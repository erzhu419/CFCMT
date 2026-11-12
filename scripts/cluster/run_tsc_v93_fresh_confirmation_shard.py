#!/usr/bin/env python3
"""Evaluate frozen V93 and three pressure references on a new seed roster.

Run through the archived run_cfcmt_sumo122.sh: its PYTHONPATH and working
directory must select the frozen V93 source/runtime, not the current checkout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as evaluator
from cf_h2o.eval.traffic_signal_causal_source_component_rollout import (
    run_source_city_component_rollout,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec, run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
from scripts.cluster.run_tsc_v93_max_pressure_shard import rollout_kwargs

METHODS = ("frozen_selector", "max_pressure", "phase_pressure", "exact_generalized_pressure")
CITY_SCENARIOS = {
    "los_angeles": ["la_1x4"],
    "jinan": ["jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"],
}
SMOKE_PAIRS = (("la_1x4", 1020485), ("jinan_3x4_real", 5622039))
REFERENCE_SPEC = PressurePolicySpec(downstream_occupancy_weight=0.02).to_dict()
RESULT_PROTOCOL = "tsc-v93-fresh-seed-confirmation-rollout-v1"
SHARD_PROTOCOL = "tsc-v93-fresh-seed-confirmation-shard-v1"
METRICS = (
    "mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
    "system_vehicle_hours", "completion_ratio", "collision_events", "collision_event_steps",
    "collision_incidents", "starting_teleports", "ending_teleports",
)


def load_inputs(parent_launch: Path, protocol_path: Path, candidate_spec: Path):
    launch = json.loads(parent_launch.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_spec.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in protocol["confirmation_seeds"])
    previous_seeds = set(candidates["confirmation_seeds"]) | set(candidates["development_seeds"])
    if not seeds or len(seeds) != len(set(seeds)) or previous_seeds.intersection(seeds):
        raise ValueError("fresh confirmation seeds must be unique and disjoint from the old 64 seeds")
    if tuple(protocol["methods"]) != METHODS or protocol["city_scenarios"] != CITY_SCENARIOS:
        raise ValueError("fresh confirmation method/scenario roster differs from its four-arm design")
    selected = [item for item in candidates["candidates"] if item["key"] == "frozen_selector"]
    expected_guards = {
        "jinan": {"risk_multiplier": 0.5, "min_context_trust": 0.0,
                  "margin": 0.0, "max_relative_rule_gap": 1.0},
        "los_angeles": None,
    }
    if (len(selected) != 1 or selected[0]["weights_by_city"] != {
            "jinan": {"atlanta": 1.0}, "los_angeles": {}}
            or selected[0]["guard_by_city"] != expected_guards):
        raise ValueError("frozen V93 source weights or city guards changed")
    return launch, protocol, selected[0], seeds


def shard_identities(seeds, *, shard_index: int, shard_count: int, smoke: bool = False):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    pairs = SMOKE_PAIRS if smoke else tuple(
        (scenario, int(seed)) for scenarios in CITY_SCENARIOS.values()
        for scenario in scenarios for seed in seeds
    )
    identities = tuple((scenario, seed, method) for scenario, seed in pairs for method in METHODS)
    return identities[shard_index::shard_count]


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, seed, method = payload["identity"]
    city = "los_angeles" if scenario == "la_1x4" else "jinan"
    root = Path(payload["results_root"]) / scenario / f"seed_{seed}" / method
    result_path, tripinfo_path = root / "result.json", root / "tripinfo.xml"
    if result_path.exists() or tripinfo_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout: {root}")
    root.mkdir(parents=True, exist_ok=True)
    common = {**payload["rollout_kwargs"], "scenario": scenario, "seed": int(seed),
              "tripinfo_out": tripinfo_path}
    contract = payload["artifact_contract"]
    started = time.monotonic()
    if method == "frozen_selector":
        candidate = payload["candidate"]
        raw_guard = candidate["guard_by_city"][city]
        guard = ContrastGuardConfig(enabled=True, **raw_guard) if raw_guard is not None else None
        result = run_source_city_component_rollout(
            main_model_path=Path(contract[f"{city}_main_model"]["path"]),
            expected_main_model_sha256=contract[f"{city}_main_model"]["sha256"],
            component_bundle_path=Path(contract[f"{city}_component_bundle"]["path"]),
            expected_component_bundle_sha256=contract[f"{city}_component_bundle"]["sha256"],
            source_city_weights=candidate["weights_by_city"][city],
            candidate_key="frozen_selector", guard_config=guard,
            allowed_seeds=tuple(payload["allowed_seeds"]), rollout_kwargs=common,
        )
    else:
        source_policy = "selected_source_prior" if method == "exact_generalized_pressure" else method
        diagnostic = ClosedLoopDiagnosticSpec(
            name=f"{method}_v93_fresh_confirmation", source_policy=source_policy,
            coordination_mode="direct" if method == "exact_generalized_pressure" else "sparse",
            result_protocol=RESULT_PROTOCOL, allowed_seeds=tuple(payload["allowed_seeds"]),
            analysis_status=payload["stage"],
        )
        result = run_external_closed_loop_rollout(
            **common, policy=diagnostic.name, diagnostic_spec=diagnostic,
        )
    metrics = result["metrics"]
    if (result["evaluation_horizon_sec"], result["control_interval_sec"], result["warmup_sec"]) != (3600, 10, 60):
        raise ValueError("frozen V93 rollout timing changed")
    reference_audit = metrics.get("reference_policy_audit", {})
    if method == "exact_generalized_pressure" and reference_audit.get("reference_policy_spec") != REFERENCE_SPEC:
        raise ValueError("exact reference is not the frozen V93 generalized pressure")
    result["prospective_confirmation"] = {
        "protocol": RESULT_PROTOCOL, "protocol_path": payload["protocol_path"],
        "stage": payload["stage"], "method": method,
        "parent_launch": payload["parent_launch"], "candidate_spec": payload["candidate_spec"],
        "source_root": payload["source_root"], "evaluator_path": str(Path(evaluator.__file__).resolve()),
    }
    _atomic_json(result_path, result)
    return {
        "method": method, "city": city, "scenario": scenario, "seed": int(seed),
        "runtime_policy": result["runtime_policy"], "stage": payload["stage"],
        "hostname": socket.gethostname(), "elapsed_sec": time.monotonic() - started,
        "result_path": str(result_path), "tripinfo_path": str(tripinfo_path),
        "evaluation_horizon_sec": result["evaluation_horizon_sec"],
        "control_interval_sec": result["control_interval_sec"], "warmup_sec": result["warmup_sec"],
        "metrics": {key: metrics[key] for key in METRICS},
        "guard_audit": metrics.get("guard_audit", {}),
        "reference_decision_requests": reference_audit.get("decision_requests"),
        "reference_post_warmup_decision_requests": reference_audit.get("post_warmup_decision_requests"),
        "source_city_weights": result.get("source_city_weights"),
        "source_guard_config": result.get("source_guard_config"),
        "target_only_anchor": result.get("target_only_anchor"),
        "frozen_inputs": {key: value for key, value in contract.items()
                          if key in ("freeze_root", "protocol_remote", "conversion_root",
                                     f"{city}_main_model", f"{city}_component_bundle")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-launch", "protocol", "candidate-spec", "results-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("shard-index", "shard-count", "workers"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="Eight historical replay cells; never use fresh seeds")
    args = parser.parse_args(argv)
    source_root = Path(os.environ["CFCMT_SOURCE_ROOT"]).resolve()
    if Path(evaluator.__file__).resolve() != source_root / "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py":
        raise ValueError("worker imported an evaluator outside CFCMT_SOURCE_ROOT")
    launch, protocol, candidate, seeds = load_inputs(args.parent_launch, args.protocol, args.candidate_spec)
    identities = shard_identities(seeds, shard_index=args.shard_index,
                                  shard_count=args.shard_count, smoke=args.smoke)
    if not identities:
        parser.error("this shard has no assigned rollout")
    root = args.results_root.resolve()
    summary_path = root / "_shards" / f"shard_{args.shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard summary: {summary_path}")
    common = {
        "artifact_contract": launch["artifact_contract"], "candidate": candidate,
        "rollout_kwargs": rollout_kwargs(launch), "results_root": str(root),
        "allowed_seeds": tuple(seed for _, seed in SMOKE_PAIRS) if args.smoke else seeds,
        "stage": "historical-equivalence-smoke" if args.smoke else "prospective-fresh-seed-confirmation",
        "protocol_path": str(args.protocol.resolve()), "parent_launch": str(args.parent_launch.resolve()),
        "candidate_spec": str(args.candidate_spec.resolve()), "source_root": str(source_root),
    }
    payloads = [{**common, "identity": identity} for identity in identities]
    workers = min(max(args.workers, 1), len(payloads))
    started, rows = time.monotonic(), []
    def progress():
        print("SCHEDULEURM_PROGRESS " + json.dumps({"current": len(rows), "total": len(payloads),
              "unit": "rollout", "elapsed_seconds": time.monotonic() - started}), flush=True)
    progress()
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_run_one, payloads, chunksize=1):
            rows.append(row)
            progress()
    rows.sort(key=lambda row: (row["scenario"], row["seed"], row["method"]))
    if {(row["scenario"], row["seed"], row["method"]) for row in rows} != set(identities):
        raise RuntimeError("fresh confirmation shard coverage failed")
    summary = {
        "protocol": SHARD_PROTOCOL, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": common["stage"], "protocol_path": common["protocol_path"],
        "parent_launch": common["parent_launch"], "candidate_spec": common["candidate_spec"],
        "source_root": str(source_root), "evaluator_path": str(Path(evaluator.__file__).resolve()),
        "shard_index": args.shard_index, "shard_count": args.shard_count, "workers": workers,
        "fresh_process_per_rollout": True, "task_count": len(rows),
        "elapsed_sec": time.monotonic() - started, "complete": True, "rows": rows,
    }
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "COMPLETE", "task_count": len(rows), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
