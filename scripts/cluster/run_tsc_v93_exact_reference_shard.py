#!/usr/bin/env python3
"""Run the exact frozen V93 generalized-pressure reference on its 224-run panel."""
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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec, run_external_closed_loop_rollout,
)
from scripts.cluster.run_tsc_external_phase_pressure_diagnostic_shard import shard_identities
from scripts.cluster.run_tsc_v93_max_pressure_shard import load_cohort, rollout_kwargs
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec

RESULT_PROTOCOL = "tsc-v93-exact-reference-supplement-rollout-v1"
SHARD_PROTOCOL = "tsc-v93-exact-reference-supplement-shard-v1"
POLICY = "exact_generalized_pressure_v93_supplement"
REFERENCE_SPEC = PressurePolicySpec(downstream_occupancy_weight=0.02).to_dict()
REFERENCE_KEY = REFERENCE_SPEC["key"]
METRICS = (
    "mean_tripinfo_waiting_time", "mean_tripinfo_duration", "mean_tripinfo_time_loss",
    "system_vehicle_hours", "completion_ratio", "collision_events", "collision_event_steps",
    "collision_incidents", "starting_teleports", "ending_teleports",
)


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, seed = payload["identity"]
    root = Path(payload["results_root"]) / scenario / f"seed_{seed}" / "exact_reference"
    result_path, tripinfo_path = root / "result.json", root / "tripinfo.xml"
    if result_path.exists() or tripinfo_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout: {root}")
    root.mkdir(parents=True, exist_ok=True)
    diagnostic = ClosedLoopDiagnosticSpec(
        name=POLICY, source_policy="selected_source_prior", coordination_mode="direct",
        result_protocol=RESULT_PROTOCOL, allowed_seeds=tuple(payload["allowed_seeds"]),
        analysis_status="exact-reference-on-reused-v93-benchmark-panel-without-refit",
    )
    started = time.monotonic()
    result = run_external_closed_loop_rollout(
        **payload["rollout_kwargs"], scenario=str(scenario), seed=int(seed), policy=POLICY,
        tripinfo_out=tripinfo_path, diagnostic_spec=diagnostic,
    )
    audit = result["metrics"]["reference_policy_audit"]
    if audit["reference_policy_spec"] != REFERENCE_SPEC:
        raise ValueError("loaded reference differs from the frozen V93 generalized pressure")
    result["reference_comparison"] = {
        "reference_key": REFERENCE_KEY,
        "reference_score": "green_q - 0.02 * green_down_occ",
        "selection": "same ordered executor-feasible states and first np.argmax as CFCMT reference",
        "execution": "direct reference requests through the shared physical executor",
        "no_model_fitting_or_parameter_selection": True,
    }
    _atomic_json(result_path, result)
    return {
        "city": result["city"], "scenario": str(scenario), "seed": int(seed),
        "policy": POLICY, "source_policy": "selected_source_prior",
        "runtime_policy": result["runtime_policy"], "reference_key": REFERENCE_KEY,
        "hostname": socket.gethostname(), "elapsed_sec": time.monotonic() - started,
        "result_path": str(result_path), "tripinfo_path": str(tripinfo_path),
        "evaluation_horizon_sec": result["evaluation_horizon_sec"],
        "control_interval_sec": result["control_interval_sec"], "warmup_sec": result["warmup_sec"],
        "reference_decision_requests": audit["decision_requests"],
        "reference_post_warmup_decision_requests": audit["post_warmup_decision_requests"],
        "reference_trace_rows": len(audit["request_trace"]),
        "metrics": {key: result["metrics"][key] for key in METRICS},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-launch", type=Path, required=True)
    parser.add_argument("--confirmation-spec", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--scenario", help="Run only this frozen scenario for a one-run smoke check")
    parser.add_argument("--seed", type=int, help="Run only this frozen seed for a one-run smoke check")
    args = parser.parse_args(argv)
    launch, seeds = load_cohort(args.parent_launch, args.confirmation_spec)
    identities = shard_identities(seeds, shard_index=args.shard_index, shard_count=args.shard_count)
    if (args.scenario is None) != (args.seed is None):
        parser.error("--scenario and --seed must be supplied together")
    if args.scenario is not None:
        identities = tuple(x for x in identities if x == (args.scenario, args.seed))
        if not identities:
            parser.error("smoke scenario/seed must belong to this frozen shard")
    summary_path = args.results_root / "_shards" / f"shard_{args.shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard summary: {summary_path}")
    common = {"allowed_seeds": seeds, "rollout_kwargs": rollout_kwargs(launch),
              "results_root": str(args.results_root.resolve())}
    payloads = [{**common, "identity": identity} for identity in identities]
    os.chdir(Path(__file__).resolve().parents[2])
    started = time.monotonic()
    workers = min(max(int(args.workers), 1), len(payloads))
    print("SCHEDULEURM_PROGRESS " + json.dumps({
        "current": 0, "total": len(payloads), "unit": "rollout", "elapsed_seconds": 0.0,
    }), flush=True)
    rows = []
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_run_one, payloads, chunksize=1):
            rows.append(row)
            print("SCHEDULEURM_PROGRESS " + json.dumps({
                "current": len(rows), "total": len(payloads), "unit": "rollout",
                "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    rows.sort(key=lambda row: (row["scenario"], row["seed"]))
    if {(row["scenario"], row["seed"]) for row in rows} != set(identities):
        raise RuntimeError("exact-reference shard coverage failed")
    summary = {
        "protocol": SHARD_PROTOCOL, "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_launch": str(args.parent_launch.resolve()),
        "confirmation_spec": str(args.confirmation_spec.resolve()),
        "hostname": socket.gethostname(), "source_policy": "selected_source_prior",
        "reference_key": REFERENCE_KEY, "shard_index": args.shard_index,
        "shard_count": args.shard_count, "workers": workers,
        "fresh_process_per_rollout": True, "task_count": len(rows),
        "elapsed_sec": time.monotonic() - started, "passed": True, "rows": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "PASS", "task_count": len(rows), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
