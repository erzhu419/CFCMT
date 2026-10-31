#!/usr/bin/env python3
"""Supplement the frozen V93 cohort with the existing MaxPressure controller."""

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
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)
from scripts.cluster.run_tsc_external_phase_pressure_diagnostic_shard import (
    shard_identities,
)


RESULT_PROTOCOL = "tsc-v93-max-pressure-supplement-rollout-v1"
SHARD_PROTOCOL = "tsc-v93-max-pressure-supplement-shard-v1"
POLICY = "max_pressure_v93_supplement"
METRICS = (
    "mean_tripinfo_waiting_time",
    "mean_tripinfo_duration",
    "mean_tripinfo_time_loss",
    "system_vehicle_hours",
    "completion_ratio",
    "collision_incidents",
    "starting_teleports",
    "ending_teleports",
)


def load_cohort(parent_launch: Path, confirmation_spec: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    launch = json.loads(parent_launch.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_spec.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in confirmation["confirmation_seeds"])
    if (
        confirmation.get("freeze_protocol")
        != "tsc-v93-causal-source-selector-disjoint-confirmation-freeze-v1"
        or len(seeds) != 56
        or len(set(seeds)) != 56
        or seeds != tuple(int(seed) for seed in launch["seeds"])
    ):
        raise ValueError("supplement must retain the original 56-seed V93 cohort")
    return launch, seeds


def rollout_kwargs(launch: Mapping[str, Any]) -> dict[str, Any]:
    """Use the archived V93 paths and their already recorded input bindings."""
    contract = launch["artifact_contract"]
    return {
        "protocol_spec_path": Path(contract["protocol_remote"]),
        "offline_authorization_path": Path(contract["offline_authorization"]["path"]),
        "expected_offline_authorization_sha256": contract["offline_authorization"]["sha256"],
        "joint_freeze_audit_path": Path(contract["joint_audit"]["path"]),
        "expected_joint_freeze_sha256": contract["joint_audit"]["sha256"],
        "freeze_root": Path(contract["freeze_root"]),
        "external_manifest_path": Path(contract["external_manifest_remote"]),
        "expected_external_manifest_sha256": launch["external_manifest_sha256"],
        "conversion_root": Path(contract["conversion_root"]),
        "conversion_manifest_path": Path(contract["conversion_manifest"]["path"]),
        "expected_conversion_manifest_sha256": contract["conversion_manifest"]["sha256"],
        "expected_conversion_tree_sha256": contract["conversion_tree_sha256"],
        "expected_sumo_version": "1.22.0",
    }


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, seed = payload["identity"]
    root = Path(payload["results_root"]) / scenario / f"seed_{seed}" / "max_pressure"
    result_path, tripinfo_path = root / "result.json", root / "tripinfo.xml"
    if result_path.exists() or tripinfo_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout: {root}")
    root.mkdir(parents=True, exist_ok=True)
    diagnostic = ClosedLoopDiagnosticSpec(
        name=POLICY,
        source_policy="max_pressure",
        coordination_mode="sparse",
        result_protocol=RESULT_PROTOCOL,
        allowed_seeds=tuple(payload["allowed_seeds"]),
        analysis_status="supplemental-baseline-on-existing-v93-confirmation-cohort",
    )
    started = time.monotonic()
    result = run_external_closed_loop_rollout(
        **payload["rollout_kwargs"],
        scenario=str(scenario),
        seed=int(seed),
        policy=POLICY,
        tripinfo_out=tripinfo_path,
        diagnostic_spec=diagnostic,
    )
    _atomic_json(result_path, result)
    return {
        "city": str(result["city"]),
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": POLICY,
        "source_policy": "max_pressure",
        "hostname": socket.gethostname(),
        "elapsed_sec": float(time.monotonic() - started),
        "result_path": str(result_path),
        "tripinfo_path": str(tripinfo_path),
        "evaluation_horizon_sec": result["evaluation_horizon_sec"],
        "control_interval_sec": result["control_interval_sec"],
        "warmup_sec": result["warmup_sec"],
        "runtime_policy": result["runtime_policy"],
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
    args = parser.parse_args(argv)
    launch, seeds = load_cohort(args.parent_launch, args.confirmation_spec)
    identities = shard_identities(seeds, shard_index=args.shard_index, shard_count=args.shard_count)
    summary_path = args.results_root / "_shards" / f"shard_{args.shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard summary: {summary_path}")
    common = {
        "allowed_seeds": seeds,
        "rollout_kwargs": rollout_kwargs(launch),
        "results_root": str(args.results_root.resolve()),
    }
    payloads = [{**common, "identity": identity} for identity in identities]
    os.chdir(Path(__file__).resolve().parents[2])
    started = time.monotonic()
    workers = min(max(int(args.workers), 1), len(payloads))
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(_run_one, payloads, chunksize=1))
    rows.sort(key=lambda row: (row["scenario"], row["seed"]))
    if {(row["scenario"], row["seed"]) for row in rows} != set(identities):
        raise RuntimeError("MaxPressure shard coverage failed")
    summary = {
        "protocol": SHARD_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_launch": str(args.parent_launch.resolve()),
        "confirmation_spec": str(args.confirmation_spec.resolve()),
        "hostname": socket.gethostname(),
        "source_policy": "max_pressure",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "workers": workers,
        "fresh_process_per_rollout": True,
        "task_count": len(rows),
        "elapsed_sec": float(time.monotonic() - started),
        "passed": True,
        "rows": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "PASS", "task_count": len(rows), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
