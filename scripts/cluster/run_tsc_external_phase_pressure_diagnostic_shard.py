#!/usr/bin/env python3
"""Run a process-isolated frozen-policy diagnostic on external-city seeds."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)
from scripts.cluster.run_tsc_topology_repaired_source_weight_shard import SCENARIOS


RESULT_PROTOCOL = "tsc-v92-external-frozen-policy-diagnostic-rollout-v1"
SHARD_PROTOCOL = "tsc-v92-external-frozen-policy-process-shard-v1"
SOURCE_POLICIES = (
    "phase_pressure",
    "h2oplus_style_dense_residual_mpc",
    "rigid_anchor_mpc",
    "simulator_only_mpc",
)


def shard_identities(
    seeds: Sequence[int], *, shard_index: int, shard_count: int
) -> tuple[tuple[str, int], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid phase-pressure shard index/count")
    identities = tuple(
        (scenario, int(seed)) for scenario in SCENARIOS for seed in seeds
    )
    return tuple(
        identity
        for index, identity in enumerate(identities)
        if index % shard_count == shard_index
    )


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, seed = payload["identity"]
    source_policy = str(payload["source_policy"])
    policy = f"{source_policy}_v92_diagnostic"
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    if result_path.exists() or tripinfo_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout: {result_path.parent}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = ClosedLoopDiagnosticSpec(
        name=policy,
        source_policy=source_policy,
        coordination_mode="sparse",
        result_protocol=RESULT_PROTOCOL,
        allowed_seeds=tuple(int(value) for value in payload["allowed_seeds"]),
        analysis_status="post_v91_redevelopment_diagnostic_not_confirmation",
    )
    started = time.monotonic()
    result = run_external_closed_loop_rollout(
        protocol_spec_path=Path(str(payload["protocol"])),
        offline_authorization_path=Path(str(payload["offline_authorization"])),
        expected_offline_authorization_sha256=str(
            payload["offline_authorization_sha256"]
        ),
        joint_freeze_audit_path=Path(str(payload["joint_audit"])),
        expected_joint_freeze_sha256=str(payload["joint_audit_sha256"]),
        freeze_root=Path(str(payload["freeze_root"])),
        external_manifest_path=Path(str(payload["external_manifest"])),
        conversion_root=Path(str(payload["conversion_root"])),
        expected_external_manifest_sha256=str(
            payload["external_manifest_sha256"]
        ),
        conversion_manifest_path=Path(str(payload["conversion_manifest"])),
        expected_conversion_manifest_sha256=str(
            payload["conversion_manifest_sha256"]
        ),
        expected_conversion_tree_sha256=str(payload["conversion_tree_sha256"]),
        expected_sumo_version="1.22.0",
        scenario=str(scenario),
        seed=int(seed),
        policy=policy,
        tripinfo_out=tripinfo_path,
        diagnostic_spec=diagnostic,
    )
    _atomic_json(result_path, result)
    metrics = result["metrics"]
    return {
        "city": str(result["city"]),
        "scenario": str(scenario),
        "seed": int(seed),
        "policy": policy,
        "source_policy": source_policy,
        "hostname": socket.gethostname(),
        "elapsed_sec": float(time.monotonic() - started),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "tripinfo_path": str(tripinfo_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "metrics": {
            key: metrics[key]
            for key in (
                "mean_tripinfo_waiting_time",
                "mean_tripinfo_duration",
                "mean_tripinfo_time_loss",
                "system_vehicle_hours",
                "completion_ratio",
                "collision_incidents",
                "starting_teleports",
                "ending_teleports",
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--source-policy", choices=SOURCE_POLICIES, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--offline-authorization-sha256", required=True)
    parser.add_argument("--joint-audit", type=Path, required=True)
    parser.add_argument("--joint-audit-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest-sha256", required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)
    identities = shard_identities(
        args.seeds,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    common = {
        "allowed_seeds": tuple(args.seeds),
        "source_policy": args.source_policy,
        "protocol": str(args.protocol),
        "offline_authorization": str(args.offline_authorization),
        "offline_authorization_sha256": args.offline_authorization_sha256,
        "joint_audit": str(args.joint_audit),
        "joint_audit_sha256": args.joint_audit_sha256,
        "freeze_root": str(args.freeze_root),
        "external_manifest": str(args.external_manifest),
        "external_manifest_sha256": args.external_manifest_sha256,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest": str(args.conversion_manifest),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
    }
    payloads = []
    for scenario, seed in identities:
        policy = f"{args.source_policy}_v92_diagnostic"
        root = args.results_root / scenario / f"seed_{seed}" / policy
        payloads.append(
            {
                **common,
                "identity": (scenario, seed),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
            }
        )
    started = time.monotonic()
    os.chdir(Path(__file__).resolve().parents[2])
    workers = min(max(int(args.workers), 1), len(payloads))
    with mp.get_context("spawn").Pool(
        processes=workers, maxtasksperchild=1
    ) as pool:
        rows = list(pool.imap_unordered(_run_one, payloads, chunksize=1))
    rows.sort(key=lambda row: (row["scenario"], row["seed"]))
    observed = {(row["scenario"], row["seed"]) for row in rows}
    if observed != set(identities):
        raise RuntimeError("phase-pressure shard coverage failed")
    summary = {
        "protocol": SHARD_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "workers": workers,
        "fresh_process_per_rollout": True,
        "source_policy": args.source_policy,
        "task_count": len(rows),
        "elapsed_sec": float(time.monotonic() - started),
        "passed": True,
        "rows": rows,
    }
    summary_path = args.results_root / "_shards" / f"shard_{args.shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard summary: {summary_path}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "PASS", "task_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
