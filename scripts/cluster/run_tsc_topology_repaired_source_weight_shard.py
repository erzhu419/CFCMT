#!/usr/bin/env python3
"""Run one process-isolated shard of topology-repaired source-weight rollouts."""

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
from cf_h2o.eval.traffic_signal_topology_repaired_source_weight_rollout import (
    RESULT_PROTOCOL,
    run_topology_repaired_source_weight_rollout,
    source_weight_key,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig


SHARD_PROTOCOL = "tsc-v92-topology-repaired-source-weight-process-shard-v1"
SCENARIOS = (
    "la_1x4",
    "jinan_3x4_real",
    "jinan_3x4_real_2000",
    "jinan_3x4_real_2500",
)


def all_identities(
    seeds: Sequence[int], weights: Sequence[float]
) -> tuple[tuple[str, int, float], ...]:
    normalized_seeds = tuple(int(value) for value in seeds)
    normalized_weights = tuple(float(value) for value in weights)
    if (
        not normalized_seeds
        or len(set(normalized_seeds)) != len(normalized_seeds)
        or not normalized_weights
        or len(set(normalized_weights)) != len(normalized_weights)
    ):
        raise ValueError("source-weight seed/weight grid is empty or duplicated")
    return tuple(
        (scenario, seed, weight)
        for scenario in SCENARIOS
        for seed in normalized_seeds
        for weight in normalized_weights
    )


def shard_identities(
    seeds: Sequence[int],
    weights: Sequence[float],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[str, int, float], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid source-weight shard index/count")
    return tuple(
        identity
        for index, identity in enumerate(all_identities(seeds, weights))
        if index % shard_count == shard_index
    )


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, seed, weight = payload["identity"]
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    if result_path.exists() or tripinfo_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout: {result_path.parent}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    city = "los_angeles" if scenario == "la_1x4" else "jinan"
    guard_values = payload.get("guard_config")
    guard = ContrastGuardConfig(**guard_values) if guard_values is not None else None
    result = run_topology_repaired_source_weight_rollout(
        model_path=Path(str(payload[f"{city}_model"])),
        expected_model_sha256=str(payload[f"{city}_model_sha256"]),
        source_weight=float(weight),
        allowed_seeds=tuple(int(value) for value in payload["allowed_seeds"]),
        guard_config=guard,
        rollout_kwargs={
            "protocol_spec_path": Path(str(payload["protocol"])),
            "offline_authorization_path": Path(
                str(payload["offline_authorization"])
            ),
            "expected_offline_authorization_sha256": str(
                payload["offline_authorization_sha256"]
            ),
            "joint_freeze_audit_path": Path(str(payload["joint_audit"])),
            "expected_joint_freeze_sha256": str(payload["joint_audit_sha256"]),
            "freeze_root": Path(str(payload["freeze_root"])),
            "external_manifest_path": Path(str(payload["external_manifest"])),
            "conversion_root": Path(str(payload["conversion_root"])),
            "expected_external_manifest_sha256": str(
                payload["external_manifest_sha256"]
            ),
            "conversion_manifest_path": Path(
                str(payload["conversion_manifest"])
            ),
            "expected_conversion_manifest_sha256": str(
                payload["conversion_manifest_sha256"]
            ),
            "expected_conversion_tree_sha256": str(
                payload["conversion_tree_sha256"]
            ),
            "expected_sumo_version": "1.22.0",
            "scenario": str(scenario),
            "seed": int(seed),
            "tripinfo_out": tripinfo_path,
        },
    )
    _atomic_json(result_path, result)
    metrics = result["metrics"]
    return {
        "city": city,
        "scenario": str(scenario),
        "seed": int(seed),
        "source_weight": float(weight),
        "guard_config": guard_values,
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
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--guard-risk-multiplier", type=float)
    parser.add_argument("--guard-min-context-trust", type=float, default=0.0)
    parser.add_argument("--guard-margin", type=float, default=0.0)
    parser.add_argument("--guard-max-relative-rule-gap", type=float, default=1.0)
    parser.add_argument("--la-model", type=Path, required=True)
    parser.add_argument("--la-model-sha256", required=True)
    parser.add_argument("--jinan-model", type=Path, required=True)
    parser.add_argument("--jinan-model-sha256", required=True)
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
        args.weights,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    guard_config = (
        {
            "enabled": True,
            "risk_multiplier": float(args.guard_risk_multiplier),
            "min_context_trust": float(args.guard_min_context_trust),
            "margin": float(args.guard_margin),
            "max_relative_rule_gap": float(args.guard_max_relative_rule_gap),
        }
        if args.guard_risk_multiplier is not None
        else None
    )
    common = {
        "allowed_seeds": tuple(args.seeds),
        "los_angeles_model": str(args.la_model),
        "los_angeles_model_sha256": args.la_model_sha256,
        "jinan_model": str(args.jinan_model),
        "jinan_model_sha256": args.jinan_model_sha256,
        "guard_config": guard_config,
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
    for scenario, seed, weight in identities:
        root = (
            args.results_root
            / scenario
            / f"seed_{seed}"
            / f"source_weight_{source_weight_key(weight)}"
        )
        payloads.append(
            {
                **common,
                "identity": (scenario, seed, weight),
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
    rows.sort(key=lambda row: (row["scenario"], row["seed"], row["source_weight"]))
    observed = {
        (row["scenario"], int(row["seed"]), float(row["source_weight"]))
        for row in rows
    }
    if observed != set(identities):
        raise RuntimeError("source-weight shard coverage failed")
    summary = {
        "protocol": SHARD_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "workers": workers,
        "fresh_process_per_rollout": True,
        "guard_config": guard_config,
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
    print(
        json.dumps(
            {
                "status": "PASS",
                "shard_index": args.shard_index,
                "task_count": len(rows),
                "elapsed_sec": summary["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    print(f"Eval complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
