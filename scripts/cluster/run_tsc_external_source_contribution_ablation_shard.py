#!/usr/bin/env python3
"""Run one process-isolated shard of the frozen v43 target-only ablation."""

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
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import (
    ANALYSIS_PROTOCOL,
    POLICY,
    RESULT_PROTOCOL,
    SPEC,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import PROTOCOL


SHARD_PROTOCOL = "tsc-v90r86-source-contribution-process-shard-v1"


def all_identities(protocol: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    scenarios = tuple(
        str(scenario)
        for city_scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].values()
        for scenario in city_scenarios
    )
    seeds = tuple(
        int(seed) for seed in protocol["target_protocol"]["closed_loop_seeds"]
    )
    identities = tuple((scenario, seed) for scenario in scenarios for seed in seeds)
    if len(scenarios) != 4 or len(seeds) != 2 or len(set(identities)) != 8:
        raise ValueError("source-contribution matrix cardinality changed")
    return identities


def shard_identities(
    protocol: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[tuple[str, int], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid source-contribution shard index/count")
    return tuple(
        identity
        for index, identity in enumerate(all_identities(protocol))
        if index % shard_count == shard_index
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_existing(
    *,
    result_path: Path,
    tripinfo_path: Path,
    identity: tuple[str, int],
) -> dict[str, Any]:
    if result_path.is_file() != tripinfo_path.is_file():
        raise RuntimeError(f"partial source-contribution result: {result_path.parent}")
    if not result_path.is_file():
        return {}
    scenario, seed = identity
    result = _read_json(result_path)
    metrics = dict(result.get("metrics", {}))
    diagnostic = dict(result.get("post_hoc_diagnostic") or {})
    if not (
        result.get("protocol") == RESULT_PROTOCOL
        and result.get("analysis_protocol") == ANALYSIS_PROTOCOL
        and result.get("scenario") == scenario
        and int(result.get("seed", -1)) == seed
        and result.get("policy") == POLICY
        and result.get("source_policy") == POLICY
        and result.get("runtime_policy") == "causal_target_only_contrast_raw"
        and diagnostic == SPEC.to_dict()
        and bool(metrics.get("ok", False))
        and int(metrics.get("starting_teleports", -1)) == 0
        and int(metrics.get("ending_teleports", -1)) == 0
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
    ):
        raise RuntimeError(f"invalid source-contribution result: {result_path}")
    return result


def _summary_row(result: Mapping[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    return {
        "city": str(result["city"]),
        "scenario": str(result["scenario"]),
        "seed": int(result["seed"]),
        "policy": POLICY,
        "hostname": socket.gethostname(),
        "elapsed_sec": float(elapsed_sec),
        "result_path": str(result["result_path"]),
        "result_sha256": str(result["result_sha256"]),
        "tripinfo_path": str(result["tripinfo_path"]),
        "tripinfo_sha256": str(result["tripinfo_sha256"]),
        "metrics": {
            key: metrics[key]
            for key in (
                "mean_tripinfo_waiting_time",
                "mean_tripinfo_duration",
                "mean_tripinfo_time_loss",
                "system_vehicle_hours",
                "completion_ratio",
                "mean_queue_per_lane",
                "collision_incidents",
                "starting_teleports",
                "ending_teleports",
                "tripinfo_count",
                "tripinfo_completed_count",
                "tripinfo_unfinished_count",
                "demand_population_at_horizon",
            )
        },
    }


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, raw_seed = tuple(payload["identity"])
    identity = (str(scenario), int(raw_seed))
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    started = time.monotonic()
    result = _validate_existing(
        result_path=result_path,
        tripinfo_path=tripinfo_path,
        identity=identity,
    )
    if not result:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_external_closed_loop_rollout(
            protocol_spec_path=Path(str(payload["protocol_path"])),
            offline_authorization_path=Path(
                str(payload["offline_authorization_path"])
            ),
            expected_offline_authorization_sha256=str(
                payload["offline_authorization_sha256"]
            ),
            joint_freeze_audit_path=Path(str(payload["joint_audit_path"])),
            expected_joint_freeze_sha256=str(payload["joint_audit_sha256"]),
            freeze_root=Path(str(payload["freeze_root"])),
            external_manifest_path=Path(str(payload["external_manifest_path"])),
            conversion_root=Path(str(payload["conversion_root"])),
            expected_external_manifest_sha256=str(
                payload["external_manifest_sha256"]
            ),
            conversion_manifest_path=Path(
                str(payload["conversion_manifest_path"])
            ),
            expected_conversion_manifest_sha256=str(
                payload["conversion_manifest_sha256"]
            ),
            expected_conversion_tree_sha256=str(payload["conversion_tree_sha256"]),
            expected_sumo_version="1.22.0",
            scenario=identity[0],
            seed=identity[1],
            policy=POLICY,
            tripinfo_out=tripinfo_path,
            diagnostic_spec=SPEC,
        )
        result["analysis_protocol"] = ANALYSIS_PROTOCOL
        _atomic_json(result_path, result)
    return _summary_row(
        {
            **result,
            "result_path": result_path,
            "result_sha256": _sha256(result_path),
            "tripinfo_path": tripinfo_path,
            "tripinfo_sha256": _sha256(tripinfo_path),
        },
        elapsed_sec=time.monotonic() - started,
    )


def _prepare_spawn_working_directory() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    os.chdir(source_root)
    return source_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    protocol = _read_json(args.protocol)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("source-contribution parent protocol changed")
    identities = shard_identities(
        protocol,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    common = {
        "protocol_path": str(args.protocol),
        "offline_authorization_path": str(args.offline_authorization),
        "offline_authorization_sha256": args.offline_authorization_sha256,
        "joint_audit_path": str(args.joint_audit),
        "joint_audit_sha256": args.joint_audit_sha256,
        "freeze_root": str(args.freeze_root),
        "external_manifest_path": str(args.external_manifest),
        "external_manifest_sha256": args.external_manifest_sha256,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_path": str(args.conversion_manifest),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
    }
    payloads = []
    for scenario, seed in identities:
        root = args.results_root / scenario / f"seed_{seed}" / POLICY
        payloads.append(
            {
                **common,
                "identity": (scenario, seed),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
            }
        )

    started = time.monotonic()
    source_root = _prepare_spawn_working_directory()
    workers = min(max(int(args.workers), 1), len(payloads))
    with mp.get_context("spawn").Pool(
        processes=workers, maxtasksperchild=1
    ) as pool:
        rows = list(pool.imap_unordered(_run_one, payloads, chunksize=1))
    rows.sort(key=lambda row: (row["scenario"], int(row["seed"])))
    observed = {(row["scenario"], int(row["seed"])) for row in rows}
    if observed != set(identities):
        raise RuntimeError("source-contribution shard coverage failed")

    summary = {
        "protocol": SHARD_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "spawn_working_directory": str(source_root),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "workers": workers,
        "fresh_process_per_rollout": True,
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
                "workers": workers,
                "elapsed_sec": summary["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
