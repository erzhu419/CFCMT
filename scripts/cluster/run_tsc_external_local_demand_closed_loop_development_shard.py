#!/usr/bin/env python3
"""Run one process-isolated shard of the v75 development matrix."""

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
from cf_h2o.eval.traffic_signal_external_local_demand_closed_loop_development import (
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
    run_local_demand_closed_loop_development,
)


SHARD_PROTOCOL = "tsc-v75r71-local-demand-development-process-shard-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shard_identities(
    protocol: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[tuple[str, str, int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid v75 shard index/count")
    return tuple(
        identity
        for index, identity in enumerate(all_rollout_identities(protocol))
        if index % shard_count == shard_index
    )


def _validate_existing(
    *,
    result_path: Path,
    tripinfo_path: Path,
    identity: tuple[str, str, int, str],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    if result_path.is_file() != tripinfo_path.is_file():
        raise RuntimeError(f"partial v75 result exists: {result_path.parent}")
    if not result_path.is_file():
        return {}
    city, scenario, seed, policy = identity
    result = _read_json(result_path)
    if not (
        result.get("protocol") == RESULT_PROTOCOL
        and result.get("city") == city
        and result.get("scenario") == scenario
        and int(result.get("seed", -1)) == seed
        and result.get("policy") == policy
        and result.get("protocol_sha256") == hashes["protocol_sha256"]
        and result.get("proposal_parent_protocol_sha256")
        == hashes["proposal_parent_protocol_sha256"]
        and result.get("hierarchical_freeze_audit_sha256")
        == hashes["hierarchical_freeze_audit_sha256"]
        and result.get("local_demand_freeze_result_sha256")
        == hashes["local_demand_freeze_result_sha256"]
        and result.get("local_demand_joint_audit_sha256")
        == hashes["local_demand_joint_audit_sha256"]
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and bool(result.get("metrics", {}).get("ok", False))
    ):
        raise RuntimeError(f"existing v75 result is invalid: {result_path}")
    return result


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = tuple(payload["identity"])
    identity = (str(raw[0]), str(raw[1]), int(raw[2]), str(raw[3]))
    city, scenario, seed, policy = identity
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    hashes = dict(payload["hashes"])
    started = time.monotonic()
    result = _validate_existing(
        result_path=result_path,
        tripinfo_path=tripinfo_path,
        identity=identity,
        hashes=hashes,
    )
    reused = bool(result)
    if not result:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_local_demand_closed_loop_development(
            protocol_path=Path(str(payload["protocol_path"])),
            proposal_parent_protocol_path=Path(
                str(payload["proposal_parent_protocol_path"])
            ),
            hierarchical_freeze_audit_path=Path(
                str(payload["hierarchical_freeze_audit_path"])
            ),
            hierarchical_freeze_root=Path(
                str(payload["hierarchical_freeze_root"])
            ),
            veto_freeze_result_path=Path(
                str(payload["veto_freeze_result_path"])
            ),
            veto_joint_audit_path=Path(str(payload["veto_joint_audit_path"])),
            veto_artifact_root=Path(str(payload["veto_artifact_root"])),
            external_manifest_path=Path(str(payload["external_manifest_path"])),
            conversion_root=Path(str(payload["conversion_root"])),
            conversion_manifest_path=Path(str(payload["conversion_manifest_path"])),
            scenario=scenario,
            seed=seed,
            policy=policy,
            tripinfo_out=tripinfo_path,
        )
        _atomic_json(result_path, result)
    metrics = dict(result["metrics"])
    guard = dict(metrics.get("guard_audit", {}))
    safety = dict(metrics.get("safety_audit", {}))
    return {
        "city": city,
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "reused": reused,
        "elapsed_sec": float(time.monotonic() - started),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "tripinfo_path": str(tripinfo_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
        "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
        "executed_overrides": int(
            guard.get("executed_overrides", guard.get("accepted_overrides", 0))
        ),
        "rejected_long_horizon_veto": int(
            guard.get("rejected_long_horizon_veto", 0)
        ),
        "collision_incidents": int(safety.get("unique_collision_incidents", 0)),
        "starting_teleports": int(metrics.get("starting_teleports", -1)),
        "ending_teleports": int(metrics.get("ending_teleports", -1)),
        "target_veto_active": bool(result["target_veto_active"]),
    }


def run_shard(
    *,
    protocol_path: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    veto_freeze_result_path: Path,
    veto_joint_audit_path: Path,
    veto_artifact_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
    results_root: Path,
    shard_index: int,
    shard_count: int,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v75 shard protocol changed")
    identities = shard_identities(
        protocol, shard_index=int(shard_index), shard_count=int(shard_count)
    )
    summary_path = Path(results_root) / "_shards" / f"shard_{shard_index}.json"
    hashes = {
        "protocol_sha256": _sha256(protocol_path),
        "proposal_parent_protocol_sha256": _sha256(proposal_parent_protocol_path),
        "hierarchical_freeze_audit_sha256": _sha256(
            hierarchical_freeze_audit_path
        ),
        "local_demand_freeze_result_sha256": _sha256(veto_freeze_result_path),
        "local_demand_joint_audit_sha256": _sha256(veto_joint_audit_path),
    }
    if summary_path.exists():
        summary = _read_json(summary_path)
        observed = {
            (
                str(row["city"]),
                str(row["scenario"]),
                int(row["seed"]),
                str(row["policy"]),
            )
            for row in summary.get("rows", ())
        }
        if (
            summary.get("protocol") != SHARD_PROTOCOL
            or summary.get("passed") is not True
            or observed != set(identities)
            or any(summary.get(key) != value for key, value in hashes.items())
        ):
            raise RuntimeError(f"completed v75 shard is invalid: {summary_path}")
        for identity in identities:
            city, scenario, seed, policy = identity
            root = Path(results_root) / city / scenario / f"seed_{seed}" / policy
            _validate_existing(
                result_path=root / "result.json",
                tripinfo_path=root / "tripinfo.xml",
                identity=identity,
                hashes=hashes,
            )
        return {**summary, "validated_summary_reuse": True}
    tasks = []
    for city, scenario, seed, policy in identities:
        root = Path(results_root) / city / scenario / f"seed_{seed}" / policy
        tasks.append(
            {
                "identity": (city, scenario, seed, policy),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
                "protocol_path": str(protocol_path),
                "proposal_parent_protocol_path": str(proposal_parent_protocol_path),
                "hierarchical_freeze_audit_path": str(
                    hierarchical_freeze_audit_path
                ),
                "hierarchical_freeze_root": str(hierarchical_freeze_root),
                "veto_freeze_result_path": str(veto_freeze_result_path),
                "veto_joint_audit_path": str(veto_joint_audit_path),
                "veto_artifact_root": str(veto_artifact_root),
                "external_manifest_path": str(external_manifest_path),
                "conversion_root": str(conversion_root),
                "conversion_manifest_path": str(conversion_manifest_path),
                "hashes": hashes,
            }
        )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process_count = min(max(int(workers), 1), len(tasks))
    context = mp.get_context("spawn")
    rows = []
    with context.Pool(processes=process_count, maxtasksperchild=1) as pool:
        for completed, row in enumerate(
            pool.imap_unordered(_run_one, tasks, chunksize=1), start=1
        ):
            rows.append(row)
            print(
                f"CFCMT_PROGRESS {completed}/{len(tasks)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    order = {identity: index for index, identity in enumerate(identities)}
    rows.sort(
        key=lambda row: order[
            (row["city"], row["scenario"], int(row["seed"]), row["policy"])
        ]
    )
    observed = {
        (row["city"], row["scenario"], int(row["seed"]), row["policy"])
        for row in rows
    }
    payload = {
        "protocol": SHARD_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "hostname": socket.gethostname(),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "workers": process_count,
        "fresh_process_per_rollout": True,
        "task_count": len(tasks),
        **hashes,
        "elapsed_sec": float(time.monotonic() - started),
        "rows": rows,
        "passed": bool(rows) and observed == set(identities),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--veto-freeze-result", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--veto-artifact-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    payload = run_shard(
        protocol_path=args.protocol,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        veto_freeze_result_path=args.veto_freeze_result,
        veto_joint_audit_path=args.veto_joint_audit,
        veto_artifact_root=args.veto_artifact_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
        results_root=args.results_root,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "node": payload["hostname"],
                "shard": payload["shard_index"],
                "tasks": payload["task_count"],
                "elapsed_sec": payload["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
