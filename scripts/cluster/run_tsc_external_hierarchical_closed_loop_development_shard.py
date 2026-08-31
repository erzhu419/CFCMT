#!/usr/bin/env python3
"""Run one process-isolated shard of the v64 closed-loop development matrix."""

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
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import _read_json
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
    run_hierarchical_closed_loop_development,
)


SHARD_PROTOCOL = "tsc-v64r60-hierarchical-closed-loop-development-process-shard-v1"


def shard_identities(
    protocol: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[tuple[str, str, int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid hierarchical development shard index/count")
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
    protocol_sha256: str,
) -> dict[str, Any]:
    if result_path.is_file() != tripinfo_path.is_file():
        raise RuntimeError(f"partial hierarchical result exists: {result_path.parent}")
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
        and result.get("protocol_sha256") == protocol_sha256
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and bool(result.get("metrics", {}).get("ok", False))
    ):
        raise RuntimeError(f"existing hierarchical result is invalid: {result_path}")
    return result


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    identity = tuple(payload["identity"])
    city, scenario, seed, policy = (
        str(identity[0]),
        str(identity[1]),
        int(identity[2]),
        str(identity[3]),
    )
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    started = time.monotonic()
    result = _validate_existing(
        result_path=result_path,
        tripinfo_path=tripinfo_path,
        identity=(city, scenario, seed, policy),
        protocol_sha256=str(payload["protocol_sha256"]),
    )
    reused = bool(result)
    if not result:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_hierarchical_closed_loop_development(
            protocol_path=Path(str(payload["protocol_path"])),
            parent_protocol_path=Path(str(payload["parent_protocol_path"])),
            offline_result_path=Path(str(payload["offline_result_path"])),
            support_audit_path=Path(str(payload["support_audit_path"])),
            freeze_audit_path=Path(str(payload["freeze_audit_path"])),
            freeze_root=Path(str(payload["freeze_root"])),
            environment_protocol_path=Path(
                str(payload["environment_protocol_path"])
            ),
            environment_parent_path=Path(str(payload["environment_parent_path"])),
            external_manifest_path=Path(str(payload["external_manifest_path"])),
            conversion_root=Path(str(payload["conversion_root"])),
            conversion_manifest_path=Path(
                str(payload["conversion_manifest_path"])
            ),
            scenario=scenario,
            seed=seed,
            policy=policy,
            tripinfo_out=tripinfo_path,
        )
        _atomic_json(result_path, result)
    metrics = dict(result["metrics"])
    guard = dict(metrics.get("guard_audit", {}))
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
    }


def _validated_completed_summary(
    *,
    summary_path: Path,
    identities: Sequence[tuple[str, str, int, str]],
    results_root: Path,
    protocol_sha256: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    rows = tuple(summary.get("rows", ()))
    row_index = {
        (
            str(row.get("city")),
            str(row.get("scenario")),
            int(row.get("seed", -1)),
            str(row.get("policy")),
        ): row
        for row in rows
    }
    if not (
        summary.get("protocol") == SHARD_PROTOCOL
        and summary.get("passed") is True
        and int(summary.get("shard_index", -1)) == int(shard_index)
        and int(summary.get("shard_count", -1)) == int(shard_count)
        and summary.get("protocol_sha256") == protocol_sha256
        and set(row_index) == set(identities)
    ):
        raise RuntimeError(f"completed hierarchical shard is invalid: {summary_path}")
    for identity in identities:
        city, scenario, seed, policy = identity
        root = Path(results_root) / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        _validate_existing(
            result_path=result_path,
            tripinfo_path=tripinfo_path,
            identity=identity,
            protocol_sha256=protocol_sha256,
        )
        row = row_index[identity]
        if (
            row.get("result_sha256") != _sha256(result_path)
            or row.get("tripinfo_sha256") != _sha256(tripinfo_path)
        ):
            raise RuntimeError(
                f"completed hierarchical shard row hash changed: {result_path}"
            )
    return {**summary, "validated_summary_reuse": True}


def run_shard(
    *,
    protocol_path: Path,
    parent_protocol_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
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
        raise ValueError("hierarchical development shard protocol changed")
    identities = shard_identities(
        protocol, shard_index=int(shard_index), shard_count=int(shard_count)
    )
    summary_path = Path(results_root) / "_shards" / f"shard_{shard_index}.json"
    protocol_sha = _sha256(protocol_path)
    if summary_path.exists():
        return _validated_completed_summary(
            summary_path=summary_path,
            identities=identities,
            results_root=results_root,
            protocol_sha256=protocol_sha,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    tasks = []
    for city, scenario, seed, policy in identities:
        root = Path(results_root) / city / scenario / f"seed_{seed}" / policy
        tasks.append(
            {
                "identity": (city, scenario, seed, policy),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha,
                "parent_protocol_path": str(parent_protocol_path),
                "offline_result_path": str(offline_result_path),
                "support_audit_path": str(support_audit_path),
                "freeze_audit_path": str(freeze_audit_path),
                "freeze_root": str(freeze_root),
                "environment_protocol_path": str(environment_protocol_path),
                "environment_parent_path": str(environment_parent_path),
                "external_manifest_path": str(external_manifest_path),
                "conversion_root": str(conversion_root),
                "conversion_manifest_path": str(conversion_manifest_path),
            }
        )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process_count = min(max(int(workers), 1), len(tasks))
    context = mp.get_context("spawn")
    rows = []
    with context.Pool(processes=process_count, maxtasksperchild=1) as pool:
        report_every = max(len(tasks) // 10, 1)
        for completed, row in enumerate(
            pool.imap_unordered(_run_one, tasks, chunksize=1), start=1
        ):
            rows.append(row)
            if completed == 1 or completed == len(tasks) or completed % report_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"CFCMT_PROGRESS {completed}/{len(tasks)} "
                    f"elapsed={elapsed:.1f}s rate={completed / elapsed:.3f}/s",
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
    passed = bool(rows) and observed == set(identities)
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
        "elapsed_sec": float(time.monotonic() - started),
        "protocol_sha256": protocol_sha,
        "parent_protocol_sha256": _sha256(parent_protocol_path),
        "offline_result_sha256": _sha256(offline_result_path),
        "support_audit_sha256": _sha256(support_audit_path),
        "freeze_audit_sha256": _sha256(freeze_audit_path),
        "passed": passed,
        "rows": rows,
    }
    _atomic_json(summary_path, payload)
    if not passed:
        raise RuntimeError(f"hierarchical shard coverage failed: {shard_index}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args(argv)
    payload = run_shard(
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
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
                "shard_index": payload["shard_index"],
                "task_count": payload["task_count"],
                "elapsed_sec": payload["elapsed_sec"],
                "passed": payload["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
