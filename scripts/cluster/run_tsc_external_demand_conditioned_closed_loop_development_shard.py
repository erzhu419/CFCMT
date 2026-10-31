#!/usr/bin/env python3
"""Run one process-isolated shard of the v71 development matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_demand_conditioned_closed_loop_development import (
    PROTOCOL,
    RESULT_PROTOCOL,
    all_method_rollout_identities,
    run_demand_conditioned_closed_loop_development,
)


SHARD_PROTOCOL = "tsc-v71r67-demand-conditioned-development-process-shard-v1"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def shard_identities(
    protocol: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[tuple[str, str, int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid v71 shard index/count")
    return tuple(
        identity
        for index, identity in enumerate(all_method_rollout_identities(protocol))
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
        raise RuntimeError(f"partial v71 rollout exists: {result_path.parent}")
    if not result_path.is_file():
        return {}
    city, scenario, seed, policy = identity
    result = _read(result_path)
    if not (
        result.get("protocol") == RESULT_PROTOCOL
        and (
            str(result.get("city")),
            str(result.get("scenario")),
            int(result.get("seed", -1)),
            str(result.get("policy")),
        )
        == (city, scenario, seed, policy)
        and result.get("protocol_sha256") == hashes["protocol_sha256"]
        and result.get("parent_protocol_sha256")
        == hashes["parent_protocol_sha256"]
        and result.get("collection_protocol_sha256")
        == hashes["collection_protocol_sha256"]
        and result.get("proposal_parent_protocol_sha256")
        == hashes["proposal_parent_protocol_sha256"]
        and result.get("hierarchical_freeze_audit_sha256")
        == hashes["freeze_audit_sha256"]
        and result.get("target_veto_joint_audit_sha256")
        == hashes["veto_joint_audit_sha256"]
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and bool(result.get("metrics", {}).get("ok", False))
    ):
        raise RuntimeError(f"existing v71 rollout is invalid: {result_path}")
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
        result = run_demand_conditioned_closed_loop_development(
            protocol_path=Path(str(payload["protocol_path"])),
            training_protocol_path=Path(str(payload["training_protocol_path"])),
            proposal_parent_protocol_path=Path(
                str(payload["proposal_parent_protocol_path"])
            ),
            freeze_audit_path=Path(str(payload["freeze_audit_path"])),
            freeze_root=Path(str(payload["freeze_root"])),
            training_result_path=Path(str(payload["training_result_path"])),
            veto_joint_audit_path=Path(str(payload["veto_joint_audit_path"])),
            veto_artifact_root=Path(str(payload["veto_artifact_root"])),
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
        "rejected_long_horizon_veto": int(
            guard.get("rejected_long_horizon_veto", 0)
        ),
    }


def _validated_summary(
    *,
    summary_path: Path,
    identities: Sequence[tuple[str, str, int, str]],
    results_root: Path,
    hashes: Mapping[str, str],
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    summary = _read(summary_path)
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
        and all(summary.get(key) == value for key, value in hashes.items())
        and set(row_index) == set(identities)
    ):
        raise RuntimeError(f"completed v71 shard is invalid: {summary_path}")
    for identity in identities:
        city, scenario, seed, policy = identity
        root = results_root / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        _validate_existing(
            result_path=result_path,
            tripinfo_path=tripinfo_path,
            identity=identity,
            hashes=hashes,
        )
        row = row_index[identity]
        if (
            row.get("result_sha256") != _sha256(result_path)
            or row.get("tripinfo_sha256") != _sha256(tripinfo_path)
        ):
            raise RuntimeError(f"completed v71 row hash changed: {result_path}")
    return {**summary, "validated_summary_reuse": True}


def run_shard(
    *,
    protocol_path: Path,
    training_protocol_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    training_result_path: Path,
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
    protocol = _read(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v71 development shard protocol changed")
    identities = shard_identities(
        protocol, shard_index=int(shard_index), shard_count=int(shard_count)
    )
    hashes = {
        "protocol_sha256": _sha256(protocol_path),
        "parent_protocol_sha256": _sha256(training_protocol_path),
        "collection_protocol_sha256": _sha256(training_result_path),
        "proposal_parent_protocol_sha256": _sha256(
            proposal_parent_protocol_path
        ),
        "freeze_audit_sha256": _sha256(freeze_audit_path),
        "veto_joint_audit_sha256": _sha256(veto_joint_audit_path),
    }
    summary_path = results_root / "_shards" / f"shard_{shard_index}.json"
    if summary_path.exists():
        return _validated_summary(
            summary_path=summary_path,
            identities=identities,
            results_root=results_root,
            hashes=hashes,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    tasks = []
    for city, scenario, seed, policy in identities:
        root = results_root / city / scenario / f"seed_{seed}" / policy
        tasks.append(
            {
                "identity": (city, scenario, seed, policy),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
                "protocol_path": str(protocol_path),
                "training_protocol_path": str(training_protocol_path),
                "proposal_parent_protocol_path": str(
                    proposal_parent_protocol_path
                ),
                "freeze_audit_path": str(freeze_audit_path),
                "freeze_root": str(freeze_root),
                "training_result_path": str(training_result_path),
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
    rows = []
    with mp.get_context("spawn").Pool(
        processes=process_count, maxtasksperchild=1
    ) as pool:
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
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--veto-artifact-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    payload = run_shard(
        protocol_path=args.protocol,
        training_protocol_path=args.training_protocol,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        training_result_path=args.training_result,
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
                "shard_index": payload["shard_index"],
                "task_count": payload["task_count"],
                "elapsed_sec": payload["elapsed_sec"],
                "passed": payload["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
