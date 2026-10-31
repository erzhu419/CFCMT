#!/usr/bin/env python3
"""Run one process-parallel shard of the frozen v50 development matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from cf_h2o.eval.traffic_signal_external_policy_horizon_expanded_development import (
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
    run_expanded_development_candidate,
)


SHARD_PROTOCOL = "tsc-v50r46-policy-horizon-expanded-development-process-shard-v1"


def shard_identities(
    protocol: Mapping[str, Any],
    adaptation: Mapping[str, Any],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[str, str, int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid v50 shard index/count")
    return tuple(
        identity
        for index, identity in enumerate(all_rollout_identities(protocol, adaptation))
        if index % shard_count == shard_index
    )


def _validate_existing(
    *,
    result_path: Path,
    tripinfo_path: Path,
    identity: tuple[str, str, int, str],
    expanded_protocol_sha256: str,
) -> dict[str, Any]:
    city, scenario, seed, candidate = identity
    if result_path.is_file() != tripinfo_path.is_file():
        raise RuntimeError(f"partial v50 result exists: {result_path.parent}")
    if not result_path.is_file():
        return {}
    result = _read_json(result_path)
    valid = (
        result.get("protocol") == RESULT_PROTOCOL
        and result.get("city") == city
        and result.get("scenario") == scenario
        and int(result.get("seed", -1)) == seed
        and result.get("policy") == candidate
        and result.get("expanded_development_protocol_sha256")
        == expanded_protocol_sha256
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and bool(result.get("metrics", {}).get("ok", False))
    )
    if not valid:
        raise RuntimeError(f"existing v50 result failed validation: {result_path}")
    return result


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    city, scenario, seed, candidate = tuple(payload["identity"])
    identity = (str(city), str(scenario), int(seed), str(candidate))
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    existing = _validate_existing(
        result_path=result_path,
        tripinfo_path=tripinfo_path,
        identity=identity,
        expanded_protocol_sha256=str(payload["expanded_protocol_sha256"]),
    )
    started = time.monotonic()
    if existing:
        result = existing
        reused = True
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_expanded_development_candidate(
            expanded_protocol_path=Path(str(payload["expanded_protocol_path"])),
            adaptation_protocol_path=Path(str(payload["adaptation_protocol_path"])),
            robustness_protocol_path=Path(str(payload["robustness_protocol_path"])),
            failed_v49_audit_path=Path(str(payload["failed_v49_audit_path"])),
            expected_failed_v49_audit_sha256=str(
                payload["expected_failed_v49_audit_sha256"]
            ),
            pressure_protocol_path=Path(str(payload["pressure_protocol_path"])),
            pressure_joint_audit_path=Path(
                str(payload["pressure_joint_audit_path"])
            ),
            expected_pressure_joint_audit_sha256=str(
                payload["expected_pressure_joint_audit_sha256"]
            ),
            pressure_freeze_root=Path(str(payload["pressure_freeze_root"])),
            candidate_key=str(candidate),
            scenario=str(scenario),
            seed=int(seed),
            rollout_kwargs={
                key: Path(str(value))
                if key.endswith("_path") or key.endswith("_root")
                else value
                for key, value in dict(payload["rollout_kwargs"]).items()
            }
            | {"tripinfo_out": tripinfo_path},
        )
        _atomic_json(result_path, result)
        reused = False
    metrics = dict(result["metrics"])
    return {
        "city": str(city),
        "scenario": str(scenario),
        "seed": int(seed),
        "candidate": str(candidate),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "reused": reused,
        "elapsed_sec": float(time.monotonic() - started),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "tripinfo_path": str(tripinfo_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
        "executed_overrides": int(
            metrics.get("guard_audit", {}).get(
                "executed_overrides",
                metrics.get("guard_audit", {}).get("accepted_overrides", 0),
            )
        ),
    }


def run_shard(
    *,
    expanded_protocol_path: Path,
    adaptation_protocol_path: Path,
    robustness_protocol_path: Path,
    failed_v49_audit_path: Path,
    expected_failed_v49_audit_sha256: str,
    pressure_protocol_path: Path,
    pressure_joint_audit_path: Path,
    expected_pressure_joint_audit_sha256: str,
    pressure_freeze_root: Path,
    results_root: Path,
    shard_index: int,
    shard_count: int,
    workers: int,
    rollout_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _read_json(expanded_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v50 expanded-development shard protocol changed")
    identities = shard_identities(
        protocol,
        adaptation,
        shard_index=int(shard_index),
        shard_count=int(shard_count),
    )
    summary_path = Path(results_root) / "_shards" / f"shard_{shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite v50 shard audit: {summary_path}")
    expanded_sha = _sha256(expanded_protocol_path)
    tasks = []
    for city, scenario, seed, candidate in identities:
        root = Path(results_root) / scenario / f"seed_{seed}" / candidate
        tasks.append(
            {
                "identity": (city, scenario, seed, candidate),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
                "expanded_protocol_path": str(expanded_protocol_path),
                "expanded_protocol_sha256": expanded_sha,
                "adaptation_protocol_path": str(adaptation_protocol_path),
                "robustness_protocol_path": str(robustness_protocol_path),
                "failed_v49_audit_path": str(failed_v49_audit_path),
                "expected_failed_v49_audit_sha256": expected_failed_v49_audit_sha256,
                "pressure_protocol_path": str(pressure_protocol_path),
                "pressure_joint_audit_path": str(pressure_joint_audit_path),
                "expected_pressure_joint_audit_sha256": expected_pressure_joint_audit_sha256,
                "pressure_freeze_root": str(pressure_freeze_root),
                "rollout_kwargs": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in rollout_kwargs.items()
                },
            }
        )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process_count = min(max(int(workers), 1), len(tasks))
    with ProcessPoolExecutor(
        max_workers=process_count, mp_context=mp.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_one, tasks, chunksize=1))
    observed = {
        (row["city"], row["scenario"], int(row["seed"]), row["candidate"])
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
        "task_count": len(tasks),
        "elapsed_sec": float(time.monotonic() - started),
        "expanded_protocol_sha256": expanded_sha,
        "adaptation_protocol_sha256": _sha256(adaptation_protocol_path),
        "robustness_protocol_sha256": _sha256(robustness_protocol_path),
        "failed_v49_audit_sha256": _sha256(failed_v49_audit_path),
        "pressure_joint_audit_sha256": _sha256(pressure_joint_audit_path),
        "passed": passed,
        "rows": rows,
    }
    _atomic_json(summary_path, payload)
    if not passed:
        raise RuntimeError(f"v50 shard coverage failed: {shard_index}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--failed-v49-audit", type=Path, required=True)
    parser.add_argument("--expected-failed-v49-audit-sha256", required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-pressure-joint-audit-sha256", required=True)
    parser.add_argument("--pressure-freeze-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--expected-offline-authorization-sha256", required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-manifest-sha256", required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--expected-conversion-manifest-sha256", required=True)
    parser.add_argument("--expected-conversion-tree-sha256", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args(argv)
    payload = run_shard(
        expanded_protocol_path=args.expanded_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        robustness_protocol_path=args.robustness_protocol,
        failed_v49_audit_path=args.failed_v49_audit,
        expected_failed_v49_audit_sha256=args.expected_failed_v49_audit_sha256,
        pressure_protocol_path=args.pressure_protocol,
        pressure_joint_audit_path=args.pressure_joint_audit,
        expected_pressure_joint_audit_sha256=args.expected_pressure_joint_audit_sha256,
        pressure_freeze_root=args.pressure_freeze_root,
        results_root=args.results_root,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        workers=args.workers,
        rollout_kwargs={
            "protocol_spec_path": args.protocol_spec,
            "offline_authorization_path": args.offline_authorization,
            "expected_offline_authorization_sha256": args.expected_offline_authorization_sha256,
            "joint_freeze_audit_path": args.joint_freeze_audit,
            "expected_joint_freeze_sha256": args.expected_joint_freeze_sha256,
            "freeze_root": args.freeze_root,
            "external_manifest_path": args.external_manifest,
            "conversion_root": args.conversion_root,
            "expected_external_manifest_sha256": args.expected_external_manifest_sha256,
            "conversion_manifest_path": args.conversion_manifest,
            "expected_conversion_manifest_sha256": args.expected_conversion_manifest_sha256,
            "expected_conversion_tree_sha256": args.expected_conversion_tree_sha256,
            "expected_sumo_version": args.expected_sumo_version,
        },
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
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
