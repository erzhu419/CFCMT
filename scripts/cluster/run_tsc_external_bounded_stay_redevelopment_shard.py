#!/usr/bin/env python3
"""Run one process-isolated shard of the frozen v88 development matrix."""

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
from cf_h2o.eval.traffic_signal_bounded_stay_redevelopment import (
    RESULT_PROTOCOL,
    all_rollout_identities,
    run_bounded_stay_redevelopment,
)
from scripts.cluster.freeze_tsc_external_v9_bounded_stay_redevelopment import (
    PROTOCOL,
)


SHARD_PROTOCOL = "tsc-v88r84-bounded-stay-process-shard-v1"


def shard_identities(
    protocol: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[tuple[str, str, int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid v88 shard index/count")
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
        raise RuntimeError(f"partial v88 result exists: {result_path.parent}")
    if not result_path.is_file():
        return {}
    city, scenario, seed, policy = identity
    result = _read_json(result_path)
    guard = dict(result.get("metrics", {}).get("guard_audit", {}))
    alignment = dict(result.get("intervention_alignment", {}))
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
        and alignment.get("passed") is True
        and int(alignment.get("effective_override_count", -1))
        == int(guard.get("effective_overrides", -2))
    ):
        raise RuntimeError(f"existing v88 result is invalid: {result_path}")
    return result


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    city, scenario, raw_seed, policy = tuple(payload["identity"])
    identity = (str(city), str(scenario), int(raw_seed), str(policy))
    city, scenario, seed, policy = identity
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    started = time.monotonic()
    result = _validate_existing(
        result_path=result_path,
        tripinfo_path=tripinfo_path,
        identity=identity,
        protocol_sha256=str(payload["protocol_sha256"]),
    )
    reused = bool(result)
    if not result:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_bounded_stay_redevelopment(
            protocol_path=Path(str(payload["protocol_path"])),
            v86_protocol_path=Path(str(payload["v86_protocol_path"])),
            v86_audit_path=Path(str(payload["v86_audit_path"])),
            v87_protocol_path=Path(str(payload["v87_protocol_path"])),
            v87_audit_path=Path(str(payload["v87_audit_path"])),
            v83_protocol_path=Path(str(payload["v83_protocol_path"])),
            v83_audit_path=Path(str(payload["v83_audit_path"])),
            v84_protocol_path=Path(str(payload["v84_protocol_path"])),
            v84_audit_path=Path(str(payload["v84_audit_path"])),
            v81_protocol_path=Path(str(payload["v81_protocol_path"])),
            v81_audit_path=Path(str(payload["v81_audit_path"])),
            artifact_protocol_path=Path(str(payload["artifact_protocol_path"])),
            artifact_result_path=Path(str(payload["artifact_result_path"])),
            artifact_audit_path=Path(str(payload["artifact_audit_path"])),
            artifact_root=Path(str(payload["artifact_root"])),
            parent_runtime_protocol_path=Path(
                str(payload["parent_runtime_protocol_path"])
            ),
            hierarchical_parent_path=Path(
                str(payload["hierarchical_parent_path"])
            ),
            offline_result_path=Path(str(payload["offline_result_path"])),
            support_audit_path=Path(str(payload["support_audit_path"])),
            hierarchical_freeze_audit_path=Path(
                str(payload["hierarchical_freeze_audit_path"])
            ),
            hierarchical_freeze_root=Path(
                str(payload["hierarchical_freeze_root"])
            ),
            environment_protocol_path=Path(
                str(payload["environment_protocol_path"])
            ),
            environment_parent_path=Path(
                str(payload["environment_parent_path"])
            ),
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
    guard = dict(metrics["guard_audit"])
    return {
        "city": city,
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "hostname": socket.gethostname(),
        "pid": int(result.get("pid", 0)),
        "reused": reused,
        "elapsed_sec": float(time.monotonic() - started),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "tripinfo_path": str(tripinfo_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
        "effective_overrides": int(guard["effective_overrides"]),
        "switch_overrides": int(guard["executed_switch_overrides"]),
        "stay_overrides": int(guard["executed_stay_overrides"]),
        "collision_incidents": int(metrics["collision_incidents"]),
        "starting_teleports": int(metrics["starting_teleports"]),
        "ending_teleports": int(metrics["ending_teleports"]),
        "intervention_alignment": result["intervention_alignment"],
    }


def _prepare_spawn_working_directory() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    os.chdir(source_root)
    return source_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v86-audit", type=Path, required=True)
    parser.add_argument("--v87-protocol", type=Path, required=True)
    parser.add_argument("--v87-audit", type=Path, required=True)
    parser.add_argument("--v83-protocol", type=Path, required=True)
    parser.add_argument("--v83-audit", type=Path, required=True)
    parser.add_argument("--v84-protocol", type=Path, required=True)
    parser.add_argument("--v84-audit", type=Path, required=True)
    parser.add_argument("--v81-protocol", type=Path, required=True)
    parser.add_argument("--v81-audit", type=Path, required=True)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-runtime-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-parent", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)

    protocol = _read_json(args.protocol)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v88 shard protocol changed")
    identities = shard_identities(
        protocol,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    workers = min(max(int(args.workers), 1), len(identities))
    protocol_sha = _sha256(args.protocol)
    summary_path = args.results_root / "_shards" / f"shard_{args.shard_index}.json"
    common = {
        "protocol_path": str(args.protocol),
        "protocol_sha256": protocol_sha,
        "v86_protocol_path": str(args.v86_protocol),
        "v86_audit_path": str(args.v86_audit),
        "v87_protocol_path": str(args.v87_protocol),
        "v87_audit_path": str(args.v87_audit),
        "v83_protocol_path": str(args.v83_protocol),
        "v83_audit_path": str(args.v83_audit),
        "v84_protocol_path": str(args.v84_protocol),
        "v84_audit_path": str(args.v84_audit),
        "v81_protocol_path": str(args.v81_protocol),
        "v81_audit_path": str(args.v81_audit),
        "artifact_protocol_path": str(args.artifact_protocol),
        "artifact_result_path": str(args.artifact_result),
        "artifact_audit_path": str(args.artifact_audit),
        "artifact_root": str(args.artifact_root),
        "parent_runtime_protocol_path": str(args.parent_runtime_protocol),
        "hierarchical_parent_path": str(args.hierarchical_parent),
        "offline_result_path": str(args.offline_result),
        "support_audit_path": str(args.support_audit),
        "hierarchical_freeze_audit_path": str(args.hierarchical_freeze_audit),
        "hierarchical_freeze_root": str(args.hierarchical_freeze_root),
        "environment_protocol_path": str(args.environment_protocol),
        "environment_parent_path": str(args.environment_parent),
        "external_manifest_path": str(args.external_manifest),
        "conversion_root": str(args.conversion_root),
        "conversion_manifest_path": str(args.conversion_manifest),
    }
    payloads = []
    for identity in identities:
        city, scenario, seed, policy = identity
        root = args.results_root / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        existing = _validate_existing(
            result_path=result_path,
            tripinfo_path=tripinfo_path,
            identity=identity,
            protocol_sha256=protocol_sha,
        )
        if not existing:
            payloads.append(
                {
                    **common,
                    "identity": identity,
                    "result_path": str(result_path),
                    "tripinfo_path": str(tripinfo_path),
                }
            )

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    spawn_working_directory = _prepare_spawn_working_directory()
    rows = []
    if payloads:
        with mp.get_context("spawn").Pool(
            processes=min(workers, len(payloads)), maxtasksperchild=1
        ) as pool:
            for completed, row in enumerate(
                pool.imap_unordered(_run_one, payloads, chunksize=1), start=1
            ):
                rows.append(row)
                print(
                    f"CFCMT_PROGRESS {completed}/{len(payloads)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    for identity in identities:
        if any(
            (row["city"], row["scenario"], row["seed"], row["policy"])
            == identity
            for row in rows
        ):
            continue
        city, scenario, seed, policy = identity
        root = args.results_root / city / scenario / f"seed_{seed}" / policy
        rows.append(
            _run_one(
                {
                    **common,
                    "identity": identity,
                    "result_path": str(root / "result.json"),
                    "tripinfo_path": str(root / "tripinfo.xml"),
                }
            )
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
    summary = {
        "protocol": SHARD_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "hostname": socket.gethostname(),
        "spawn_working_directory": str(spawn_working_directory),
        "protocol_sha256": protocol_sha,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "workers": workers,
        "fresh_process_per_rollout": True,
        "task_count": len(identities),
        "elapsed_sec": float(time.monotonic() - started),
        "passed": passed,
        "rows": rows,
    }
    if summary_path.exists():
        existing_summary = _read_json(summary_path)
        if existing_summary.get("passed") is not True:
            raise RuntimeError(f"invalid existing v88 shard summary: {summary_path}")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(summary_path, summary)
    if not passed:
        raise RuntimeError(f"v88 shard coverage failed: {args.shard_index}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "shard_index": args.shard_index,
                "task_count": len(identities),
                "workers": workers,
                "elapsed_sec": summary["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
