#!/usr/bin/env python3
"""Run one process-isolated shard of the frozen V117 confirmation matrix."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_rollout import (
    RESULT_PROTOCOL,
    STATIC_INPUT_AUDIT_PROTOCOL,
    audit_static_inputs,
    run_confirmation_rollout,
)


SHARD_PROTOCOL = "tsc-v117-pure-waiting-confirmation-shard-v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def shard_identities(
    freeze: Mapping[str, Any], *, shard_index: int, shard_count: int
) -> tuple[tuple[int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid V117 shard index/count")
    arms = tuple(str(value) for value in freeze["active_arms"])
    seeds = tuple(int(value) for value in freeze["fresh_seeds"])
    identities = tuple(
        (seed, arm)
        for index, seed in enumerate(seeds)
        if index % shard_count == shard_index
        for arm in arms
    )
    if len(identities) != len(set(identities)):
        raise ValueError("V117 shard identities are not unique")
    return identities


def _validate_existing(
    *, result_path: Path, tripinfo_path: Path, seed: int, arm: str, freeze_sha: str
) -> dict[str, Any]:
    if result_path.is_file() != tripinfo_path.is_file():
        raise RuntimeError(f"partial V117 result exists: {result_path.parent}")
    if not result_path.is_file():
        return {}
    result = _read_json(result_path)
    metrics = dict(result.get("metrics", {}))
    if (
        result.get("protocol") != RESULT_PROTOCOL
        or int(result.get("seed", -1)) != int(seed)
        or result.get("arm") != arm
        or result.get("freeze_sha256") != freeze_sha
        or result.get("tripinfo_evidence", {}).get("sha256")
        != _sha256(tripinfo_path)
        or not bool(metrics.get("ok", False))
        or int(metrics.get("starting_teleports", -1)) != 0
        or int(metrics.get("ending_teleports", -1)) != 0
    ):
        raise RuntimeError(f"invalid existing V117 result: {result_path}")
    return result


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(payload["seed"])
    arm = str(payload["arm"])
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    started = time.monotonic()
    result = _validate_existing(
        result_path=result_path,
        tripinfo_path=tripinfo_path,
        seed=seed,
        arm=arm,
        freeze_sha=str(payload["freeze_sha256"]),
    )
    reused = bool(result)
    if not result:
        result = run_confirmation_rollout(
            freeze_path=Path(str(payload["freeze_path"])),
            expected_freeze_sha256=str(payload["freeze_sha256"]),
            static_input_audit_path=Path(str(payload["static_input_audit_path"])),
            expected_static_input_audit_sha256=str(
                payload["static_input_audit_sha256"]
            ),
            manifest_path=Path(str(payload["manifest_path"])),
            arm=arm,
            seed=seed,
            tripinfo_out=tripinfo_path,
        )
        _atomic_json(result_path, result)
    metrics = dict(result["metrics"])
    guard = dict(metrics.get("guard_audit", {}))
    return {
        "seed": seed,
        "arm": arm,
        "hostname": socket.gethostname(),
        "reused": reused,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "tripinfo_path": str(tripinfo_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
        "collision_incidents": int(metrics["collision_incidents"]),
        "starting_teleports": int(metrics["starting_teleports"]),
        "ending_teleports": int(metrics["ending_teleports"]),
        "effective_overrides": int(guard.get("effective_overrides", 0)),
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)
    freeze = _read_json(args.freeze)
    if (
        _sha256(args.freeze) != str(args.freeze_sha256)
        or freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decision") != AUTHORIZATION_DECISION
        or int(freeze.get("matrix_size", -1))
        != len(freeze.get("fresh_seeds", ())) * len(freeze.get("active_arms", ()))
        or not 1 <= int(args.workers) <= 20
    ):
        raise ValueError("V117 shard freeze or worker contract changed")
    identities = shard_identities(
        freeze,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    summary_path = args.results_root / "_shards" / f"shard_{args.shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"V117 shard summary already exists: {summary_path}")
    audit_path = (
        args.results_root
        / "_input_audits"
        / f"shard_{args.shard_index}.json"
    )
    if audit_path.exists():
        audit = _read_json(audit_path)
        if (
            audit.get("protocol") != STATIC_INPUT_AUDIT_PROTOCOL
            or audit.get("status") != "PASS"
            or audit.get("hostname") != socket.gethostname()
            or audit.get("freeze_sha256") != str(args.freeze_sha256)
        ):
            raise RuntimeError("existing V117 shard input audit is invalid")
    else:
        audit = audit_static_inputs(
            freeze_path=args.freeze,
            expected_freeze_sha256=args.freeze_sha256,
            manifest_path=args.manifest,
            conversion_root=args.conversion_root,
            conversion_manifest_path=args.conversion_manifest,
        )
        _atomic_json(audit_path, audit)
    audit_sha = _sha256(audit_path)
    common = {
        "freeze_path": str(args.freeze),
        "freeze_sha256": str(args.freeze_sha256),
        "static_input_audit_path": str(audit_path),
        "static_input_audit_sha256": audit_sha,
        "manifest_path": str(args.manifest),
    }
    payloads = []
    for seed, arm in identities:
        root = args.results_root / f"seed_{seed}" / arm
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        if not _validate_existing(
            result_path=result_path,
            tripinfo_path=tripinfo_path,
            seed=seed,
            arm=arm,
            freeze_sha=str(args.freeze_sha256),
        ):
            payloads.append(
                {
                    **common,
                    "seed": seed,
                    "arm": arm,
                    "result_path": str(result_path),
                    "tripinfo_path": str(tripinfo_path),
                }
            )
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    source_root = Path(__file__).resolve().parents[2]
    os.chdir(source_root)
    rows = []
    if payloads:
        workers = min(max(int(args.workers), 1), len(payloads))
        with mp.get_context("spawn").Pool(
            processes=workers, maxtasksperchild=1
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
    observed = {(int(row["seed"]), str(row["arm"])) for row in rows}
    for seed, arm in identities:
        if (seed, arm) in observed:
            continue
        root = args.results_root / f"seed_{seed}" / arm
        rows.append(
            _run_one(
                {
                    **common,
                    "seed": seed,
                    "arm": arm,
                    "result_path": str(root / "result.json"),
                    "tripinfo_path": str(root / "tripinfo.xml"),
                }
            )
        )
    order = {identity: index for index, identity in enumerate(identities)}
    rows.sort(key=lambda row: order[(int(row["seed"]), str(row["arm"]))])
    if {(int(row["seed"]), str(row["arm"])) for row in rows} != set(identities):
        raise RuntimeError("V117 shard result matrix is incomplete")
    summary = {
        "protocol": SHARD_PROTOCOL,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "source_root": str(source_root),
        "freeze_path": str(args.freeze),
        "freeze_sha256": str(args.freeze_sha256),
        "static_input_audit_path": str(audit_path),
        "static_input_audit_sha256": audit_sha,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "worker_count": min(max(int(args.workers), 1), len(identities)),
        "identity_count": len(identities),
        "rows": rows,
        "elapsed_sec": float(time.monotonic() - started),
    }
    _atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "shard": args.shard_index,
                "identity_count": len(identities),
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
