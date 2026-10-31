#!/usr/bin/env python3
"""Run independent offline component families concurrently for one target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


MODULE = "cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _family_command(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest: Path,
    target: str,
    family: str,
    result_path: Path,
    cache_workers: int,
    target_group_budget: int = 0,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE,
        "--cache-root",
        str(cache_root),
        "--baseline-result",
        str(baseline_result),
        "--manifest",
        str(manifest),
        "--targets",
        str(target),
        "--target-group-budget",
        str(int(target_group_budget)),
        "--model-families",
        str(family),
        "--workers",
        str(max(int(cache_workers), 1)),
        "--fit-parallel-workers",
        "1",
        "--target-workers",
        "1",
        "--target-executor",
        "process",
        "--fit-protocol",
        "screening",
        "--out",
        str(result_path),
    ]


def _validate_result(
    path: Path,
    *,
    target: str,
    family: str,
    target_group_budget: int = 0,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_budget = int(target_group_budget)
    if payload.get("target_group_budget") != expected_budget:
        raise ValueError(f"{path}: target budget mismatch")
    if payload.get("model_families") != [family]:
        raise ValueError(f"{path}: unexpected model families")
    targets = payload.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError(f"{path}: expected exactly one target result")
    row = targets[0]
    if row.get("target") != target:
        raise ValueError(f"{path}: unexpected target {row.get('target')!r}")
    if row.get("target_group_budget") != expected_budget:
        raise ValueError(f"{path}: target row budget mismatch")
    excluded_group_ids = row.get("excluded_group_ids")
    if not isinstance(excluded_group_ids, list):
        raise ValueError(f"{path}: excluded target groups are not a list")
    if len(excluded_group_ids) != len(set(str(value) for value in excluded_group_ids)):
        raise ValueError(f"{path}: duplicate excluded target groups")
    if len(excluded_group_ids) != expected_budget:
        raise ValueError(
            f"{path}: expected {expected_budget} excluded target groups, "
            f"found {len(excluded_group_ids)}"
        )
    diagnostics = row.get("model_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{path}: missing model diagnostics")
    adaptation_groups = int(diagnostics.get("target_adaptation_groups", -1))
    if not 0 <= adaptation_groups <= expected_budget:
        raise ValueError(f"{path}: invalid target adaptation group count")
    if (expected_budget == 0) != (adaptation_groups == 0):
        raise ValueError(f"{path}: target adaptation budget accounting mismatch")
    evaluation = row.get("offline_evaluation", {})
    family_row = evaluation.get("families", {}).get(family)
    if not isinstance(family_row, dict):
        raise ValueError(f"{path}: missing family evaluation")
    group_count = int(evaluation.get("evaluation_group_count", -1))
    if group_count <= 0 or int(family_row.get("group_count", -1)) != group_count:
        raise ValueError(f"{path}: action-group accounting mismatch")
    regret = float(family_row["mean_normalized_action_regret"])
    if not (regret >= 0.0 and regret < float("inf")):
        raise ValueError(f"{path}: invalid action regret")
    return {
        "family": family,
        "result_path": str(path),
        "result_sha256": _sha256(path),
        "evaluation_group_count": group_count,
        "excluded_group_count": len(excluded_group_ids),
        "target_adaptation_groups": adaptation_groups,
        "mean_normalized_action_regret": regret,
        "optimal_action_rate": float(family_row["optimal_action_rate"]),
    }


def run_shard(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest: Path,
    target: str,
    model_families: Sequence[str],
    out_root: Path,
    cache_workers: int,
    target_group_budget: int = 0,
) -> dict[str, Any]:
    if not model_families or len(set(model_families)) != len(model_families):
        raise ValueError("model families must be nonempty and unique")
    if int(target_group_budget) < 0:
        raise ValueError("target group budget must be nonnegative")
    for path in (cache_root, baseline_result, manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    target_root = out_root / target
    target_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    processes: list[dict[str, Any]] = []
    for family in model_families:
        family_root = target_root / family
        family_root.mkdir(parents=True, exist_ok=True)
        result_path = family_root / "result.json"
        log_path = family_root / "run.log"
        if result_path.exists() or log_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing family output: {family_root}"
            )
        command = _family_command(
            cache_root=cache_root,
            baseline_result=baseline_result,
            manifest=manifest,
            target=target,
            family=family,
            result_path=result_path,
            cache_workers=cache_workers,
            target_group_budget=target_group_budget,
        )
        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(
            {
                "family": family,
                "command": command,
                "process": process,
                "pid": int(process.pid),
                "log_handle": log_handle,
                "log_path": log_path,
                "result_path": result_path,
                "started_monotonic": time.monotonic(),
            }
        )

    _atomic_json(
        target_root / "launch_manifest.json",
        {
            "protocol": "parallel-independent-family-processes-v1",
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "started_at_utc": started_at,
            "target": target,
            "target_group_budget": int(target_group_budget),
            "model_families": list(model_families),
            "cache_root": str(cache_root.resolve()),
            "baseline_result": str(baseline_result.resolve()),
            "baseline_sha256": _sha256(baseline_result),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "children": [
                {
                    "family": row["family"],
                    "pid": row["pid"],
                    "command": row["command"],
                    "log_path": str(row["log_path"]),
                    "result_path": str(row["result_path"]),
                }
                for row in processes
            ],
        },
    )

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending = list(processes)
    while pending:
        for row in list(pending):
            returncode = row["process"].poll()
            if returncode is None:
                continue
            row["log_handle"].close()
            elapsed = float(time.monotonic() - row["started_monotonic"])
            status = {
                "family": row["family"],
                "pid": row["pid"],
                "returncode": int(returncode),
                "elapsed_sec": elapsed,
                "log_path": str(row["log_path"]),
                "result_path": str(row["result_path"]),
            }
            if int(returncode) == 0:
                try:
                    status.update(
                        _validate_result(
                            row["result_path"],
                            target=target,
                            family=row["family"],
                            target_group_budget=target_group_budget,
                        )
                    )
                    completed.append(status)
                except Exception as error:
                    status["validation_error"] = repr(error)
                    failures.append(status)
            else:
                failures.append(status)
            pending.remove(row)
            print(
                "CFCMT_COMPONENT_SHARD_PROGRESS "
                f"target={target} completed={len(completed) + len(failures)}/"
                f"{len(processes)} family={row['family']} rc={int(returncode)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if pending:
            time.sleep(2.0)

    payload = {
        "protocol": "parallel-independent-family-processes-v1",
        "hostname": socket.gethostname(),
        "target": target,
        "target_group_budget": int(target_group_budget),
        "model_families": list(model_families),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": float(time.monotonic() - started),
        "passed": not failures and len(completed) == len(processes),
        "completed": sorted(completed, key=lambda row: model_families.index(row["family"])),
        "failures": failures,
    }
    _atomic_json(target_root / "shard_result.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"component shard failed: {failures}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-families", nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--target-group-budget", type=int, default=0)
    args = parser.parse_args()
    payload = run_shard(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest=args.manifest,
        target=args.target,
        model_families=tuple(args.model_families),
        out_root=args.out_root,
        cache_workers=args.cache_workers,
        target_group_budget=args.target_group_budget,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
