#!/usr/bin/env python3
"""Run one independently placed v38 Salt Lake confirmation target."""

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

from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    CONFIRMATION_FAMILIES,
    TARGET_GROUP_BUDGET,
    TARGETS,
)


MODULE = "cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation"


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


def run_shard(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest: Path,
    protocol_spec: Path,
    target: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    out_root: Path,
    cache_workers: int,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(f"unknown frozen Salt Lake target: {target}")
    for path in (cache_root, baseline_result, manifest, protocol_spec):
        if not path.exists():
            raise FileNotFoundError(path)
    target_root = out_root / target
    target_root.mkdir(parents=True, exist_ok=True)
    result_path = target_root / "result.json"
    log_path = target_root / "run.log"
    launch_path = target_root / "launch_manifest.json"
    shard_path = target_root / "shard_result.json"
    if any(path.exists() for path in (result_path, log_path, launch_path, shard_path)):
        raise FileExistsError(f"refusing to overwrite Salt confirmation shard: {target_root}")
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--cache-root",
        str(cache_root),
        "--baseline-result",
        str(baseline_result),
        "--manifest",
        str(manifest),
        "--protocol-spec",
        str(protocol_spec),
        "--target",
        target,
        "--expected-cache-sha256",
        expected_cache_sha256,
        "--expected-baseline-sha256",
        expected_baseline_sha256,
        "--workers",
        str(max(int(cache_workers), 1)),
        "--out",
        str(result_path),
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        launch_path,
        {
            "protocol": "v38r34-single-target-global-pairwise-launch-v1",
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "started_at_utc": started_at,
            "target": target,
            "cache_root": str(cache_root.resolve()),
            "expected_cache_sha256": expected_cache_sha256,
            "baseline_result": str(baseline_result.resolve()),
            "expected_baseline_sha256": expected_baseline_sha256,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "protocol_spec": str(protocol_spec.resolve()),
            "protocol_spec_sha256": _sha256(protocol_spec),
            "command": command,
            "result_path": str(result_path),
            "log_path": str(log_path),
        },
    )
    started = time.monotonic()
    with log_path.open("wb") as log_handle:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    payload: dict[str, Any] = {
        "protocol": "v38r34-single-target-global-pairwise-shard-v1",
        "hostname": socket.gethostname(),
        "target": target,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": float(time.monotonic() - started),
        "returncode": int(completed.returncode),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "passed": False,
    }
    if completed.returncode == 0 and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluation = result.get("offline_evaluation", {})
        family_rows = evaluation.get("families", {})
        valid = (
            result.get("target") == target
            and int(result.get("target_group_budget", -1)) == TARGET_GROUP_BUDGET
            and len(result.get("selected_group_ids", ())) == TARGET_GROUP_BUDGET
            and result.get("model_families") == list(CONFIRMATION_FAMILIES)
            and set(family_rows) == set(CONFIRMATION_FAMILIES)
            and int(evaluation.get("evaluation_group_count", -1)) > 0
        )
        if valid:
            payload.update(
                {
                    "passed": True,
                    "result_sha256": _sha256(result_path),
                    "evaluation_group_count": int(
                        evaluation["evaluation_group_count"]
                    ),
                    "family_regrets": {
                        family: float(
                            family_rows[family]["mean_normalized_action_regret"]
                        )
                        for family in CONFIRMATION_FAMILIES
                    },
                }
            )
        else:
            payload["validation_error"] = "result contract mismatch"
    _atomic_json(shard_path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"Salt confirmation shard failed: {payload}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=16)
    args = parser.parse_args(argv)
    payload = run_shard(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest=args.manifest,
        protocol_spec=args.protocol_spec,
        target=args.target,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        out_root=args.out_root,
        cache_workers=args.cache_workers,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
