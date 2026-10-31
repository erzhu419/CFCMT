#!/usr/bin/env python3
"""Run one independently placed v41 seed-heldout ensemble target."""

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

from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import CANDIDATE_KEYS
from cf_h2o.eval.traffic_signal_seed_heldout_anchored_development import (
    ADAPTATION_SEEDS,
    EVALUATION_SEEDS,
    FOLD_COUNT,
    GROUPS_PER_FOLD,
    TARGET_GROUP_BUDGET,
    TRAINING_GROUPS_PER_FOLD,
)


MODULE = "cf_h2o.eval.traffic_signal_seed_heldout_anchored_development"


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
        raise FileExistsError(f"refusing to overwrite v41 shard: {target_root}")
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
            "protocol": "v41r37-single-target-seed-heldout-ensemble-launch-v1",
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
        "protocol": "v41r37-single-target-seed-heldout-ensemble-shard-v1",
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
        selected = result.get("selected_group_ids", ())
        evaluation = result.get("evaluation_group_ids", ())
        folds = result.get("fold_results", ())
        oof = result.get("oof_candidates", {})
        ensemble = result.get("ensemble_evaluation", {})
        fold_contract = all(
            len(fold.get("training_group_ids", ())) == TRAINING_GROUPS_PER_FOLD
            and len(fold.get("validation_group_ids", ())) == GROUPS_PER_FOLD
            and set(fold.get("oof_candidate_metrics", {})) == set(CANDIDATE_KEYS)
            and set(fold.get("heldout_seed_candidate_metrics", {}))
            == set(CANDIDATE_KEYS)
            for fold in folds
        )
        valid = (
            result.get("target") == target
            and int(result.get("target_group_budget", -1)) == TARGET_GROUP_BUDGET
            and tuple(result.get("adaptation_seeds", ())) == ADAPTATION_SEEDS
            and tuple(result.get("evaluation_seeds", ())) == EVALUATION_SEEDS
            and len(selected) == TARGET_GROUP_BUDGET
            and len(set(selected)) == TARGET_GROUP_BUDGET
            and bool(evaluation)
            and not set(selected) & set(evaluation)
            and len(folds) == FOLD_COUNT
            and fold_contract
            and int(result.get("oof_candidate_count", -1)) == len(CANDIDATE_KEYS)
            and set(oof) == set(CANDIDATE_KEYS)
            and int(ensemble.get("model_ensemble_size", -1)) == FOLD_COUNT
            and int(ensemble.get("evaluation_group_count", -1)) == len(evaluation)
            and set(ensemble.get("candidates", {})) == set(CANDIDATE_KEYS)
        )
        if valid:
            payload.update(
                {
                    "passed": True,
                    "result_sha256": _sha256(result_path),
                    "fold_count": len(folds),
                    "candidate_count": len(oof),
                    "evaluation_group_count": len(evaluation),
                }
            )
        else:
            payload["validation_error"] = "result contract mismatch"
    _atomic_json(shard_path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"v41 seed-heldout ensemble shard failed: {payload}")
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
