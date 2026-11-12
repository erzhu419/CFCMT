#!/usr/bin/env python3
"""Execute the separately authorized 63-cell recovery of preempted shard 4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def identity(row):
    return row["scenario"], int(row["seed"]), row["method"]


def recovery_identities(manifest, first_attempt, original_shard):
    groups = {name: [identity(row) for row in first_attempt[name]]
              for name in ("completed", "interrupted", "unstarted")}
    if tuple(map(len, groups.values())) != (116, 4, 59):
        raise ValueError("First-attempt inventory must retain 116 complete, 4 interrupted, 59 unstarted")
    all_original = [key for values in groups.values() for key in values]
    if len(all_original) != len(set(all_original)) or set(all_original) != set(original_shard):
        raise ValueError("First-attempt inventory must partition the original 179 identities")
    complete, interrupted, unstarted = (set(groups[name]) for name in groups)
    selected = [identity(row) for row in manifest["execute"]]
    expected = [key for key in original_shard if key not in complete]
    if selected != expected or set(selected) != interrupted | unstarted:
        raise ValueError("Recovery must contain exactly the 4 interrupted and 59 unstarted identities")
    if [identity(row) for row in manifest["reuse_completed"]] != groups["completed"]:
        raise ValueError("Completed first-attempt records must all be retained without rerun")
    for row in manifest["execute"]:
        kind = "technical_retry" if identity(row) in interrupted else "first_execution"
        if row["execution_kind"] != kind:
            raise ValueError("Retry versus first-execution provenance changed")
    if first_attempt["task_id"] != manifest["original_task_id"] or first_attempt["task_id"] != "t213641":
        raise ValueError("Recovery refers to a different first-attempt task")
    return selected


def main():
    import run_tsc_v93_fresh_confirmation_shard as worker

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "first-attempt", "parent-launch", "protocol", "candidate-spec", "results-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    first_attempt = json.loads(args.first_attempt.read_text())
    source_root = Path(os.environ["CFCMT_SOURCE_ROOT"]).resolve()
    if Path(worker.evaluator.__file__).resolve() != source_root / "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py":
        raise ValueError("Recovery imported an evaluator outside the frozen CFCMT_SOURCE_ROOT")
    launch, protocol, candidate, seeds = worker.load_inputs(args.parent_launch, args.protocol, args.candidate_spec)
    original = worker.shard_identities(seeds, shard_index=4, shard_count=5)
    selected = recovery_identities(manifest, first_attempt, original)
    root = args.results_root.resolve()
    if str(root) != manifest["recovery_results_root"] or root.name != "evaluation_recovery_attempt2":
        raise ValueError("Recovery output must use its declared isolated attempt2 directory")
    summary_path = root / "_shards/recovery_4.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite recovery summary: {summary_path}")
    common = {
        "artifact_contract": launch["artifact_contract"], "candidate": candidate,
        "rollout_kwargs": worker.rollout_kwargs(launch), "results_root": str(root),
        "allowed_seeds": seeds, "stage": "prospective-fresh-seed-confirmation",
        "protocol_path": str(args.protocol.resolve()), "parent_launch": str(args.parent_launch.resolve()),
        "candidate_spec": str(args.candidate_spec.resolve()), "source_root": str(source_root),
    }
    payloads = [{**common, "identity": key} for key in selected]
    kinds = {identity(row): row["execution_kind"] for row in manifest["execute"]}
    started, rows = time.monotonic(), []
    def progress():
        print("SCHEDULEURM_PROGRESS " + json.dumps({"current": len(rows), "total": 63,
              "unit": "rollout", "elapsed_seconds": time.monotonic() - started}), flush=True)
    progress()
    with mp.get_context("spawn").Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(worker._run_one, payloads, chunksize=1):
            row["recovery_execution_kind"] = kinds[identity(row)]
            row["original_task_id"] = manifest["original_task_id"]
            rows.append(row)
            progress()
    if len(rows) != 63 or {identity(row) for row in rows} != set(selected):
        raise RuntimeError("Recovery coverage differs from the prescribed 63 identities")
    rows.sort(key=identity)
    summary = {
        "protocol": "tsc-v93-fresh-seed-confirmation-recovery-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
        "original_task_id": manifest["original_task_id"], "original_shard_index": 4,
        "original_shard_count": 5, "first_attempt_records": str(args.first_attempt.resolve()),
        "recovery_manifest": str(args.manifest.resolve()), "source_root": str(source_root),
        "evaluator_path": str(Path(worker.evaluator.__file__).resolve()),
        "task_count": len(rows), "retry_count": 4, "first_execution_count": 59,
        "preserved_completed_count": 116, "workers": args.workers,
        "elapsed_sec": time.monotonic() - started, "rows": rows,
    }
    worker._atomic_json(summary_path, summary)
    print(json.dumps({"status": "COMPLETE", "task_count": len(rows), "summary": str(summary_path)}))


if __name__ == "__main__":
    main()
