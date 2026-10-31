"""Run one CPU shard with an independent original-runtime process per rollout."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def run_child(index):
    outcome = None
    try:
        # Inherit the wrapper's original-snapshot cwd, Python and environment.
        outcome = subprocess.run([sys.executable, "-u", str(HERE / "run.py"),
            "--index", str(index)], capture_output=True, text=True, check=False)
        if outcome.returncode:
            raise RuntimeError(f"Rollout process exited {outcome.returncode}")
        row = json.loads((HERE / "summaries" / f"run_{index:03d}" / "summary.json").read_text())
        if row.get("complete") is not True or row.get("run_index") != index:
            raise RuntimeError("Child summary is incomplete or has the wrong rollout index")
        return row, None
    except Exception as error:
        failure = dict(index=index, error=str(error),
            returncode=outcome.returncode if outcome is not None else None)
        if outcome is not None:
            for name in ("stdout", "stderr"):
                lines = [line for line in getattr(outcome, name).splitlines()
                    if "CFCMT_PROGRESS" not in line and "SCHEDULEURM_PROGRESS" not in line]
                failure[name + "_tail"] = "\n".join(lines)[-3000:]
        print(json.dumps(dict(rollout_failed=failure)), flush=True)
        return None, failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args(argv)
    if not (0 <= args.start < args.end <= 80 and args.workers > 0 and args.shard_index >= 0):
        parser.error("Require 0 <= start < end <= 80, positive workers and nonnegative shard index")
    total, finished, started = args.end - args.start, 0, time.monotonic()
    records, failures = [], []

    def progress():
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=finished,
            total=total, unit="rollout", elapsed_seconds=time.monotonic() - started)),
            flush=True)

    progress()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_child, index) for index in range(args.start, args.end)}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row, failure = future.result()
                (records if failure is None else failures).append(row if failure is None else failure)
                finished += 1
                progress()
            if not done:
                progress()
    output = dict(records=sorted(records, key=lambda row: row["run_index"]),
        failed_indices=sorted(row["index"] for row in failures), failures=failures,
        complete=not failures, count=len(records), finished_count=finished, expected=total,
        start=args.start, end=args.end, shard_index=args.shard_index,
        elapsed_seconds=time.monotonic() - started)
    path = HERE / "shard_summaries" / f"shard_{args.shard_index}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
