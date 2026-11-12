"""Run the frozen Hangzhou paired response roster serially, stopping on first failure."""
import argparse
import json
from pathlib import Path
import traceback

from measure_arm import measure_arm


def save(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def run(roster_path, cfg, directory):
    roster = json.loads(roster_path.read_text())
    points = {tuple(p) for p in roster["count_pairs"]}
    cases = roster["cases"]
    expected = {(s, r, p) for s in (32061101, 32061102) for r, p in points}
    if (roster["status"] != "FROZEN_FOR_SEPARATE_NOT_RUN_READOUT" or len(points) != 7
            or len(cases) != 14 or {(c["seed"], c["right_near"], c["straight_near"])
                                    for c in cases} != expected):
        raise ValueError("Input does not contain the frozen fourteen paired cases")
    directory.mkdir(parents=True, exist_ok=False)
    pairs = []
    receipt = dict(status="RUNNING", planned_pairs=len(cases), completed_pairs=0,
                   prefix_replays_started=0, branches_started=0, branches_completed=0,
                   matching_starts=0, fits=0, closed_loop_policy_evaluations=0)
    save(directory / "pairs.json", pairs)
    save(directory / "result.json", receipt)
    try:
        for index, case in enumerate(cases, 1):
            reference = None
            row = dict(case)
            for arm in ("high", "low"):
                cell = directory / f"case_{index:02d}" / arm
                cell.mkdir(parents=True)
                receipt["last_attempt"] = dict(index=index, seed=case["seed"],
                                               time_sec=case["time_sec"], arm=arm)

                def checkpoint(start):
                    nonlocal reference
                    if arm == "high":
                        reference = start
                    elif start != reference:
                        keys = sorted(k for k in start.keys() | reference.keys()
                                      if start.get(k) != reference.get(k))
                        raise ValueError(f"Paired checkpoint differs in {keys}")
                    else:
                        receipt["matching_starts"] += 1

                def branch_started():
                    receipt["branches_started"] += 1
                    save(directory / "result.json", receipt)

                try:
                    info = measure_arm(case, arm, cfg, cell, on_checkpoint=checkpoint,
                                       on_branch_start=branch_started)
                finally:
                    receipt["prefix_replays_started"] += int((cell / "prefix_started.json").is_file())
                row[arm] = info
                receipt["branches_completed"] += 1
                save(directory / "result.json", receipt)
            pairs.append(row)
            receipt["completed_pairs"] = len(pairs)
            save(directory / "pairs.json", pairs)
            save(directory / "result.json", receipt)
            print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=index, total=len(cases),
                                                            unit="pairs")), flush=True)
        if any(receipt[k] != 28 for k in ("prefix_replays_started", "branches_started", "branches_completed")) or receipt["matching_starts"] != 14:
            raise ValueError("Attempt counts differ from the fixed plan")
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt.update(status="HOLD", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    save(directory / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--sumocfg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.roster, args.sumocfg, args.output)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "COMPLETE" else 2)
