"""Execute the 28 fresh Jinan pairs in roster order with independent PP prefix replays."""
import argparse
import json
from pathlib import Path
import traceback

from measure_target import measure_arm


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def frozen_cases(roster, plan):
    seeds = [33061101, 33061102, 33061103, 33061104]
    points = [tuple(p["point"]) for p in plan["source_models"]["points"]]
    cases = roster["cases"]
    expected = {(s, *p) for s in seeds for p in points}
    actual = {(c["seed"], c["right_near"], c["straight_near"]) for c in cases}
    if (roster["status"] != "PASS_FRESH_TARGET_OPPORTUNITY"
            or roster["seeds"] != seeds or plan["seeds"] != seeds
            or set(map(tuple, roster["count_pairs"])) != set(points)
            or len(expected) != 28 or len(cases) != 28 or actual != expected
            or len({(c["seed"], c["time_sec"]) for c in cases}) != 28):
        raise ValueError("Expected all seven frozen points at distinct checkpoints in each of four seeds")
    return cases


def run(roster_path, plan_path, cfg, registry_path, directory):
    cases = frozen_cases(json.loads(roster_path.read_text()), json.loads(plan_path.read_text()))
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
                save(directory / "result.json", receipt)

                def checkpoint(start):
                    nonlocal reference
                    if arm == "high":
                        reference = start
                    elif start != reference:
                        mismatches = sorted(k for k in start.keys() | reference.keys()
                                            if start.get(k) != reference.get(k))
                        raise ValueError(f"Paired checkpoint differs in {mismatches}")
                    else:
                        receipt["matching_starts"] += 1

                def branch_started():
                    receipt["branches_started"] += 1
                    save(directory / "result.json", receipt)

                try:
                    row[arm] = measure_arm(case, arm, cfg, cell, checkpoint, branch_started,
                                           registry_path=registry_path)
                finally:
                    receipt["prefix_replays_started"] += int((cell / "prefix_started.json").is_file())
                receipt["branches_completed"] += 1
                save(directory / "result.json", receipt)
            pairs.append(row)
            receipt["completed_pairs"] = len(pairs)
            save(directory / "pairs.json", pairs)
            save(directory / "result.json", receipt)
            print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=index, total=len(cases),
                                                            unit="pairs")), flush=True)
        if (any(receipt[k] != 56 for k in ("prefix_replays_started", "branches_started", "branches_completed"))
                or receipt["matching_starts"] != 28):
            raise ValueError("Attempt counts differ from the frozen 28-pair design")
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt.update(status="HOLD", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    save(directory / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("roster", "plan", "sumocfg", "registry", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.roster, args.plan, args.sumocfg, args.registry, args.output)
    print(json.dumps(receipt), flush=True)
    raise SystemExit(0 if receipt["status"] == "COMPLETE" else 2)
