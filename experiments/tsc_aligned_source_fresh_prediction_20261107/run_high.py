"""Replay each frozen Jinan checkpoint once and measure the native high action."""
import argparse
import json
from pathlib import Path
import traceback

from measure_target import measure_arm
from readout import arm_data
from score_fresh import score


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def frozen_cases(roster, plan):
    cases = roster["cases"]
    points = sorted(tuple(row[:2]) for row in plan["development_counts"]["rows"])
    expected = [(seed, point) for seed in plan["target_seeds"] for point in points]
    measured = [(c["seed"], (c["right_near"], c["straight_near"])) for c in cases]
    if (roster["status"] != "PASS_BALANCED_COVERAGE" or measured != expected
            or len(set(expected)) != 24 or len({(c["seed"], c["time_sec"]) for c in cases}) != 24
            or any(tuple(c["point"]) != point for c, (_, point) in zip(cases, expected))):
        raise ValueError("Measurement requires the frozen 24 unique cases from a full census")
    return cases


def run(roster_path, plan_path, cfg, registry, output):
    from cf_h2o.sumo_runtime import load_libsumo

    plan = json.loads(plan_path.read_text())
    cases = frozen_cases(json.loads(roster_path.read_text()), plan)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    receipt = dict(status="RUNNING", planned_cases=len(cases), completed_cases=0, matching_starts=0,
                   prefix_replays_started=0, branches_started=0, branches_completed=0,
                   fits=0, closed_loop_policy_evaluations=0)
    save(output / "result.json", receipt)
    save(output / "cases.json", rows)
    try:
        for index, case in enumerate(cases, 1):
            arm_dir = output / f"case_{index:02d}"
            arm_dir.mkdir()
            lane, targets = case["straight_lane"], case["straight_target_ids"]
            reference = json.loads((roster_path.parent / case["start_file"]).read_text())
            length = None
            receipt["last_attempt"] = dict(index=index, seed=case["seed"], time_sec=case["time_sec"])
            save(output / "result.json", receipt)

            def checkpoint(start):
                nonlocal length
                if start != reference:
                    keys = [k for k in start.keys() | reference.keys() if start.get(k) != reference.get(k)]
                    raise ValueError(f"Census and replay starts differ in {sorted(keys)}")
                api = load_libsumo()
                length = api.lane.getLength(lane)
                near_limit = 10 * api.lane.getMaxSpeed(lane)
                vehicles = start["vehicles"]
                order = sorted((length - v["lane_position"], vid) for vid, v in vehicles.items()
                               if v["lane_id"] == lane and length - v["lane_position"] <= near_limit)
                if (len(order) != case["straight_near"] or [vid for _, vid in order[:2]] != targets
                        or any(vehicles[vid]["speed"] != 0 or api.vehicle.getAccel(vid) != 2.0 for vid in targets)):
                    raise ValueError("Census rank, rest condition or native accel differs at replay")
                receipt["matching_starts"] += 1

            def branch_started():
                receipt["branches_started"] += 1
                save(output / "result.json", receipt)

            try:
                arm = measure_arm(case, "high", cfg, arm_dir, checkpoint, branch_started, registry_path=registry)
            finally:
                receipt["prefix_replays_started"] += int((arm_dir / "prefix_started.json").is_file())
            receipt["branches_completed"] += 1
            if (arm["right_nominal"] != case["right_sequence"]
                    or arm["straight_nominal"] != case["high_straight_sequence"]):
                raise ValueError("The nominal high action no longer matches this frozen case")
            metrics, _ = arm_data(arm_dir, case, targets, lane, length)
            crossed = metrics["second"]["entry_s"] is not None
            row = {key: case[key] for key in ("seed", "time_sec", "tls_id", "point", "straight_target_ids")}
            row.update(crossed=crossed, high_candidate=case["high_candidate"], **metrics)
            rows.append(row)
            receipt["completed_cases"] = len(rows)
            save(output / "cases.json", rows)
            save(output / "result.json", receipt)
            print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=index, total=len(cases), unit="cases")),
                  flush=True)
        if any(receipt[k] != 24 for k in ("matching_starts", "prefix_replays_started",
                                          "branches_started", "branches_completed", "completed_cases")):
            raise ValueError("Replay or measurement counts differ from the frozen 24-case design")
        scores = score(plan, rows)
        save(output / "scores.json", scores)
        receipt.update(status="COMPLETE", prediction_decision=scores["status"])
    except Exception as exc:
        receipt.update(status="HOLD_EXECUTION", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    save(output / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("roster", "plan", "sumocfg", "registry", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.roster, args.plan, args.sumocfg, args.registry, args.output)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "COMPLETE" else 2)
