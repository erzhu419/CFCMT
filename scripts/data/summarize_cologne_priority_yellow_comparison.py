"""Summarize six fresh Cologne runs with priority-preserving yellow signals."""

import argparse
import json
from pathlib import Path

if __package__:
    from .summarize_cologne_repaired_controller_comparison import ARMS, SEEDS, SERVICE_KEYS, service_summary
else:
    from summarize_cologne_repaired_controller_comparison import ARMS, SEEDS, SERVICE_KEYS, service_summary

PROTOCOL = "tsc-v154i-cologne-priority-yellow-comparison-v1"
BASELINE_PROTOCOL = "tsc-v154f-cologne-repaired-controller-comparison-v1"
CHANGE_KEYS = ("mean_tripinfo_waiting_time", "system_vehicle_hours", "arrived", "collision_incidents")
METRIC_KEYS = (*SERVICE_KEYS, "collision_incidents", "collision_events", "starting_teleports", "ending_teleports")
EXPECTED = [(seed, arm) for seed in SEEDS for arm in ARMS]


def ordered_cells(cells):
    keys = [(row["seed"], row["arm"]) for row in cells]
    if len(keys) != 6 or set(keys) != set(EXPECTED):
        raise ValueError("Require all six frozen cells exactly once")
    by_key = dict(zip(keys, cells))
    return [by_key[key] for key in EXPECTED]


def aggregate(cells, baseline=None):
    rows = ordered_cells(cells)
    bindings = ("repaired_sumocfg", "source_bindings", "executor_source")
    if any(row["protocol"] != PROTOCOL or row.get("reused", False)
           or any(not row[key] or row[key] != rows[0][key] for key in bindings) for row in rows):
        raise ValueError("Require fresh runs with the same protocol, network, controller and executor source")
    complete = all(row["run_valid"] for row in rows)
    safety = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        valid = all(row["run_valid"] for row in selected)
        zero_count = sum(row["run_valid"] and row["metrics"]["collision_incidents"] == 0
                         and row["metrics"]["collision_events"] == 0 for row in selected)
        safety[arm] = {
            "all_seed_runs_valid": valid, "zero_collision_seed_count": zero_count,
            "zero_collision_status": "UNAVAILABLE" if not valid else "PASS" if zero_count == 3 else "FAIL",
            **{key + "_total": sum(row["metrics"][key] for row in selected) if valid else None
               for key in ("collision_incidents", "collision_events")},
        }
    result = {
        "protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
        "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
        **{key: rows[0][key] for key in bindings}, "safety_by_arm": safety,
        "all_seed_service": service_summary(rows, SEEDS) if complete else None,
        "additional_seed_service": service_summary(rows, (52282, 63792)) if complete else None,
        "cells": [{**{key: row[key] for key in ("seed", "arm", "run_valid", "status", "zero_incident_status")},
                   "metrics": {key: row["metrics"].get(key) for key in METRIC_KEYS},
                   "collision_lane_pairs": row["collision_inventory"]["incident_counts_by_first_participant_lane_pair"],
                   **{key: row.get(key) for key in ("source_result", "task_id", "elapsed_sec")}}
                  for row in rows],
        "new_simulation_count": 6, "reused_cell_count": 0,
        "new_simulation_elapsed_seconds_sum": sum(row["elapsed_sec"] for row in rows),
    }
    if baseline is not None:
        if baseline["protocol"] != BASELINE_PROTOCOL or baseline["repaired_sumocfg"] != rows[0]["repaired_sumocfg"]:
            raise ValueError("Baseline must be V154F on the same repaired network")
        old_rows = ordered_cells(baseline["cells"])
        result["changes_from_v154f"] = [
            {"seed": new["seed"], "arm": new["arm"], "both_runs_valid": new["run_valid"] and old["run_valid"],
             "metrics": {key: {"before": old["metrics"].get(key), "after": new["metrics"].get(key),
                               "after_minus_before": new["metrics"][key] - old["metrics"][key]
                               if new["run_valid"] and old["run_valid"] else None} for key in CHANGE_KEYS}}
            for new, old in zip(rows, old_rows)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-root", "launch-record", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    args = parser.parse_args()
    launch = json.loads(args.launch_record.read_text())
    tasks = {row["signature"]: row["id"] for row in json.loads(launch["scheduler_stdout"])["submitted"]}
    cells, retrieved_bytes = [], 0
    for seed, arm in EXPECTED:
        path = args.results_root / f"seed_{seed}" / arm / "result.json"
        row = json.loads(path.read_text())
        if (row["seed"], row["arm"]) != (seed, arm):
            raise ValueError(f"Cell does not match result path: {path}")
        row.update(source_result=str(path), task_id=tasks[f"CFCMT/v154i/priority-yellow-v1/cologne1/{seed}/{arm}"])
        cells.append(row)
        retrieved_bytes += path.stat().st_size
    baseline = json.loads(args.baseline_summary.read_text()) if args.baseline_summary else None
    result = aggregate(cells, baseline)
    result.update(retrieved_result_bytes=retrieved_bytes, launch_record=str(args.launch_record),
                  baseline_summary=str(args.baseline_summary) if args.baseline_summary else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("matrix_status", "safety_by_arm", "retrieved_result_bytes")}))


if __name__ == "__main__":
    main()
