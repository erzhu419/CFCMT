"""Compare the Cologne1 B100 fraction refit with the frozen V154I controls."""

import argparse
import json
from pathlib import Path

if __package__:
    from .summarize_cologne_priority_yellow_comparison import (
        PROTOCOL as BASELINE_PROTOCOL, ordered_cells, CHANGE_KEYS, METRIC_KEYS,
    )
    from .summarize_cologne_repaired_controller_comparison import ARMS, SEEDS, service_summary
else:
    from summarize_cologne_priority_yellow_comparison import (
        PROTOCOL as BASELINE_PROTOCOL, ordered_cells, CHANGE_KEYS, METRIC_KEYS,
    )
    from summarize_cologne_repaired_controller_comparison import ARMS, SEEDS, service_summary

PROTOCOL = "tsc-v154j-cologne-local-fraction-refit-v2"


def safety_summary(rows):
    valid = all(row["run_valid"] for row in rows)
    keys = ("collision_incidents", "collision_events", "starting_teleports", "ending_teleports")
    zero = sum(row["run_valid"] and all(row["metrics"][key] == 0 for key in keys) for row in rows)
    return {"all_seed_runs_valid": valid, "zero_incident_seed_count": zero,
            "zero_incident_status": "UNAVAILABLE" if not valid else "PASS" if zero == 3 else "FAIL",
            **{key + "_total": sum(row["metrics"][key] for row in rows) if valid else None for key in keys}}


def compact_cell(row, reused):
    keys = ("seed", "arm", "run_valid", "status", "zero_incident_status", "source_result",
            "task_id", "elapsed_sec", "source_root", "model_path", "occupancy_equation_protocol")
    return {**{key: row[key] for key in keys if key in row}, "reused": reused,
            "source_protocol": BASELINE_PROTOCOL if reused else PROTOCOL,
            "metrics": {key: row["metrics"].get(key) for key in METRIC_KEYS},
            "collision_lane_pairs": row.get("collision_inventory", {}).get(
                "incident_counts_by_first_participant_lane_pair", row.get("collision_lane_pairs", []))}


def aggregate(cells, fit, baseline):
    if len(cells) != 3 or {row["seed"] for row in cells} != set(SEEDS):
        raise ValueError("Require exactly the three new rigid evaluation seeds")
    rows = sorted(cells, key=lambda row: SEEDS.index(row["seed"]))
    if baseline["protocol"] != BASELINE_PROTOCOL or fit["protocol"] != PROTOCOL:
        raise ValueError("Require the local v2 fit and V154I baseline")
    old = ordered_cells(baseline["cells"])
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["arm"] != ARMS[0] or row["scenario"] != "cologne1"
                or row["repaired_sumocfg"] != baseline["repaired_sumocfg"]
                or not row["source_root"] or row["source_root"] != rows[0]["source_root"]
                or row["model_path"] != fit.get("model_path")
                or row["occupancy_equation_protocol"] != fit.get("occupancy_equation_protocol")
                or row.get("reused", False)):
            raise ValueError("New rigid cells must share the fitted model, source and fixed Cologne1 network")
    pp = [row for row in old if row["arm"] == ARMS[1]]
    old_rigid = [row for row in old if row["arm"] == ARMS[0]]
    fit_valid = (fit["status"] == "PASS" and fit.get("training_group_count") == 100
                 and len(set(fit.get("selected_groups", ()))) == 100
                 and fit.get("frozen_candidate") == "constant_alpha_0")
    complete = fit_valid and all(row["run_valid"] for row in [*rows, *old])
    training = {key: value for key, value in fit.items()
                if key not in ("selected_groups", "selection", "cells", "anchor_feature_names")}
    training["selection"] = {
        seed: {"pool_groups": item["pool_groups"], "quota": item["quota"],
               "selected_group_count": len(item["selected_groups"])}
        for seed, item in fit.get("selection", {}).items()}
    training["collection_cells"] = [
        {**{key: value for key, value in cell.items()
             if key not in ("retained_groups", "censored_groups", "unavailable_groups")},
         **{key + "_count": len(cell[key]) for key in
            ("retained_groups", "censored_groups", "unavailable_groups") if key in cell}}
        for cell in fit.get("cells", [])]
    return {
        "protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
        "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
        "repaired_sumocfg": baseline["repaired_sumocfg"], "source_root": rows[0]["source_root"],
        "baseline_source_bindings": baseline["source_bindings"],
        "baseline_executor_source": baseline["executor_source"], "training": training,
        "safety_by_arm": {ARMS[0]: safety_summary(rows), ARMS[1]: safety_summary(pp),
                          "old_rigid_target_only": safety_summary(old_rigid)},
        "all_seed_service": service_summary([*rows, *pp], SEEDS) if complete else None,
        "additional_seed_service": service_summary([*rows, *pp], (52282, 63792)) if complete else None,
        "changes_from_v154i_rigid": [
            {"seed": new["seed"], "before_result": before.get("source_result"),
             "before_task_id": before.get("task_id"), "both_runs_valid": new["run_valid"] and before["run_valid"],
             "metrics": {key: {"before": before["metrics"].get(key), "after": new["metrics"].get(key),
                               "after_minus_before": new["metrics"][key] - before["metrics"][key]
                               if new["run_valid"] and before["run_valid"] else None} for key in CHANGE_KEYS}}
            for new, before in zip(rows, old_rigid)],
        "cells": [compact_cell(row, False) for row in rows] + [compact_cell(row, True) for row in pp],
        "reused_phase_pressure_task_ids": [row["task_id"] for row in pp],
        "new_simulation_count": 3, "reused_cell_count": 3,
        "new_simulation_elapsed_seconds_sum": sum(row["elapsed_sec"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-root", "launch-record", "baseline-summary", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads(args.launch_record.read_text())
    tasks = {row["signature"]: row["id"] for row in json.loads(launch["scheduler_stdout"])["submitted"]}
    cells, retrieved_bytes = [], 0
    for seed in SEEDS:
        path = args.results_root / "evaluation" / f"seed_{seed}" / ARMS[0] / "result.json"
        row = json.loads(path.read_text())
        if (row["seed"], row["arm"]) != (seed, ARMS[0]):
            raise ValueError(f"Evaluation result does not match its path: {path}")
        row.update(source_result=str(path), task_id=tasks[f"CFCMT/v154j/local-fraction-v2/evaluate/{seed}"])
        cells.append(row)
        retrieved_bytes += path.stat().st_size
    fit_path = args.results_root / "fit_v2" / "result.json"
    result = aggregate(cells, json.loads(fit_path.read_text()), json.loads(args.baseline_summary.read_text()))
    result.update(retrieved_result_bytes=retrieved_bytes + fit_path.stat().st_size,
                  refs={"fit_result": str(fit_path), "baseline_summary": str(args.baseline_summary),
                        "launch_record": str(args.launch_record)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("matrix_status", "safety_by_arm", "retrieved_result_bytes")}))


if __name__ == "__main__":
    main()
