"""Compare the frozen rigid threshold with ungated rigid and PhasePressure."""

import argparse
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize_cologne_local_fraction_refit import PROTOCOL as BASELINE_PROTOCOL, safety_summary
    from .summarize_cologne_priority_yellow_comparison import ordered_cells, METRIC_KEYS
    from .summarize_cologne_repaired_controller_comparison import SEEDS, SERVICE_KEYS
else:
    from summarize_cologne_local_fraction_refit import PROTOCOL as BASELINE_PROTOCOL, safety_summary
    from summarize_cologne_priority_yellow_comparison import ordered_cells, METRIC_KEYS
    from summarize_cologne_repaired_controller_comparison import SEEDS, SERVICE_KEYS

PROTOCOL = "tsc-v154m-cologne-threshold-closed-loop-v1"
THRESHOLD_PROTOCOL = "tsc-v154l-cologne-rigid-oof-threshold-v1"
THRESHOLD = 0.3580414874633295
GATED, UNGATED, PP = "rigid_thresholded", "rigid_target_only", "phase_pressure"
ARMS = (UNGATED, GATED, PP)


def service_summary(cells, seeds):
    by_key = {(row["seed"], row["arm"]): row for row in cells}
    means = {arm: {metric: mean(by_key[seed, arm]["metrics"][metric] for seed in seeds)
                   for metric in SERVICE_KEYS} for arm in ARMS}
    result = {"seeds": list(seeds), "weighting": "equal_seed", "means_by_arm": means}
    for name, arm in (("new_vs_ungated", UNGATED), ("new_vs_phase_pressure", PP)):
        result[name] = {
            metric: {"difference_mean": means[GATED][metric] - means[arm][metric],
                     "relative_difference_of_means_percent":
                         100 * (means[GATED][metric] / means[arm][metric] - 1)
                         if means[arm][metric] != 0 else None,
                     "paired_differences": [
                         {"seed": seed, "new_minus_reference": by_key[seed, GATED]["metrics"][metric]
                          - by_key[seed, arm]["metrics"][metric]} for seed in seeds]}
            for metric in SERVICE_KEYS}
    return result


def aggregate(cells, baseline):
    if len(cells) != 3 or {row["seed"] for row in cells} != set(SEEDS):
        raise ValueError("Require exactly the three frozen threshold evaluation seeds")
    if baseline["protocol"] != BASELINE_PROTOCOL:
        raise ValueError("Use the V154J local B100 comparison as baseline")
    old = ordered_cells(baseline["cells"])
    rows = sorted(cells, key=lambda row: SEEDS.index(row["seed"]))
    training, frozen = baseline["training"], rows[0]["frozen_threshold"]
    model_path, source_root = training["model_path"], baseline["source_root"]
    if (training["status"] != "PASS" or training["training_group_count"] != 100
            or training["frozen_candidate"] != "constant_alpha_0"
            or frozen["protocol"] != THRESHOLD_PROTOCOL or frozen["threshold"] != THRESHOLD
            or frozen["mode"] != "minimum_advantage"
            or frozen["comparison"] != "predicted_advantage > threshold + 1e-12"
            or frozen["model_path"] != model_path or frozen["source_root"] != source_root):
        raise ValueError("Require the unchanged B100 model and exact frozen V154L threshold")
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["arm"] != GATED or row["scenario"] != "cologne1"
                or row["repaired_sumocfg"] != baseline["repaired_sumocfg"]
                or row["source_root"] != source_root or row["model_path"] != model_path
                or row["occupancy_equation_protocol"] != training["occupancy_equation_protocol"]
                or row["frozen_threshold"] != frozen or row.get("reused", False)):
            raise ValueError("New runs must share the fixed network, model, core source and threshold")
    if any(row["source_root"] != source_root or row["model_path"] != model_path
           for row in old if row["arm"] == UNGATED):
        raise ValueError("Ungated comparison must use the same original B100 model and source")
    combined = [*rows, *old]
    complete = all(row["run_valid"] for row in combined)
    fields = ("seed", "arm", "run_valid", "status", "zero_incident_status", "source_result", "task_id",
              "elapsed_sec", "source_root", "model_path", "occupancy_equation_protocol", "tooling_root")
    compact = [{**{key: row[key] for key in fields if key in row}, "reused": False,
                "source_protocol": PROTOCOL,
                "metrics": {key: row["metrics"].get(key) for key in METRIC_KEYS},
                "collision_lane_pairs": row["collision_inventory"]["incident_counts_by_first_participant_lane_pair"]}
               for row in rows]
    compact.extend({**row, "reused": True} for row in old)
    by_key = {(row["seed"], row["arm"]): row for row in compact}
    return {
        "protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
        "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
        "repaired_sumocfg": baseline["repaired_sumocfg"], "source_root": source_root,
        "model_path": model_path, "training_group_count": training["training_group_count"],
        "frozen_threshold": frozen,
        "baseline_source_bindings": baseline["baseline_source_bindings"],
        "baseline_executor_source": baseline["baseline_executor_source"],
        "cells": [by_key[seed, arm] for seed in SEEDS for arm in ARMS],
        "safety_by_arm": {arm: safety_summary([row for row in combined if row["arm"] == arm]) for arm in ARMS},
        "all_seed_service": service_summary(combined, SEEDS) if complete else None,
        "additional_seed_service": service_summary(combined, (52282, 63792)) if complete else None,
        "new_task_ids": [row["task_id"] for row in rows],
        "reused_task_ids": {arm: [row["task_id"] for row in old if row["arm"] == arm] for arm in (UNGATED, PP)},
        "new_simulation_count": 3, "reused_cell_count": 6,
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
        path = args.results_root / "evaluation" / f"seed_{seed}" / GATED / "result.json"
        row = json.loads(path.read_text())
        if (row["seed"], row["arm"]) != (seed, GATED):
            raise ValueError(f"Evaluation result does not match its path: {path}")
        row.update(source_result=str(path), task_id=tasks[f"CFCMT/v154m/threshold-closed-loop-v1/evaluate/{seed}"])
        cells.append(row)
        retrieved_bytes += path.stat().st_size
    result = aggregate(cells, json.loads(args.baseline_summary.read_text()))
    result.update(retrieved_result_bytes=retrieved_bytes,
                  refs={"baseline_summary": str(args.baseline_summary), "launch_record": str(args.launch_record)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("matrix_status", "safety_by_arm", "retrieved_result_bytes")}))


if __name__ == "__main__":
    main()
