"""Compare the frozen raw-target controller with matched X and PP runs."""

import argparse
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize_cologne_spaced_no_stay import (
        PROTOCOL as BASELINE_PROTOCOL, NEW as NORMALIZED, SPACING, PP, GATED,
        PP_PROTOCOL, THRESHOLD_PROTOCOL, OCCUPANCY_PROTOCOL, COMPARISON,
        SEEDS, SERVICE_KEYS, METRIC_KEYS, safety_summary, execution_summary as spaced_execution_summary)
else:
    from summarize_cologne_spaced_no_stay import (
        PROTOCOL as BASELINE_PROTOCOL, NEW as NORMALIZED, SPACING, PP, GATED,
        PP_PROTOCOL, THRESHOLD_PROTOCOL, OCCUPANCY_PROTOCOL, COMPARISON,
        SEEDS, SERVICE_KEYS, METRIC_KEYS, safety_summary, execution_summary as spaced_execution_summary)

PROTOCOL = "tsc-v155b-cologne-raw-target-closed-loop-v1"
CALIBRATION_PROTOCOL = "tsc-v155a-cologne-raw-target-refit-v1"
NEW = "raw_target_spaced_no_stay_rigid"
ARMS = (NEW, NORMALIZED, PP)
LANES, SAMPLES = 8, 3541


def service_summary(cells, seeds):
    by_key = {(row["seed"], row["arm"]): row for row in cells}
    keys = (*SERVICE_KEYS, "mean_queue_per_lane")
    means = {arm: {metric: mean(by_key[seed, arm]["metrics"][metric] for seed in seeds)
                   for metric in keys} for arm in ARMS}
    result = {"seeds": list(seeds), "weighting": "equal_seed", "means_by_arm": means,
              "primary_metric": "mean_queue", "training_target_units_metric": "mean_queue_per_lane"}
    for name, reference in (("raw_target_vs_normalized_target", NORMALIZED), ("raw_target_vs_phase_pressure", PP)):
        result[name] = {
            metric: {"difference_mean": means[NEW][metric] - means[reference][metric],
                     "relative_difference_of_means_percent":
                         100 * (means[NEW][metric] / means[reference][metric] - 1)
                         if means[reference][metric] != 0 else None,
                     "paired_differences": [
                         {"seed": seed, "raw_target_minus_reference":
                          by_key[seed, NEW]["metrics"][metric] - by_key[seed, reference]["metrics"][metric]}
                         for seed in seeds]}
            for metric in keys}
    return result


def execution_summary(rows, frozen, baseline):
    prior = baseline["execution_diagnosis"]
    prior_view = {"per_seed": prior["per_seed"], "totals": prior["totals"],
                  "maximum_green_elapsed_sec_by_seed": [
                      {"seed": row["seed"], "by_tls": row["maximum_green_elapsed_sec_by_tls"]}
                      for row in prior["spacing_by_seed"]]}
    result = spaced_execution_summary(rows, frozen, {"execution_diagnosis": prior_view})
    result["normalized_target_baseline"] = result.pop("continuous_baseline")
    return result


def aggregate(protocol, cells, baseline):
    if (protocol["protocol"] != PROTOCOL or protocol["spacing_contract"] != SPACING or len(cells) != 3
            or {row["seed"] for row in cells} != set(SEEDS)):
        raise ValueError("Require the frozen B protocol and exactly the three original evaluation seeds")
    frozen, source, controls = protocol["frozen_threshold"], protocol["source_root"], protocol["controls"]
    if (frozen["protocol"] != THRESHOLD_PROTOCOL or frozen["calibration_protocol"] != CALIBRATION_PROTOCOL
            or frozen["score_units"] != "raw_native_halted_per_lane_cost_difference"
            or frozen["threshold"] != 0 or frozen["mode"] != "minimum_advantage"
            or frozen["comparison"] != COMPARISON or frozen["source_root"] != source
            or frozen["ablation_originator_path"] != protocol["ablation_originator_path"]):
        raise ValueError("Require the frozen A raw-target model/threshold and unchanged S veto")
    if (baseline["protocol"] != BASELINE_PROTOCOL or baseline["seeds"] != list(SEEDS)
            or baseline["source_root"] != source or baseline["repaired_sumocfg"] != protocol["sumocfg"]
            or baseline["model_path"] != controls["old_model_path"]
            or baseline["frozen_threshold"] != controls["old_frozen_threshold"]
            or baseline["spacing_contract"] != SPACING
            or any(baseline[key] != protocol[key] for key in ("threshold_originator_path", "ablation_originator_path"))
            or baseline["new_task_ids"] != controls["spaced_model_tasks"]
            or baseline["reused_task_ids"][PP] != controls["phase_pressure_tasks"]):
        raise ValueError("Reuse exactly the fixed X model and same-network PP comparison")
    reused = [row for row in baseline["cells"] if row["arm"] in (NORMALIZED, PP)]
    if (len(reused) != 6 or {(row["seed"], row["arm"]) for row in reused}
            != {(seed, arm) for seed in SEEDS for arm in (NORMALIZED, PP)}):
        raise ValueError("Require exactly six X and PP baseline cells")
    for row in reused:
        is_model = row["arm"] == NORMALIZED
        tasks = controls["spaced_model_tasks" if is_model else "phase_pressure_tasks"]
        if (row["source_protocol"] != (BASELINE_PROTOCOL if is_model else PP_PROTOCOL)
                or row["task_id"] != tasks[SEEDS.index(row["seed"])]) or (
                is_model and (row["source_root"] != source or row["model_path"] != controls["old_model_path"]
                              or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL)):
            raise ValueError("A baseline cell changed its source, model or task binding")
    rows = sorted(cells, key=lambda row: SEEDS.index(row["seed"]))
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["arm"] != GATED or row["scenario"] != "cologne1"
                or row["source_root"] != source or row["repaired_sumocfg"] != protocol["sumocfg"]
                or row["model_path"] != frozen["model_path"] or row["frozen_threshold"] != frozen
                or any(row[key] != protocol[key] for key in ("threshold_originator_path", "ablation_originator_path"))
                or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL or row.get("reused", False)):
            raise ValueError("New runs must use the frozen raw-target model, source, network and execution rule")
        metrics = row["metrics"]
        if row["run_valid"] and (metrics["controlled_lane_count"] != LANES
                or metrics["fixed_horizon_sample_seconds"] != SAMPLES
                or abs(metrics["mean_queue"] - metrics["halted_vehicle_seconds"] / SAMPLES) > 1e-12
                or abs(metrics["mean_queue_per_lane"] - metrics["mean_queue"] / LANES) > 1e-12):
            raise ValueError("Primary halted-queue cost must retain the fixed sample and lane denominators")
    fields = ("seed", "run_valid", "status", "zero_incident_status", "source_result", "task_id",
              "elapsed_sec", "source_root", "model_path", "occupancy_equation_protocol", "validity_checks")
    compact = [{**{key: row[key] for key in fields if key in row}, "arm": NEW, "source_arm": GATED,
                "source_protocol": PROTOCOL, "reused": False,
                "metrics": {key: row["metrics"].get(key) for key in (*METRIC_KEYS, "mean_queue_per_lane")},
                "collision_lane_pairs": row["collision_inventory"]["incident_counts_by_first_participant_lane_pair"]}
               for row in rows]
    compact.extend({**row, "reused": True, "metrics": {
        **row["metrics"], "mean_queue_per_lane": row["metrics"]["mean_queue"] / LANES}} for row in reused)
    by_key = {(row["seed"], row["arm"]): row for row in compact}
    compact = [by_key[seed, arm] for seed in SEEDS for arm in ARMS]
    complete = all(row["run_valid"] for row in compact)
    return {"protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
            "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
            "repaired_sumocfg": protocol["sumocfg"], "source_root": source,
            "model_path": frozen["model_path"], "frozen_threshold": frozen,
            "old_model_path": controls["old_model_path"], "old_frozen_threshold": controls["old_frozen_threshold"],
            "spacing_contract": SPACING,
            "primary_efficiency": {"metric": "mean_queue", "definition": "mean SUMO halted vehicles across eight controlled lanes",
                "samples_per_run": SAMPLES, "controlled_lane_count": LANES,
                "training_target_equivalent": "mean_queue_per_lane = mean_queue / 8",
                "other_service_metrics": "Descriptive outcomes; no additional optimization objective"},
            **{key: baseline[key] for key in ("baseline_source_bindings", "baseline_executor_source")},
            **{key: protocol[key] for key in ("threshold_originator_path", "ablation_originator_path")},
            "cells": compact, "safety_by_arm": {arm: safety_summary([row for row in compact if row["arm"] == arm])
                                                 for arm in ARMS},
            "all_seed_service": service_summary(compact, SEEDS) if complete else None,
            "additional_seed_service": service_summary(compact, (52282, 63792)) if complete else None,
            "execution_diagnosis": execution_summary(rows, frozen, baseline),
            "new_task_ids": [row["task_id"] for row in rows],
            "reused_task_ids": {arm: [by_key[seed, arm]["task_id"] for seed in SEEDS] for arm in (NORMALIZED, PP)},
            "new_simulation_count": 3, "reused_cell_count": 6, "new_model_fit_count": 0, "threshold_retuned": False,
            "new_simulation_elapsed_seconds_sum": sum(row["elapsed_sec"] for row in rows),
            "interpretation": "The A raw-target model and its frozen zero threshold replace the P/T model under the same S veto and450-second spacing. The primary measure is halted queue. Complete colliding runs contribute to all efficiency means; invalid runs disable means. These previously inspected seeds provide development evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "results-root", "launch-record", "baseline-summary", "out"):
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
        row.update(source_result=str(path), task_id=tasks[f"CFCMT/v155b/raw-target-spaced-v1/evaluate/{seed}"])
        cells.append(row)
        retrieved_bytes += path.stat().st_size
    result = aggregate(json.loads(args.protocol.read_text()), cells, json.loads(args.baseline_summary.read_text()))
    result.update(retrieved_result_bytes=retrieved_bytes,
                  refs={"baseline_summary": str(args.baseline_summary), "launch_record": str(args.launch_record),
                        "protocol": str(args.protocol)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("matrix_status", "primary_efficiency", "retrieved_result_bytes")}))


if __name__ == "__main__":
    main()
