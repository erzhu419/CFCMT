"""Compare the fixed native-label B100 model with original thresholded rigid and PP."""

import argparse
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize_cologne_local_fraction_refit import safety_summary
    from .summarize_cologne_priority_yellow_comparison import METRIC_KEYS
    from .summarize_cologne_repaired_controller_comparison import SEEDS, SERVICE_KEYS
else:
    from summarize_cologne_local_fraction_refit import safety_summary
    from summarize_cologne_priority_yellow_comparison import METRIC_KEYS
    from summarize_cologne_repaired_controller_comparison import SEEDS, SERVICE_KEYS

PROTOCOL = "tsc-v154q-cologne-native-label-closed-loop-v1"
BASELINE_PROTOCOL = "tsc-v154m-cologne-threshold-closed-loop-v1"
PP_PROTOCOL = "tsc-v154i-cologne-priority-yellow-comparison-v1"
CALIBRATION_PROTOCOL = "tsc-v154p-cologne-native-b100-v1"
THRESHOLD_PROTOCOL = "tsc-v154l-cologne-rigid-oof-threshold-v1"
OCCUPANCY_PROTOCOL = "sumo-lane-occupancy-fraction-equations-v1"
COMPARISON = "predicted_advantage > threshold + 1e-12"
GATED = "rigid_thresholded"
NEW, OLD, PP = "new_native_rigid", "original_thresholded_rigid", "phase_pressure"
ARMS = (NEW, OLD, PP)


def service_summary(cells, seeds):
    by_key = {(row["seed"], row["arm"]): row for row in cells}
    means = {arm: {metric: mean(by_key[seed, arm]["metrics"][metric] for seed in seeds)
                   for metric in SERVICE_KEYS} for arm in ARMS}
    result = {"seeds": list(seeds), "weighting": "equal_seed", "means_by_arm": means}
    for name, arm in (("new_vs_original_thresholded", OLD), ("new_vs_phase_pressure", PP)):
        result[name] = {
            metric: {"difference_mean": means[NEW][metric] - means[arm][metric],
                     "relative_difference_of_means_percent":
                         100 * (means[NEW][metric] / means[arm][metric] - 1)
                         if means[arm][metric] != 0 else None,
                     "paired_differences": [
                         {"seed": seed, "new_minus_reference": by_key[seed, NEW]["metrics"][metric]
                          - by_key[seed, arm]["metrics"][metric]} for seed in seeds]}
            for metric in SERVICE_KEYS}
    return result


def threshold_application(rows, frozen):
    per_seed = []
    for row in rows:
        diag = row["originator_diagnostics"]
        if (diag["threshold"] != frozen["threshold"]
                or diag["threshold_mode"] != frozen["mode"]
                or diag["threshold_comparison"] != COMPARISON
                or diag["threshold_model_path"] != frozen["model_path"]):
            raise ValueError("Observed threshold application must match the frozen native model")
        item = {"seed": row["seed"], "decisions": diag["decision_count"],
                "ungated_rigid_proposals_on_gated_trajectory": diag["ungated_rigid_proposal_count"],
                "accepted_overrides": diag["phase_pressure_override_count"],
                "threshold_rejected_proposals": diag["threshold_rejected_proposal_count"]}
        if (item["accepted_overrides"] + item["threshold_rejected_proposals"]
                != item["ungated_rigid_proposals_on_gated_trajectory"]):
            raise ValueError("Threshold proposal accounting is incomplete")
        per_seed.append(item)
    totals = {key: sum(row[key] for row in per_seed) for key in per_seed[0] if key != "seed"}
    return {"per_seed": per_seed, "totals": totals,
            "threshold": frozen["threshold"], "comparison": COMPARISON,
            "accepted_fraction_of_decisions": totals["accepted_overrides"] / totals["decisions"]
            if totals["decisions"] else None,
            "description": "Counts refer to the new native-label controller trajectories; accepted model proposals are distinct from executed phase switches."}


def aggregate(protocol, cells, baseline):
    if (protocol["protocol"] != PROTOCOL or len(cells) != 3
            or {row["seed"] for row in cells} != set(SEEDS)):
        raise ValueError("Require the frozen V154Q protocol and exactly three original evaluation seeds")
    frozen = protocol["frozen_threshold"]
    source = protocol["source_root"]
    if (frozen["protocol"] != THRESHOLD_PROTOCOL
            or frozen.get("calibration_protocol") != CALIBRATION_PROTOCOL
            or frozen["threshold"] != 0.0 or frozen["mode"] != "minimum_advantage"
            or frozen["comparison"] != COMPARISON or frozen["source_root"] != source):
        raise ValueError("Require the unchanged native B100 model and P-calibrated threshold zero")
    if (baseline["protocol"] != BASELINE_PROTOCOL or baseline["seeds"] != list(SEEDS)
            or baseline["source_root"] != source
            or baseline["repaired_sumocfg"] != protocol["sumocfg"]
            or baseline["training_group_count"] != 100
            or baseline["new_task_ids"] != protocol["controls"]["original_thresholded_tasks"]
            or baseline["reused_task_ids"][PP] != protocol["controls"]["phase_pressure_tasks"]):
        raise ValueError("Reuse the original fixed-network V154M comparison")
    old_frozen = baseline["frozen_threshold"]
    if (old_frozen["protocol"] != THRESHOLD_PROTOCOL or old_frozen["threshold"] != 0.3580414874633295
            or old_frozen["mode"] != "minimum_advantage" or old_frozen["comparison"] != COMPARISON
            or old_frozen["model_path"] != baseline["model_path"] or old_frozen["source_root"] != source
            or old_frozen["model_path"] == frozen["model_path"]):
        raise ValueError("The original model and threshold must retain their original binding")
    rows = sorted(cells, key=lambda row: SEEDS.index(row["seed"]))
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["arm"] != GATED or row["scenario"] != "cologne1"
                or row["source_root"] != source or row["repaired_sumocfg"] != protocol["sumocfg"]
                or row["model_path"] != frozen["model_path"] or row["frozen_threshold"] != frozen
                or row["threshold_originator_path"] != protocol["threshold_originator_path"]
                or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL or row.get("reused", False)):
            raise ValueError("New runs must share the frozen native model, threshold, source, originator and network")
    old = [row for row in baseline["cells"] if row["arm"] in (GATED, PP)]
    if (len(old) != 6 or {(row["seed"], row["arm"]) for row in old}
            != {(seed, arm) for seed in SEEDS for arm in (GATED, PP)}):
        raise ValueError("Require exactly three original thresholded and three original PP cells")
    for row in old:
        if row["arm"] == GATED:
            if (row["source_protocol"] != BASELINE_PROTOCOL or row["model_path"] != baseline["model_path"]
                    or row["source_root"] != source or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL
                    or row["task_id"] != baseline["new_task_ids"][SEEDS.index(row["seed"])]):
                raise ValueError("Original thresholded control provenance changed")
        elif (row["source_protocol"] != PP_PROTOCOL
              or row["task_id"] != baseline["reused_task_ids"][PP][SEEDS.index(row["seed"])]):
            raise ValueError("Reuse the existing V154I PhasePressure trajectories")
    fields = ("seed", "run_valid", "status", "zero_incident_status", "source_result", "task_id",
              "elapsed_sec", "source_root", "model_path", "occupancy_equation_protocol", "validity_checks")
    compact = [{**{key: row[key] for key in fields if key in row}, "arm": NEW, "source_arm": GATED,
                "source_protocol": PROTOCOL, "reused": False,
                "metrics": {key: row["metrics"].get(key) for key in METRIC_KEYS},
                "collision_lane_pairs": row["collision_inventory"]["incident_counts_by_first_participant_lane_pair"]}
               for row in rows]
    compact.extend({**row, "source_arm": row["arm"], "arm": OLD if row["arm"] == GATED else PP,
                    "reused": True} for row in old)
    by_key = {(row["seed"], row["arm"]): row for row in compact}
    compact = [by_key[seed, arm] for seed in SEEDS for arm in ARMS]
    complete = all(row["run_valid"] for row in compact)
    return {
        "protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
        "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
        "repaired_sumocfg": protocol["sumocfg"], "source_root": source,
        "model_path": frozen["model_path"], "frozen_threshold": frozen,
        "original_model_path": baseline["model_path"], "original_frozen_threshold": old_frozen,
        "baseline_source_bindings": baseline["baseline_source_bindings"],
        "baseline_executor_source": baseline["baseline_executor_source"],
        "threshold_originator_path": protocol["threshold_originator_path"],
        "cells": compact, "safety_by_arm": {arm: safety_summary([row for row in compact if row["arm"] == arm])
                                             for arm in ARMS},
        "all_seed_service": service_summary(compact, SEEDS) if complete else None,
        "additional_seed_service": service_summary(compact, (52282, 63792)) if complete else None,
        "threshold_application": threshold_application(rows, frozen),
        "new_task_ids": [row["task_id"] for row in rows],
        "reused_task_ids": {arm: [by_key[seed, arm]["task_id"] for seed in SEEDS] for arm in (OLD, PP)},
        "new_simulation_count": 3, "reused_cell_count": 6, "new_model_fit_count": 0,
        "threshold_retuned": False,
        "new_simulation_elapsed_seconds_sum": sum(row["elapsed_sec"] for row in rows),
        "interpretation": "Complete colliding runs remain in service means. Three original seeds include geometry-development seed41242; the additional-two-seed summary is prespecified. This is a comparison of the complete recalibrated native-label controller with the previous controller, not an isolated effect of labels with threshold held constant.",
    }


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
        row.update(source_result=str(path),
                   task_id=tasks[f"CFCMT/v154q/native-label-closed-loop-v1/evaluate/{seed}"])
        cells.append(row)
        retrieved_bytes += path.stat().st_size
    result = aggregate(json.loads(args.protocol.read_text()), cells, json.loads(args.baseline_summary.read_text()))
    result.update(retrieved_result_bytes=retrieved_bytes,
                  refs={"baseline_summary": str(args.baseline_summary), "launch_record": str(args.launch_record),
                        "protocol": str(args.protocol)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("matrix_status", "safety_by_arm", "retrieved_result_bytes")}))


if __name__ == "__main__":
    main()
