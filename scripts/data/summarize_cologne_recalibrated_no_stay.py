"""Compare the frozen T-calibrated no-stay controller with S and PhasePressure."""

import argparse
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize_cologne_no_stay_closed_loop import (
        PROTOCOL as BASELINE_PROTOCOL, ORIGINATOR_PROTOCOL, PP_PROTOCOL,
        THRESHOLD_PROTOCOL, CALIBRATION_PROTOCOL as NATIVE_CALIBRATION_PROTOCOL,
        OCCUPANCY_PROTOCOL, COMPARISON, GATED, SEEDS, SERVICE_KEYS, METRIC_KEYS,
        safety_summary)
else:
    from summarize_cologne_no_stay_closed_loop import (
        PROTOCOL as BASELINE_PROTOCOL, ORIGINATOR_PROTOCOL, PP_PROTOCOL,
        THRESHOLD_PROTOCOL, CALIBRATION_PROTOCOL as NATIVE_CALIBRATION_PROTOCOL,
        OCCUPANCY_PROTOCOL, COMPARISON, GATED, SEEDS, SERVICE_KEYS, METRIC_KEYS,
        safety_summary)

PROTOCOL = "tsc-v154u-cologne-recalibrated-no-stay-closed-loop-v1"
CALIBRATION_PROTOCOL = "tsc-v154t-cologne-no-stay-oof-threshold-v1"
THRESHOLD = 0.20790450432178334
NEW, ZERO, PP = "recalibrated_no_stay_rigid", "no_stay_native_rigid", "phase_pressure"
ARMS = (NEW, ZERO, PP)


def service_summary(cells, seeds):
    by_key = {(row["seed"], row["arm"]): row for row in cells}
    means = {arm: {metric: mean(by_key[seed, arm]["metrics"][metric] for seed in seeds)
                   for metric in SERVICE_KEYS} for arm in ARMS}
    result = {"seeds": list(seeds), "weighting": "equal_seed", "means_by_arm": means}
    for name, reference in (("recalibrated_vs_threshold_zero", ZERO),
                            ("recalibrated_vs_phase_pressure", PP)):
        result[name] = {
            metric: {"difference_mean": means[NEW][metric] - means[reference][metric],
                     "relative_difference_of_means_percent":
                         100 * (means[NEW][metric] / means[reference][metric] - 1)
                         if means[reference][metric] != 0 else None,
                     "paired_differences": [
                         {"seed": seed, "recalibrated_minus_reference":
                          by_key[seed, NEW]["metrics"][metric] - by_key[seed, reference]["metrics"][metric]}
                         for seed in seeds]}
            for metric in SERVICE_KEYS}
    return result


def execution_summary(rows, frozen, baseline):
    per_seed = []
    for row in rows:
        diag = row["originator_diagnostics"]
        if (diag["protocol"] != ORIGINATOR_PROTOCOL or diag["threshold"] != frozen["threshold"]
                or diag["threshold_mode"] != frozen["mode"]
                or diag["threshold_comparison"] != COMPARISON
                or diag["threshold_model_path"] != frozen["model_path"]):
            raise ValueError("Observed originator must use the frozen T threshold and P model")
        item = {"seed": row["seed"], "decisions": diag["decision_count"],
                "ungated_rigid_proposals_on_no_stay_trajectory": diag["ungated_rigid_proposal_count"],
                "accepted_overrides": diag["phase_pressure_override_count"],
                "threshold_rejected_proposals": diag["threshold_rejected_proposal_count"],
                "stay_veto_count": diag["model_stay_veto_count"],
                "executed_stay_overrides": row["metrics"]["guard_audit"]["executed_stay_overrides"],
                "phase_switches": row["metrics"]["phase_execution_audit"]["switches"]}
        if (item["accepted_overrides"] + item["threshold_rejected_proposals"] + item["stay_veto_count"]
                != item["ungated_rigid_proposals_on_no_stay_trajectory"]):
            raise ValueError("Uncapped proposals must equal accepted, threshold-rejected and stay-vetoed counts")
        if item["executed_stay_overrides"] != 0:
            raise ValueError("The no-stay controller executed a model stay override")
        per_seed.append(item)
    totals = {key: sum(row[key] for row in per_seed) for key in per_seed[0] if key != "seed"}
    prior = baseline["execution_diagnosis"]
    return {"per_seed": per_seed, "totals": totals, "threshold": frozen["threshold"],
            "comparison": COMPARISON,
            "maximum_green_elapsed_sec_by_seed": [
                {"seed": row["seed"], "by_tls": row["maximum_green_elapsed_sec_by_tls"]} for row in rows],
            "accepted_fraction_of_decisions": totals["accepted_overrides"] / totals["decisions"]
            if totals["decisions"] else None,
            "threshold_zero_baseline": {key: prior[key] for key in
                                        ("per_seed", "totals", "maximum_green_elapsed_sec_by_seed")},
            "description": "Uncapped proposal counts follow threshold rejection, then stay veto, then acceptance on each controller's own trajectory. Accepted overrides and phase switches are distinct counts."}


def aggregate(protocol, cells, baseline):
    if (protocol["protocol"] != PROTOCOL or len(cells) != 3
            or {row["seed"] for row in cells} != set(SEEDS)):
        raise ValueError("Require the frozen V154U protocol and exactly three original evaluation seeds")
    frozen, source = protocol["frozen_threshold"], protocol["source_root"]
    if (frozen["protocol"] != THRESHOLD_PROTOCOL
            or frozen.get("calibration_protocol") != CALIBRATION_PROTOCOL
            or frozen["threshold"] != THRESHOLD or frozen["mode"] != "minimum_advantage"
            or frozen["comparison"] != COMPARISON or frozen["source_root"] != source
            or frozen["ablation_originator_path"] != protocol["ablation_originator_path"]):
        raise ValueError("Require the frozen T-calibrated threshold and unchanged S no-stay rule")
    controls, old = protocol["controls"], baseline["frozen_threshold"]
    if (baseline["protocol"] != BASELINE_PROTOCOL or baseline["seeds"] != list(SEEDS)
            or baseline["source_root"] != source or baseline["repaired_sumocfg"] != protocol["sumocfg"]
            or baseline["model_path"] != frozen["model_path"]
            or old["protocol"] != THRESHOLD_PROTOCOL or old["calibration_protocol"] != NATIVE_CALIBRATION_PROTOCOL
            or old["threshold"] != 0.0 or old["mode"] != "minimum_advantage" or old["comparison"] != COMPARISON
            or any(old[key] != frozen[key] for key in ("source_root", "model_path", "fit_result"))
            or any(baseline[key] != protocol[key] for key in ("threshold_originator_path", "ablation_originator_path"))
            or baseline["new_task_ids"] != controls["no_stay_threshold_zero_tasks"]
            or baseline["reused_task_ids"][PP] != controls["phase_pressure_tasks"]):
        raise ValueError("Reuse the unchanged S threshold-zero model and original same-network PP controls")
    reused = [row for row in baseline["cells"] if row["arm"] in (ZERO, PP)]
    if (len(reused) != 6 or {(row["seed"], row["arm"]) for row in reused}
            != {(seed, arm) for seed in SEEDS for arm in (ZERO, PP)}):
        raise ValueError("Require exactly the six frozen S and PP comparison cells")
    for row in reused:
        expected = BASELINE_PROTOCOL if row["arm"] == ZERO else PP_PROTOCOL
        tasks = controls["no_stay_threshold_zero_tasks" if row["arm"] == ZERO else "phase_pressure_tasks"]
        if (row["source_protocol"] != expected or row["task_id"] != tasks[SEEDS.index(row["seed"])]) or (
                row["arm"] == ZERO and (row["source_root"] != source or row["model_path"] != frozen["model_path"]
                                         or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL)):
            raise ValueError("A reused cell has a changed source, model or task binding")
    rows = sorted(cells, key=lambda row: SEEDS.index(row["seed"]))
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["arm"] != GATED or row["scenario"] != "cologne1"
                or row["source_root"] != source or row["repaired_sumocfg"] != protocol["sumocfg"]
                or row["model_path"] != frozen["model_path"] or row["frozen_threshold"] != frozen
                or any(row[key] != protocol[key] for key in ("threshold_originator_path", "ablation_originator_path"))
                or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL or row.get("reused", False)):
            raise ValueError("New runs must share the frozen model, threshold, source, no-stay originator and network")
    fields = ("seed", "run_valid", "status", "zero_incident_status", "source_result", "task_id",
              "elapsed_sec", "source_root", "model_path", "occupancy_equation_protocol", "validity_checks")
    compact = [{**{key: row[key] for key in fields if key in row}, "arm": NEW, "source_arm": GATED,
                "source_protocol": PROTOCOL, "reused": False,
                "metrics": {key: row["metrics"].get(key) for key in METRIC_KEYS},
                "collision_lane_pairs": row["collision_inventory"]["incident_counts_by_first_participant_lane_pair"]}
               for row in rows]
    compact.extend({**row, "reused": True} for row in reused)
    by_key = {(row["seed"], row["arm"]): row for row in compact}
    compact = [by_key[seed, arm] for seed in SEEDS for arm in ARMS]
    complete = all(row["run_valid"] for row in compact)
    return {
        "protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
        "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
        "repaired_sumocfg": protocol["sumocfg"], "source_root": source,
        "model_path": frozen["model_path"], "frozen_threshold": frozen,
        "threshold_zero_frozen_threshold": old,
        **{key: baseline[key] for key in ("baseline_source_bindings", "baseline_executor_source")},
        **{key: protocol[key] for key in ("threshold_originator_path", "ablation_originator_path")},
        "cells": compact, "safety_by_arm": {arm: safety_summary([row for row in compact if row["arm"] == arm])
                                             for arm in ARMS},
        "all_seed_service": service_summary(compact, SEEDS) if complete else None,
        "additional_seed_service": service_summary(compact, (52282, 63792)) if complete else None,
        "execution_diagnosis": execution_summary(rows, frozen, baseline),
        "new_task_ids": [row["task_id"] for row in rows],
        "reused_task_ids": {arm: [by_key[seed, arm]["task_id"] for seed in SEEDS] for arm in (ZERO, PP)},
        "new_simulation_count": 3, "reused_cell_count": 6, "new_model_fit_count": 0,
        "threshold_retuned": False,
        "new_simulation_elapsed_seconds_sum": sum(row["elapsed_sec"] for row in rows),
        "interpretation": "Only the frozen T threshold changes relative to S; model, no-stay rule, source and network remain fixed. Complete colliding runs remain in means; invalid runs disable means. The additional-two-seed summary excludes geometry-development seed41242. These previously inspected seeds provide development evidence, not independent confirmation.",
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
        row.update(source_result=str(path), task_id=tasks[f"CFCMT/v154u/recalibrated-no-stay-v1/evaluate/{seed}"])
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
