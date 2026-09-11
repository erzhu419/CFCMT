"""Compare fixed 450-second model spacing with continuous no-stay and PP control."""

import argparse
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize_cologne_recalibrated_no_stay import (
        PROTOCOL as BASELINE_PROTOCOL, CALIBRATION_PROTOCOL, THRESHOLD, NEW as CONTINUOUS,
        ORIGINATOR_PROTOCOL, PP_PROTOCOL, THRESHOLD_PROTOCOL, OCCUPANCY_PROTOCOL,
        COMPARISON, GATED, PP, SEEDS, SERVICE_KEYS, METRIC_KEYS, safety_summary)
else:
    from summarize_cologne_recalibrated_no_stay import (
        PROTOCOL as BASELINE_PROTOCOL, CALIBRATION_PROTOCOL, THRESHOLD, NEW as CONTINUOUS,
        ORIGINATOR_PROTOCOL, PP_PROTOCOL, THRESHOLD_PROTOCOL, OCCUPANCY_PROTOCOL,
        COMPARISON, GATED, PP, SEEDS, SERVICE_KEYS, METRIC_KEYS, safety_summary)

PROTOCOL = "tsc-v154x-cologne-spaced-no-stay-closed-loop-v1"
NEW = "spaced_no_stay_rigid"
ARMS = (NEW, CONTINUOUS, PP)
SPACING = {"control_interval_sec": 10, "cooldown_intervals": 44,
           "minimum_intervention_gap_sec": 450, "model_action_sec": 10, "pp_continuation_sec": 440}


def service_summary(cells, seeds):
    by_key = {(row["seed"], row["arm"]): row for row in cells}
    means = {arm: {metric: mean(by_key[seed, arm]["metrics"][metric] for seed in seeds)
                   for metric in SERVICE_KEYS} for arm in ARMS}
    result = {"seeds": list(seeds), "weighting": "equal_seed", "means_by_arm": means}
    for name, reference in (("spaced_vs_continuous", CONTINUOUS), ("spaced_vs_phase_pressure", PP)):
        result[name] = {
            metric: {"difference_mean": means[NEW][metric] - means[reference][metric],
                     "relative_difference_of_means_percent":
                         100 * (means[NEW][metric] / means[reference][metric] - 1)
                         if means[reference][metric] != 0 else None,
                     "paired_differences": [
                         {"seed": seed, "spaced_minus_reference":
                          by_key[seed, NEW]["metrics"][metric] - by_key[seed, reference]["metrics"][metric]}
                         for seed in seeds]}
            for metric in SERVICE_KEYS}
    return result


def execution_summary(rows, frozen, baseline, *, originator_protocol=ORIGINATOR_PROTOCOL):
    counts, traces = [], []
    for row in rows:
        diag, metrics = row["originator_diagnostics"], row["metrics"]
        guard, deployment = metrics["guard_audit"], metrics["residual_deployment"]
        if (diag["protocol"] != originator_protocol or diag["threshold"] != frozen["threshold"]
                or diag["threshold_mode"] != frozen["mode"] or diag["threshold_comparison"] != COMPARISON
                or diag["threshold_model_path"] != frozen["model_path"]):
            raise ValueError("Observed originator must retain the frozen T threshold and P model")
        if (row["spacing_contract"] != SPACING or deployment["mode"] != "direct"
                or deployment["cooldown_intervals"] != 44 or deployment["cooldown_intervals_override"] != 44):
            raise ValueError("Observed direct deployment must apply the fixed 44-interval cooldown")
        trace = metrics["accepted_intervention_trace"]
        accepted = guard["accepted_overrides"]
        if (accepted != len(trace) or len(trace) > 8
                or any(guard[key] != accepted for key in
                       ("executed_overrides", "accepted_switch_overrides", "executed_switch_overrides"))
                or any(guard[key] != 0 for key in ("proposed_stay_overrides", "accepted_stay_overrides", "executed_stay_overrides"))
                or any(value != 0 for key, value in guard.items()
                       if key.startswith("rejected_") and key != "rejected_cooldown")
                or any(not (item["request_accepted"] and item["execution_effective"]
                            and item["override_kind"] == "switch") for item in trace)):
            raise ValueError("Accepted trace must contain every effective switch, with only cooldown rejections")
        times = [item["time_sec"] for item in trace]
        gaps = [right - left for left, right in zip(times, times[1:])]
        if any(gap < 450 for gap in gaps):
            raise ValueError("Actual model interventions must be at least 450 seconds apart")
        item = {"seed": row["seed"], "decisions": diag["decision_count"],
                "raw_model_proposals": diag["ungated_rigid_proposal_count"],
                "threshold_rejected_proposals": diag["threshold_rejected_proposal_count"],
                "stay_veto_count": diag["model_stay_veto_count"],
                "originator_passed_proposals": diag["phase_pressure_override_count"],
                "actual_accepted_overrides": accepted, "cooldown_rejected_proposals": guard["rejected_cooldown"],
                "executed_switch_overrides": guard["executed_switch_overrides"],
                "executed_stay_overrides": guard["executed_stay_overrides"],
                "rejected_executor": guard["rejected_executor"],
                "phase_switches": metrics["phase_execution_audit"]["switches"]}
        if (item["raw_model_proposals"] != item["threshold_rejected_proposals"] + item["stay_veto_count"]
                + item["originator_passed_proposals"]
                or item["originator_passed_proposals"] != accepted + item["cooldown_rejected_proposals"]
                or guard["proposed_overrides"] != item["originator_passed_proposals"]):
            raise ValueError("Originator filters and coordinator cooldown must account for every proposal")
        counts.append(item)
        traces.append({"seed": row["seed"], "accepted_times_sec": times,
                       "minimum_intervention_gap_sec": min(gaps) if gaps else None,
                       "accepted_intervention_trace": trace,
                       "maximum_green_elapsed_sec_by_tls": row["maximum_green_elapsed_sec_by_tls"]})
    return {"per_seed": counts,
            "totals": {key: sum(item[key] for item in counts) for key in counts[0] if key != "seed"},
            "spacing_by_seed": traces,
            "continuous_baseline": {key: baseline["execution_diagnosis"][key] for key in
                                    ("per_seed", "totals", "maximum_green_elapsed_sec_by_seed")},
            "description": "Uncapped raw proposals first pass threshold and stay filters, then coordinator cooldown. Cooldown-rejected proposals are distinct from all PP decisions. Complete accepted traces verify actual switch spacing."}


def aggregate(protocol, cells, baseline):
    if (protocol["protocol"] != PROTOCOL or protocol["spacing_contract"] != SPACING or len(cells) != 3
            or {row["seed"] for row in cells} != set(SEEDS)):
        raise ValueError("Require the frozen X spacing protocol and exactly three evaluation seeds")
    frozen, source, controls = protocol["frozen_threshold"], protocol["source_root"], protocol["controls"]
    if (frozen["protocol"] != THRESHOLD_PROTOCOL or frozen["calibration_protocol"] != CALIBRATION_PROTOCOL
            or frozen["threshold"] != THRESHOLD or frozen["mode"] != "minimum_advantage"
            or frozen["comparison"] != COMPARISON or frozen["source_root"] != source
            or frozen["ablation_originator_path"] != protocol["ablation_originator_path"]):
        raise ValueError("Require the unchanged P model, T threshold and S no-stay rule")
    if (baseline["protocol"] != BASELINE_PROTOCOL or baseline["seeds"] != list(SEEDS)
            or baseline["source_root"] != source or baseline["repaired_sumocfg"] != protocol["sumocfg"]
            or baseline["model_path"] != frozen["model_path"] or baseline["frozen_threshold"] != frozen
            or any(baseline[key] != protocol[key] for key in ("threshold_originator_path", "ablation_originator_path"))
            or baseline["new_task_ids"] != controls["continuous_model_tasks"]
            or baseline["reused_task_ids"][PP] != controls["phase_pressure_tasks"]):
        raise ValueError("Reuse the frozen U continuous controller and its original same-network PP controls")
    reused = [row for row in baseline["cells"] if row["arm"] in (CONTINUOUS, PP)]
    if (len(reused) != 6 or {(row["seed"], row["arm"]) for row in reused}
            != {(seed, arm) for seed in SEEDS for arm in (CONTINUOUS, PP)}):
        raise ValueError("Require exactly six U and PP comparison cells")
    for row in reused:
        continuous = row["arm"] == CONTINUOUS
        tasks = controls["continuous_model_tasks" if continuous else "phase_pressure_tasks"]
        if (row["source_protocol"] != (BASELINE_PROTOCOL if continuous else PP_PROTOCOL)
                or row["task_id"] != tasks[SEEDS.index(row["seed"])]) or (
                continuous and (row["source_root"] != source or row["model_path"] != frozen["model_path"]
                                or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL)):
            raise ValueError("A reused cell changed its source, model or task binding")
    rows = sorted(cells, key=lambda row: SEEDS.index(row["seed"]))
    for row in rows:
        if (row["protocol"] != PROTOCOL or row["arm"] != GATED or row["scenario"] != "cologne1"
                or row["source_root"] != source or row["repaired_sumocfg"] != protocol["sumocfg"]
                or row["model_path"] != frozen["model_path"] or row["frozen_threshold"] != frozen
                or any(row[key] != protocol[key] for key in ("threshold_originator_path", "ablation_originator_path"))
                or row["occupancy_equation_protocol"] != OCCUPANCY_PROTOCOL or row.get("reused", False)):
            raise ValueError("New cells must share the fixed controller, model, source and network")
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
    return {"protocol": PROTOCOL, "matrix_status": "COMPLETE" if complete else "INVALID",
            "seeds": list(SEEDS), "geometry_development_seed": 41242, "additional_seeds": [52282, 63792],
            "repaired_sumocfg": protocol["sumocfg"], "source_root": source,
            "model_path": frozen["model_path"], "frozen_threshold": frozen, "spacing_contract": SPACING,
            **{key: baseline[key] for key in ("baseline_source_bindings", "baseline_executor_source")},
            **{key: protocol[key] for key in ("threshold_originator_path", "ablation_originator_path")},
            "cells": compact, "safety_by_arm": {arm: safety_summary([row for row in compact if row["arm"] == arm])
                                                 for arm in ARMS},
            "all_seed_service": service_summary(compact, SEEDS) if complete else None,
            "additional_seed_service": service_summary(compact, (52282, 63792)) if complete else None,
            "execution_diagnosis": execution_summary(rows, frozen, baseline),
            "new_task_ids": [row["task_id"] for row in rows],
            "reused_task_ids": {arm: [by_key[seed, arm]["task_id"] for seed in SEEDS] for arm in (CONTINUOUS, PP)},
            "new_simulation_count": 3, "reused_cell_count": 6, "new_model_fit_count": 0, "threshold_retuned": False,
            "new_simulation_elapsed_seconds_sum": sum(row["elapsed_sec"] for row in rows),
            "interpretation": "Only residual cooldown changes from U: an accepted model action reserves 10 seconds, followed by 440 seconds of PP continuation. Complete colliding runs remain in means; invalid runs disable means. These previously inspected seeds provide development evidence; the additional-two-seed summary excludes geometry-development seed41242."}


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
        row.update(source_result=str(path), task_id=tasks[f"CFCMT/v154x/spaced-no-stay-v1/evaluate/{seed}"])
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
