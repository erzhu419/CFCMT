"""Summarize the fixed 21 native-prefix model-versus-PP action comparisons."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

PROTOCOL = "tsc-v154y-cologne-spaced-action-counterfactual-v1"
RAW_PROTOCOL = "tsc-v155c-cologne-raw-target-action-counterfactual-v1"
SEEDS = (52282, 41242, 63792)
ARMS = ("model_once_then_pp", "pp_only")
TOLERANCE = 1e-12
SCORE_UNITS = "group-normalized predicted score advantage"
RAW_SCORE_UNITS = "raw_native_halted_per_lane_cost_difference"
GAIN_UNITS = "native 450-second mean halted vehicles per controlled lane"
PAIR_CHECKS = ("both_target_captured", "physical_start_exact", "pending_start_exact", "prefix_decisions_exact")
BRANCH_CHECKS = ("evaluator_complete", "native_reference_trace_exact", "full_horizon", "target_captured",
                 "selected_action_exact", "requested_action_exact", "pp_reference_exact",
                 "predicted_advantage_exact", "single_tls_eight_lanes", "active_population_accounting")
LABEL_CONTRACT = {"control_interval_sec": 10, "model_action_sec": 10, "pp_continuation_sec": 440,
                  "horizon_sec": 450, "cost_mode": "halted_queue", "controlled_lane_count": 8}


def action_sign(gain):
    return "beneficial" if gain > TOLERANCE else "harmful" if gain < -TOLERANCE else "tie"


def association(rows, score_units=SCORE_UNITS):
    predicted = [row["predicted_advantage"] for row in rows]
    measured = [row["actual_advantage"] for row in rows]
    pmean, amean = mean(predicted), mean(measured)
    centered_p, centered_a = [p - pmean for p in predicted], [a - amean for a in measured]
    pvar, avar = sum(p * p for p in centered_p), sum(a * a for a in centered_a)
    return {"count": len(rows),
            "score_units": score_units, "gain_units": GAIN_UNITS,
            "pearson": sum(p * a for p, a in zip(centered_p, centered_a)) / math.sqrt(pvar * avar)
            if pvar > 0 and avar > 0 else None,
            "pearson_unavailable_reason": None if pvar > 0 and avar > 0 else "constant_prediction_or_measured_gain",
            "interpretation": "Descriptive association of raw predicted and measured gains in matching units."
            if score_units == RAW_SCORE_UNITS else
            "Descriptive association across differently scaled quantities; it is not a prediction calibration error."}


def sign_agreement(rows):
    agreement = sum(action_sign(row["predicted_advantage"]) == action_sign(row["actual_advantage"]) for row in rows)
    return {"score_sign_agreement_count": agreement, "score_sign_disagreement_count": len(rows) - agreement}


def raw_prediction_diagnostics(rows):
    errors = [row["predicted_advantage"] - row["actual_advantage"] for row in rows]
    return {"count": len(rows), "units": GAIN_UNITS, "prediction_bias": mean(errors),
            "prediction_mean_absolute_error": mean(abs(error) for error in errors),
            "overprediction_count": sum(error > TOLERANCE for error in errors),
            "underprediction_count": sum(error < -TOLERANCE for error in errors),
            "equal_prediction_count": sum(abs(error) <= TOLERANCE for error in errors),
            "error_definition": "raw_predicted_gain_minus_raw_measured_gain"}


def segment_summary(rows):
    if not all(len(row.get("per_150_seconds", [])) == 3 for row in rows):
        return None
    return [{"segment": label, "offset_start_sec": 150 * index, "offset_end_sec": 150 * (index + 1),
             "model_mean_cost": mean(row["per_150_seconds"][index]["model_cost"] for row in rows),
             "phase_pressure_mean_cost": mean(row["per_150_seconds"][index]["phase_pressure_cost"] for row in rows),
             "mean_actual_advantage": mean(row["per_150_seconds"][index]["actual_advantage"] for row in rows)}
            for index, label in enumerate(("early", "middle", "late"))]


def cost_summary(rows, raw_scores=False):
    score_units = RAW_SCORE_UNITS if raw_scores else SCORE_UNITS
    per_seed = []
    for seed in SEEDS:
        subset = [row for row in rows if row["seed"] == seed]
        per_seed.append({"seed": seed, "pairs": len(subset),
            "model_mean_cost": mean(row["model_cost"] for row in subset),
            "phase_pressure_mean_cost": mean(row["phase_pressure_cost"] for row in subset),
            "mean_actual_advantage": mean(row["actual_advantage"] for row in subset),
            "mean_predicted_advantage": mean(row["predicted_advantage"] for row in subset),
            **{sign: sum(row["sign"] == sign for row in subset) for sign in ("beneficial", "harmful", "tie")},
            **sign_agreement(subset), "descriptive_score_gain_association": association(subset, score_units),
            **({"raw_prediction_diagnostics": raw_prediction_diagnostics(subset),
                "per_150_seconds": segment_summary(subset)} if raw_scores else {})})
    model = mean(row["model_mean_cost"] for row in per_seed)
    pp = mean(row["phase_pressure_mean_cost"] for row in per_seed)
    return {"pair_count": len(rows), "weighting": "equal_seed", "per_seed": per_seed,
            "score_units": score_units, "gain_units": GAIN_UNITS,
            "model_mean_cost": model, "phase_pressure_mean_cost": pp,
            "mean_actual_advantage": mean(row["mean_actual_advantage"] for row in per_seed),
            "mean_predicted_advantage": mean(row["mean_predicted_advantage"] for row in per_seed),
            "model_minus_phase_pressure_percent": 100 * (model / pp - 1) if pp else None,
            **{sign: sum(row[sign] for row in per_seed) for sign in ("beneficial", "harmful", "tie")},
            **sign_agreement(rows), "descriptive_score_gain_association": association(rows, score_units),
            **({"raw_prediction_diagnostics": raw_prediction_diagnostics(rows),
                "per_150_seconds": segment_summary(rows)} if raw_scores else {})}


def required_checks_pass(checks, required):
    return all(checks.get(key) is True for key in required) and all(value is True for value in checks.values())


def compact_pair(window, result, group, common_errors, raw_scores=False):
    errors = list(common_errors)
    row = {"seed": window["seed"], "time_sec": window["time_sec"], "end_time_sec": window["window_end_sec"],
           "predicted_advantage": window["selected_trace"]["priority"],
           "model_cost": None, "phase_pressure_cost": None, "actual_advantage": None, "sign": None}
    if group is None:
        row.update(status="MISSING", valid=False, validity_errors=[*errors, "missing_pair"])
        return row
    if (group["seed"] != window["seed"] or group["target_time_sec"] != window["time_sec"]
            or group["end_time_sec"] != window["window_end_sec"] or group["selected_trace"] != window["selected_trace"]):
        errors.append("frozen_window_or_action_changed")
    if group["status"] != "PASS" or not required_checks_pass(group.get("checks", {}), PAIR_CHECKS):
        errors.append("native_pair_checks_failed")
    runs = group.get("runs", {})
    diagnostic = {}
    for arm, cost_key in zip(ARMS, ("model_cost", "phase_pressure_cost")):
        branch = runs.get(arm, {})
        cost = branch.get("mean_halted_per_lane")
        finite_cost = isinstance(cost, (int, float)) and math.isfinite(cost)
        row[cost_key] = cost if finite_cost else None
        if (branch.get("status") != "PASS" or branch.get("sample_count") != 450 or not finite_cost
                or not required_checks_pass(branch.get("checks", {}), BRANCH_CHECKS)):
            errors.append(arm + "_incomplete_or_invalid")
        diagnostic[arm] = {
            phase: {key: branch.get(field, {}).get(key) for key in
                    ("raw_collision_events", "collision_event_steps", "unique_collision_incidents",
                     "starting_teleports", "ending_teleports")}
            for phase, field in (("prefix", "prefix_safety"), ("window", "safety"))}
        diagnostic[arm]["runtime_error"] = branch.get("runtime_error")
    row.update(valid=not errors, status="PASS" if not errors else "INVALID", validity_errors=errors,
               pair_checks=group.get("checks", {}), branch_diagnostics=diagnostic,
               task_id=result.get("task_id"), source_result=result.get("source_result"))
    if row["valid"]:
        row["actual_advantage"] = row["phase_pressure_cost"] - row["model_cost"]
        row["sign"] = action_sign(row["actual_advantage"])
        row["score_sign_agrees"] = action_sign(row["predicted_advantage"]) == row["sign"]
        if raw_scores:
            row["prediction_error"] = row["predicted_advantage"] - row["actual_advantage"]
            row["per_150_seconds"] = []
            segments = [runs[arm].get("per_150_seconds", []) for arm in ARMS]
            if all(len(items) == 3 for items in segments):
                for items, cost_key in zip(segments, ("model_cost", "phase_pressure_cost")):
                    if abs(mean(item["mean_halted_per_lane"] for item in items) - row[cost_key]) > TOLERANCE:
                        raise ValueError("Three equal150-second segment means must reproduce the450-second cost")
                for index, (model, pp) in enumerate(zip(*segments)):
                    start = window["time_sec"] + index * 150
                    if any((item["start_sec"], item["end_sec"]) != (start, start + 150) for item in (model, pp)):
                        raise ValueError("Segment costs must match the three fixed150-second intervals")
                    row["per_150_seconds"].append({"offset_start_sec": index * 150, "offset_end_sec": (index + 1) * 150,
                        "model_cost": model["mean_halted_per_lane"], "phase_pressure_cost": pp["mean_halted_per_lane"],
                        "actual_advantage": pp["mean_halted_per_lane"] - model["mean_halted_per_lane"]})
    return row


def aggregate(protocol, results):
    raw_scores = protocol["protocol"] == RAW_PROTOCOL
    if raw_scores:
        frozen = protocol["frozen_threshold"]
        if (frozen.get("calibration_protocol") != "tsc-v155a-cologne-raw-target-refit-v1"
                or frozen.get("score_units") != RAW_SCORE_UNITS or protocol.get("score_units") != RAW_SCORE_UNITS
                or frozen["threshold"] != 0 or frozen["mode"] != "minimum_advantage"):
            raise ValueError("Raw prediction errors require the frozen A raw-score model and threshold zero")
    windows = protocol["windows"]
    keys = [(window["seed"], window["time_sec"]) for window in windows]
    if (protocol["protocol"] not in (PROTOCOL, RAW_PROTOCOL) or protocol["seeds"] != list(SEEDS)
            or tuple(protocol["branches"]) != ARMS or protocol["cooldown_intervals"] != 44
            or protocol["label_contract"] != LABEL_CONTRACT or len(set(keys)) != 21 or len(keys) != 21
            or any(sum(seed == window["seed"] for window in windows) != 7 for seed in SEEDS)
            or any(window["window_end_sec"] != window["time_sec"] + 450
                   or window["window_end_sec"] > 28800 for window in windows)):
        raise ValueError("Require exactly the21 frozen complete action windows and original scalar label contract")
    by_seed = {result["seed"]: result for result in results}
    if len(by_seed) != len(results) or set(by_seed) - set(SEEDS):
        raise ValueError("Per-seed results contain a duplicated or unfrozen evaluation seed")
    rows, seed_records = [], []
    for seed in SEEDS:
        result = by_seed.get(seed)
        expected = [window for window in windows if window["seed"] == seed]
        if result is None:
            rows.extend(compact_pair(window, None, None, ["missing_seed_result"], raw_scores) for window in expected)
            seed_records.append({"seed": seed, "status": "MISSING"})
            continue
        errors = []
        for key in ("protocol", "source_root", "sumocfg", "frozen_threshold", "cooldown_intervals", "label_contract") + (
                ("score_units",) if raw_scores else ()):
            if result.get(key) != protocol[key]:
                errors.append("seed_binding_changed:" + key)
        groups = result.get("groups", [])
        times = [group["target_time_sec"] for group in groups]
        if len(set(times)) != len(times) or set(times) - {window["time_sec"] for window in expected}:
            errors.append("duplicated_or_unfrozen_group")
        by_time = {group["target_time_sec"]: group for group in groups}
        rows.extend(compact_pair(window, result, by_time.get(window["time_sec"]), errors, raw_scores) for window in expected)
        seed_records.append({"seed": seed, "status": result["status"], "task_id": result.get("task_id"),
                             "source_result": result.get("source_result"), "binding_errors": errors})
    complete = all(row["valid"] for row in rows)
    ordered = sorted(rows, key=lambda row: (SEEDS.index(row["seed"]), row["time_sec"]))
    extrema_fields = ("seed", "time_sec", "model_cost", "phase_pressure_cost", "predicted_advantage", "actual_advantage", "sign")
    extrema = None
    if complete:
        extrema = {name: {key: choice(ordered, key=lambda row: row["actual_advantage"])[key] for key in extrema_fields}
                   for name, choice in (("best_measured_window", max), ("worst_measured_window", min))}
    return {"protocol": protocol["protocol"], "status": "COMPLETE" if complete else "INCOMPLETE",
            "source_root": protocol["source_root"], "sumocfg": protocol["sumocfg"],
            "frozen_threshold": protocol["frozen_threshold"], "cooldown_intervals": 44,
            "score_units": RAW_SCORE_UNITS if raw_scores else SCORE_UNITS, "gain_units": GAIN_UNITS,
            "label_contract": LABEL_CONTRACT, "selection_rule": protocol["selection_rule"],
            "expected_pair_count": 21, "valid_pair_count": sum(row["valid"] for row in rows),
            "missing_pair_count": sum(row["status"] == "MISSING" for row in rows),
            "seeds": list(SEEDS), "seed_results": seed_records, "pairs": ordered,
            "summary": cost_summary(ordered, raw_scores) if complete else None, "descriptive_extrema": extrema,
            "accounting": {"planned_new_simulations": 42,
                           "new_simulations_reported": sum(result["new_simulation_count"] for result in results),
                           "new_model_fits": 0, "threshold_retuned": False, "spacing_retuned": False,
                           "runner_elapsed_sec_sum": sum(result["elapsed_sec"] for result in results),
                           "retrieved_result_count": len(results),
                           "task_ids": [result.get("task_id") for result in results]},
            "interpretation": "Positive native gain means PP cost exceeds model cost. " + (
                "Raw predicted and measured gains share native units, so signed bias and absolute prediction error are valid. "
                if raw_scores else "Model scores use group normalization, while native gains retain halted-vehicle units; signs and descriptive association are reported without subtracting the two scales. The two replay arms cannot reconstruct the original eight-action group normalization. ")
                + "Collisions remain branch diagnostics and do not exclude valid pairs. Summary and extrema require all21 valid pairs and select no threshold or policy.",
            "collision_accounting": "Prefix and window counts remain separate for each branch; repeated-prefix incidents are not summed as unique experiment-wide incidents.",
            "limitations": protocol.get("limitations", [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "results-root", "launch-record", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    signature = "CFCMT/v155c/raw-target-action-counterfactual-v1" if protocol["protocol"] == RAW_PROTOCOL else "CFCMT/v154y/spaced-action-counterfactual-v1"
    launch = json.loads(args.launch_record.read_text())
    tasks = {row["signature"]: row["id"] for row in json.loads(launch["scheduler_stdout"])["submitted"]}
    results, retrieved_bytes = [], 0
    for seed in SEEDS:
        path = args.results_root / "replay" / f"seed_{seed}" / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result["seed"] != seed:
            raise ValueError(f"Result seed does not match path: {path}")
        result.update(source_result=str(path), task_id=tasks[f"{signature}/{seed}"])
        results.append(result)
        retrieved_bytes += path.stat().st_size
    summary = aggregate(protocol, results)
    summary["accounting"]["retrieved_result_bytes"] = retrieved_bytes
    summary["refs"] = {"protocol": str(args.protocol), "launch_record": str(args.launch_record)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "valid_pair_count", "summary", "accounting")}))


if __name__ == "__main__":
    main()
