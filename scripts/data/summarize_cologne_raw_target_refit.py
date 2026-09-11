"""Compare the raw-target B100 refit with the fixed normalized-target OOF control."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

PROTOCOL = "tsc-v155a-cologne-raw-target-refit-v1"
BALANCED_PROTOCOL = "tsc-v155d-cologne-group-balanced-raw-refit-v1"
SEEDS = (2027, 3037, 4047)
GROUP_COUNTS = {"2027": 34, "3037": 33, "4047": 33}
COST_UNITS = "native 450-second mean halted vehicles per controlled lane"
OLD_SCORE_UNITS = "group-normalized predicted score advantage"
NEW_SCORE_UNITS = "raw predicted cost advantage in native halted vehicles per controlled lane"
TOLERANCE = 1e-12
CHECKS = ("training_roster_exact", "feature_order_exact", "original_t_baseline_exact", "raw_target_not_clipped",
          "sample_weight_source_unchanged", "serialized_scores_exact", "reference_predictions_zero")
BALANCED_CHECKS = ("training_roster_exact", "feature_order_exact", "original_a_baseline_exact", "raw_target_not_clipped",
                   "base_group_weights_uniform", "serialized_scores_exact", "reference_predictions_zero")


def comparison(newcalib, oldcalib, old_score_units=OLD_SCORE_UNITS):
    """Compare selected policies in native cost units, retaining all100 denominators."""
    old, new = oldcalib["selected"], newcalib["selected"]
    for policy in (old, new):
        if (policy["groups"] != 100 or policy["weighting"] != "equal_seed"
                or {seed: row["groups"] for seed, row in policy["per_seed"].items()} != GROUP_COUNTS
                or any(policy[key] != sum(row[key] for row in policy["per_seed"].values())
                       for key in ("accepted_overrides", "beneficial", "harmful", "equal_cost"))
                or any(row["accepted_overrides"] != row["beneficial"] + row["harmful"] + row["equal_cost"]
                       for row in policy["per_seed"].values())
                or any(abs(policy[key] - mean(row[key] for row in policy["per_seed"].values())) > TOLERANCE
                       for key in ("mean_cost", "mean_phase_pressure_cost", "mean_cost_minus_phase_pressure"))):
            raise ValueError("Both policies must retain the same100 OOF groups and seed denominators")
    per_seed = []
    for seed in map(str, SEEDS):
        before, after = old["per_seed"][seed], new["per_seed"][seed]
        if abs(before["mean_phase_pressure_cost"] - after["mean_phase_pressure_cost"]) > TOLERANCE:
            raise ValueError("The matched native labels or PP reference changed")
        per_seed.append({"seed": int(seed), "groups": before["groups"],
            "old_policy_cost": before["mean_cost"], "new_policy_cost": after["mean_cost"],
            "phase_pressure_cost": before["mean_phase_pressure_cost"],
            "new_minus_old_policy_cost": after["mean_cost"] - before["mean_cost"],
            "new_minus_phase_pressure_cost": after["mean_cost"] - after["mean_phase_pressure_cost"],
            "old_outcomes": {key: before[key] for key in ("accepted_overrides", "beneficial", "harmful", "equal_cost")},
            "new_outcomes": {key: after[key] for key in ("accepted_overrides", "beneficial", "harmful", "equal_cost")}})
    old_cost, new_cost = [mean(row[key] for row in per_seed) for key in ("old_policy_cost", "new_policy_cost")]
    pp = mean(row["phase_pressure_cost"] for row in per_seed)
    return {"groups": 100, "weighting": "equal_seed", "cost_units": COST_UNITS,
            "old_score_units": old_score_units, "new_score_units": NEW_SCORE_UNITS,
            "old_selected_policy": old, "new_selected_policy": new,
            "new_minus_old_policy_cost": new_cost - old_cost,
            "new_relative_to_old_policy_percent": 100 * (new_cost / old_cost - 1) if old_cost else None,
            "new_minus_phase_pressure_cost": new_cost - pp,
            "new_relative_to_phase_pressure_percent": 100 * (new_cost / pp - 1) if pp else None,
            "per_seed": per_seed,
            "threshold_interpretation": "Both models predict native raw cost advantage; policy effects use matched OOF costs."
            if old_score_units == NEW_SCORE_UNITS else
            "Each threshold belongs to its model's score units; their numerical difference is not a comparison of control strength."}


def _action_metrics(rows):
    return {"groups": len(rows),
            "effective_action_changed": sum(row["old_effective_index"] != row["new_effective_index"] for row in rows),
            "deployed_action_changed": sum(row["old_deployed_index"] != row["new_deployed_index"] for row in rows),
            "new_lower_cost": sum(row["new_cost"] < row["old_cost"] - TOLERANCE for row in rows),
            "new_higher_cost": sum(row["new_cost"] > row["old_cost"] + TOLERANCE for row in rows),
            "equal_cost": sum(abs(row["new_cost"] - row["old_cost"]) <= TOLERANCE for row in rows),
            **{name: sum((row["old_deployed_index"] != row["reference_index"],
                         row["new_deployed_index"] != row["reference_index"]) == pair for row in rows)
               for name, pair in (("both_accept", (True, True)), ("old_accept_only", (True, False)),
                                  ("new_accept_only", (False, True)), ("neither_accept", (False, False)))},
            **{key: mean(row[key] for row in rows) for key in
               ("old_cost", "new_cost", "reference_cost", "old_regret", "new_regret", "old_cost_rank", "new_cost_rank")}}


def action_comparison(matched_actions, old_threshold, new_threshold, old_score_key="old_normalized_predicted_advantage"):
    if (len(matched_actions) != 100 or len({row["group_id"] for row in matched_actions}) != 100
            or {str(seed): sum(row["seed"] == seed for row in matched_actions) for seed in SEEDS} != GROUP_COUNTS):
        raise ValueError("Action comparison requires all100 matched groups with original seed counts")
    rows = []
    for original in matched_actions:
        row = dict(original)
        costs, ref = row["label_costs"], row["reference_index"]
        if len(costs) != 8 or not all(math.isfinite(cost) for cost in costs) or ref not in range(8):
            raise ValueError("All eight native action costs and the PP reference must be preserved")
        row["reference_cost"], row["oracle_cost"] = costs[ref], min(costs)
        for arm, threshold, score_key in (("old", old_threshold, old_score_key),
                                          ("new", new_threshold, "new_raw_predicted_advantage")):
            effective, deployed, score = row[arm + "_effective_index"], row[arm + "_deployed_index"], row[score_key]
            if effective not in range(8) or deployed not in range(8) or not math.isfinite(score):
                raise ValueError("Matched action indices and predictions must be valid")
            accepted = threshold is not None and effective != ref and score > threshold + TOLERANCE
            if deployed != (effective if accepted else ref):
                raise ValueError("Deployed actions must follow each model's own frozen threshold")
            row[arm + "_cost"] = costs[deployed]
            row[arm + "_regret"] = costs[deployed] - min(costs)
            row[arm + "_cost_rank"] = 1 + sum(cost < costs[deployed] - TOLERANCE for cost in costs)
        row["new_minus_old_cost"] = row["new_cost"] - row["old_cost"]
        rows.append(row)
    per_seed = {str(seed): _action_metrics([row for row in rows if row["seed"] == seed]) for seed in SEEDS}
    return {"rows": rows, "groups": 100, "weighting": "equal_seed_for_costs_and_regrets",
            "per_seed": per_seed,
            "totals": {key: sum(subset[key] for subset in per_seed.values()) for key in
                       ("effective_action_changed", "deployed_action_changed", "new_lower_cost", "new_higher_cost",
                        "equal_cost", "both_accept", "old_accept_only", "new_accept_only", "neither_accept")},
            "means": {key: mean(subset[key] for subset in per_seed.values()) for key in
                      ("old_cost", "new_cost", "reference_cost", "old_regret", "new_regret", "old_cost_rank", "new_cost_rank")},
            "rank_definition": "One plus the number of the eight raw candidate costs strictly below deployed cost; equal-cost actions share rank."}


def aggregate(protocol, result):
    balanced = protocol["protocol"] == BALANCED_PROTOCOL
    if (protocol["protocol"] not in (PROTOCOL, BALANCED_PROTOCOL) or result["protocol"] != protocol["protocol"] or result["status"] != "COMPLETE"
            or result["selected_groups"] != protocol["training_group_ids"] or len(result["selected_groups"]) != 100
            or result["folds"] != protocol["folds"]
            or result["anchor_feature_names"] != protocol["expected_anchor_feature_names"]
            or any(result[key] != protocol[key] for key in ("source_root", "sumocfg", "old_model_path"))
            or result["score_units"] != "raw_native_halted_per_lane_cost_difference"
            or result["old_calibration"]["selected"] != protocol["expected_original_selected"]):
        raise ValueError("Require the complete frozen raw-target fit, original features/folds and designated control")
    checks = result.get("checks", {})
    fits = result["fit_diagnostics"]
    if (any(checks.get(key) is not True for key in (BALANCED_CHECKS if balanced else CHECKS))
            or not all(value is True for value in checks.values())
            or not fits["full"] or len(fits["oof"]) != 3
            or sorted(row["heldout_seed"] for row in fits["oof"]) != list(SEEDS)
            or result["new_model_fits"] != 4 or result["full_model_fits"] != 1 or result["oof_fold_fits"] != 3
            or any(result[key] != 0 for key in ("inactive_correction_fits", "new_labels", "new_simulations"))):
        raise ValueError("Raw targets, the declared weights, serialized scores and all four fits must be verified")
    if balanced:
        old_frozen = protocol["old_frozen_threshold"]
        if (old_frozen.get("calibration_protocol") != PROTOCOL
                or old_frozen.get("score_units") != "raw_native_halted_per_lane_cost_difference"):
            raise ValueError("The weight ablation must compare against the frozen A raw-target model")
        for diagnostic in [fits["full"], *(row["diagnostics"] for row in fits["oof"])]:
            if (diagnostic["sample_weight_source_protocol"] != "domain_group_balanced_v1"
                    or diagnostic["sign_balance"]["enabled"] is not False
                    or any(abs(diagnostic[key] - 1) > TOLERANCE for key in
                           ("sample_weight_min", "sample_weight_max", "sample_weight_mean"))):
                raise ValueError("Each D fit must use uniform domain/group candidate weights without sign balancing")
    selected, frozen = result["calibration"]["selected"], result["frozen_threshold"]
    if (frozen["model_path"] != result["model_path"] or frozen["source_root"] != protocol["source_root"]
            or frozen["calibration_protocol"] != protocol["protocol"] or frozen["threshold"] != selected["threshold"]
            or frozen["mode"] != selected["mode"]
            or frozen["ablation_originator_path"] != protocol["ablation_originator_path"]):
        raise ValueError("The saved threshold must bind the new model and selected raw-score policy")
    compared = comparison(result["calibration"], result["old_calibration"], NEW_SCORE_UNITS if balanced else OLD_SCORE_UNITS)
    actions = action_comparison(result["matched_actions"], protocol["old_frozen_threshold"]["threshold"], selected["threshold"],
                                "old_raw_predicted_advantage" if balanced else "old_normalized_predicted_advantage")
    if {row["group_id"] for row in actions["rows"]} != set(protocol["training_group_ids"]):
        raise ValueError("Matched actions must use the original100 group IDs")
    for seed in map(str, SEEDS):
        rows = [row for row in actions["rows"] if row["seed"] == int(seed)]
        for arm, calibration in (("old", result["old_calibration"]), ("new", result["calibration"])):
            metrics = calibration["selected"]["per_seed"][seed]
            accepted = [row for row in rows if row[arm + "_deployed_index"] != row["reference_index"]]
            gains = [row["reference_cost"] - row[arm + "_cost"] for row in accepted]
            actual = {"accepted_overrides": len(accepted), "beneficial": sum(gain > TOLERANCE for gain in gains),
                      "harmful": sum(gain < -TOLERANCE for gain in gains), "equal_cost": sum(abs(gain) <= TOLERANCE for gain in gains)}
            if (abs(mean(row[arm + "_cost"] for row in rows) - metrics["mean_cost"]) > TOLERANCE
                    or abs(mean(row["reference_cost"] for row in rows) - metrics["mean_phase_pressure_cost"]) > TOLERANCE
                    or any(actual[key] != metrics[key] for key in actual)):
                raise ValueError("The100 matched deployed actions must reproduce both selected policy costs and outcomes")
    return {"protocol": protocol["protocol"], "status": "COMPLETE",
            **{key: result[key] for key in ("source_root", "sumocfg", "old_model_path", "model_path", "frozen_threshold",
                                          "score_units", "selected_groups", "folds", "anchor_feature_names", "fit_diagnostics", "checks")},
            "policy_comparison": compared, "action_comparison": actions,
            "proposal_accounting": result["proposal_accounting"],
            "accounting": {key: result[key] for key in ("new_model_fits", "full_model_fits", "oof_fold_fits",
                                                       "inactive_correction_fits", "new_labels", "new_simulations", "elapsed_sec")},
            "threshold_selection_data": "Original100 training-seed OOF groups only; same no-stay veto and threshold-selection rule",
            "limitations": protocol["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "result", "launch-record", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    report = aggregate(json.loads(args.protocol.read_text()), result)
    launch = json.loads(args.launch_record.read_text())
    submitted = json.loads(launch["scheduler_stdout"])["submitted"]
    signature = "CFCMT/v155d/group-balanced-raw-refit/v1" if report["protocol"] == BALANCED_PROTOCOL else "CFCMT/v155a/raw-target-refit/v1"
    task = next(row["id"] for row in submitted if row["signature"] == signature)
    report["accounting"].update(task_id=task, retrieved_result_bytes=args.result.stat().st_size)
    report["refs"] = {"protocol": str(args.protocol), "result": str(args.result), "launch_record": str(args.launch_record)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "policy_comparison": report["policy_comparison"],
                      "action_totals": report["action_comparison"]["totals"], "accounting": report["accounting"]}))


if __name__ == "__main__":
    main()
