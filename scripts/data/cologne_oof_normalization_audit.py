"""Audit the frozen no-stay OOF selections in their original eight-action scale."""

from statistics import mean

import numpy as np

from cf_h2o.traffic_signal.action_scaling import action_group_range

TOLERANCE = 1e-12
TARGET_CLIP = 5.0


def _distribution(values):
    return {"count": len(values), "minimum": float(np.min(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "maximum": float(np.max(values)) if values else None}


def _normalized_diagnostics(rows):
    predicted = np.asarray([row["normalized_predicted_gain"] for row in rows])
    truth = np.asarray([row["normalized_true_gain"] for row in rows])
    residuals = predicted - truth
    correlation = None
    if len(rows) > 1 and np.ptp(predicted) > 0 and np.ptp(truth) > 0:
        correlation = float(np.corrcoef(predicted, truth)[0, 1])
    return {"accepted_groups": len(rows), "weighting": "equal_accepted_group",
            "bias": float(np.mean(residuals)) if len(rows) else None,
            "mean_absolute_error": float(np.mean(np.abs(residuals))) if len(rows) else None,
            "score_vs_normalized_gain_pearson": correlation,
            "residual_definition": "normalized_predicted_gain_minus_normalized_true_gain"}


def _gain_sums(rows, key):
    gains = [row[key] for row in rows]
    return {"positive_gain_sum": sum(gain for gain in gains if gain > TOLERANCE),
            "negative_gain_sum": sum(gain for gain in gains if gain < -TOLERANCE),
            "net_gain_sum": sum(gains), "weighting": "equal_accepted_group_sum"}


def audit_records(effective_records, threshold):
    """Keep the existing gate and eight-action labels; never refit or reselect."""
    if not effective_records or len({row["group_id"] for row in effective_records}) != len(effective_records):
        raise ValueError("Require the complete nonduplicated effective OOF roster")
    accepted, all_ranges, policy_rows = [], [], []
    for row in effective_records:
        costs, scores = np.asarray(row["label_costs"], dtype=float), np.asarray(row["predicted_scores"], dtype=float)
        ref, sel = row["reference_index"], row["selected_index"]
        if (costs.shape != (8,) or scores.shape != (8,) or ref not in range(8) or sel not in range(8)
                or not np.all(np.isfinite(costs)) or not np.all(np.isfinite(scores))):
            raise ValueError("Require all eight finite raw labels and aligned OOF scores")
        predicted = float(scores[ref] - scores[sel])
        if row["different_action"] != (ref != sel) or abs(predicted - row["predicted_advantage"]) > TOLERANCE:
            raise ValueError("Effective OOF action or cached advantage changed")
        delta = costs - costs[ref]
        raw_range = float(np.ptp(costs))
        scale = action_group_range(delta)
        all_ranges.append({"raw_cost_range": raw_range, "training_scale": scale})
        passes = threshold is not None and row["different_action"] and predicted > threshold + TOLERANCE
        policy_rows.append({"seed": row["seed"], "reference_cost": float(costs[ref]),
                            "policy_cost": float(costs[sel] if passes else costs[ref]), "accepted": passes})
        if not passes:
            continue
        truth = -float(np.clip(delta[sel] / scale, -TARGET_CLIP, TARGET_CLIP))
        raw_gain = float(costs[ref] - costs[sel])
        sign = "beneficial" if raw_gain > TOLERANCE else "harmful" if raw_gain < -TOLERANCE else "tie"
        accepted.append({"group_id": row["group_id"], "seed": row["seed"],
            "reference_index": ref, "selected_index": sel, "label_costs": costs.tolist(),
            "raw_cost_minimum": float(np.min(costs)), "raw_cost_maximum": float(np.max(costs)),
            "raw_cost_range": raw_range, "training_scale": scale,
            "normalized_predicted_gain": predicted, "raw_actual_gain": raw_gain,
            "normalized_true_gain": truth, "normalized_residual": predicted - truth, "sign": sign})

    seeds = sorted({row["seed"] for row in effective_records})
    per_seed_policy = {}
    for seed in seeds:
        subset = [row for row in policy_rows if row["seed"] == seed]
        pp = mean(row["reference_cost"] for row in subset)
        policy = mean(row["policy_cost"] for row in subset)
        per_seed_policy[str(seed)] = {"groups": len(subset), "accepted_overrides": sum(row["accepted"] for row in subset),
            "mean_cost": policy, "mean_phase_pressure_cost": pp,
            "mean_cost_minus_phase_pressure": policy - pp,
            "mean_actual_advantage": mean(row["reference_cost"] - row["policy_cost"] for row in subset)}

    harmful = [row for row in accepted if row["sign"] == "harmful"]
    raw_harm = -sum(row["raw_actual_gain"] for row in harmful)
    normalized_harm = -sum(row["normalized_true_gain"] for row in harmful)
    for key, prefix, total in (("raw_actual_gain", "raw", raw_harm),
                               ("normalized_true_gain", "normalized", normalized_harm)):
        ordered = sorted(harmful, key=lambda row: (row[key], row["group_id"]))
        rank = 0
        previous = None
        for index, row in enumerate(ordered, 1):
            if row[key] != previous:
                rank = index
            row[prefix + "_harm_rank"] = rank
            row[prefix + "_harm_share"] = -row[key] / total
            previous = row[key]
    accepted_by_seed = {str(seed): [row for row in accepted if row["seed"] == seed] for seed in seeds}
    largest = min(harmful, key=lambda row: (row["raw_actual_gain"], row["group_id"])) if harmful else None
    largest_fields = ("group_id", "seed", "raw_actual_gain", "normalized_true_gain", "training_scale",
                      "raw_harm_share", "normalized_harm_share", "raw_harm_rank", "normalized_harm_rank")
    return {"groups": len(effective_records), "accepted_count": len(accepted), "threshold": threshold,
            "threshold_comparison": "predicted_advantage > threshold + 1e-12",
            "outcome_counts": {sign: sum(row["sign"] == sign for row in accepted) for sign in ("beneficial", "harmful", "tie")},
            "units": {"score_and_normalized_gain": "full-eight-action group-normalized contrast",
                      "raw_gain": "native 450-second mean halted vehicles per controlled lane"},
            "normalization": {"scale_input": "all_eight_raw_costs_minus_reference_cost",
                "scale_function": "cf_h2o.traffic_signal.action_scaling.action_group_range",
                "normalized_training_target": "clip((cost_a-cost_reference)/scale,-5,5)",
                "normalized_true_gain": "negative normalized training target for selected action",
                "target_clip": TARGET_CLIP},
            "accepted_records": accepted,
            "normalized_prediction_diagnostics": {"overall": _normalized_diagnostics(accepted),
                "per_seed": {seed: _normalized_diagnostics(rows) for seed, rows in accepted_by_seed.items()}},
            "raw_policy": {"groups": len(effective_records), "weighting": "equal_seed", "per_seed": per_seed_policy,
                **{key: mean(row[key] for row in per_seed_policy.values()) for key in
                   ("mean_cost", "mean_phase_pressure_cost", "mean_cost_minus_phase_pressure", "mean_actual_advantage")}},
            "gain_sums": {"raw": _gain_sums(accepted, "raw_actual_gain"),
                          "normalized": _gain_sums(accepted, "normalized_true_gain")},
            "range_distribution": {name: {key: _distribution([row[key] for row in rows])
                                           for key in ("raw_cost_range", "training_scale")}
                                   for name, rows in (("all_groups", all_ranges), ("accepted_groups", accepted), ("harmful_groups", harmful))},
            "harm_attribution": {"count": len(harmful), "weighting": "equal_harmful_group_sum",
                "raw_harm_sum": raw_harm, "normalized_harm_sum": normalized_harm,
                "largest_raw_harm": {key: largest[key] for key in largest_fields} if largest else None},
            "interpretation": "Residuals compare predicted and true gains in the original eight-action normalized scale. Raw policy costs retain every OOF group in the denominator. Harm shares describe how existing selected actions contribute under two scales; they do not estimate the result of removing normalization or refitting."}
