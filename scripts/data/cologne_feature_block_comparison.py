"""Compare the fixed expanded-feature and original raw-cost OOF policies."""

from collections import Counter
import math
from statistics import mean

SEED_COUNTS = {2027: 34, 3037: 33, 4047: 33}
TOLERANCE = 1e-12
COUNT_KEYS = ("accepted_overrides", "beneficial", "harmful", "equal_cost")
COST_KEYS = ("mean_cost", "mean_phase_pressure_cost", "mean_cost_minus_phase_pressure")


def _policy(rows, arm):
    per_seed = {}
    for seed in SEED_COUNTS:
        selected = [row for row in rows if row["seed"] == seed]
        gains = [row["reference_cost"] - row[arm + "_cost"] for row in selected if arm and row[arm + "_accepted"]]
        costs = [row[arm + "_cost"] if arm else row["reference_cost"] for row in selected]
        per_seed[str(seed)] = {
            "groups": len(selected), "accepted_overrides": len(gains),
            "beneficial": sum(gain > TOLERANCE for gain in gains),
            "harmful": sum(gain < -TOLERANCE for gain in gains),
            "equal_cost": sum(abs(gain) <= TOLERANCE for gain in gains),
            "mean_cost": mean(costs), "mean_phase_pressure_cost": mean(row["reference_cost"] for row in selected),
            "mean_cost_minus_phase_pressure": mean(cost - row["reference_cost"] for cost, row in zip(costs, selected))}
    return {"groups": len(rows), "weighting": "equal_seed", "threshold": 0. if arm else None,
            "mode": "minimum_advantage" if arm else "phase_pressure_only", "per_seed": per_seed,
            **{key: sum(values[key] for values in per_seed.values()) for key in COUNT_KEYS},
            **{key: mean(values[key] for values in per_seed.values()) for key in COST_KEYS}}


def _changes(rows):
    return {"groups": len(rows),
            "deployed_action_changed": sum(row["old_deployed_index"] != row["new_deployed_index"] for row in rows),
            "model_lower_cost": sum(row["new_cost"] < row["old_cost"] - TOLERANCE for row in rows),
            "model_higher_cost": sum(row["new_cost"] > row["old_cost"] + TOLERANCE for row in rows),
            "same_cost": sum(abs(row["new_cost"] - row["old_cost"]) <= TOLERANCE for row in rows),
            "changed_action_same_cost": sum(row["old_deployed_index"] != row["new_deployed_index"]
                                            and abs(row["new_cost"] - row["old_cost"]) <= TOLERANCE for row in rows)}


def _cost_comparison(model, previous, pp):
    return {"model_minus_previous_model_cost": model - previous,
            "model_vs_previous_model_percent": 100 * (model / previous - 1) if previous else None,
            "model_minus_phase_pressure_cost": model - pp,
            "model_vs_phase_pressure_percent": 100 * (model / pp - 1) if pp else None}


def compare_actions(matched_actions):
    """Derive deployment from the fixed zero threshold and retain all100 groups."""
    if (len(matched_actions) != 100 or len({row["group_id"] for row in matched_actions}) != 100
            or Counter(row["seed"] for row in matched_actions) != SEED_COUNTS):
        raise ValueError("Require all100 matched groups with the original34/33/33 seed counts")
    rows = []
    for original in matched_actions:
        row = dict(original)
        costs, reference = row["label_costs"], row["reference_index"]
        if len(costs) != 8 or not all(math.isfinite(cost) for cost in costs) or reference not in range(8):
            raise ValueError("Each group must preserve all eight native costs and its PP reference")
        row["reference_cost"] = costs[reference]
        for arm in ("old", "new"):
            effective, deployed = row[arm + "_effective_index"], row[arm + "_deployed_index"]
            predicted = row[arm + "_raw_predicted_advantage"]
            if effective not in range(8) or deployed not in range(8) or not math.isfinite(predicted):
                raise ValueError("Both models require valid effective/deployed actions and raw scores")
            accepted = effective != reference and predicted > TOLERANCE
            if deployed != (effective if accepted else reference):
                raise ValueError("Deployed actions must follow the fixed zero threshold after the S action veto")
            row[arm + "_cost"], row[arm + "_accepted"] = costs[deployed], accepted
        rows.append(row)
    model, previous, pp = _policy(rows, "new"), _policy(rows, "old"), _policy(rows, None)
    return {"groups": 100, "weighting": "equal_seed", "cost_units": "native450-second mean halted vehicles per controlled lane",
            "model": model, "previous_model": previous, "phase_pressure": pp,
            "action_changes": {"overall": _changes(rows), "per_seed": {
                str(seed): _changes([row for row in rows if row["seed"] == seed]) for seed in SEED_COUNTS}},
            "comparisons": {"overall": _cost_comparison(model["mean_cost"], previous["mean_cost"], pp["mean_cost"]),
                            "per_seed": {str(seed): _cost_comparison(model["per_seed"][str(seed)]["mean_cost"],
                                                                      previous["per_seed"][str(seed)]["mean_cost"],
                                                                      pp["per_seed"][str(seed)]["mean_cost"])
                                         for seed in SEED_COUNTS}}}
