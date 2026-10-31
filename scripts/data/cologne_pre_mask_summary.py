"""Compare the frozen A policy with masking allowed actions before selection."""

import math
from statistics import mean

if __package__:
    from .cologne_feature_block_comparison import SEED_COUNTS, TOLERANCE, compare_actions
else:
    from cologne_feature_block_comparison import SEED_COUNTS, TOLERANCE, compare_actions

MISSED = "missed_beneficial_switch"
OPTIMAL = "achieves_label_optimum"
METRICS = ("original_regret", "remaining_regret", "net_regret_recovery", "gross_benefit", "introduced_harm")
OUTCOMES = ("accepted_overrides", "beneficial", "harmful", "equal_cost")


def _recovery(rows, denominator):
    changes = [row for row in rows if row["premask_deployed_index"] != row["baseline_deployed_index"]]
    gains = [row["baseline_policy_cost"] - row["premask_policy_cost"] for row in rows]
    changed_gains = [row["reference_cost"] - row["premask_policy_cost"] for row in changes]
    sums = {"original_regret": sum(row["baseline_regret"] for row in rows),
            "remaining_regret": sum(row["premask_regret"] for row in rows),
            "net_regret_recovery": sum(gains), "gross_benefit": sum(max(gain, 0.) for gain in gains),
            "introduced_harm": sum(max(-gain, 0.) for gain in gains)}
    return {"groups": len(rows), "changed_actions": len(changes),
            "new_override_outcomes": {"accepted_overrides": len(changes),
                                      "beneficial": sum(gain > TOLERANCE for gain in changed_gains),
                                      "harmful": sum(gain < -TOLERANCE for gain in changed_gains),
                                      "equal_cost": sum(abs(gain) <= TOLERANCE for gain in changed_gains)},
            "raw_sums": sums, "contributions": {key: value / denominator for key, value in sums.items()}}


def _combine(parts):
    return {"groups": sum(part["groups"] for part in parts),
            "changed_actions": sum(part["changed_actions"] for part in parts),
            "new_override_outcomes": {key: sum(part["new_override_outcomes"][key] for part in parts) for key in OUTCOMES},
            "raw_sums": {key: sum(part["raw_sums"][key] for part in parts) for key in METRICS},
            "contributions": {key: mean(part["contributions"][key] for part in parts) for key in METRICS}}


def summarize_pre_mask(rows):
    """Keep100 groups in policy costs and use that same denominator for recovery."""
    matched = [{"group_id": row["group_id"], "seed": row["seed"], "reference_index": row["reference_index"],
                "label_costs": row["label_costs"],
                **{prefix + key: row[source + field] for prefix, source in (("old_", "baseline_"), ("new_", "premask_"))
                   for key, field in (("effective_index", "selected_index"), ("deployed_index", "deployed_index"),
                                      ("raw_predicted_advantage", "predicted_advantage"))}}
               for row in rows]
    report = compare_actions(matched)
    veto = [row for row in rows if row["original_model_stay_vetoed"]]
    other = [row for row in rows if not row["original_model_stay_vetoed"]]
    if (len(veto) != 32 or len(other) != 68
            or sum(row["baseline_category"] == MISSED for row in veto) != 23
            or sum(row["baseline_category"] == OPTIMAL for row in veto) != 9
            or any(row["baseline_deployed_index"] != row["reference_index"] for row in veto)
            or any(row["baseline_deployed_index"] != row["premask_deployed_index"] for row in other)):
        raise ValueError("Only the32 original stay-veto groups may change; retain their23 missed and9 PP-optimal groups")
    for row in rows:
        costs, reference = row["label_costs"], row["reference_index"]
        if (not math.isfinite(row["reference_cost"]) or not math.isfinite(row["oracle_cost"])
                or abs(row["reference_cost"] - costs[reference]) > TOLERANCE):
            raise ValueError("Preserve the native reference and allowed-action oracle costs")
        for arm in ("baseline", "premask"):
            cost, regret = row[arm + "_policy_cost"], row[arm + "_regret"]
            if (not math.isfinite(cost) or not math.isfinite(regret)
                    or abs(cost - costs[row[arm + "_deployed_index"]]) > TOLERANCE
                    or abs(regret - (cost - row["oracle_cost"])) > TOLERANCE or regret < -TOLERANCE):
                raise ValueError("Both policies must use matched native costs and the same allowed-action oracle")
        if row["original_model_stay_vetoed"] and ((row["baseline_category"] == MISSED) != (row["baseline_regret"] > TOLERANCE)):
            raise ValueError("The original veto subgroups must retain H's available-gain classification")
    per_seed = {}
    for seed, denominator in SEED_COUNTS.items():
        selected = [row for row in veto if row["seed"] == seed]
        per_seed[str(seed)] = {**_recovery(selected, denominator), "all_group_denominator": denominator,
                              "missed_veto": _recovery([row for row in selected if row["baseline_category"] == MISSED], denominator),
                              "pp_optimal_veto": _recovery([row for row in selected if row["baseline_category"] == OPTIMAL], denominator)}
    overall = {**_combine(list(per_seed.values())), "all_group_denominator": 100,
               **{subset: _combine([part[subset] for part in per_seed.values()]) for subset in ("missed_veto", "pp_optimal_veto")}}
    for block in [overall, *per_seed.values()]:
        for metrics in [block, block["missed_veto"], block["pp_optimal_veto"]]:
            values = metrics["contributions"]
            metrics["recovered_fraction_of_original_regret"] = (values["net_regret_recovery"] / values["original_regret"]
                                                                if values["original_regret"] > 0 else None)
            if (abs(values["original_regret"] - values["remaining_regret"] - values["net_regret_recovery"]) > TOLERANCE
                    or abs(values["gross_benefit"] - values["introduced_harm"] - values["net_regret_recovery"]) > TOLERANCE):
                raise ValueError("Regret recovery must equal gross benefit minus introduced harm")
    if abs(report["comparisons"]["overall"]["model_minus_previous_model_cost"]
           + overall["contributions"]["net_regret_recovery"]) > TOLERANCE:
        raise ValueError("The32 veto groups must account for the complete100-group policy cost change")
    return {**report, "veto_recovery": {"overall": overall, "per_seed": per_seed},
            "checks": {"all100_groups_retained": True, "original32_veto_subgroups_retained": True,
                       "other68_deployed_actions_unchanged": True, "regret_recovery_accounts_for_policy_change": True},
            "contribution_denominator": "Each veto subgroup sum is divided by all34/33/33 groups of its seed; overall contributions average those three values."}
