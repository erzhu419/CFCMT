"""Decompose the fixed100-group policy cost using equal training-seed weights."""

from collections import Counter
import math
from statistics import mean

SEED_COUNTS = {2027: 34, 3037: 33, 4047: 33}
CATEGORIES = ("achieves_label_optimum", "unnecessary_override_when_pp_optimal",
              "suboptimal_switch_when_benefit_available", "missed_beneficial_switch")
FIELDS = ("reference_cost", "policy_cost", "oracle_cost", "pp_to_oracle_gain", "realized_gain", "regret")
TOLERANCE = 1e-12


def _sums(rows):
    return {field: sum(row[field] for row in rows) for field in FIELDS}


def _outcomes(rows):
    gains = [row["realized_gain"] for row in rows if row["deployed_index"] != row["reference_index"]]
    return {"accepted_overrides": len(gains), "beneficial": sum(gain > TOLERANCE for gain in gains),
            "harmful": sum(gain < -TOLERANCE for gain in gains),
            "equal_cost": sum(abs(gain) <= TOLERANCE for gain in gains)}


def _seed_summary(rows):
    sums = _sums(rows)
    means = {field: value / len(rows) for field, value in sums.items()}
    categories = {}
    for category in CATEGORIES:
        subset = [row for row in rows if row["category"] == category]
        totals = _sums(subset)
        categories[category] = {"groups": len(subset), "raw_sums": totals,
                                "contributions": {field: value / len(rows) for field, value in totals.items()}}
    return {"groups": len(rows), "means": means, "raw_sums": sums,
            "policy_outcomes": _outcomes(rows), "categories": categories}


def _add_fractions(summary):
    potential, regret = summary["means"]["pp_to_oracle_gain"], summary["means"]["regret"]
    summary["realized_fraction_of_available_gain"] = summary["means"]["realized_gain"] / potential if potential > 0 else None
    for category in summary["categories"].values():
        category["regret_share"] = category["contributions"]["regret"] / regret if regret > 0 else None


def summarize_loss(rows):
    """Category contributions use all groups in each seed, including optimal ones."""
    if (len(rows) != 100 or len({row["group_id"] for row in rows}) != 100
            or Counter(row["seed"] for row in rows) != SEED_COUNTS):
        raise ValueError("Require all100 original groups with34/33/33 seed counts")
    for row in rows:
        if not all(math.isfinite(row[field]) for field in FIELDS):
            raise ValueError("The policy loss decomposition needs finite native label costs and gains")
        reference, policy, oracle = (row[field] for field in FIELDS[:3])
        computed = (reference - oracle, reference - policy, policy - oracle)
        if (any(abs(row[field] - value) > TOLERANCE for field, value in zip(FIELDS[3:], computed))
                or abs(row["pp_to_oracle_gain"] - row["realized_gain"] - row["regret"]) > TOLERANCE
                or oracle > min(reference, policy) + TOLERANCE
                or (row["deployed_index"] == row["reference_index"] and abs(reference - policy) > TOLERANCE)):
            raise ValueError("Native costs must satisfy potential = realized gain + regret and the PP action identity")
        expected_category = (CATEGORIES[0] if row["regret"] <= TOLERANCE else
                             CATEGORIES[1] if row["pp_to_oracle_gain"] <= TOLERANCE else
                             CATEGORIES[3] if row["deployed_index"] == row["reference_index"] else CATEGORIES[2])
        if row["category"] != expected_category:
            raise ValueError("The four loss categories must match optimum, available gain and deployed action")
    per_seed = {str(seed): _seed_summary([row for row in rows if row["seed"] == seed]) for seed in SEED_COUNTS}
    overall = {"groups": len(rows), "weighting": "equal_seed",
               "means": {field: mean(values["means"][field] for values in per_seed.values()) for field in FIELDS},
               "raw_sums": _sums(rows), "policy_outcomes": _outcomes(rows),
               "categories": {category: {
                   "groups": sum(values["categories"][category]["groups"] for values in per_seed.values()),
                   "raw_sums": {field: sum(values["categories"][category]["raw_sums"][field] for values in per_seed.values()) for field in FIELDS},
                   "contributions": {field: mean(values["categories"][category]["contributions"][field] for values in per_seed.values()) for field in FIELDS}}
                   for category in CATEGORIES}}
    for summary in [overall, *per_seed.values()]:
        _add_fractions(summary)
        if (sum(category["groups"] for category in summary["categories"].values()) != summary["groups"]
                or abs(summary["means"]["pp_to_oracle_gain"] - summary["means"]["realized_gain"] - summary["means"]["regret"]) > TOLERANCE
                or abs(sum(summary["categories"][category]["contributions"]["regret"] for category in CATEGORIES[1:])
                       - summary["means"]["regret"]) > TOLERANCE):
            raise ValueError("All groups and all regret must remain in the exhaustive loss decomposition")
    return {**overall, "per_seed": per_seed,
            "checks": {"all100_groups_retained": len(rows) == 100, "category_counts_exhaustive": True,
                       "potential_equals_realized_plus_regret": True, "three_loss_categories_account_for_regret": True},
            "contribution_denominator": "Each category sum is divided by all34/33/33 groups of its seed; overall contributions average the three seed contributions.",
            "cost_units": "native450-second mean halted vehicles per controlled lane"}
