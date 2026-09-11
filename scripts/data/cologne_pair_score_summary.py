"""Separate PP-benefit sign errors from ordering errors between allowed switches."""

from collections import Counter
import math
from statistics import mean

SEED_COUNTS = {2027: 34, 3037: 33, 4047: 33}
TOLERANCE = 1e-12
CATEGORIES = {
    "switch_vs_pp": ("strict_sign_agreement", "false_positive_benefit", "strict_missed_benefit",
                     "prediction_tie", "label_tie", "both_tie"),
    "switch_pairs": ("strict_order_agreement", "strict_order_reversal", "prediction_tie", "label_tie", "both_tie")}
REVERSALS = {"switch_vs_pp": ("false_positive_benefit", "strict_missed_benefit"),
             "switch_pairs": ("strict_order_reversal",)}


def _seed_metrics(rows, family):
    blocks = [row[family] for row in rows]
    categories = CATEGORIES[family]
    return {"groups": len(rows), "groups_without_comparisons": sum(block["eligible_comparisons"] == 0 for block in blocks),
            "fraction_of_groups_with_comparisons": mean(block["eligible_comparisons"] > 0 for block in blocks),
            "eligible_comparisons": sum(block["eligible_comparisons"] for block in blocks),
            "counts": {key: sum(block["counts"][key] for block in blocks) for key in categories},
            "absolute_label_gap_sums": {key: sum(block["absolute_label_gap_sums"][key] for block in blocks) for key in categories},
            "total_absolute_label_gap": sum(block["total_absolute_label_gap"] for block in blocks),
            "fractions": {key: mean(block["counts"][key] / block["eligible_comparisons"]
                                     if block["eligible_comparisons"] else 0. for block in blocks) for key in categories},
            "absolute_label_gap_contributions": {
                key: mean(block["absolute_label_gap_sums"][key] / block["eligible_comparisons"]
                          if block["eligible_comparisons"] else 0. for block in blocks) for key in categories},
            "mean_absolute_label_gap": mean(block["total_absolute_label_gap"] / block["eligible_comparisons"]
                                            if block["eligible_comparisons"] else 0. for block in blocks)}


def _overall(per_seed, family):
    parts = list(per_seed.values())
    return {**{key: sum(part[key] for part in parts) for key in
               ("groups", "groups_without_comparisons", "eligible_comparisons", "total_absolute_label_gap")},
            **{key: mean(part[key] for part in parts) for key in
               ("fraction_of_groups_with_comparisons", "mean_absolute_label_gap")},
            **{field: {key: sum(part[field][key] for part in parts) for key in CATEGORIES[family]}
               for field in ("counts", "absolute_label_gap_sums")},
            **{field: {key: mean(part[field][key] for part in parts) for key in CATEGORIES[family]}
               for field in ("fractions", "absolute_label_gap_contributions")}}


def _add_ratios(metrics, family):
    gap = sum(metrics["absolute_label_gap_contributions"][key] for key in REVERSALS[family])
    total = metrics["mean_absolute_label_gap"]
    metrics.update(strict_reversal_fraction=sum(metrics["fractions"][key] for key in REVERSALS[family]),
                   reversed_label_gap_contribution=gap,
                   reversed_gap_ratio=gap / total if total > 0 else None,
                   prediction_tie_label_gap_contribution=metrics["absolute_label_gap_contributions"]["prediction_tie"])


def summarize_pair_scores(rows):
    """Average each group's all-comparison fractions before averaging seeds."""
    if (len(rows) != 100 or len({row["group_id"] for row in rows}) != 100
            or Counter(row["seed"] for row in rows) != SEED_COUNTS):
        raise ValueError("Require all100 original groups with34/33/33 seed counts")
    for row in rows:
        indices, count = row["eligible_switch_indices"], row["eligible_switch_count"]
        if (count != len(indices) or len(set(indices)) != count or row["reference_index"] in indices
                or row["reference_index"] not in range(8) or any(index not in range(8) for index in indices)
                or row["candidate_count"] != 8 or row["allowed_candidate_count"] != count + 1):
            raise ValueError("Eligible switches must preserve the allowed non-reference candidate set")
        for family, categories in CATEGORIES.items():
            block = row[family]
            expected_count = count if family == "switch_vs_pp" else count * (count - 1) // 2
            counts, gaps = block["counts"], block["absolute_label_gap_sums"]
            if (block["eligible_comparisons"] != expected_count or set(counts) != set(categories) or set(gaps) != set(categories)
                    or any(not isinstance(value, int) or value < 0 for value in counts.values())
                    or sum(counts.values()) != expected_count
                    or any(not math.isfinite(value) or value < 0 for value in gaps.values())
                    or not math.isfinite(block["total_absolute_label_gap"])
                    or abs(sum(gaps.values()) - block["total_absolute_label_gap"]) > TOLERANCE
                    or any(counts[key] == 0 and gaps[key] != 0 for key in categories)):
                raise ValueError("Each family must retain every eligible comparison, tie and absolute label gap")
    result = {}
    for family in CATEGORIES:
        per_seed = {str(seed): _seed_metrics([row for row in rows if row["seed"] == seed], family) for seed in SEED_COUNTS}
        overall = _overall(per_seed, family)
        for metrics in [overall, *per_seed.values()]:
            _add_ratios(metrics, family)
        result[family] = {"overall": overall, "per_seed": per_seed}
    return {"groups": 100, "weighting": "all_comparisons_within_group_then_equal_group_within_seed_then_equal_seed",
            "candidate_count": sum(row["candidate_count"] for row in rows),
            "eligible_switch_count": sum(row["eligible_switch_count"] for row in rows), **result,
            "checks": {"all100_groups_retained": True, "all_eligible_comparisons_retained": True,
                       "ties_retained_in_comparison_denominators": True, "absolute_label_gap_totals_exact": True},
            "weighting_definition": "Category counts and absolute label gap sums are divided by all eligible comparisons of each group, including ties. "
                                    "Groups with no comparisons contribute zero and remain in their original34/33/33 group denominator. "
                                    "Overall values average the three seed means; reversed_gap_ratio divides weighted gap contributions.",
            "interpretation": "The two families describe separate cached-score errors; their gap contributions are not deployment regret."}
