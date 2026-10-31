"""Pair completeness, fold-specific scaling and label-sign groups for V155F."""

import pytest

from scripts.data import summarize_cologne_omitted_feature_audit as summary


def fixture():
    queries, folds = [], []
    seed_outcomes = {2027: (5, 11, 1), 3037: (13, 3, 1), 4047: (13, 1, 0)}
    for index, (seed, counts) in enumerate(seed_outcomes.items()):
        folds.append({"heldout_seed": seed, "feature_std": [0.] + [2. ** (index + 1)] * 12,
                      "feature_min": [2.] + [0.] * 12, "feature_max": [2.] + [20.] * 12,
                      "constant_features": [summary.FEATURE_NAMES[0]]})
        neighbor_seed = list(seed_outcomes)[(index + 1) % 3]
        for outcome, count, gain in zip(("beneficial", "harmful", "tie"), counts, (.1, -.1, 0.)):
            for number in range(count):
                queries.append({"group_id": f"seed{seed}:{outcome}:{number}", "seed": seed,
                                "actual_gain": gain, "outcome": outcome, "feature_values": [4.] + [10.] * 12,
                                "neighbors": [{"group_id": f"seed{neighbor_seed}:neighbor{i}", "seed": neighbor_seed,
                                               "raw_gain": raw_gain, "feature_values": [2.] + [4.] * 12}
                                              for i, raw_gain in enumerate((.2, -.1, 0.))]})
    return {"feature_names": list(summary.FEATURE_NAMES), "query_count": 48, "pair_count": 144,
            "queries": queries, "folds": folds}


def test_all_pairs_and_original_query_groups_are_retained():
    report = summary.summarize_audit(fixture())
    assert report["feature_names"] == [
        "current_green_elapsed_norm", "green_q_max", "reference_green_q_max", "delta_green_q_max",
        "red_q_max", "reference_red_q_max", "delta_red_q_max", "green_q_cv", "reference_green_q_cv",
        "delta_green_q_cv", "red_q_cv", "reference_red_q_cv", "delta_red_q_cv"]
    assert report["query_count"] == 48 and report["pair_count"] == 144
    assert report["query_outcome_counts"] == {"beneficial": 31, "harmful": 15, "tie": 2}
    assert report["query_seed_counts"] == {"2027": 17, "3037": 17, "4047": 14}
    assert report["overall"]["gain_sign_relation_counts"] == {"agree": 46, "disagree": 46, "tie_involved": 52}
    assert [report["per_seed"][str(seed)]["pair_count"] for seed in summary.SEED_COUNTS] == [51, 51, 42]
    assert [report["per_query_outcome"][outcome]["pair_count"] for outcome in summary.OUTCOME_COUNTS] == [93, 45, 6]
    assert report["per_query_outcome"]["tie"]["gain_sign_relation_counts"] == {"agree": 0, "disagree": 0, "tie_involved": 6}
    assert all(report["checks"].values())


def test_raw_differences_use_each_query_fold_for_standardization():
    report = summary.summarize_audit(fixture())
    feature = summary.FEATURE_NAMES[1]
    assert report["overall"]["features"][feature]["absolute_raw_difference"] == {
        "count": 144, "minimum": 6., "median": 6., "maximum": 6.}
    for seed, gap, count in ((2027, 3., 51), (3037, 1.5, 51), (4047, .75, 42)):
        stats = report["per_seed"][str(seed)]["features"][feature]
        assert stats["absolute_standardized_difference"] == {
            "count": count, "minimum": gap, "median": gap, "maximum": gap}
    stats = report["overall"]["by_gain_sign_relation"]["disagree"]["features"][feature]
    assert stats["absolute_standardized_difference"]["count"] == 46
    assert stats["absolute_standardized_difference"]["median"] == 1.5


def test_constant_features_keep_raw_gaps_but_have_no_standardized_values():
    audit = fixture()
    audit["queries"][0]["feature_values"][0] = 2.  # Three genuine equal-value pairs remain included.
    report = summary.summarize_audit(audit)
    stats = report["overall"]["features"][summary.FEATURE_NAMES[0]]
    assert stats["constant_pair_count"] == 144 and stats["constant_mismatch_count"] == 141
    assert stats["absolute_raw_difference"] == {"count": 144, "minimum": 0., "median": 2., "maximum": 2.}
    assert stats["absolute_standardized_difference"] == {"count": 0, "minimum": None, "median": None, "maximum": None}


def test_sign_relation_uses_raw_gain_and_preserves_tolerance_ties():
    assert summary.gain_sign_relation(-.2, -.1) == "agree"
    assert summary.gain_sign_relation(.2, -.1) == "disagree"
    assert summary.gain_sign_relation(1e-12, -.1) == "tie_involved"
    assert summary.gain_sign_relation(.2, -1e-12) == "tie_involved"
    assert summary.gain_sign_relation(.2, -1.01e-12) == "disagree"


@pytest.mark.parametrize("change", ["missing_pair", "wrong_outcome", "wrong_feature_order", "zero_nonconstant_scale"])
def test_incomplete_pairs_or_wrong_units_cannot_produce_summary(change):
    audit = fixture()
    if change == "missing_pair":
        audit["queries"][0]["neighbors"].pop()
    elif change == "wrong_outcome":
        audit["queries"][0]["actual_gain"] = -.1
    elif change == "wrong_feature_order":
        audit["feature_names"].reverse()
    else:
        audit["folds"][0]["feature_std"][1] = 0.
    with pytest.raises(ValueError):
        summary.summarize_audit(audit)
