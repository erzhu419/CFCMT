"""Pair/category weighting keeps ties and original group denominators."""

import pytest

from scripts.data.cologne_pair_score_summary import CATEGORIES, summarize_pair_scores


def block(family, entries):
    counts, gaps = dict.fromkeys(CATEGORIES[family], 0), dict.fromkeys(CATEGORIES[family], 0.)
    for category, count, gap in entries:
        counts[category], gaps[category] = count, gap
    return {"eligible_comparisons": sum(counts.values()), "counts": counts,
            "absolute_label_gap_sums": gaps, "total_absolute_label_gap": sum(gaps.values())}


def fixture():
    rows = []
    for seed, count in ((2027, 34), (3037, 33), (4047, 33)):
        for index in range(count):
            rows.append({"group_id": f"cologne1:seed{seed}:{index}", "seed": seed, "reference_index": 0,
                         "eligible_switch_indices": [1], "eligible_switch_count": 1,
                         "candidate_count": 8, "allowed_candidate_count": 2,
                         "switch_vs_pp": block("switch_vs_pp", [("strict_sign_agreement", 1, 1.)]),
                         "switch_pairs": block("switch_pairs", [])})
    return rows


def test_group_then_seed_weighting_and_ratio_use_weighted_gap_contributions():
    rows = fixture()
    for row in rows:
        if row["seed"] == 2027:
            row.update(eligible_switch_indices=[1, 2, 3], eligible_switch_count=3, allowed_candidate_count=4,
                       switch_vs_pp=block("switch_vs_pp", [("strict_sign_agreement", 1, 3.), ("false_positive_benefit", 1, 6.), ("prediction_tie", 1, 9.)]),
                       switch_pairs=block("switch_pairs", [("strict_order_agreement", 1, 3.), ("strict_order_reversal", 1, 6.), ("prediction_tie", 1, 9.)]))
        elif row["seed"] == 3037:
            row.update(eligible_switch_indices=[1, 2], eligible_switch_count=2, allowed_candidate_count=3,
                       switch_vs_pp=block("switch_vs_pp", [("strict_sign_agreement", 1, 1.), ("strict_missed_benefit", 1, 1.)]),
                       switch_pairs=block("switch_pairs", [("strict_order_reversal", 1, 1.)]))
    report = summarize_pair_scores(rows)
    pp, switches = report["switch_vs_pp"]["overall"], report["switch_pairs"]["overall"]
    assert pp["counts"]["false_positive_benefit"] == 34 and pp["counts"]["strict_missed_benefit"] == 33
    assert pp["fractions"]["false_positive_benefit"] == pytest.approx(1. / 9)
    assert pp["fractions"]["strict_missed_benefit"] == pytest.approx(1. / 6)
    assert pp["reversed_label_gap_contribution"] == pytest.approx(2.5 / 3)
    assert pp["mean_absolute_label_gap"] == pytest.approx(8. / 3)
    assert pp["reversed_gap_ratio"] == pytest.approx(2.5 / 8)
    assert pp["reversed_gap_ratio"] != pytest.approx(237. / 711)  # Pooled raw gaps weight large groups differently.
    assert switches["reversed_gap_ratio"] == pytest.approx(3. / 7)
    assert switches["reversed_gap_ratio"] != pytest.approx(237. / 645)
    assert switches["fraction_of_groups_with_comparisons"] == pytest.approx(2. / 3)
    assert sum(switches["fractions"].values()) == pytest.approx(2. / 3)
    assert report["switch_pairs"]["per_seed"]["4047"]["groups_without_comparisons"] == 33


def test_ties_retain_all_comparison_denominators_and_tiny_label_gaps():
    rows = fixture()
    for row in rows:
        row.update(eligible_switch_indices=[1, 2, 3], eligible_switch_count=3, allowed_candidate_count=4)
        for family in CATEGORIES:
            row[family] = block(family, [("prediction_tie", 1, 2.), ("label_tie", 1, 5e-13), ("both_tie", 1, 0.)])
    report = summarize_pair_scores(rows)
    for family in CATEGORIES:
        metrics = report[family]["overall"]
        assert metrics["eligible_comparisons"] == 300
        assert metrics["fractions"]["prediction_tie"] == pytest.approx(1. / 3)
        assert metrics["fractions"]["label_tie"] == pytest.approx(1. / 3)
        assert metrics["fractions"]["both_tie"] == pytest.approx(1. / 3)
        assert metrics["absolute_label_gap_contributions"]["label_tie"] > 0
        assert metrics["prediction_tie_label_gap_contribution"] == pytest.approx(2. / 3)
        assert metrics["reversed_gap_ratio"] == 0.
    assert all(report["checks"].values())


def test_zero_comparison_groups_remain_in_all100_denominator():
    rows = fixture()
    for row in rows:
        row.update(eligible_switch_indices=[], eligible_switch_count=0, allowed_candidate_count=1)
        for family in CATEGORIES:
            row[family] = block(family, [])
    report = summarize_pair_scores(rows)
    for family in CATEGORIES:
        metrics = report[family]["overall"]
        assert metrics["groups"] == metrics["groups_without_comparisons"] == 100
        assert metrics["eligible_comparisons"] == 0 and sum(metrics["fractions"].values()) == 0.
        assert metrics["mean_absolute_label_gap"] == 0. and metrics["reversed_gap_ratio"] is None


@pytest.mark.parametrize("change", ["missing_group", "lost_tie", "wrong_pair_count", "wrong_gap_sum"])
def test_missing_groups_comparisons_or_label_gaps_are_rejected(change):
    rows = fixture()
    if change == "missing_group":
        rows.pop()
    elif change == "lost_tie":
        rows[0]["switch_vs_pp"]["counts"]["prediction_tie"] = 1
    elif change == "wrong_pair_count":
        rows[0]["switch_pairs"]["eligible_comparisons"] = 1
    else:
        rows[0]["switch_vs_pp"]["total_absolute_label_gap"] += .1
    with pytest.raises(ValueError):
        summarize_pair_scores(rows)
