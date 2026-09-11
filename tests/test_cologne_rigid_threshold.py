import pytest

from scripts.data.audit_cologne_local_rigid_ranking import group_record
from scripts.data.calibrate_cologne_rigid_threshold import select_threshold, threshold_metrics


def record(seed, interval, predicted, pp_cost, rigid_cost):
    return group_record(
        f"cologne1:seed{seed}:{interval}:cluster_357187_359543",
        ["PP", "rigid"], [pp_cost, rigid_cost], [0, -predicted], 0, 1,
    )


def test_none_means_pp_only_and_numeric_threshold_requires_strict_excess():
    rows = [record(2027, i, advantage, 10, 0)
            for i, advantage in enumerate((1.0, 1.0 + 0.5e-12, 1.0 + 2e-12))]
    pp = threshold_metrics(rows, None)
    numeric = threshold_metrics(rows, 1.0)
    assert pp["mode"] == "phase_pressure_only"
    assert pp["accepted_overrides"] == 0
    assert pp["mean_cost"] == 10
    assert numeric["accepted_overrides"] == numeric["beneficial"] == 1
    assert numeric["mean_cost"] == pytest.approx(20 / 3)
    assert numeric["mean_cost_minus_phase_pressure"] == pytest.approx(-10 / 3)


def test_unequal_group_counts_use_equal_seed_cost_not_pooled_group_cost():
    rows = [record(2027, i, 1, 10, 0) for i in range(3)]
    rows += [record(seed, 6, 1, 0, 12) for seed in (3037, 4047)]
    result = select_threshold(rows)
    ungated = result["ungated_rigid"]
    assert ungated["groups"] == ungated["accepted_overrides"] == 5
    assert ungated["weighting"] == "equal_seed"
    assert ungated["per_seed"]["2027"]["groups"] == 3
    assert ungated["mean_cost"] == 8
    assert ungated["mean_phase_pressure_cost"] == pytest.approx(10 / 3)
    assert ungated["mean_cost_minus_phase_pressure"] == pytest.approx(14 / 3)
    # Pooled groups would prefer rigid (24/5 < 30/5); the prespecified seed mean does not.
    assert result["selected"]["threshold"] is None


def test_threshold_uses_actual_raw_cost_even_when_high_confidence_action_is_harmful():
    rows = [record(2027, 6, 1, 10, 0), record(2027, 11, 4, 10, 11)]
    result = select_threshold(rows)
    assert result["selected"]["threshold"] == 0
    assert result["selected"]["mean_cost"] == 5.5
    assert result["selected"]["beneficial"] == result["selected"]["harmful"] == 1
    assert threshold_metrics(rows, 1)["mean_cost"] == 10.5


def test_duplicate_positive_advantages_cannot_be_split_to_keep_only_good_group():
    rows = [record(2027, 6, 2, 10, 2), record(2027, 11, 2, 10, 20)]
    result = select_threshold(rows)
    assert result["candidate_count"] == 3
    assert [row["threshold"] for row in result["curve"]] == [None, 0.0, 2.0]
    assert [row["accepted_overrides"] for row in result["curve"]] == [0, 2, 0]
    assert result["selected"]["threshold"] is None


def test_equal_cost_tie_prefers_explicit_pp_only():
    result = select_threshold([record(2027, 6, 1, 2, 2)])
    assert result["ungated_rigid"]["equal_cost"] == 1
    assert result["selected"]["threshold"] is None
    assert result["selected"]["accepted_overrides"] == 0


def test_equal_cost_numeric_threshold_tie_prefers_larger_threshold():
    rows = [record(2027, 6, 1, 5, 5), record(2027, 11, 2, 10, 0)]
    result = select_threshold(rows)
    assert result["selected"]["threshold"] == 1
    assert result["selected"]["accepted_overrides"] == 1
    assert result["selected"]["mean_cost"] == result["ungated_rigid"]["mean_cost"] == 2.5
    assert result["phase_pressure"]["mean_cost"] == 7.5
