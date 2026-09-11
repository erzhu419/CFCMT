"""Matched eight-action fixtures distinguish normalized errors from raw policy gain."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.data.audit_cologne_local_rigid_ranking import group_record
from scripts.data.cologne_oof_normalization_audit import audit_records


def record(index, seed, costs, score=0.6, selected=1):
    scores = [0, -score, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    return group_record(f"cologne1:seed{seed}:{index}:tls", [f"phase{i}" for i in range(8)], costs, scores, 0, selected)


def test_full_eight_action_scale_and_correctly_normalized_residual():
    # Two-arm range would be 2; full action range is 10, so the true gain is .2.
    row = record(0, 2027, [4, 2, 0, 10, 5, 6, 7, 8])
    result = audit_records([row], 0.2)
    accepted = result["accepted_records"][0]
    assert accepted["raw_cost_range"] == accepted["training_scale"] == 10
    assert accepted["raw_actual_gain"] == 2
    assert accepted["normalized_true_gain"] == pytest.approx(0.2)
    assert accepted["normalized_residual"] == pytest.approx(0.4)
    stats = result["normalized_prediction_diagnostics"]["overall"]
    assert stats["bias"] == stats["mean_absolute_error"] == pytest.approx(0.4)
    assert stats["score_vs_normalized_gain_pearson"] is None


def test_scale_floor_uses_contrast_not_high_absolute_cost_level():
    costs = [10000, 10000 + 1e-10, 10000, 10000, 10000, 10000, 10000, 10000]
    result = audit_records([record(0, 2027, costs)], 0)
    accepted = result["accepted_records"][0]
    assert accepted["training_scale"] == 1e-8
    assert accepted["raw_cost_range"] < accepted["training_scale"]
    assert accepted["normalized_true_gain"] == pytest.approx((costs[0] - costs[1]) / 1e-8)


def test_cost_rescaling_preserves_normalized_error_but_changes_raw_gain():
    costs = np.asarray([4, 2, 0, 10, 5, 6, 7, 8])
    original = audit_records([record(0, 2027, costs)], 0)["accepted_records"][0]
    scaled = audit_records([record(0, 2027, costs * 100)], 0)["accepted_records"][0]
    assert scaled["raw_actual_gain"] == 100 * original["raw_actual_gain"]
    assert scaled["training_scale"] == 100 * original["training_scale"]
    assert scaled["normalized_true_gain"] == original["normalized_true_gain"]
    assert scaled["normalized_residual"] == original["normalized_residual"]
    assert scaled["normalized_residual"] != scaled["normalized_predicted_gain"] - scaled["raw_actual_gain"]


def test_flat_eight_action_group_is_retained_as_tie():
    result = audit_records([record(0, 2027, [5] * 8)], 0)
    row = result["accepted_records"][0]
    assert row["raw_cost_range"] == 0 and row["training_scale"] == 1e-8
    assert row["sign"] == "tie" and row["normalized_true_gain"] == 0
    assert result["outcome_counts"] == {"beneficial": 0, "harmful": 0, "tie": 1}


def test_raw_policy_retains_rejected_and_post_stay_rows_in_seed_denominators():
    rows = [record(0, 2027, [4, 2, 0, 10, 5, 6, 7, 8]),
            record(1, 2027, [4, 2, 0, 10, 5, 6, 7, 8], score=0.2),
            record(2, 2027, [4, 2, 0, 10, 5, 6, 7, 8], selected=0),
            record(3, 3037, [2, 3, 0, 10, 5, 6, 7, 8])]
    original = deepcopy(rows)
    result = audit_records(rows, 0.2)
    policy = result["raw_policy"]
    assert result["groups"] == 4 and result["accepted_count"] == 2
    assert policy["per_seed"]["2027"]["groups"] == 3
    assert policy["per_seed"]["2027"]["mean_cost"] == pytest.approx(10 / 3)
    assert policy["mean_cost"] == pytest.approx((10 / 3 + 3) / 2)
    assert policy["mean_phase_pressure_cost"] == 3
    assert policy["mean_actual_advantage"] == pytest.approx((2 / 3 - 1) / 2)
    assert rows == original


def test_raw_harm_concentration_can_reverse_normalized_harm_ranks():
    rows = [record(0, 2027, [2, 3, 0, 100, 4, 5, 6, 7]),
            record(1, 3037, [0, 0.5, 0, 1, 0.1, 0.2, 0.3, 0.4])]
    result = audit_records(rows, 0)
    first, second = result["accepted_records"]
    assert [first["raw_harm_rank"], second["raw_harm_rank"]] == [1, 2]
    assert [first["normalized_harm_rank"], second["normalized_harm_rank"]] == [2, 1]
    assert first["raw_harm_share"] == pytest.approx(2 / 3)
    assert first["normalized_harm_share"] == pytest.approx(0.01 / 0.51)
    assert result["gain_sums"]["raw"]["negative_gain_sum"] == -1.5
    assert result["gain_sums"]["normalized"]["negative_gain_sum"] == pytest.approx(-0.51)
    assert result["harm_attribution"]["largest_raw_harm"]["group_id"] == first["group_id"]


def test_positive_nonconstant_normalized_association_and_clip_contract():
    rows = [record(0, 2027, [1, 0, 2, 3, 4, 5, 6, 10], score=0.2),
            record(1, 2027, [2, 0, 1, 3, 4, 5, 6, 10], score=0.4)]
    result = audit_records(rows, 0)
    assert result["normalized_prediction_diagnostics"]["overall"]["score_vs_normalized_gain_pearson"] == pytest.approx(1)
    assert result["normalization"]["target_clip"] == 5
    assert all(abs(row["normalized_true_gain"]) <= 1 for row in result["accepted_records"])


@pytest.mark.parametrize("change", ["two_arms", "cached_advantage", "duplicate"])
def test_incomplete_or_stale_inputs_rejected(change):
    rows = [record(0, 2027, np.arange(8))]
    if change == "two_arms":
        rows[0]["label_costs"] = rows[0]["label_costs"][:2]
    elif change == "cached_advantage":
        rows[0]["predicted_advantage"] += 1
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        audit_records(rows, 0)
