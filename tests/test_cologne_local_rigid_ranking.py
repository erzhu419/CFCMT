import pytest

from scripts.data.audit_cologne_local_rigid_ranking import group_record, summarize


def record(costs, scores, *, selected=1, interval=6):
    return group_record(
        f"cologne1:seed2027:{interval}:cluster_357187_359543",
        [f"phase{i}" for i in range(len(costs))], costs, scores, reference=0, selected=selected,
    )


def test_positive_advantage_means_lower_cost_and_regret_uses_oracle():
    row = record([10, 7, 5], [0, -4, -1])
    assert row["seed"] == 2027
    assert row["interval_index"] == 6
    assert row["phase_pressure_cost"] == 10
    assert row["rigid_cost"] == 7
    assert row["oracle_cost"] == 5
    assert row["rigid_minus_phase_pressure"] == -3
    assert row["predicted_advantage"] == 4
    assert row["actual_advantage"] == 3
    assert row["rigid_regret"] == 2
    assert row["phase_pressure_regret"] == 5


def test_predicted_improvement_can_be_an_actual_harm():
    row = record([5, 10, 3], [0, -4, -2])
    result = summarize([row])
    assert row["predicted_advantage"] == 4
    assert row["actual_advantage"] == -5
    assert row["rigid_minus_phase_pressure"] == 5
    assert row["rigid_regret"] == 7
    assert row["phase_pressure_regret"] == 2
    assert result["predicted_positive_advantage_outcomes"] == {"beneficial": 0, "harmful": 1, "equal_cost": 0}
    assert result["positive_prediction_mean_actual_advantage"] == -5


def test_different_action_with_equal_cost_is_neither_harmful_nor_beneficial():
    row = record([4, 4, 9], [0, -0.1, 2])
    result = summarize([row])
    assert row["different_action"] is True
    assert row["actual_advantage"] == row["rigid_regret"] == row["phase_pressure_regret"] == 0
    assert result["different_action_groups"] == 1
    assert result["different_action_outcomes"] == {"beneficial": 0, "harmful": 0, "equal_cost": 1}
    assert result["rigid_optimal_groups"] == result["phase_pressure_optimal_groups"] == 1


def test_keeping_phase_pressure_can_still_have_oracle_regret():
    row = record([7, 3], [-2, 4], selected=0)
    result = summarize([row])
    assert row["different_action"] is False
    assert row["predicted_advantage"] == row["actual_advantage"] == 0
    assert row["rigid_regret"] == row["phase_pressure_regret"] == 4
    assert result["different_action_groups"] == result["predicted_positive_advantage_groups"] == 0
    assert result["positive_prediction_mean_actual_advantage"] is None


def test_summary_weights_actual_groups_equally_not_candidate_rows_or_fixed_split_size():
    result = summarize([
        record([10, 4], [0, -6]),
        record([2, 6, 7, 8, 9], [0, -3, -1, 0, 1], interval=11),
    ])
    assert result["groups"] == 2
    assert result["candidate_rows"] == 7
    assert result["weighting"] == "equal_group"
    assert result["mean"] == pytest.approx({
        "phase_pressure_cost": 6, "rigid_cost": 5, "oracle_cost": 3,
        "rigid_minus_phase_pressure": -1, "rigid_regret": 2, "phase_pressure_regret": 3,
        "predicted_advantage": 4.5, "actual_advantage": 1,
    })
    assert result["all_groups_outcomes"] == {"beneficial": 1, "harmful": 1, "equal_cost": 0}
    assert result["rigid_optimal_groups"] == result["phase_pressure_optimal_groups"] == 1
