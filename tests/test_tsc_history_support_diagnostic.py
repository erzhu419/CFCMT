from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import diagnose_tsc_history_support as diagnostic


def joined_example():
    data = SimpleNamespace(
        metadata={
            "action_group_ids": ["first"] * 8 + ["second"] * 8,
            "candidate_states": [f"phase{i}" for i in range(8)] * 2,
            "is_reference": [i in (2, 10) for i in range(16)],
            "row_seeds": [11] * 8 + [33] * 8,
            "row_scenarios": ["base"] * 8 + ["high"] * 8,
        },
        targets={"absolute_cost": 100 + np.arange(16, dtype=float)},
    )
    record = dict(group_id="second", selected_state="phase6", reference_state="phase2",
                  selected_row=0, reference_row=999, selected_cost=114,
                  phase_pressure_cost=110, seed=33, scenario="high")
    return data, record


def test_join_uses_group_and_phase_not_fold_local_row_numbers():
    data, record = joined_example()
    rows, reference, selected = diagnostic.resolve_selected(data, record)
    np.testing.assert_array_equal(rows, np.arange(8, 16))
    assert reference == 2 and selected == 6
    record["selected_row"], record["reference_row"] = 12345, -12345
    same = diagnostic.resolve_selected(data, record)
    np.testing.assert_array_equal(same[0], rows)
    assert same[1:] == (reference, selected)


@pytest.mark.parametrize("field,value,match", [
    ("selected_state", "missing", "joined uniquely"),
    ("selected_cost", 113, "cost differs"),
    ("seed", 11, "seed or demand"),
    ("reference_state", "phase3", "cost differs"),
])
def test_join_rejects_wrong_phase_cost_or_seed(field, value, match):
    data, record = joined_example()
    record[field] = value
    with pytest.raises(ValueError, match=match):
        diagnostic.resolve_selected(data, record)


def test_feasible_oracle_excludes_stay_that_cancels_pp_switch():
    costs = [50, 60, 10, 45, 80, 70, 65, 62]
    switching = [1, 1, 0, 1, 1, 1, 1, 1]
    outcome = diagnostic.action_outcome(costs, switching, reference=0, selected=4)
    assert outcome == dict(
        gain_vehicle_sec=-30, harm_vehicle_sec=30,
        feasible_headroom_vehicle_sec=5, regret_vehicle_sec=35,
        missed_benefit_vehicle_sec=5, best_second_gap_vehicle_sec=5,
        unconstrained_headroom_vehicle_sec=40,
        category="harmful_despite_beneficial_alternative",
    )
    with pytest.raises(ValueError, match="no-stay"):
        diagnostic.action_outcome(costs, switching, reference=0, selected=2)
    costs[3] = 55
    pp_optimal = diagnostic.action_outcome(costs, switching, reference=0, selected=4)
    assert pp_optimal["feasible_headroom_vehicle_sec"] == 0
    assert pp_optimal["regret_vehicle_sec"] == 30
    assert pp_optimal["category"] == "harmful_when_pp_optimal"
    switching[0] = 0
    switching[2] = 1
    # A PP stay permits all switching alternatives.
    unrestricted = diagnostic.action_outcome(costs, switching, reference=0, selected=2)
    assert unrestricted["gain_vehicle_sec"] == 40
    assert unrestricted["regret_vehicle_sec"] == 0


def test_support_uses_train_scaling_and_other_seed_calibration():
    train = np.array([[0.0], [0.1], [10.0], [10.1]])
    arguments = dict(train=train, train_seeds=np.array([1, 1, 2, 2]),
                     train_groups=np.array(["a", "b", "c", "d"]),
                     train_scenarios=np.array(["base"] * 4), train_gains=None)
    result = diagnostic.support(np.array([[1.0], [1000.0]]), query_seeds=[3, 3],
                                query_scenarios=["base", "base"], **arguments)
    alone = diagnostic.support(np.array([[1.0]]), query_seeds=[3],
                               query_scenarios=["base"], **arguments)
    assert result[0]["distance"] == pytest.approx(0.9 / train.std(), abs=1e-12)
    assert result[0] == alone[0]  # Extreme held-out state cannot change the scale.
    # Same-seed neighbors are 0.1 apart; cross-seed neighbors are about 10 apart.
    # A calibration that merely removes self rows would incorrectly yield 100%.
    assert result[0]["training_percentile"] == 0
    assert result[1]["training_percentile"] == 1
    assert result[0]["neighbor_gains_vehicle_sec"] is None
    with pytest.raises(ValueError, match="seeds leaked"):
        diagnostic.support(np.array([[1.0]]), query_seeds=[1],
                           query_scenarios=["base"], **arguments)


def test_action_neighbors_collapse_seven_candidates_to_one_state_group():
    train = np.array([[10 * group + .1 * action, 4.0]
                      for group in range(6) for action in range(7)])
    groups = np.repeat([f"g{i}" for i in range(6)], 7)
    result = diagnostic.support(
        np.array([[.01, 5.0]]), train, query_seeds=[99],
        train_seeds=np.repeat(np.arange(6), 7), train_groups=groups,
        train_scenarios=np.array(["base"] * 42), query_scenarios=["base"],
        train_gains=np.arange(42),
    )[0]
    assert result["neighbor_groups"] == [f"g{i}" for i in range(5)]
    assert result["neighbor_gains_vehicle_sec"] == [0, 7, 14, 21, 28]
    assert result["novel_constant_columns"] == 1


def test_same_demand_neighbors_separate_interleaved_opposite_label_populations():
    # Nearby A/B states have opposite labels. Within each demand, pairs 0.1 apart
    # share a seed, so legitimate same-demand calibration distances are about 10.
    positions = [0, .1, 10, 10.1, 20, 20.1]
    train = np.array([[value + offset] for value in positions for offset in (0, 1)])
    scenarios = np.array(["A", "B"] * 6)
    groups = np.array([f"{demand}{i}" for i in range(6) for demand in ("A", "B")])
    seeds = np.array([seed for i in range(6) for seed in (1 + i // 2, 4 + i // 2)])
    gains = np.array([100, -100] * 6)
    result = diagnostic.support(
        np.array([[1.2]]), train, query_seeds=[99], train_seeds=seeds,
        train_groups=groups, train_scenarios=scenarios, query_scenarios=["A"],
        train_gains=gains,
    )[0]
    assert result["nearest_other_demand"]
    assert result["neighbor_groups"][0] == "B1"
    assert set(result["neighbor_gains_vehicle_sec"]) == {-100, 100}
    assert result["same_demand_neighbor_gains_vehicle_sec"] == [100] * 5
    assert result["same_demand_distance"] == pytest.approx(1.1 / train.std(), abs=1e-12)
    # Using same-seed A pairs or B neighbors in the calibration would give 100%.
    assert result["same_demand_percentile"] == 0
