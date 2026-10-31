from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_counterfactual_signal_audit import (
    _dilution_summary,
    summarize_scenario_datasets,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _dataset(
    scenario: str,
    tls_count: int,
    interval_cost: list[float],
    next_queue: list[float],
) -> MechanismDataset:
    rows = len(interval_cost)
    return MechanismDataset(
        feature_names=("x",),
        features=np.zeros((rows, 1)),
        context_names=("z",),
        context=np.ones((rows, 1)),
        priors={
            "interval_cost": np.zeros(rows),
            "next_total_queue": np.zeros(rows),
        },
        targets={
            "interval_cost": np.asarray(interval_cost, dtype=float),
            "next_total_queue": np.asarray(next_queue, dtype=float),
        },
        domains=np.asarray([scenario] * rows),
        metadata={
            "scenario": scenario,
            "action_group_ids": ["a", "a", "a", "b", "b", "b"],
            "controllable_tls_count": tls_count,
            "counterfactual_cost_scope": (
                "global_system_vehicles_per_controlled_lane"
            ),
        },
    )


def test_signal_summary_measures_group_range_and_action_alignment():
    dataset = _dataset(
        "small",
        1,
        [1.0, 2.0, 3.0, 4.0, 4.0, 4.0],
        [10.0, 20.0, 30.0, 8.0, 8.0, 8.0],
    )
    result = summarize_scenario_datasets([dataset])
    signal = result["targets"]["interval_cost"]
    alignment = result["primary_to_local_alignment"]["next_total_queue"]
    assert result["action_group_count"] == 2
    assert signal["within_group_range_median"] == 1.0
    assert signal["exact_zero_range_fraction"] == 0.5
    assert alignment["best_action_overlap_fraction"] == 1.0
    assert alignment["nonconstant_group_count"] == 1


def test_dilution_summary_detects_inverse_network_scaling():
    scenarios = {
        "one": {
            "controllable_tls_count": 1,
            "targets": {"interval_cost": {"within_group_range_median": 1.0}},
        },
        "ten": {
            "controllable_tls_count": 10,
            "targets": {"interval_cost": {"within_group_range_median": 0.1}},
        },
        "hundred": {
            "controllable_tls_count": 100,
            "targets": {"interval_cost": {"within_group_range_median": 0.01}},
        },
    }
    result = _dilution_summary(scenarios)
    assert result["log_tls_vs_log_primary_range_pearson"] < -0.99
    assert np.isclose(result["log_log_slope"], -1.0)
