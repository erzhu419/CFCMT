from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_pilot import (
    compare_counterfactual_costs,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _dataset(costs: list[float]) -> MechanismDataset:
    size = len(costs)
    return MechanismDataset(
        feature_names=("feature",),
        features=np.arange(size, dtype=float).reshape(-1, 1),
        context_names=("context",),
        context=np.ones((size, 1), dtype=float),
        priors={"interval_cost": np.zeros(size, dtype=float)},
        targets={
            "interval_cost": np.asarray(costs, dtype=float),
            "one_step_total_queue": np.arange(size, dtype=float),
        },
        domains=np.asarray(["city"] * size, dtype=str),
        metadata={
            "action_group_ids": ["group-a", "group-a", "group-b", "group-b"],
            "candidate_states": ["a0", "a1", "b0", "b1"],
        },
    )


def test_compare_counterfactual_costs_matches_trajectories_and_new_labels() -> None:
    legacy = _dataset([2.0, 1.0, 3.0, 5.0])
    waiting = _dataset([4.0, 1.0, 8.0, 2.0])
    result = compare_counterfactual_costs(legacy, waiting)
    assert result["passed"] is True
    assert result["exact_action_key_set"] is True
    assert result["trajectory_invariance"]["passed"] is True
    assert result["informative_waiting_group_count"] == 2
    assert result["same_best_action_fraction"] == 0.5


def test_compare_counterfactual_costs_rejects_mismatched_actions() -> None:
    legacy = _dataset([2.0, 1.0, 3.0, 5.0])
    waiting = _dataset([4.0, 1.0, 8.0, 2.0])
    waiting.metadata["candidate_states"][-1] = "different"
    result = compare_counterfactual_costs(legacy, waiting)
    assert result["passed"] is False
    assert result["exact_action_key_set"] is False
