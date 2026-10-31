from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTEXT_NAMES_V3,
    FEATURE_NAMES_V3,
    policy_consistent_mechanism_priors_v4,
)
from cf_h2o.eval.traffic_signal_topology_context_repair import (
    REPAIR_PROTOCOL,
    repair_dataset_topology_context,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def test_repair_changes_only_context_and_context_dependent_priors() -> None:
    rows = 3
    features = np.zeros((rows, len(FEATURE_NAMES_V3)), dtype=float)
    old_context = np.zeros((rows, len(CONTEXT_NAMES_V3)), dtype=float)
    old_priors = policy_consistent_mechanism_priors_v4(
        features,
        old_context,
        control_interval_sec=10,
        counterfactual_horizon_intervals=6,
    )
    targets = {name: np.arange(rows, dtype=float) for name in old_priors}
    dataset = MechanismDataset(
        feature_names=FEATURE_NAMES_V3,
        features=features,
        context_names=CONTEXT_NAMES_V3,
        context=old_context,
        priors=old_priors,
        targets=targets,
        domains=np.asarray(["test"] * rows),
        metadata={
            "scenario": "test",
            "control_interval_sec": 10,
            "counterfactual_horizon_intervals": 6,
        },
    )
    corrected = np.linspace(0.0, 1.0, len(CONTEXT_NAMES_V3))

    repaired = repair_dataset_topology_context(dataset, corrected)

    assert np.array_equal(repaired.features, dataset.features)
    assert all(
        np.array_equal(repaired.targets[name], dataset.targets[name])
        for name in dataset.targets
    )
    assert np.allclose(repaired.context, corrected[None, :])
    assert repaired.metadata["static_topology_context_repair"]["protocol"] == REPAIR_PROTOCOL
    assert set(repaired.priors) == set(dataset.priors)
