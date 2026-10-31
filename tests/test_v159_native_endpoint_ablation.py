from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_v159_native_endpoint_ablation as v159
from cf_h2o.traffic_signal.action_ranker import (
    CausalReferenceEndpointResidualRegressor,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "cf_h2o/config/traffic_signal_tsc_v159_native_endpoint_ablation.json"
)


def _endpoint_dataset() -> MechanismDataset:
    return MechanismDataset(
        feature_names=("state", "action"),
        features=np.asarray(
            [
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 5.0],
                [2.0, 6.0],
            ],
            dtype=float,
        ),
        context_names=("context",),
        context=np.zeros((4, 1), dtype=float),
        priors={},
        targets={"interval_cost": np.asarray([0.0, -1.0, 0.0, 1.0])},
        domains=np.asarray(["a", "a", "b", "b"]),
        metadata={
            "action_group_ids": ["group-a", "group-a", "group-b", "group-b"],
            "is_reference": [True, False, True, False],
        },
    )


def test_frozen_v159_protocol_validates() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    v159.validate_protocol(protocol)
    changed = json.loads(json.dumps(protocol))
    changed["development_gate"]["minimum_improved_seeds"] = 6
    with pytest.raises(ValueError, match="frozen protocol changed"):
        v159.validate_protocol(changed)


def test_endpoint_design_retains_absolute_action_location() -> None:
    dataset = _endpoint_dataset()
    model = CausalReferenceEndpointResidualRegressor(
        state_feature_names=("state",),
        action_feature_names=("action",),
        endpoint_feature_names=("action",),
    )
    model._resolve_features(dataset)
    design, reference = model._design(dataset)
    assert design.shape == (4, 3)
    np.testing.assert_array_equal(reference, [True, False, True, False])
    np.testing.assert_allclose(design[[1, 3], 1], [1.0, 1.0])
    np.testing.assert_allclose(design[[1, 3], 2], [0.5, 5.5])
    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]["mean"]
    assert diagnostics["residual_protocol"] == "causal_reference_endpoint_residual_v1"
    assert diagnostics["design_feature_count"] == 3
    np.testing.assert_allclose(prediction[reference], 0.0)


def test_endpoint_model_requires_every_declared_feature() -> None:
    dataset = _endpoint_dataset()
    model = CausalReferenceEndpointResidualRegressor(
        state_feature_names=("state",),
        action_feature_names=("action", "missing"),
        endpoint_feature_names=("action",),
    )
    with pytest.raises(KeyError, match="missing features"):
        model.fit(dataset)


def test_v159_gate_requires_seed_and_demand_consistency() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    comparison = {
        "mean_difference": -0.001,
        "paired_bootstrap_95": [-0.002, -0.0001],
        "improved_seeds": 8,
    }
    demand = {scenario: -0.0001 for scenario in protocol["development"]["scenarios"]}
    assert v159._gate(protocol, comparison, demand)["passed"] is True
    demand["jinan_3x4_real_2500"] = 0.0001
    gate = v159._gate(protocol, comparison, demand)
    assert gate["passed"] is False
    assert gate["checks"]["every_demand_noninferior_passed"] is False
