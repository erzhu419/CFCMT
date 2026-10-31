from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    local_candidate_key,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
    METHOD_POLICY,
    POLICIES,
    _fitted_runtime_bundle,
)


class _StaticModel:
    def __init__(self, mean, uncertainty):
        self.mean = np.asarray(mean, dtype=float)
        self.uncertainty = np.asarray(uncertainty, dtype=float)

    def predict(self, dataset):
        return {
            "control_cost": {
                "mean": self.mean.copy(),
                "uncertainty": self.uncertainty.copy(),
                "context_trust": np.ones(dataset.size, dtype=float),
            }
        }


def _one_group_dataset():
    return SimpleNamespace(
        size=2,
        metadata={
            "action_group_ids": np.asarray(["group", "group"]),
            "is_reference": np.asarray([True, False]),
        },
    )


def test_frozen_constant_blend_reproduces_adjusted_score_interpolation():
    model = FrozenAnchoredBlendModel(
        anchor_model=_StaticModel([0.0, 1.0], [0.0, 0.2]),
        correction_model=_StaticModel([0.0, -1.0], [0.0, 0.4]),
        candidate="constant_alpha_0p5",
        anchor_objective_mode="control_only",
        correction_objective_mode="control_only",
    )

    prediction = model.predict(_one_group_dataset())["control_cost"]

    assert np.allclose(prediction["mean"], [0.0, 0.0])
    assert np.allclose(prediction["uncertainty"], [0.0, 0.3])
    assert model.audit.to_dict()["mean_alpha"] == 0.5


def test_frozen_local_blend_uses_same_disagreement_and_confidence_rule():
    model = FrozenAnchoredBlendModel(
        anchor_model=_StaticModel([0.0, 1.0], [0.0, 0.1]),
        correction_model=_StaticModel([0.0, -1.0], [0.0, 0.1]),
        candidate=local_candidate_key(1.0, 0.25),
        anchor_objective_mode="control_only",
        correction_objective_mode="control_only",
    )

    prediction = model.predict(_one_group_dataset())["control_cost"]
    audit = model.audit.to_dict()

    assert np.allclose(prediction["mean"], [0.0, 0.0])
    assert audit["mean_alpha"] == 0.5
    assert audit["nonzero_alpha_fraction"] == 1.0


def test_closed_loop_policy_order_matches_preregistered_matrix():
    assert POLICIES == (
        METHOD_POLICY,
        "fixed_time",
        "phase_pressure",
        "max_pressure",
        "rigid_anchor_mpc",
        "simulator_only_mpc",
        "h2oplus_style_dense_residual_mpc",
    )


def test_online_runtime_bundle_rejects_non_control_objective():
    offline = SimpleNamespace(
        prior_spec=SimpleNamespace(key="phase_pressure"),
        objective_modes={"family": "mechanism_weighted"},
    )
    payload = {"model_ensemble": [offline]}

    try:
        _fitted_runtime_bundle(
            offline_payload=payload,
            family="family",
            model=_StaticModel([0.0], [0.0]),
            prediction_horizon_sec=60,
        )
    except ValueError as exc:
        assert "control_only" in str(exc)
    else:
        raise AssertionError("non-control objective was accepted")
