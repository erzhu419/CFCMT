from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    FrozenCausalSourceSelectorModel,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import _group_adjusted_scores
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


class _FixedModel:
    def __init__(self, score, uncertainty=None, trust=None) -> None:
        self.score = np.asarray(score, dtype=float)
        self.uncertainty = np.asarray(
            np.zeros_like(self.score) if uncertainty is None else uncertainty,
            dtype=float,
        )
        self.trust = np.asarray(
            np.ones_like(self.score) if trust is None else trust,
            dtype=float,
        )

    def predict(self, _dataset):
        return {
            "control_cost": {
                "mean": self.score,
                "uncertainty": self.uncertainty,
                "context_trust": self.trust,
            }
        }


def _dataset() -> MechanismDataset:
    return MechanismDataset(
        feature_names=("feature",),
        features=np.zeros((3, 1)),
        context_names=("context",),
        context=np.zeros((3, 1)),
        priors={},
        targets={"interval_cost": np.zeros(3)},
        domains=np.asarray(["target"] * 3),
        metadata={
            "action_group_ids": np.asarray(["group"] * 3),
            "is_reference": np.asarray([True, False, False]),
        },
    )


@pytest.mark.parametrize(
    ("profile", "expected_source"),
    (
        ("all_sources_mean", np.asarray([0.0, -0.5, -1.5])),
        ("all_sources_median", np.asarray([0.0, -0.5, -1.5])),
        ("all_sources_worst", np.asarray([0.0, 1.0, -1.0])),
    ),
)
def test_selector_runtime_matches_frozen_ensemble_algebra(
    profile, expected_source
) -> None:
    dataset = _dataset()
    model = FrozenCausalSourceSelectorModel(
        source_models={
            "source_a": _FixedModel([0.0, -2.0, -1.0]),
            "source_b": _FixedModel([0.0, 1.0, -2.0]),
        },
        source_objective_modes={
            "source_a": "control_only",
            "source_b": "control_only",
        },
        source_profile=profile,
        source_weight=0.5,
        minimum_source_support=0.0,
        target_only_model=_FixedModel([0.0, -0.2, -0.2]),
    )

    score, _, _, _ = _group_adjusted_scores(
        dataset,
        model.predict(dataset),
        objective_mode="control_only",
    )
    np.testing.assert_allclose(
        score,
        0.5 * np.asarray([0.0, -0.2, -0.2]) + 0.5 * expected_source,
    )


def test_selector_runtime_applies_cross_source_support_before_guard() -> None:
    dataset = _dataset()
    model = FrozenCausalSourceSelectorModel(
        source_models={
            "source_a": _FixedModel([0.0, -2.0, -1.0]),
            "source_b": _FixedModel([0.0, 1.0, -2.0]),
        },
        source_objective_modes={
            "source_a": "control_only",
            "source_b": "control_only",
        },
        source_profile="all_sources_mean",
        source_weight=1.0,
        minimum_source_support=0.6,
        target_only_model=_FixedModel([0.0, 0.0, 0.0]),
    )

    prediction = model.predict(dataset)["control_cost"]
    assert prediction["context_trust"][1] == pytest.approx(0.0)
    assert prediction["context_trust"][2] == pytest.approx(1.0)
    assert model.support_rejected_rows == 2


def test_zero_shot_selector_runtime_never_requires_target_model() -> None:
    dataset = _dataset()
    model = FrozenCausalSourceSelectorModel(
        source_models={"source_a": _FixedModel([0.0, -1.0, 1.0])},
        source_objective_modes={"source_a": "control_only"},
        source_profile="source_a",
        source_weight=1.0,
        minimum_source_support=0.0,
        target_only_model=None,
    )
    score, _, _, _ = _group_adjusted_scores(
        dataset,
        model.predict(dataset),
        objective_mode="control_only",
    )
    np.testing.assert_allclose(score, [0.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="require weight one"):
        FrozenCausalSourceSelectorModel(
            source_models={"source_a": _FixedModel([0.0, -1.0, 1.0])},
            source_objective_modes={"source_a": "control_only"},
            source_profile="source_a",
            source_weight=0.5,
            minimum_source_support=0.0,
            target_only_model=None,
        )
