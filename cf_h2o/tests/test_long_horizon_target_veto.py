import numpy as np

from cf_h2o.traffic_signal.long_horizon_target_veto import (
    LongHorizonTargetVeto,
    LongHorizonTargetVetoConfig,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


class _PredictionModel:
    def __init__(self, mean, uncertainty, trust) -> None:
        self.values = {
            "mean": np.asarray(mean, dtype=float),
            "uncertainty": np.asarray(uncertainty, dtype=float),
            "context_trust": np.asarray(trust, dtype=float),
        }

    def predict(self, _dataset):
        return {"control_cost": self.values}


def _dataset() -> MechanismDataset:
    return MechanismDataset(
        feature_names=("x",),
        features=np.asarray([[0.0], [1.0]]),
        context_names=("c",),
        context=np.asarray([[0.0], [0.0]]),
        priors={"interval_cost": np.zeros(2)},
        targets={"interval_cost": np.asarray([0.0, -1.0])},
        domains=np.asarray(["city", "city"]),
        metadata={
            "action_group_ids": ["g", "g"],
            "is_reference": [True, False],
        },
    )


def test_long_horizon_veto_accepts_only_negative_supported_upper_bound() -> None:
    accepted = LongHorizonTargetVeto(
        model=_PredictionModel([0.0, -0.5], [0.0, 0.1], [1.0, 0.8]),
        config=LongHorizonTargetVetoConfig(
            enabled=True, risk_multiplier=1.0, min_context_trust=0.75
        ),
    ).evaluate(_dataset(), candidate_index=1, reference_index=0)
    rejected = LongHorizonTargetVeto(
        model=_PredictionModel([0.0, -0.05], [0.0, 0.1], [1.0, 0.8]),
        config=LongHorizonTargetVetoConfig(
            enabled=True, risk_multiplier=1.0, min_context_trust=0.75
        ),
    ).evaluate(_dataset(), candidate_index=1, reference_index=0)

    assert accepted["eligible"] is True
    assert accepted["upper"] < 0.0
    assert rejected["eligible"] is False
    assert rejected["rejection"] == "target_confidence"


def test_long_horizon_veto_rejects_unsupported_action() -> None:
    decision = LongHorizonTargetVeto(
        model=_PredictionModel([0.0, -1.0], [0.0, 0.0], [1.0, 0.2]),
        config=LongHorizonTargetVetoConfig(
            enabled=True, risk_multiplier=0.0, min_context_trust=0.5
        ),
    ).evaluate(_dataset(), candidate_index=1, reference_index=0)

    assert decision["eligible"] is False
    assert decision["rejection"] == "target_support"
