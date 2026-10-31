from __future__ import annotations

import torch

from cf_h2o.rl.h2o_mcwm_bridge import H2OFactorTrustBridge
from cf_h2o.schemas import TransitionBatch
from cf_h2o.trust.factor_trust import FactorTrustEstimator, FactorWiseTrustWeightProvider
from cf_h2o.trust.qdelta_adapter import QDeltaTrustAdapter
from cf_h2o.trust.weight_composer import WeightComposer


def _mechanism_outputs(batch_size: int = 6, parent_mask_value: float = 1.0, residual_scale: float = 0.1):
    residual = torch.full((batch_size, 2), residual_scale)
    return {
        "demand": {
            "base": torch.zeros(batch_size, 2),
            "residual": residual,
            "mean": residual,
            "parent_mask": torch.full((3,), parent_mask_value),
            "child_names": ["waiting@t1", "load@t1"],
        },
        "speed": {
            "base": torch.zeros(batch_size, 1),
            "residual": torch.full((batch_size, 1), residual_scale),
            "mean": torch.full((batch_size, 1), residual_scale),
            "parent_mask": torch.full((2,), parent_mask_value),
            "child_names": ["travel_time@t1"],
        },
    }


def test_factor_trust_shapes_and_ranges():
    estimator = FactorTrustEstimator(["demand", "speed"], {"w_min": 0.05, "w_max": 5.0})
    outputs = _mechanism_outputs(batch_size=5)
    trust = estimator(
        outputs,
        mechanism_uncertainty={
            "demand": torch.zeros(5),
            "speed": torch.ones(5) * 0.2,
        },
    )

    assert set(trust) == {"demand", "speed"}
    for value in trust.values():
        assert value.shape == (5,)
        assert torch.isfinite(value).all()
        assert torch.all(value >= 0.05)
        assert torch.all(value <= 5.0)


def test_weight_composer_modes_are_finite_and_clipped():
    trust = {
        "demand": torch.tensor([0.01, 0.5, 2.0]),
        "speed": torch.tensor([0.2, 0.5, 20.0]),
    }

    for mode in ("min", "arithmetic_mean", "product", "geometric_mean", "reward_path_weighted"):
        composer = WeightComposer(mode=mode, w_min=0.05, w_max=5.0, reward_path_weights={"speed": 2.0})
        weight = composer.compose(trust)
        assert weight.shape == (3,)
        assert torch.isfinite(weight).all()
        assert torch.all(weight >= 0.05)
        assert torch.all(weight <= 5.0)


def test_graph_uncertainty_highers_lowers_trust():
    estimator = FactorTrustEstimator(["demand"], {"graph_uncertainty_scale": 1.0})
    certain = estimator({"demand": _mechanism_outputs(parent_mask_value=1.0)["demand"]})["demand"]
    uncertain = estimator({"demand": _mechanism_outputs(parent_mask_value=0.5)["demand"]})["demand"]

    assert torch.all(uncertain < certain)


def test_rollout_horizon_does_not_increase_trust():
    estimator = FactorTrustEstimator(["demand"], {"horizon_decay": 0.8})
    outputs = {"demand": _mechanism_outputs()["demand"]}

    h1 = estimator(outputs, rollout_horizon=1)["demand"]
    h5 = estimator(outputs, rollout_horizon=5)["demand"]

    assert torch.all(h5 <= h1)


class _DummyWorldModel:
    def predict(self, batch: TransitionBatch, theta_dict=None):
        batch_size = batch.batch_size
        residual = batch.observations[:, :1] * 0.0 + 0.2
        return {
            "mechanism_outputs": {
                "demand": {
                    "base": torch.zeros(batch_size, 1, device=batch.device),
                    "residual": residual,
                    "mean": residual,
                    "parent_mask": torch.ones(2, device=batch.device),
                    "child_names": ["waiting@t1"],
                },
            },
            "mechanism_uncertainty": {"demand": torch.zeros(batch_size, device=batch.device)},
        }


class _CallableQDelta:
    def __call__(self, observations, actions):
        return torch.full((observations.shape[0],), 0.5, device=observations.device)


def _sim_batch(batch_size: int = 4):
    return {
        "observations": torch.randn(batch_size, 3),
        "actions": torch.randn(batch_size, 1),
        "rewards": torch.randn(batch_size),
        "next_observations": torch.randn(batch_size, 3),
        "dones": torch.zeros(batch_size),
    }


def test_factor_trust_provider_returns_detached_h2o_weight():
    estimator = FactorTrustEstimator(["demand"], {"w_min": 0.05, "w_max": 5.0})
    composer = WeightComposer("geometric_mean", w_min=0.05, w_max=5.0)
    provider = FactorWiseTrustWeightProvider(
        _DummyWorldModel(),
        estimator,
        composer,
        qdelta_adapter=QDeltaTrustAdapter(_CallableQDelta()),
        config={"trust_warmup_steps": 2, "joint_train_trust": False},
    )
    batch = _sim_batch()

    warmup = provider.compute_weight(batch, step=0)
    weight = provider.compute_weight(batch, step=3)

    assert torch.allclose(warmup, torch.ones(4))
    assert weight.shape == (4,)
    assert torch.isfinite(weight).all()
    assert not weight.requires_grad
    assert torch.all(weight < 1.0)


class _DummyH2O:
    def __init__(self):
        self._total_steps = 0
        self.external_sim_weight_provider = None

    def train(self, batch_size, pretrain_steps=0):
        self._total_steps += 1
        return {"batch_size": batch_size}


class _GradProvider:
    def compute_weight(self, sim_batch, step=0):
        return torch.ones(sim_batch["observations"].shape[0], requires_grad=True) * 0.7


def test_h2o_factor_trust_bridge_warmup_and_detach():
    h2o = _DummyH2O()
    bridge = H2OFactorTrustBridge(h2o, _GradProvider(), config={"trust_warmup_steps": 5, "joint_train_trust": False})
    batch = _sim_batch(batch_size=3)

    warmup = bridge.compute_sim_weight(batch)
    h2o._total_steps = 6
    weighted = bridge.compute_sim_weight(batch)

    assert torch.allclose(warmup, torch.ones(3))
    assert torch.allclose(weighted, torch.full((3,), 0.7))
    assert not weighted.requires_grad
    assert h2o.external_sim_weight_provider.__self__ is bridge
