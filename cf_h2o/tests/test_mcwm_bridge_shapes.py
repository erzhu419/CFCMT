from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cf_h2o.data.h2o_buffer_adapter import transition_batch_from_h2o
from cf_h2o.rl.h2o_mcwm_bridge import H2OMCWMBridge
from cf_h2o.schemas import TransitionBatch
from cf_h2o.world_model.mcwm_adapter import MCWMAdapter


def test_mcwm_adapter_predict_and_trust_shapes():
    obs_dim = 4
    act_dim = 2
    batch_size = 8
    adapter = MCWMAdapter(
        obs_dim,
        act_dim,
        {
            "ensemble_size": 3,
            "hidden_dim": 16,
            "residual_hidden_dim": 8,
            "w_min": 0.05,
            "w_max": 5.0,
        },
        device="cpu",
    )

    observations = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, act_dim)

    pred = adapter.predict(observations, actions, deterministic=True)
    assert pred["next_observations"].shape == (batch_size, obs_dim)
    assert pred["rewards"].shape == (batch_size,)
    assert pred["epistemic"].shape == (batch_size,)
    assert pred["aleatoric"].shape == (batch_size,)
    assert torch.isfinite(pred["next_observations"]).all()
    assert torch.isfinite(pred["rewards"]).all()

    weights = adapter.trust_weight(observations, actions)
    assert weights.shape == (batch_size,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.05)
    assert torch.all(weights <= 5.0)


def test_mcwm_sim_model_converges_on_linear_synthetic_dynamics():
    torch.manual_seed(7)
    np.random.seed(7)

    obs_dim = 3
    act_dim = 1
    n_train = 384
    n_eval = 64
    observations = torch.randn(n_train + n_eval, obs_dim) * 0.5
    actions = torch.randn(n_train + n_eval, act_dim) * 0.5
    delta = 0.2 * observations + 0.15 * actions.repeat(1, obs_dim)
    rewards = (observations[:, 0] - 0.5 * observations[:, 1] + 0.25 * actions[:, 0])
    next_observations = observations + delta
    dones = torch.zeros(n_train + n_eval)

    train_batch = TransitionBatch(
        observations=observations[:n_train],
        actions=actions[:n_train],
        rewards=rewards[:n_train],
        next_observations=next_observations[:n_train],
        dones=dones[:n_train],
    )

    adapter = MCWMAdapter(
        obs_dim,
        act_dim,
        {
            "ensemble_size": 2,
            "hidden_dim": 32,
            "train_epochs_sim": 60,
            "batch_size": 64,
            "val_ratio": 0.2,
            "patience": 60,
        },
        device="cpu",
    )

    eval_obs = observations[n_train:]
    eval_actions = actions[n_train:]
    eval_next = next_observations[n_train:]
    with torch.no_grad():
        before = adapter.predict(eval_obs, eval_actions, deterministic=True)["next_observations"]
        before_mse = torch.mean((before - eval_next) ** 2).item()

    adapter.fit_sim_model(train_batch)

    with torch.no_grad():
        after = adapter.predict(eval_obs, eval_actions, deterministic=True)["next_observations"]
        after_mse = torch.mean((after - eval_next) ** 2).item()

    assert after_mse < before_mse * 0.5
    assert after_mse < 0.05


def test_h2o_batch_adapter_preserves_metadata_without_policy_shortcutting():
    batch_size = 4
    h2o_batch = {
        "observations": torch.randn(batch_size, 3),
        "actions": torch.randn(batch_size, 1),
        "rewards": torch.randn(batch_size),
        "next_observations": torch.randn(batch_size, 3),
        "dones": torch.zeros(batch_size),
        "z_t": torch.randn(batch_size, 30),
        "z_t1": torch.randn(batch_size, 30),
        "_indices": torch.arange(batch_size),
    }

    converted = transition_batch_from_h2o(
        h2o_batch,
        source="real",
        line_id=["7X"] * batch_size,
        route_id=["7X_main"] * batch_size,
    )

    assert converted.observations.shape == (batch_size, 3)
    assert converted.z_t.shape == (batch_size, 30)
    assert converted.z_t1.shape == (batch_size, 30)
    assert converted.source == ["real"] * batch_size
    assert converted.line_id == ["7X"] * batch_size
    assert converted.route_id == ["7X_main"] * batch_size
    assert "_indices" in converted.metadata


class _ZeroLogitDiscriminator(nn.Module):
    def forward(self, z_t, z_t1):
        return torch.zeros(z_t.shape[0], 1, device=z_t.device)


def test_h2oplus_default_sim_weight_matches_old_discriminator_ratio():
    root = Path(__file__).resolve().parents[2]
    simple_sac = root / "H2Oplus" / "SimpleSAC"
    sys.path.insert(0, str(simple_sac))
    try:
        from h2oplus_bus import H2OPlusBus
    finally:
        sys.path.remove(str(simple_sac))

    h2o = object.__new__(H2OPlusBus)
    h2o.discriminator = _ZeroLogitDiscriminator()
    h2o.external_sim_weight_provider = None
    sim_batch = {
        "observations": torch.randn(5, 3),
        "actions": torch.randn(5, 1),
        "next_observations": torch.randn(5, 3),
        "z_t": torch.randn(5, 30),
        "z_t1": torch.randn(5, 30),
    }

    weights = h2o.compute_sim_weight(sim_batch)
    assert weights.shape == (5,)
    assert torch.allclose(weights, torch.ones(5), atol=1e-6)

    h2o.external_sim_weight_provider = lambda batch: torch.full((batch["observations"].shape[0],), 2.5)
    external_weights = h2o.compute_sim_weight(sim_batch)
    assert torch.allclose(external_weights, torch.full((5,), 2.5))


class _DummyMCWM:
    def trust_weight(self, observations, actions):
        return torch.full((observations.shape[0],), 0.75, device=observations.device)


class _DummyH2O:
    def __init__(self):
        self._total_steps = 0
        self.external_sim_weight_provider = None

    def train(self, batch_size, pretrain_steps=0):
        self._total_steps += 1
        batch = {
            "observations": torch.randn(batch_size, 3),
            "actions": torch.randn(batch_size, 1),
        }
        weight = self.external_sim_weight_provider(batch)
        return {"step": self._total_steps, "weight_mean": float(weight.mean())}


def test_bridge_replaces_weight_provider_for_three_steps():
    h2o = _DummyH2O()
    bridge = H2OMCWMBridge(h2o, _DummyMCWM(), config={"trust_warmup_steps": 0})

    metrics = [bridge.train_step(6) for _ in range(3)]

    assert [m["step"] for m in metrics] == [1, 2, 3]
    assert all(m["weight_mean"] == 0.75 for m in metrics)
    assert h2o.external_sim_weight_provider.__self__ is bridge
