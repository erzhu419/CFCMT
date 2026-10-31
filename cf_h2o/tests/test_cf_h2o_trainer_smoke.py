from __future__ import annotations

import pytest
import torch

from cf_h2o.latent.factor_encoder import TimeVaryingFactorEncoder
from cf_h2o.rl.cf_h2o_trainer import CFH2OTrainer
from cf_h2o.rl.policy_inputs import build_policy_input
from cf_h2o.schemas import GraphPosterior, GraphSpec, TransitionBatch
from cf_h2o.trust.factor_trust import FactorTrustEstimator


class _DummyReplayBuffer:
    def __init__(self, obs_dim=3, act_dim=1, size=64):
        generator = torch.Generator().manual_seed(23)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model_batches = []
        self.data = {}
        for scope, offset in (("real", 0.0), ("sim", 0.2)):
            obs = torch.randn(size, obs_dim, generator=generator) + offset
            act = torch.randn(size, act_dim, generator=generator)
            next_obs = obs + 0.1 * act.repeat(1, obs_dim)
            reward = -obs[:, 0] + 0.1 * act[:, 0]
            self.data[scope] = {
                "observations": obs,
                "actions": act,
                "rewards": reward,
                "next_observations": next_obs,
                "dones": torch.zeros(size),
            }

    def sample(self, batch_size, scope=None, type=None):
        scope = scope or "real"
        source = self.data[scope]
        idx = torch.arange(batch_size) % source["observations"].shape[0]
        batch = {key: value[idx].clone() for key, value in source.items()}
        batch["obs_names"] = ["waiting", "travel_time", "headway"]
        batch["action_names"] = ["holding"]
        return batch

    def add_model_batch(self, batch: TransitionBatch):
        self.model_batches.append(batch)


class _DummyH2O:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self._total_steps = 0
        self.external_sim_weight_provider = None

    def train(self, batch_size, pretrain_steps=0):
        self._total_steps += 1
        metric = {"h2o_loss": float(self._total_steps), "sqrt_IS_ratio": 1.0}
        if self.external_sim_weight_provider is not None:
            sim_batch = self.replay_buffer.sample(batch_size, scope="sim")
            weight = self.external_sim_weight_provider(sim_batch)
            metric["sqrt_IS_ratio"] = float(weight.sqrt().mean())
        return metric


class _DummyWorldModel:
    def __init__(self):
        self.fit_sim_calls = 0
        self.fit_residual_calls = 0

    def fit_sim_modules(self, sim_data):
        self.fit_sim_calls += 1
        return {"initial_loss": 1.0, "final_loss": 0.5}

    def fit_residual_modules(self, real_data, sim_data=None, theta_dict=None):
        self.fit_residual_calls += 1
        return {"initial_loss": 0.8, "final_loss": 0.2, "mode": "aligned", "residual_trained": True}

    def predict(self, batch: TransitionBatch, theta_dict=None):
        batch_size = batch.batch_size
        device = batch.device
        demand_res = torch.full((batch_size, 1), 0.1, device=device)
        speed_res = torch.full((batch_size, 1), 0.2, device=device)
        dwell_res = torch.full((batch_size, 1), 0.05, device=device)
        return {
            "next_observations": batch.observations + 0.1,
            "rewards": batch.rewards + 0.1,
            "mechanism_outputs": {
                "demand": {
                    "base": torch.zeros(batch_size, 1, device=device),
                    "residual": demand_res,
                    "mean": demand_res,
                    "parent_mask": torch.ones(2, device=device),
                    "child_names": ["waiting@t1"],
                },
                "speed": {
                    "base": torch.zeros(batch_size, 1, device=device),
                    "residual": speed_res,
                    "mean": speed_res,
                    "parent_mask": torch.ones(2, device=device) * 0.8,
                    "child_names": ["travel_time@t1"],
                },
                "dwell": {
                    "base": torch.zeros(batch_size, 1, device=device),
                    "residual": dwell_res,
                    "mean": dwell_res,
                    "parent_mask": torch.ones(2, device=device),
                    "child_names": ["headway@t1"],
                },
            },
            "mechanism_uncertainty": {
                "demand": torch.zeros(batch_size, device=device),
                "speed": torch.zeros(batch_size, device=device) + 0.1,
                "dwell": torch.zeros(batch_size, device=device),
            },
        }

    def rollout(self, init_batch, policy, horizon, theta_encoder=None):
        actions = policy(init_batch.observations)
        return TransitionBatch(
            observations=init_batch.observations,
            actions=actions,
            rewards=torch.zeros(init_batch.batch_size),
            next_observations=init_batch.observations + 0.1,
            dones=torch.zeros(init_batch.batch_size),
            source=["model"] * init_batch.batch_size,
            metadata=dict(init_batch.metadata),
        )


def _graph_posterior():
    node_names = [
        "waiting@t",
        "travel_time@t",
        "headway@t",
        "holding@t",
        "waiting@t1",
        "travel_time@t1",
        "headway@t1",
        "reward@t1",
    ]
    n_nodes = len(node_names)
    hard_mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    hard_mask[:4, 4:] = True
    edge_probs = torch.where(hard_mask, torch.full((n_nodes, n_nodes), 0.6), torch.zeros(n_nodes, n_nodes))
    graph = GraphSpec(
        node_names=node_names,
        adjacency=(edge_probs > 0.5).float(),
        hard_mask=hard_mask,
        edge_probs=edge_probs,
        node_groups={},
    )
    return GraphPosterior(graphs=[graph], log_weights=torch.zeros(1), edge_marginals=edge_probs)


def _trainer():
    replay = _DummyReplayBuffer()
    h2o = _DummyH2O(replay)
    world_model = _DummyWorldModel()
    factor_encoder = TimeVaryingFactorEncoder(
        input_dim=3,
        mechanism_names=["demand", "speed", "dwell"],
        latent_dims={"demand": 2, "speed": 2, "dwell": 2},
        hidden_dim=12,
    )
    trust_estimator = FactorTrustEstimator(["demand", "speed", "dwell"], {"w_min": 0.05, "w_max": 5.0})
    trainer = CFH2OTrainer(
        h2o,
        world_model,
        factor_encoder,
        _graph_posterior(),
        trust_estimator,
        replay,
        {
            "batch_size": 8,
            "act_dim": 1,
            "training": {
                "model_rollout_start_step": 1,
                "model_rollout_interval": 1,
                "model_rollout_horizon": 1,
            },
            "latent": {"history_len": 4},
            "trust": {"trust_warmup_steps": 0, "w_min": 0.05, "w_max": 5.0},
            "policy_input": {"use_theta": True, "use_local_graph_embedding": False},
        },
    )
    return trainer, replay, h2o, world_model


def test_full_trainer_smoke_runs_ten_steps_and_keeps_metrics():
    trainer, replay, h2o, world_model = _trainer()
    wm_metrics = trainer.train_world_model()

    last_metrics = {}
    for _ in range(10):
        last_metrics = trainer.train_step()

    assert h2o._total_steps == 10
    assert world_model.fit_sim_calls == 1
    assert world_model.fit_residual_calls == 1
    assert "world_model/residual_loss_dwell" in wm_metrics
    assert "h2o_loss" in last_metrics
    assert "h2o/sqrt_IS_ratio" in last_metrics
    assert "theta_norm/demand" in last_metrics
    assert "trust/speed_mean" in last_metrics
    assert "graph/entropy" in last_metrics
    assert "world_model/residual_loss_dwell" in last_metrics
    assert replay.model_batches
    assert replay.model_batches[-1].source == ["model"] * replay.model_batches[-1].batch_size


def test_policy_input_excludes_forbidden_fields_and_rejects_explicit_shortcuts():
    observations = torch.randn(4, 3)
    theta = {"demand": torch.randn(4, 2)}
    features = build_policy_input(
        observations,
        obs_names=["waiting", "travel_time", "headway"],
        theta_dict=theta,
        metadata={"source": "sim", "domain_id": torch.zeros(4, dtype=torch.long)},
    )

    assert features.values.shape == (4, 5)
    assert all("source" not in name and "domain" not in name and "future" not in name for name in features.names)

    with pytest.raises(ValueError):
        build_policy_input(
            observations,
            metadata={"policy_input_fields": ["obs/waiting", "source"]},
        )
