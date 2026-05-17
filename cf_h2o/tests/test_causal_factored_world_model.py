from __future__ import annotations

import pytest
import torch

from cf_h2o.graph.feature_registry import FeatureRegistry
from cf_h2o.schemas import GraphPosterior, GraphSpec, TransitionBatch
from cf_h2o.world_model.causal_factored_residual import CausalFactoredResidualWorldModel
from cf_h2o.world_model.mcwm_adapter import MCWMAdapter


OBS_NAMES = ["waiting", "travel_time", "headway"]
ACTION_NAMES = ["holding"]


def _make_data(n_samples: int = 256, include_paired: bool = True):
    generator = torch.Generator().manual_seed(19)
    waiting = torch.randn(n_samples, generator=generator)
    travel_time = torch.randn(n_samples, generator=generator)
    headway = torch.randn(n_samples, generator=generator)
    holding = torch.randn(n_samples, generator=generator)
    theta = {
        "demand": torch.randn(n_samples, 1, generator=generator),
        "speed": torch.randn(n_samples, 1, generator=generator),
        "headway": torch.randn(n_samples, 1, generator=generator),
        "reward": torch.randn(n_samples, 1, generator=generator),
    }
    noise = lambda scale=0.01: scale * torch.randn(n_samples, generator=generator)

    sim_waiting_next = 0.70 * waiting + 0.10 * holding + noise()
    sim_travel_next = 0.60 * travel_time - 0.20 * holding + noise()
    sim_headway_next = 0.50 * headway + 0.30 * travel_time - 0.10 * holding + noise()
    sim_reward = -0.60 * waiting - 0.40 * headway - 0.10 * holding + noise()

    waiting_residual = 0.30 * theta["demand"][:, 0] + 0.20 * holding
    travel_residual = -0.35 * theta["speed"][:, 0] + 0.10 * travel_time
    headway_residual = 0.25 * theta["headway"][:, 0] + 0.15 * holding
    reward_residual = -0.20 * theta["reward"][:, 0] + 0.10 * holding - 0.05 * waiting

    observations = torch.stack([waiting, travel_time, headway], dim=1)
    actions = holding.reshape(-1, 1)
    sim_next = torch.stack([sim_waiting_next, sim_travel_next, sim_headway_next], dim=1)
    real_next = sim_next + torch.stack([waiting_residual, travel_residual, headway_residual], dim=1)
    real_reward = sim_reward + reward_residual
    metadata = {"obs_names": OBS_NAMES, "action_names": ACTION_NAMES}
    if include_paired:
        metadata = {
            **metadata,
            "sim_next_observations": sim_next,
            "sim_rewards": sim_reward,
        }
    real_data = TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=real_reward,
        next_observations=real_next,
        dones=torch.zeros(n_samples),
        metadata=metadata,
    )
    sim_data = TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=sim_reward,
        next_observations=sim_next,
        dones=torch.zeros(n_samples),
        metadata={"obs_names": OBS_NAMES, "action_names": ACTION_NAMES},
    )
    return sim_data, real_data, theta


def _posterior_and_specs(batch: TransitionBatch):
    registry = FeatureRegistry.from_transition_dataset(batch)
    hard_mask = registry.build_temporal_hard_mask()
    edge_probs = torch.where(hard_mask, torch.full_like(hard_mask, 0.02, dtype=torch.float32), torch.zeros_like(hard_mask, dtype=torch.float32))
    idx = registry.node_index
    edges = {
        ("waiting@t", "waiting@t1"): 1.0,
        ("holding@t", "waiting@t1"): 0.5,
        ("travel_time@t", "travel_time@t1"): 1.0,
        ("holding@t", "travel_time@t1"): 1.0,
        ("headway@t", "headway@t1"): 1.0,
        ("travel_time@t", "headway@t1"): 1.0,
        ("holding@t", "headway@t1"): 1.0,
        ("waiting@t", "reward@t1"): 1.0,
        ("headway@t", "reward@t1"): 1.0,
        ("holding@t", "reward@t1"): 1.0,
    }
    for (src, dst), prob in edges.items():
        edge_probs[idx[src], idx[dst]] = prob
    graph = GraphSpec(
        node_names=registry.node_names,
        adjacency=(edge_probs > 0.8).float(),
        hard_mask=hard_mask,
        edge_probs=edge_probs,
        node_groups=registry.node_groups(),
    )
    posterior = GraphPosterior(
        graphs=[graph],
        log_weights=torch.zeros(1),
        edge_marginals=edge_probs,
        diagnostics={"node_names": registry.node_names},
    )
    specs = [
        {
            "name": "demand",
            "child_names": ["waiting@t1"],
            "parent_names": ["waiting@t", "holding@t"],
            "latent_dim": 1,
            "loss_type": "mse",
        },
        {
            "name": "speed",
            "child_names": ["travel_time@t1"],
            "parent_names": ["travel_time@t", "holding@t"],
            "latent_dim": 1,
            "loss_type": "mse",
        },
        {
            "name": "headway",
            "child_names": ["headway@t1"],
            "parent_names": ["headway@t", "travel_time@t", "holding@t"],
            "latent_dim": 1,
            "loss_type": "mse",
        },
        {
            "name": "reward",
            "child_names": ["reward@t1"],
            "parent_names": ["waiting@t", "headway@t", "holding@t"],
            "latent_dim": 1,
            "loss_type": "mse",
        },
    ]
    return posterior, specs


def _model(batch: TransitionBatch):
    posterior, specs = _posterior_and_specs(batch)
    return CausalFactoredResidualWorldModel(
        specs,
        posterior,
        {
            "hidden_dim": 32,
            "batch_size": 64,
            "train_epochs_sim": 60,
            "train_epochs_residual": 80,
            "lr": 3e-3,
            "residual_lr": 3e-3,
            "weight_decay": 0.0,
        },
    )


def test_causal_factored_predict_shapes_match_monolithic_mcwm():
    sim_data, real_data, theta = _make_data(n_samples=32)
    model = _model(real_data)
    monolithic = MCWMAdapter(obs_dim=3, act_dim=1, config={"ensemble_size": 2, "hidden_dim": 16}, device="cpu")

    mono_pred = monolithic.predict(real_data.observations, real_data.actions, deterministic=True)
    factored_pred = model.predict(real_data, theta)

    assert factored_pred["next_observations"].shape == mono_pred["next_observations"].shape
    assert factored_pred["rewards"].shape == mono_pred["rewards"].shape
    assert set(factored_pred["mechanism_outputs"]) == {"demand", "speed", "headway", "reward"}


def test_parent_inputs_exclude_forbidden_future_and_shortcut_nodes():
    _, real_data, _ = _make_data(n_samples=16)
    model = _model(real_data)

    for spec in model.mechanism_specs:
        assert all(parent.endswith("@t") for parent in spec.parent_names)
        assert all("@t1" not in parent for parent in spec.parent_names)
        assert all("domain" not in parent and "source" not in parent and "city" not in parent for parent in spec.parent_names)
        parents, mask = model.parent_inputs(real_data, spec.name)
        assert parents.shape[1] == len(spec.parent_names)
        assert mask.shape == (len(spec.parent_names),)

    posterior, specs = _posterior_and_specs(real_data)
    bad = dict(specs[0])
    bad["parent_names"] = ["reward@t1"]
    with pytest.raises(ValueError):
        CausalFactoredResidualWorldModel([bad], posterior, {"hidden_dim": 8})


def test_graph_posterior_soft_mask_broadcasts_into_mechanism_module():
    _, real_data, _ = _make_data(n_samples=8)
    model = _model(real_data)

    parents, mask = model.parent_inputs(real_data, "demand")
    assert torch.allclose(mask, torch.tensor([1.0, 0.5]))
    out = model.base_modules["demand"](parents, mask=mask)

    assert out["mean"].shape == (8, 1)
    assert torch.isfinite(out["mean"]).all()


def test_paired_residual_training_loss_decreases_on_synthetic_mechanisms():
    sim_data, real_data, theta = _make_data(n_samples=256)
    model = _model(real_data)

    sim_metrics = model.fit_sim_modules(sim_data)
    residual_metrics = model.fit_residual_modules(real_data, theta_dict=theta)
    pred = model.predict(real_data, theta)
    real_mse = torch.mean((pred["next_observations"] - real_data.next_observations) ** 2).item()
    uncorrected_mse = torch.mean((sim_data.next_observations - real_data.next_observations) ** 2).item()

    assert sim_metrics["final_loss"] < sim_metrics["initial_loss"]
    assert residual_metrics["mode"] == "paired"
    assert residual_metrics["final_loss"] < residual_metrics["initial_loss"] * 0.4
    assert real_mse < uncorrected_mse


def test_aligned_residual_training_path_runs_without_paired_targets():
    sim_data, real_data, theta = _make_data(n_samples=96, include_paired=False)
    model = _model(real_data)

    metrics = model.fit_residual_modules(real_data, sim_data=sim_data, theta_dict=theta)

    assert metrics["residual_trained"] is True
    assert metrics["mode"] == "aligned"
    assert "alignment_distance_mean" in metrics
