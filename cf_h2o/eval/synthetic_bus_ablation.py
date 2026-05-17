"""End-to-end synthetic bus mechanism ablation for CF-H2O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cf_h2o.graph.feature_registry import FeatureRegistry
from cf_h2o.schemas import GraphPosterior, GraphSpec, MechanismSpec, TransitionBatch
from cf_h2o.world_model.causal_factored_residual import CausalFactoredResidualWorldModel


OBS_NAMES = ["waiting", "dwell", "travel_time", "headway"]
ACTION_NAMES = ["holding"]
MECHANISMS = ["demand", "dwell", "speed", "headway", "reward"]


@dataclass
class SyntheticAblationResult:
    uncalibrated_mse: float
    monolithic_linear_mse: float
    wrong_mask_mse: float
    causal_factored_mse: float
    residual_initial_loss: float
    residual_final_loss: float
    wrong_residual_final_loss: float


def run_synthetic_causal_ablation(config: dict[str, Any] | None = None) -> SyntheticAblationResult:
    """Run a small controlled mechanism-transfer ablation.

    The real dynamics are generated as an uncalibrated simulator plus
    mechanism-specific residuals with theta-parent interactions. The correct
    causal-factored residual model receives the right parent sets; the wrong
    mask model receives shuffled parent sets. A monolithic linear residual
    baseline receives all raw variables but no interaction features.
    """

    config = dict(config or {})
    torch.manual_seed(int(config.get("seed", 31)))
    n_train = int(config.get("n_train", 192))
    n_eval = int(config.get("n_eval", 128))
    train = _make_split(n_train, seed=int(config.get("seed", 31)))
    eval_split = _make_split(n_eval, seed=int(config.get("seed", 31)) + 1)

    correct_posterior = _posterior(train.real, correct=True)
    correct_model = _make_model(
        _specs(correct=True),
        correct_posterior,
        config,
    )
    correct_metrics = correct_model.fit_residual_modules(train.real, theta_dict=train.theta)
    corrected_next, corrected_reward = _exact_sim_plus_residual(correct_model, eval_split.real, eval_split.theta)

    wrong_model = _make_model(_specs(correct=False), _posterior(train.real, correct=False), config)
    wrong_metrics = wrong_model.fit_residual_modules(train.real, theta_dict=train.theta)
    wrong_next, wrong_reward = _exact_sim_plus_residual(wrong_model, eval_split.real, eval_split.theta)

    mono_next, mono_reward = _fit_eval_monolithic_linear(train, eval_split, ridge=float(config.get("ridge", 1e-3)))

    sim_next = eval_split.real.metadata["sim_next_observations"]
    sim_reward = eval_split.real.metadata["sim_rewards"]
    return SyntheticAblationResult(
        uncalibrated_mse=_transition_mse(sim_next, sim_reward, eval_split.real.next_observations, eval_split.real.rewards),
        monolithic_linear_mse=_transition_mse(mono_next, mono_reward, eval_split.real.next_observations, eval_split.real.rewards),
        wrong_mask_mse=_transition_mse(wrong_next, wrong_reward, eval_split.real.next_observations, eval_split.real.rewards),
        causal_factored_mse=_transition_mse(corrected_next, corrected_reward, eval_split.real.next_observations, eval_split.real.rewards),
        residual_initial_loss=float(correct_metrics["initial_loss"]),
        residual_final_loss=float(correct_metrics["final_loss"]),
        wrong_residual_final_loss=float(wrong_metrics["final_loss"]),
    )


@dataclass
class _Split:
    sim: TransitionBatch
    real: TransitionBatch
    theta: dict[str, torch.Tensor]


def _make_split(n: int, seed: int) -> _Split:
    generator = torch.Generator().manual_seed(seed)
    waiting = torch.randn(n, generator=generator)
    dwell = torch.randn(n, generator=generator)
    travel_time = torch.randn(n, generator=generator)
    headway = torch.randn(n, generator=generator)
    holding = torch.randn(n, generator=generator)
    noise = lambda scale=0.01: scale * torch.randn(n, generator=generator)
    theta = {
        "demand": torch.randn(n, 1, generator=generator),
        "dwell": torch.randn(n, 1, generator=generator),
        "speed": torch.randn(n, 1, generator=generator),
        "headway": torch.randn(n, 1, generator=generator),
        "reward": torch.randn(n, 1, generator=generator),
    }

    sim_waiting = 0.65 * waiting + 0.05 * holding + noise()
    sim_dwell = 0.55 * dwell + 0.20 * waiting + 0.10 * holding + noise()
    sim_travel = 0.70 * travel_time - 0.10 * holding + noise()
    sim_headway = 0.60 * headway + 0.25 * travel_time - 0.05 * holding + noise()
    sim_reward = -0.55 * waiting - 0.35 * headway - 0.10 * holding + noise()

    demand_res = theta["demand"][:, 0] * (0.45 * waiting + 0.30 * holding)
    dwell_res = theta["dwell"][:, 0] * (0.35 * waiting - 0.25 * dwell + 0.25 * holding)
    speed_res = theta["speed"][:, 0] * (0.40 * travel_time - 0.25 * holding)
    headway_res = theta["headway"][:, 0] * (0.45 * headway + 0.25 * travel_time - 0.20 * holding)
    reward_res = theta["reward"][:, 0] * (-0.35 * waiting - 0.30 * headway + 0.20 * holding)

    observations = torch.stack([waiting, dwell, travel_time, headway], dim=1)
    actions = holding.reshape(-1, 1)
    sim_next = torch.stack([sim_waiting, sim_dwell, sim_travel, sim_headway], dim=1)
    real_next = sim_next + torch.stack([demand_res, dwell_res, speed_res, headway_res], dim=1)
    real_reward = sim_reward + reward_res

    common_metadata = {"obs_names": OBS_NAMES, "action_names": ACTION_NAMES}
    real = TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=real_reward,
        next_observations=real_next,
        dones=torch.zeros(n),
        metadata={
            **common_metadata,
            "sim_next_observations": sim_next,
            "sim_rewards": sim_reward,
        },
    )
    sim = TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=sim_reward,
        next_observations=sim_next,
        dones=torch.zeros(n),
        metadata=common_metadata,
    )
    return _Split(sim=sim, real=real, theta=theta)


def _specs(correct: bool) -> list[MechanismSpec]:
    if correct:
        spec_data = [
            ("demand", ["waiting@t1"], ["waiting@t", "holding@t"]),
            ("dwell", ["dwell@t1"], ["waiting@t", "dwell@t", "holding@t"]),
            ("speed", ["travel_time@t1"], ["travel_time@t", "holding@t"]),
            ("headway", ["headway@t1"], ["headway@t", "travel_time@t", "holding@t"]),
            ("reward", ["reward@t1"], ["waiting@t", "headway@t", "holding@t"]),
        ]
    else:
        spec_data = [
            ("demand", ["waiting@t1"], ["dwell@t", "travel_time@t"]),
            ("dwell", ["dwell@t1"], ["travel_time@t", "headway@t"]),
            ("speed", ["travel_time@t1"], ["waiting@t", "headway@t"]),
            ("headway", ["headway@t1"], ["waiting@t", "dwell@t"]),
            ("reward", ["reward@t1"], ["dwell@t", "travel_time@t"]),
        ]
    return [
        MechanismSpec(
            name=name,
            child_names=children,
            parent_names=parents,
            latent_dim=1,
            output_dim=len(children),
            loss_type="mse",
        )
        for name, children, parents in spec_data
    ]


def _posterior(batch: TransitionBatch, correct: bool) -> GraphPosterior:
    registry = FeatureRegistry.from_transition_dataset(batch)
    hard_mask = registry.build_temporal_hard_mask()
    edge_probs = torch.where(hard_mask, torch.full(hard_mask.shape, 0.02), torch.zeros(hard_mask.shape))
    idx = registry.node_index
    for spec in _specs(correct):
        for parent in spec.parent_names:
            for child in spec.child_names:
                edge_probs[idx[parent], idx[child]] = 1.0
    graph = GraphSpec(
        node_names=registry.node_names,
        adjacency=(edge_probs > 0.8).float(),
        hard_mask=hard_mask,
        edge_probs=edge_probs,
        node_groups=registry.node_groups(),
    )
    return GraphPosterior(graphs=[graph], log_weights=torch.zeros(1), edge_marginals=edge_probs)


def _make_model(
    specs: list[MechanismSpec],
    posterior: GraphPosterior,
    config: dict[str, Any],
) -> CausalFactoredResidualWorldModel:
    return CausalFactoredResidualWorldModel(
        specs,
        posterior,
        {
            "hidden_dim": int(config.get("hidden_dim", 24)),
            "batch_size": int(config.get("batch_size", 64)),
            "train_epochs_residual": int(config.get("train_epochs_residual", 80)),
            "residual_lr": float(config.get("residual_lr", 3e-3)),
            "weight_decay": float(config.get("weight_decay", 0.0)),
        },
    )


def _exact_sim_plus_residual(
    model: CausalFactoredResidualWorldModel,
    batch: TransitionBatch,
    theta: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    sim_next = batch.metadata["sim_next_observations"].clone()
    sim_reward = batch.metadata["sim_rewards"].clone()
    corrected_next = sim_next.clone()
    corrected_reward = sim_reward.clone()
    for spec in model.mechanism_specs:
        parents, mask = model.parent_inputs(batch, spec.name)
        theta_m = theta[spec.name].to(device=parents.device, dtype=parents.dtype)
        residual = model.residual_modules[spec.name](parents, theta=theta_m, mask=mask)["mean"].detach()
        for offset, child_name in enumerate(spec.child_names):
            value = residual[:, offset]
            if child_name == "reward@t1":
                corrected_reward = corrected_reward + value
            else:
                obs_name = child_name[:-3]
                corrected_next[:, OBS_NAMES.index(obs_name)] = corrected_next[:, OBS_NAMES.index(obs_name)] + value
    return corrected_next, corrected_reward


def _fit_eval_monolithic_linear(train: _Split, eval_split: _Split, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = _monolithic_features(train.real, train.theta)
    eval_x = _monolithic_features(eval_split.real, eval_split.theta)
    train_y = torch.cat(
        [
            train.real.next_observations - train.real.metadata["sim_next_observations"],
            (train.real.rewards - train.real.metadata["sim_rewards"]).reshape(-1, 1),
        ],
        dim=1,
    )
    eye = torch.eye(train_x.shape[1], dtype=train_x.dtype, device=train_x.device)
    coeff = torch.linalg.solve(train_x.T @ train_x + ridge * eye, train_x.T @ train_y)
    residual = eval_x @ coeff
    next_obs = eval_split.real.metadata["sim_next_observations"] + residual[:, : len(OBS_NAMES)]
    reward = eval_split.real.metadata["sim_rewards"] + residual[:, len(OBS_NAMES)]
    return next_obs, reward


def _monolithic_features(batch: TransitionBatch, theta: dict[str, torch.Tensor]) -> torch.Tensor:
    theta_values = torch.cat([theta[name] for name in MECHANISMS], dim=1)
    return torch.cat(
        [
            batch.observations,
            batch.actions,
            theta_values,
            torch.ones(batch.batch_size, 1, dtype=batch.observations.dtype, device=batch.device),
        ],
        dim=1,
    )


def _transition_mse(pred_next, pred_reward, target_next, target_reward) -> float:
    next_mse = (pred_next - target_next).pow(2).mean()
    reward_mse = (pred_reward.reshape(-1) - target_reward.reshape(-1)).pow(2).mean()
    return float((next_mse + reward_mse).detach().cpu())
