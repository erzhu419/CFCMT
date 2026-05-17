"""End-to-end synthetic bus mechanism ablation for CF-H2O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cf_h2o.graph.dag_gflownet import DAGGFlowNetDiscoverer
from cf_h2o.graph.feature_registry import FeatureRegistry
from cf_h2o.rl.h2o_mcwm_bridge import H2OFactorTrustBridge
from cf_h2o.schemas import GraphPosterior, GraphSpec, MechanismSpec, TransitionBatch
from cf_h2o.trust.factor_trust import FactorTrustEstimator, FactorWiseTrustWeightProvider
from cf_h2o.trust.weight_composer import WeightComposer
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


@dataclass
class DataEfficiencyPoint:
    n_train: int
    causal_sparse_mse: float
    dense_unfactored_mse: float

    @property
    def causal_to_dense_ratio(self) -> float:
        return self.causal_sparse_mse / max(self.dense_unfactored_mse, 1e-12)


@dataclass
class SyntheticDataEfficiencyResult:
    points: list[DataEfficiencyPoint]
    target_mse: float
    causal_n_at_target: int | None
    dense_n_at_target: int | None
    sample_efficiency_gain: float


@dataclass
class LearnedDAGDataEfficiencyPoint:
    n_train: int
    learned_gflownet_mse: float
    oracle_sparse_mse: float
    dense_unfactored_mse: float
    parent_recall: float
    parent_precision: float
    average_parent_count: float
    initial_tb_loss: float
    final_tb_loss: float
    sample_log_reward_mean: float

    @property
    def learned_to_dense_ratio(self) -> float:
        return self.learned_gflownet_mse / max(self.dense_unfactored_mse, 1e-12)


@dataclass
class LearnedDAGDataEfficiencyResult:
    points: list[LearnedDAGDataEfficiencyPoint]
    target_mse: float
    learned_n_at_target: int | None
    oracle_n_at_target: int | None
    dense_n_at_target: int | None
    sample_efficiency_gain: float


@dataclass
class ImperfectSimH2OOnlineResult:
    uncalibrated_sim_mse: float
    corrected_sim_mse: float
    good_sim_weight_mean: float
    imperfect_sim_weight_mean: float
    bad_sim_weight_mean: float
    bridge_weight_mean: float
    weighted_sim_loss: float
    weight_ess_ratio: float


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


def run_synthetic_data_efficiency_ablation(config: dict[str, Any] | None = None) -> SyntheticDataEfficiencyResult:
    """Measure the sample-efficiency gain from sparse causal mechanisms.

    The causal estimator receives only the known parent-theta interaction terms
    for each mechanism. The dense estimator receives all variables and all
    pairwise interactions jointly, which contains the right terms but buries
    them among many irrelevant couplings.
    """

    config = dict(config or {})
    seed = int(config.get("seed", 41))
    train_sizes = [int(value) for value in config.get("train_sizes", [8, 16, 32, 64, 96])]
    eval_split = _make_split(int(config.get("n_eval", 512)), seed=seed + 1000)
    ridge = float(config.get("ridge", 1.0))
    target_mse = float(config.get("target_mse", 0.02))

    points = []
    for n_train in train_sizes:
        train = _make_split(n_train, seed=seed)
        causal_next, causal_reward = _fit_eval_causal_sparse_ridge(train, eval_split, ridge=ridge)
        dense_next, dense_reward = _fit_eval_dense_unfactored_ridge(train, eval_split, ridge=ridge)
        points.append(
            DataEfficiencyPoint(
                n_train=n_train,
                causal_sparse_mse=_transition_mse(
                    causal_next,
                    causal_reward,
                    eval_split.real.next_observations,
                    eval_split.real.rewards,
                ),
                dense_unfactored_mse=_transition_mse(
                    dense_next,
                    dense_reward,
                    eval_split.real.next_observations,
                    eval_split.real.rewards,
                ),
            )
        )

    causal_n = _first_n_at_target(points, "causal_sparse_mse", target_mse)
    dense_n = _first_n_at_target(points, "dense_unfactored_mse", target_mse)
    if causal_n is None:
        gain = 0.0
    elif dense_n is None:
        gain = float("inf")
    else:
        gain = dense_n / causal_n

    return SyntheticDataEfficiencyResult(
        points=points,
        target_mse=target_mse,
        causal_n_at_target=causal_n,
        dense_n_at_target=dense_n,
        sample_efficiency_gain=float(gain),
    )


def run_learned_dag_data_efficiency_ablation(config: dict[str, Any] | None = None) -> LearnedDAGDataEfficiencyResult:
    """Measure data efficiency when the parent sets come from DAG-GFlowNet."""

    config = dict(config or {})
    seed = int(config.get("seed", 41))
    train_sizes = [int(value) for value in config.get("train_sizes", [16, 32, 64])]
    eval_split = _make_split(int(config.get("n_eval", 512)), seed=seed + 1000)
    ridge = float(config.get("ridge", 1.0))
    target_mse = float(config.get("target_mse", 0.06))
    parent_top_k = int(config.get("parent_top_k", 4))

    points = []
    for n_train in train_sizes:
        train = _make_split(n_train, seed=seed)
        posterior = _learn_gflownet_posterior(train.real, config, seed=seed + 700 + n_train)
        registry = FeatureRegistry.from_transition_dataset(train.real)
        learned_specs = _learned_specs_from_posterior(posterior, registry, parent_top_k=parent_top_k)
        learned_next, learned_reward = _fit_eval_sparse_ridge_with_specs(train, eval_split, learned_specs, ridge)
        oracle_next, oracle_reward = _fit_eval_causal_sparse_ridge(train, eval_split, ridge)
        dense_next, dense_reward = _fit_eval_dense_unfactored_ridge(train, eval_split, ridge)
        recall, precision, avg_parent_count = _parent_recovery_stats(learned_specs)
        train_metrics = posterior.diagnostics.get("train", {})
        points.append(
            LearnedDAGDataEfficiencyPoint(
                n_train=n_train,
                learned_gflownet_mse=_transition_mse(
                    learned_next,
                    learned_reward,
                    eval_split.real.next_observations,
                    eval_split.real.rewards,
                ),
                oracle_sparse_mse=_transition_mse(
                    oracle_next,
                    oracle_reward,
                    eval_split.real.next_observations,
                    eval_split.real.rewards,
                ),
                dense_unfactored_mse=_transition_mse(
                    dense_next,
                    dense_reward,
                    eval_split.real.next_observations,
                    eval_split.real.rewards,
                ),
                parent_recall=recall,
                parent_precision=precision,
                average_parent_count=avg_parent_count,
                initial_tb_loss=float(train_metrics.get("initial_tb_loss", 0.0)),
                final_tb_loss=float(train_metrics.get("final_tb_loss", 0.0)),
                sample_log_reward_mean=float(posterior.diagnostics.get("sample_log_reward_mean", 0.0)),
            )
        )

    learned_n = _first_n_at_target(points, "learned_gflownet_mse", target_mse)
    oracle_n = _first_n_at_target(points, "oracle_sparse_mse", target_mse)
    dense_n = _first_n_at_target(points, "dense_unfactored_mse", target_mse)
    if learned_n is None:
        gain = 0.0
    elif dense_n is None:
        gain = float("inf")
    else:
        gain = dense_n / learned_n

    return LearnedDAGDataEfficiencyResult(
        points=points,
        target_mse=target_mse,
        learned_n_at_target=learned_n,
        oracle_n_at_target=oracle_n,
        dense_n_at_target=dense_n,
        sample_efficiency_gain=float(gain),
    )


def run_imperfect_sim_h2o_online_ablation(config: dict[str, Any] | None = None) -> ImperfectSimH2OOnlineResult:
    """Validate imperfect simulator use through the H2O+ external weight hook."""

    config = dict(config or {})
    seed = int(config.get("seed", 53))
    train = _make_split(int(config.get("n_train", 160)), seed=seed)
    eval_split = _make_split(int(config.get("n_eval", 128)), seed=seed + 1)
    model_config = {
        "hidden_dim": int(config.get("hidden_dim", 24)),
        "batch_size": int(config.get("batch_size", 64)),
        "train_epochs_residual": int(config.get("train_epochs_residual", 70)),
        "residual_lr": float(config.get("residual_lr", 3e-3)),
    }
    model = _make_model(_specs(correct=True), _posterior(train.real, correct=True), model_config)
    model.fit_residual_modules(train.real, theta_dict=train.theta)
    corrected_next, corrected_reward = _exact_sim_plus_residual(model, eval_split.real, eval_split.theta)

    estimator = FactorTrustEstimator(
        MECHANISMS,
        {
            "w_min": float(config.get("w_min", 0.05)),
            "w_max": float(config.get("w_max", 5.0)),
            "residual_scale": float(config.get("residual_scale", 1.4)),
            "graph_uncertainty_scale": 0.0,
        },
    )
    provider = FactorWiseTrustWeightProvider(
        model,
        estimator,
        WeightComposer(mode="geometric_mean", w_min=float(config.get("w_min", 0.05)), w_max=float(config.get("w_max", 5.0))),
        theta_provider=_theta_from_batch_metadata,
        config={"trust_warmup_steps": 0},
    )
    h2o = _SyntheticH2OOnlineHarness(_to_h2o_sim_batch(eval_split, theta_scale=1.0))
    bridge = H2OFactorTrustBridge(h2o, provider, config={"trust_warmup_steps": 0})

    good_weight = bridge.compute_sim_weight(_to_h2o_sim_batch(eval_split, theta_scale=0.25))
    imperfect_weight = bridge.compute_sim_weight(_to_h2o_sim_batch(eval_split, theta_scale=1.0))
    bad_weight = bridge.compute_sim_weight(_to_h2o_sim_batch(eval_split, theta_scale=2.0))
    h2o_metrics = bridge.train_step(batch_size=min(64, eval_split.real.batch_size))

    sim_next = eval_split.real.metadata["sim_next_observations"]
    sim_reward = eval_split.real.metadata["sim_rewards"]
    return ImperfectSimH2OOnlineResult(
        uncalibrated_sim_mse=_transition_mse(sim_next, sim_reward, eval_split.real.next_observations, eval_split.real.rewards),
        corrected_sim_mse=_transition_mse(corrected_next, corrected_reward, eval_split.real.next_observations, eval_split.real.rewards),
        good_sim_weight_mean=float(good_weight.mean().detach().cpu()),
        imperfect_sim_weight_mean=float(imperfect_weight.mean().detach().cpu()),
        bad_sim_weight_mean=float(bad_weight.mean().detach().cpu()),
        bridge_weight_mean=float(h2o_metrics["bridge_weight_mean"]),
        weighted_sim_loss=float(h2o_metrics["weighted_sim_loss"]),
        weight_ess_ratio=float(h2o_metrics["weight_ess_ratio"]),
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


def _fit_eval_causal_sparse_ridge(train: _Split, eval_split: _Split, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    return _fit_eval_sparse_ridge_with_specs(train, eval_split, _specs(correct=True), ridge)


def _fit_eval_sparse_ridge_with_specs(
    train: _Split,
    eval_split: _Split,
    specs: list[MechanismSpec],
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = {}
    train_y = _residual_targets(train)
    for spec in specs:
        child_idx = _child_output_index(spec.name)
        features = _causal_sparse_features(train, spec)
        coefficients[spec.name] = _ridge_solve(features, train_y[:, child_idx : child_idx + 1], ridge)

    residual = torch.zeros(eval_split.real.batch_size, len(OBS_NAMES) + 1, dtype=eval_split.real.observations.dtype)
    for spec in specs:
        child_idx = _child_output_index(spec.name)
        residual[:, child_idx : child_idx + 1] = _causal_sparse_features(eval_split, spec) @ coefficients[spec.name]
    next_obs = eval_split.real.metadata["sim_next_observations"] + residual[:, : len(OBS_NAMES)]
    reward = eval_split.real.metadata["sim_rewards"] + residual[:, len(OBS_NAMES)]
    return next_obs, reward


def _fit_eval_dense_unfactored_ridge(train: _Split, eval_split: _Split, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = _dense_unfactored_features(train)
    eval_x = _dense_unfactored_features(eval_split)
    coeff = _ridge_solve(train_x, _residual_targets(train), ridge)
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


def _residual_targets(split: _Split) -> torch.Tensor:
    return torch.cat(
        [
            split.real.next_observations - split.real.metadata["sim_next_observations"],
            (split.real.rewards - split.real.metadata["sim_rewards"]).reshape(-1, 1),
        ],
        dim=1,
    )


def _causal_sparse_features(split: _Split, spec: MechanismSpec) -> torch.Tensor:
    values = []
    for parent in spec.parent_names:
        parent_value = _parent_value(split, parent).reshape(-1, 1)
        values.append(parent_value * split.theta[spec.name])
    values.append(torch.ones(split.real.batch_size, 1, dtype=split.real.observations.dtype, device=split.real.device))
    return torch.cat(values, dim=1)


def _dense_unfactored_features(split: _Split) -> torch.Tensor:
    batch = split.real
    theta_values = torch.cat([split.theta[name] for name in MECHANISMS], dim=1)
    base = torch.cat([batch.observations, batch.actions, theta_values], dim=1)
    pairwise = []
    for row in range(base.shape[1]):
        for col in range(row, base.shape[1]):
            pairwise.append((base[:, row] * base[:, col]).reshape(-1, 1))
    return torch.cat(
        [
            base,
            *pairwise,
            torch.ones(batch.batch_size, 1, dtype=batch.observations.dtype, device=batch.device),
        ],
        dim=1,
    )


def _ridge_solve(features: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    eye = torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
    return torch.linalg.solve(features.T @ features + float(ridge) * eye, features.T @ targets)


def _learn_gflownet_posterior(batch: TransitionBatch, config: dict[str, Any], seed: int) -> GraphPosterior:
    registry = FeatureRegistry.from_transition_dataset(batch)
    discoverer = DAGGFlowNetDiscoverer(
        {
            "train_steps": int(config.get("gflownet_train_steps", 120)),
            "num_samples": int(config.get("gflownet_num_samples", 24)),
            "max_parents": int(config.get("gflownet_max_parents", 4)),
            "reward_scale": float(config.get("gflownet_reward_scale", 0.002)),
            "max_log_reward": float(config.get("gflownet_max_log_reward", 80.0)),
            "complexity_penalty": float(config.get("gflownet_complexity_penalty", 0.45)),
            "seed": int(seed),
        }
    )
    return discoverer.fit(batch, registry=registry)


def _learned_specs_from_posterior(
    posterior: GraphPosterior,
    registry: FeatureRegistry,
    *,
    parent_top_k: int,
) -> list[MechanismSpec]:
    hard_mask = posterior.graphs[0].hard_mask.to(dtype=torch.bool)
    edge_marginals = posterior.edge_marginals.detach().cpu()
    specs = []
    for oracle_spec in _specs(correct=True):
        child = oracle_spec.child_names[0]
        child_idx = registry.node_index[child]
        allowed = torch.where(hard_mask[:, child_idx])[0]
        if allowed.numel() == 0:
            parents = []
        else:
            probs = edge_marginals[allowed, child_idx]
            order = torch.argsort(probs, descending=True)[: max(1, int(parent_top_k))]
            parents = [registry.node_names[int(allowed[idx])] for idx in order.tolist()]
        specs.append(
            MechanismSpec(
                name=oracle_spec.name,
                child_names=list(oracle_spec.child_names),
                parent_names=parents,
                latent_dim=1,
                output_dim=len(oracle_spec.child_names),
                loss_type="mse",
            )
        )
    return specs


def _parent_recovery_stats(specs: list[MechanismSpec]) -> tuple[float, float, float]:
    true_edges = {(spec.name, parent) for spec in _specs(correct=True) for parent in spec.parent_names}
    learned_edges = {(spec.name, parent) for spec in specs for parent in spec.parent_names}
    matched = true_edges & learned_edges
    recall = len(matched) / max(1, len(true_edges))
    precision = len(matched) / max(1, len(learned_edges))
    avg_parent_count = sum(len(spec.parent_names) for spec in specs) / max(1, len(specs))
    return float(recall), float(precision), float(avg_parent_count)


def _child_output_index(mechanism_name: str) -> int:
    mapping = {"demand": 0, "dwell": 1, "speed": 2, "headway": 3, "reward": 4}
    return mapping[mechanism_name]


def _parent_value(split: _Split, parent_name: str) -> torch.Tensor:
    name = parent_name[:-2] if parent_name.endswith("@t") else parent_name
    if name in OBS_NAMES:
        return split.real.observations[:, OBS_NAMES.index(name)]
    if name in ACTION_NAMES:
        return split.real.actions[:, ACTION_NAMES.index(name)]
    raise KeyError(f"Unknown parent name: {parent_name}")


def _first_n_at_target(points: list[DataEfficiencyPoint], field: str, target_mse: float) -> int | None:
    for point in sorted(points, key=lambda item: item.n_train):
        if float(getattr(point, field)) <= target_mse:
            return point.n_train
    return None


def _to_h2o_sim_batch(split: _Split, theta_scale: float) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "observations": split.real.observations,
        "actions": split.real.actions,
        "rewards": split.real.metadata["sim_rewards"],
        "next_observations": split.real.metadata["sim_next_observations"],
        "dones": split.real.dones,
        "z_t": torch.zeros(split.real.batch_size, 1, dtype=split.real.observations.dtype, device=split.real.device),
        "z_t1": torch.zeros(split.real.batch_size, 1, dtype=split.real.observations.dtype, device=split.real.device),
        "obs_names": OBS_NAMES,
        "action_names": ACTION_NAMES,
        "source": "sim",
        "line_id": ["synthetic-line"] * split.real.batch_size,
        "route_id": ["synthetic-route"] * split.real.batch_size,
    }
    for name in MECHANISMS:
        batch[f"theta_{name}"] = split.theta[name] * float(theta_scale)
    return batch


def _theta_from_batch_metadata(batch: TransitionBatch) -> dict[str, torch.Tensor]:
    return {name: batch.metadata[f"theta_{name}"] for name in MECHANISMS}


class _SyntheticH2OOnlineHarness:
    def __init__(self, sim_batch: dict[str, Any]):
        self.sim_batch = sim_batch
        self._total_steps = 1
        self.external_sim_weight_provider = None

    def train(self, batch_size: int, pretrain_steps: int = 0) -> dict[str, float]:
        del pretrain_steps
        self._total_steps += 1
        if self.external_sim_weight_provider is None:
            raise RuntimeError("external_sim_weight_provider must be installed")
        batch = _slice_h2o_batch(self.sim_batch, batch_size)
        weight = self.external_sim_weight_provider(batch)
        transition_energy = (batch["next_observations"] - batch["observations"]).pow(2).mean(dim=1)
        reward_energy = (batch["rewards"].reshape(-1) - batch["rewards"].reshape(-1).mean()).pow(2)
        proxy_td_error = transition_energy + reward_energy
        ess = (weight.sum() ** 2) / (weight.pow(2).sum() + 1e-8)
        return {
            "bridge_weight_mean": float(weight.mean().detach().cpu()),
            "weighted_sim_loss": float((weight * proxy_td_error).mean().detach().cpu()),
            "weight_ess_ratio": float((ess / max(weight.numel(), 1)).detach().cpu()),
        }


def _slice_h2o_batch(batch: dict[str, Any], batch_size: int) -> dict[str, Any]:
    limit = min(int(batch_size), int(batch["observations"].shape[0]))
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch["observations"].shape[0],):
            sliced[key] = value[:limit]
        elif isinstance(value, list) and len(value) == batch["observations"].shape[0]:
            sliced[key] = value[:limit]
        else:
            sliced[key] = value
    return sliced


def _transition_mse(pred_next, pred_reward, target_next, target_reward) -> float:
    next_mse = (pred_next - target_next).pow(2).mean()
    reward_mse = (pred_reward.reshape(-1) - target_reward.reshape(-1)).pow(2).mean()
    return float((next_mse + reward_mse).detach().cpu())
