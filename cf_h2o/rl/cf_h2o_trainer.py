"""Stage 6 full CF-H2O training orchestrator."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch

from cf_h2o.data.h2o_buffer_adapter import transition_batch_from_h2o
from cf_h2o.latent.factor_encoder import build_history_windows
from cf_h2o.latent.factor_prior import theta_norm_metrics
from cf_h2o.rl.h2o_mcwm_bridge import H2OFactorTrustBridge
from cf_h2o.rl.policy_inputs import build_policy_input
from cf_h2o.schemas import GraphPosterior, TransitionBatch
from cf_h2o.trust.factor_trust import FactorTrustEstimator, FactorWiseTrustWeightProvider
from cf_h2o.trust.weight_composer import WeightComposer


class CFH2OTrainer:
    """Coordinate real/sim/model batches, theta inference, trust, and H2O+ updates."""

    def __init__(
        self,
        h2o_algo: Any,
        world_model: Any,
        factor_encoder: Any,
        graph_posterior: GraphPosterior,
        trust_estimator: FactorTrustEstimator,
        replay_buffer: Any,
        config: dict[str, Any] | None = None,
    ):
        self.h2o = h2o_algo
        self.world_model = world_model
        self.factor_encoder = factor_encoder
        self.graph_posterior = graph_posterior
        self.trust_estimator = trust_estimator
        self.replay_buffer = replay_buffer if replay_buffer is not None else getattr(h2o_algo, "replay_buffer", None)
        self.config = dict(config or {})
        self.training_config = dict(self.config.get("training", {}))
        self.batch_size = int(self.config.get("batch_size", self.training_config.get("batch_size", 256)))
        self.total_steps = 0
        self.last_world_model_metrics: dict[str, Any] = {}
        self.last_model_rollout: TransitionBatch | None = None

        trust_config = dict(self.config.get("trust", {}))
        composer_config = dict(self.config.get("composer", {}))
        composer = WeightComposer(
            mode=composer_config.get("mode", trust_config.get("mode", "geometric_mean")),
            w_min=float(trust_config.get("w_min", 0.05)),
            w_max=float(trust_config.get("w_max", 5.0)),
            reward_path_weights=composer_config.get("reward_path_weights"),
        )
        provider_config = {
            **trust_config,
            "trust_warmup_steps": int(trust_config.get("trust_warmup_steps", self.training_config.get("trust_warmup_steps", 0))),
        }
        self.trust_provider = FactorWiseTrustWeightProvider(
            self.world_model,
            self.trust_estimator,
            composer,
            theta_provider=self.infer_theta,
            config=provider_config,
        )
        if self.h2o is not None:
            self.bridge = H2OFactorTrustBridge(self.h2o, self.trust_provider, self.replay_buffer, provider_config)
        else:
            self.bridge = None

    def pretrain_policy_on_real(self, steps: int) -> list[dict[str, Any]]:
        metrics = []
        pretrain_steps = int(steps) + int(getattr(self.h2o, "_total_steps", 0))
        for _ in range(int(steps)):
            metrics.append(dict(self.h2o.train(self.batch_size, pretrain_steps=pretrain_steps)))
        return metrics

    def train_world_model(self) -> dict[str, Any]:
        sim_batch = self._sample_transition_batch("sim")
        real_batch = self._sample_transition_batch("real")
        theta = self.infer_theta(real_batch)
        metrics: dict[str, Any] = {}
        if hasattr(self.world_model, "fit_sim_modules"):
            sim_metrics = self.world_model.fit_sim_modules(sim_batch)
            metrics.update({f"world_model/sim_{key}": value for key, value in sim_metrics.items() if _metric_scalar(value)})
        if hasattr(self.world_model, "fit_residual_modules"):
            residual_metrics = self.world_model.fit_residual_modules(real_batch, sim_data=sim_batch, theta_dict=theta)
            metrics.update({f"world_model/residual_{key}": value for key, value in residual_metrics.items() if _metric_scalar(value)})
            if "final_loss" in residual_metrics:
                metrics.setdefault("world_model/residual_loss_dwell", float(residual_metrics["final_loss"]))
        self.last_world_model_metrics = metrics
        return metrics

    def train_step(self) -> dict[str, Any]:
        self.total_steps += 1
        real_batch = self._sample_transition_batch("real")
        sim_batch = self._sample_transition_batch("sim")
        theta = self.infer_theta(real_batch)
        metrics: dict[str, Any] = {
            "trainer/step": self.total_steps,
            "graph/entropy": self._graph_entropy(),
        }
        metrics.update(theta_norm_metrics(theta))

        trust_weight = self.trust_provider.compute_weight(_batch_to_dict(sim_batch), step=int(getattr(self.h2o, "_total_steps", self.total_steps)))
        metrics.update(_trust_metrics(self.trust_provider, sim_batch, trust_weight))

        if self._should_generate_model_rollout():
            rollout = self._generate_model_rollout(real_batch, theta)
            self._store_model_rollout(rollout)
            metrics["model_rollout/size"] = rollout.batch_size
            metrics["model_rollout/horizon"] = int(self._current_rollout_horizon())

        h2o_metrics = dict(self.h2o.train(self.batch_size, pretrain_steps=int(self.training_config.get("pretrain_real_steps", 0))))
        metrics.update(h2o_metrics)
        metrics.update({f"h2o/{key}": value for key, value in h2o_metrics.items() if _metric_scalar(value)})

        if self.last_world_model_metrics:
            metrics.update(self.last_world_model_metrics)
        metrics.setdefault("world_model/residual_loss_dwell", self.last_world_model_metrics.get("world_model/residual_final_loss", 0.0))
        return metrics

    def infer_theta(self, batch: TransitionBatch) -> dict[str, torch.Tensor]:
        if self.factor_encoder is None:
            return {}
        features = batch.metadata.get("history_features")
        masks = batch.metadata.get("history_masks")
        if features is None:
            input_dim = int(getattr(self.factor_encoder, "input_dim", batch.observations.shape[1]))
            if input_dim <= batch.observations.shape[1]:
                base = batch.observations[:, :input_dim]
            else:
                pad = batch.observations.new_zeros(batch.batch_size, input_dim - batch.observations.shape[1])
                base = torch.cat([batch.observations, pad], dim=1)
            history_len = int(self.config.get("latent", {}).get("history_len", self.training_config.get("history_len", 8)))
            features, masks = build_history_windows(base, history_len=history_len)
        else:
            features = features.to(device=batch.device, dtype=batch.observations.dtype)
            masks = masks.to(device=batch.device, dtype=batch.observations.dtype) if masks is not None else None
        with torch.no_grad():
            return self.factor_encoder(features, masks)

    def build_policy_input(
        self,
        batch: TransitionBatch,
        *,
        local_graph_embedding: Optional[torch.Tensor] = None,
        theta_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        theta_dict = theta_dict if theta_dict is not None else self.infer_theta(batch)
        policy_config = dict(self.config.get("policy_input", {}))
        return build_policy_input(
            batch.observations,
            obs_names=batch.metadata.get("obs_names") or batch.metadata.get("observation_names"),
            local_graph_embedding=local_graph_embedding if policy_config.get("use_local_graph_embedding", False) else None,
            theta_dict=theta_dict if policy_config.get("use_theta", True) else None,
            selected_theta=policy_config.get("selected_theta"),
            metadata=batch.metadata,
        )

    def _sample_transition_batch(self, scope: str) -> TransitionBatch:
        raw = self.replay_buffer.sample(self.batch_size, scope=scope)
        return transition_batch_from_h2o(raw, source=scope)

    def _should_generate_model_rollout(self) -> bool:
        start = int(self.training_config.get("model_rollout_start_step", 0))
        interval = int(self.training_config.get("model_rollout_interval", 1))
        return self.total_steps >= start and interval > 0 and self.total_steps % interval == 0

    def _current_rollout_horizon(self) -> int:
        return int(self.training_config.get("model_rollout_horizon", 1))

    def _generate_model_rollout(self, init_batch: TransitionBatch, theta: dict[str, torch.Tensor]) -> TransitionBatch:
        policy = self._policy_callable()
        horizon = self._current_rollout_horizon()
        if hasattr(self.world_model, "rollout"):
            try:
                rollout = self.world_model.rollout(init_batch, policy, horizon, lambda _features: theta)
            except TypeError:
                rollout = self.world_model.rollout(init_batch.observations, policy, horizon)
        else:
            actions = policy(init_batch.observations)
            pred = self.world_model.predict(init_batch, theta)
            rollout = TransitionBatch(
                observations=init_batch.observations,
                actions=actions,
                rewards=pred["rewards"],
                next_observations=pred["next_observations"],
                dones=torch.zeros(init_batch.batch_size, device=init_batch.device, dtype=init_batch.observations.dtype),
                source=["model"] * init_batch.batch_size,
                metadata=dict(init_batch.metadata),
            )
        if rollout.source is None:
            rollout.source = ["model"] * rollout.batch_size
        return rollout

    def _policy_callable(self) -> Callable[[torch.Tensor], torch.Tensor]:
        policy = getattr(self.h2o, "policy", None)
        if policy is None:
            act_dim = int(self.config.get("act_dim", 1))
            return lambda obs: obs.new_zeros(obs.shape[0], act_dim)
        if hasattr(policy, "sample_actions"):
            return lambda obs: policy.sample_actions(obs)
        if hasattr(policy, "act"):
            return lambda obs: policy.act(obs)
        return policy

    def _store_model_rollout(self, rollout: TransitionBatch) -> None:
        self.last_model_rollout = rollout
        if hasattr(self.replay_buffer, "add_model_batch"):
            self.replay_buffer.add_model_batch(rollout)
            return
        if hasattr(self.replay_buffer, "append_transition_batch"):
            self.replay_buffer.append_transition_batch(rollout)
            return
        if hasattr(self.replay_buffer, "append_traj"):
            z_t = rollout.z_t if rollout.z_t is not None else rollout.observations.new_zeros(rollout.batch_size, int(self.config.get("z_dim", 1)))
            z_t1 = rollout.z_t1 if rollout.z_t1 is not None else z_t
            self.replay_buffer.append_traj(
                rollout.observations.detach().cpu().numpy(),
                rollout.actions.detach().cpu().numpy(),
                rollout.rewards.detach().cpu().numpy(),
                rollout.next_observations.detach().cpu().numpy(),
                rollout.dones.detach().cpu().numpy(),
                z_t.detach().cpu().numpy(),
                z_t1.detach().cpu().numpy(),
            )
            return
        if hasattr(self.replay_buffer, "append"):
            z_dim = int(self.config.get("z_dim", 1))
            for idx in range(rollout.batch_size):
                z_t = rollout.observations.new_zeros(z_dim)
                self.replay_buffer.append(
                    rollout.observations[idx].detach().cpu().numpy(),
                    rollout.actions[idx].detach().cpu().numpy(),
                    float(rollout.rewards[idx].detach().cpu()),
                    rollout.next_observations[idx].detach().cpu().numpy(),
                    float(rollout.dones[idx].detach().cpu()),
                    z_t.detach().cpu().numpy(),
                    z_t.detach().cpu().numpy(),
                )

    def _graph_entropy(self) -> float:
        edge_probs = self.graph_posterior.edge_marginals.detach().clamp(1e-8, 1.0 - 1e-8)
        entropy = -(edge_probs * torch.log(edge_probs) + (1.0 - edge_probs) * torch.log(1.0 - edge_probs))
        if self.graph_posterior.graphs:
            mask = self.graph_posterior.graphs[0].hard_mask.to(device=edge_probs.device)
            if mask.any():
                entropy = entropy[mask]
        return float(entropy.mean().cpu())


def _batch_to_dict(batch: TransitionBatch) -> dict[str, Any]:
    return {
        "observations": batch.observations,
        "actions": batch.actions,
        "rewards": batch.rewards,
        "next_observations": batch.next_observations,
        "dones": batch.dones,
        "z_t": batch.z_t,
        "z_t1": batch.z_t1,
        "source": batch.source,
        **batch.metadata,
    }


def _trust_metrics(provider: FactorWiseTrustWeightProvider, sim_batch: TransitionBatch, weight: torch.Tensor) -> dict[str, float]:
    theta = provider.theta_provider(sim_batch) if provider.theta_provider is not None else None
    pred = provider.world_model.predict(sim_batch, theta)
    trust = provider.trust_estimator(
        pred["mechanism_outputs"],
        mechanism_uncertainty=pred.get("mechanism_uncertainty"),
    )
    metrics = {
        "trust/total_mean": float(weight.detach().mean().cpu()),
        "trust/total_min": float(weight.detach().min().cpu()),
        "trust/total_max": float(weight.detach().max().cpu()),
    }
    for name, value in trust.items():
        metrics[f"trust/{name}_mean"] = float(value.detach().mean().cpu())
    return metrics


def _metric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool))
