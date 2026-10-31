"""Causal-factored residual world model for Stage 4."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from cf_h2o.schemas import GraphPosterior, GraphSpec, MechanismSpec, TransitionBatch
from cf_h2o.world_model.mechanism_modules import MechanismModule


FORBIDDEN_PARENT_MARKERS = ("@t1", "reward@t1", "source", "domain", "city", "real_sim", "label")


class CausalFactoredResidualWorldModel(nn.Module):
    """Mechanism-level simulator model plus residual adapter.

    Base modules learn the uncalibrated simulator transition. Residual modules
    learn real-minus-sim corrections and are conditioned on mechanism theta.
    """

    def __init__(
        self,
        mechanism_specs: list[MechanismSpec | Mapping[str, Any]],
        graph_posterior: GraphPosterior,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.config = dict(config or {})
        self.graph_posterior = graph_posterior
        self.node_names = _posterior_node_names(graph_posterior)
        self.node_index = {name: idx for idx, name in enumerate(self.node_names)}
        self.mechanism_specs = [self._normalize_spec(spec) for spec in mechanism_specs]
        self.hidden_dim = int(self.config.get("hidden_dim", 128))
        self.base_modules = nn.ModuleDict()
        self.residual_modules = nn.ModuleDict()
        self._spec_by_name: dict[str, MechanismSpec] = {}

        for spec in self.mechanism_specs:
            self._validate_spec(spec)
            self._spec_by_name[spec.name] = spec
            self.base_modules[spec.name] = MechanismModule(
                parent_dim=len(spec.parent_names),
                theta_dim=0,
                output_dim=len(spec.child_names),
                hidden_dim=self.hidden_dim,
            )
            self.residual_modules[spec.name] = MechanismModule(
                parent_dim=len(spec.parent_names),
                theta_dim=int(spec.latent_dim),
                output_dim=len(spec.child_names),
                hidden_dim=self.hidden_dim,
                zero_init_output=True,
            )

    def fit_sim_modules(self, sim_data: TransitionBatch) -> dict[str, Any]:
        """Train base mechanism modules on simulator transitions."""

        return self._train_modules(
            sim_data,
            self.base_modules,
            target_mode="target",
            epochs=int(self.config.get("train_epochs_sim", 100)),
            lr=float(self.config.get("lr", 1e-3)),
            batch_size=int(self.config.get("batch_size", 256)),
            theta_dict=None,
        )

    def fit_residual_modules(
        self,
        real_data: TransitionBatch,
        sim_data: TransitionBatch | None = None,
        theta_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, Any]:
        """Train residual modules with paired or nearest-neighbor aligned data."""

        paired = self._paired_sim_targets(real_data)
        mode = "paired"
        alignment_info: dict[str, Any] = {}
        if paired is None:
            if sim_data is None:
                return {
                    "residual_trained": False,
                    "reason": "missing paired sim targets and sim_data for alignment",
                }
            paired, alignment_info = self._aligned_sim_targets(real_data, sim_data)
            mode = "aligned"

        train_data = real_data
        train_theta = theta_dict
        keep_mask = paired.pop("_keep_mask", None)
        if keep_mask is not None:
            keep_idx = torch.where(keep_mask.to(device=real_data.device))[0]
            train_data = _index_batch(real_data, keep_idx)
            paired["sim_next_observations"] = paired["sim_next_observations"][keep_idx]
            paired["sim_rewards"] = paired["sim_rewards"][keep_idx]
            train_theta = _index_theta_dict(theta_dict, keep_idx)

        metrics = self._train_modules(
            train_data,
            self.residual_modules,
            target_mode="residual",
            override_next_observations=paired["sim_next_observations"],
            override_rewards=paired["sim_rewards"],
            epochs=int(self.config.get("train_epochs_residual", 100)),
            lr=float(self.config.get("residual_lr", self.config.get("lr", 1e-3))),
            batch_size=int(self.config.get("batch_size", 256)),
            theta_dict=train_theta,
        )
        metrics.update({"residual_trained": True, "mode": mode})
        metrics.update(alignment_info)
        return metrics

    def predict(
        self,
        batch: TransitionBatch,
        theta_dict: Optional[dict[str, torch.Tensor]] = None,
        graph_sample: GraphSpec | None = None,
    ) -> dict[str, Any]:
        """Predict next observations/reward using base + residual mechanisms."""

        node_values = _batch_node_values(batch)
        obs_names = _obs_names(batch)
        batch_size = batch.batch_size
        next_obs = batch.observations.new_zeros(batch_size, batch.observations.shape[1])
        reward = batch.rewards.new_zeros(batch_size)
        mechanism_outputs: dict[str, dict[str, torch.Tensor]] = {}
        mechanism_uncertainty: dict[str, torch.Tensor] = {}

        for spec in self.mechanism_specs:
            parents, mask = self.parent_inputs(batch, spec.name, graph_sample=graph_sample, node_values=node_values)
            theta = self._theta_for(theta_dict, spec.name, batch_size, parents.device, parents.dtype)
            base = self.base_modules[spec.name](parents, mask=mask)
            residual = self.residual_modules[spec.name](parents, theta=theta, mask=mask)
            mean = base["mean"] + residual["mean"]
            mechanism_outputs[spec.name] = {
                "base": base["mean"],
                "residual": residual["mean"],
                "mean": mean,
                "parent_mask": mask.detach(),
                "parent_names": list(spec.parent_names),
                "child_names": list(spec.child_names),
            }
            mechanism_uncertainty[spec.name] = base["uncertainty"] + residual["uncertainty"]
            for child_offset, child_name in enumerate(spec.child_names):
                child_value = mean[:, child_offset]
                if child_name == "reward@t1":
                    reward = child_value
                elif child_name.endswith("@t1"):
                    obs_name = child_name[:-3]
                    if obs_name in obs_names:
                        next_obs[:, obs_names.index(obs_name)] = child_value

        return {
            "next_observations": next_obs,
            "rewards": reward.reshape(-1),
            "reward": reward.reshape(-1),
            "mechanism_outputs": mechanism_outputs,
            "mechanism_uncertainty": mechanism_uncertainty,
        }

    def rollout(
        self,
        init_batch: TransitionBatch,
        policy: Callable[[torch.Tensor], torch.Tensor],
        horizon: int,
        theta_encoder: Optional[Callable[[torch.Tensor], dict[str, torch.Tensor]]] = None,
    ) -> TransitionBatch:
        """Generate short model rollouts from the factored predictor."""

        observations = init_batch.observations
        obs_items = []
        action_items = []
        reward_items = []
        next_obs_items = []
        done_items = []
        sources = []
        for _ in range(int(horizon)):
            with torch.no_grad():
                actions = policy(observations)
                if isinstance(actions, tuple):
                    actions = actions[0]
                step_batch = TransitionBatch(
                    observations=observations,
                    actions=actions,
                    rewards=torch.zeros(observations.shape[0], device=observations.device, dtype=observations.dtype),
                    next_observations=observations,
                    dones=torch.zeros(observations.shape[0], device=observations.device, dtype=observations.dtype),
                    metadata=dict(init_batch.metadata),
                )
                theta = theta_encoder(observations.unsqueeze(1)) if theta_encoder is not None else None
                pred = self.predict(step_batch, theta)
            obs_items.append(observations)
            action_items.append(actions)
            reward_items.append(pred["rewards"])
            next_obs_items.append(pred["next_observations"])
            done_items.append(torch.zeros_like(pred["rewards"]))
            sources.extend(["model"] * observations.shape[0])
            observations = pred["next_observations"].detach()

        return TransitionBatch(
            observations=torch.cat(obs_items, dim=0),
            actions=torch.cat(action_items, dim=0),
            rewards=torch.cat(reward_items, dim=0),
            next_observations=torch.cat(next_obs_items, dim=0),
            dones=torch.cat(done_items, dim=0),
            source=sources,
            metadata=dict(init_batch.metadata),
        )

    def parent_inputs(
        self,
        batch: TransitionBatch,
        mechanism_name: str,
        *,
        graph_sample: GraphSpec | None = None,
        node_values: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parent matrix and soft parent mask for one mechanism."""

        spec = self._spec_by_name[mechanism_name]
        node_values = node_values or _batch_node_values(batch)
        missing = [name for name in spec.parent_names if name not in node_values]
        if missing:
            raise KeyError(f"Missing parent node values for {mechanism_name}: {missing}")
        parents = torch.cat([node_values[name] for name in spec.parent_names], dim=1)
        mask = self.parent_mask(mechanism_name, graph_sample=graph_sample).to(device=parents.device, dtype=parents.dtype)
        return parents, mask

    def parent_mask(self, mechanism_name: str, graph_sample: GraphSpec | None = None) -> torch.Tensor:
        """Soft mask over configured parent names, averaged over child nodes."""

        spec = self._spec_by_name[mechanism_name]
        edge_matrix = graph_sample.adjacency if graph_sample is not None else self.graph_posterior.edge_marginals
        edge_matrix = edge_matrix.detach().cpu()
        mask_values = []
        for parent in spec.parent_names:
            if parent not in self.node_index:
                mask_values.append(1.0)
                continue
            parent_idx = self.node_index[parent]
            child_probs = []
            for child in spec.child_names:
                if child in self.node_index:
                    child_probs.append(float(edge_matrix[parent_idx, self.node_index[child]]))
            mask_values.append(max(child_probs) if child_probs else 1.0)
        return torch.tensor(mask_values, dtype=torch.float32)

    def _train_modules(
        self,
        data: TransitionBatch,
        modules: nn.ModuleDict,
        *,
        target_mode: str,
        epochs: int,
        lr: float,
        batch_size: int,
        theta_dict: Optional[dict[str, torch.Tensor]],
        override_next_observations: torch.Tensor | None = None,
        override_rewards: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(modules.parameters(), lr=lr, weight_decay=float(self.config.get("weight_decay", 0.0)))
        indices = torch.arange(data.batch_size, device=data.device)
        epoch_losses: list[float] = []
        for _ in range(max(1, int(epochs))):
            perm = indices[torch.randperm(indices.numel(), device=indices.device)]
            running = []
            for start in range(0, data.batch_size, max(1, int(batch_size))):
                batch_idx = perm[start : start + max(1, int(batch_size))]
                sub_data = _index_batch(data, batch_idx)
                sub_override_next = override_next_observations[batch_idx] if override_next_observations is not None else None
                sub_override_rewards = override_rewards[batch_idx] if override_rewards is not None else None
                node_values = _batch_node_values(sub_data)
                override_values = (
                    _batch_node_values(
                        sub_data,
                        override_next_observations=sub_override_next,
                        override_rewards=sub_override_rewards,
                    )
                    if target_mode == "residual"
                    else None
                )
                optimizer.zero_grad()
                losses = []
                for spec in self.mechanism_specs:
                    parents, mask = self.parent_inputs(sub_data, spec.name, node_values=node_values)
                    theta = self._theta_for_indexed(theta_dict, spec.name, batch_idx, parents.device, parents.dtype)
                    pred = modules[spec.name](
                        parents,
                        theta=theta if modules is self.residual_modules else None,
                        mask=mask,
                    )
                    target = _target_for_spec(spec, node_values)
                    if target_mode == "residual":
                        if override_values is None:
                            raise RuntimeError("residual target mode requires override values")
                        target = target - _target_for_spec(spec, override_values)
                    losses.append(modules[spec.name].loss(pred, target, spec.loss_type))
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modules.parameters(), float(self.config.get("grad_clip_norm", 10.0)))
                optimizer.step()
                running.append(float(loss.detach().cpu()))
            epoch_losses.append(sum(running) / max(1, len(running)))
        return {
            "initial_loss": epoch_losses[0],
            "final_loss": epoch_losses[-1],
            "loss_history": epoch_losses,
            "epochs": max(1, int(epochs)),
        }

    def _paired_sim_targets(self, real_data: TransitionBatch) -> Optional[dict[str, torch.Tensor]]:
        sim_next = real_data.metadata.get("sim_next_observations")
        sim_rewards = real_data.metadata.get("sim_rewards")
        if sim_next is None or sim_rewards is None:
            return None
        sim_next = _as_tensor_like(sim_next, real_data.next_observations)
        sim_rewards = _as_tensor_like(sim_rewards, real_data.rewards).reshape(-1)
        return {"sim_next_observations": sim_next, "sim_rewards": sim_rewards}

    def _aligned_sim_targets(self, real_data: TransitionBatch, sim_data: TransitionBatch) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        real_raw = _alignment_features(real_data)
        sim_raw = _alignment_features(sim_data).to(device=real_raw.device)
        real_features, sim_features = _standardize_pair_features(real_raw, sim_raw)
        distances = torch.cdist(real_features, sim_features)
        min_dist, nn_idx = distances.min(dim=1)
        max_distance = float(self.config.get("alignment_max_distance", float("inf")))
        keep = min_dist <= max_distance
        if not keep.any():
            quantile = float(self.config.get("alignment_keep_quantile", 0.5))
            threshold = torch.quantile(min_dist, quantile)
            keep = min_dist <= threshold
        sim_next = sim_data.next_observations[nn_idx].to(device=real_data.device, dtype=real_data.observations.dtype)
        sim_rewards = sim_data.rewards[nn_idx].reshape(-1).to(device=real_data.device, dtype=real_data.rewards.dtype)
        return (
            {"sim_next_observations": sim_next, "sim_rewards": sim_rewards, "_keep_mask": keep.detach()},
            {
                "alignment_distance_mean": float(min_dist.mean().detach().cpu()),
                "alignment_keep_ratio": float(keep.float().mean().detach().cpu()),
            },
        )

    def _theta_for(
        self,
        theta_dict: Optional[dict[str, torch.Tensor]],
        mechanism_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        spec = self._spec_by_name[mechanism_name]
        if int(spec.latent_dim) <= 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if theta_dict is None or mechanism_name not in theta_dict:
            return torch.zeros(batch_size, int(spec.latent_dim), device=device, dtype=dtype)
        theta = theta_dict[mechanism_name].to(device=device, dtype=dtype)
        if theta.shape != (batch_size, int(spec.latent_dim)):
            raise ValueError(f"theta[{mechanism_name}] must be {(batch_size, int(spec.latent_dim))}, got {tuple(theta.shape)}")
        return theta

    def _theta_for_indexed(
        self,
        theta_dict: Optional[dict[str, torch.Tensor]],
        mechanism_name: str,
        batch_idx: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        spec = self._spec_by_name[mechanism_name]
        if int(spec.latent_dim) <= 0:
            return torch.zeros(batch_idx.numel(), 0, device=device, dtype=dtype)
        if theta_dict is None or mechanism_name not in theta_dict:
            return torch.zeros(batch_idx.numel(), int(spec.latent_dim), device=device, dtype=dtype)
        return theta_dict[mechanism_name][batch_idx].to(device=device, dtype=dtype)

    def _normalize_spec(self, spec: MechanismSpec | Mapping[str, Any]) -> MechanismSpec:
        if isinstance(spec, MechanismSpec):
            data = asdict(spec)
        elif is_dataclass(spec):
            data = asdict(spec)
        else:
            data = dict(spec)
        child_names = list(data["child_names"])
        parent_names = list(data.get("parent_names", []))
        if not parent_names:
            parent_names = self._infer_parent_names(child_names)
        return MechanismSpec(
            name=str(data["name"]),
            child_names=child_names,
            parent_names=parent_names,
            latent_dim=int(data.get("latent_dim", 0)),
            output_dim=len(child_names),
            loss_type=str(data.get("loss_type", "mse")),
        )

    def _infer_parent_names(self, child_names: list[str]) -> list[str]:
        parent_indices = set()
        threshold = float(self.config.get("parent_threshold", 0.3))
        for child in child_names:
            if child not in self.node_index:
                continue
            child_idx = self.node_index[child]
            parent_idx = torch.where(self.graph_posterior.edge_marginals[:, child_idx].detach().cpu() >= threshold)[0].tolist()
            parent_indices.update(parent_idx)
        return [
            self.node_names[idx]
            for idx in sorted(parent_indices)
            if self.node_names[idx].endswith("@t") and not _is_forbidden_parent(self.node_names[idx])
        ]

    def _validate_spec(self, spec: MechanismSpec) -> None:
        if len(spec.child_names) != int(spec.output_dim):
            raise ValueError(f"{spec.name}: output_dim must match child_names length")
        for child in spec.child_names:
            if child not in self.node_index:
                raise KeyError(f"{spec.name}: child node {child!r} is not in graph posterior")
            if not child.endswith("@t1"):
                raise ValueError(f"{spec.name}: child node must be @t1, got {child!r}")
        for parent in spec.parent_names:
            if _is_forbidden_parent(parent) or not parent.endswith("@t"):
                raise ValueError(f"{spec.name}: forbidden or non-present parent {parent!r}")
            if parent not in self.node_index:
                raise KeyError(f"{spec.name}: parent node {parent!r} is not in graph posterior")


def _posterior_node_names(posterior: GraphPosterior) -> list[str]:
    if posterior.graphs:
        return list(posterior.graphs[0].node_names)
    return list(posterior.diagnostics.get("node_names", []))


def _is_forbidden_parent(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in FORBIDDEN_PARENT_MARKERS)


def _obs_names(batch: TransitionBatch) -> list[str]:
    names = batch.metadata.get("obs_names") or batch.metadata.get("observation_names")
    return list(names) if names is not None else [f"obs_{idx}" for idx in range(int(batch.observations.shape[1]))]


def _action_names(batch: TransitionBatch) -> list[str]:
    names = batch.metadata.get("action_names")
    act_dim = int(batch.actions.shape[1]) if batch.actions.ndim > 1 else 1
    return list(names) if names is not None else [f"action_{idx}" for idx in range(act_dim)]


def _z_names(batch: TransitionBatch) -> list[str]:
    names = batch.metadata.get("z_names")
    if batch.z_t is None:
        return []
    return list(names) if names is not None else [f"z_{idx}" for idx in range(int(batch.z_t.shape[1]))]


def _batch_node_values(
    batch: TransitionBatch,
    *,
    override_next_observations: torch.Tensor | None = None,
    override_rewards: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    obs = batch.observations
    next_obs = override_next_observations if override_next_observations is not None else batch.next_observations
    rewards = override_rewards if override_rewards is not None else batch.rewards
    actions = batch.actions if batch.actions.ndim > 1 else batch.actions.reshape(-1, 1)
    values: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(_obs_names(batch)):
        values[f"{name}@t"] = obs[:, idx : idx + 1]
        values[f"{name}@t1"] = next_obs[:, idx : idx + 1].to(device=obs.device, dtype=obs.dtype)
    for idx, name in enumerate(_action_names(batch)):
        values[f"{name}@t"] = actions[:, idx : idx + 1]
    values["reward@t1"] = rewards.reshape(-1, 1).to(device=obs.device, dtype=obs.dtype)
    if batch.z_t is not None:
        z_t1 = batch.z_t1 if batch.z_t1 is not None else torch.zeros_like(batch.z_t)
        for idx, name in enumerate(_z_names(batch)):
            values[f"{name}@t"] = batch.z_t[:, idx : idx + 1]
            values[f"{name}@t1"] = z_t1[:, idx : idx + 1]
    if batch.global_time is not None:
        values["global_time@t"] = batch.global_time.reshape(-1, 1).to(device=obs.device, dtype=obs.dtype)
    return values


def _target_for_spec(spec: MechanismSpec, node_values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([node_values[name] for name in spec.child_names], dim=1)


def _index_batch(batch: TransitionBatch, indices: torch.Tensor) -> TransitionBatch:
    def idx(value):
        return value[indices] if isinstance(value, torch.Tensor) else value

    return TransitionBatch(
        observations=idx(batch.observations),
        actions=idx(batch.actions),
        rewards=idx(batch.rewards),
        next_observations=idx(batch.next_observations),
        dones=idx(batch.dones),
        z_t=idx(batch.z_t),
        z_t1=idx(batch.z_t1),
        domain_id=idx(batch.domain_id),
        line_id=batch.line_id,
        route_id=batch.route_id,
        global_time=idx(batch.global_time),
        snapshot_t=batch.snapshot_t,
        snapshot_t1=batch.snapshot_t1,
        source=batch.source,
        metadata=dict(batch.metadata),
    )


def _index_theta_dict(theta_dict: Optional[dict[str, torch.Tensor]], indices: torch.Tensor) -> Optional[dict[str, torch.Tensor]]:
    if theta_dict is None:
        return None
    return {name: theta[indices] for name, theta in theta_dict.items()}


def _as_tensor_like(value: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _alignment_features(batch: TransitionBatch) -> torch.Tensor:
    actions = batch.actions if batch.actions.ndim > 1 else batch.actions.reshape(-1, 1)
    items = [batch.observations, actions]
    if batch.z_t is not None:
        items.append(batch.z_t)
    return torch.cat([item.to(dtype=batch.observations.dtype, device=batch.observations.device) for item in items], dim=1)


def _standardize_pair_features(real: torch.Tensor, sim: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if sim is None:
        mean = real.mean(dim=0, keepdim=True)
        std = real.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (real - mean) / std, None
    combined = torch.cat([real, sim], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (real - mean) / std, (sim - mean) / std
