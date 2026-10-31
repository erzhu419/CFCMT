"""Factor-wise simulator/model trust estimation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from cf_h2o.data.h2o_buffer_adapter import transition_batch_from_h2o
from cf_h2o.schemas import TransitionBatch
from cf_h2o.trust.weight_composer import WeightComposer


class FactorTrustEstimator(nn.Module):
    """Estimate one trust vector per causal mechanism."""

    def __init__(self, mechanism_names: list[str], config: dict[str, Any] | None = None):
        super().__init__()
        self.mechanism_names = list(mechanism_names)
        self.config = dict(config or {})
        self.w_min = float(self.config.get("w_min", 0.05))
        self.w_max = float(self.config.get("w_max", 5.0))
        self.residual_scale = float(self.config.get("residual_scale", 1.0))
        self.uncertainty_scale = float(self.config.get("uncertainty_scale", 1.0))
        self.graph_uncertainty_scale = float(self.config.get("graph_uncertainty_scale", 1.0))
        self.alignment_scale = float(self.config.get("alignment_scale", 1.0))
        self.target_error_scale = float(self.config.get("target_error_scale", 1.0))
        self.horizon_decay = float(self.config.get("horizon_decay", 0.95))

    def forward(
        self,
        mechanism_outputs: dict[str, Mapping[str, Any]],
        real_targets: dict[str, torch.Tensor] | None = None,
        *,
        mechanism_uncertainty: dict[str, torch.Tensor] | None = None,
        alignment_quality: dict[str, torch.Tensor] | torch.Tensor | None = None,
        qdelta_weight: dict[str, torch.Tensor] | torch.Tensor | None = None,
        rollout_horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Return ``{mechanism_name: trust [B]}``."""

        trust: dict[str, torch.Tensor] = {}
        for name in self.mechanism_names:
            if name not in mechanism_outputs:
                continue
            output = mechanism_outputs[name]
            reference = _reference_tensor(output)
            batch_size = reference.shape[0]
            weight = torch.ones(batch_size, device=reference.device, dtype=reference.dtype)

            residual = _optional_tensor(output.get("residual"), reference)
            if residual is not None:
                residual_mag = _row_mean_square(residual).sqrt()
                weight = weight * torch.exp(-self.residual_scale * residual_mag)

            uncertainty = _mechanism_value(name, mechanism_uncertainty, output.get("uncertainty"), reference)
            if uncertainty is not None:
                weight = weight * torch.exp(-self.uncertainty_scale * uncertainty.reshape(-1).clamp_min(0.0))

            graph_uncertainty = _graph_uncertainty(output, reference)
            weight = weight * torch.exp(-self.graph_uncertainty_scale * graph_uncertainty)

            alignment = _mechanism_value(name, alignment_quality, output.get("alignment_distance"), reference)
            if alignment is not None:
                weight = weight * torch.exp(-self.alignment_scale * alignment.reshape(-1).clamp_min(0.0))

            qdelta = _mechanism_value(name, qdelta_weight, output.get("qdelta_weight"), reference)
            if qdelta is not None:
                weight = weight * qdelta.reshape(-1).clamp_min(0.0)

            if real_targets is not None:
                target = _target_for_output(name, output, real_targets, reference)
                if target is not None:
                    pred = _optional_tensor(output.get("mean"), reference)
                    if pred is not None:
                        error = _row_mean_square(pred - target)
                        weight = weight * torch.exp(-self.target_error_scale * error)

            if int(rollout_horizon) > 1:
                weight = weight * (self.horizon_decay ** (int(rollout_horizon) - 1))

            trust[name] = torch.clamp(
                torch.nan_to_num(weight, nan=self.w_min, posinf=self.w_max, neginf=self.w_min),
                self.w_min,
                self.w_max,
            )
        return trust


class FactorWiseTrustWeightProvider:
    """Callable provider installed into ``H2OPlusBus.compute_sim_weight``."""

    def __init__(
        self,
        world_model: Any,
        trust_estimator: FactorTrustEstimator,
        composer: WeightComposer | None = None,
        *,
        theta_provider: Optional[Callable[[TransitionBatch], dict[str, torch.Tensor]]] = None,
        qdelta_adapter: Any = None,
        config: dict[str, Any] | None = None,
    ):
        self.world_model = world_model
        self.trust_estimator = trust_estimator
        self.composer = composer or WeightComposer()
        self.theta_provider = theta_provider
        self.qdelta_adapter = qdelta_adapter
        self.config = dict(config or {})

    def compute_weight(
        self,
        sim_batch: Mapping[str, Any],
        *,
        step: int = 0,
        rollout_horizon: int | None = None,
    ) -> torch.Tensor:
        observations = sim_batch["observations"]
        if int(step) < int(self.config.get("trust_warmup_steps", 0)):
            return torch.ones(observations.shape[0], dtype=observations.dtype, device=observations.device)

        batch = transition_batch_from_h2o(sim_batch, source=sim_batch.get("source", "sim"))
        theta = self.theta_provider(batch) if self.theta_provider is not None else None
        pred = self.world_model.predict(batch, theta)
        qdelta_weight = None
        if self.qdelta_adapter is not None:
            qdelta_weight = self.qdelta_adapter.weight(batch.observations, batch.actions)
        trust = self.trust_estimator(
            pred["mechanism_outputs"],
            mechanism_uncertainty=pred.get("mechanism_uncertainty"),
            qdelta_weight=qdelta_weight,
            rollout_horizon=int(rollout_horizon if rollout_horizon is not None else self.config.get("rollout_horizon", 1)),
        )
        weight = self.composer.compose(trust)
        if not bool(self.config.get("joint_train_trust", False)):
            weight = weight.detach()
        return weight.reshape(-1).to(device=observations.device, dtype=observations.dtype)

    def __call__(self, sim_batch: Mapping[str, Any]) -> torch.Tensor:
        return self.compute_weight(sim_batch)


def _reference_tensor(output: Mapping[str, Any]) -> torch.Tensor:
    for key in ("mean", "residual", "base"):
        value = output.get(key)
        if torch.is_tensor(value):
            return value
    raise KeyError("mechanism output must contain at least one tensor among mean/residual/base")


def _optional_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _row_mean_square(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.pow(2)
    return value.pow(2).mean(dim=tuple(range(1, value.ndim)))


def _graph_uncertainty(output: Mapping[str, Any], reference: torch.Tensor) -> torch.Tensor:
    mask = output.get("parent_mask")
    if mask is None:
        return reference.new_zeros(reference.shape[0])
    mask = _optional_tensor(mask, reference).clamp(0.0, 1.0)
    entropy = -(mask * torch.log(mask.clamp_min(1e-8)) + (1.0 - mask) * torch.log((1.0 - mask).clamp_min(1e-8)))
    max_entropy = torch.log(reference.new_tensor(2.0))
    norm_entropy = (entropy / max_entropy).mean()
    return reference.new_full((reference.shape[0],), float(norm_entropy.detach().cpu()))


def _mechanism_value(
    name: str,
    values: dict[str, torch.Tensor] | torch.Tensor | None,
    fallback: Any,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if values is None:
        return _optional_tensor(fallback, reference)
    if isinstance(values, Mapping):
        if name not in values:
            return _optional_tensor(fallback, reference)
        return _optional_tensor(values[name], reference)
    return _optional_tensor(values, reference)


def _target_for_output(
    name: str,
    output: Mapping[str, Any],
    real_targets: dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if name in real_targets:
        return _optional_tensor(real_targets[name], reference)
    child_names = output.get("child_names") or []
    items = []
    for child in child_names:
        if child in real_targets:
            items.append(_optional_tensor(real_targets[child], reference).reshape(reference.shape[0], -1))
    if items:
        return torch.cat(items, dim=1)
    return None

