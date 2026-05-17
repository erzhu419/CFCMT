"""Policy input construction with shortcut guards."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from cf_h2o.schemas import FeatureTensor


FORBIDDEN_POLICY_FIELD_MARKERS = (
    "source",
    "real_sim",
    "domain_id",
    "city_id",
    "snapshot_t1",
    "reward@t1",
    "reward_t1",
    "next_observation",
    "next_state",
    "future",
)


def build_policy_input(
    observations: torch.Tensor,
    *,
    obs_names: Optional[list[str]] = None,
    local_graph_embedding: Optional[torch.Tensor] = None,
    theta_dict: Optional[dict[str, torch.Tensor]] = None,
    selected_theta: Optional[list[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FeatureTensor:
    """Build policy features from allowed present-time inputs only."""

    if observations.ndim != 2:
        raise ValueError(f"observations must be [B, D], got {tuple(observations.shape)}")
    metadata = dict(metadata or {})
    _raise_on_forbidden_explicit_inputs(metadata)

    batch_size = observations.shape[0]
    names = list(obs_names) if obs_names is not None else [f"obs_{idx}" for idx in range(observations.shape[1])]
    if len(names) != observations.shape[1]:
        raise ValueError("obs_names length must match observation dim")
    groups = {"observations": list(range(len(names)))}
    tensors = [observations]
    feature_names = [f"obs/{name}" for name in names]

    if local_graph_embedding is not None:
        local_graph_embedding = _check_batch(local_graph_embedding, batch_size, "local_graph_embedding").to(
            device=observations.device,
            dtype=observations.dtype,
        )
        start = len(feature_names)
        feature_names.extend([f"local_graph/{idx}" for idx in range(local_graph_embedding.shape[1])])
        groups["local_graph"] = list(range(start, len(feature_names)))
        tensors.append(local_graph_embedding)

    if theta_dict is not None:
        theta_names = selected_theta if selected_theta is not None else sorted(theta_dict)
        start = len(feature_names)
        theta_tensors = []
        for mechanism in theta_names:
            if mechanism not in theta_dict:
                continue
            theta = _check_batch(theta_dict[mechanism], batch_size, f"theta[{mechanism}]").to(
                device=observations.device,
                dtype=observations.dtype,
            )
            theta_tensors.append(theta)
            feature_names.extend([f"theta/{mechanism}_{idx}" for idx in range(theta.shape[1])])
        if theta_tensors:
            groups["theta"] = list(range(start, len(feature_names)))
            tensors.append(torch.cat(theta_tensors, dim=1))

    _assert_no_forbidden_feature_names(feature_names)
    values = torch.cat(tensors, dim=1)
    return FeatureTensor(names=feature_names, values=values, groups=groups, metadata={"policy_safe": True})


def _check_batch(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.reshape(batch_size, 1)
    if value.ndim != 2 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must be [B, D], got {tuple(value.shape)}")
    return value


def _raise_on_forbidden_explicit_inputs(metadata: Mapping[str, Any]) -> None:
    explicit = metadata.get("policy_input_fields") or []
    for field in explicit:
        lowered = str(field).lower()
        if any(marker in lowered for marker in FORBIDDEN_POLICY_FIELD_MARKERS):
            raise ValueError(f"Forbidden policy input field requested: {field}")


def _assert_no_forbidden_feature_names(names: list[str]) -> None:
    for name in names:
        lowered = name.lower()
        if any(marker in lowered for marker in FORBIDDEN_POLICY_FIELD_MARKERS):
            raise ValueError(f"Forbidden policy feature generated: {name}")

