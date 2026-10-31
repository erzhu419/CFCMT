"""Feature registry for temporal DAG discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from cf_h2o.schemas import TransitionBatch


KEYWORD_GROUPS = {
    "headway": ["headway", "hw", "forward", "backward", "gap"],
    "demand": ["waiting", "arrival_rate", "passenger", "queue"],
    "dwell": ["dwell", "boarding", "alighting", "stop_time"],
    "speed": ["speed", "travel_time", "runtime", "segment"],
    "load": ["load", "onboard", "capacity"],
    "reward": ["reward"],
    "action": ["action", "holding", "hold"],
    "domain": ["domain", "source", "city", "real_sim", "label"],
    "theta": ["theta", "latent"],
}


@dataclass(frozen=True)
class FeatureDef:
    """One scalar temporal DAG node."""

    name: str
    group: str
    time_index: str
    source: str
    dim: int = 1

    @property
    def node_name(self) -> str:
        return f"{self.name}@{self.time_index}"


def infer_feature_group(name: str) -> str:
    lowered = name.lower()
    for group, keywords in KEYWORD_GROUPS.items():
        if any(keyword in lowered for keyword in keywords):
            return group
    return "unknown_state"


class FeatureRegistry:
    """Registry of present-time parent nodes and next-time child nodes."""

    def __init__(self):
        self.features: list[FeatureDef] = []

    def register(self, name: str, group: str | None, time_index: str, source: str, dim: int = 1):
        """Register a scalar feature or expand a vector feature into scalars."""

        if time_index not in {"t", "t1"}:
            raise ValueError(f"time_index must be 't' or 't1', got {time_index!r}")
        group = group or infer_feature_group(name)
        if int(dim) <= 1:
            self.features.append(FeatureDef(str(name), str(group), time_index, str(source), 1))
            return
        for idx in range(int(dim)):
            item_name = f"{name}_{idx}"
            self.features.append(FeatureDef(item_name, group, time_index, str(source), 1))

    @classmethod
    def from_transition_dataset(
        cls,
        dataset: TransitionBatch,
        obs_names: Optional[list[str]] = None,
        action_names: Optional[list[str]] = None,
        z_names: Optional[list[str]] = None,
    ) -> "FeatureRegistry":
        """Build a temporal registry from a transition batch.

        Raw line_id, route_id, source/domain labels are deliberately not
        registered as policy or graph nodes. They remain metadata.
        """

        registry = cls()
        metadata = dataset.metadata or {}
        obs_names = obs_names or metadata.get("obs_names") or metadata.get("observation_names")
        action_names = action_names or metadata.get("action_names")
        z_names = z_names or metadata.get("z_names")

        obs_dim = int(dataset.observations.shape[1])
        act_dim = int(dataset.actions.shape[1]) if dataset.actions.ndim > 1 else 1
        obs_names = list(obs_names) if obs_names is not None else [f"obs_{idx}" for idx in range(obs_dim)]
        action_names = list(action_names) if action_names is not None else [f"action_{idx}" for idx in range(act_dim)]
        if len(obs_names) != obs_dim:
            raise ValueError(f"obs_names length {len(obs_names)} does not match obs_dim {obs_dim}")
        if len(action_names) != act_dim:
            raise ValueError(f"action_names length {len(action_names)} does not match act_dim {act_dim}")

        for name in obs_names:
            group = infer_feature_group(name)
            registry.register(name, group, "t", "observations")
            registry.register(name, group, "t1", "next_observations")
        for name in action_names:
            registry.register(name, "action", "t", "actions")
        registry.register("reward", "reward", "t1", "rewards")

        if dataset.z_t is not None and dataset.z_t1 is not None:
            z_dim = int(dataset.z_t.shape[1])
            z_names = list(z_names) if z_names is not None else [f"z_{idx}" for idx in range(z_dim)]
            if len(z_names) != z_dim:
                raise ValueError(f"z_names length {len(z_names)} does not match z_dim {z_dim}")
            for name in z_names:
                group = infer_feature_group(name)
                registry.register(name, group, "t", "z_t")
                registry.register(name, group, "t1", "z_t1")

        if dataset.global_time is not None:
            registry.register("global_time", "unknown_state", "t", "global_time")

        return registry

    @property
    def node_names(self) -> list[str]:
        return [feature.node_name for feature in self.features]

    @property
    def node_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.node_names)}

    def get(self, node_name: str) -> FeatureDef:
        return self.features[self.node_index[node_name]]

    def infer_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for feature in self.features:
            groups.setdefault(feature.group, []).append(feature.node_name)
        return groups

    def node_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for idx, feature in enumerate(self.features):
            groups.setdefault(feature.group, []).append(idx)
        return groups

    def parent_node_indices(self) -> list[int]:
        return [
            idx
            for idx, feature in enumerate(self.features)
            if feature.time_index == "t" and feature.group not in {"domain"}
        ]

    def child_node_indices(self) -> list[int]:
        return [
            idx
            for idx, feature in enumerate(self.features)
            if feature.time_index == "t1" and feature.group in {"unknown_state", "headway", "demand", "dwell", "speed", "load", "reward"}
        ]

    def build_temporal_hard_mask(self) -> torch.Tensor:
        """Return mask[i, j] for allowed edge node_i -> node_j."""

        n_nodes = len(self.features)
        mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
        for src_idx, src in enumerate(self.features):
            if src.time_index != "t" or src.group == "domain":
                continue
            for dst_idx, dst in enumerate(self.features):
                if dst.time_index != "t1":
                    continue
                if dst.group == "action":
                    continue
                if src_idx == dst_idx:
                    continue
                mask[src_idx, dst_idx] = True
        return mask

    def values_from_dataset(self, dataset: TransitionBatch) -> torch.Tensor:
        """Build a dense matrix with columns aligned to ``node_names``."""

        columns: list[torch.Tensor] = []
        obs_cols = _split_columns(dataset.observations)
        next_obs_cols = _split_columns(dataset.next_observations)
        action_cols = _split_columns(dataset.actions if dataset.actions.ndim > 1 else dataset.actions.reshape(-1, 1))
        reward_col = dataset.rewards.reshape(-1, 1).to(dtype=dataset.observations.dtype, device=dataset.observations.device)
        z_cols = _split_columns(dataset.z_t) if dataset.z_t is not None else []
        z_next_cols = _split_columns(dataset.z_t1) if dataset.z_t1 is not None else []
        global_time_col = (
            dataset.global_time.reshape(-1, 1).to(dtype=dataset.observations.dtype, device=dataset.observations.device)
            if dataset.global_time is not None
            else None
        )

        source_offsets = {"observations": 0, "next_observations": 0, "actions": 0, "z_t": 0, "z_t1": 0}
        for feature in self.features:
            if feature.source == "observations":
                columns.append(obs_cols[source_offsets["observations"]])
                source_offsets["observations"] += 1
            elif feature.source == "next_observations":
                columns.append(next_obs_cols[source_offsets["next_observations"]])
                source_offsets["next_observations"] += 1
            elif feature.source == "actions":
                columns.append(action_cols[source_offsets["actions"]])
                source_offsets["actions"] += 1
            elif feature.source == "rewards":
                columns.append(reward_col)
            elif feature.source == "z_t":
                columns.append(z_cols[source_offsets["z_t"]])
                source_offsets["z_t"] += 1
            elif feature.source == "z_t1":
                columns.append(z_next_cols[source_offsets["z_t1"]])
                source_offsets["z_t1"] += 1
            elif feature.source == "global_time" and global_time_col is not None:
                columns.append(global_time_col)
            else:
                raise KeyError(f"Cannot materialize feature {feature.node_name} from source {feature.source}")

        return torch.cat(columns, dim=1)


def _split_columns(tensor: torch.Tensor | None) -> list[torch.Tensor]:
    if tensor is None:
        return []
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    return [tensor[:, idx : idx + 1] for idx in range(int(tensor.shape[1]))]


def mechanism_specs_from_registry(
    registry: FeatureRegistry,
    edge_marginals: torch.Tensor | None = None,
    min_parent_prob: float = 0.3,
) -> list[dict[str, Any]]:
    """Create JSON/YAML-friendly mechanism candidates from groups."""

    groups = registry.node_groups()
    node_names = registry.node_names
    specs: list[dict[str, Any]] = []
    mechanism_groups = [group for group in sorted(groups) if group not in {"action", "domain", "theta"}]
    for group in mechanism_groups:
        child_indices = [idx for idx in groups[group] if registry.features[idx].time_index == "t1"]
        if not child_indices:
            continue
        parent_names: list[str] = []
        if edge_marginals is not None:
            parent_indices = set()
            for child_idx in child_indices:
                parents = torch.where(edge_marginals[:, child_idx] >= float(min_parent_prob))[0].tolist()
                parent_indices.update(parents)
            parent_names = [node_names[idx] for idx in sorted(parent_indices)]
        specs.append(
            {
                "name": group,
                "child_names": [node_names[idx] for idx in child_indices],
                "parent_names": parent_names,
                "latent_dim": 1 if group == "reward" else 2,
                "output_dim": len(child_indices),
                "loss_type": "mse",
            }
        )
    if not specs:
        child_names = [node_names[idx] for idx in registry.child_node_indices()]
        specs.append(
            {
                "name": "state_group_0",
                "child_names": child_names,
                "parent_names": [],
                "latent_dim": 2,
                "output_dim": len(child_names),
                "loss_type": "mse",
            }
        )
    return specs

