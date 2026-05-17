"""Shared data contracts for CF-H2O.

The dataclasses keep policy-time fields separate from target/metadata fields so
later stages can enforce no future leakage and no source-domain shortcutting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TransitionBatch:
    """Transition tensors.

    Shapes:
        observations: [B, obs_dim]
        actions: [B, act_dim]
        rewards: [B] or [B, 1]
        next_observations: [B, obs_dim]
        dones: [B] or [B, 1]

    `z_t1`, `snapshot_t1`, and next variables are targets/context for model
    training and trust estimation. They must not be fed into policy inputs.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    z_t: Optional[torch.Tensor] = None
    z_t1: Optional[torch.Tensor] = None
    domain_id: Optional[torch.Tensor] = None
    line_id: Optional[list[str]] = None
    route_id: Optional[list[str]] = None
    global_time: Optional[torch.Tensor] = None
    snapshot_t: Optional[list[dict[str, Any]]] = None
    snapshot_t1: Optional[list[dict[str, Any]]] = None
    source: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def device(self) -> torch.device:
        return self.observations.device

    def to(self, device: torch.device | str) -> "TransitionBatch":
        """Move tensor fields to `device`, preserving Python metadata."""

        def move(value):
            return value.to(device) if isinstance(value, torch.Tensor) else value

        return TransitionBatch(
            observations=move(self.observations),
            actions=move(self.actions),
            rewards=move(self.rewards),
            next_observations=move(self.next_observations),
            dones=move(self.dones),
            z_t=move(self.z_t),
            z_t1=move(self.z_t1),
            domain_id=move(self.domain_id),
            line_id=self.line_id,
            route_id=self.route_id,
            global_time=move(self.global_time),
            snapshot_t=self.snapshot_t,
            snapshot_t1=self.snapshot_t1,
            source=self.source,
            metadata=dict(self.metadata),
        )


@dataclass
class FeatureTensor:
    """Named feature matrix.

    Shapes:
        values: [B, D]
    """

    names: list[str]
    values: torch.Tensor
    groups: dict[str, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSpec:
    """Temporal mechanism graph specification.

    Shapes:
        adjacency: [N, N], float in [0, 1], edge i -> j
        hard_mask: [N, N], bool; False means edge forbidden
        edge_probs: [N, N], posterior probability
    """

    node_names: list[str]
    adjacency: torch.Tensor
    hard_mask: torch.Tensor
    edge_probs: torch.Tensor
    node_groups: dict[str, list[int]]
    graph_type: str = "temporal_mechanism_graph"
    version: str = "0.1"


@dataclass
class GraphPosterior:
    """Posterior over temporal mechanism graphs.

    Shapes:
        log_weights: [K]
        edge_marginals: [N, N]
    """

    graphs: list[GraphSpec]
    log_weights: torch.Tensor
    edge_marginals: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MechanismSpec:
    """Specification for a mechanism-level residual module."""

    name: str
    child_names: list[str]
    parent_names: list[str]
    latent_dim: int
    output_dim: int
    loss_type: str


@dataclass
class MechanismBatch:
    """Mechanism-level training batch.

    Shapes:
        parents: [B, P_m]
        child_target: [B, O_m]
        actions: [B, act_dim]
        theta: [B, latent_dim] if provided
        mask: [P_m] if provided
    """

    mechanism_name: str
    parents: torch.Tensor
    child_target: torch.Tensor
    actions: torch.Tensor
    theta: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
