"""Deterministic local-graph feature encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cf_h2o.graph.local_neighborhood import local_graph_feature_dim, local_graph_to_tensor


class LocalGraphEncoder(nn.Module):
    """Encode padded local graph features into a fixed-size embedding."""

    def __init__(self, input_dims: dict[str, Any] | None = None, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        input_dims = dict(input_dims or {})
        self.input_dim = int(input_dims.get("feature_dim", input_dims.get("input_dim", local_graph_feature_dim())))
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, local_graph_batch: Any) -> torch.Tensor:
        """Return local embedding ``h_local`` with shape ``[B, out_dim]``."""

        param = next(self.parameters())
        features = local_graph_to_tensor(local_graph_batch, device=param.device, dtype=param.dtype)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D local graph features, got shape {tuple(features.shape)}")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {features.shape[1]}")
        return self.net(features)

