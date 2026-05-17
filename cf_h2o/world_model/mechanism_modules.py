"""Mechanism-level modules for causal-factored world models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class MechanismModule(nn.Module):
    """Small deterministic mechanism head with optional theta conditioning."""

    def __init__(
        self,
        parent_dim: int,
        theta_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        *,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.parent_dim = int(parent_dim)
        self.theta_dim = int(theta_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = self.parent_dim + self.theta_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        if zero_init_output:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        parents: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic mean/logvar-style output dict."""

        if parents.ndim != 2:
            raise ValueError(f"parents must be [B, P], got {tuple(parents.shape)}")
        if parents.shape[1] != self.parent_dim:
            raise ValueError(f"Expected parent dim {self.parent_dim}, got {parents.shape[1]}")
        if mask is not None:
            parents = parents * _broadcast_mask(mask, parents)

        if self.theta_dim > 0:
            if theta is None:
                theta = parents.new_zeros(parents.shape[0], self.theta_dim)
            if theta.ndim != 2 or theta.shape != (parents.shape[0], self.theta_dim):
                raise ValueError(f"theta must be [B, {self.theta_dim}], got {tuple(theta.shape)}")
            inputs = torch.cat([parents, theta.to(dtype=parents.dtype, device=parents.device)], dim=-1)
        else:
            inputs = parents

        mean = self.net(inputs)
        logvar = torch.zeros_like(mean)
        return {
            "mean": mean,
            "logvar": logvar,
            "uncertainty": torch.zeros(mean.shape[0], device=mean.device, dtype=mean.dtype),
        }

    def loss(self, pred: dict[str, torch.Tensor], target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
        target = target.reshape_as(pred["mean"])
        if loss_type == "gaussian_nll":
            logvar = pred["logvar"].clamp(-10.0, 5.0)
            return (0.5 * (logvar + (target - pred["mean"]).pow(2) / logvar.exp())).mean()
        if loss_type == "poisson_nll":
            return F.poisson_nll_loss(pred["mean"].clamp_min(1e-6), target.clamp_min(0.0), log_input=False)
        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(pred["mean"], target)
        return F.mse_loss(pred["mean"], target)


def _broadcast_mask(mask: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=parents.device, dtype=parents.dtype)
    if mask.ndim == 1:
        if mask.shape[0] != parents.shape[1]:
            raise ValueError(f"mask dim {mask.shape[0]} does not match parent dim {parents.shape[1]}")
        return mask.unsqueeze(0)
    if mask.ndim == 2:
        if mask.shape != parents.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} does not match parents {tuple(parents.shape)}")
        return mask
    raise ValueError(f"mask must be [P] or [B, P], got {tuple(mask.shape)}")

