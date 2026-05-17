"""Simple priors and metrics for latent mechanism factors."""

from __future__ import annotations

import torch
from torch import nn


class StandardNormalFactorPrior(nn.Module):
    """Standard normal prior over each mechanism factor."""

    def forward(self, theta_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if not theta_dict:
            return torch.zeros(())
        losses = []
        for theta in theta_dict.values():
            losses.append(0.5 * (theta**2).mean())
        return torch.stack(losses).mean()

    def log_prob(self, theta_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: (-0.5 * theta.pow(2) - 0.5 * torch.log(theta.new_tensor(2.0 * torch.pi))).sum(dim=-1)
            for name, theta in theta_dict.items()
        }


def theta_norm_metrics(theta_dict: dict[str, torch.Tensor], prefix: str = "theta_norm") -> dict[str, float]:
    return {
        f"{prefix}/{name}": float(theta.detach().norm(dim=-1).mean().cpu())
        for name, theta in theta_dict.items()
    }

