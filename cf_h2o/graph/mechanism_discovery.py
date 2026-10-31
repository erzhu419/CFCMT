"""Masked parent learning primitives for Stage 2 graph discovery."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class MaskedParentLearner(nn.Module):
    """Small masked MLP matching the Stage 2 API.

    AutoDAGDiscoverer uses the deterministic ridge scorer below for stable
    bootstrap discovery, while this module keeps the trainable API available
    for later concrete-mask upgrades.
    """

    def __init__(self, parent_dim: int, child_dim: int, hidden_dim: int, mask_type: str = "concrete"):
        super().__init__()
        self.parent_dim = int(parent_dim)
        self.child_dim = int(child_dim)
        self.hidden_dim = int(hidden_dim)
        self.mask_type = mask_type
        self.mask_logits = nn.Parameter(torch.zeros(self.parent_dim))
        self.net = nn.Sequential(
            nn.Linear(self.parent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.child_dim),
        )

    def forward(self, parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_probs = torch.sigmoid(self.mask_logits)
        pred = self.net(parents * mask_probs)
        return pred, mask_probs

    def loss(self, pred, target, mask_probs, prior_logits=None, lambda_sparse: float = 1e-3, lambda_prior: float = 0.1):
        target = target.reshape_as(pred)
        loss = F.mse_loss(pred, target) + float(lambda_sparse) * mask_probs.mean()
        if prior_logits is not None:
            prior_probs = torch.sigmoid(prior_logits.to(mask_probs.device, dtype=mask_probs.dtype))
            loss = loss + float(lambda_prior) * F.binary_cross_entropy(mask_probs, prior_probs)
        return loss


def ridge_parent_edge_probabilities(
    parents: torch.Tensor,
    child: torch.Tensor,
    *,
    l2: float = 1e-3,
    score_threshold: float = 0.08,
    score_temperature: float = 0.04,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate parent probabilities using standardized ridge coefficients."""

    if parents.ndim != 2:
        raise ValueError(f"parents must be [B, P], got {tuple(parents.shape)}")
    if child.ndim == 1:
        child = child.reshape(-1, 1)
    if child.ndim != 2:
        raise ValueError(f"child must be [B, C], got {tuple(child.shape)}")
    if parents.shape[0] != child.shape[0]:
        raise ValueError("parents and child batch sizes differ")

    x = _standardize(parents)
    y = _standardize(child)
    parent_dim = x.shape[1]
    if parent_dim == 0:
        return torch.zeros(0, device=parents.device, dtype=parents.dtype), {"mse": 0.0, "scores": []}

    eye = torch.eye(parent_dim, device=x.device, dtype=x.dtype)
    xtx = x.T @ x / max(1, x.shape[0]) + float(l2) * eye
    xty = x.T @ y / max(1, x.shape[0])
    coeff = torch.linalg.solve(xtx, xty)
    scores = coeff.abs().mean(dim=1)
    probs = torch.sigmoid((scores - float(score_threshold)) / float(score_temperature))
    pred = x @ coeff
    mse = F.mse_loss(pred, y).detach().item()
    return probs.clamp(0.0, 1.0).detach(), {"mse": mse, "scores": scores.detach().cpu().tolist()}


def apply_max_parents(probs: torch.Tensor, max_parents: int | None) -> torch.Tensor:
    if max_parents is None or int(max_parents) <= 0 or probs.numel() <= int(max_parents):
        return probs
    max_parents = int(max_parents)
    keep = torch.zeros_like(probs, dtype=torch.bool)
    top_idx = torch.topk(probs, k=max_parents).indices
    keep[top_idx] = True
    return torch.where(keep, probs, torch.minimum(probs, torch.full_like(probs, 0.01)))


def residual_domain_stability_loss(residuals: torch.Tensor, domain_id: torch.Tensor | None) -> torch.Tensor:
    """First-version domain stability loss: variance of domain residual means."""

    if domain_id is None:
        return torch.zeros((), device=residuals.device, dtype=residuals.dtype)
    unique_domains = torch.unique(domain_id)
    if unique_domains.numel() <= 1:
        return torch.zeros((), device=residuals.device, dtype=residuals.dtype)
    means = []
    for domain in unique_domains:
        mask = domain_id == domain
        if mask.any():
            means.append(residuals[mask].mean(dim=0))
    if len(means) <= 1:
        return torch.zeros((), device=residuals.device, dtype=residuals.dtype)
    stacked = torch.stack(means, dim=0)
    return stacked.var(dim=0, unbiased=False).mean()


def _standardize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (tensor - mean) / std

