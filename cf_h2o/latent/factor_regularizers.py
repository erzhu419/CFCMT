"""Regularizers for latent mechanism factors."""

from __future__ import annotations

import torch


def temporal_smoothness_loss(theta_seq):
    """Penalize rapid changes in theta sequences.

    Accepts either a tensor ``[B, T, D]`` or a dict of such tensors.
    """

    if isinstance(theta_seq, dict):
        losses = [temporal_smoothness_loss(value) for value in theta_seq.values()]
        if not losses:
            return torch.zeros(())
        return torch.stack(losses).mean()
    if theta_seq.ndim < 3 or theta_seq.shape[1] < 2:
        return theta_seq.new_zeros(())
    return ((theta_seq[:, 1:] - theta_seq[:, :-1]) ** 2).mean()


def mechanism_independence_loss(theta_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Penalize correlation among mechanism factors."""

    if len(theta_dict) <= 1:
        first = next(iter(theta_dict.values()), None)
        return torch.zeros((), device=first.device, dtype=first.dtype) if first is not None else torch.zeros(())

    theta = torch.cat([value.reshape(value.shape[0], -1) for value in theta_dict.values()], dim=1)
    if theta.shape[0] < 2 or theta.shape[1] < 2:
        return theta.new_zeros(())
    theta = theta - theta.mean(dim=0, keepdim=True)
    theta = theta / theta.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    corr = theta.T @ theta / max(1, theta.shape[0])
    off_diag = corr - torch.diag(torch.diag(corr))
    return (off_diag**2).mean()


def domain_contrast_loss(theta_dict: dict[str, torch.Tensor], domain_id: torch.Tensor | None) -> torch.Tensor:
    """Optional diagnostic loss encouraging non-collapsed domain factors.

    ``domain_id`` is used only in the loss, not as an encoder input.
    """

    first = next(iter(theta_dict.values()), None)
    if first is None:
        return torch.zeros(())
    if domain_id is None:
        return first.new_zeros(())

    theta = torch.cat([value.reshape(value.shape[0], -1) for value in theta_dict.values()], dim=1)
    domain_id = domain_id.to(device=theta.device).reshape(-1)
    unique_domains = torch.unique(domain_id)
    if unique_domains.numel() <= 1 or theta.shape[0] < 2:
        return theta.new_zeros(())

    centroids = []
    within_terms = []
    for domain in unique_domains:
        mask = domain_id == domain
        if mask.sum() == 0:
            continue
        domain_theta = theta[mask]
        centroid = domain_theta.mean(dim=0)
        centroids.append(centroid)
        within_terms.append(((domain_theta - centroid) ** 2).mean())
    if len(centroids) <= 1:
        return theta.new_zeros(())

    centroids_tensor = torch.stack(centroids, dim=0)
    within = torch.stack(within_terms).mean()
    between = centroids_tensor.var(dim=0, unbiased=False).mean()
    return within / (between + 1e-6)

