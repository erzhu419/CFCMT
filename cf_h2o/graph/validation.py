"""Validation helpers for temporal graph discovery."""

from __future__ import annotations

import torch


def forbidden_edge_max_probability(edge_marginals: torch.Tensor, hard_mask: torch.Tensor) -> float:
    forbidden = ~hard_mask
    if not forbidden.any():
        return 0.0
    return float(edge_marginals[forbidden].max().detach().cpu())


def edge_auc(edge_marginals: torch.Tensor, true_edges: set[tuple[int, int]], hard_mask: torch.Tensor) -> float:
    """Compute AUC over allowed edges without requiring sklearn."""

    positives = []
    negatives = []
    edge_probs = edge_marginals.detach().cpu()
    hard = hard_mask.detach().cpu()
    for src_idx in range(edge_probs.shape[0]):
        for dst_idx in range(edge_probs.shape[1]):
            if not bool(hard[src_idx, dst_idx]):
                continue
            score = float(edge_probs[src_idx, dst_idx])
            if (src_idx, dst_idx) in true_edges:
                positives.append(score)
            else:
                negatives.append(score)
    if not positives or not negatives:
        raise ValueError("AUC requires at least one positive and one negative allowed edge")
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total

