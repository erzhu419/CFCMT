"""Composition rules for mechanism-wise trust weights."""

from __future__ import annotations

from typing import Mapping

import torch


class WeightComposer:
    """Compose per-mechanism trust weights into one transition weight."""

    def __init__(
        self,
        mode: str = "geometric_mean",
        w_min: float = 0.05,
        w_max: float = 5.0,
        reward_path_weights: Mapping[str, float] | None = None,
    ):
        self.mode = str(mode)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.reward_path_weights = dict(reward_path_weights or {})

    def compose(self, trust_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if not trust_dict:
            raise ValueError("trust_dict cannot be empty")
        names = list(trust_dict)
        values = [trust_dict[name].reshape(-1) for name in names]
        batch_size = values[0].shape[0]
        if any(value.shape[0] != batch_size for value in values):
            raise ValueError("all trust tensors must have the same batch size")
        stacked = torch.stack(values, dim=0)

        if self.mode == "min":
            weight = stacked.min(dim=0).values
        elif self.mode == "arithmetic_mean":
            weight = stacked.mean(dim=0)
        elif self.mode == "product":
            weight = stacked.prod(dim=0)
        elif self.mode == "reward_path_weighted":
            weight = self._reward_path_weighted(names, stacked)
        elif self.mode == "geometric_mean":
            weight = torch.exp(torch.log(stacked.clamp_min(1e-8)).mean(dim=0))
        else:
            raise ValueError(f"Unknown trust composition mode: {self.mode}")
        return torch.clamp(torch.nan_to_num(weight, nan=self.w_min, posinf=self.w_max, neginf=self.w_min), self.w_min, self.w_max)

    def _reward_path_weighted(self, names: list[str], stacked: torch.Tensor) -> torch.Tensor:
        if not self.reward_path_weights:
            return torch.exp(torch.log(stacked.clamp_min(1e-8)).mean(dim=0))
        raw_weights = stacked.new_tensor([float(self.reward_path_weights.get(name, 1.0)) for name in names])
        raw_weights = raw_weights.clamp_min(0.0)
        if float(raw_weights.sum()) <= 0.0:
            raw_weights = torch.ones_like(raw_weights)
        norm_weights = raw_weights / raw_weights.sum()
        return torch.exp((torch.log(stacked.clamp_min(1e-8)) * norm_weights[:, None]).sum(dim=0))

