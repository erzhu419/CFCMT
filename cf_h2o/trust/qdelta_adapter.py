"""Optional adapters for MC-WM QDelta trust critics."""

from __future__ import annotations

from typing import Any

import torch


class QDeltaTrustAdapter:
    """Wrap a QDelta-like object behind a torch-facing trust API."""

    def __init__(self, qdelta: Any, w_min: float = 0.05, w_max: float = 1.0):
        self.qdelta = qdelta
        self.w_min = float(w_min)
        self.w_max = float(w_max)

    @torch.no_grad()
    def weight(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if hasattr(self.qdelta, "weight"):
            weight = self.qdelta.weight(observations, actions)
        elif callable(self.qdelta):
            weight = self.qdelta(observations, actions)
        else:
            raise TypeError("qdelta must expose .weight(obs, action) or be callable")
        if not torch.is_tensor(weight):
            weight = torch.as_tensor(weight, dtype=observations.dtype, device=observations.device)
        return torch.clamp(weight.reshape(-1).to(device=observations.device, dtype=observations.dtype), self.w_min, self.w_max)

