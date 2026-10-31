"""Time-varying latent mechanism factor encoder."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn


DEFAULT_MECHANISM_NAMES = ["demand", "speed", "dwell", "headway", "reward"]
DEFAULT_LATENT_DIMS = {"demand": 2, "speed": 2, "dwell": 2, "headway": 2, "reward": 1}


class TimeVaryingFactorEncoder(nn.Module):
    """Infer mechanism factors from policy-safe historical features.

    The encoder intentionally does not accept domain_id, source labels, or
    real/sim flags. Those can be used in auxiliary losses outside the forward
    pass, but not as direct inputs for theta inference.
    """

    def __init__(
        self,
        input_dim: int,
        mechanism_names: list[str] | None = None,
        latent_dims: dict[str, int] | None = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.mechanism_names = list(mechanism_names or DEFAULT_MECHANISM_NAMES)
        self.latent_dims = dict(latent_dims or DEFAULT_LATENT_DIMS)
        self.hidden_dim = int(hidden_dim)

        missing = [name for name in self.mechanism_names if name not in self.latent_dims]
        if missing:
            raise KeyError(f"Missing latent dims for mechanisms: {missing}")

        self.step_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.context_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.hidden_dim, int(self.latent_dims[name])) for name in self.mechanism_names}
        )

    @classmethod
    def from_mechanism_specs(
        cls,
        input_dim: int,
        mechanism_specs: Sequence[Any],
        hidden_dim: int = 128,
    ) -> "TimeVaryingFactorEncoder":
        names: list[str] = []
        dims: dict[str, int] = {}
        for spec in mechanism_specs:
            if isinstance(spec, Mapping):
                name = str(spec["name"])
                latent_dim = int(spec.get("latent_dim", DEFAULT_LATENT_DIMS.get(name, 1)))
            else:
                name = str(spec.name)
                latent_dim = int(spec.latent_dim)
            names.append(name)
            dims[name] = latent_dim
        return cls(input_dim=input_dim, mechanism_names=names, latent_dims=dims, hidden_dim=hidden_dim)

    def forward(self, history_features: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return ``{mechanism_name: theta [B, latent_dim]}``.

        Args:
            history_features: ``[B, T, D]`` policy-safe history features.
            masks: optional ``[B, T]`` padding mask, where 1 means valid.
        """

        context = self._encode_context(history_features, masks)
        return {name: self.heads[name](context) for name in self.mechanism_names}

    def forward_sequence(self, history_features: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return per-prefix theta sequences ``{name: [B, T, latent_dim]}``."""

        if history_features.ndim != 3:
            raise ValueError(f"history_features must be [B, T, D], got {tuple(history_features.shape)}")
        seq_items: dict[str, list[torch.Tensor]] = {name: [] for name in self.mechanism_names}
        for end in range(1, history_features.shape[1] + 1):
            prefix_masks = masks[:, :end] if masks is not None else None
            theta = self.forward(history_features[:, :end], prefix_masks)
            for name in self.mechanism_names:
                seq_items[name].append(theta[name])
        return {name: torch.stack(items, dim=1) for name, items in seq_items.items()}

    def _encode_context(self, history_features: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        if history_features.ndim != 3:
            raise ValueError(f"history_features must be [B, T, D], got {tuple(history_features.shape)}")
        if history_features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {history_features.shape[-1]}")

        mask = _normalize_mask(history_features, masks)
        masked_features = history_features * mask.unsqueeze(-1)
        step_hidden = self.step_encoder(masked_features.reshape(-1, self.input_dim)).reshape(
            history_features.shape[0],
            history_features.shape[1],
            self.hidden_dim,
        )
        step_hidden = step_hidden * mask.unsqueeze(-1)

        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_hidden = step_hidden.sum(dim=1) / counts
        last_hidden = _last_valid(step_hidden, mask)
        context = torch.cat([mean_hidden, last_hidden], dim=-1)
        return self.context_layer(context)


def build_history_windows(
    features: torch.Tensor,
    history_len: int,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left-padded rolling history windows.

    Args:
        features: ``[N, D]`` sequence ordered by time.
        history_len: number of past/current steps per window.
        masks: optional ``[N]`` validity mask for source rows.

    Returns:
        windows: ``[N, history_len, D]``
        window_masks: ``[N, history_len]``
    """

    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    history_len = int(history_len)
    if history_len <= 0:
        raise ValueError("history_len must be positive")

    n_steps, dim = features.shape
    device = features.device
    dtype = features.dtype
    source_mask = (
        torch.ones(n_steps, device=device, dtype=dtype)
        if masks is None
        else masks.to(device=device, dtype=dtype).reshape(n_steps)
    )
    windows = torch.zeros(n_steps, history_len, dim, device=device, dtype=dtype)
    window_masks = torch.zeros(n_steps, history_len, device=device, dtype=dtype)
    for row in range(n_steps):
        start = max(0, row - history_len + 1)
        src = features[start : row + 1]
        src_mask = source_mask[start : row + 1]
        dst_start = history_len - src.shape[0]
        windows[row, dst_start:] = src
        window_masks[row, dst_start:] = src_mask
    windows = windows * window_masks.unsqueeze(-1)
    return windows, window_masks


def _normalize_mask(history_features: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
    if masks is None:
        return torch.ones(
            history_features.shape[:2],
            device=history_features.device,
            dtype=history_features.dtype,
        )
    if masks.shape != history_features.shape[:2]:
        raise ValueError(f"masks must be [B, T], got {tuple(masks.shape)}")
    return masks.to(device=history_features.device, dtype=history_features.dtype).clamp(0.0, 1.0)


def _last_valid(step_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_counts = mask.sum(dim=1).to(dtype=torch.long)
    batch_size, _, hidden_dim = step_hidden.shape
    result = torch.zeros(batch_size, hidden_dim, device=step_hidden.device, dtype=step_hidden.dtype)
    has_valid = valid_counts > 0
    if not has_valid.any():
        return result
    last_indices = (valid_counts[has_valid] - 1).clamp_min(0)
    compressed = []
    valid_hidden = step_hidden[has_valid]
    valid_mask = mask[has_valid].bool()
    for row_hidden, row_mask, last_index in zip(valid_hidden, valid_mask, last_indices):
        valid_rows = row_hidden[row_mask]
        if valid_rows.numel() == 0:
            compressed.append(torch.zeros(hidden_dim, device=step_hidden.device, dtype=step_hidden.dtype))
        else:
            compressed.append(valid_rows[last_index])
    result[has_valid] = torch.stack(compressed, dim=0)
    return result

