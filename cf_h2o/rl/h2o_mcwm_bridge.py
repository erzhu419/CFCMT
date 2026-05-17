"""Stage 0 bridge from H2O+ mixed training to MC-WM trust weights."""

from __future__ import annotations

from typing import Any

import torch


class H2OMCWMBridge:
    """Attach MC-WM trust weights to a copied `H2OPlusBus` instance."""

    def __init__(self, h2o_algo: Any, mcwm_adapter: Any, replay_buffer: Any = None, config: dict[str, Any] | None = None):
        self.h2o = h2o_algo
        self.mcwm = mcwm_adapter
        self.replay_buffer = replay_buffer if replay_buffer is not None else getattr(h2o_algo, "replay_buffer", None)
        self.config = dict(config or {})
        self.h2o.external_sim_weight_provider = self.compute_sim_weight

    def compute_sim_weight(self, sim_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute detached MC-WM trust weights for a H2O+ sim batch.

        Args:
            sim_batch: H2O+ batch dict containing `observations` [B, obs_dim]
                and `actions` [B, act_dim].

        Returns:
            [B] tensor on the same device as observations.
        """

        observations = sim_batch["observations"]
        actions = sim_batch["actions"]
        if int(getattr(self.h2o, "_total_steps", 0)) < int(self.config.get("trust_warmup_steps", 0)):
            return torch.ones(observations.shape[0], dtype=observations.dtype, device=observations.device)
        return self.mcwm.trust_weight(observations, actions).reshape(-1).detach()

    def train_step(self, batch_size: int, pretrain_steps: int = 0) -> dict[str, Any]:
        """Run one H2O+ training step with the bridge-installed weight provider."""

        return self.h2o.train(batch_size, pretrain_steps=pretrain_steps)
