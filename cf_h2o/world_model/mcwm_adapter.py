"""Stage 0 adapter around the copied MC-WM residual world model."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from cf_h2o.schemas import TransitionBatch
from mc_wm.residual.world_model import CorrectedWorldModel, ResidualAdapter, WorldModelEnsemble


def _tensor_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _config_get(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default) if config is not None else default


class MCWMAdapter:
    """Torch-facing adapter for MC-WM's numpy-based residual world model.

    Public API uses `TransitionBatch` and torch tensors. The copied MC-WM
    implementation is left intact and receives numpy arrays internally.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict[str, Any] | None = None, device: str = "cpu"):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.config = dict(config or {})
        self.device = torch.device(device)
        self.w_min = float(_config_get(self.config, "w_min", 0.05))
        self.w_max = float(_config_get(self.config, "w_max", 5.0))

        self.sim_model = WorldModelEnsemble(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            K=int(_config_get(self.config, "ensemble_size", 5)),
            hidden=int(_config_get(self.config, "hidden_dim", 200)),
            lr=float(_config_get(self.config, "lr", 1e-3)),
            weight_decay=float(_config_get(self.config, "weight_decay", 1e-4)),
            device=str(self.device),
        )
        self.residual = ResidualAdapter(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=int(_config_get(self.config, "residual_hidden_dim", 64)),
            lr=float(_config_get(self.config, "residual_lr", 1e-3)),
            weight_decay=float(_config_get(self.config, "residual_weight_decay", 1e-4)),
            device=str(self.device),
        )
        self.corrected_model = CorrectedWorldModel(self.sim_model, self.residual)

    def fit_sim_model(self, sim_dataset: TransitionBatch) -> dict[str, Any]:
        """Train `M_sim` on simulator transitions."""

        self.sim_model.fit(
            _tensor_to_numpy(sim_dataset.observations),
            _tensor_to_numpy(sim_dataset.actions),
            _tensor_to_numpy(sim_dataset.next_observations),
            _tensor_to_numpy(sim_dataset.rewards).reshape(-1),
            n_epochs=int(_config_get(self.config, "train_epochs_sim", 100)),
            batch_size=int(_config_get(self.config, "batch_size", 256)),
            val_ratio=float(_config_get(self.config, "val_ratio", 0.1)),
            patience=int(_config_get(self.config, "patience", 20)),
        )
        return {"sim_model_trained": True}

    def fit_residual(self, paired_or_aligned_dataset: TransitionBatch) -> dict[str, Any]:
        """Train the residual adapter.

        Stage 0 supports paired residual data if `metadata` contains
        `sim_next_observations` and `sim_rewards`. Otherwise it records that no
        residual was trained and leaves the corrected model equal to `M_sim`.
        """

        sim_next = paired_or_aligned_dataset.metadata.get("sim_next_observations")
        sim_rewards = paired_or_aligned_dataset.metadata.get("sim_rewards")
        if sim_next is None or sim_rewards is None:
            return {
                "residual_trained": False,
                "reason": "missing paired sim_next_observations/sim_rewards",
            }

        self.residual.fit(
            _tensor_to_numpy(paired_or_aligned_dataset.observations),
            _tensor_to_numpy(paired_or_aligned_dataset.actions),
            _tensor_to_numpy(sim_next),
            _tensor_to_numpy(sim_rewards).reshape(-1),
            _tensor_to_numpy(paired_or_aligned_dataset.next_observations),
            _tensor_to_numpy(paired_or_aligned_dataset.rewards).reshape(-1),
            n_epochs=int(_config_get(self.config, "train_epochs_residual", 100)),
            batch_size=int(_config_get(self.config, "batch_size", 256)),
            patience=int(_config_get(self.config, "patience", 20)),
        )
        return {"residual_trained": True}

    def predict(self, observations: torch.Tensor, actions: torch.Tensor, deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Predict next transition tensors.

        Args:
            observations: [B, obs_dim]
            actions: [B, act_dim]

        Returns:
            next_observations: [B, obs_dim]
            rewards: [B]
            epistemic: [B]
            aleatoric: [B] filled with zeros in Stage 0
        """

        obs_np = _tensor_to_numpy(observations)
        act_np = _tensor_to_numpy(actions)
        next_obs_np, rewards_np = self.corrected_model.predict(obs_np, act_np, deterministic=deterministic)
        epistemic_np = self.sim_model.get_disagreement(obs_np, act_np)

        return {
            "next_observations": torch.as_tensor(next_obs_np, dtype=observations.dtype, device=observations.device),
            "rewards": torch.as_tensor(rewards_np, dtype=observations.dtype, device=observations.device).reshape(-1),
            "epistemic": torch.as_tensor(epistemic_np, dtype=observations.dtype, device=observations.device).reshape(-1),
            "aleatoric": torch.zeros(observations.shape[0], dtype=observations.dtype, device=observations.device),
        }

    def rollout(self, init_observations: torch.Tensor, policy: Callable[[torch.Tensor], torch.Tensor], horizon: int) -> TransitionBatch:
        """Generate model rollouts.

        Args:
            init_observations: [B, obs_dim]
            policy: callable mapping observations [B, obs_dim] to actions [B, act_dim]
            horizon: rollout length

        Returns:
            Flattened `TransitionBatch` with B * horizon transitions.
        """

        obs_items = []
        action_items = []
        reward_items = []
        next_obs_items = []
        done_items = []
        sources = []

        obs = init_observations.to(self.device)
        for _ in range(int(horizon)):
            with torch.no_grad():
                actions = policy(obs)
                if isinstance(actions, tuple):
                    actions = actions[0]
            pred = self.predict(obs, actions, deterministic=False)
            next_obs = pred["next_observations"]

            obs_items.append(obs)
            action_items.append(actions)
            reward_items.append(pred["rewards"])
            next_obs_items.append(next_obs)
            done_items.append(torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device))
            sources.extend(["model"] * int(obs.shape[0]))
            obs = next_obs.detach()

        return TransitionBatch(
            observations=torch.cat(obs_items, dim=0),
            actions=torch.cat(action_items, dim=0),
            rewards=torch.cat(reward_items, dim=0),
            next_observations=torch.cat(next_obs_items, dim=0),
            dones=torch.cat(done_items, dim=0),
            source=sources,
        )

    def trust_weight(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return simulator/model trust weights in `[w_min, w_max]`.

        The Stage 0 rule is monotone decreasing in ensemble disagreement. This
        is intentionally detached before entering H2O+ critic losses.
        """

        obs_np = _tensor_to_numpy(observations)
        act_np = _tensor_to_numpy(actions)
        disagreement = torch.as_tensor(
            self.sim_model.get_disagreement(obs_np, act_np),
            dtype=observations.dtype,
            device=observations.device,
        ).reshape(-1)
        scale = torch.clamp(disagreement.detach().median(), min=1e-6)
        weight = torch.exp(-disagreement / scale)
        return torch.clamp(weight, self.w_min, self.w_max).detach()
