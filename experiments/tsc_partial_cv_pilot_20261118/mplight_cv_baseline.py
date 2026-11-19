"""Sensor-matched FRAP/MPLight-style decentralized Double-DQN baseline under partial-CV sensing.

Architectural & non-leakage parity invariants:
1. Observations and rewards read STRICTLY from `PartialCvObserver` (`p_cv = 0.10`) and
   `SafePhaseExecutor` — unequipped vehicles are never queried during training or evaluation.
2. Uses a parameter-shared phase-competition Q-network over 78-dim candidate representations
   `z_i = [x_g (18), phi_i (30), phi_i - mean_{j != i}(phi_j) (30)]`, matching CFCMT's
   feature interface and applying the identical NEMA `feasible_indices_under_max_red` shield.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def build_phase_competition_inputs(
    state_vec: np.ndarray, action_mat: np.ndarray
) -> np.ndarray:
    """Build (K, 78) phase-competition matrix `[x_g, phi_i, phi_i - mean_{j!=i}(phi_j)]`."""
    k_act = int(action_mat.shape[0])
    s_rep = np.repeat(np.asarray(state_vec, dtype=np.float64)[None, :], k_act, axis=0)
    phi = np.asarray(action_mat, dtype=np.float64)
    if k_act <= 1:
        comp = np.zeros_like(phi)
    else:
        sum_all = np.sum(phi, axis=0, keepdims=True)
        mean_others = (sum_all - phi) / float(k_act - 1)
        comp = phi - mean_others
    # Clip extreme outliers for neural stability
    return np.clip(np.hstack([s_rep, phi, comp]), -25.0, 25.0)


@dataclass
class ReplayTransition:
    z_cur: np.ndarray  # (K, 78)
    action_idx: int
    reward: float
    z_next: np.ndarray  # (K', 78)
    done: bool


class MPLightCvQNetwork:
    """Shared 2-layer ReLU phase-competition Q-network with Adam + Double-DQN Huber updates."""

    def __init__(
        self,
        *,
        in_dim: int = 78,
        h1: int = 64,
        h2: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 0.90,
        tau: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.h1 = int(h1)
        self.h2 = int(h2)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.gamma = float(gamma)
        self.tau = float(tau)

        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, h1)),
            "b1": np.zeros(h1, dtype=np.float64),
            "W2": rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2)),
            "b2": np.zeros(h2, dtype=np.float64),
            "W3": rng.normal(0.0, np.sqrt(1.0 / h2), size=(h2, 1)),
            "b3": np.zeros(1, dtype=np.float64),
        }
        self.target_params: dict[str, np.ndarray] = {
            k: v.copy() for k, v in self.params.items()
        }
        self.adam_m: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.adam_v: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.params.items()
        }
        self.step_count: int = 0

    def clone(self) -> "MPLightCvQNetwork":
        cp = MPLightCvQNetwork(
            in_dim=self.in_dim,
            h1=self.h1,
            h2=self.h2,
            lr=self.lr,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            tau=self.tau,
        )
        cp.params = {k: v.copy() for k, v in self.params.items()}
        cp.target_params = {k: v.copy() for k, v in self.target_params.items()}
        return cp

    @staticmethod
    def _forward_with(
        x: np.ndarray, p: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u1 = x @ p["W1"] + p["b1"]
        a1 = np.maximum(u1, 0.0)
        u2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(u2, 0.0)
        q = (a2 @ p["W3"] + p["b3"]).reshape(-1)
        return q, u1, a1, u2, a2

    def predict_q(self, z_mat: np.ndarray, *, use_target: bool = False) -> np.ndarray:
        p = self.target_params if use_target else self.params
        q, _, _, _, _ = self._forward_with(z_mat, p)
        return q

    def score_group(self, state_vec: np.ndarray, action_mat: np.ndarray) -> np.ndarray:
        """Return cost-convention scores `-Q(s, a)` (lower = better phase, matching CFCMT)."""
        z = build_phase_competition_inputs(state_vec, action_mat)
        return -self.predict_q(z, use_target=False)

    def train_minibatch(self, batch: Sequence[ReplayTransition]) -> float:
        if not batch:
            return 0.0
        n_b = len(batch)
        x_sel = np.vstack([tr.z_cur[tr.action_idx] for tr in batch])
        targets = np.zeros(n_b, dtype=np.float64)
        for idx, tr in enumerate(batch):
            if tr.done or tr.z_next.shape[0] == 0:
                targets[idx] = tr.reward
            else:
                q_online_next = self.predict_q(tr.z_next, use_target=False)
                best_next = int(np.argmax(q_online_next))
                q_targ_next = self.predict_q(tr.z_next, use_target=True)
                targets[idx] = tr.reward + self.gamma * float(q_targ_next[best_next])

        q_pred, u1, a1, u2, a2 = self._forward_with(x_sel, self.params)
        diff = q_pred - targets
        # Huber loss (delta = 2.0) gradient wrt q_pred
        delta = 2.0
        abs_d = np.abs(diff)
        loss = float(
            np.mean(
                np.where(
                    abs_d <= delta, 0.5 * (diff ** 2), delta * (abs_d - 0.5 * delta)
                )
            )
        )
        dq = (np.clip(diff, -delta, delta) / float(n_b)).reshape(-1, 1)

        grads: dict[str, np.ndarray] = {}
        grads["W3"] = a2.T @ dq + self.weight_decay * self.params["W3"]
        grads["b3"] = np.sum(dq, axis=0)

        da2 = dq @ self.params["W3"].T
        du2 = da2 * (u2 > 0.0)
        grads["W2"] = a1.T @ du2 + self.weight_decay * self.params["W2"]
        grads["b2"] = np.sum(du2, axis=0)

        da1 = du2 @ self.params["W2"].T
        du1 = da1 * (u1 > 0.0)
        grads["W1"] = x_sel.T @ du1 + self.weight_decay * self.params["W1"]
        grads["b1"] = np.sum(du1, axis=0)

        self.step_count += 1
        b1_adam, b2_adam, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            g_clip = np.clip(g, -5.0, 5.0)
            self.adam_m[k] = b1_adam * self.adam_m[k] + (1.0 - b1_adam) * g_clip
            self.adam_v[k] = b2_adam * self.adam_v[k] + (1.0 - b2_adam) * (g_clip ** 2)
            m_hat = self.adam_m[k] / (1.0 - b1_adam ** self.step_count)
            v_hat = self.adam_v[k] / (1.0 - b2_adam ** self.step_count)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            self.target_params[k] = (
                (1.0 - self.tau) * self.target_params[k] + self.tau * self.params[k]
            )
        return loss
