"""Target-labeled long-horizon veto for frozen cross-city action proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


VETO_PROTOCOL = "target-simulator-300s-recovery-causal-veto-v1"


@dataclass(frozen=True)
class LongHorizonTargetVetoConfig:
    enabled: bool
    risk_multiplier: float
    min_context_trust: float
    rollout_value_horizon_sec: int = 300

    def __post_init__(self) -> None:
        if (
            float(self.risk_multiplier) < 0.0
            or not 0.0 <= float(self.min_context_trust) <= 1.0
            or int(self.rollout_value_horizon_sec) <= 0
        ):
            raise ValueError("invalid long-horizon target-veto configuration")


class LongHorizonTargetVeto:
    """Veto, but never originate or redirect, one frozen proposal action."""

    def __init__(self, *, model: Any, config: LongHorizonTargetVetoConfig) -> None:
        self.model = model
        self.config = config

    def evaluate(
        self,
        dataset: MechanismDataset,
        *,
        candidate_index: int,
        reference_index: int,
    ) -> dict[str, float | bool | str]:
        candidate = int(candidate_index)
        reference = int(reference_index)
        if not 0 <= candidate < dataset.size or not 0 <= reference < dataset.size:
            raise IndexError("target-veto action index is outside the contrast group")
        is_reference = np.asarray(
            dataset.metadata.get("is_reference", ()), dtype=bool
        )
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
        if (
            is_reference.shape != (dataset.size,)
            or groups.shape != (dataset.size,)
            or not bool(is_reference[reference])
            or bool(is_reference[candidate])
            or groups[candidate] != groups[reference]
        ):
            raise ValueError("target-veto candidate/reference identity is invalid")
        prediction = self.model.predict(dataset).get("control_cost", {})
        mean = np.asarray(prediction.get("mean", ()), dtype=float)
        uncertainty = np.asarray(prediction.get("uncertainty", ()), dtype=float)
        trust = np.asarray(prediction.get("context_trust", ()), dtype=float)
        if not (
            mean.shape == uncertainty.shape == trust.shape == (dataset.size,)
            and np.isfinite(mean).all()
            and np.isfinite(uncertainty).all()
            and np.isfinite(trust).all()
        ):
            raise ValueError("target-veto prediction bundle is invalid")
        predicted_delta = float(mean[candidate] - mean[reference])
        predicted_uncertainty = float(
            uncertainty[candidate] + uncertainty[reference]
        )
        context_trust = float(min(trust[candidate], trust[reference]))
        upper = predicted_delta + float(self.config.risk_multiplier) * max(
            predicted_uncertainty, 0.0
        )
        if not self.config.enabled:
            rejection = "disabled"
        elif context_trust < float(self.config.min_context_trust):
            rejection = "target_support"
        elif upper >= 0.0:
            rejection = "target_confidence"
        else:
            rejection = "none"
        return {
            "protocol": VETO_PROTOCOL,
            "eligible": rejection == "none",
            "rejection": rejection,
            "predicted_delta": predicted_delta,
            "uncertainty": max(predicted_uncertainty, 0.0),
            "context_trust": context_trust,
            "upper": float(upper),
            "rollout_value_horizon_sec": int(
                self.config.rollout_value_horizon_sec
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": VETO_PROTOCOL,
            "role": "execution_veto_only_never_action_originator",
            "target_labels": "target_simulator_counterfactual_recovery_value",
            "zero_shot": False,
            "config": asdict(self.config),
        }
