"""Demand-schedule-conditioned execution veto for a frozen CFCMT proposal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import DemandScheduleProfile
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)


VETO_PROTOCOL = "target-simulator-450s-demand-conditioned-causal-veto-v1"
ARTIFACT_PROTOCOL = "cfcmt-demand-conditioned-target-veto-model-v1"


@dataclass(frozen=True)
class DemandConditionedVetoConfig:
    enabled: bool
    acceptance_quantile: float
    score_threshold: float
    rollout_value_horizon_sec: int = 450

    def __post_init__(self) -> None:
        if (
            not 0.0 < float(self.acceptance_quantile) < 1.0
            or not np.isfinite(float(self.score_threshold))
            or int(self.rollout_value_horizon_sec) != 450
        ):
            raise ValueError("invalid demand-conditioned veto configuration")


class DemandConditionedTargetVeto:
    """Veto an existing proposal using local state and static demand schedule."""

    def __init__(
        self,
        *,
        model: ProposalConditionalRegressor,
        config: DemandConditionedVetoConfig,
        candidate_feature_names: Sequence[str],
        demand_profile: DemandScheduleProfile,
    ) -> None:
        self.model = model
        self.config = config
        self.candidate_feature_names = tuple(
            str(value) for value in candidate_feature_names
        )
        self.demand_profile = demand_profile
        expected = (*self.candidate_feature_names, *demand_profile.feature_names)
        if (
            not self.candidate_feature_names
            or len(set(self.candidate_feature_names))
            != len(self.candidate_feature_names)
            or tuple(model.feature_names) != expected
        ):
            raise ValueError("demand-conditioned model feature contract changed")

    def _combined_matrix(self, dataset: MechanismDataset) -> np.ndarray:
        index = {name: position for position, name in enumerate(dataset.feature_names)}
        missing = [name for name in self.candidate_feature_names if name not in index]
        if missing:
            raise KeyError(f"demand-conditioned dataset missing features: {missing}")
        columns = np.asarray(
            [index[name] for name in self.candidate_feature_names], dtype=int
        )
        candidate = np.asarray(dataset.features[:, columns], dtype=float)
        demand = np.repeat(
            self.demand_profile.vector()[None, :], dataset.size, axis=0
        )
        combined = np.concatenate([candidate, demand], axis=1)
        if combined.shape != (dataset.size, len(self.model.feature_names)):
            raise ValueError("demand-conditioned runtime matrix changed")
        return combined

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
            raise IndexError("demand-conditioned action index is outside the group")
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
            raise ValueError("demand-conditioned candidate/reference is invalid")
        score = float(self.model.predict_matrix(self._combined_matrix(dataset))[candidate])
        if not self.config.enabled:
            rejection = "disabled"
        elif score > float(self.config.score_threshold):
            rejection = "demand_conditioned_score"
        else:
            rejection = "none"
        return {
            "protocol": VETO_PROTOCOL,
            "eligible": rejection == "none",
            "rejection": rejection,
            "predicted_delta": score,
            "uncertainty": float(self.model.calibration_error),
            "context_trust": 1.0,
            "upper": score,
            "score_threshold": float(self.config.score_threshold),
            "acceptance_quantile": float(self.config.acceptance_quantile),
            "rollout_value_horizon_sec": int(
                self.config.rollout_value_horizon_sec
            ),
            "demand_profile_input_sha256": self.demand_profile.input_sha256,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": VETO_PROTOCOL,
            "role": "demand_conditioned_execution_veto_only",
            "action_originator": False,
            "target_labels": "target_simulator_450s_counterfactual_recovery_value",
            "demand_context_labels": "none",
            "zero_shot": False,
            "candidate_feature_names": list(self.candidate_feature_names),
            "demand_feature_names": list(self.demand_profile.feature_names),
            "demand_profile_input_sha256": self.demand_profile.input_sha256,
            "config": asdict(self.config),
        }
