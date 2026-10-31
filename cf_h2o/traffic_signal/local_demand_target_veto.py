"""Route-projected local-demand execution veto for frozen CFCMT proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import DemandScheduleProfile
from cf_h2o.traffic_signal.demand_timeline_context import TLSLocalTopologyContext
from cf_h2o.traffic_signal.local_demand_timeline_context import (
    LocalDemandTimelineContext,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)


VETO_PROTOCOL = "target-simulator-450s-local-demand-causal-veto-v1"
ARTIFACT_PROTOCOL = "cfcmt-local-demand-target-veto-model-v1"


@dataclass(frozen=True)
class LocalDemandVetoConfig:
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
            raise ValueError("invalid local-demand veto configuration")


class LocalDemandTargetVeto:
    """Veto an existing proposal using label-free route and signal context."""

    def __init__(
        self,
        *,
        model: ProposalConditionalRegressor,
        config: LocalDemandVetoConfig,
        candidate_feature_names: Sequence[str],
        demand_profile: DemandScheduleProfile,
        topology_context: TLSLocalTopologyContext,
        local_demand_context: LocalDemandTimelineContext,
        target_transform: str,
        target_scale: float,
    ) -> None:
        self.model = model
        self.config = config
        self.candidate_feature_names = tuple(
            str(value) for value in candidate_feature_names
        )
        self.demand_profile = demand_profile
        self.topology_context = topology_context
        self.local_demand_context = local_demand_context
        self.target_transform = str(target_transform)
        self.target_scale = float(target_scale)
        expected = (
            *self.candidate_feature_names,
            *demand_profile.feature_names,
            *topology_context.feature_names,
            *local_demand_context.feature_names,
        )
        if (
            not self.candidate_feature_names
            or len(set(self.candidate_feature_names))
            != len(self.candidate_feature_names)
            or tuple(model.feature_names) != expected
            or demand_profile.input_sha256 != topology_context.input_sha256
            or demand_profile.input_sha256 != local_demand_context.input_sha256
            or self.target_transform != "raw_scenario_robust_scaled"
            or not np.isfinite(self.target_scale)
            or self.target_scale <= 0.0
        ):
            raise ValueError("local-demand model feature contract changed")

    def _combined_matrix(self, dataset: MechanismDataset) -> np.ndarray:
        index = {name: position for position, name in enumerate(dataset.feature_names)}
        missing = [name for name in self.candidate_feature_names if name not in index]
        if missing:
            raise KeyError(f"local-demand dataset missing features: {missing}")
        tls_ids = tuple(str(value) for value in dataset.metadata.get("row_tls", ()))
        times = np.asarray(dataset.metadata.get("row_times", ()), dtype=float)
        if (
            len(tls_ids) != dataset.size
            or times.shape != (dataset.size,)
            or not np.isfinite(times).all()
        ):
            raise ValueError("local-demand runtime metadata changed")
        columns = np.asarray(
            [index[name] for name in self.candidate_feature_names], dtype=int
        )
        candidate = np.asarray(dataset.features[:, columns], dtype=float)
        demand = np.repeat(
            self.demand_profile.vector()[None, :], dataset.size, axis=0
        )
        topology = np.vstack(
            [self.topology_context.vector_for(tls_id) for tls_id in tls_ids]
        )
        local = np.vstack(
            [
                self.local_demand_context.vector_at(tls_id, time_sec)
                for tls_id, time_sec in zip(tls_ids, times)
            ]
        )
        combined = np.concatenate([candidate, demand, topology, local], axis=1)
        if (
            combined.shape != (dataset.size, len(self.model.feature_names))
            or not np.isfinite(combined).all()
        ):
            raise ValueError("local-demand runtime matrix changed")
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
            raise IndexError("local-demand action index is outside the group")
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
            raise ValueError("local-demand candidate/reference is invalid")
        score = float(self.model.predict_matrix(self._combined_matrix(dataset))[candidate])
        if not self.config.enabled:
            rejection = "disabled"
        elif score > float(self.config.score_threshold):
            rejection = "local_demand_score"
        else:
            rejection = "none"
        tls_id = str(dataset.metadata["row_tls"][candidate])
        time_sec = float(dataset.metadata["row_times"][candidate])
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
            "runtime_tls_id": tls_id,
            "runtime_time_sec": time_sec,
            "demand_profile_input_sha256": self.demand_profile.input_sha256,
            "target_transform": self.target_transform,
            "target_scale": self.target_scale,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": VETO_PROTOCOL,
            "role": "local_demand_execution_veto_only",
            "action_originator": False,
            "target_labels": "target_simulator_450s_counterfactual_recovery_value",
            "target_transform": self.target_transform,
            "target_scale": self.target_scale,
            "context_labels": "none",
            "zero_shot": False,
            "candidate_feature_names": list(self.candidate_feature_names),
            "demand_feature_names": list(self.demand_profile.feature_names),
            "topology_feature_names": list(self.topology_context.feature_names),
            "local_demand_feature_names": list(
                self.local_demand_context.feature_names
            ),
            "demand_profile_input_sha256": self.demand_profile.input_sha256,
            "config": asdict(self.config),
        }
