"""Closed-loop deployment of the V150K state-conditioned source utility model."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    CONTRAST_FEATURES,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_CORE_FAMILY_V3,
    CONTRAST_FEATURES_V3,
    FEATURE_NAMES_V3,
    FittedContrastModelsV3,
    evaluate_policy_v3,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    _rigid_score,
    residual_corrected_rigid_score,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    RUNTIME_MODEL_PROTOCOL,
)
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.action_contrast import action_group_ids
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import pressure_spec
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    mechanism_parameter_design,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
from cf_h2o.traffic_signal.right_of_way_context import (
    NEIGHBOR_EXECUTION_STATE_FEATURES,
    RIGHT_OF_WAY_ACTION_FEATURES,
    RIGHT_OF_WAY_CONTEXT_FEATURES,
    NetworkRightOfWayContext,
    candidate_right_of_way_features,
    net_file_from_sumocfg,
    read_network_right_of_way_context,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    FittedUtilityGate,
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
    apply_utility_gate,
    pressure_align_candidate_scores,
    pressure_pairwise_candidate_scores,
)


RESULT_PROTOCOL = "tsc-v150l-state-conditioned-source-closed-loop-rollout-v3"
ORIGINATOR_PROTOCOL = "tsc-v150l-state-conditioned-source-originator-v1"
RUNTIME_POLICY = f"{CFCMT_CORE_FAMILY_V3}_contrast_source_utility"
POLICY_ARMS = (
    "phase_pressure",
    "rigid_target_only",
    "selected_source_corrected_rigid",
    "same_capacity_placebo_corrected_rigid",
)
SOURCE_ARMS = frozenset(POLICY_ARMS[1:])


def assess_zero_incident_diagnostic(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve raw incidents without deciding paired deployment safety."""

    starting_teleports = int(metrics.get("starting_teleports", -1))
    ending_teleports = int(metrics.get("ending_teleports", -1))
    collision_events = int(metrics.get("collision_events", -1))
    collision_incidents = int(metrics.get("collision_incidents", -1))
    passed = (
        starting_teleports == 0
        and ending_teleports == 0
        and collision_events == 0
    )
    return {
        "protocol": "zero-teleport-zero-collision-observed-rollout-v1",
        "passed": bool(passed),
        "starting_teleports": starting_teleports,
        "ending_teleports": ending_teleports,
        "collision_events": collision_events,
        "collision_incidents": collision_incidents,
        "handling": "record_result_then_apply_paired_aggregate_safety_gate",
    }


@dataclass(frozen=True)
class StateConditionedSourceDecision:
    reference_index: int
    proposed_index: int
    selected_index: int
    rigid_selected_index: int
    learned_differs: bool
    eligible: bool
    priority: float
    rejection: str | None
    arm: str
    source_city_admitted: bool
    utility_gate_selected: bool
    selected_candidate_key: str | None
    source_changed_rigid_action: bool
    predicted_score: float
    reference_score: float
    rigid_predicted_score: float
    candidate_score_constraint: str | None
    candidate_changed_rigid_group_count: int
    candidate_pressure_aligned_change_group_count: int
    candidate_off_reference_rejected_group_count: int
    source_changed_to_reference_action: bool
    source_changed_off_reference_action: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class StateConditionedSourceUtilityOriginator:
    """Rank feasible phases with a frozen target model and gated source prior."""

    selection_layer = "state_conditioned_source_utility"

    def __init__(
        self,
        *,
        runtime_payload: Mapping[str, Any],
        network_context: NetworkRightOfWayContext,
        arm: str,
        expected_runtime_protocol: str = RUNTIME_MODEL_PROTOCOL,
        candidate_score_constraint: str | None = None,
        originator_protocol: str = ORIGINATOR_PROTOCOL,
    ) -> None:
        self.payload = dict(runtime_payload)
        self.network_context = network_context
        self.arm = str(arm)
        self.expected_runtime_protocol = str(expected_runtime_protocol)
        self.candidate_score_constraint = candidate_score_constraint
        self.originator_protocol = str(originator_protocol)
        self._validate_payload()
        self._neighbor_features: dict[str, np.ndarray] = {}
        self._interval_count = 0
        self._decision_count = 0
        self._override_count = 0
        self._utility_selected_count = 0
        self._source_changed_action_count = 0
        self._source_changed_to_reference_action_count = 0
        self._source_changed_off_reference_action_count = 0
        self._candidate_changed_rigid_group_count = 0
        self._candidate_pressure_aligned_change_group_count = 0
        self._candidate_off_reference_rejected_group_count = 0
        self._candidate_counts: Counter[str] = Counter()
        self._neighbor_observation_ratios: list[float] = []

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        expected_city: str,
        network_context: NetworkRightOfWayContext,
        arm: str,
        expected_runtime_protocol: str = RUNTIME_MODEL_PROTOCOL,
        candidate_score_constraint: str | None = None,
        originator_protocol: str = ORIGINATOR_PROTOCOL,
    ) -> "StateConditionedSourceUtilityOriginator":
        model_path = Path(path)
        if _sha256(model_path) != str(expected_sha256):
            raise ValueError("V150L runtime model identity changed")
        payload = pickle.loads(model_path.read_bytes())
        if not isinstance(payload, Mapping) or str(payload.get("city")) != str(
            expected_city
        ):
            raise ValueError("V150L runtime model belongs to another city")
        return cls(
            runtime_payload=payload,
            network_context=network_context,
            arm=arm,
            expected_runtime_protocol=expected_runtime_protocol,
            candidate_score_constraint=candidate_score_constraint,
            originator_protocol=originator_protocol,
        )

    def _validate_payload(self) -> None:
        if self.arm not in SOURCE_ARMS:
            raise ValueError(f"unknown V150L source arm: {self.arm!r}")
        if self.payload.get("protocol") != self.expected_runtime_protocol:
            raise ValueError("V150L runtime model protocol changed")
        if self.payload.get("occupancy_equation_protocol") != OCCUPANCY_EQUATION_PROTOCOL:
            raise ValueError("V150L runtime model occupancy equation contract mismatch")
        if self.candidate_score_constraint not in {
            None,
            PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
            PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
        }:
            raise ValueError("V150L candidate score constraint is unknown")
        if self.payload.get("candidate_score_constraint") != (
            self.candidate_score_constraint
        ):
            raise ValueError("V150L runtime and deployment constraints differ")
        feature_names = tuple(str(value) for value in self.payload.get(
            "mechanism_feature_names", ()
        ))
        scale = np.asarray(self.payload.get("feature_scale", ()), dtype=float)
        target_beta = np.asarray(self.payload.get("target_beta", ()), dtype=float)
        candidate_betas = dict(self.payload.get("candidate_betas", {}))
        placebo_betas = dict(self.payload.get("placebo_betas", {}))
        if (
            not feature_names
            or scale.shape != (len(feature_names),)
            or target_beta.shape != scale.shape
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or not np.all(np.isfinite(target_beta))
            or not candidate_betas
            or set(candidate_betas) != set(placebo_betas)
            or any(
                np.asarray(beta, dtype=float).shape != scale.shape
                or not np.all(np.isfinite(np.asarray(beta, dtype=float)))
                for beta in (*candidate_betas.values(), *placebo_betas.values())
            )
            or self.payload.get("rigid_model") is None
            or not isinstance(self.payload.get("source_utility_gate"), FittedUtilityGate)
            or not isinstance(self.payload.get("placebo_utility_gate"), FittedUtilityGate)
        ):
            raise ValueError("V150L runtime model payload is incomplete")

    def prepare_interval(
        self,
        *,
        states: Mapping[str, Any],
        executors: Mapping[str, Any],
        routing_graph: Any,
        control_interval_sec: int,
    ) -> None:
        """Cache simultaneous neighboring execution state before action ranking."""

        state_ids = {str(value) for value in states}
        executor_ids = {str(value) for value in executors}
        if state_ids != executor_ids:
            raise ValueError("V150L states and executors are not aligned")
        adjacency = {
            str(source): tuple(str(target) for target in targets)
            for source, targets in routing_graph.adjacency.items()
        }
        summaries = {
            tls_id: self._execution_summary(
                tls_id,
                executors[tls_id],
                control_interval_sec=int(control_interval_sec),
            )
            for tls_id in sorted(state_ids)
        }
        neighbor_features: dict[str, np.ndarray] = {}
        for tls_id in sorted(state_ids):
            downstream = set(adjacency.get(tls_id, ()))
            upstream = {
                source
                for source, targets in adjacency.items()
                if tls_id in set(targets)
            }
            neighbors = tuple(sorted((downstream | upstream) - {tls_id}))
            observed = [summaries[name] for name in neighbors if name in summaries]
            observation_ratio = (
                len(observed) / len(neighbors) if neighbors else 1.0
            )
            neighbor_features[tls_id] = np.asarray(
                [
                    observation_ratio,
                    _mean([row["protected"] for row in observed]),
                    _mean([row["permissive"] for row in observed]),
                    _mean([row["elapsed"] for row in observed]),
                    _mean([row["switching"] for row in observed]),
                    _mean([row["clearance"] for row in observed]),
                ],
                dtype=float,
            )
            self._neighbor_observation_ratios.append(float(observation_ratio))
        self._neighbor_features = neighbor_features
        self._interval_count += 1

    def _execution_summary(
        self, tls_id: str, executor: Any, *, control_interval_sec: int
    ) -> dict[str, float]:
        state = (
            str(executor.target_green_state)
            if bool(executor.is_switching)
            else str(executor.current_green_state)
        )
        topology = candidate_right_of_way_features(
            self.network_context, tls_id, state
        )
        topo_index = {
            name: index for index, name in enumerate(RIGHT_OF_WAY_ACTION_FEATURES)
        }
        timing = executor.timings[str(executor.current_green_state)]
        clearance = (
            min(
                max(float(executor.remaining_sec), 0.0)
                / max(float(control_interval_sec), 1.0),
                1.5,
            )
            if bool(executor.is_switching)
            else 0.0
        )
        return {
            "protected": float(topology[topo_index["protected_green_ratio"]]),
            "permissive": float(topology[topo_index["permissive_green_ratio"]]),
            "elapsed": float(
                min(
                    float(executor.green_elapsed_sec)
                    / max(float(timing.min_green_sec), 1.0),
                    4.0,
                )
            ),
            "switching": float(bool(executor.is_switching)),
            "clearance": float(clearance),
        }

    def _augment_contrast(
        self, contrast: MechanismDataset, *, reference_index: int
    ) -> MechanismDataset:
        reference = int(reference_index)
        candidate_states = np.asarray(
            contrast.metadata.get("candidate_states", ()), dtype=str
        )
        row_tls = np.asarray(contrast.metadata.get("row_tls", ()), dtype=str)
        if (
            contrast.metadata.get("occupancy_equation_protocol") != OCCUPANCY_EQUATION_PROTOCOL
            or tuple(contrast.feature_names[: len(FEATURE_NAMES_V3)])
            != tuple(FEATURE_NAMES_V3)
            or tuple(contrast.metadata.get("contrast_features", ()))
            != tuple(CONTRAST_FEATURES_V3)
            or candidate_states.shape != (contrast.size,)
            or row_tls.shape != (contrast.size,)
            or np.unique(row_tls).size != 1
            or not 0 <= reference < contrast.size
        ):
            raise ValueError("V150L runtime contrast schema changed")
        tls_id = str(row_tls[0])
        if tls_id not in self._neighbor_features:
            raise RuntimeError("V150L originator was not prepared for this interval")
        base_absolute = np.asarray(
            contrast.features[:, : len(FEATURE_NAMES_V3)], dtype=float
        )
        topology = np.vstack(
            [
                candidate_right_of_way_features(
                    self.network_context, tls_id, state
                )
                for state in candidate_states
            ]
        )
        neighbor = np.repeat(
            self._neighbor_features[tls_id][None, :], contrast.size, axis=0
        )
        absolute = np.column_stack((base_absolute, topology, neighbor))
        absolute_names = (*FEATURE_NAMES_V3, *RIGHT_OF_WAY_CONTEXT_FEATURES)
        absolute_index = {
            name: index for index, name in enumerate(absolute_names)
        }
        selected = np.asarray(
            [absolute_index[name] for name in CONTRAST_FEATURES], dtype=int
        )
        reference_features = np.repeat(
            absolute[reference, selected][None, :], contrast.size, axis=0
        )
        augmented = np.column_stack(
            (absolute, reference_features, absolute[:, selected] - reference_features)
        )
        feature_names = (
            *absolute_names,
            *(f"reference_{name}" for name in CONTRAST_FEATURES),
            *(f"delta_{name}" for name in CONTRAST_FEATURES),
        )
        if not np.all(np.isfinite(augmented)):
            raise ValueError("V150L runtime context contains non-finite values")
        references = np.arange(contrast.size) == reference
        return MechanismDataset(
            feature_names=tuple(feature_names),
            features=augmented,
            context_names=contrast.context_names,
            context=np.asarray(contrast.context, dtype=float).copy(),
            priors={
                name: np.asarray(values, dtype=float).copy()
                for name, values in contrast.priors.items()
            },
            targets={
                name: np.asarray(values, dtype=float).copy()
                for name, values in contrast.targets.items()
            },
            domains=np.asarray(contrast.domains).copy(),
            metadata={
                **dict(contrast.metadata),
                "reference_rows": [reference] * contrast.size,
                "is_reference": references.tolist(),
                "contrast_features": list(CONTRAST_FEATURES),
                "right_of_way_context_protocol": (
                    "deploy-observable-right-of-way-and-simultaneous-neighbor-state-v1"
                ),
                "right_of_way_feature_names": list(RIGHT_OF_WAY_CONTEXT_FEATURES),
            },
        )

    @staticmethod
    def _argmin_with_reference_tie_break(
        values: np.ndarray,
        *,
        reference: int,
        candidate_states: np.ndarray,
    ) -> int:
        minimum = float(np.min(values))
        tied = np.flatnonzero(
            np.isclose(values, minimum, rtol=0.0, atol=1e-12)
        )
        if reference in {int(value) for value in tied}:
            return int(reference)
        return min(
            (int(value) for value in tied),
            key=lambda row: (str(candidate_states[row]), row),
        )

    def select(
        self, contrast: MechanismDataset, *, reference_index: int
    ) -> StateConditionedSourceDecision:
        reference = int(reference_index)
        augmented = self._augment_contrast(contrast, reference_index=reference)
        design = mechanism_parameter_design(augmented)
        expected_names = tuple(
            str(value) for value in self.payload["mechanism_feature_names"]
        )
        if design.feature_names != expected_names:
            raise ValueError("V150L mechanism design differs from the frozen model")
        scale = np.asarray(self.payload["feature_scale"], dtype=float)
        x = np.asarray(design.values, dtype=float) / scale
        target_score = x @ np.asarray(self.payload["target_beta"], dtype=float)
        rigid_score = _rigid_score(self.payload["rigid_model"], augmented)
        references = np.asarray(
            augmented.metadata.get("is_reference", ()), dtype=bool
        )
        source_betas = {
            str(key): np.asarray(value, dtype=float)
            for key, value in dict(self.payload["candidate_betas"]).items()
        }
        placebo_betas = {
            str(key): np.asarray(value, dtype=float)
            for key, value in dict(self.payload["placebo_betas"]).items()
        }
        betas = source_betas
        gate = self.payload["source_utility_gate"]
        if self.arm == "same_capacity_placebo_corrected_rigid":
            betas = placebo_betas
            gate = self.payload["placebo_utility_gate"]
        candidate_scores = {
            key: residual_corrected_rigid_score(
                rigid_score,
                x @ beta,
                target_score,
                references,
            )
            for key, beta in betas.items()
        }
        constraint_audit = None
        if self.arm != "rigid_target_only":
            if (
                self.candidate_score_constraint
                == PRESSURE_ALIGNED_CANDIDATE_PROTOCOL
            ):
                candidate_scores, constraint_audit = pressure_align_candidate_scores(
                    augmented,
                    rigid_score,
                    candidate_scores,
                )
            elif (
                self.candidate_score_constraint
                == PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL
            ):
                candidate_scores, constraint_audit = pressure_pairwise_candidate_scores(
                    augmented,
                    rigid_score,
                    candidate_scores,
                )
        application = {
            "selected_group_count": 0,
            "selected_candidate_counts": {},
        }
        should_apply_gate = bool(
            self.arm != "rigid_target_only" and self.payload["admitted"]
        )
        if should_apply_gate:
            score, application = apply_utility_gate(
                augmented,
                rigid_score,
                candidate_scores,
                gate,
            )
        else:
            score = np.asarray(rigid_score, dtype=float).copy()
        candidate_states = np.asarray(
            contrast.metadata.get("candidate_states", ()), dtype=str
        )
        proposed = self._argmin_with_reference_tie_break(
            np.asarray(score, dtype=float),
            reference=reference,
            candidate_states=candidate_states,
        )
        rigid_selected = self._argmin_with_reference_tie_break(
            np.asarray(rigid_score, dtype=float),
            reference=reference,
            candidate_states=candidate_states,
        )
        differs = proposed != reference
        selected_counts = dict(application.get("selected_candidate_counts", {}))
        selected_key = next(iter(selected_counts), None)
        utility_selected = bool(
            int(application.get("selected_group_count", 0)) > 0
        )
        source_changed_action = proposed != rigid_selected
        source_changed_to_reference = bool(
            source_changed_action and proposed == reference
        )
        source_changed_off_reference = bool(
            source_changed_action and proposed != reference
        )
        candidate_changed = int(
            (constraint_audit or {}).get("changed_rigid_group_count", 0)
        )
        candidate_aligned = int(
            (constraint_audit or {}).get(
                "pressure_aligned_change_group_count", 0
            )
        )
        candidate_rejected = int(
            (constraint_audit or {}).get(
                "off_reference_change_rejected_group_count", 0
            )
        )
        self._decision_count += 1
        self._override_count += int(differs)
        self._utility_selected_count += int(utility_selected)
        self._source_changed_action_count += int(source_changed_action)
        self._source_changed_to_reference_action_count += int(
            source_changed_to_reference
        )
        self._source_changed_off_reference_action_count += int(
            source_changed_off_reference
        )
        self._candidate_changed_rigid_group_count += candidate_changed
        self._candidate_pressure_aligned_change_group_count += candidate_aligned
        self._candidate_off_reference_rejected_group_count += candidate_rejected
        if selected_key is not None:
            self._candidate_counts[str(selected_key)] += 1
        return StateConditionedSourceDecision(
            reference_index=reference,
            proposed_index=proposed,
            selected_index=proposed,
            rigid_selected_index=rigid_selected,
            learned_differs=differs,
            eligible=differs,
            priority=max(float(score[reference] - score[proposed]), 0.0),
            rejection=None,
            arm=self.arm,
            source_city_admitted=bool(self.payload["admitted"]),
            utility_gate_selected=utility_selected,
            selected_candidate_key=(
                str(selected_key) if selected_key is not None else None
            ),
            source_changed_rigid_action=source_changed_action,
            predicted_score=float(score[proposed]),
            reference_score=float(score[reference]),
            rigid_predicted_score=float(rigid_score[rigid_selected]),
            candidate_score_constraint=self.candidate_score_constraint,
            candidate_changed_rigid_group_count=candidate_changed,
            candidate_pressure_aligned_change_group_count=candidate_aligned,
            candidate_off_reference_rejected_group_count=candidate_rejected,
            source_changed_to_reference_action=source_changed_to_reference,
            source_changed_off_reference_action=source_changed_off_reference,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": self.originator_protocol,
            "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
            "arm": self.arm,
            "city": str(self.payload["city"]),
            "target_budget": int(self.payload["target_budget"]),
            "source_city_admitted": bool(self.payload["admitted"]),
            "interval_count": int(self._interval_count),
            "decision_count": int(self._decision_count),
            "phase_pressure_override_count": int(self._override_count),
            "utility_gate_selected_count": int(self._utility_selected_count),
            "source_changed_rigid_action_count": int(
                self._source_changed_action_count
            ),
            "source_changed_to_reference_action_count": int(
                self._source_changed_to_reference_action_count
            ),
            "source_changed_off_reference_action_count": int(
                self._source_changed_off_reference_action_count
            ),
            "candidate_score_constraint": self.candidate_score_constraint,
            "candidate_changed_rigid_group_count": int(
                self._candidate_changed_rigid_group_count
            ),
            "candidate_pressure_aligned_change_group_count": int(
                self._candidate_pressure_aligned_change_group_count
            ),
            "candidate_off_reference_rejected_group_count": int(
                self._candidate_off_reference_rejected_group_count
            ),
            "selected_candidate_counts": dict(sorted(self._candidate_counts.items())),
            "mean_neighbor_observation_ratio": (
                float(np.mean(self._neighbor_observation_ratios))
                if self._neighbor_observation_ratios
                else 0.0
            ),
            "target_labels_used_at_fit": True,
            "target_outcomes_used_online": False,
            "source_identity_used_by_utility_gate": False,
        }


def _runtime_models(
    originator: StateConditionedSourceUtilityOriginator,
    *,
    prediction_horizon_sec: int,
) -> FittedContrastModelsV3:
    family = CFCMT_CORE_FAMILY_V3
    prior = pressure_spec("phase_pressure")
    return FittedContrastModelsV3(
        family_models={family: None},
        prior_policy=prior.key,
        prior_spec=prior,
        prediction_horizon_sec=int(prediction_horizon_sec),
        objective_modes={family: "control_only"},
        guards={},
        regularizers={},
        target_support=None,
        hierarchy_layers={},
        diagnostics={"runtime": originator.originator_protocol},
        action_originator=originator,
    )


def run_rollout(
    *,
    manifest_path: Path,
    conversion_root: Path,
    scenario: str,
    city: str,
    seed: int,
    arm: str,
    runtime_model_path: Path | None,
    expected_runtime_model_sha256: str | None,
    duration_sec: float,
    warmup_sec: float,
    control_interval_sec: int,
    prediction_horizon_sec: int,
    tripinfo_path: Path,
    expected_runtime_model_protocol: str = RUNTIME_MODEL_PROTOCOL,
    candidate_score_constraint: str | None = None,
    result_protocol: str = RESULT_PROTOCOL,
    originator_protocol: str = ORIGINATOR_PROTOCOL,
    scientific_status: str = "seven-city-closed-loop-development-rollout",
    runtime_policy: str = RUNTIME_POLICY,
) -> dict[str, Any]:
    started = time.monotonic()
    if arm not in POLICY_ARMS:
        raise ValueError(f"unknown V150L policy arm: {arm!r}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    manifest = load_traffic_signal_manifest(manifest_path)
    if (
        scenario not in manifest.sumocfgs
        or str(manifest.city_groups[scenario]) != str(city)
    ):
        raise ValueError("V150L scenario and city do not match the manifest")
    sumocfg = Path(manifest.sumocfgs[scenario]).resolve()

    originator = None
    models = None
    deployed_policy = "phase_pressure"
    model_sha = None
    if arm != "phase_pressure":
        if runtime_model_path is None or expected_runtime_model_sha256 is None:
            raise ValueError("V150L source arms require a frozen runtime model")
        network_context = read_network_right_of_way_context(
            net_file_from_sumocfg(sumocfg)
        )
        originator = StateConditionedSourceUtilityOriginator.load(
            runtime_model_path,
            expected_sha256=expected_runtime_model_sha256,
            expected_city=city,
            network_context=network_context,
            arm=arm,
            expected_runtime_protocol=expected_runtime_model_protocol,
            candidate_score_constraint=candidate_score_constraint,
            originator_protocol=originator_protocol,
        )
        models = _runtime_models(
            originator, prediction_horizon_sec=prediction_horizon_sec
        )
        deployed_policy = str(runtime_policy)
        model_sha = str(expected_runtime_model_sha256)

    tripinfo = Path(tripinfo_path)
    if tripinfo.exists():
        raise FileExistsError(f"refusing to overwrite V150L tripinfo: {tripinfo}")
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = evaluate_policy_v3(
            sumo_api=load_libsumo(),
            sumocfg=sumocfg,
            scenario=str(scenario),
            policy=deployed_policy,
            models=models,
            duration_sec=float(duration_sec),
            control_interval_sec=int(control_interval_sec),
            warmup_sec=float(warmup_sec),
            seed=int(seed),
            tripinfo_output=tripinfo,
            residual_coordination_mode="direct",
            residual_cooldown_intervals_override=0,
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"V150L rollout failed: {metrics.get('error')}")
        zero_incident_diagnostic = assess_zero_incident_diagnostic(metrics)
        if not tripinfo.is_file() or tripinfo.stat().st_size <= 0:
            raise RuntimeError("V150L rollout lacks tripinfo output")
    finally:
        tripinfo.unlink(missing_ok=True)
    return {
        "protocol": str(result_protocol),
        "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": str(scientific_status),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "sumo_version": libsumo_version(),
        "manifest_sha256": _sha256(manifest_path),
        "runtime_model_sha256": model_sha,
        "city": str(city),
        "scenario": str(scenario),
        "seed": int(seed),
        "arm": str(arm),
        "runtime_policy": deployed_policy,
        "duration_sec": float(duration_sec),
        "warmup_sec": float(warmup_sec),
        "control_interval_sec": int(control_interval_sec),
        "prediction_horizon_sec": (
            int(prediction_horizon_sec) if originator is not None else None
        ),
        "target_labels_used_at_fit": originator is not None,
        "target_outcomes_used_online": False,
        "zero_incident_diagnostic": zero_incident_diagnostic,
        "originator_diagnostics": (
            originator.diagnostics() if originator is not None else None
        ),
        "metrics": metrics,
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=POLICY_ARMS, required=True)
    parser.add_argument("--runtime-model", type=Path)
    parser.add_argument("--runtime-model-sha256")
    parser.add_argument("--duration-sec", type=float, default=3600.0)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--prediction-horizon-sec", type=int, default=450)
    parser.add_argument("--tripinfo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150L result: {args.out}")
    result = run_rollout(
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        scenario=str(args.scenario),
        city=str(args.city),
        seed=int(args.seed),
        arm=str(args.arm),
        runtime_model_path=args.runtime_model,
        expected_runtime_model_sha256=args.runtime_model_sha256,
        duration_sec=float(args.duration_sec),
        warmup_sec=float(args.warmup_sec),
        control_interval_sec=int(args.control_interval_sec),
        prediction_horizon_sec=int(args.prediction_horizon_sec),
        tripinfo_path=args.tripinfo,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "DONE",
                "city": result["city"],
                "scenario": result["scenario"],
                "seed": result["seed"],
                "arm": result["arm"],
                "mean_waiting_time": result["metrics"].get(
                    "mean_tripinfo_waiting_time"
                ),
                "zero_incident_observed": result["zero_incident_diagnostic"][
                    "passed"
                ],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
