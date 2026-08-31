"""Counterfactual action-contrast CFCMT for safe cross-network TSC.

V2 showed that accurate source-domain absolute transition prediction does not
guarantee transferable action rankings. V3 collects paired one-step source
counterfactuals from identical SUMO snapshots and learns only the candidate
minus rule-reference mechanism effects. Target networks still contribute no
transition labels: only static summaries and online local state are observed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.sumo_runtime import sumo_scratch_root, sumo_state_directory
from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import _parse_tripinfo_metrics, _scenario_summary
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import (
    CONTEXT_NAMES_V2,
    FEATURE_NAMES_V2,
    OUTPUT_NAMES_V2,
    LocalTransitionState,
    _behavior_candidate,
    _candidate_for_state,
    _candidate_aggregates,
    _controlled_lane_set,
    _lane_queue,
    _read_local_state,
    _rule_candidate,
    _start_sumo,
    _tls_phase_infos,
    candidate_features_v2,
    uncalibrated_mechanism_priors,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    SUMO_EXECUTION_PROTOCOL,
    _reload_sumo_state,
)
from cf_h2o.traffic_signal.action_contrast import (
    DEFAULT_CONTRAST_FEATURES,
    action_group_ids,
    build_action_contrast_dataset,
    make_target_action_contrasts,
    select_source_only_reference_policy,
)
from cf_h2o.traffic_signal.action_scaling import (
    GROUP_ACTION_REGRET_SCALE_PROTOCOL,
    action_group_range,
)
from cf_h2o.traffic_signal.dynamic_graph_context import (
    SIGNAL_GRAPH_ACTION_FEATURES,
    SIGNAL_GRAPH_FEATURES,
    SignalRoutingGraph,
    build_dynamic_signal_contexts,
    build_signal_routing_graph,
    candidate_graph_features,
)
from cf_h2o.traffic_signal.boosted_mechanism_world_model import (
    BoostedCausalTransferWorldModel,
    BoostedFitConfig,
    BoostedMechanismWorldModel,
)
from cf_h2o.traffic_signal.mechanism_world_model import (
    DenseResidualWorldModel,
    InvariantMechanismWorldModel,
    MechanismDataset,
    MechanismDefinition,
    MechanismFitConfig,
    PriorMechanismWorldModel,
)
from cf_h2o.traffic_signal.generalized_pressure import (
    PressurePolicySpec,
    pressure_scores_from_feature_matrix,
    pressure_spec,
)
from cf_h2o.traffic_signal.action_ranker import (
    AntisymmetricPairwiseActionRegressor,
    CAUSAL_CORE_RANKING_PARENTS,
    CAUSAL_RIGID_RANKING_PARENTS,
    ActionAdvantageConfig,
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
    PairwiseActionAdvantageRegressor,
    PairwiseActionRanker,
    TargetAdaptedActionAdvantageRegressor,
    TargetOnlyActionAdvantageRegressor,
    domain_normalized_action_target,
)
from cf_h2o.traffic_signal.network_context import TOPOLOGY_CONTEXT_NAMES, static_topology_context
from cf_h2o.traffic_signal.safe_phase_controller import (
    SafePhaseExecutor,
    aggregate_phase_audits,
    build_safe_phase_executors,
)
from cf_h2o.traffic_signal.target_action_support import (
    HierarchicalTargetActionSupport,
    TargetActionSupport,
)
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    BoostedDirectRolloutValueCausalMechanismAdvantageModel,
    CausalMechanismAdvantageConfig,
    CausalMechanismAdvantageModel,
    DirectRolloutValueCausalMechanismAdvantageModel,
    FusedCausalMechanismAdvantageModel,
    FusedPhysicalResidualCausalMechanismAdvantageModel,
    FusedRigidActionAdvantageModel,
    PhysicalResidualCausalMechanismAdvantageModel,
    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL,
    RolloutValueResidualCausalMechanismAdvantageModel,
    SOURCE_OOF_RIDGE_LCB_EXPERT_GATE_PROTOCOL,
)
from cf_h2o.traffic_signal.intervention_graph import (
    TlsInterventionGraph,
    add_routing_conflicts,
    build_tls_intervention_graph,
    greedy_independent_interventions,
)
from cf_h2o.traffic_signal.source_gated_pairwise import (
    SourceGatedPairwiseActionRegressor,
)


FOCAL_SELECTION_PROTOCOL_V3 = "least_covered_feasible_round_robin_v1"


STATE_SNAPSHOT_PROTOCOL_V3 = {
    "save_rng": True,
    "precision": 8,
    "thread_rngs": 1,
    "reload_mode": "full_network_load",
    "behavior_capture": "uninterrupted_first_pass",
    "branch_evaluation": "snapshot_full_reload_second_pass",
    "focal_selection": FOCAL_SELECTION_PROTOCOL_V3,
    "safe_phase_execution": (
        "min-green-yellow-all-red-full-junction-occupancy-clearance-v4"
    ),
}


STRICT_SAFETY_MONITORING_V3 = {
    "junction_collisions": "audit_raw_events_and_unique_incidents",
    "starting_teleports": "fail_fast",
    "ending_teleports": "fail_fast",
    "counterfactual_collision_handling": "censor_entire_matched_action_group",
    "deployment_collision_admission": "paired_baseline_noninferiority",
    "failure_mode": "teleport_fail_fast_collision_audit",
}


COUNTERFACTUAL_SAFETY_PROTOCOL_V3 = (
    "paired-action-group-symmetric-censor-on-any-branch-collision-v1"
)


CONTRAST_MECHANISM_DEFINITIONS = (
    MechanismDefinition(
        name="queue_propagation",
        target_name="next_total_queue",
        parent_variants={
            "minimal": (
                "total_q",
                "total_veh",
                "delta_green_q",
                "delta_green_down_q",
                "delta_switch_indicator",
                "delta_clearance_fraction",
            ),
            "physical": (
                "total_q",
                "total_veh",
                "mean_occ",
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "green_down_q",
                "reference_green_down_q",
                "delta_green_down_q",
                "service_pressure",
                "reference_service_pressure",
                "delta_service_pressure",
                "delta_switch_indicator",
                "delta_clearance_fraction",
                "graph_upstream_q_mean",
                "graph_downstream_q_mean",
                "graph_two_hop_q_mean",
                "graph_global_q_per_tls",
                "network_active_per_tls",
                "delta_green_signal_down_q_mean",
                "delta_green_corridor_pressure",
            ),
        },
        clip_min=-500.0,
        clip_max=500.0,
    ),
    MechanismDefinition(
        name="served_movement",
        target_name="next_green_queue",
        parent_variants={
            "minimal": (
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "green_veh",
                "delta_green_veh",
                "delta_clearance_fraction",
            ),
            "physical": (
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "green_veh",
                "reference_green_veh",
                "delta_green_veh",
                "green_occ",
                "delta_green_occ",
                "green_down_q",
                "delta_green_down_q",
                "service_pressure",
                "delta_service_pressure",
                "delta_switch_indicator",
                "delta_clearance_fraction",
                "graph_downstream_q_mean",
                "delta_green_signal_down_q_mean",
                "delta_green_corridor_pressure",
            ),
        },
        clip_min=-500.0,
        clip_max=500.0,
    ),
    MechanismDefinition(
        name="red_accumulation",
        target_name="next_red_queue",
        parent_variants={
            "minimal": (
                "red_q",
                "reference_red_q",
                "delta_red_q",
                "red_veh",
                "delta_red_veh",
            ),
            "physical": (
                "red_q",
                "reference_red_q",
                "delta_red_q",
                "red_veh",
                "reference_red_veh",
                "delta_red_veh",
                "red_occ",
                "delta_red_occ",
                "red_pressure",
                "delta_red_pressure",
                "delta_switch_indicator",
                "delta_clearance_fraction",
            ),
        },
        clip_min=-500.0,
        clip_max=500.0,
    ),
    MechanismDefinition(
        name="spillback",
        target_name="next_downstream_occupancy",
        parent_variants={
            "minimal": (
                "green_down_occ",
                "reference_green_down_occ",
                "delta_green_down_occ",
                "green_down_q",
                "delta_green_down_q",
            ),
            "physical": (
                "green_down_occ",
                "reference_green_down_occ",
                "delta_green_down_occ",
                "green_down_q",
                "reference_green_down_q",
                "delta_green_down_q",
                "green_q",
                "delta_green_q",
                "green_veh",
                "delta_green_veh",
                "service_pressure",
                "delta_service_pressure",
                "delta_clearance_fraction",
                "graph_downstream_q_mean",
                "graph_neighbor_occ_mean",
                "green_signal_down_occ_mean",
                "delta_green_signal_down_occ_mean",
                "delta_green_signal_down_occ_max",
            ),
        },
        clip_min=-100.0,
        clip_max=100.0,
    ),
    MechanismDefinition(
        name="mobility",
        target_name="next_mean_speed",
        parent_variants={
            "minimal": (
                "mean_speed",
                "total_q",
                "delta_green_q",
                "delta_clearance_fraction",
            ),
            "physical": (
                "mean_speed",
                "total_q",
                "mean_occ",
                "green_q",
                "delta_green_q",
                "green_down_occ",
                "delta_green_down_occ",
                "service_pressure",
                "delta_service_pressure",
                "delta_switch_indicator",
                "delta_clearance_fraction",
                "graph_neighbor_q_max",
                "graph_neighbor_occ_mean",
                "delta_green_signal_down_occ_mean",
            ),
        },
        clip_min=-30.0,
        clip_max=30.0,
    ),
    MechanismDefinition(
        name="terminal_clearance",
        target_name="terminal_system_load",
        parent_variants={
            "minimal": (
                "total_q",
                "network_active_per_tls",
                "delta_green_q",
                "delta_switch_indicator",
            ),
            "physical": (
                "total_q",
                "total_veh",
                "network_active_per_tls",
                "graph_global_q_per_tls",
                "graph_global_q_p90",
                "graph_neighbor_q_max",
                "green_q",
                "delta_green_q",
                "red_q",
                "delta_red_q",
                "delta_green_signal_down_q_mean",
                "delta_green_corridor_pressure",
                "delta_service_pressure",
                "delta_switch_indicator",
                "delta_clearance_fraction",
            ),
        },
        clip_min=-1000.0,
        clip_max=1000.0,
    ),
    MechanismDefinition(
        name="control_cost",
        target_name="interval_cost",
        parent_variants={
            "minimal": (
                "total_q",
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "delta_green_down_q",
                "delta_switch_indicator",
                "delta_clearance_fraction",
            ),
            "physical": (
                "total_q",
                "total_veh",
                "mean_occ",
                "mean_speed",
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "red_q",
                "delta_red_q",
                "green_down_q",
                "delta_green_down_q",
                "green_down_occ",
                "delta_green_down_occ",
                "service_pressure",
                "delta_service_pressure",
                "red_pressure",
                "delta_red_pressure",
                "delta_switch_indicator",
                "delta_clearance_fraction",
                "graph_upstream_q_mean",
                "graph_downstream_q_mean",
                "graph_neighbor_q_max",
                "graph_two_hop_q_mean",
                "graph_global_q_per_tls",
                "graph_global_q_p90",
                "graph_focal_q_percentile",
                "network_active_per_tls",
                "green_signal_down_q_mean",
                "delta_green_signal_down_q_mean",
                "delta_green_signal_down_occ_mean",
                "delta_green_signal_reach_ratio",
                "delta_green_corridor_pressure",
            ),
        },
        clip_min=-1000.0,
        clip_max=1000.0,
    ),
)

ONE_STEP_TARGET_BY_TERMINAL_TARGET_V4 = {
    "next_total_queue": "one_step_total_queue",
    "next_green_queue": "one_step_green_queue",
    "next_red_queue": "one_step_red_queue",
    "next_downstream_occupancy": "one_step_downstream_occupancy",
    "next_mean_speed": "one_step_mean_speed",
}

ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4 = tuple(
    replace(
        definition,
        target_name=ONE_STEP_TARGET_BY_TERMINAL_TARGET_V4.get(
            definition.target_name,
            definition.target_name,
        ),
    )
    for definition in CONTRAST_MECHANISM_DEFINITIONS
)

CONTROL_COST_CONTRAST_DEFINITION = tuple(
    definition for definition in CONTRAST_MECHANISM_DEFINITIONS if definition.name == "control_cost"
)
PHYSICAL_COMPONENT_DEFINITION_NAMES_V3 = {
    "cfcmt_physical_queue_only": ("queue_propagation",),
    "cfcmt_physical_red_only": ("red_accumulation",),
    "cfcmt_physical_spillback_only": ("spillback",),
    "cfcmt_physical_served_only": ("served_movement",),
    "cfcmt_physical_mobility_only": ("mobility",),
    "cfcmt_physical_mobility_queue": ("mobility", "queue_propagation"),
    "cfcmt_physical_mobility_red": ("mobility", "red_accumulation"),
    "cfcmt_physical_mobility_spillback": ("mobility", "spillback"),
    "cfcmt_physical_mobility_served": ("mobility", "served_movement"),
}
PHYSICAL_COMPONENT_FAMILIES_V3 = tuple(PHYSICAL_COMPONENT_DEFINITION_NAMES_V3)
ONE_STEP_PHYSICAL_COMPONENT_DEFINITION_NAMES_V4 = {
    "cfcmt_one_step_physical_queue_only": ("queue_propagation",),
    "cfcmt_one_step_physical_red_only": ("red_accumulation",),
    "cfcmt_one_step_physical_spillback_only": ("spillback",),
    "cfcmt_one_step_physical_served_only": ("served_movement",),
    "cfcmt_one_step_physical_mobility_only": ("mobility",),
    "cfcmt_one_step_physical_mobility_queue": (
        "mobility",
        "queue_propagation",
    ),
    "cfcmt_one_step_physical_mobility_red": (
        "mobility",
        "red_accumulation",
    ),
    "cfcmt_one_step_physical_mobility_spillback": ("mobility", "spillback"),
    "cfcmt_one_step_physical_mobility_served": (
        "mobility",
        "served_movement",
    ),
}
ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4 = tuple(
    ONE_STEP_PHYSICAL_COMPONENT_DEFINITION_NAMES_V4
)
ONE_STEP_PHYSICAL_FULL_FAMILY_V4 = "cfcmt_one_step_physical_mechanism"
ONE_STEP_RIGID_RESIDUAL_COMPONENT_DEFINITION_NAMES_V5 = {
    family.replace("cfcmt_one_step_physical_", "cfcmt_one_step_rigid_residual_"): names
    for family, names in ONE_STEP_PHYSICAL_COMPONENT_DEFINITION_NAMES_V4.items()
}
ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5 = tuple(
    ONE_STEP_RIGID_RESIDUAL_COMPONENT_DEFINITION_NAMES_V5
)
ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5 = (
    "cfcmt_one_step_rigid_residual_mechanism"
)
ONE_STEP_SELECTIVE_RESIDUAL_FAMILY_CONFIG_V6 = {
    f"cfcmt_one_step_selective_residual_{base}_q{int(quantile * 100):02d}": (
        mechanism_names,
        quantile,
    )
    for base, mechanism_names in {
        "mobility": ("mobility",),
        "mobility_queue": ("mobility", "queue_propagation"),
    }.items()
    for quantile in (0.50, 0.75, 0.90)
}
ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6 = tuple(
    ONE_STEP_SELECTIVE_RESIDUAL_FAMILY_CONFIG_V6
)
ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7 = (
    "cfcmt_rollout_value_rigid_residual"
)
DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8 = (
    "cfcmt_direct_rollout_value_rigid_residual"
)
BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9 = (
    "cfcmt_boosted_direct_rollout_value_rigid_residual"
)
GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10 = (
    "causal_group_normalized_rigid_advantage"
)
ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11 = (
    "causal_antisymmetric_pairwise_advantage"
)
SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12 = (
    "cfcmt_source_gated_pairwise_advantage"
)
MODEL_FAMILIES_V3 = (
    "simulator",
    "dense",
    "sparse",
    "cfcmt",
    "dense_boosted",
    "causal_boosted",
    "cfcmt_boosted",
    "dense_ranker",
    "causal_ranker",
    "dense_advantage",
    "dense_balanced_advantage",
    "causal_advantage",
    "causal_core_advantage",
    "causal_balanced_advantage",
    "causal_rigid_advantage",
    "cfcmt_mechanism",
    "cfcmt_fused",
    "cfcmt_fused_rigid",
    "cfcmt_physical_mechanism",
    "cfcmt_physical_fused",
    *PHYSICAL_COMPONENT_FAMILIES_V3,
    ONE_STEP_PHYSICAL_FULL_FAMILY_V4,
    *ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4,
    ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5,
    *ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5,
    *ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6,
    ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
    DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
    BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
    GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10,
    ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11,
    SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12,
    "cfcmt_physical_latent_mechanism",
    "cfcmt_physical_latent_fused",
    "causal_target_adapter",
    "causal_target_only",
)

NORMALIZED_ACTION_UNIT_FAMILIES_V3 = frozenset(
    {
        "dense_advantage",
        "dense_balanced_advantage",
        "causal_advantage",
        "causal_core_advantage",
        "causal_balanced_advantage",
        "causal_rigid_advantage",
        "cfcmt_mechanism",
        "cfcmt_fused",
        "cfcmt_fused_rigid",
        "cfcmt_physical_mechanism",
        "cfcmt_physical_fused",
        *PHYSICAL_COMPONENT_FAMILIES_V3,
        ONE_STEP_PHYSICAL_FULL_FAMILY_V4,
        *ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4,
        ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5,
        *ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5,
        *ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6,
        ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
        DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
        BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
        GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10,
        ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11,
        SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12,
        "cfcmt_physical_latent_mechanism",
        "cfcmt_physical_latent_fused",
        "causal_target_adapter",
        "causal_target_only",
    }
)
CFCMT_CORE_FAMILY_V3 = "causal_rigid_advantage"
CFCMT_TARGET_SPECIALIST_FAMILY_V3 = "causal_target_only"
CFCMT_HIERARCHY_PROTOCOL_V3 = (
    "disjoint-target-calibrated-single-expert-mechanism-target-rigid-pressure-v2"
)
CFCMT_HIERARCHY_SELECTION_PROTOCOL_V3 = (
    "disjoint-target-group-anchor-first-single-expert-v1"
)
CFCMT_HIERARCHY_LAYERS_V3 = (
    "mechanism_refinement",
    "target_specialist",
    "causal_core_fallback",
)

OBJECTIVE_WEIGHTS_V3: Mapping[str, Mapping[str, float]] = {
    "control_only": {"control_cost": 1.0},
    "queue_only": {"queue_propagation": 1.0},
    "queue_spillback": {
        "queue_propagation": 1.0,
        "spillback": 0.025,
        "mobility": -0.02,
    },
    "factorized_v2": {
        "control_cost": 1.0,
        "queue_propagation": 0.12,
        "red_accumulation": 0.04,
        "spillback": 0.015,
        "mobility": -0.015,
    },
    "causal_composite": {
        "control_cost": 0.40,
        "queue_propagation": 0.60,
        "red_accumulation": 0.15,
        "spillback": 0.025,
        "mobility": -0.02,
    },
}

CONTEXT_NAMES_V3 = (*CONTEXT_NAMES_V2, *TOPOLOGY_CONTEXT_NAMES)
FEATURE_NAMES_V3 = (*FEATURE_NAMES_V2, *SIGNAL_GRAPH_FEATURES)
CONTRAST_FEATURES_V3 = (*DEFAULT_CONTRAST_FEATURES, *SIGNAL_GRAPH_ACTION_FEATURES)
OUTPUT_NAMES_V3 = (*OUTPUT_NAMES_V2, "terminal_system_load")
ONE_STEP_OUTPUT_NAMES_V4 = tuple(ONE_STEP_TARGET_BY_TERMINAL_TARGET_V4.values())
OUTPUT_NAMES_V4 = (*OUTPUT_NAMES_V3, *ONE_STEP_OUTPUT_NAMES_V4)
MULTIHORIZON_PREFIX_TARGET_PREFIX = "prefix_mean_cost_"
POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4 = (
    "one-control-interval-local-mechanisms-plus-pressure-rollout-value-v1"
)
WAITING_ALIGNED_ESTIMAND_PROTOCOL_V5 = (
    "one-control-interval-local-mechanisms-plus-halted-queue-rollout-value-v1"
)
COUNTERFACTUAL_COST_MODES_V5 = (
    "system_vehicle_load",
    "halted_queue",
)
MIN_CONTEXT_SUPPORT_V3 = 0.10
MIN_OPERATIONAL_TOTAL_VEHICLES_V3 = 1.0


def counterfactual_cost_contract_v5(mode: str) -> dict[str, str]:
    value = str(mode)
    if value == "halted_queue":
        return {
            "mode": value,
            "estimand_protocol": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V5,
            "scope": "global_halted_vehicles_per_controlled_lane",
            "population": "sumo_last_step_halting_number_on_controlled_lanes",
            "normalization": "controlled_lane_count",
        }
    if value == "system_vehicle_load":
        return {
            "mode": value,
            "estimand_protocol": POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
            "scope": "global_system_vehicles_per_controlled_lane",
            "population": "active_inserted_plus_pending_insertion_vehicles",
            "normalization": "controlled_lane_count",
        }
    raise ValueError(f"unknown counterfactual cost mode: {value}")


def rollout_prefix_target_name(horizon_sec: int) -> str:
    horizon = int(horizon_sec)
    if horizon <= 0:
        raise ValueError("rollout prefix horizon must be positive")
    return f"{MULTIHORIZON_PREFIX_TARGET_PREFIX}{horizon}s"


def normalize_rollout_prefix_horizons(
    horizons_sec: Sequence[int] | None,
    *,
    rollout_horizon_sec: int,
) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in (horizons_sec or ())}))
    if any(value <= 0 for value in horizons):
        raise ValueError("rollout prefix horizons must be positive")
    if any(value > int(rollout_horizon_sec) for value in horizons):
        raise ValueError("rollout prefix horizon exceeds the collected rollout")
    return horizons


def rollout_prefix_cost_targets(
    interval_costs: Sequence[float],
    horizons_sec: Sequence[int],
) -> dict[str, float]:
    costs = np.asarray(interval_costs, dtype=float)
    if costs.ndim != 1 or not np.all(np.isfinite(costs)):
        raise ValueError("rollout interval costs must be a finite vector")
    horizons = normalize_rollout_prefix_horizons(
        horizons_sec,
        rollout_horizon_sec=int(costs.size),
    )
    return {
        rollout_prefix_target_name(horizon): float(np.mean(costs[:horizon]))
        for horizon in horizons
    }


@dataclass(frozen=True)
class ContrastGuardConfig:
    enabled: bool = False
    risk_multiplier: float = 1.0
    min_context_trust: float = 0.0
    margin: float = 0.0
    max_relative_rule_gap: float = 1.0


@dataclass(frozen=True)
class PriorRegularizationConfig:
    enabled: bool = False
    blend_weight: float = 0.0
    risk_multiplier: float = 0.0
    min_context_trust: float = 0.0


def hierarchical_guard_decision_v3(
    *,
    learned_differs: bool,
    predicted_delta: float,
    uncertainty: float,
    relative_rule_gap: float,
    pressure_context_trust: float,
    conformal_context_trust: float,
    total_vehicles: float,
    regularizer: PriorRegularizationConfig,
    guard: ContrastGuardConfig,
) -> dict[str, float | bool | str | None]:
    """Apply a frozen pressure gate and, when enabled, a conformal gate."""

    pressure_objective = (
        float(relative_rule_gap)
        + float(regularizer.blend_weight) * float(predicted_delta)
        + float(regularizer.risk_multiplier) * float(uncertainty)
    )
    conformal_upper = (
        float(predicted_delta)
        + float(guard.risk_multiplier) * float(uncertainty)
        + float(guard.margin)
    )
    rejection = None
    if not learned_differs:
        rejection = "agreement"
    elif not regularizer.enabled:
        rejection = "disabled"
    elif float(pressure_context_trust) < float(regularizer.min_context_trust):
        rejection = "pressure_support"
    elif float(total_vehicles) < MIN_OPERATIONAL_TOTAL_VEHICLES_V3:
        rejection = "service_safety"
    elif pressure_objective >= 0.0:
        rejection = "pressure_confidence"
    elif guard.enabled and float(conformal_context_trust) < float(
        guard.min_context_trust
    ):
        rejection = "conformal_support"
    elif guard.enabled and float(relative_rule_gap) > float(
        guard.max_relative_rule_gap
    ) + 1e-12:
        rejection = "conformal_service_safety"
    elif guard.enabled and conformal_upper >= 0.0:
        rejection = "conformal_confidence"
    eligible = rejection is None
    priority = (
        min(-pressure_objective, -conformal_upper)
        if eligible and guard.enabled
        else (-pressure_objective if eligible else 0.0)
    )
    return {
        "eligible": bool(eligible),
        "rejection": rejection,
        "priority": float(max(priority, 0.0)),
        "pressure_objective": float(pressure_objective),
        "conformal_upper": float(conformal_upper),
    }


@dataclass(frozen=True)
class ResidualExecutionTrustRegionConfig:
    """Observable-state gate applied after model selection and before execution."""

    enabled: bool = False
    min_priority: float = 0.0
    min_total_queue: float = 0.0
    min_total_vehicles: float = 0.0
    max_mean_speed: float | None = None
    max_simultaneous_overrides: int | None = None
    global_cooldown_intervals: int = 0
    allow_stay_during_global_cooldown: bool = False
    max_stay_overrides_per_global_cooldown: int | None = None

    def __post_init__(self) -> None:
        if (
            float(self.min_priority) < 0.0
            or float(self.min_total_queue) < 0.0
            or float(self.min_total_vehicles) < 0.0
            or (
                self.max_mean_speed is not None
                and float(self.max_mean_speed) < 0.0
            )
            or (
                self.max_simultaneous_overrides is not None
                and int(self.max_simultaneous_overrides) < 1
            )
            or int(self.global_cooldown_intervals) < 0
            or (
                self.max_stay_overrides_per_global_cooldown is not None
                and int(self.max_stay_overrides_per_global_cooldown) < 0
            )
        ):
            raise ValueError("invalid residual execution trust-region threshold")


def _minimum_intervention_gap_sec(
    cooldown_intervals: int, control_interval_sec: int
) -> int:
    """Return the earliest next control epoch after a cooldown is installed."""

    if int(cooldown_intervals) < 0 or int(control_interval_sec) <= 0:
        raise ValueError("invalid intervention cooldown timing")
    return (int(cooldown_intervals) + 1) * int(control_interval_sec)


@dataclass
class ContrastGuardAudit:
    decisions: int = 0
    warmup_prior_decisions: int = 0
    prior_agreements: int = 0
    proposed_overrides: int = 0
    accepted_overrides: int = 0
    executed_overrides: int = 0
    proposed_switch_overrides: int = 0
    proposed_stay_overrides: int = 0
    accepted_switch_overrides: int = 0
    accepted_stay_overrides: int = 0
    executed_switch_overrides: int = 0
    executed_stay_overrides: int = 0
    rejected_executor: int = 0
    rejected_disabled: int = 0
    rejected_context: int = 0
    rejected_local_support: int = 0
    rejected_confidence: int = 0
    rejected_service_safety: int = 0
    rejected_minimum_priority: int = 0
    rejected_minimum_queue: int = 0
    rejected_minimum_vehicles: int = 0
    rejected_maximum_speed: int = 0
    rejected_global_cooldown: int = 0
    rejected_global_stay_budget: int = 0
    rejected_long_horizon_veto: int = 0
    rejected_cooldown: int = 0
    rejected_coordination: int = 0
    hierarchical_decisions: int = 0
    target_originator_decisions: int = 0
    selected_mechanism_refinements: int = 0
    selected_target_specialists: int = 0
    selected_causal_core_fallbacks: int = 0
    selected_target_originators: int = 0
    hierarchical_prior_fallbacks: int = 0

    def to_dict(self) -> dict[str, float | int]:
        proposed = max(self.proposed_overrides, 1)
        decisions = max(self.decisions, 1)
        return {
            **asdict(self),
            "effective_overrides": (
                self.executed_switch_overrides + self.executed_stay_overrides
            ),
            "agreement_rate": self.prior_agreements / decisions,
            "proposal_rate": self.proposed_overrides / decisions,
            "acceptance_rate": self.accepted_overrides / proposed,
        }


@dataclass
class FittedContrastModelsV3:
    family_models: dict[str, Any]
    prior_policy: str
    prior_spec: PressurePolicySpec
    prediction_horizon_sec: int
    objective_modes: dict[str, str]
    guards: dict[str, ContrastGuardConfig]
    regularizers: dict[str, PriorRegularizationConfig]
    target_support: TargetActionSupport | HierarchicalTargetActionSupport | None
    hierarchy_layers: dict[str, tuple[str, ...]]
    diagnostics: dict[str, Any]
    execution_veto: Any | None = None
    action_originator: Any | None = None


@dataclass
class _UncertaintyScaledModelV3:
    base_model: Any
    scale: float

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        prediction = self.base_model.predict(dataset)
        return {
            mechanism: {
                **values,
                "uncertainty": np.asarray(values["uncertainty"], dtype=float)
                * float(self.scale),
            }
            for mechanism, values in prediction.items()
        }


@dataclass(frozen=True)
class ContrastProposal:
    prior_candidate: Any
    selected_candidate: Any
    learned_differs: bool
    eligible: bool
    priority: float
    rejection: str | None = None
    selection_layer: str = "single"
    veto_diagnostics: Mapping[str, Any] | None = None
    originator_diagnostics: Mapping[str, Any] | None = None


def _candidate_phase_state(candidate: Any) -> str:
    return str(getattr(candidate, "state", candidate))


def _override_execution_kind(
    proposal: ContrastProposal, current_green_state: str
) -> str | None:
    """Classify an eligible residual by whether it induces a phase transition."""

    if not proposal.learned_differs or not proposal.eligible:
        return None
    prior_state = _candidate_phase_state(proposal.prior_candidate)
    selected_state = _candidate_phase_state(proposal.selected_candidate)
    if selected_state == prior_state:
        return None
    if selected_state == str(current_green_state):
        return "stay"
    return "switch"


def _restore_executors(
    executors: Mapping[str, SafePhaseExecutor],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    for tls_id in sorted(executors):
        executors[tls_id].restore(snapshots[tls_id])


def _collision_debug_samples(
    *,
    sumo_api: Any,
    collisions: Sequence[Any],
    executors: Mapping[str, SafePhaseExecutor] | None = None,
    time_sec: float | None = None,
    max_samples: int = 4,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    vehicle_ids = {str(vehicle_id) for vehicle_id in sumo_api.vehicle.getIDList()}
    tls_ids = {str(tls_id) for tls_id in sumo_api.trafficlight.getIDList()}
    for collision in tuple(collisions)[: max(int(max_samples), 0)]:
        sample: dict[str, Any] = {
            "time": float(
                sumo_api.simulation.getTime() if time_sec is None else time_sec
            ),
            "repr": repr(collision),
        }
        for name in (
            "collider",
            "victim",
            "colliderType",
            "victimType",
            "colliderSpeed",
            "victimSpeed",
            "type",
            "lane",
            "pos",
        ):
            value = getattr(collision, name, None)
            if value is not None:
                sample[name] = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else str(value)
                )
        participant_lanes: set[str] = set()
        participants: dict[str, dict[str, Any]] = {}
        for role in ("collider", "victim"):
            vehicle_id = str(getattr(collision, role, ""))
            if not vehicle_id or vehicle_id not in vehicle_ids:
                continue
            lane_id = str(sumo_api.vehicle.getLaneID(vehicle_id))
            participant_lanes.add(lane_id)
            position = sumo_api.vehicle.getPosition(vehicle_id)
            participants[role] = {
                "vehicle_id": vehicle_id,
                "lane_id": lane_id,
                "road_id": str(sumo_api.vehicle.getRoadID(vehicle_id)),
                "speed": float(sumo_api.vehicle.getSpeed(vehicle_id)),
                "lane_position": float(sumo_api.vehicle.getLanePosition(vehicle_id)),
                "position": [float(position[0]), float(position[1])],
            }
        sample["participants"] = participants
        collision_lane = str(sample.get("lane", ""))
        if collision_lane:
            participant_lanes.add(collision_lane)
        relevant_tls = sorted(
            tls_id
            for tls_id in tls_ids
            if any(lane_id.startswith(f":{tls_id}_") for lane_id in participant_lanes)
        )
        sample["relevant_tls_states"] = {
            tls_id: str(sumo_api.trafficlight.getRedYellowGreenState(tls_id))
            for tls_id in relevant_tls
        }
        if executors is not None:
            sample["relevant_executor_states"] = {
                tls_id: executors[tls_id].snapshot()
                for tls_id in relevant_tls
                if tls_id in executors
            }
        samples.append(sample)
    return samples


class UnsafeRolloutError(RuntimeError):
    """A rollout violated a hard constraint or a branch safety constraint."""

    def __init__(self, message: str, safety_audit: Mapping[str, Any]):
        super().__init__(message)
        self.safety_audit = dict(safety_audit)


def _new_safety_ledger() -> dict[str, Any]:
    return {
        "raw_collision_events": 0,
        "collision_event_steps": 0,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "collision_samples": [],
        "_collision_incident_keys": set(),
    }


def _collision_incident_key(collision: Any) -> tuple[str, str, str, str]:
    participants = sorted(
        (
            str(getattr(collision, "collider", "")),
            str(getattr(collision, "victim", "")),
        )
    )
    return (
        participants[0],
        participants[1],
        str(getattr(collision, "type", "")),
        str(getattr(collision, "lane", "")),
    )


def _record_safety_step(
    *,
    ledger: dict[str, Any],
    sumo_api: Any,
    executors: Mapping[str, SafePhaseExecutor],
    starting_teleports: int,
    ending_teleports: int,
    collisions: Sequence[Any],
) -> None:
    ledger["starting_teleports"] += int(starting_teleports)
    ledger["ending_teleports"] += int(ending_teleports)
    ledger["raw_collision_events"] += len(collisions)
    ledger["collision_event_steps"] += int(bool(collisions))
    ledger["_collision_incident_keys"].update(
        _collision_incident_key(collision) for collision in collisions
    )
    remaining_samples = max(12 - len(ledger["collision_samples"]), 0)
    if remaining_samples:
        ledger["collision_samples"].extend(
            _collision_debug_samples(
                sumo_api=sumo_api,
                collisions=collisions,
                executors=executors,
                time_sec=float(sumo_api.simulation.getTime()),
                max_samples=remaining_samples,
            )
        )


def _finalize_safety_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    incident_keys = ledger.get("_collision_incident_keys", set())
    return {
        "raw_collision_events": int(ledger.get("raw_collision_events", 0)),
        "collision_event_steps": int(ledger.get("collision_event_steps", 0)),
        "unique_collision_incidents": int(len(incident_keys)),
        "starting_teleports": int(ledger.get("starting_teleports", 0)),
        "ending_teleports": int(ledger.get("ending_teleports", 0)),
        "collision_samples": list(ledger.get("collision_samples", ()))[:12],
        "no_teleport_passed": bool(
            int(ledger.get("starting_teleports", 0)) == 0
            and int(ledger.get("ending_teleports", 0)) == 0
        ),
    }


def _advance_with_actions(
    *,
    sumo_api: Any,
    executors: Mapping[str, SafePhaseExecutor],
    actions: Mapping[str, str],
    control_interval_sec: int,
    cost_lanes: Sequence[str] | None = None,
    system_cost_lane_count: int | None = None,
    collision_mode: str = "fail",
    safety_ledger: dict[str, Any] | None = None,
) -> list[float]:
    if collision_mode not in {"fail", "audit"}:
        raise ValueError(f"unknown collision handling mode: {collision_mode}")
    ledger = safety_ledger if safety_ledger is not None else _new_safety_ledger()
    for tls_id in sorted(actions):
        executors[tls_id].request(actions[tls_id])
    costs: list[float] = []
    for _ in range(int(control_interval_sec)):
        before = float(sumo_api.simulation.getTime())
        sumo_api.simulationStep()
        elapsed = max(float(sumo_api.simulation.getTime()) - before, 0.0)
        for tls_id in sorted(executors):
            executors[tls_id].advance(elapsed)
        starting_teleports = int(sumo_api.simulation.getStartingTeleportNumber())
        ending_teleports = int(sumo_api.simulation.getEndingTeleportNumber())
        collisions = tuple(sumo_api.simulation.getCollisions())
        _record_safety_step(
            ledger=ledger,
            sumo_api=sumo_api,
            executors=executors,
            starting_teleports=starting_teleports,
            ending_teleports=ending_teleports,
            collisions=collisions,
        )
        if starting_teleports or ending_teleports:
            audit = _finalize_safety_ledger(ledger)
            raise UnsafeRolloutError(
                "teleport in SUMO rollout at "
                f"time={float(sumo_api.simulation.getTime()):.3f}: "
                f"starting_teleports={starting_teleports}, "
                f"ending_teleports={ending_teleports}, "
                f"safety_audit={json.dumps(audit, sort_keys=True)}",
                audit,
            )
        if collisions and collision_mode == "fail":
            audit = _finalize_safety_ledger(ledger)
            raise UnsafeRolloutError(
                "junction collision in counterfactual rollout at "
                f"time={float(sumo_api.simulation.getTime()):.3f}: "
                f"collisions={len(collisions)}, "
                f"safety_audit={json.dumps(audit, sort_keys=True)}",
                audit,
            )
        if system_cost_lane_count is not None:
            active = float(sumo_api.vehicle.getIDCount())
            pending = float(len(sumo_api.simulation.getPendingVehicles()))
            costs.append((active + pending) / max(int(system_cost_lane_count), 1))
        elif cost_lanes is not None:
            costs.append(float(sum(_lane_queue(sumo_api, lane) for lane in cost_lanes)))
    return costs


def _counterfactual_branch_outcome_v3(
    *,
    sumo_api: Any,
    infos: Mapping[str, Any],
    states: Mapping[str, LocalTransitionState],
    executors: Mapping[str, SafePhaseExecutor],
    focal_id: str,
    candidate: Any,
    context: np.ndarray,
    control_interval_sec: int,
    counterfactual_horizon_intervals: int,
    controlled_lane_count: int,
    controlled_lanes: Sequence[str] | None = None,
    counterfactual_cost_mode: str = "system_vehicle_load",
    safety_ledger: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, float], dict[str, float]]:
    mode = counterfactual_cost_contract_v5(counterfactual_cost_mode)["mode"]
    if mode == "halted_queue":
        cost_lanes = tuple(str(value) for value in (controlled_lanes or ()))
        if not cost_lanes:
            raise ValueError("halted-queue counterfactual requires controlled lanes")
        advance_cost_args = {
            "cost_lanes": cost_lanes,
            "system_cost_lane_count": None,
        }
        cost_divisor = float(max(int(controlled_lane_count), 1))
    else:
        advance_cost_args = {
            "cost_lanes": None,
            "system_cost_lane_count": int(controlled_lane_count),
        }
        cost_divisor = 1.0
    actions = {
        tls_id: _rule_candidate(
            states[tls_id],
            executors[tls_id],
            "phase_pressure",
        ).state
        for tls_id in sorted(infos)
    }
    actions[focal_id] = candidate.state
    interval_costs = [
        float(value) / cost_divisor
        for value in _advance_with_actions(
            sumo_api=sumo_api,
            executors=executors,
            actions=actions,
            control_interval_sec=control_interval_sec,
            collision_mode="fail",
            safety_ledger=safety_ledger,
            **advance_cost_args,
        )
    ]
    one_step_state = _read_local_state(
        sumo_api,
        focal_id,
        infos[focal_id],
        context,
    )
    one_step_aggregate = _candidate_aggregates(one_step_state, candidate)
    for _ in range(max(int(counterfactual_horizon_intervals) - 1, 0)):
        rollout_states = {
            tls_id: _read_local_state(
                sumo_api,
                tls_id,
                info,
                context,
            )
            for tls_id, info in sorted(infos.items())
        }
        rollout_actions = {
            tls_id: _rule_candidate(
                rollout_states[tls_id],
                executors[tls_id],
                "phase_pressure",
            ).state
            for tls_id in sorted(infos)
        }
        interval_costs.extend(
            float(value) / cost_divisor
            for value in _advance_with_actions(
                sumo_api=sumo_api,
                executors=executors,
                actions=rollout_actions,
                control_interval_sec=control_interval_sec,
                collision_mode="fail",
                safety_ledger=safety_ledger,
                **advance_cost_args,
            )
        )
    terminal_state = _read_local_state(
        sumo_api,
        focal_id,
        infos[focal_id],
        context,
    )
    return (
        interval_costs,
        one_step_aggregate,
        _candidate_aggregates(terminal_state, candidate),
    )


def _counterfactual_outcome_vector(
    interval_costs: Sequence[float],
    one_step_aggregate: Mapping[str, float],
    terminal_aggregate: Mapping[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            *interval_costs,
            one_step_aggregate["total_q"],
            one_step_aggregate["green_q"],
            one_step_aggregate["red_q"],
            one_step_aggregate["green_down_occ"],
            one_step_aggregate["mean_speed"],
            terminal_aggregate["total_q"],
            terminal_aggregate["green_q"],
            terminal_aggregate["red_q"],
            terminal_aggregate["green_down_occ"],
            terminal_aggregate["mean_speed"],
        ],
        dtype=float,
    )


def _scenario_context_v3(
    *,
    sumocfg: Path,
    infos: Mapping[str, Any],
    control_interval_sec: int,
) -> np.ndarray:
    base = _scenario_summary(
        sumocfg=sumocfg,
        infos=dict(infos),
        control_interval_sec=control_interval_sec,
    )
    topology = static_topology_context(sumocfg, infos)
    context = np.concatenate([base, topology])
    if context.shape != (len(CONTEXT_NAMES_V3),):
        raise ValueError("V3 scenario context has an invalid shape")
    return context


def _read_graph_states_v3(
    sumo_api: Any,
    infos: Mapping[str, Any],
    context: np.ndarray,
    routing_graph: SignalRoutingGraph,
) -> dict[str, LocalTransitionState]:
    states = {
        str(tls_id): _read_local_state(sumo_api, str(tls_id), info, context)
        for tls_id, info in sorted(infos.items())
    }
    graph_contexts = build_dynamic_signal_contexts(
        states,
        infos,
        active_vehicles=float(sumo_api.vehicle.getIDCount()),
        routing_graph=routing_graph,
    )
    return {
        tls_id: replace(state, graph_context=graph_contexts[tls_id])
        for tls_id, state in states.items()
    }


def _least_covered_focal_tls_v3(
    *,
    eligible_tls: Sequence[str],
    controllable_tls: Sequence[str],
    selection_counts: Mapping[str, int],
    opportunity_index: int,
    max_focal_tls: int,
) -> tuple[str, ...]:
    """Select feasible signals while balancing cumulative TLS coverage."""

    controllable = tuple(str(tls_id) for tls_id in controllable_tls)
    eligible = {str(tls_id) for tls_id in eligible_tls}
    if not controllable or not eligible:
        return ()
    unknown = eligible - set(controllable)
    if unknown:
        raise ValueError(f"eligible TLS are absent from controllable set: {sorted(unknown)}")
    count = min(max(int(max_focal_tls), 1), len(eligible))
    start = (int(opportunity_index) * count) % len(controllable)
    static_rank = {tls_id: index for index, tls_id in enumerate(controllable)}
    ordered = sorted(
        eligible,
        key=lambda tls_id: (
            int(selection_counts.get(tls_id, 0)),
            (static_rank[tls_id] - start) % len(controllable),
            tls_id,
        ),
    )
    return tuple(ordered[:count])


def candidate_features_v3(
    state: LocalTransitionState,
    candidate: Any,
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
) -> np.ndarray:
    local = candidate_features_v2(
        state,
        candidate,
        executor,
        control_interval_sec=control_interval_sec,
    )
    aggregate = _candidate_aggregates(state, candidate)
    graph = candidate_graph_features(
        state.graph_context,
        state.info,
        candidate.state,
        green_queue=float(aggregate["green_q"]),
    )
    features = np.concatenate([local, graph])
    if features.shape != (len(FEATURE_NAMES_V3),):
        raise ValueError("V3 candidate feature vector has an invalid shape")
    return features


def uncalibrated_mechanism_priors_v3(
    features: np.ndarray,
    context: np.ndarray,
    *,
    control_interval_sec: int,
) -> dict[str, np.ndarray]:
    priors = uncalibrated_mechanism_priors(
        features,
        context,
        control_interval_sec=control_interval_sec,
    )
    priors["terminal_system_load"] = np.asarray(
        priors["interval_cost"], dtype=float
    ).copy()
    return priors


def policy_consistent_mechanism_priors_v4(
    features: np.ndarray,
    context: np.ndarray,
    *,
    control_interval_sec: int,
    counterfactual_horizon_intervals: int,
) -> dict[str, np.ndarray]:
    """Build analytic priors at the same horizon as each observed estimand."""

    horizon_intervals = max(int(counterfactual_horizon_intervals), 1)
    horizon_priors = uncalibrated_mechanism_priors_v3(
        features,
        context,
        control_interval_sec=int(control_interval_sec) * horizon_intervals,
    )
    one_step_priors = uncalibrated_mechanism_priors_v3(
        features,
        context,
        control_interval_sec=int(control_interval_sec),
    )
    return {
        **horizon_priors,
        **{
            one_step_name: np.asarray(
                one_step_priors[terminal_name],
                dtype=float,
            ).copy()
            for terminal_name, one_step_name in (
                ONE_STEP_TARGET_BY_TERMINAL_TARGET_V4.items()
            )
        },
    }


def collect_counterfactual_transitions_v3(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    max_focal_tls: int = 4,
    counterfactual_horizon_intervals: int = 3,
    counterfactual_cost_mode: str = "system_vehicle_load",
    behavior_policy: str = "phase_pressure",
    collection_shard_index: int = 0,
    collection_shard_count: int = 1,
    rollout_prefix_horizons_sec: Sequence[int] | None = None,
) -> MechanismDataset:
    """Collect matched actions from an uninterrupted trace, then branch in pass two."""

    cost_contract = counterfactual_cost_contract_v5(counterfactual_cost_mode)
    if int(collection_shard_count) < 1:
        raise ValueError("collection_shard_count must be positive")
    if not 0 <= int(collection_shard_index) < int(collection_shard_count):
        raise ValueError("collection_shard_index must be in [0, collection_shard_count)")
    rollout_horizon_sec = int(control_interval_sec) * max(
        int(counterfactual_horizon_intervals), 1
    )
    prefix_horizons = normalize_rollout_prefix_horizons(
        rollout_prefix_horizons_sec,
        rollout_horizon_sec=rollout_horizon_sec,
    )
    prefix_target_names = tuple(
        rollout_prefix_target_name(value) for value in prefix_horizons
    )
    output_names = (*OUTPUT_NAMES_V4, *prefix_target_names)
    _start_sumo(sumo_api, sumocfg, seed)
    rows_features: list[np.ndarray] = []
    rows_context: list[np.ndarray] = []
    rows_targets: dict[str, list[float]] = {name: [] for name in output_names}
    row_groups: list[str] = []
    row_tls: list[str] = []
    row_times: list[float] = []
    row_candidate_states: list[str] = []
    executors: dict[str, SafePhaseExecutor] = {}
    intervention_graph: TlsInterventionGraph | None = None
    branch_count = 0
    candidate_group_count = 0
    retained_group_count = 0
    censored_group_count = 0
    candidate_branch_count = 0
    attempted_branch_count = 0
    observed_unsafe_branch_count = 0
    censored_branch_count = 0
    censored_group_ids: list[str] = []
    counterfactual_collision_samples: list[dict[str, Any]] = []
    counterfactual_raw_collision_events = 0
    counterfactual_collision_event_steps = 0
    counterfactual_collision_incident_signatures: set[str] = set()
    restore_count = 0
    replay_checks = 0
    replay_max_abs_difference = 0.0
    try:
        infos = _tls_phase_infos(sumo_api)
        routing_graph = build_signal_routing_graph(sumo_api, infos)
        intervention_graph = add_routing_conflicts(
            build_tls_intervention_graph(infos), routing_graph.adjacency
        )
        controlled_lanes = tuple(sorted(_controlled_lane_set(infos)))
        controlled_lane_count = max(len(controlled_lanes), 1)
        controllable_tls = tuple(
            sorted(tls_id for tls_id, info in infos.items() if len(info.candidates) > 1)
        )
        focal_selection_counts = {tls_id: 0 for tls_id in controllable_tls}
        context = _scenario_context_v3(
            sumocfg=sumocfg,
            infos=infos,
            control_interval_sec=control_interval_sec,
        )
        executors = build_safe_phase_executors(sumo_api, infos)
        if behavior_policy not in {"phase_pressure", "exploratory_mixture"}:
            raise ValueError(f"unknown counterfactual behavior policy: {behavior_policy}")
        rng = np.random.default_rng(seed + 3701)
        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + float(duration_sec)
        interval_idx = 0
        collection_opportunity_idx = 0
        behavior_trace = hashlib.sha256()
        behavior_safety_ledger = _new_safety_ledger()
        behavior_phase_execution_audit: dict[str, Any] = {}
        captured_snapshots: list[dict[str, Any]] = []
        scratch_root = sumo_scratch_root()
        with sumo_state_directory(prefix=f"cfcmt_v3_{scenario}_") as temp_dir:
            # Pass 1 is never reloaded. This keeps the behavior trajectory
            # independent of which counterfactual shard this process owns.
            while (
                float(sumo_api.simulation.getTime()) < end_time
                and int(sumo_api.simulation.getMinExpectedNumber()) > 0
            ):
                states = _read_graph_states_v3(
                    sumo_api, infos, context, routing_graph
                )
                behavior = (
                    {
                        tls_id: _rule_candidate(
                            states[tls_id],
                            executors[tls_id],
                            "phase_pressure",
                        )
                        for tls_id in sorted(infos)
                    }
                    if behavior_policy == "phase_pressure"
                    else {
                        tls_id: _behavior_candidate(states[tls_id], executors[tls_id], rng)
                        for tls_id in sorted(infos)
                    }
                )
                behavior_states = {tls_id: candidate.state for tls_id, candidate in behavior.items()}
                now = float(sumo_api.simulation.getTime())
                behavior_trace.update(np.asarray([now, float(interval_idx)], dtype="<f8").tobytes())
                for tls_id in sorted(infos):
                    behavior_trace.update(str(tls_id).encode("utf-8"))
                    behavior_trace.update(b"\0")
                    behavior_trace.update(behavior_states[tls_id].encode("utf-8"))
                    behavior_trace.update(b"\0")
                    behavior_trace.update(
                        np.asarray(
                            candidate_features_v3(
                                states[tls_id],
                                behavior[tls_id],
                                executors[tls_id],
                                control_interval_sec=control_interval_sec,
                            ),
                            dtype="<f8",
                        ).tobytes()
                    )
                eligible = [
                    tls_id
                    for tls_id in sorted(infos)
                    if len(executors[tls_id].feasible_states_now()) > 1
                ]
                focal_ids: list[str] = []
                if now >= begin_time + float(warmup_sec) and eligible:
                    selected_focal_ids = _least_covered_focal_tls_v3(
                        eligible_tls=eligible,
                        controllable_tls=controllable_tls,
                        selection_counts=focal_selection_counts,
                        opportunity_index=collection_opportunity_idx,
                        max_focal_tls=max_focal_tls,
                    )
                    for tls_id in selected_focal_ids:
                        focal_selection_counts[tls_id] += 1
                    assigned_shard = collection_opportunity_idx % int(collection_shard_count)
                    opportunity_idx = collection_opportunity_idx
                    collection_opportunity_idx += 1
                    if assigned_shard == int(collection_shard_index):
                        focal_ids = list(selected_focal_ids)

                if focal_ids:
                    snapshots = {tls_id: executor.snapshot() for tls_id, executor in executors.items()}
                    state_path = temp_dir / f"counterfactual_state_{opportunity_idx:06d}.xml"
                    sumo_api.simulation.saveState(str(state_path))
                    candidates_by_tls = {
                        focal_id: tuple(
                            _candidate_for_state(infos[focal_id], state)
                            for state in executors[focal_id].feasible_states_now()
                        )
                        for focal_id in focal_ids
                    }
                    candidate_rows_by_tls = {
                        focal_id: {
                            candidate.state: candidate_features_v3(
                                states[focal_id],
                                candidate,
                                executors[focal_id],
                                control_interval_sec=control_interval_sec,
                            )
                            for candidate in candidates_by_tls[focal_id]
                        }
                        for focal_id in focal_ids
                    }
                    captured_snapshots.append(
                        {
                            "state_path": state_path,
                            "now": now,
                            "interval_idx": interval_idx,
                            "opportunity_idx": opportunity_idx,
                            "focal_ids": tuple(focal_ids),
                            "states": states,
                            "executor_snapshots": snapshots,
                            "candidates_by_tls": candidates_by_tls,
                            "candidate_rows_by_tls": candidate_rows_by_tls,
                        }
                    )

                remaining = max(int(math.ceil(end_time - float(sumo_api.simulation.getTime()))), 0)
                if remaining <= 0:
                    break
                _advance_with_actions(
                    sumo_api=sumo_api,
                    executors=executors,
                    actions=behavior_states,
                    control_interval_sec=min(int(control_interval_sec), remaining),
                    collision_mode="audit",
                    safety_ledger=behavior_safety_ledger,
                )
                interval_idx += 1

            behavior_phase_execution_audit = aggregate_phase_audits(executors)

            # Pass 2 branches only from snapshots generated by the untouched
            # first pass. Reloads can no longer perturb later behavior states.
            for captured in captured_snapshots:
                state_path = Path(captured["state_path"])
                now = float(captured["now"])
                captured_interval_idx = int(captured["interval_idx"])
                focal_ids = tuple(str(value) for value in captured["focal_ids"])
                states = captured["states"]
                snapshots = captured["executor_snapshots"]
                for focal_id in focal_ids:
                    group = f"{scenario}:seed{int(seed)}:{captured_interval_idx}:{focal_id}"
                    candidates = captured["candidates_by_tls"][focal_id]
                    candidate_rows = captured["candidate_rows_by_tls"][focal_id]
                    candidate_group_count += 1
                    candidate_branch_count += len(candidates)
                    buffered_rows: list[dict[str, Any]] = []
                    unsafe_group = False
                    for candidate_idx, candidate in enumerate(candidates):
                        _reload_sumo_state(
                            sumo_api,
                            sumocfg,
                            seed,
                            state_path,
                            begin_time=now,
                        )
                        _restore_executors(executors, snapshots)
                        restore_count += 1
                        attempted_branch_count += 1
                        branch_safety_ledger = _new_safety_ledger()
                        try:
                            (
                                interval_costs,
                                one_step_aggregate,
                                terminal_aggregate,
                            ) = (
                                _counterfactual_branch_outcome_v3(
                                    sumo_api=sumo_api,
                                    infos=infos,
                                    states=states,
                                    executors=executors,
                                    focal_id=focal_id,
                                    candidate=candidate,
                                    context=context,
                                    control_interval_sec=control_interval_sec,
                                    counterfactual_horizon_intervals=(
                                        counterfactual_horizon_intervals
                                    ),
                                    controlled_lane_count=controlled_lane_count,
                                    controlled_lanes=controlled_lanes,
                                    counterfactual_cost_mode=counterfactual_cost_mode,
                                    safety_ledger=branch_safety_ledger,
                                )
                            )
                        except UnsafeRolloutError as error:
                            audit = dict(error.safety_audit)
                            if int(audit.get("starting_teleports", 0)) or int(
                                audit.get("ending_teleports", 0)
                            ):
                                raise RuntimeError(
                                    "hard teleport failure in counterfactual branch for "
                                    f"scenario={scenario}, seed={int(seed)}, "
                                    f"snapshot_time={now:.3f}, group={group}, "
                                    f"focal_tls={focal_id}, candidate_index={candidate_idx}, "
                                    f"candidate_state={candidate.state!r}: {error}"
                                ) from error
                            unsafe_group = True
                            observed_unsafe_branch_count += 1
                            counterfactual_raw_collision_events += int(
                                audit.get("raw_collision_events", 0)
                            )
                            counterfactual_collision_event_steps += int(
                                audit.get("collision_event_steps", 0)
                            )
                            for sample in audit.get("collision_samples", ()):
                                signature_payload = (
                                    str(scenario),
                                    int(seed),
                                    float(sample.get("time", now)),
                                    str(sample.get("collider", "")),
                                    str(sample.get("victim", "")),
                                    str(sample.get("type", "")),
                                    str(sample.get("lane", "")),
                                )
                                counterfactual_collision_incident_signatures.add(
                                    hashlib.sha256(
                                        json.dumps(
                                            signature_payload,
                                            separators=(",", ":"),
                                        ).encode("utf-8")
                                    ).hexdigest()[:20]
                                )
                                if len(counterfactual_collision_samples) < 20:
                                    counterfactual_collision_samples.append(
                                        {
                                            "group": group,
                                            "focal_tls": focal_id,
                                            "candidate_index": int(candidate_idx),
                                            "candidate_state": candidate.state,
                                            **dict(sample),
                                        }
                                    )
                            break
                        should_audit = (
                            candidate_idx == 0
                            and (
                                replay_checks == 0
                                or int(captured["opportunity_idx"]) % 12 == 0
                            )
                        )
                        if should_audit:
                            _reload_sumo_state(
                                sumo_api,
                                sumocfg,
                                seed,
                                state_path,
                                begin_time=now,
                            )
                            _restore_executors(executors, snapshots)
                            restore_count += 1
                            (
                                replay_costs,
                                replay_one_step_aggregate,
                                replay_terminal_aggregate,
                            ) = _counterfactual_branch_outcome_v3(
                                sumo_api=sumo_api,
                                infos=infos,
                                states=states,
                                executors=executors,
                                focal_id=focal_id,
                                candidate=candidate,
                                context=context,
                                control_interval_sec=control_interval_sec,
                                counterfactual_horizon_intervals=(
                                    counterfactual_horizon_intervals
                                ),
                                controlled_lane_count=controlled_lane_count,
                                controlled_lanes=controlled_lanes,
                                counterfactual_cost_mode=counterfactual_cost_mode,
                                safety_ledger=_new_safety_ledger(),
                            )
                            observed = _counterfactual_outcome_vector(
                                interval_costs,
                                one_step_aggregate,
                                terminal_aggregate,
                            )
                            replayed = _counterfactual_outcome_vector(
                                replay_costs,
                                replay_one_step_aggregate,
                                replay_terminal_aggregate,
                            )
                            difference = float(np.max(np.abs(observed - replayed)))
                            difference_index = int(np.argmax(np.abs(observed - replayed)))
                            difference_label = (
                                f"system_load_second_{difference_index}"
                                if difference_index < len(interval_costs)
                                else (
                                    "one_step_total_queue",
                                    "one_step_green_queue",
                                    "one_step_red_queue",
                                    "one_step_downstream_occupancy",
                                    "one_step_mean_speed",
                                    "next_total_queue",
                                    "next_green_queue",
                                    "next_red_queue",
                                    "next_downstream_occupancy",
                                    "next_mean_speed",
                                )[difference_index - len(interval_costs)]
                            )
                            replay_checks += 1
                            replay_max_abs_difference = max(
                                replay_max_abs_difference, difference
                            )
                            if not np.allclose(observed, replayed, rtol=0.0, atol=1e-8):
                                raise RuntimeError(
                                    "SUMO counterfactual replay is not deterministic for "
                                    f"{scenario} seed={seed} time={now} tls={focal_id}; "
                                    f"max_abs_difference={difference:.6g} at {difference_label} "
                                    f"({observed[difference_index]:.9g} vs "
                                    f"{replayed[difference_index]:.9g})"
                                )
                        buffered_rows.append(
                            {
                                "features": candidate_rows[candidate.state],
                                "one_step_total_queue": one_step_aggregate[
                                    "total_q"
                                ],
                                "one_step_green_queue": one_step_aggregate[
                                    "green_q"
                                ],
                                "one_step_red_queue": one_step_aggregate["red_q"],
                                "one_step_downstream_occupancy": one_step_aggregate[
                                    "green_down_occ"
                                ],
                                "one_step_mean_speed": one_step_aggregate[
                                    "mean_speed"
                                ],
                                "next_total_queue": terminal_aggregate["total_q"],
                                "next_green_queue": terminal_aggregate["green_q"],
                                "next_red_queue": terminal_aggregate["red_q"],
                                "next_downstream_occupancy": terminal_aggregate[
                                    "green_down_occ"
                                ],
                                "next_mean_speed": terminal_aggregate["mean_speed"],
                                "interval_cost": float(np.mean(interval_costs)),
                                "terminal_system_load": float(interval_costs[-1]),
                                **rollout_prefix_cost_targets(
                                    interval_costs,
                                    prefix_horizons,
                                ),
                                "candidate_state": candidate.state,
                            }
                        )
                    if unsafe_group:
                        censored_group_count += 1
                        censored_branch_count += len(candidates)
                        censored_group_ids.append(group)
                        continue
                    if len(buffered_rows) != len(candidates):
                        raise AssertionError(
                            f"incomplete safe action group {group}: "
                            f"{len(buffered_rows)} of {len(candidates)} branches"
                        )
                    retained_group_count += 1
                    for buffered in buffered_rows:
                        rows_features.append(buffered["features"])
                        rows_context.append(context)
                        for target_name in output_names:
                            rows_targets[target_name].append(buffered[target_name])
                        row_groups.append(group)
                        row_tls.append(focal_id)
                        row_times.append(now)
                        row_candidate_states.append(buffered["candidate_state"])
                        branch_count += 1
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass

    if retained_group_count + censored_group_count != candidate_group_count:
        raise AssertionError("counterfactual action-group accounting is inconsistent")
    if branch_count + censored_branch_count != candidate_branch_count:
        raise AssertionError("counterfactual branch accounting is inconsistent")
    behavior_safety_audit = _finalize_safety_ledger(behavior_safety_ledger)
    counterfactual_safety_audit = {
        "protocol": COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
        "candidate_groups": int(candidate_group_count),
        "retained_groups": int(retained_group_count),
        "censored_groups": int(censored_group_count),
        "candidate_branches": int(candidate_branch_count),
        "attempted_branches": int(attempted_branch_count),
        "retained_branches": int(branch_count),
        "censored_branches": int(censored_branch_count),
        "unevaluated_censored_branches": int(
            candidate_branch_count - attempted_branch_count
        ),
        "observed_unsafe_branches": int(observed_unsafe_branch_count),
        "raw_collision_events_observed": int(counterfactual_raw_collision_events),
        "collision_event_steps_observed": int(counterfactual_collision_event_steps),
        "unique_collision_incidents_observed": int(
            len(counterfactual_collision_incident_signatures)
        ),
        "collision_incident_signatures": sorted(
            counterfactual_collision_incident_signatures
        ),
        "censored_group_fraction": float(
            censored_group_count / max(candidate_group_count, 1)
        ),
        "censored_group_ids": list(censored_group_ids),
        "collision_samples": list(counterfactual_collision_samples),
        "starting_teleports": 0,
        "ending_teleports": 0,
        "no_teleport_passed": True,
        "symmetric_group_censoring_passed": bool(
            branch_count + censored_branch_count == candidate_branch_count
        ),
    }
    features = np.vstack(rows_features) if rows_features else np.zeros((0, len(FEATURE_NAMES_V3)))
    contexts = np.vstack(rows_context) if rows_context else np.zeros((0, len(CONTEXT_NAMES_V3)))
    priors = policy_consistent_mechanism_priors_v4(
        features,
        contexts,
        control_interval_sec=int(control_interval_sec),
        counterfactual_horizon_intervals=int(counterfactual_horizon_intervals),
    )
    for horizon, target_name in zip(prefix_horizons, prefix_target_names):
        priors[target_name] = np.asarray(
            uncalibrated_mechanism_priors_v3(
                features,
                contexts,
                control_interval_sec=int(horizon),
            )["interval_cost"],
            dtype=float,
        ).copy()
    return MechanismDataset(
        feature_names=FEATURE_NAMES_V3,
        features=features,
        context_names=CONTEXT_NAMES_V3,
        context=contexts,
        priors=priors,
        targets={name: np.asarray(values, dtype=float) for name, values in rows_targets.items()},
        domains=np.asarray([scenario] * features.shape[0]),
        metadata={
            "scenario": scenario,
            "sumocfg": str(sumocfg),
            "seed": int(seed),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "max_focal_tls": int(max_focal_tls),
            "counterfactual_horizon_intervals": int(counterfactual_horizon_intervals),
            "counterfactual_horizon_sec": int(control_interval_sec)
            * max(int(counterfactual_horizon_intervals), 1),
            "action_group_ids": row_groups,
            "row_tls": row_tls,
            "row_times": row_times,
            "candidate_states": row_candidate_states,
            "controllable_tls_ids": list(controllable_tls),
            "controllable_tls_count": len(controllable_tls),
            "covered_tls_ids": sorted(set(row_tls)),
            "covered_tls_count": len(set(row_tls)),
            "tls_coverage_fraction": len(set(row_tls)) / max(len(controllable_tls), 1),
            "counterfactual_branches": int(branch_count),
            "state_restores": int(restore_count),
            "state_scratch_root": str(scratch_root),
            "state_snapshot_protocol": dict(STATE_SNAPSHOT_PROTOCOL_V3),
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "behavior_trace_sha256": behavior_trace.hexdigest(),
            "behavior_interval_count": int(interval_idx),
            "captured_snapshot_count": len(captured_snapshots),
            "counterfactual_replay_audit": {
                "checks": int(replay_checks),
                "period_intervals": 12,
                "absolute_tolerance": 1e-8,
                "max_abs_difference": float(replay_max_abs_difference),
                "not_applicable": bool(replay_checks == 0),
                "passed": bool(
                    (replay_checks > 0 and replay_max_abs_difference <= 1e-8)
                    or (branch_count == 0 and replay_checks == 0)
                ),
            },
            "phase_execution_audit": behavior_phase_execution_audit,
            "behavior_safety_audit": behavior_safety_audit,
            "counterfactual_safety_audit": counterfactual_safety_audit,
            "protocol": "matched_source_counterfactual_symmetric_safety_censor_v6",
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "counterfactual_estimand": cost_contract["estimand_protocol"],
            "mechanism_output_names": list(output_names),
            "local_mechanism_horizon_sec": int(control_interval_sec),
            "rollout_value_horizon_sec": int(control_interval_sec)
            * max(int(counterfactual_horizon_intervals), 1),
            "rollout_prefix_horizons_sec": list(prefix_horizons),
            "mechanism_estimands": {
                "one_step_local_state": {
                    "output_names": list(ONE_STEP_OUTPUT_NAMES_V4),
                    "horizon_sec": int(control_interval_sec),
                    "policy": "candidate_action_for_first_control_interval",
                    "analytic_prior_horizon_sec": int(control_interval_sec),
                },
                "legacy_terminal_local_state": {
                    "output_names": list(OUTPUT_NAMES_V2[:-1]),
                    "horizon_sec": int(control_interval_sec)
                    * max(int(counterfactual_horizon_intervals), 1),
                    "policy": "candidate_action_then_phase_pressure",
                    "analytic_prior_horizon_sec": int(control_interval_sec)
                    * max(int(counterfactual_horizon_intervals), 1),
                },
                "rollout_action_value": {
                    "output_names": ["interval_cost", "terminal_system_load"],
                    "horizon_sec": int(control_interval_sec)
                    * max(int(counterfactual_horizon_intervals), 1),
                    "policy": "candidate_action_then_phase_pressure",
                    "cost_mode": cost_contract["mode"],
                },
                "rollout_prefix_action_values": {
                    "output_names": list(prefix_target_names),
                    "horizons_sec": list(prefix_horizons),
                    "aggregation": "arithmetic_mean_of_one_second_cost_samples_from_branch_start",
                    "policy": "candidate_action_then_phase_pressure",
                    "cost_mode": cost_contract["mode"],
                },
            },
            "counterfactual_cost_mode": cost_contract["mode"],
            "counterfactual_cost_scope": cost_contract["scope"],
            "counterfactual_cost_population": cost_contract["population"],
            "counterfactual_cost_normalization": cost_contract["normalization"],
            "intervention_graph": intervention_graph.to_dict() if intervention_graph is not None else {},
            "signal_routing_graph": {
                "adjacency": {
                    name: list(neighbors)
                    for name, neighbors in routing_graph.adjacency.items()
                },
                "edge_count": int(
                    sum(len(neighbors) for neighbors in routing_graph.adjacency.values())
                ),
            },
            "behavior_policy": behavior_policy,
            "collection_shard_index": int(collection_shard_index),
            "collection_shard_count": int(collection_shard_count),
            "eligible_collection_opportunities": int(collection_opportunity_idx),
            "focal_selection_protocol": FOCAL_SELECTION_PROTOCOL_V3,
            "focal_selection_count_by_tls": {
                tls_id: int(focal_selection_counts[tls_id])
                for tls_id in sorted(focal_selection_counts)
            },
            "focal_selection_min_count": min(focal_selection_counts.values(), default=0),
            "focal_selection_max_count": max(focal_selection_counts.values(), default=0),
        },
    )


def _merge_behavior_safety_audits_v3(
    datasets: Sequence[MechanismDataset],
) -> dict[str, Any]:
    audits = [dict(item.metadata.get("behavior_safety_audit", {})) for item in datasets]
    if any(not audit for audit in audits):
        raise ValueError("counterfactual dataset is missing behavior safety audit")
    identities = {
        (str(item.metadata.get("scenario", "")), int(item.metadata.get("seed", -1)))
        for item in datasets
    }
    if len(identities) == 1:
        encoded = {json.dumps(audit, sort_keys=True) for audit in audits}
        if len(encoded) != 1:
            raise ValueError("counterfactual shards disagree on behavior safety audit")
        return dict(audits[0])
    return {
        "raw_collision_events": int(
            sum(int(audit.get("raw_collision_events", 0)) for audit in audits)
        ),
        "collision_event_steps": int(
            sum(int(audit.get("collision_event_steps", 0)) for audit in audits)
        ),
        "unique_collision_incidents": int(
            sum(int(audit.get("unique_collision_incidents", 0)) for audit in audits)
        ),
        "starting_teleports": int(
            sum(int(audit.get("starting_teleports", 0)) for audit in audits)
        ),
        "ending_teleports": int(
            sum(int(audit.get("ending_teleports", 0)) for audit in audits)
        ),
        "collision_samples": [
            sample
            for audit in audits
            for sample in audit.get("collision_samples", ())
        ][:20],
        "no_teleport_passed": all(
            bool(audit.get("no_teleport_passed", False)) for audit in audits
        ),
        "source_audit_count": len(audits),
    }


def _merge_counterfactual_safety_audits_v3(
    datasets: Sequence[MechanismDataset],
) -> dict[str, Any]:
    audits = [
        dict(item.metadata.get("counterfactual_safety_audit", {}))
        for item in datasets
    ]
    if any(
        audit.get("protocol") != COUNTERFACTUAL_SAFETY_PROTOCOL_V3
        for audit in audits
    ):
        raise ValueError("counterfactual dataset safety-censor protocol mismatch")
    integer_fields = (
        "candidate_groups",
        "retained_groups",
        "censored_groups",
        "candidate_branches",
        "attempted_branches",
        "retained_branches",
        "censored_branches",
        "unevaluated_censored_branches",
        "observed_unsafe_branches",
        "raw_collision_events_observed",
        "collision_event_steps_observed",
        "starting_teleports",
        "ending_teleports",
    )
    totals = {
        name: int(sum(int(audit.get(name, 0)) for audit in audits))
        for name in integer_fields
    }
    censored_group_ids = [
        str(group)
        for audit in audits
        for group in audit.get("censored_group_ids", ())
    ]
    if len(censored_group_ids) != len(set(censored_group_ids)):
        raise ValueError("counterfactual censored action-group ids overlap")
    collision_incident_signatures = sorted(
        {
            str(signature)
            for audit in audits
            for signature in audit.get("collision_incident_signatures", ())
        }
    )
    merged = {
        "protocol": COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
        **totals,
        "unique_collision_incidents_observed": len(
            collision_incident_signatures
        ),
        "collision_incident_signatures": collision_incident_signatures,
        "censored_group_fraction": float(
            totals["censored_groups"] / max(totals["candidate_groups"], 1)
        ),
        "censored_group_ids": censored_group_ids,
        "collision_samples": [
            sample
            for audit in audits
            for sample in audit.get("collision_samples", ())
        ][:20],
        "no_teleport_passed": all(
            bool(audit.get("no_teleport_passed", False)) for audit in audits
        ),
        "symmetric_group_censoring_passed": all(
            bool(audit.get("symmetric_group_censoring_passed", False))
            for audit in audits
        ),
        "source_audit_count": len(audits),
    }
    if (
        totals["candidate_groups"]
        != totals["retained_groups"] + totals["censored_groups"]
        or totals["candidate_branches"]
        != totals["retained_branches"] + totals["censored_branches"]
    ):
        raise ValueError("counterfactual merged safety accounting is inconsistent")
    return merged


def merge_counterfactual_datasets_v3(datasets: Sequence[MechanismDataset]) -> MechanismDataset:
    if not datasets:
        raise ValueError("cannot merge an empty counterfactual dataset sequence")
    first = datasets[0]
    output_names = tuple(
        str(name)
        for name in first.metadata.get(
            "mechanism_output_names",
            tuple(first.targets),
        )
    )
    if (
        not output_names
        or set(output_names) != set(first.targets)
        or set(output_names) != set(first.priors)
    ):
        raise ValueError("counterfactual dataset has an invalid output schema")
    seen_groups: set[str] = set()
    for dataset_index, item in enumerate(datasets):
        if item.feature_names != first.feature_names or item.context_names != first.context_names:
            raise ValueError("counterfactual datasets use incompatible schemas")
        item_output_names = tuple(
            str(name)
            for name in item.metadata.get(
                "mechanism_output_names",
                tuple(item.targets),
            )
        )
        if (
            item_output_names != output_names
            or set(item.targets) != set(output_names)
            or set(item.priors) != set(output_names)
        ):
            raise ValueError("counterfactual datasets use incompatible output schemas")
        if int(item.metadata.get("counterfactual_horizon_sec", -1)) != int(
            first.metadata.get("counterfactual_horizon_sec", -1)
        ):
            raise ValueError("counterfactual datasets use incompatible prediction horizons")
        if int(item.metadata.get("local_mechanism_horizon_sec", -1)) != int(
            first.metadata.get("local_mechanism_horizon_sec", -1)
        ):
            raise ValueError(
                "counterfactual datasets use incompatible local mechanism horizons"
            )
        if item.metadata.get("mechanism_estimands") != first.metadata.get(
            "mechanism_estimands"
        ):
            raise ValueError("counterfactual datasets use incompatible estimands")
        if item.metadata.get("counterfactual_cost_scope") != first.metadata.get(
            "counterfactual_cost_scope"
        ):
            raise ValueError("counterfactual datasets use incompatible cost scopes")
        if item.metadata.get("counterfactual_cost_mode") != first.metadata.get(
            "counterfactual_cost_mode"
        ):
            raise ValueError("counterfactual datasets use incompatible cost modes")
        if tuple(item.metadata.get("rollout_prefix_horizons_sec", ())) != tuple(
            first.metadata.get("rollout_prefix_horizons_sec", ())
        ):
            raise ValueError("counterfactual datasets use incompatible prefix horizons")
        if item.metadata.get("behavior_policy") != first.metadata.get("behavior_policy"):
            raise ValueError("counterfactual datasets use incompatible behavior policies")
        if item.metadata.get("state_snapshot_protocol") != first.metadata.get(
            "state_snapshot_protocol"
        ):
            raise ValueError("counterfactual datasets use incompatible snapshot protocols")
        if item.metadata.get("sumo_execution_protocol") != first.metadata.get(
            "sumo_execution_protocol"
        ):
            raise ValueError("counterfactual datasets use incompatible SUMO execution protocols")
        if item.metadata.get("strict_safety_monitoring") != first.metadata.get(
            "strict_safety_monitoring"
        ):
            raise ValueError("counterfactual datasets use incompatible safety monitoring protocols")
        replay_audit = item.metadata.get("counterfactual_replay_audit", {})
        if not replay_audit.get("passed", False):
            raise ValueError("counterfactual dataset failed deterministic replay audit")
        safety_audit = item.metadata.get("counterfactual_safety_audit", {})
        safety_protocol_matches = (
            safety_audit.get("protocol") == COUNTERFACTUAL_SAFETY_PROTOCOL_V3
        )
        no_teleport_passed = bool(safety_audit.get("no_teleport_passed", False))
        symmetric_censoring_passed = bool(
            safety_audit.get("symmetric_group_censoring_passed", False)
        )
        retained_branches = int(safety_audit.get("retained_branches", -1))
        if (
            not safety_protocol_matches
            or not no_teleport_passed
            or not symmetric_censoring_passed
            or retained_branches != item.size
        ):
            raise ValueError(
                "counterfactual dataset failed symmetric safety audit: "
                f"dataset_index={dataset_index}, "
                f"protocol_matches={safety_protocol_matches}, "
                f"no_teleport_passed={no_teleport_passed}, "
                f"symmetric_group_censoring_passed={symmetric_censoring_passed}, "
                f"retained_branches={retained_branches}, rows={item.size}"
            )
        groups = {str(value) for value in item.metadata.get("action_group_ids", ())}
        overlap = seen_groups & groups
        if overlap:
            raise ValueError(f"counterfactual action-group ids collide across datasets: {sorted(overlap)[:3]}")
        seen_groups.update(groups)
    aligned_metadata: dict[str, list[Any]] = {}
    for key in ("row_tls", "row_times", "candidate_states"):
        present = [key in item.metadata for item in datasets]
        if any(present) and not all(present):
            raise ValueError(f"counterfactual datasets have incomplete row metadata: {key}")
        if not all(present):
            continue
        values: list[Any] = []
        for item in datasets:
            item_values = list(item.metadata[key])
            if len(item_values) != item.size:
                raise ValueError(f"counterfactual row metadata is misaligned: {key}")
            values.extend(item_values)
        aligned_metadata[key] = values

    return MechanismDataset(
        feature_names=first.feature_names,
        features=np.vstack([item.features for item in datasets]),
        context_names=first.context_names,
        context=np.vstack([item.context for item in datasets]),
        priors={
            name: np.concatenate([item.priors[name] for item in datasets])
            for name in output_names
        },
        targets={
            name: np.concatenate([item.targets[name] for item in datasets])
            for name in output_names
        },
        domains=np.concatenate([item.domains for item in datasets]),
        metadata={
            "action_group_ids": [group for item in datasets for group in item.metadata["action_group_ids"]],
            **aligned_metadata,
            "source_datasets": [dict(item.metadata) for item in datasets],
            "protocol": first.metadata.get(
                "protocol",
                "matched_source_counterfactual_symmetric_safety_censor_v5",
            ),
            "counterfactual_horizon_sec": int(first.metadata["counterfactual_horizon_sec"]),
            "counterfactual_estimand": first.metadata.get(
                "counterfactual_estimand",
                "first_action_then_phase_pressure_rollout_value",
            ),
            "mechanism_output_names": list(output_names),
            "local_mechanism_horizon_sec": int(
                first.metadata.get("local_mechanism_horizon_sec", -1)
            ),
            "rollout_value_horizon_sec": int(
                first.metadata.get(
                    "rollout_value_horizon_sec",
                    first.metadata["counterfactual_horizon_sec"],
                )
            ),
            "mechanism_estimands": first.metadata.get("mechanism_estimands"),
            "rollout_prefix_horizons_sec": list(
                first.metadata.get("rollout_prefix_horizons_sec", ())
            ),
            "counterfactual_cost_mode": first.metadata.get(
                "counterfactual_cost_mode"
            ),
            "counterfactual_cost_scope": first.metadata.get("counterfactual_cost_scope"),
            "counterfactual_cost_population": first.metadata.get(
                "counterfactual_cost_population"
            ),
            "counterfactual_cost_normalization": first.metadata.get(
                "counterfactual_cost_normalization"
            ),
            "behavior_policy": first.metadata.get("behavior_policy"),
            "state_snapshot_protocol": first.metadata.get("state_snapshot_protocol"),
            "sumo_execution_protocol": first.metadata.get("sumo_execution_protocol"),
            "strict_safety_monitoring": first.metadata.get(
                "strict_safety_monitoring"
            ),
            "behavior_safety_audit": _merge_behavior_safety_audits_v3(datasets),
            "counterfactual_safety_audit": (
                _merge_counterfactual_safety_audits_v3(datasets)
            ),
            "counterfactual_replay_audit": {
                "checks": int(
                    sum(
                        int(item.metadata["counterfactual_replay_audit"]["checks"])
                        for item in datasets
                    )
                ),
                "max_abs_difference": float(
                    max(
                        float(
                            item.metadata["counterfactual_replay_audit"][
                                "max_abs_difference"
                            ]
                        )
                        for item in datasets
                    )
                ),
                "passed": True,
            },
        },
    )


def _subset_contrast(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    mask = np.asarray(mask, dtype=bool)
    metadata = dict(dataset.metadata)
    for key in ("action_group_ids", "reference_rows", "is_reference"):
        values = np.asarray(metadata.get(key, ()))
        if values.shape == (dataset.size,):
            metadata[key] = values[mask].tolist()
    subset = dataset.subset(mask)
    return MechanismDataset(
        feature_names=subset.feature_names,
        features=subset.features,
        context_names=subset.context_names,
        context=subset.context,
        priors=subset.priors,
        targets=subset.targets,
        domains=subset.domains,
        metadata=metadata,
    )


def _fit_family_v3(
    dataset: MechanismDataset,
    family: str,
    cfcmt_config: MechanismFitConfig,
    definitions: Sequence[MechanismDefinition] = CONTRAST_MECHANISM_DEFINITIONS,
    target_domain: str | None = None,
) -> Any:
    if family == "simulator":
        model = PriorMechanismWorldModel(definitions)
    elif family == "dense":
        model = DenseResidualWorldModel(definitions)
    elif family == "sparse":
        model = InvariantMechanismWorldModel(
            definitions,
            replace(cfcmt_config, latent_ranks=(0,), adaptation_shrinkages=(0.0,)),
        )
    elif family == "cfcmt":
        model = InvariantMechanismWorldModel(definitions, cfcmt_config)
    elif family == "dense_boosted":
        model = BoostedMechanismWorldModel(
            definitions,
            causal=False,
            config=BoostedFitConfig(),
        )
    elif family == "causal_boosted":
        model = BoostedMechanismWorldModel(
            definitions,
            causal=True,
            config=BoostedFitConfig(),
        )
    elif family == "cfcmt_boosted":
        model = BoostedCausalTransferWorldModel(
            definitions,
            boosted_config=BoostedFitConfig(),
            adaptation_config=replace(
                cfcmt_config,
                latent_ranks=tuple(rank for rank in cfcmt_config.latent_ranks if rank <= 1) or (0,),
            ),
        )
    elif family == "dense_ranker":
        model = PairwiseActionRanker(causal=False)
    elif family == "causal_ranker":
        model = PairwiseActionRanker(causal=True)
    elif family == "dense_advantage":
        model = PairwiseActionAdvantageRegressor(causal=False)
    elif family == "dense_balanced_advantage":
        model = PairwiseActionAdvantageRegressor(
            causal=False,
            config=ActionAdvantageConfig(
                candidate_only=True,
                balance_candidate_signs=True,
            ),
        )
    elif family == "causal_advantage":
        model = PairwiseActionAdvantageRegressor(causal=True)
    elif family == "causal_core_advantage":
        model = PairwiseActionAdvantageRegressor(
            causal=True,
            causal_feature_names=CAUSAL_CORE_RANKING_PARENTS,
        )
    elif family == "causal_balanced_advantage":
        model = PairwiseActionAdvantageRegressor(
            causal=True,
            config=ActionAdvantageConfig(
                candidate_only=True,
                balance_candidate_signs=True,
            ),
            causal_feature_names=CAUSAL_CORE_RANKING_PARENTS,
        )
    elif family == "causal_rigid_advantage":
        model = PairwiseActionAdvantageRegressor(
            causal=True,
            config=ActionAdvantageConfig(
                candidate_only=True,
                balance_candidate_signs=True,
            ),
            causal_feature_names=CAUSAL_RIGID_RANKING_PARENTS,
        )
    elif family == GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10:
        model = PairwiseActionAdvantageRegressor(
            causal=True,
            config=ActionAdvantageConfig(
                candidate_only=True,
                balance_candidate_signs=True,
                target_normalization_protocol=(
                    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
                ),
            ),
            causal_feature_names=CAUSAL_RIGID_RANKING_PARENTS,
        )
    elif family == ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11:
        model = AntisymmetricPairwiseActionRegressor()
    elif family == SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12:
        model = SourceGatedPairwiseActionRegressor()
    elif family == "cfcmt_mechanism":
        model = CausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
        )
    elif family == "cfcmt_fused":
        model = FusedCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
        )
    elif family == "cfcmt_fused_rigid":
        model = FusedRigidActionAdvantageModel(
            definitions,
            target_domain=target_domain,
        )
    elif family == "cfcmt_physical_mechanism":
        model = PhysicalResidualCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            latent=False,
        )
    elif family == "cfcmt_physical_fused":
        model = FusedPhysicalResidualCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            latent=False,
        )
    elif family == ONE_STEP_PHYSICAL_FULL_FAMILY_V4:
        model = PhysicalResidualCausalMechanismAdvantageModel(
            ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4,
            target_domain=target_domain,
            latent=False,
        )
    elif family == ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5:
        model = PhysicalResidualCausalMechanismAdvantageModel(
            ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4,
            target_domain=target_domain,
            config=CausalMechanismAdvantageConfig(
                source_stack_protocol=(
                    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
                )
            ),
            latent=False,
        )
    elif family in PHYSICAL_COMPONENT_DEFINITION_NAMES_V3:
        mechanism_names = PHYSICAL_COMPONENT_DEFINITION_NAMES_V3[family]
        mechanism_name_set = set(mechanism_names)
        selected_definitions = tuple(
            definition
            for definition in definitions
            if definition.name in mechanism_name_set
        )
        selected_names = tuple(definition.name for definition in selected_definitions)
        if set(selected_names) != mechanism_name_set or len(selected_names) != len(
            mechanism_names
        ):
            raise ValueError(
                f"{family}: expected definitions {mechanism_names!r}, "
                f"found {selected_names!r}"
            )
        model = PhysicalResidualCausalMechanismAdvantageModel(
            selected_definitions,
            target_domain=target_domain,
            latent=False,
        )
    elif family in ONE_STEP_PHYSICAL_COMPONENT_DEFINITION_NAMES_V4:
        mechanism_names = ONE_STEP_PHYSICAL_COMPONENT_DEFINITION_NAMES_V4[family]
        mechanism_name_set = set(mechanism_names)
        selected_definitions = tuple(
            definition
            for definition in ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4
            if definition.name in mechanism_name_set
        )
        selected_names = tuple(definition.name for definition in selected_definitions)
        if set(selected_names) != mechanism_name_set or len(selected_names) != len(
            mechanism_names
        ):
            raise ValueError(
                f"{family}: expected one-step definitions {mechanism_names!r}, "
                f"found {selected_names!r}"
            )
        model = PhysicalResidualCausalMechanismAdvantageModel(
            selected_definitions,
            target_domain=target_domain,
            latent=False,
        )
    elif family in ONE_STEP_RIGID_RESIDUAL_COMPONENT_DEFINITION_NAMES_V5:
        mechanism_names = ONE_STEP_RIGID_RESIDUAL_COMPONENT_DEFINITION_NAMES_V5[
            family
        ]
        mechanism_name_set = set(mechanism_names)
        selected_definitions = tuple(
            definition
            for definition in ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4
            if definition.name in mechanism_name_set
        )
        selected_names = tuple(definition.name for definition in selected_definitions)
        if set(selected_names) != mechanism_name_set or len(selected_names) != len(
            mechanism_names
        ):
            raise ValueError(
                f"{family}: expected one-step definitions {mechanism_names!r}, "
                f"found {selected_names!r}"
            )
        model = PhysicalResidualCausalMechanismAdvantageModel(
            selected_definitions,
            target_domain=target_domain,
            config=CausalMechanismAdvantageConfig(
                source_stack_protocol=(
                    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
                )
            ),
            latent=False,
        )
    elif family in ONE_STEP_SELECTIVE_RESIDUAL_FAMILY_CONFIG_V6:
        mechanism_names, error_quantile = (
            ONE_STEP_SELECTIVE_RESIDUAL_FAMILY_CONFIG_V6[family]
        )
        mechanism_name_set = set(mechanism_names)
        selected_definitions = tuple(
            definition
            for definition in ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4
            if definition.name in mechanism_name_set
        )
        selected_names = tuple(definition.name for definition in selected_definitions)
        if set(selected_names) != mechanism_name_set or len(selected_names) != len(
            mechanism_names
        ):
            raise ValueError(
                f"{family}: expected selective one-step definitions "
                f"{mechanism_names!r}, found {selected_names!r}"
            )
        model = PhysicalResidualCausalMechanismAdvantageModel(
            selected_definitions,
            target_domain=target_domain,
            config=CausalMechanismAdvantageConfig(
                source_stack_protocol=(
                    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
                ),
                source_expert_gate_protocol=(
                    SOURCE_OOF_RIDGE_LCB_EXPERT_GATE_PROTOCOL
                ),
                source_expert_gate_error_quantile=float(error_quantile),
            ),
            latent=False,
        )
    elif family == ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7:
        model = RolloutValueResidualCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            config=CausalMechanismAdvantageConfig(
                source_stack_protocol=(
                    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
                )
            ),
        )
    elif family == DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8:
        model = DirectRolloutValueCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            config=CausalMechanismAdvantageConfig(
                source_stack_protocol=(
                    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
                )
            ),
        )
    elif family == BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9:
        model = BoostedDirectRolloutValueCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            config=CausalMechanismAdvantageConfig(
                source_stack_protocol=(
                    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
                )
            ),
        )
    elif family == "cfcmt_physical_latent_mechanism":
        model = PhysicalResidualCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            latent=True,
        )
    elif family == "cfcmt_physical_latent_fused":
        model = FusedPhysicalResidualCausalMechanismAdvantageModel(
            definitions,
            target_domain=target_domain,
            latent=True,
        )
    elif family == "causal_target_adapter":
        model = TargetAdaptedActionAdvantageRegressor(target_domain=target_domain)
    elif family == "causal_target_only":
        model = TargetAdaptedActionAdvantageRegressor(
            target_domain=target_domain,
            prior_group_strength=0.0,
            max_target_weight=1.0,
        )
    elif family == "causal_target_only_v2":
        model = TargetOnlyActionAdvantageRegressor(target_domain=target_domain)
    else:
        raise ValueError(f"unknown V3 model family: {family}")
    diagnostics = model.fit(dataset)
    return model, diagnostics


def _fit_family_with_source_loo_v3(
    dataset: MechanismDataset,
    *,
    family: str,
    cfcmt_config: MechanismFitConfig,
    target_domain: str | None,
    parallel_workers: int = 1,
    reused_source_model: CausalMechanismAdvantageModel | None = None,
    reused_prediction_bundle: Mapping[
        str,
        tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]],
    ]
    | None = None,
) -> tuple[
    Any,
    dict[str, Any],
    dict[str, tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]]],
    dict[str, Any],
]:
    """Fit the full-source model and independent source-city LOO folds."""

    heldouts = tuple(str(value) for value in np.unique(dataset.domains))
    folds = []
    for heldout in heldouts:
        train = _subset_contrast(dataset, dataset.domains != heldout)
        valid = _subset_contrast(dataset, dataset.domains == heldout)
        if np.unique(train.domains).size >= 2:
            folds.append((heldout, train, valid))
    if (
        family == ONE_STEP_PHYSICAL_FULL_FAMILY_V4
        or family in ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4
        or family == ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5
        or family in ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5
        or family in ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6
    ):
        source_definitions = ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4
    else:
        source_definitions = (
            CONTRAST_MECHANISM_DEFINITIONS
            if family in {
            "cfcmt_boosted",
            "cfcmt_mechanism",
            "cfcmt_fused",
            "cfcmt_fused_rigid",
            "cfcmt_physical_mechanism",
            "cfcmt_physical_fused",
            "cfcmt_physical_latent_mechanism",
            "cfcmt_physical_latent_fused",
            ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
            DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
            BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
            }
            or family in PHYSICAL_COMPONENT_FAMILIES_V3
            else CONTROL_COST_CONTRAST_DEFINITION
        )

    reusable_predictions = dict(reused_prediction_bundle or {})
    if reusable_predictions:
        expected_folds = {heldout: valid for heldout, _, valid in folds}
        if set(reusable_predictions) != set(expected_folds):
            raise ValueError(
                "source-LOO reuse requires identical held-out domains: "
                f"expected={sorted(expected_folds)}, "
                f"provided={sorted(reusable_predictions)}"
            )
        for heldout, (valid, _) in reusable_predictions.items():
            expected = expected_folds[heldout]
            aligned_metadata_keys = (
                "action_group_ids",
                "is_reference",
                "reference_rows",
            )
            if (
                valid.feature_names != expected.feature_names
                or valid.context_names != expected.context_names
                or valid.size != expected.size
                or not np.array_equal(valid.features, expected.features)
                or not np.array_equal(valid.context, expected.context)
                or not np.array_equal(valid.domains, expected.domains)
                or set(valid.priors) != set(expected.priors)
                or any(
                    not np.array_equal(valid.priors[name], expected.priors[name])
                    for name in expected.priors
                )
                or set(valid.targets) != set(expected.targets)
                or any(
                    not np.array_equal(valid.targets[name], expected.targets[name])
                    for name in expected.targets
                )
                or any(
                    not np.array_equal(
                        np.asarray(valid.metadata.get(key, ())),
                        np.asarray(expected.metadata.get(key, ())),
                    )
                    for key in aligned_metadata_keys
                )
            ):
                raise ValueError(
                    f"source-LOO reuse dataset mismatch for held-out domain {heldout!r}"
                )

    def fit_fold(
        heldout: str,
        train: MechanismDataset,
        valid: MechanismDataset,
    ) -> tuple[str, MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]]:
        model, _ = _fit_family_v3(
            train,
            family,
            cfcmt_config,
            definitions=source_definitions,
        )
        return heldout, valid, model.predict(valid)

    def fit_full() -> tuple[Any, dict[str, Any]]:
        if reused_source_model is not None:
            if family == "cfcmt_fused":
                model = FusedCausalMechanismAdvantageModel(
                    source_definitions,
                    target_domain=target_domain,
                )
            elif family in {
                "cfcmt_physical_fused",
                "cfcmt_physical_latent_fused",
            }:
                model = FusedPhysicalResidualCausalMechanismAdvantageModel(
                    source_definitions,
                    target_domain=target_domain,
                    latent=family == "cfcmt_physical_latent_fused",
                )
            else:
                raise ValueError(
                    "frozen source-model reuse is only defined for matched fused families"
                )
            diagnostics = model.fit_from_fitted_source(
                dataset,
                reused_source_model,
            )
            return model, diagnostics
        return _fit_family_v3(
            dataset,
            family,
            cfcmt_config,
            target_domain=target_domain,
        )

    requested_workers = max(int(parallel_workers), 1)
    actual_workers = min(requested_workers, len(folds) + 1)
    if reusable_predictions:
        model, diagnostics = fit_full()
        fold_results = [
            (heldout, expected_folds[heldout], prediction)
            for heldout, (_, prediction) in reusable_predictions.items()
        ]
        actual_workers = 1
    elif actual_workers <= 1:
        model, diagnostics = fit_full()
        fold_results = [fit_fold(*fold) for fold in folds]
    else:
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            model_future = pool.submit(fit_full)
            fold_futures = [pool.submit(fit_fold, *fold) for fold in folds]
            model, diagnostics = model_future.result()
            fold_results = [future.result() for future in fold_futures]
    predictions = {
        heldout: (valid, prediction)
        for heldout, valid, prediction in sorted(fold_results, key=lambda item: item[0])
    }
    return model, dict(diagnostics), predictions, {
        "protocol": "full_source_plus_source_city_loo_thread_pool_v1",
        "requested_workers": requested_workers,
        "actual_workers": actual_workers,
        "full_source_fit_count": 0 if reused_source_model is not None else 1,
        "reused_full_source_fit_count": 1 if reused_source_model is not None else 0,
        "source_loo_fit_count": 0 if reusable_predictions else len(folds),
        "reused_source_loo_fit_count": len(folds) if reusable_predictions else 0,
        "reuse_protocol": (
            "exact_matched_family_artifacts_v1"
            if reused_source_model is not None or reusable_predictions
            else "none"
        ),
    }


def _group_adjusted_scores(
    dataset: MechanismDataset,
    prediction: Mapping[str, Mapping[str, np.ndarray]],
    *,
    objective_mode: str = "control_only",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = action_group_ids(dataset)
    is_reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    if objective_mode not in OBJECTIVE_WEIGHTS_V3:
        raise ValueError(f"unknown V3 mechanism objective: {objective_mode!r}")
    weights = OBJECTIVE_WEIGHTS_V3[objective_mode]
    missing = set(weights) - set(prediction)
    if missing:
        raise KeyError(f"objective {objective_mode!r} is missing mechanism predictions: {sorted(missing)}")
    score = np.zeros(dataset.size, dtype=float)
    uncertainty = np.zeros(dataset.size, dtype=float)
    trust_rows = []
    for mechanism, weight in weights.items():
        score += float(weight) * np.asarray(prediction[mechanism]["mean"], dtype=float)
        uncertainty += abs(float(weight)) * np.asarray(
            prediction[mechanism]["uncertainty"],
            dtype=float,
        )
        trust_rows.append(np.asarray(prediction[mechanism]["context_trust"], dtype=float))
    trust = np.minimum.reduce(trust_rows)
    reference_rows = np.empty(dataset.size, dtype=int)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        reference = rows[is_reference[rows]]
        if reference.size != 1:
            raise ValueError(f"contrast group {group!r} does not have exactly one reference")
        reference_rows[rows] = int(reference[0])
    score -= score[reference_rows]
    uncertainty += uncertainty[reference_rows]
    score[is_reference] = 0.0
    uncertainty[is_reference] = 0.0
    return score, uncertainty, trust, reference_rows


def _relative_contrast_rule_gap(dataset: MechanismDataset) -> np.ndarray:
    raw_score = pressure_scores_from_feature_matrix(
        dataset.features,
        dataset.feature_names,
        dataset.metadata.get(
            "reference_policy_spec", dataset.metadata.get("reference_policy", "")
        ),
    )
    groups = action_group_ids(dataset)
    is_reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    relative = np.zeros(dataset.size, dtype=float)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        reference = rows[is_reference[rows]]
        if reference.size != 1:
            raise ValueError(
                f"contrast group {group!r} does not have exactly one rule reference"
            )
        reference_score = float(raw_score[int(reference[0])])
        relative[rows] = np.maximum(reference_score - raw_score[rows], 0.0) / max(
            abs(reference_score), 1.0
        )
    return relative


def _source_loo_action_records_v3(
    dataset: MechanismDataset,
    *,
    family: str,
    cfcmt_config: MechanismFitConfig,
    objective_mode: str = "control_only",
    prediction_bundle: Mapping[
        str,
        tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]],
    ]
    | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    records: dict[str, dict[str, np.ndarray]] = {}
    trust_values = []
    if prediction_bundle is None:
        prediction_bundle = {}
        for heldout in np.unique(dataset.domains):
            train = _subset_contrast(dataset, dataset.domains != heldout)
            valid = _subset_contrast(dataset, dataset.domains == heldout)
            if np.unique(train.domains).size < 2:
                continue
            model, _ = _fit_family_v3(
                train,
                family,
                cfcmt_config,
                definitions=(
                    ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4
                    if family == ONE_STEP_PHYSICAL_FULL_FAMILY_V4
                    or family in ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4
                    or family == ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5
                    or family in ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5
                    or family in ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6
                    else CONTRAST_MECHANISM_DEFINITIONS
                    if family in {
                        "cfcmt_mechanism",
                        "cfcmt_fused",
                        "cfcmt_fused_rigid",
                        "cfcmt_physical_mechanism",
                        "cfcmt_physical_fused",
                        "cfcmt_physical_latent_mechanism",
                        "cfcmt_physical_latent_fused",
                        ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
                        DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
                        BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
                    }
                    or family in PHYSICAL_COMPONENT_FAMILIES_V3
                    or objective_mode != "control_only"
                    else CONTROL_COST_CONTRAST_DEFINITION
                ),
            )
            prediction_bundle[str(heldout)] = (valid, model.predict(valid))
    for heldout, (valid, prediction) in prediction_bundle.items():
        score, uncertainty, trust, reference_rows = _group_adjusted_scores(
            valid,
            prediction,
            objective_mode=objective_mode,
        )
        target_clip = (
            8.0
            if family in {
                "cfcmt_mechanism",
                "cfcmt_fused",
                "cfcmt_fused_rigid",
                "cfcmt_physical_mechanism",
                "cfcmt_physical_fused",
                "cfcmt_physical_latent_mechanism",
                "cfcmt_physical_latent_fused",
                ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
                DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
                BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
            }
            or family in PHYSICAL_COMPONENT_FAMILIES_V3
            or family == ONE_STEP_PHYSICAL_FULL_FAMILY_V4
            or family in ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4
            or family == ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5
            or family in ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5
            or family in ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6
            else 5.0
            if family in NORMALIZED_ACTION_UNIT_FAMILIES_V3
            else None
        )
        actual, action_scales = domain_normalized_action_target(
            valid,
            "interval_cost",
            target_clip=target_clip,
        )
        row_scales = np.asarray(
            [action_scales[str(domain)] for domain in valid.domains],
            dtype=float,
        )
        prediction_was_normalized = family in NORMALIZED_ACTION_UNIT_FAMILIES_V3
        if not prediction_was_normalized:
            score = score / row_scales
            uncertainty = uncertainty / row_scales
        terminal_name = (
            "terminal_system_load"
            if "terminal_system_load" in valid.targets
            else "interval_cost"
        )
        actual_terminal, terminal_scales = domain_normalized_action_target(
            valid,
            terminal_name,
        )
        records[str(heldout)] = {
            "score": score,
            "uncertainty": uncertainty,
            "trust": trust,
            "reference_rows": reference_rows,
            "actual": actual,
            "actual_raw": np.asarray(valid.targets["interval_cost"], dtype=float),
            "actual_terminal": actual_terminal,
            "actual_terminal_raw": np.asarray(valid.targets[terminal_name], dtype=float),
            "action_unit": "within-domain-median-action-range",
            "prediction_was_normalized": prediction_was_normalized,
            "action_scales": action_scales,
            "terminal_action_scales": terminal_scales,
            "groups": action_group_ids(valid),
            "rule_gap": _relative_contrast_rule_gap(valid),
            "rule_gap_unit": "reference-normalized-pressure-gap",
            "relative_rule_gap": _relative_contrast_rule_gap(valid),
        }
        trust_values.extend(trust.tolist())
    min_trust = max(
        MIN_CONTEXT_SUPPORT_V3,
        float(np.clip(np.quantile(trust_values, 0.02), 0.0, 0.25)) if trust_values else 0.0,
    )
    return records, min_trust


def _calibrate_source_loo_uncertainty_v3(
    records_bundle: tuple[dict[str, dict[str, np.ndarray]], float],
    *,
    quantile: float = 0.90,
) -> tuple[
    tuple[dict[str, dict[str, np.ndarray]], float],
    dict[str, Any],
]:
    """Inflate uncertainty to cover source-city held-out action errors."""

    records, min_trust = records_bundle
    domain_scales: dict[str, float] = {}
    for domain, record in records.items():
        score = np.asarray(record["score"], dtype=float)
        actual = np.asarray(record["actual"], dtype=float)
        uncertainty = np.asarray(record["uncertainty"], dtype=float)
        reference_rows = np.asarray(record["reference_rows"], dtype=int)
        candidate = np.arange(score.size) != reference_rows
        usable = candidate & np.isfinite(score) & np.isfinite(actual) & np.isfinite(
            uncertainty
        )
        if not np.any(usable):
            domain_scales[str(domain)] = 1.0
            continue
        error = np.abs(actual[usable] - score[usable])
        denominator = np.maximum(uncertainty[usable], 1e-8)
        domain_scales[str(domain)] = max(
            float(np.quantile(error / denominator, float(quantile))),
            1.0,
        )
    scale = max(domain_scales.values(), default=1.0)
    scaled_records = {
        domain: {
            **record,
            "uncertainty": np.asarray(record["uncertainty"], dtype=float) * scale,
        }
        for domain, record in records.items()
    }
    return (scaled_records, min_trust), {
        "protocol": "domain-normalized-worst-source-city-quantile-inflation-v2",
        "action_unit": "within-domain-median-action-range",
        "quantile": float(quantile),
        "scale": float(scale),
        "domain_scales": domain_scales,
        "source_domain_count": len(domain_scales),
        "shrinkage_allowed": False,
    }


def calibrate_mechanism_objective_v3(
    dataset: MechanismDataset,
    *,
    family: str,
    cfcmt_config: MechanismFitConfig,
    objective_modes: Sequence[str] = tuple(OBJECTIVE_WEIGHTS_V3),
    min_material_gain: float = 0.005,
    prediction_bundle: Mapping[
        str,
        tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]],
    ]
    | None = None,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]]],
]:
    """Select whether multi-mechanism predictions improve source action ranking."""

    domain_rows: dict[str, dict[str, float]] = {mode: {} for mode in objective_modes}
    if prediction_bundle is None:
        computed_bundle: dict[
            str,
            tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]],
        ] = {}
        for heldout in np.unique(dataset.domains):
            train = _subset_contrast(dataset, dataset.domains != heldout)
            valid = _subset_contrast(dataset, dataset.domains == heldout)
            if np.unique(train.domains).size < 2:
                continue
            model, _ = _fit_family_v3(
                train,
                family,
                cfcmt_config,
                definitions=CONTRAST_MECHANISM_DEFINITIONS,
            )
            computed_bundle[str(heldout)] = (valid, model.predict(valid))
        prediction_bundle = computed_bundle
    else:
        prediction_bundle = dict(prediction_bundle)
    for heldout, (valid, prediction) in prediction_bundle.items():
        actual = np.asarray(valid.targets["interval_cost"], dtype=float)
        groups = action_group_ids(valid)
        for mode in objective_modes:
            score, _, _, _ = _group_adjusted_scores(
                valid,
                prediction,
                objective_mode=mode,
            )
            regrets = []
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                selected = int(rows[int(np.argmin(score[rows]))])
                values = actual[rows]
                scale = action_group_range(values)
                regrets.append((float(actual[selected]) - float(np.min(values))) / scale)
            domain_rows[mode][str(heldout)] = float(np.mean(regrets)) if regrets else 0.0

    grid = []
    for mode in objective_modes:
        values = np.asarray(list(domain_rows[mode].values()), dtype=float)
        robust = float(np.mean(values) + 0.25 * np.std(values) + 0.25 * np.max(values))
        complexity = max(len(OBJECTIVE_WEIGHTS_V3[mode]) - 1, 0)
        grid.append(
            {
                "objective_mode": mode,
                "mean_normalized_regret": float(np.mean(values)),
                "worst_domain_regret": float(np.max(values)),
                "robust_normalized_regret": robust,
                "selection_score": robust + 0.001 * complexity,
                "domain_normalized_regret": domain_rows[mode],
            }
        )
    control = next(row for row in grid if row["objective_mode"] == "control_only")
    best = min(grid, key=lambda row: (row["selection_score"], len(OBJECTIVE_WEIGHTS_V3[row["objective_mode"]])))
    material_gain = float(control["selection_score"] - best["selection_score"])
    if best["objective_mode"] != "control_only" and material_gain < float(min_material_gain):
        selected = "control_only"
        gate_reason = "multi_mechanism_gain_below_margin"
    else:
        selected = str(best["objective_mode"])
        gate_reason = "selected_by_source_loo"
    return (
        selected,
        {
            "family": family,
            "selected": selected,
            "gate_reason": gate_reason,
            "min_material_gain": float(min_material_gain),
            "action_regret_scale_protocol": GROUP_ACTION_REGRET_SCALE_PROTOCOL,
            "material_gain_vs_control_only": material_gain,
            "grid": grid,
        },
        prediction_bundle,
    )


def calibrate_contrast_guard_v3(
    dataset: MechanismDataset,
    *,
    family: str,
    cfcmt_config: MechanismFitConfig,
    thresholds: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    max_relative_rule_gaps: Sequence[float] = (0.0, 0.10, 0.25, 0.50, 1.0),
    noninferiority_tolerance: float = 0.05,
    terminal_noninferiority_tolerance: float | None = None,
    selection_mode: str = "strict_worst_domain",
    confidence_z: float = 1.645,
    max_harm_fraction: float = 0.20,
    min_improving_fraction: float = 0.10,
    records_bundle: tuple[dict[str, dict[str, np.ndarray]], float] | None = None,
) -> tuple[ContrastGuardConfig, dict[str, Any]]:
    """Select override confidence using only leave-one-source action outcomes."""

    records, min_trust = records_bundle or _source_loo_action_records_v3(
        dataset,
        family=family,
        cfcmt_config=cfcmt_config,
    )

    grid = []
    for threshold in thresholds:
        for max_rule_gap in max_relative_rule_gaps:
            domain_means = {}
            domain_terminal_means = {}
            accepted = 0
            decisions = 0
            for domain, record in records.items():
                selected_actual = []
                selected_terminal = []
                for group in np.unique(record["groups"]):
                    rows = np.flatnonzero(record["groups"] == group)
                    reference = int(record["reference_rows"][rows[0]])
                    learned = int(rows[int(np.argmin(record["score"][rows]))])
                    decisions += 1
                    use_learned = (
                        learned != reference
                        and float(record["trust"][learned]) >= min_trust
                        and float(record["relative_rule_gap"][learned])
                        <= float(max_rule_gap) + 1e-12
                        and float(record["score"][learned])
                        + float(threshold) * float(record["uncertainty"][learned])
                        < 0.0
                    )
                    if use_learned:
                        accepted += 1
                        selected_actual.append(float(record["actual"][learned]))
                        selected_terminal.append(
                            float(record["actual_terminal"][learned])
                        )
                    else:
                        selected_actual.append(0.0)
                        selected_terminal.append(0.0)
                domain_means[domain] = (
                    float(np.mean(selected_actual)) if selected_actual else 0.0
                )
                domain_terminal_means[domain] = (
                    float(np.mean(selected_terminal)) if selected_terminal else 0.0
                )
            mean_delta = float(np.mean(list(domain_means.values()))) if domain_means else 0.0
            worst_delta = max(domain_means.values(), default=0.0)
            values = np.asarray(list(domain_means.values()), dtype=float)
            terminal_values = np.asarray(
                list(domain_terminal_means.values()), dtype=float
            )
            standard_error = (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else float("inf")
            )
            upper_confidence_bound = mean_delta + float(confidence_z) * standard_error
            harm_fraction = float(
                np.mean(values > float(noninferiority_tolerance))
            ) if values.size else 0.0
            improving_fraction = float(np.mean(values < 0.0)) if values.size else 0.0
            terminal_harm_fraction = float(
                np.mean(terminal_values > float(terminal_noninferiority_tolerance or 0.0))
            ) if terminal_values.size else 0.0
            grid.append(
                {
                    "risk_multiplier": float(threshold),
                    "max_relative_rule_gap": float(max_rule_gap),
                    "mean_actual_delta": mean_delta,
                    "worst_domain_delta": float(worst_delta),
                    "mean_terminal_delta": float(np.mean(terminal_values))
                    if terminal_values.size
                    else 0.0,
                    "worst_terminal_delta": float(np.max(terminal_values))
                    if terminal_values.size
                    else 0.0,
                    "terminal_harm_fraction": terminal_harm_fraction,
                    "standard_error": standard_error,
                    "upper_confidence_bound": upper_confidence_bound,
                    "harm_fraction": harm_fraction,
                    "improving_fraction": improving_fraction,
                    "accepted_overrides": int(accepted),
                    "decisions": int(decisions),
                    "domain_mean_delta": domain_means,
                    "domain_terminal_mean_delta": domain_terminal_means,
                }
            )
    def terminal_feasible(row: Mapping[str, Any]) -> bool:
        return bool(
            terminal_noninferiority_tolerance is None
            or (
                row["mean_terminal_delta"] <= float(terminal_noninferiority_tolerance)
                and row["terminal_harm_fraction"] <= float(max_harm_fraction)
            )
        )
    if selection_mode == "strict_worst_domain":
        feasible = [
            row
            for row in grid
            if row["accepted_overrides"] > 0
            and row["mean_actual_delta"] < 0.0
            and row["worst_domain_delta"] <= float(noninferiority_tolerance)
            and terminal_feasible(row)
        ]
    elif selection_mode == "mean_ucb":
        feasible = [
            row
            for row in grid
            if row["accepted_overrides"] >= 2
            and row["upper_confidence_bound"] < 0.0
            and row["harm_fraction"] <= float(max_harm_fraction)
            and row["improving_fraction"] >= float(min_improving_fraction)
            and terminal_feasible(row)
        ]
    else:
        raise ValueError(f"unknown contrast guard selection mode: {selection_mode}")
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                row["mean_actual_delta"],
                row["mean_terminal_delta"],
                row["max_relative_rule_gap"],
                -row["risk_multiplier"],
            ),
        )
        config = ContrastGuardConfig(
            enabled=True,
            risk_multiplier=float(selected["risk_multiplier"]),
            min_context_trust=min_trust,
            margin=0.0,
            max_relative_rule_gap=float(selected["max_relative_rule_gap"]),
        )
    else:
        selected = None
        config = ContrastGuardConfig(enabled=False, min_context_trust=min_trust)
    return config, {
        "family": family,
        "objective_units": {
            "rule_gap": "reference-normalized-pressure-gap",
            "model_score": "within-domain-median-action-range",
            "uncertainty": "within-domain-median-action-range",
        },
        "source_domains": sorted(records),
        "selection_mode": selection_mode,
        "confidence_z": float(confidence_z),
        "max_harm_fraction": float(max_harm_fraction),
        "min_improving_fraction": float(min_improving_fraction),
        "noninferiority_tolerance": float(noninferiority_tolerance),
        "terminal_noninferiority_tolerance": terminal_noninferiority_tolerance,
        "selected": selected,
        "grid": grid,
    }


def calibrate_prior_regularization_v3(
    dataset: MechanismDataset,
    *,
    family: str,
    cfcmt_config: MechanismFitConfig,
    blend_weights: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    risk_multipliers: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    noninferiority_tolerance: float = 0.0,
    min_mean_improvement: float = 1e-4,
    min_override_fraction: float = 0.02,
    min_improving_domain_fraction: float = 1.0 / 3.0,
    trust_quantiles: Sequence[float] = (0.0, 0.10, 0.25, 0.50, 0.75),
    records_bundle: tuple[dict[str, dict[str, np.ndarray]], float] | None = None,
) -> tuple[PriorRegularizationConfig, dict[str, Any]]:
    """Select a physics-prior/model blend on held-out source action groups."""

    records, min_trust = records_bundle or _source_loo_action_records_v3(
        dataset,
        family=family,
        cfcmt_config=cfcmt_config,
    )
    all_trust = np.concatenate(
        [np.asarray(record["trust"], dtype=float) for record in records.values()]
    ) if records else np.asarray([0.0])
    trust_thresholds = sorted(
        {
            float(np.clip(np.quantile(all_trust, quantile), MIN_CONTEXT_SUPPORT_V3, 0.95))
            for quantile in trust_quantiles
        }
        | {float(min_trust)}
    )
    grid = []
    for blend_weight in blend_weights:
        for risk_multiplier in risk_multipliers:
            for trust_threshold in trust_thresholds:
                domain_means = {}
                overrides = 0
                decisions = 0
                for domain, record in records.items():
                    selected_actual = []
                    for group in np.unique(record["groups"]):
                        rows = np.flatnonzero(record["groups"] == group)
                        reference = int(record["reference_rows"][rows[0]])
                        objective = (
                            record["rule_gap"][rows]
                            + float(blend_weight) * record["score"][rows]
                            + float(risk_multiplier) * record["uncertainty"][rows]
                        )
                        objective = np.asarray(objective, dtype=float)
                        objective[record["trust"][rows] < float(trust_threshold)] = np.inf
                        reference_local = int(np.flatnonzero(rows == reference)[0])
                        objective[reference_local] = 0.0
                        selected = int(rows[int(np.argmin(objective))])
                        decisions += 1
                        if selected != reference:
                            overrides += 1
                        selected_actual.append(float(record["actual"][selected]))
                    domain_means[domain] = float(np.mean(selected_actual)) if selected_actual else 0.0
                mean_delta = float(np.mean(list(domain_means.values()))) if domain_means else 0.0
                worst_delta = max(domain_means.values(), default=0.0)
                improving_domains = int(
                    sum(value < -float(min_mean_improvement) for value in domain_means.values())
                )
                grid.append(
                    {
                        "blend_weight": float(blend_weight),
                        "risk_multiplier": float(risk_multiplier),
                        "min_context_trust": float(trust_threshold),
                        "mean_actual_delta": mean_delta,
                        "worst_domain_delta": float(worst_delta),
                        "overrides": int(overrides),
                        "decisions": int(decisions),
                        "improving_domains": improving_domains,
                        "domain_mean_delta": domain_means,
                    }
                )
    domain_count = len(records)
    required_improving_domains = max(
        min(2, domain_count),
        int(math.ceil(float(min_improving_domain_fraction) * domain_count)),
    )
    feasible = [
        row
        for row in grid
        if row["overrides"] >= max(2, int(math.ceil(float(min_override_fraction) * row["decisions"])))
        and row["mean_actual_delta"] < -float(min_mean_improvement)
        and row["worst_domain_delta"] <= float(noninferiority_tolerance) + 1e-12
        and row["improving_domains"] >= required_improving_domains
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                row["mean_actual_delta"],
                row["worst_domain_delta"],
                row["blend_weight"],
                -row["risk_multiplier"],
                -row["min_context_trust"],
            ),
        )
        config = PriorRegularizationConfig(
            enabled=True,
            blend_weight=float(selected["blend_weight"]),
            risk_multiplier=float(selected["risk_multiplier"]),
            min_context_trust=float(selected["min_context_trust"]),
        )
    else:
        selected = None
        config = PriorRegularizationConfig()
    return config, {
        "family": family,
        "objective_units": {
            "rule_gap": "reference-normalized-pressure-gap",
            "model_score": "within-domain-median-action-range",
            "uncertainty": "within-domain-median-action-range",
        },
        "source_domains": sorted(records),
        "source_min_context_trust": min_trust,
        "trust_thresholds": trust_thresholds,
        "noninferiority_tolerance": float(noninferiority_tolerance),
        "min_mean_improvement": float(min_mean_improvement),
        "min_override_fraction": float(min_override_fraction),
        "min_improving_domain_fraction": float(min_improving_domain_fraction),
        "required_improving_domains": int(required_improving_domains),
        "selected": selected,
        "grid": grid,
    }


def fit_contrast_models_v3(
    absolute_dataset: MechanismDataset,
    *,
    cfcmt_config: MechanismFitConfig | None = None,
    target_context: np.ndarray | None = None,
    prior_policy_override: str | PressurePolicySpec | Mapping[str, Any] | None = None,
    prior_selection_diagnostics: Mapping[str, Any] | None = None,
    model_families: Sequence[str] = MODEL_FAMILIES_V3,
    target_domain: str | None = None,
    parallel_workers: int = 1,
) -> FittedContrastModelsV3:
    config = cfcmt_config or MechanismFitConfig()
    if "interval_cost" not in config.action_ranking_targets:
        config = replace(
            config,
            action_ranking_targets=(*config.action_ranking_targets, "interval_cost"),
        )
    if prior_policy_override is None:
        prior_policy, prior_diagnostics = select_source_only_reference_policy(
            absolute_dataset,
            target_context=target_context,
        )
        prior = pressure_spec(prior_policy)
    else:
        prior = pressure_spec(prior_policy_override)
        prior_policy = prior.key
        prior_diagnostics = dict(prior_selection_diagnostics or {})
        prior_diagnostics.setdefault("selected", prior_policy)
    contrast = build_action_contrast_dataset(
        absolute_dataset,
        reference_policy=prior,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    models = {}
    fit_diagnostics = {}
    guards = {}
    regularizers = {}
    objective_modes = {}
    objective_diagnostics = {}
    guard_diagnostics = {}
    regularizer_diagnostics = {}
    raw_family_models: dict[str, Any] = {}
    source_loo_prediction_bundles: dict[
        str,
        dict[str, tuple[MechanismDataset, Mapping[str, Mapping[str, np.ndarray]]]],
    ] = {}
    requested_families = tuple(dict.fromkeys(str(family) for family in model_families))
    unknown_families = set(requested_families) - set(MODEL_FAMILIES_V3)
    if unknown_families:
        raise ValueError(f"unknown requested V3 model families: {sorted(unknown_families)}")
    fit_started = time.monotonic()
    for family_index, family in enumerate(requested_families, start=1):
        family_started = time.monotonic()
        reused_source_family = {
            "cfcmt_fused": "cfcmt_mechanism",
            "cfcmt_physical_fused": "cfcmt_physical_mechanism",
            "cfcmt_physical_latent_fused": "cfcmt_physical_latent_mechanism",
        }.get(family)
        reused_loo_family = {
            "cfcmt_fused": "cfcmt_mechanism",
            "cfcmt_fused_rigid": "causal_rigid_advantage",
            "cfcmt_physical_fused": "cfcmt_physical_mechanism",
            "cfcmt_physical_latent_fused": "cfcmt_physical_latent_mechanism",
        }.get(family)
        (
            models[family],
            fit_diagnostics[family],
            prediction_bundle,
            parallel_diagnostics,
        ) = _fit_family_with_source_loo_v3(
            contrast,
            family=family,
            cfcmt_config=config,
            target_domain=target_domain,
            parallel_workers=parallel_workers,
            reused_source_model=(
                raw_family_models.get(reused_source_family)
                if reused_source_family is not None
                else None
            ),
            reused_prediction_bundle=(
                source_loo_prediction_bundles.get(reused_loo_family)
                if reused_loo_family is not None
                else None
            ),
        )
        raw_family_models[family] = models[family]
        source_loo_prediction_bundles[family] = dict(prediction_bundle)
        fit_diagnostics[family]["source_loo_parallelism"] = parallel_diagnostics
        if family == "cfcmt_boosted":
            (
                objective_modes[family],
                objective_diagnostics[family],
                prediction_bundle,
            ) = calibrate_mechanism_objective_v3(
                contrast,
                family=family,
                cfcmt_config=config,
                prediction_bundle=prediction_bundle,
            )
        else:
            objective_modes[family] = "control_only"
            objective_diagnostics[family] = {
                "family": family,
                "selected": "control_only",
                "gate_reason": "matched_baseline_control_objective",
                "grid": [],
            }
        records_bundle = _source_loo_action_records_v3(
            contrast,
            family=family,
            cfcmt_config=config,
            objective_mode=objective_modes[family],
            prediction_bundle=prediction_bundle,
        )
        records_bundle, uncertainty_diagnostics = (
            _calibrate_source_loo_uncertainty_v3(records_bundle)
        )
        uncertainty_scale = float(uncertainty_diagnostics["scale"])
        models[family] = _UncertaintyScaledModelV3(
            base_model=models[family],
            scale=uncertainty_scale,
        )
        fit_diagnostics[family]["uncertainty_calibration"] = (
            uncertainty_diagnostics
        )
        guards[family], guard_diagnostics[family] = calibrate_contrast_guard_v3(
            contrast,
            family=family,
            cfcmt_config=config,
            records_bundle=records_bundle,
        )
        regularizers[family], regularizer_diagnostics[family] = calibrate_prior_regularization_v3(
            contrast,
            family=family,
            cfcmt_config=config,
            records_bundle=records_bundle,
        )
        family_elapsed = time.monotonic() - family_started
        fit_diagnostics[family]["family_total_elapsed_sec"] = float(family_elapsed)
        print(
            "CFCMT_FAMILY_PROGRESS "
            f"target_domain={target_domain or 'source_only'} "
            f"{family_index}/{len(requested_families)} family={family} "
            f"family_elapsed={family_elapsed:.1f}s "
            f"total_elapsed={time.monotonic() - fit_started:.1f}s",
            flush=True,
        )
    total_fit_elapsed = time.monotonic() - fit_started
    return FittedContrastModelsV3(
        family_models=models,
        prior_policy=prior_policy,
        prior_spec=prior,
        prediction_horizon_sec=int(absolute_dataset.metadata["counterfactual_horizon_sec"]),
        objective_modes=objective_modes,
        guards=guards,
        regularizers=regularizers,
        target_support=None,
        hierarchy_layers={
            "guard": CFCMT_HIERARCHY_LAYERS_V3,
            "regularized": CFCMT_HIERARCHY_LAYERS_V3,
        },
        diagnostics={
            "source_domains": sorted(str(item) for item in np.unique(absolute_dataset.domains)),
            "source_absolute_rows": int(absolute_dataset.size),
            "source_contrast_rows": int(contrast.size),
            "prior_selection": prior_diagnostics,
            "fit": fit_diagnostics,
            "mechanism_objective_selection": objective_diagnostics,
            "guard_calibration": guard_diagnostics,
            "prior_regularization_calibration": regularizer_diagnostics,
            "uncertainty_calibration": {
                family: fit_diagnostics[family]["uncertainty_calibration"]
                for family in requested_families
            },
            "cfcmt_hierarchy": {
                "protocol": CFCMT_HIERARCHY_PROTOCOL_V3,
                "layers": list(CFCMT_HIERARCHY_LAYERS_V3),
                "target_specialist_requires_fitted_target_head": True,
            },
            "fit_runtime": {
                "protocol": "monotonic_family_wall_clock_v1",
                "total_elapsed_sec": float(total_fit_elapsed),
                "family_elapsed_sec": {
                    family: float(
                        fit_diagnostics[family]["family_total_elapsed_sec"]
                    )
                    for family in requested_families
                },
            },
        },
    )


def calibrate_target_policy_selection_v3(
    calibration_absolute_dataset: MechanismDataset,
    *,
    fitted_models: FittedContrastModelsV3,
    cfcmt_config: MechanismFitConfig | None = None,
    min_calibration_groups: int = 4,
    guard_selection_mode: str = "strict_worst_domain",
) -> dict[str, Any]:
    """Select target-specific safety controls on held-out adaptation groups.

    The fitted models must not have seen ``calibration_absolute_dataset``. Each
    action group is treated as a separate validation block, so a configuration
    is accepted only when its selected action is non-inferior in every held-out
    group and improves a material fraction of groups.
    """

    groups = action_group_ids(calibration_absolute_dataset)
    unique_groups = tuple(np.unique(groups).tolist())
    if len(unique_groups) < int(min_calibration_groups):
        return {
            "eligible": False,
            "reason": "insufficient_disjoint_target_calibration_groups",
            "group_count": len(unique_groups),
            "min_calibration_groups": int(min_calibration_groups),
            "guards": {},
            "regularizers": {},
            "diagnostics": {},
        }

    config = cfcmt_config or MechanismFitConfig()
    contrast = build_action_contrast_dataset(
        calibration_absolute_dataset,
        reference_policy=fitted_models.prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    selected_guards: dict[str, ContrastGuardConfig] = {}
    selected_regularizers: dict[str, PriorRegularizationConfig] = {}
    diagnostics: dict[str, Any] = {}
    for family, model in fitted_models.family_models.items():
        prediction_bundle = {}
        for group_index, group in enumerate(unique_groups):
            valid = _subset_contrast(contrast, action_group_ids(contrast) == group)
            prediction = model.predict(valid)
            if fitted_models.target_support is not None:
                local_support = fitted_models.target_support.support(valid)
                prediction = {
                    mechanism: {
                        **values,
                        "context_trust": np.minimum(
                            np.asarray(values["context_trust"], dtype=float),
                            local_support,
                        ),
                    }
                    for mechanism, values in prediction.items()
                }
            prediction_bundle[f"target_calibration_group_{group_index:04d}"] = (
                valid,
                prediction,
            )
        records_bundle = _source_loo_action_records_v3(
            contrast,
            family=family,
            cfcmt_config=config,
            objective_mode=fitted_models.objective_modes[family],
            prediction_bundle=prediction_bundle,
        )
        selected_guards[family], guard_diagnostics = calibrate_contrast_guard_v3(
            contrast,
            family=family,
            cfcmt_config=config,
            noninferiority_tolerance=0.0,
            terminal_noninferiority_tolerance=0.0,
            selection_mode=guard_selection_mode,
            records_bundle=records_bundle,
        )
        selected_regularizers[family], regularizer_diagnostics = (
            calibrate_prior_regularization_v3(
                contrast,
                family=family,
                cfcmt_config=config,
                noninferiority_tolerance=0.0,
                min_mean_improvement=1e-4,
                min_override_fraction=0.05,
                min_improving_domain_fraction=0.25,
                records_bundle=records_bundle,
            )
        )
        diagnostics[family] = {
            "guard": guard_diagnostics,
            "regularizer": regularizer_diagnostics,
        }
    return {
        "eligible": True,
        "reason": "disjoint_target_group_calibration",
        "group_count": len(unique_groups),
        "min_calibration_groups": int(min_calibration_groups),
        "guards": selected_guards,
        "regularizers": selected_regularizers,
        "diagnostics": diagnostics,
    }


def _absolute_target_candidates(
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
    prediction_horizon_sec: int,
) -> tuple[tuple[Any, ...], MechanismDataset]:
    candidates = tuple(
        _candidate_for_state(state.info, phase_state)
        for phase_state in executor.feasible_states_now()
    )
    features = np.vstack(
        [
            candidate_features_v3(
                state,
                candidate,
                executor,
                control_interval_sec=control_interval_sec,
            )
            for candidate in candidates
        ]
    )
    context = np.repeat(state.context[None, :], len(candidates), axis=0)
    priors = uncalibrated_mechanism_priors_v3(
        features,
        context,
        control_interval_sec=prediction_horizon_sec,
    )
    dataset = MechanismDataset(
        feature_names=FEATURE_NAMES_V3,
        features=features,
        context_names=CONTEXT_NAMES_V3,
        context=context,
        priors=priors,
        targets={name: values.copy() for name, values in priors.items()},
        domains=np.asarray(["target"] * len(candidates)),
        metadata={
            "action_group_ids": ["target_state"] * len(candidates),
            "row_tls": [str(state.tls_id)] * len(candidates),
            "row_times": [float(state.sim_time)] * len(candidates),
            "candidate_states": [str(candidate.state) for candidate in candidates],
        },
    )
    return candidates, dataset


def select_target_capacity_layers_v3(
    *,
    target_specialist_fitted: bool,
    guards: Mapping[str, ContrastGuardConfig],
    regularizers: Mapping[str, PriorRegularizationConfig],
    calibration_diagnostics: Mapping[str, Mapping[str, Any]],
    min_mechanism_material_gain: float = 0.01,
) -> dict[str, Any]:
    """Select one deployable expert using disjoint target calibration groups.

    Independently calibrated confidence bounds are not necessarily comparable
    across model families. The selector therefore makes one target-level
    capacity choice before deployment instead of taking the largest bound at
    every state. Once a target head exists, the target specialist replaces the
    source-only rigid core as the anchor; the mechanism model may replace that
    anchor only after a material held-out calibration gain.
    """

    anchor_family = (
        CFCMT_TARGET_SPECIALIST_FAMILY_V3
        if target_specialist_fitted
        else CFCMT_CORE_FAMILY_V3
    )
    anchor_layer = (
        "target_specialist"
        if target_specialist_fitted
        else "causal_core_fallback"
    )

    def candidate(
        family: str,
        *,
        mode: str,
        configs: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        config = configs.get(family)
        selected = (
            calibration_diagnostics.get(family, {})
            .get(mode, {})
            .get("selected")
        )
        if config is None or not bool(getattr(config, "enabled", False)) or not selected:
            return None
        domain_values = np.asarray(
            list(dict(selected.get("domain_mean_delta", {})).values()),
            dtype=float,
        )
        mean_delta = float(selected.get("mean_actual_delta", 0.0))
        worst_delta = float(selected.get("worst_domain_delta", 0.0))
        dispersion = float(np.std(domain_values)) if domain_values.size else 0.0
        robust_score = mean_delta + 0.25 * dispersion + 0.25 * worst_delta
        return {
            "family": family,
            "mean_actual_delta": mean_delta,
            "worst_domain_delta": worst_delta,
            "calibration_dispersion": dispersion,
            "robust_score": float(robust_score),
            "calibration_group_count": int(domain_values.size),
        }

    selected_layers: dict[str, tuple[str, ...]] = {}
    mode_diagnostics: dict[str, Any] = {}
    for mode, configs in (("guard", guards), ("regularizer", regularizers)):
        anchor = candidate(anchor_family, mode=mode, configs=configs)
        mechanism = candidate("cfcmt_mechanism", mode=mode, configs=configs)
        if anchor is None:
            layers: tuple[str, ...] = ()
            selected_family = None
            reason = "calibrated_anchor_unavailable"
        elif (
            mechanism is not None
            and float(mechanism["robust_score"])
            <= float(anchor["robust_score"])
            - float(min_mechanism_material_gain)
        ):
            layers = ("mechanism_refinement",)
            selected_family = "cfcmt_mechanism"
            reason = "mechanism_materially_improves_over_calibrated_anchor"
        else:
            layers = (anchor_layer,)
            selected_family = anchor_family
            reason = (
                "calibrated_anchor_selected"
                if mechanism is not None
                else "calibrated_anchor_only"
            )
        hierarchy_mode = "guard" if mode == "guard" else "regularized"
        selected_layers[hierarchy_mode] = layers
        mode_diagnostics[hierarchy_mode] = {
            "selected_family": selected_family,
            "enabled_layers": list(layers),
            "reason": reason,
            "anchor": anchor,
            "mechanism": mechanism,
        }
    return {
        "protocol": CFCMT_HIERARCHY_SELECTION_PROTOCOL_V3,
        "target_specialist_fitted": bool(target_specialist_fitted),
        "anchor_family": anchor_family,
        "anchor_layer": anchor_layer,
        "min_mechanism_material_gain": float(min_mechanism_material_gain),
        "layers": selected_layers,
        "modes": mode_diagnostics,
    }


def has_fitted_target_specialist_v3(model: Any) -> bool:
    """Return whether a wrapped target-adapted model has a trained target head."""

    current = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if (
            getattr(current, "target_model", None) is not None
            and float(getattr(current, "target_weight", 0.0)) > 0.0
        ):
            return True
        current = getattr(current, "base_model", None)
    return False


def _select_hierarchical_proposal(
    refinement: ContrastProposal,
    causal_core: ContrastProposal,
    target_specialist: ContrastProposal | None = None,
    *,
    enabled_layers: Sequence[str] = CFCMT_HIERARCHY_LAYERS_V3,
) -> ContrastProposal:
    """Choose the strongest admissible bound in the CFCMT fallback hierarchy.

    Every proposal is relative to the same pressure-policy action. Ties retain
    the lower-capacity causal core, so a richer layer must provide a strictly
    stronger conservative improvement bound to replace it.
    """

    enabled = set(enabled_layers)
    layered = [(causal_core, "causal_core_fallback")]
    if target_specialist is not None:
        layered.append((target_specialist, "target_specialist"))
    layered.append((refinement, "mechanism_refinement"))
    eligible = [
        item
        for item in layered
        if item[1] in enabled and item[0].eligible
    ]
    if eligible:
        selected, layer = max(eligible, key=lambda item: float(item[0].priority))
        return replace(selected, selection_layer=layer)

    considered = [item for item in layered if item[1] in enabled]
    learned_differs = any(proposal.learned_differs for proposal, _ in considered)
    rejection = next(
        (
            proposal.rejection
            for proposal, _ in considered
            if proposal.learned_differs and proposal.rejection is not None
        ),
        None,
    )
    return ContrastProposal(
        prior_candidate=causal_core.prior_candidate,
        selected_candidate=causal_core.prior_candidate,
        learned_differs=bool(learned_differs),
        eligible=False,
        priority=0.0,
        rejection=rejection,
        selection_layer="pressure_prior_fallback",
    )


def _contrast_proposal(
    *,
    model: Any,
    family: str,
    models: FittedContrastModelsV3,
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    control_interval_sec: int,
    guarded: bool,
    regularized: bool,
    pressure_gated: bool = False,
    hierarchical_guarded: bool = False,
    _allow_hierarchy: bool = True,
) -> ContrastProposal:
    if (
        _allow_hierarchy
        and getattr(models, "action_originator", None) is None
        and family == "cfcmt_mechanism"
        and (guarded or regularized or pressure_gated or hierarchical_guarded)
        and CFCMT_CORE_FAMILY_V3 in models.family_models
    ):
        refinement = _contrast_proposal(
            model=model,
            family=family,
            models=models,
            state=state,
            executor=executor,
            control_interval_sec=control_interval_sec,
            guarded=guarded,
            regularized=regularized,
            pressure_gated=pressure_gated,
            hierarchical_guarded=hierarchical_guarded,
            _allow_hierarchy=False,
        )
        causal_core = _contrast_proposal(
            model=models.family_models[CFCMT_CORE_FAMILY_V3],
            family=CFCMT_CORE_FAMILY_V3,
            models=models,
            state=state,
            executor=executor,
            control_interval_sec=control_interval_sec,
            guarded=guarded,
            regularized=regularized,
            pressure_gated=pressure_gated,
            hierarchical_guarded=hierarchical_guarded,
            _allow_hierarchy=False,
        )
        target_specialist = None
        if (
            CFCMT_TARGET_SPECIALIST_FAMILY_V3 in models.family_models
            and has_fitted_target_specialist_v3(
                models.family_models[CFCMT_TARGET_SPECIALIST_FAMILY_V3]
            )
        ):
            target_specialist = _contrast_proposal(
                model=models.family_models[CFCMT_TARGET_SPECIALIST_FAMILY_V3],
                family=CFCMT_TARGET_SPECIALIST_FAMILY_V3,
                models=models,
                state=state,
                executor=executor,
                control_interval_sec=control_interval_sec,
                guarded=guarded,
                regularized=regularized,
                pressure_gated=pressure_gated,
                hierarchical_guarded=hierarchical_guarded,
                _allow_hierarchy=False,
            )
        hierarchy_mode = (
            "regularized"
            if regularized or pressure_gated or hierarchical_guarded
            else "guard"
        )
        return _select_hierarchical_proposal(
            refinement,
            causal_core,
            target_specialist,
            enabled_layers=models.hierarchy_layers.get(
                hierarchy_mode,
                CFCMT_HIERARCHY_LAYERS_V3,
            ),
        )

    candidates, absolute = _absolute_target_candidates(
        state,
        executor,
        control_interval_sec=control_interval_sec,
        prediction_horizon_sec=models.prediction_horizon_sec,
    )
    if len(candidates) == 1:
        return ContrastProposal(
            prior_candidate=candidates[0],
            selected_candidate=candidates[0],
            learned_differs=False,
            eligible=False,
            priority=0.0,
        )
    contrast, reference = make_target_action_contrasts(
        absolute,
        reference_policy=models.prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    action_originator = getattr(models, "action_originator", None)
    if action_originator is not None:
        decision = action_originator.select(
            contrast,
            reference_index=reference,
        )
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[int(decision.selected_index)],
            learned_differs=bool(decision.learned_differs),
            eligible=bool(decision.eligible),
            priority=float(decision.priority),
            rejection=decision.rejection,
            selection_layer="target_spatiotemporal_latent",
            originator_diagnostics=decision.to_dict(),
        )
    score, uncertainty, trust, _ = _group_adjusted_scores(
        contrast,
        model.predict(contrast),
        objective_mode=models.objective_modes[family],
    )
    relative_rule_gap = _relative_contrast_rule_gap(contrast)
    base_trust = np.asarray(trust, dtype=float)
    if isinstance(models.target_support, HierarchicalTargetActionSupport):
        support_components = models.target_support.component_support(contrast)
        pressure_local_support = np.asarray(
            support_components["pressure"], dtype=float
        )
        conformal_local_support = np.asarray(
            support_components["conformal"], dtype=float
        )
    else:
        local_support = (
            models.target_support.support(contrast)
            if models.target_support is not None
            else np.ones(contrast.size, dtype=float)
        )
        pressure_local_support = np.asarray(local_support, dtype=float)
        conformal_local_support = np.asarray(local_support, dtype=float)
    pressure_trust = np.minimum(base_trust, pressure_local_support)
    conformal_trust = np.minimum(base_trust, conformal_local_support)
    local_support = np.minimum(pressure_local_support, conformal_local_support)
    trust = np.minimum(pressure_trust, conformal_trust)
    learned = int(np.argmin(score))
    if learned == reference:
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[reference],
            learned_differs=False,
            eligible=False,
            priority=0.0,
        )
    if hierarchical_guarded:
        feature_index = {
            name: index for index, name in enumerate(contrast.feature_names)
        }
        if "total_veh" not in feature_index:
            raise KeyError("hierarchical contrast lacks total_veh")
        decision = hierarchical_guard_decision_v3(
            learned_differs=True,
            predicted_delta=float(score[learned]),
            uncertainty=float(uncertainty[learned]),
            relative_rule_gap=float(relative_rule_gap[learned]),
            pressure_context_trust=float(pressure_trust[learned]),
            conformal_context_trust=float(conformal_trust[learned]),
            total_vehicles=float(
                contrast.features[learned, feature_index["total_veh"]]
            ),
            regularizer=models.regularizers[family],
            guard=models.guards[family],
        )
        if not bool(decision["eligible"]):
            rejection = str(decision["rejection"])
            if rejection in {"pressure_support", "conformal_support"}:
                rejection = "local_support"
            elif rejection in {
                "pressure_confidence",
                "conformal_confidence",
            }:
                rejection = "confidence"
            elif rejection == "conformal_service_safety":
                rejection = "service_safety"
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection=rejection,
            )
        veto_diagnostics = None
        if models.execution_veto is not None:
            veto_diagnostics = models.execution_veto.evaluate(
                contrast,
                candidate_index=learned,
                reference_index=reference,
            )
            if not bool(veto_diagnostics["eligible"]):
                return ContrastProposal(
                    prior_candidate=candidates[reference],
                    selected_candidate=candidates[reference],
                    learned_differs=True,
                    eligible=False,
                    priority=0.0,
                    rejection="long_horizon_veto",
                    veto_diagnostics=veto_diagnostics,
                )
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[learned],
            learned_differs=True,
            eligible=True,
            priority=float(decision["priority"]),
            veto_diagnostics=veto_diagnostics,
        )
    if pressure_gated:
        config = models.regularizers[family]
        if not config.enabled:
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection="disabled",
            )
        if float(trust[learned]) < float(config.min_context_trust):
            rejection = (
                "local_support"
                if float(local_support[learned]) < float(config.min_context_trust)
                else "context"
            )
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection=rejection,
            )
        feature_index = {
            name: index for index, name in enumerate(contrast.feature_names)
        }
        if "total_veh" not in feature_index:
            raise KeyError("pressure-gated contrast lacks total_veh")
        if (
            float(contrast.features[learned, feature_index["total_veh"]])
            < MIN_OPERATIONAL_TOTAL_VEHICLES_V3
        ):
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection="service_safety",
            )
        objective = (
            float(relative_rule_gap[learned])
            + float(config.blend_weight) * float(score[learned])
            + float(config.risk_multiplier) * float(uncertainty[learned])
        )
        if objective >= 0.0:
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection="confidence",
            )
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[learned],
            learned_differs=True,
            eligible=True,
            priority=max(-objective, 0.0),
        )
    if regularized:
        config = models.regularizers[family]
        if not config.enabled:
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection="disabled",
            )
        objective = (
            _relative_contrast_rule_gap(contrast)
            + float(config.blend_weight) * score
            + float(config.risk_multiplier) * uncertainty
        )
        objective = np.asarray(objective, dtype=float)
        objective[trust < float(config.min_context_trust)] = np.inf
        objective[reference] = 0.0
        selected = int(np.argmin(objective))
        if selected == reference:
            if float(trust[learned]) < float(config.min_context_trust):
                rejection = (
                    "local_support"
                    if float(local_support[learned]) < float(config.min_context_trust)
                    else "context"
                )
            else:
                rejection = "confidence"
            return ContrastProposal(
                prior_candidate=candidates[reference],
                selected_candidate=candidates[reference],
                learned_differs=True,
                eligible=False,
                priority=0.0,
                rejection=rejection,
            )
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[selected],
            learned_differs=True,
            eligible=True,
            priority=max(-float(objective[selected]), 0.0),
        )
    if not guarded:
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[learned],
            learned_differs=True,
            eligible=True,
            priority=max(-float(score[learned]), 0.0),
        )

    config = models.guards[family]
    if not config.enabled:
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[reference],
            learned_differs=True,
            eligible=False,
            priority=0.0,
            rejection="disabled",
        )
    if float(trust[learned]) < float(config.min_context_trust):
        rejection = (
            "local_support"
            if float(local_support[learned]) < float(config.min_context_trust)
            else "context"
        )
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[reference],
            learned_differs=True,
            eligible=False,
            priority=0.0,
            rejection=rejection,
        )
    if float(relative_rule_gap[learned]) > float(config.max_relative_rule_gap) + 1e-12:
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[reference],
            learned_differs=True,
            eligible=False,
            priority=0.0,
            rejection="service_safety",
        )
    upper_cost = (
        float(score[learned])
        + float(config.risk_multiplier) * float(uncertainty[learned])
        + float(config.margin)
    )
    if upper_cost >= 0.0:
        return ContrastProposal(
            prior_candidate=candidates[reference],
            selected_candidate=candidates[reference],
            learned_differs=True,
            eligible=False,
            priority=0.0,
            rejection="confidence",
        )
    return ContrastProposal(
        prior_candidate=candidates[reference],
        selected_candidate=candidates[learned],
        learned_differs=True,
        eligible=True,
        priority=max(-upper_cost, 0.0),
    )


def _coordinate_contrast_proposals(
    proposals: Mapping[str, ContrastProposal],
    *,
    cooldowns: dict[str, int],
    cooldown_intervals: int,
    max_simultaneous_overrides: int | None,
    audit: ContrastGuardAudit,
    conflicts: Mapping[str, Sequence[str]] | None = None,
    current_green_states: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    candidates = []
    override_kinds: dict[str, str] = {}
    for tls_id, proposal in proposals.items():
        audit.decisions += 1
        if proposal.selection_layer in {
            "mechanism_refinement",
            "target_specialist",
            "causal_core_fallback",
            "pressure_prior_fallback",
        }:
            audit.hierarchical_decisions += 1
        if proposal.selection_layer == "target_spatiotemporal_latent":
            audit.target_originator_decisions += 1
        if proposal.selection_layer == "pressure_prior_fallback":
            audit.hierarchical_prior_fallbacks += 1
        if not proposal.learned_differs:
            audit.prior_agreements += 1
            continue
        if not proposal.eligible:
            if proposal.rejection == "disabled":
                audit.rejected_disabled += 1
            elif proposal.rejection == "local_support":
                audit.rejected_local_support += 1
            elif proposal.rejection == "context":
                audit.rejected_context += 1
            elif proposal.rejection == "confidence":
                audit.rejected_confidence += 1
            elif proposal.rejection == "service_safety":
                audit.rejected_service_safety += 1
            elif proposal.rejection == "minimum_priority":
                audit.rejected_minimum_priority += 1
            elif proposal.rejection == "minimum_queue":
                audit.rejected_minimum_queue += 1
            elif proposal.rejection == "minimum_vehicles":
                audit.rejected_minimum_vehicles += 1
            elif proposal.rejection == "maximum_speed":
                audit.rejected_maximum_speed += 1
            elif proposal.rejection == "global_cooldown":
                audit.rejected_global_cooldown += 1
            elif proposal.rejection == "global_stay_budget":
                audit.rejected_global_stay_budget += 1
            elif proposal.rejection == "long_horizon_veto":
                audit.rejected_long_horizon_veto += 1
            continue
        audit.proposed_overrides += 1
        if current_green_states is not None:
            kind = _override_execution_kind(
                proposal, current_green_states[str(tls_id)]
            )
            if kind is not None:
                override_kinds[str(tls_id)] = kind
                if kind == "stay":
                    audit.proposed_stay_overrides += 1
                else:
                    audit.proposed_switch_overrides += 1
        if int(cooldowns.get(tls_id, 0)) > 0:
            audit.rejected_cooldown += 1
            continue
        candidates.append((float(proposal.priority), str(tls_id)))

    conflict_map = conflicts or {tls_id: () for _, tls_id in candidates}
    active_cooldowns = {
        tls_id for tls_id, remaining in cooldowns.items() if int(remaining) > 0
    }
    temporally_blocked = {
        tls_id
        for _, tls_id in candidates
        if tls_id in active_cooldowns
        or any(neighbor in active_cooldowns for neighbor in conflict_map.get(tls_id, ()))
    }
    if temporally_blocked:
        audit.rejected_cooldown += len(temporally_blocked)
    priorities = {
        tls_id: priority
        for priority, tls_id in candidates
        if tls_id not in temporally_blocked
    }
    selected, spatially_rejected = greedy_independent_interventions(
        priorities,
        conflict_map,
        max_selected=max_simultaneous_overrides,
    )
    selected_ids = set(selected)
    audit.accepted_overrides += len(selected_ids)
    audit.accepted_stay_overrides += sum(
        override_kinds.get(tls_id) == "stay" for tls_id in selected_ids
    )
    audit.accepted_switch_overrides += sum(
        override_kinds.get(tls_id) == "switch" for tls_id in selected_ids
    )
    audit.selected_mechanism_refinements += sum(
        proposals[tls_id].selection_layer == "mechanism_refinement"
        for tls_id in selected_ids
    )
    audit.selected_target_specialists += sum(
        proposals[tls_id].selection_layer == "target_specialist"
        for tls_id in selected_ids
    )
    audit.selected_causal_core_fallbacks += sum(
        proposals[tls_id].selection_layer == "causal_core_fallback"
        for tls_id in selected_ids
    )
    audit.selected_target_originators += sum(
        proposals[tls_id].selection_layer == "target_spatiotemporal_latent"
        for tls_id in selected_ids
    )
    audit.rejected_coordination += len(spatially_rejected)
    actions = {
        tls_id: proposal.selected_candidate if tls_id in selected_ids else proposal.prior_candidate
        for tls_id, proposal in proposals.items()
    }
    for tls_id in tuple(cooldowns):
        cooldowns[tls_id] = max(int(cooldowns[tls_id]) - 1, 0)
    for tls_id in selected_ids:
        cooldowns[tls_id] = max(int(cooldown_intervals), 0)
    return actions


def _apply_residual_execution_trust_region(
    proposals: Mapping[str, ContrastProposal],
    states: Mapping[str, LocalTransitionState],
    config: ResidualExecutionTrustRegionConfig | None,
    *,
    global_cooldown_active: bool = False,
    current_green_states: Mapping[str, str] | None = None,
    stay_bypass_remaining: int | None = None,
) -> dict[str, ContrastProposal]:
    if config is None or not bool(config.enabled):
        return dict(proposals)
    filtered: dict[str, ContrastProposal] = {}
    for tls_id, proposal in proposals.items():
        rejection = None
        if proposal.eligible:
            state = states[tls_id]
            total_queue = float(sum(state.q_by_lane.values()))
            total_vehicles = float(sum(state.veh_by_lane.values()))
            speeds = tuple(float(value) for value in state.speed_by_lane.values())
            mean_speed = float(np.mean(speeds)) if speeds else 0.0
            stay_override = bool(
                current_green_states is not None
                and _override_execution_kind(
                    proposal, current_green_states[str(tls_id)]
                )
                == "stay"
            )
            if global_cooldown_active:
                if not (
                    config.allow_stay_during_global_cooldown and stay_override
                ):
                    rejection = "global_cooldown"
                elif (
                    config.max_stay_overrides_per_global_cooldown is not None
                    and int(stay_bypass_remaining or 0) <= 0
                ):
                    rejection = "global_stay_budget"
            elif float(proposal.priority) < float(config.min_priority):
                rejection = "minimum_priority"
            elif total_queue < float(config.min_total_queue):
                rejection = "minimum_queue"
            elif total_vehicles < float(config.min_total_vehicles):
                rejection = "minimum_vehicles"
            elif (
                config.max_mean_speed is not None
                and mean_speed > float(config.max_mean_speed)
            ):
                rejection = "maximum_speed"
        if rejection is None:
            filtered[tls_id] = proposal
        else:
            filtered[tls_id] = replace(
                proposal,
                selected_candidate=proposal.prior_candidate,
                eligible=False,
                priority=0.0,
                rejection=rejection,
            )
    return filtered


def _policy_family_v3(policy: str) -> str | None:
    for family in MODEL_FAMILIES_V3:
        if policy.startswith(f"{family}_contrast_"):
            return family
    return None


def _pressure_candidate_v3(
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    spec: str | PressurePolicySpec | Mapping[str, Any],
    *,
    control_interval_sec: int,
) -> Any:
    candidates = tuple(
        _candidate_for_state(state.info, phase_state)
        for phase_state in executor.feasible_states_now()
    )
    features = np.vstack(
        [
            candidate_features_v2(
                state,
                candidate,
                executor,
                control_interval_sec=control_interval_sec,
            )
            for candidate in candidates
        ]
    )
    scores = pressure_scores_from_feature_matrix(features, FEATURE_NAMES_V2, spec)
    return candidates[int(np.argmax(scores))]


def evaluate_policy_v3(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    policy: str,
    models: FittedContrastModelsV3 | None,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    tripinfo_output: Path | None = None,
    pressure_spec_override: PressurePolicySpec | Mapping[str, Any] | None = None,
    residual_coordination_mode: str = "sparse",
    residual_cooldown_intervals_override: int | None = None,
    residual_execution_trust_region: ResidualExecutionTrustRegionConfig
    | None = None,
) -> dict[str, Any]:
    coordination_modes = {"sparse", "spatial_only", "direct"}
    if residual_coordination_mode not in coordination_modes:
        raise ValueError(
            "residual_coordination_mode must be one of "
            f"{sorted(coordination_modes)}, got {residual_coordination_mode!r}"
        )
    if (
        residual_cooldown_intervals_override is not None
        and int(residual_cooldown_intervals_override) < 0
    ):
        raise ValueError("residual cooldown override must be nonnegative")
    execution_trust_region = (
        residual_execution_trust_region
        or ResidualExecutionTrustRegionConfig()
    )
    _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=tripinfo_output)
    queues: list[float] = []
    departed = 0
    arrived = 0
    loaded = 0
    executors: dict[str, SafePhaseExecutor] = {}
    guard_audit = ContrastGuardAudit()
    accepted_intervention_trace: list[dict[str, Any]] = []
    residual_cooldowns: dict[str, int] = {}
    residual_global_cooldown_remaining = 0
    residual_global_stay_bypass_remaining = 0
    max_simultaneous_overrides: int | None = None
    cooldown_intervals = 0
    intervention_graph: TlsInterventionGraph | None = None
    queue_proxies: list[float] = []
    active_vehicle_counts: list[float] = []
    pending_vehicle_counts: list[float] = []
    starting_teleports = 0
    ending_teleports = 0
    collision_events = 0
    collision_event_steps = 0
    collision_incident_keys: set[tuple[str, str, str, str]] = set()
    collision_samples: list[dict[str, Any]] = []
    emergency_stops = 0
    closed = False
    try:
        infos = _tls_phase_infos(sumo_api)
        routing_graph = build_signal_routing_graph(sumo_api, infos)
        lanes = _controlled_lane_set(infos)
        intervention_graph = add_routing_conflicts(
            build_tls_intervention_graph(infos), routing_graph.adjacency
        )
        context = _scenario_context_v3(
            sumocfg=sumocfg,
            infos=infos,
            control_interval_sec=control_interval_sec,
        )
        if policy != "fixed_program":
            executors = build_safe_phase_executors(sumo_api, infos)
            residual_cooldowns = {tls_id: 0 for tls_id in infos}
        learned_family = _policy_family_v3(policy)
        if learned_family is not None:
            if models is None or learned_family not in models.family_models:
                raise ValueError(f"unknown or unfitted V3 policy: {policy}")
            sparse_cooldown_intervals = max(
                int(math.ceil(models.prediction_horizon_sec / max(control_interval_sec, 1))) - 1,
                0,
            )
            cooldown_intervals = (
                sparse_cooldown_intervals
                if residual_coordination_mode == "sparse"
                else 0
            )
            if residual_cooldown_intervals_override is not None:
                cooldown_intervals = int(residual_cooldown_intervals_override)
        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + float(duration_sec)
        while float(sumo_api.simulation.getTime()) < end_time:
            if policy != "fixed_program":
                states = _read_graph_states_v3(
                    sumo_api, infos, context, routing_graph
                )
                if learned_family is not None:
                    residual_active = float(sumo_api.simulation.getTime()) >= (
                        begin_time + float(warmup_sec)
                    )
                    if not residual_active:
                        guard_audit.warmup_prior_decisions += len(states)
                        for tls_id, state in states.items():
                            candidate = _pressure_candidate_v3(
                                state,
                                executors[tls_id],
                                models.prior_spec,
                                control_interval_sec=control_interval_sec,
                            )
                            executors[tls_id].request(candidate.state)
                    else:
                        model = models.family_models[learned_family]
                        proposals = {
                            tls_id: _contrast_proposal(
                                model=model,
                                family=learned_family,
                                models=models,
                                state=state,
                                executor=executors[tls_id],
                                control_interval_sec=control_interval_sec,
                                guarded=policy.endswith("_guard"),
                                regularized=policy.endswith("_regularized"),
                                pressure_gated=policy.endswith("_pressure_gate"),
                                hierarchical_guarded=policy.endswith(
                                    "_hierarchical_guard"
                                ),
                            )
                            for tls_id, state in states.items()
                        }
                        executor_snapshots = {
                            tls_id: executors[tls_id].snapshot()
                            for tls_id in states
                        }
                        current_green_states = {
                            tls_id: str(snapshot["current_green_state"])
                            for tls_id, snapshot in executor_snapshots.items()
                        }
                        global_cooldown_active_this_interval = bool(
                            execution_trust_region.enabled
                            and residual_global_cooldown_remaining > 0
                        )
                        stay_bypass_remaining_before = int(
                            residual_global_stay_bypass_remaining
                        )
                        proposals = _apply_residual_execution_trust_region(
                            proposals,
                            states,
                            execution_trust_region,
                            global_cooldown_active=(
                                global_cooldown_active_this_interval
                            ),
                            current_green_states=current_green_states,
                            stay_bypass_remaining=(
                                residual_global_stay_bypass_remaining
                            ),
                        )
                        if residual_global_cooldown_remaining > 0:
                            residual_global_cooldown_remaining -= 1
                        if execution_trust_region.enabled:
                            max_simultaneous_overrides = (
                                execution_trust_region.max_simultaneous_overrides
                            )
                            if (
                                global_cooldown_active_this_interval
                                and execution_trust_region.max_stay_overrides_per_global_cooldown
                                is not None
                            ):
                                max_simultaneous_overrides = min(
                                    int(max_simultaneous_overrides)
                                    if max_simultaneous_overrides is not None
                                    else residual_global_stay_bypass_remaining,
                                    residual_global_stay_bypass_remaining,
                                )
                        coordinated = _coordinate_contrast_proposals(
                            proposals,
                            cooldowns=residual_cooldowns,
                            cooldown_intervals=cooldown_intervals,
                            max_simultaneous_overrides=max_simultaneous_overrides,
                            audit=guard_audit,
                            conflicts=(
                                intervention_graph.conflicts
                                if residual_coordination_mode != "direct"
                                else {tls_id: () for tls_id in proposals}
                            ),
                            current_green_states=current_green_states,
                        )
                        executed_override_this_interval = False
                        executed_stay_during_global_cooldown = 0
                        for tls_id, candidate in coordinated.items():
                            proposal = proposals[tls_id]
                            is_override = (
                                str(candidate.state)
                                != str(proposal.prior_candidate.state)
                            )
                            executor_before = executor_snapshots[tls_id]
                            override_kind = (
                                _override_execution_kind(
                                    proposal,
                                    str(executor_before["current_green_state"]),
                                )
                                if is_override
                                else None
                            )
                            request_accepted = executors[tls_id].request(
                                candidate.state
                            )
                            stay_effective = bool(
                                is_override
                                and override_kind == "stay"
                                and str(executor_before["mode"]) == "green"
                                and str(candidate.state)
                                == str(executor_before["current_green_state"])
                            )
                            execution_effective = bool(
                                is_override and (request_accepted or stay_effective)
                            )
                            if is_override and request_accepted:
                                guard_audit.executed_overrides += 1
                                guard_audit.executed_switch_overrides += 1
                                executed_override_this_interval = True
                            elif stay_effective:
                                guard_audit.executed_stay_overrides += 1
                                if global_cooldown_active_this_interval:
                                    executed_stay_during_global_cooldown += 1
                            elif is_override:
                                guard_audit.rejected_executor += 1
                            if is_override and len(accepted_intervention_trace) < 256:
                                state = states[tls_id]
                                lane_speeds = tuple(
                                    float(value)
                                    for value in state.speed_by_lane.values()
                                )
                                lane_occupancies = tuple(
                                    float(value)
                                    for value in state.occ_by_lane.values()
                                )
                                accepted_intervention_trace.append(
                                    {
                                        "time_sec": float(
                                            sumo_api.simulation.getTime()
                                        ),
                                        "tls_id": str(tls_id),
                                        "prior_phase_state": str(
                                            proposal.prior_candidate.state
                                        ),
                                        "selected_phase_state": str(
                                            candidate.state
                                        ),
                                        "priority": float(proposal.priority),
                                        "selection_layer": str(
                                            proposal.selection_layer
                                        ),
                                        "target_veto": (
                                            dict(proposal.veto_diagnostics)
                                            if proposal.veto_diagnostics is not None
                                            else None
                                        ),
                                        "target_originator": (
                                            dict(proposal.originator_diagnostics)
                                            if proposal.originator_diagnostics is not None
                                            else None
                                        ),
                                        "request_accepted": bool(
                                            request_accepted
                                        ),
                                        "execution_effective": execution_effective,
                                        "override_kind": override_kind,
                                        "executor_mode_before": str(
                                            executor_before["mode"]
                                        ),
                                        "current_green_state_before": str(
                                            executor_before[
                                                "current_green_state"
                                            ]
                                        ),
                                        "global_cooldown_active_before": bool(
                                            global_cooldown_active_this_interval
                                        ),
                                        "stay_bypass_remaining_before": int(
                                            stay_bypass_remaining_before
                                        ),
                                        "green_elapsed_sec_before": float(
                                            executor_before[
                                                "green_elapsed_sec"
                                            ]
                                        ),
                                        "total_queue": float(
                                            sum(state.q_by_lane.values())
                                        ),
                                        "total_vehicles": float(
                                            sum(state.veh_by_lane.values())
                                        ),
                                        "mean_speed": float(
                                            np.mean(lane_speeds)
                                        )
                                        if lane_speeds
                                        else 0.0,
                                        "mean_occupancy": float(
                                            np.mean(lane_occupancies)
                                        )
                                        if lane_occupancies
                                        else 0.0,
                                    }
                                )
                        if (
                            execution_trust_region.enabled
                            and executed_override_this_interval
                        ):
                            residual_global_cooldown_remaining = max(
                                int(
                                    execution_trust_region.global_cooldown_intervals
                                ),
                                0,
                            )
                            residual_global_stay_bypass_remaining = (
                                max(
                                    int(
                                        execution_trust_region.max_stay_overrides_per_global_cooldown
                                    ),
                                    0,
                                )
                                if residual_global_cooldown_remaining > 0
                                and execution_trust_region.max_stay_overrides_per_global_cooldown
                                is not None
                                else 0
                            )
                        elif (
                            execution_trust_region.enabled
                            and executed_stay_during_global_cooldown > 0
                            and execution_trust_region.max_stay_overrides_per_global_cooldown
                            is not None
                        ):
                            residual_global_stay_bypass_remaining = max(
                                residual_global_stay_bypass_remaining
                                - executed_stay_during_global_cooldown,
                                0,
                            )
                        if residual_global_cooldown_remaining == 0:
                            residual_global_stay_bypass_remaining = 0
                else:
                    for tls_id, state in states.items():
                        executor = executors[tls_id]
                        if policy in {"max_pressure", "phase_pressure", "spillback_pressure"}:
                            candidate = _pressure_candidate_v3(
                                state,
                                executor,
                                policy,
                                control_interval_sec=control_interval_sec,
                            )
                        elif policy == "configured_pressure":
                            if pressure_spec_override is None:
                                raise ValueError("configured_pressure requires pressure_spec_override")
                            candidate = _pressure_candidate_v3(
                                state,
                                executor,
                                pressure_spec_override,
                                control_interval_sec=control_interval_sec,
                            )
                        elif policy == "selected_source_prior":
                            if models is None:
                                raise ValueError("selected_source_prior requires fitted source selector")
                            candidate = _pressure_candidate_v3(
                                state,
                                executor,
                                models.prior_spec,
                                control_interval_sec=control_interval_sec,
                            )
                        else:
                            raise ValueError(f"unknown V3 policy: {policy}")
                        executor.request(candidate.state)
            remaining = min(int(control_interval_sec), max(int(math.ceil(end_time - float(sumo_api.simulation.getTime()))), 0))
            for _ in range(remaining):
                before = float(sumo_api.simulation.getTime())
                sumo_api.simulationStep()
                now = float(sumo_api.simulation.getTime())
                elapsed = max(now - before, 0.0)
                for executor in executors.values():
                    executor.advance(elapsed)
                departed += int(sumo_api.simulation.getDepartedNumber())
                arrived += int(sumo_api.simulation.getArrivedNumber())
                loaded += int(sumo_api.simulation.getLoadedNumber())
                step_starting_teleports = int(
                    sumo_api.simulation.getStartingTeleportNumber()
                )
                step_ending_teleports = int(
                    sumo_api.simulation.getEndingTeleportNumber()
                )
                starting_teleports += step_starting_teleports
                ending_teleports += step_ending_teleports
                if step_starting_teleports or step_ending_teleports:
                    raise RuntimeError(
                        "teleport violates the fixed-population evaluation protocol at "
                        f"time={now:.3f}: starting={step_starting_teleports}, "
                        f"ending={step_ending_teleports}"
                    )
                step_collisions = tuple(sumo_api.simulation.getCollisions())
                collision_events += len(step_collisions)
                collision_event_steps += int(bool(step_collisions))
                collision_samples.extend(
                    _collision_debug_samples(
                        sumo_api=sumo_api,
                        collisions=step_collisions,
                        executors=executors,
                        time_sec=now,
                        max_samples=max(20 - len(collision_samples), 0),
                    )
                )
                collision_incident_keys.update(
                    _collision_incident_key(collision)
                    for collision in step_collisions
                )
                emergency_stops += int(
                    sumo_api.simulation.getEmergencyStoppingVehiclesNumber()
                )
                if now >= begin_time + float(warmup_sec):
                    queues.append(
                        float(sum(sumo_api.lane.getLastStepHaltingNumber(lane) for lane in lanes))
                    )
                    queue_proxies.append(float(sum(_lane_queue(sumo_api, lane) for lane in lanes)))
                    active_vehicle_counts.append(float(sumo_api.vehicle.getIDCount()))
                    pending_vehicle_counts.append(
                        float(len(sumo_api.simulation.getPendingVehicles()))
                    )
        controlled_lane_count = max(len(lanes), 1)
        pending_at_horizon = int(len(sumo_api.simulation.getPendingVehicles()))
        active_at_horizon = int(sumo_api.vehicle.getIDCount())
        demand_population_at_horizon = int(departed) + pending_at_horizon
        sample_seconds = len(queues)
        halted_vehicle_seconds = float(np.sum(queues))
        active_vehicle_seconds = float(np.sum(active_vehicle_counts))
        pending_vehicle_seconds = float(np.sum(pending_vehicle_counts))
        system_vehicle_seconds = active_vehicle_seconds + pending_vehicle_seconds
        metrics = {
            "ok": True,
            "policy": policy,
            "mean_queue": float(np.mean(queues)) if queues else float("nan"),
            "p90_queue": float(np.quantile(queues, 0.90)) if queues else float("nan"),
            "mean_queue_per_lane": float(np.mean(queues)) / controlled_lane_count
            if queues
            else float("nan"),
            "p90_queue_per_lane": float(np.quantile(queues, 0.90)) / controlled_lane_count
            if queues
            else float("nan"),
            "mean_queue_proxy": float(np.mean(queue_proxies))
            if queue_proxies
            else float("nan"),
            "mean_queue_proxy_per_lane": float(np.mean(queue_proxies)) / controlled_lane_count
            if queue_proxies
            else float("nan"),
            "controlled_lane_count": len(lanes),
            "fixed_horizon_sample_seconds": int(sample_seconds),
            "halted_vehicle_seconds": halted_vehicle_seconds,
            "halted_vehicle_hours": halted_vehicle_seconds / 3600.0,
            "active_vehicle_seconds": active_vehicle_seconds,
            "active_vehicle_hours": active_vehicle_seconds / 3600.0,
            "mean_active_vehicles": float(np.mean(active_vehicle_counts))
            if active_vehicle_counts
            else float("nan"),
            "mean_active_vehicles_per_controlled_lane": float(np.mean(active_vehicle_counts))
            / controlled_lane_count
            if active_vehicle_counts
            else float("nan"),
            "pending_vehicle_seconds": pending_vehicle_seconds,
            "pending_vehicle_hours": pending_vehicle_seconds / 3600.0,
            "system_vehicle_seconds": system_vehicle_seconds,
            "system_vehicle_hours": system_vehicle_seconds / 3600.0,
            "mean_system_vehicles": float(
                np.mean(np.asarray(active_vehicle_counts) + np.asarray(pending_vehicle_counts))
            )
            if active_vehicle_counts
            else float("nan"),
            "mean_system_vehicles_per_controlled_lane": float(
                np.mean(np.asarray(active_vehicle_counts) + np.asarray(pending_vehicle_counts))
            )
            / controlled_lane_count
            if active_vehicle_counts
            else float("nan"),
            "departed": int(departed),
            "arrived": int(arrived),
            "loaded": int(loaded),
            "demand_population_at_horizon": demand_population_at_horizon,
            "throughput_ratio": float(arrived / max(departed, 1)),
            "departure_service_ratio": float(
                departed / max(demand_population_at_horizon, 1)
            ),
            "completion_ratio": float(
                arrived / max(demand_population_at_horizon, 1)
            ),
            "starting_teleports": int(starting_teleports),
            "ending_teleports": int(ending_teleports),
            "collision_events": int(collision_events),
            "collision_event_steps": int(collision_event_steps),
            "collision_incidents": int(len(collision_incident_keys)),
            "collision_samples": collision_samples,
            "emergency_stops": int(emergency_stops),
            "teleport_rate": float(starting_teleports / max(departed, 1)),
            "collision_rate": float(collision_events / max(departed, 1)),
            "collision_incident_rate": float(
                len(collision_incident_keys) / max(departed, 1)
            ),
            "emergency_stop_rate": float(emergency_stops / max(departed, 1)),
            "pending_vehicles_at_horizon": pending_at_horizon,
            "active_vehicles_at_horizon": active_at_horizon,
            "phase_execution_audit": aggregate_phase_audits(executors) if executors else {},
            "guard_audit": guard_audit.to_dict() if learned_family is not None else {},
            "accepted_intervention_trace": (
                accepted_intervention_trace if learned_family is not None else []
            ),
            "residual_deployment": {
                "mode": residual_coordination_mode,
                "estimand": (
                    "partial_interference_first_action_then_prior_rollout"
                    if residual_coordination_mode == "sparse"
                    else "simultaneous_local_action_with_spatial_partial_interference"
                    if residual_coordination_mode == "spatial_only"
                    else "simultaneous_independent_local_action"
                ),
                "coordination": (
                    "temporal_cooldown_plus_greedy_priority_independent_set"
                    if residual_coordination_mode == "sparse"
                    else "greedy_priority_independent_set"
                    if residual_coordination_mode == "spatial_only"
                    else "none"
                ),
                "max_simultaneous_overrides": "conflict_graph_only"
                if max_simultaneous_overrides is None
                and residual_coordination_mode != "direct"
                else max_simultaneous_overrides,
                "cooldown_intervals": cooldown_intervals,
                "cooldown_sec": cooldown_intervals * int(control_interval_sec),
                "minimum_same_or_conflicting_intervention_gap_sec": (
                    _minimum_intervention_gap_sec(
                        cooldown_intervals, int(control_interval_sec)
                    )
                ),
                "cooldown_intervals_override": residual_cooldown_intervals_override,
                "global_cooldown_intervals": int(
                    execution_trust_region.global_cooldown_intervals
                ),
                "global_cooldown_sec": int(
                    execution_trust_region.global_cooldown_intervals
                )
                * int(control_interval_sec),
                "minimum_network_global_intervention_gap_sec": (
                    _minimum_intervention_gap_sec(
                        int(execution_trust_region.global_cooldown_intervals),
                        int(control_interval_sec),
                    )
                ),
                "global_cooldown_remaining_at_horizon": int(
                    residual_global_cooldown_remaining
                ),
                "stay_bypass_remaining_at_horizon": int(
                    residual_global_stay_bypass_remaining
                ),
                "global_cooldown_contract": (
                    "phase_transition_only_cooldown_with_bounded_stay_override_bypass"
                    if execution_trust_region.global_cooldown_intervals > 0
                    and execution_trust_region.allow_stay_during_global_cooldown
                    and execution_trust_region.max_stay_overrides_per_global_cooldown
                    is not None
                    else "phase_transition_only_cooldown_with_stay_override_bypass"
                    if execution_trust_region.global_cooldown_intervals > 0
                    and execution_trust_region.allow_stay_during_global_cooldown
                    else "one_executed_residual_then_network_prior_only"
                    if execution_trust_region.global_cooldown_intervals > 0
                    else "disabled"
                ),
                "execution_trust_region": asdict(execution_trust_region),
                "activation_sec": float(warmup_sec),
                "activation_protocol": "match_counterfactual_collection_warmup_v1",
                "conflict_graph_applied": residual_coordination_mode != "direct",
                "intervention_graph": intervention_graph.to_dict()
                if intervention_graph is not None
                else {},
            }
            if learned_family is not None
            else {},
            "protocol": (
                "safe_min_green_yellow_all_red_full_junction_occupancy_clearance_v4"
                if executors
                else "native_sumo_program"
            ),
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "queue_metric_definition": "sum_of_sumo_last_step_halting_number_on_controlled_lanes",
            "queue_proxy_definition": "halting_plus_0.35_vehicle_count_plus_0.025_occupancy",
            "fixed_horizon_metric_population": "active_inserted_plus_sumo_pending_insertion_vehicles_after_warmup",
            "demand_population_definition": "cumulative_departed_plus_sumo_pending_insertion_vehicles_at_horizon",
            "loaded_definition": "cumulative_sumo_getLoadedNumber_event_count_diagnostic_only",
            "tripinfo_metric_population": (
                "all_departed_vehicles_including_write_unfinished_records_at_horizon"
            ),
            "tripinfo_unfinished_semantics": (
                "arrival_minus_one_with_duration_waiting_and_time_loss_accrued_to_horizon"
            ),
            "prior_policy": models.prior_policy if models is not None else None,
            "prior_policy_spec": models.prior_spec.to_dict() if models is not None else None,
            "configured_pressure_spec": pressure_spec(pressure_spec_override).to_dict()
            if pressure_spec_override is not None
            else None,
        }
    except Exception as exc:
        metrics = {"ok": False, "policy": policy, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            sumo_api.close()
            closed = True
        except Exception:
            pass
    if closed:
        metrics.update(_parse_tripinfo_metrics(tripinfo_output))
    return metrics
