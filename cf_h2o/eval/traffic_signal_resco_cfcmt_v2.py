"""Second-generation RESCO benchmark for mechanism transfer.

This runner replaces the legacy scalar next-queue residual benchmark.  It uses
safe phase execution, records multiple local mechanisms, and keeps target
information budgets explicit.  The module is intentionally separate from the
legacy runner so historical results cannot be mistaken for V2 results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo, runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import (
    EXTENDED_SCENARIOS,
    _parse_tripinfo_metrics,
    _route_departure_stats,
    _scenario_summary,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    DEFAULT_ENV_ROOT,
    GREEN_CHARS,
    PhaseCandidate,
    TlsPhaseInfo,
    _controlled_lane_set,
    _ensure_env_root,
    _lane_queue,
    _scenario_sumocfg,
    _start_sumo,
    _tls_phase_infos,
)
from cf_h2o.traffic_signal.mechanism_world_model import (
    DenseResidualWorldModel,
    InvariantMechanismWorldModel,
    MechanismDataset,
    MechanismDefinition,
    MechanismFitConfig,
    PriorMechanismWorldModel,
)
from cf_h2o.traffic_signal.safe_phase_controller import (
    SafePhaseExecutor,
    aggregate_phase_audits,
    build_safe_phase_executors,
)


DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_resco_cfcmt_v2.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_resco_cfcmt_v2.md")

FEATURE_NAMES_V2 = (
    "total_q",
    "total_veh",
    "mean_speed",
    "mean_occ",
    "green_q",
    "red_q",
    "green_veh",
    "red_veh",
    "green_occ",
    "red_occ",
    "green_down_q",
    "green_down_occ",
    "green_link_ratio",
    "green_lane_ratio",
    "phase_duration_norm",
    "lane_count_norm",
    "candidate_count_norm",
    "time_sin",
    "time_cos",
    "service_pressure",
    "red_pressure",
    "switch_indicator",
    "clearance_fraction",
    "current_phase_overlap",
    "current_green_elapsed_norm",
    "green_q_mean",
    "green_q_max",
    "green_q_cv",
    "red_q_max",
    "red_q_cv",
    "green_speed_mean",
    "red_speed_mean",
    "green_occ_max",
    "red_occ_max",
    "green_down_q_max",
    "green_down_occ_max",
    "green_down_occ_std",
    "green_movement_ratio",
    "green_outgoing_ratio",
    "green_movement_down_q_mean",
    "green_movement_down_occ_mean",
    "green_movement_down_occ_max",
    "movement_service_pressure",
    "queue_concentration",
)

CONTEXT_NAMES_V2 = (
    "demand_per_tls",
    "demand_per_lane",
    "log_trip_count",
    "demand_horizon_norm",
    "tls_count_norm",
    "network_lane_count_norm",
    "candidate_count_mean_norm",
    "green_ratio_mean",
    "phase_duration_mean_norm",
)

OUTPUT_NAMES_V2 = (
    "next_total_queue",
    "next_green_queue",
    "next_red_queue",
    "next_downstream_occupancy",
    "next_mean_speed",
    "interval_cost",
)


TSC_MECHANISM_DEFINITIONS = (
    MechanismDefinition(
        name="queue_propagation",
        target_name="next_total_queue",
        parent_variants={
            "minimal": ("total_q", "total_veh", "green_lane_ratio", "switch_indicator", "clearance_fraction"),
            "physical": (
                "total_q",
                "total_veh",
                "green_q",
                "red_q",
                "mean_occ",
                "green_down_q",
                "green_lane_ratio",
                "service_pressure",
                "red_pressure",
                "switch_indicator",
                "clearance_fraction",
            ),
        },
        clip_min=0.0,
        clip_max=500.0,
    ),
    MechanismDefinition(
        name="served_movement",
        target_name="next_green_queue",
        parent_variants={
            "minimal": ("green_q", "green_veh", "green_lane_ratio", "clearance_fraction"),
            "physical": (
                "green_q",
                "green_veh",
                "green_occ",
                "green_down_q",
                "green_down_occ",
                "green_lane_ratio",
                "phase_duration_norm",
                "service_pressure",
                "switch_indicator",
                "clearance_fraction",
                "current_phase_overlap",
            ),
        },
        clip_min=0.0,
        clip_max=500.0,
    ),
    MechanismDefinition(
        name="red_accumulation",
        target_name="next_red_queue",
        parent_variants={
            "minimal": ("red_q", "red_veh", "green_lane_ratio"),
            "physical": (
                "red_q",
                "red_veh",
                "red_occ",
                "total_veh",
                "green_lane_ratio",
                "red_pressure",
                "switch_indicator",
                "clearance_fraction",
            ),
        },
        clip_min=0.0,
        clip_max=500.0,
    ),
    MechanismDefinition(
        name="spillback",
        target_name="next_downstream_occupancy",
        parent_variants={
            "minimal": ("green_down_occ", "green_down_q", "green_q", "green_veh"),
            "physical": (
                "green_down_occ",
                "green_down_q",
                "green_q",
                "green_veh",
                "mean_occ",
                "mean_speed",
                "green_link_ratio",
                "service_pressure",
                "clearance_fraction",
            ),
        },
        clip_min=0.0,
        clip_max=100.0,
    ),
    MechanismDefinition(
        name="mobility",
        target_name="next_mean_speed",
        parent_variants={
            "minimal": ("mean_speed", "total_q", "mean_occ"),
            "physical": (
                "mean_speed",
                "total_q",
                "mean_occ",
                "green_down_occ",
                "green_down_q",
                "service_pressure",
                "red_pressure",
                "switch_indicator",
                "clearance_fraction",
            ),
        },
        clip_min=0.0,
        clip_max=30.0,
    ),
    MechanismDefinition(
        name="control_cost",
        target_name="interval_cost",
        parent_variants={
            "minimal": ("total_q", "red_q", "green_down_q", "green_down_occ", "switch_indicator", "clearance_fraction"),
            "physical": (
                "total_q",
                "red_q",
                "mean_occ",
                "mean_speed",
                "green_down_occ",
                "green_down_q",
                "service_pressure",
                "red_pressure",
                "switch_indicator",
                "clearance_fraction",
                "current_phase_overlap",
            ),
        },
        clip_min=0.0,
        clip_max=1000.0,
    ),
)


@dataclass(frozen=True)
class LocalTransitionState:
    tls_id: str
    info: TlsPhaseInfo
    q_by_lane: Mapping[str, float]
    veh_by_lane: Mapping[str, float]
    speed_by_lane: Mapping[str, float]
    occ_by_lane: Mapping[str, float]
    down_q_by_lane: Mapping[str, float]
    down_occ_by_lane: Mapping[str, float]
    context: np.ndarray
    sim_time: float
    movement_down_q: Mapping[tuple[str, str], float] = field(default_factory=dict)
    movement_down_occ: Mapping[tuple[str, str], float] = field(default_factory=dict)
    graph_context: Any = None


@dataclass
class FittedTransferModelsV2:
    simulator: PriorMechanismWorldModel
    dense: DenseResidualWorldModel
    cfcmt: InvariantMechanismWorldModel
    diagnostics: dict[str, Any]
    guard_configs: dict[str, "GuardConfigV2"]


@dataclass(frozen=True)
class GuardConfigV2:
    prior_policy: str = "max_pressure"
    margin: float = 0.0
    risk_multiplier: float = 0.25
    min_context_trust: float = 0.10


@dataclass
class GuardAuditV2:
    decisions: int = 0
    prior_agreements: int = 0
    proposed_overrides: int = 0
    accepted_overrides: int = 0
    rejected_margin: int = 0
    rejected_context: int = 0
    predicted_advantage_sum: float = 0.0
    accepted_advantage_sum: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        decisions = max(self.decisions, 1)
        proposed = max(self.proposed_overrides, 1)
        return {
            "decisions": self.decisions,
            "prior_agreements": self.prior_agreements,
            "proposed_overrides": self.proposed_overrides,
            "accepted_overrides": self.accepted_overrides,
            "rejected_margin": self.rejected_margin,
            "rejected_context": self.rejected_context,
            "agreement_rate": self.prior_agreements / decisions,
            "proposal_rate": self.proposed_overrides / decisions,
            "acceptance_rate": self.accepted_overrides / proposed,
            "predicted_advantage_mean": self.predicted_advantage_sum / proposed,
            "accepted_advantage_mean": self.accepted_advantage_sum / max(self.accepted_overrides, 1),
        }


def _outgoing_by_lane(info: TlsPhaseInfo) -> dict[str, tuple[str, ...]]:
    outgoing: dict[str, list[str]] = {lane: [] for lane in info.incoming_lanes}
    for link_group in info.controlled_links:
        if not link_group or not link_group[0]:
            continue
        in_lane = str(link_group[0][0])
        out_lane = str(link_group[0][1])
        outgoing.setdefault(in_lane, [])
        if out_lane not in outgoing[in_lane]:
            outgoing[in_lane].append(out_lane)
    return {lane: tuple(values) for lane, values in outgoing.items()}


def _read_local_state(
    sumo_api: Any,
    tls_id: str,
    info: TlsPhaseInfo,
    context: np.ndarray,
) -> LocalTransitionState:
    outgoing = _outgoing_by_lane(info)
    q_by_lane: dict[str, float] = {}
    veh_by_lane: dict[str, float] = {}
    speed_by_lane: dict[str, float] = {}
    occ_by_lane: dict[str, float] = {}
    down_occ_by_lane: dict[str, float] = {}
    down_q_by_lane: dict[str, float] = {}
    movement_down_occ: dict[tuple[str, str], float] = {}
    movement_down_q: dict[tuple[str, str], float] = {}
    outgoing_cache: dict[str, tuple[float, float]] = {}
    for lane in info.incoming_lanes:
        q_by_lane[lane] = _lane_queue(sumo_api, lane)
        veh_by_lane[lane] = float(sumo_api.lane.getLastStepVehicleNumber(lane))
        speed_by_lane[lane] = float(sumo_api.lane.getLastStepMeanSpeed(lane))
        occ_by_lane[lane] = float(sumo_api.lane.getLastStepOccupancy(lane))
        downstream = []
        downstream_q = []
        for out_lane in outgoing.get(lane, ()):
            try:
                if out_lane not in outgoing_cache:
                    outgoing_cache[out_lane] = (
                        _lane_queue(sumo_api, out_lane),
                        float(sumo_api.lane.getLastStepOccupancy(out_lane)),
                    )
                out_q, out_occ = outgoing_cache[out_lane]
                downstream.append(out_occ)
                downstream_q.append(out_q)
                movement_down_q[(lane, out_lane)] = out_q
                movement_down_occ[(lane, out_lane)] = out_occ
            except Exception:
                continue
        down_occ_by_lane[lane] = float(np.mean(downstream)) if downstream else 0.0
        down_q_by_lane[lane] = float(np.mean(downstream_q)) if downstream_q else 0.0
    return LocalTransitionState(
        tls_id=str(tls_id),
        info=info,
        q_by_lane=q_by_lane,
        veh_by_lane=veh_by_lane,
        speed_by_lane=speed_by_lane,
        occ_by_lane=occ_by_lane,
        down_q_by_lane=down_q_by_lane,
        down_occ_by_lane=down_occ_by_lane,
        context=np.asarray(context, dtype=float),
        sim_time=float(sumo_api.simulation.getTime()),
        movement_down_q=movement_down_q,
        movement_down_occ=movement_down_occ,
    )


def _green_lane_ids(info: TlsPhaseInfo, candidate: PhaseCandidate) -> tuple[str, ...]:
    lanes = []
    for link_index, char in enumerate(candidate.state):
        if char not in GREEN_CHARS or link_index >= len(info.controlled_links):
            continue
        link_group = info.controlled_links[link_index]
        if not link_group or not link_group[0]:
            continue
        lane = str(link_group[0][0])
        if lane not in lanes:
            lanes.append(lane)
    return tuple(lanes)


def _phase_overlap(left: str, right: str) -> float:
    active_left = {idx for idx, char in enumerate(left) if char in GREEN_CHARS}
    active_right = {idx for idx, char in enumerate(right) if char in GREEN_CHARS}
    union = active_left | active_right
    return float(len(active_left & active_right) / max(len(union), 1))


def _candidate_aggregates(state: LocalTransitionState, candidate: PhaseCandidate) -> dict[str, float]:
    lanes = tuple(state.q_by_lane)
    green = set(_green_lane_ids(state.info, candidate))
    green_lanes = [lane for lane in lanes if lane in green]
    red_lanes = [lane for lane in lanes if lane not in green]
    green_q_values = np.asarray([state.q_by_lane[lane] for lane in green_lanes], dtype=float)
    red_q_values = np.asarray([state.q_by_lane[lane] for lane in red_lanes], dtype=float)
    green_speed_values = np.asarray([state.speed_by_lane[lane] for lane in green_lanes], dtype=float)
    red_speed_values = np.asarray([state.speed_by_lane[lane] for lane in red_lanes], dtype=float)
    green_occ_values = np.asarray([state.occ_by_lane[lane] for lane in green_lanes], dtype=float)
    red_occ_values = np.asarray([state.occ_by_lane[lane] for lane in red_lanes], dtype=float)
    green_down_q_values = np.asarray([state.down_q_by_lane[lane] for lane in green_lanes], dtype=float)
    green_down_occ_values = np.asarray(
        [state.down_occ_by_lane[lane] for lane in green_lanes],
        dtype=float,
    )
    green_movements: list[tuple[str, str]] = []
    for link_index, char in enumerate(candidate.state):
        if char not in GREEN_CHARS or link_index >= len(state.info.controlled_links):
            continue
        link_group = state.info.controlled_links[link_index]
        if not link_group or not link_group[0] or len(link_group[0]) < 2:
            continue
        movement = (str(link_group[0][0]), str(link_group[0][1]))
        if movement not in green_movements:
            green_movements.append(movement)
    movement_down_q = np.asarray(
        [
            state.movement_down_q.get(movement, state.down_q_by_lane.get(movement[0], 0.0))
            for movement in green_movements
        ],
        dtype=float,
    )
    movement_down_occ = np.asarray(
        [
            state.movement_down_occ.get(
                movement,
                state.down_occ_by_lane.get(movement[0], 0.0),
            )
            for movement in green_movements
        ],
        dtype=float,
    )
    movement_counts: dict[str, int] = {}
    for in_lane, _ in green_movements:
        movement_counts[in_lane] = movement_counts.get(in_lane, 0) + 1
    movement_pressure = 0.0
    for index, (in_lane, _) in enumerate(green_movements):
        queue_share = state.q_by_lane.get(in_lane, 0.0) / max(movement_counts[in_lane], 1)
        downstream_q = movement_down_q[index] if index < movement_down_q.size else 0.0
        downstream_occ = movement_down_occ[index] if index < movement_down_occ.size else 0.0
        movement_pressure += max(queue_share - downstream_q, 0.0) * max(
            1.0 - downstream_occ / 120.0,
            0.0,
        )
    aggregate = {
        "total_q": float(sum(state.q_by_lane.values())),
        "total_veh": float(sum(state.veh_by_lane.values())),
        "mean_speed": float(np.mean(list(state.speed_by_lane.values()))) if state.speed_by_lane else 0.0,
        "mean_occ": float(np.mean(list(state.occ_by_lane.values()))) if state.occ_by_lane else 0.0,
        "green_q": float(sum(state.q_by_lane[lane] for lane in green_lanes)),
        "red_q": float(sum(state.q_by_lane[lane] for lane in red_lanes)),
        "green_veh": float(sum(state.veh_by_lane[lane] for lane in green_lanes)),
        "red_veh": float(sum(state.veh_by_lane[lane] for lane in red_lanes)),
        "green_occ": float(np.mean([state.occ_by_lane[lane] for lane in green_lanes])) if green_lanes else 0.0,
        "red_occ": float(np.mean([state.occ_by_lane[lane] for lane in red_lanes])) if red_lanes else 0.0,
        "green_down_q": float(sum(state.down_q_by_lane[lane] for lane in green_lanes)),
        "green_down_occ": float(np.mean([state.down_occ_by_lane[lane] for lane in green_lanes])) if green_lanes else 0.0,
        "green_q_mean": _safe_mean(green_q_values),
        "green_q_max": _safe_max(green_q_values),
        "green_q_cv": _safe_cv(green_q_values),
        "red_q_max": _safe_max(red_q_values),
        "red_q_cv": _safe_cv(red_q_values),
        "green_speed_mean": _safe_mean(green_speed_values),
        "red_speed_mean": _safe_mean(red_speed_values),
        "green_occ_max": _safe_max(green_occ_values),
        "red_occ_max": _safe_max(red_occ_values),
        "green_down_q_max": _safe_max(green_down_q_values),
        "green_down_occ_max": _safe_max(green_down_occ_values),
        "green_down_occ_std": _safe_std(green_down_occ_values),
        "green_movement_down_q_mean": _safe_mean(movement_down_q),
        "green_movement_down_occ_mean": _safe_mean(movement_down_occ),
        "green_movement_down_occ_max": _safe_max(movement_down_occ),
        "movement_service_pressure": float(movement_pressure),
        "queue_concentration": float(
            max(_safe_max(green_q_values), _safe_max(red_q_values))
            / max(sum(state.q_by_lane.values()), 1.0)
        ),
    }
    lane_count = max(len(lanes), 1)
    link_count = max(len(state.info.controlled_links), 1)
    aggregate["green_link_ratio"] = float(candidate.green_count / link_count)
    aggregate["green_lane_ratio"] = float(len(green_lanes) / lane_count)
    aggregate["green_movement_ratio"] = float(len(green_movements) / link_count)
    aggregate["green_outgoing_ratio"] = float(
        len({out_lane for _, out_lane in green_movements}) / link_count
    )
    aggregate["service_pressure"] = max(
        aggregate["green_q"] - aggregate["green_down_q"],
        0.0,
    ) * max(0.0, 1.0 - aggregate["green_down_occ"] / 120.0)
    aggregate["red_pressure"] = aggregate["red_q"] * (1.0 - 0.5 * aggregate["green_lane_ratio"])
    return aggregate


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_cv(values: np.ndarray) -> float:
    return _safe_std(values) / max(abs(_safe_mean(values)), 1.0)


def candidate_features_v2(
    state: LocalTransitionState,
    candidate: PhaseCandidate,
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
) -> np.ndarray:
    aggregate = _candidate_aggregates(state, candidate)
    switching = candidate.state != executor.current_green_state or executor.is_switching
    if executor.is_switching:
        clearance_sec = max(float(executor.remaining_sec), 0.0)
    else:
        clearance_sec = executor.projected_clearance_sec(candidate.state)
    timing = executor.timings[executor.current_green_state]
    day_fraction = (state.sim_time % 86400.0) / 86400.0
    values = {
        **aggregate,
        "phase_duration_norm": float(candidate.duration / 90.0),
        "lane_count_norm": float(len(state.info.incoming_lanes) / 12.0),
        "candidate_count_norm": float(len(state.info.candidates) / 12.0),
        "time_sin": float(math.sin(math.tau * day_fraction)),
        "time_cos": float(math.cos(math.tau * day_fraction)),
        "switch_indicator": float(switching),
        "clearance_fraction": float(min(clearance_sec / max(float(control_interval_sec), 1.0), 1.5)),
        "current_phase_overlap": _phase_overlap(executor.current_green_state, candidate.state),
        "current_green_elapsed_norm": float(min(executor.green_elapsed_sec / max(timing.min_green_sec, 1.0), 4.0)),
    }
    return np.asarray([values[name] for name in FEATURE_NAMES_V2], dtype=float)


def uncalibrated_mechanism_priors(
    features: np.ndarray,
    context: np.ndarray,
    *,
    control_interval_sec: int,
) -> dict[str, np.ndarray]:
    """Generic non-target-calibrated mechanism equations."""

    x = np.asarray(features, dtype=float)
    z = np.asarray(context, dtype=float)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES_V2)}
    total_q = x[:, idx["total_q"]]
    total_veh = x[:, idx["total_veh"]]
    mean_speed = x[:, idx["mean_speed"]]
    mean_occ = x[:, idx["mean_occ"]]
    green_q = x[:, idx["green_q"]]
    red_q = x[:, idx["red_q"]]
    green_veh = x[:, idx["green_veh"]]
    green_down = x[:, idx["green_down_occ"]]
    green_down_q = x[:, idx["green_down_q"]]
    green_link_ratio = x[:, idx["green_link_ratio"]]
    green_lane_ratio = x[:, idx["green_lane_ratio"]]
    red_pressure = x[:, idx["red_pressure"]]
    clearance_fraction = np.clip(x[:, idx["clearance_fraction"]], 0.0, 1.0)

    arrival = z[:, 0] + 0.18 * z[:, 1] + 0.025 * total_veh + 0.008 * red_pressure
    available_green = np.clip(1.0 - clearance_fraction, 0.0, 1.0)
    downstream_factor = np.clip(1.0 - green_down / 135.0, 0.15, 1.0)
    capacity = (
        0.09
        * float(control_interval_sec)
        * (1.0 + 3.0 * green_lane_ratio + 1.4 * green_link_ratio)
        * available_green
        * downstream_factor
    )
    service = np.minimum(np.maximum(green_q - 0.25 * green_down_q, 0.0) + 0.3 * green_veh, capacity)
    next_green = np.clip(green_q + arrival * green_lane_ratio - service, 0.0, 500.0)
    next_red = np.clip(
        red_q + arrival * (1.0 - green_lane_ratio) - 0.015 * red_q * available_green,
        0.0,
        500.0,
    )
    next_total = np.clip(next_green + next_red, 0.0, 500.0)
    next_down = np.clip(
        green_down + 0.45 * service - 0.10 * mean_speed - 0.025 * np.maximum(100.0 - mean_occ, 0.0),
        0.0,
        100.0,
    )
    next_speed = np.clip(
        mean_speed
        + 0.12 * (13.0 - mean_speed)
        - 0.035 * total_q / np.maximum(x[:, idx["lane_count_norm"]] * 12.0, 1.0)
        - 0.012 * green_down,
        0.0,
        30.0,
    )
    interval_cost = np.clip(
        0.5 * (total_q + next_total)
        + 0.08 * red_pressure
        + 0.04 * next_down
        + 0.8 * float(control_interval_sec) * clearance_fraction,
        0.0,
        1000.0,
    )
    return {
        "next_total_queue": next_total,
        "next_green_queue": next_green,
        "next_red_queue": next_red,
        "next_downstream_occupancy": next_down,
        "next_mean_speed": next_speed,
        "interval_cost": interval_cost,
    }


def _candidate_for_state(info: TlsPhaseInfo, state: str) -> PhaseCandidate:
    for candidate in info.candidates:
        if candidate.state == state:
            return candidate
    return max(info.candidates, key=lambda candidate: _phase_overlap(candidate.state, state))


def _pressure_score(
    state: LocalTransitionState,
    candidate: PhaseCandidate,
    *,
    mode: str,
) -> float:
    aggregate = _candidate_aggregates(state, candidate)
    if mode == "phase_pressure":
        return float(aggregate["green_q"])
    if mode == "spillback_pressure":
        return float(aggregate["service_pressure"])
    return float(aggregate["green_q"] - aggregate["green_down_q"])


def _behavior_candidate(
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    rng: np.random.Generator,
) -> PhaseCandidate:
    feasible = [
        _candidate_for_state(state.info, candidate_state)
        for candidate_state in executor.feasible_states_now()
    ]
    draw = float(rng.uniform())
    if draw < 0.18:
        return feasible[int(rng.integers(0, len(feasible)))]
    mode = "phase_pressure" if draw < 0.58 else "spillback_pressure"
    return max(feasible, key=lambda candidate: _pressure_score(state, candidate, mode=mode))


def _lane_waiting_time(sumo_api: Any, lane: str) -> float:
    try:
        return float(sumo_api.lane.getWaitingTime(lane))
    except Exception:
        return 0.0


def collect_mechanism_transitions(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
) -> MechanismDataset:
    """Collect behavior transitions using the same safe action protocol as evaluation."""

    _start_sumo(sumo_api, sumocfg, seed)
    rows_features: list[np.ndarray] = []
    rows_context: list[np.ndarray] = []
    rows_targets: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES_V2}
    row_tls: list[str] = []
    row_times: list[float] = []
    executors: dict[str, SafePhaseExecutor] = {}
    try:
        infos = _tls_phase_infos(sumo_api)
        context = _scenario_summary(sumocfg=sumocfg, infos=infos, control_interval_sec=control_interval_sec)
        executors = build_safe_phase_executors(sumo_api, infos)
        rng = np.random.default_rng(seed + 1709)
        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + float(duration_sec)
        while float(sumo_api.simulation.getTime()) < end_time and int(sumo_api.simulation.getMinExpectedNumber()) > 0:
            states = {
                tls_id: _read_local_state(sumo_api, tls_id, info, context)
                for tls_id, info in infos.items()
            }
            chosen = {
                tls_id: _behavior_candidate(states[tls_id], executors[tls_id], rng)
                for tls_id in infos
            }
            features = {
                tls_id: candidate_features_v2(
                    states[tls_id],
                    chosen[tls_id],
                    executors[tls_id],
                    control_interval_sec=control_interval_sec,
                )
                for tls_id in infos
            }
            for tls_id, candidate in chosen.items():
                executors[tls_id].request(candidate.state)

            interval_costs: dict[str, list[float]] = {tls_id: [] for tls_id in infos}
            for _ in range(int(control_interval_sec)):
                before = float(sumo_api.simulation.getTime())
                sumo_api.simulationStep()
                now = float(sumo_api.simulation.getTime())
                elapsed = max(now - before, 0.0)
                for executor in executors.values():
                    executor.advance(elapsed)
                for tls_id, info in infos.items():
                    queue = sum(_lane_queue(sumo_api, lane) for lane in info.incoming_lanes)
                    waiting = sum(_lane_waiting_time(sumo_api, lane) for lane in info.incoming_lanes)
                    interval_costs[tls_id].append(float(queue + 0.015 * waiting))
                if now >= end_time:
                    break

            next_states = {
                tls_id: _read_local_state(sumo_api, tls_id, info, context)
                for tls_id, info in infos.items()
            }
            now = float(sumo_api.simulation.getTime())
            if now >= begin_time + float(warmup_sec):
                for tls_id, candidate in chosen.items():
                    next_aggregate = _candidate_aggregates(next_states[tls_id], candidate)
                    rows_features.append(features[tls_id])
                    rows_context.append(context)
                    rows_targets["next_total_queue"].append(next_aggregate["total_q"])
                    rows_targets["next_green_queue"].append(next_aggregate["green_q"])
                    rows_targets["next_red_queue"].append(next_aggregate["red_q"])
                    rows_targets["next_downstream_occupancy"].append(next_aggregate["green_down_occ"])
                    rows_targets["next_mean_speed"].append(next_aggregate["mean_speed"])
                    rows_targets["interval_cost"].append(float(np.mean(interval_costs[tls_id])))
                    row_tls.append(str(tls_id))
                    row_times.append(now)
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass

    feature_array = np.vstack(rows_features) if rows_features else np.zeros((0, len(FEATURE_NAMES_V2)), dtype=float)
    context_array = np.vstack(rows_context) if rows_context else np.zeros((0, len(CONTEXT_NAMES_V2)), dtype=float)
    priors = uncalibrated_mechanism_priors(
        feature_array,
        context_array,
        control_interval_sec=control_interval_sec,
    )
    return MechanismDataset(
        feature_names=FEATURE_NAMES_V2,
        features=feature_array,
        context_names=CONTEXT_NAMES_V2,
        context=context_array,
        priors=priors,
        targets={name: np.asarray(values, dtype=float) for name, values in rows_targets.items()},
        domains=np.asarray([scenario] * feature_array.shape[0]),
        metadata={
            "scenario": scenario,
            "sumocfg": str(sumocfg),
            "seed": int(seed),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "row_tls": row_tls,
            "row_times": row_times,
            "phase_execution_audit": aggregate_phase_audits(executors) if executors else {},
            "protocol": "safe_min_green_yellow_all_red_v2",
        },
    )


def merge_mechanism_datasets(datasets: Sequence[MechanismDataset]) -> MechanismDataset:
    if not datasets:
        raise ValueError("cannot merge an empty dataset sequence")
    first = datasets[0]
    if any(item.feature_names != first.feature_names or item.context_names != first.context_names for item in datasets):
        raise ValueError("all datasets must use the same feature and context schema")
    return MechanismDataset(
        feature_names=first.feature_names,
        features=np.vstack([item.features for item in datasets]),
        context_names=first.context_names,
        context=np.vstack([item.context for item in datasets]),
        priors={name: np.concatenate([item.priors[name] for item in datasets]) for name in OUTPUT_NAMES_V2},
        targets={name: np.concatenate([item.targets[name] for item in datasets]) for name in OUTPUT_NAMES_V2},
        domains=np.concatenate([item.domains for item in datasets]),
        metadata={"source_datasets": [dict(item.metadata) for item in datasets]},
    )


def fit_transfer_models_v2(
    dataset: MechanismDataset,
    *,
    cfcmt_config: MechanismFitConfig | None = None,
    calibrate_guards: bool = True,
) -> FittedTransferModelsV2:
    """Fit all residual families on exactly the same source transitions."""

    simulator = PriorMechanismWorldModel(TSC_MECHANISM_DEFINITIONS)
    dense = DenseResidualWorldModel(TSC_MECHANISM_DEFINITIONS)
    cfcmt = InvariantMechanismWorldModel(TSC_MECHANISM_DEFINITIONS, cfcmt_config)
    diagnostics = {
        "simulator": simulator.fit(dataset),
        "dense": dense.fit(dataset),
        "cfcmt": cfcmt.fit(dataset),
        "source_rows": dataset.size,
        "source_domains": sorted(str(item) for item in np.unique(dataset.domains)),
    }
    if calibrate_guards:
        guard_configs = {
            family: calibrate_guard_config_v2(
                dataset,
                family=family,
                cfcmt_config=cfcmt_config,
            )
            for family in ("simulator", "dense", "cfcmt")
        }
    else:
        guard_configs = {family: GuardConfigV2() for family in ("simulator", "dense", "cfcmt")}
    diagnostics["guard_calibration"] = {
        family: {
            "prior_policy": config.prior_policy,
            "margin": config.margin,
            "risk_multiplier": config.risk_multiplier,
            "min_context_trust": config.min_context_trust,
        }
        for family, config in guard_configs.items()
    }
    return FittedTransferModelsV2(
        simulator=simulator,
        dense=dense,
        cfcmt=cfcmt,
        diagnostics=diagnostics,
        guard_configs=guard_configs,
    )


def _fit_model_family(
    dataset: MechanismDataset,
    family: str,
    cfcmt_config: MechanismFitConfig | None,
) -> Any:
    if family == "simulator":
        model = PriorMechanismWorldModel(TSC_MECHANISM_DEFINITIONS)
    elif family == "dense":
        model = DenseResidualWorldModel(TSC_MECHANISM_DEFINITIONS)
    elif family == "cfcmt":
        model = InvariantMechanismWorldModel(TSC_MECHANISM_DEFINITIONS, cfcmt_config)
    else:
        raise ValueError(f"unknown model family: {family}")
    model.fit(dataset)
    return model


def _actual_transition_objective(dataset: MechanismDataset) -> np.ndarray:
    return (
        np.asarray(dataset.targets["interval_cost"], dtype=float)
        + 0.12 * np.asarray(dataset.targets["next_total_queue"], dtype=float)
        + 0.04 * np.asarray(dataset.targets["next_red_queue"], dtype=float)
        + 0.015 * np.asarray(dataset.targets["next_downstream_occupancy"], dtype=float)
        - 0.015 * np.asarray(dataset.targets["next_mean_speed"], dtype=float)
    )


def calibrate_guard_config_v2(
    dataset: MechanismDataset,
    *,
    family: str,
    cfcmt_config: MechanismFitConfig | None = None,
    coverage_quantile: float = 0.90,
    prior_policy: str = "max_pressure",
) -> GuardConfigV2:
    """Calibrate a model's guard radius with source-domain leave-one errors."""

    normalized_errors = []
    signed_errors = []
    trust_values = []
    domains = np.unique(dataset.domains)
    for heldout in domains:
        train_mask = dataset.domains != heldout
        valid_mask = ~train_mask
        if np.unique(dataset.domains[train_mask]).size < 2:
            continue
        model = _fit_model_family(dataset.subset(train_mask), family, cfcmt_config)
        valid = dataset.subset(valid_mask)
        predicted, uncertainty, trust = _candidate_objective(model.predict(valid))
        actual = _actual_transition_objective(valid)
        absolute_error = np.abs(actual - predicted)
        normalized_errors.extend((absolute_error / np.maximum(uncertainty, 1e-6)).tolist())
        signed_errors.extend((actual - predicted).tolist())
        trust_values.extend(trust.tolist())

    if normalized_errors:
        risk_multiplier = float(np.quantile(normalized_errors, coverage_quantile))
        risk_multiplier = float(np.clip(risk_multiplier, 1.0, 5.0))
        positive_bias = max(float(np.quantile(signed_errors, coverage_quantile)), 0.0)
        margin = 0.10 * positive_bias
        min_context_trust = float(np.clip(np.quantile(trust_values, 0.05), 0.02, 0.25))
    else:
        risk_multiplier = 1.0
        margin = 0.0
        min_context_trust = 0.05
    return GuardConfigV2(
        prior_policy=prior_policy,
        margin=margin,
        risk_multiplier=risk_multiplier,
        min_context_trust=min_context_trust,
    )


def _candidate_dataset(
    state: LocalTransitionState,
    candidates: Sequence[PhaseCandidate],
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
) -> MechanismDataset:
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
    context = np.repeat(state.context[None, :], len(candidates), axis=0)
    priors = uncalibrated_mechanism_priors(features, context, control_interval_sec=control_interval_sec)
    return MechanismDataset(
        feature_names=FEATURE_NAMES_V2,
        features=features,
        context_names=CONTEXT_NAMES_V2,
        context=context,
        priors=priors,
        targets={name: values.copy() for name, values in priors.items()},
        domains=np.asarray(["target"] * len(candidates)),
        metadata={"information_budget": "target_static_plus_online_state_no_labels"},
    )


def _candidate_objective(
    prediction: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return expected cost, uncertainty radius, and context trust."""

    cost = np.asarray(prediction["control_cost"]["mean"], dtype=float)
    cost = (
        cost
        + 0.12 * np.asarray(prediction["queue_propagation"]["mean"], dtype=float)
        + 0.04 * np.asarray(prediction["red_accumulation"]["mean"], dtype=float)
        + 0.015 * np.asarray(prediction["spillback"]["mean"], dtype=float)
        - 0.015 * np.asarray(prediction["mobility"]["mean"], dtype=float)
    )
    uncertainty = (
        np.asarray(prediction["control_cost"]["uncertainty"], dtype=float)
        + 0.12 * np.asarray(prediction["queue_propagation"]["uncertainty"], dtype=float)
        + 0.04 * np.asarray(prediction["red_accumulation"]["uncertainty"], dtype=float)
        + 0.015 * np.asarray(prediction["spillback"]["uncertainty"], dtype=float)
        + 0.015 * np.asarray(prediction["mobility"]["uncertainty"], dtype=float)
    )
    trust = np.minimum.reduce(
        [
            np.asarray(prediction[name]["context_trust"], dtype=float)
            for name in ("control_cost", "queue_propagation", "red_accumulation", "spillback", "mobility")
        ]
    )
    return cost, np.maximum(uncertainty, 0.0), np.clip(trust, 0.0, 1.0)


def _model_candidate_scores(
    model: Any,
    state: LocalTransitionState,
    candidates: Sequence[PhaseCandidate],
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = _candidate_dataset(
        state,
        candidates,
        executor,
        control_interval_sec=control_interval_sec,
    )
    return _candidate_objective(model.predict(dataset))


def _feasible_candidates(state: LocalTransitionState, executor: SafePhaseExecutor) -> tuple[PhaseCandidate, ...]:
    return tuple(_candidate_for_state(state.info, phase_state) for phase_state in executor.feasible_states_now())


def _rule_candidate(
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    policy: str,
) -> PhaseCandidate:
    candidates = _feasible_candidates(state, executor)
    if policy not in {"max_pressure", "phase_pressure", "spillback_pressure"}:
        raise ValueError(f"unknown pressure policy: {policy}")
    return max(candidates, key=lambda candidate: _pressure_score(state, candidate, mode=policy))


def _model_candidate(
    model: Any,
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
) -> PhaseCandidate:
    candidates = _feasible_candidates(state, executor)
    score, _, _ = _model_candidate_scores(
        model,
        state,
        candidates,
        executor,
        control_interval_sec=control_interval_sec,
    )
    return candidates[int(np.argmin(score))]


def _guarded_candidate(
    model: Any,
    state: LocalTransitionState,
    executor: SafePhaseExecutor,
    *,
    control_interval_sec: int,
    guard: GuardConfigV2,
    audit: GuardAuditV2,
) -> PhaseCandidate:
    candidates = _feasible_candidates(state, executor)
    score, uncertainty, trust = _model_candidate_scores(
        model,
        state,
        candidates,
        executor,
        control_interval_sec=control_interval_sec,
    )
    learned_index = int(np.argmin(score))
    prior_candidate = _rule_candidate(state, executor, guard.prior_policy)
    prior_index = next(idx for idx, candidate in enumerate(candidates) if candidate.state == prior_candidate.state)
    audit.decisions += 1
    if learned_index == prior_index:
        audit.prior_agreements += 1
        return prior_candidate

    audit.proposed_overrides += 1
    advantage = float(score[prior_index] - score[learned_index])
    audit.predicted_advantage_sum += advantage
    if float(trust[learned_index]) < float(guard.min_context_trust):
        audit.rejected_context += 1
        return prior_candidate
    # If both action-cost errors are covered by their source-calibrated radii,
    # the pairwise ranking error is bounded by the sum of the two radii.
    risk_radius = float(guard.risk_multiplier) * (
        float(uncertainty[learned_index]) + float(uncertainty[prior_index])
    )
    if advantage <= float(guard.margin) + risk_radius:
        audit.rejected_margin += 1
        return prior_candidate
    audit.accepted_overrides += 1
    audit.accepted_advantage_sum += advantage
    return candidates[learned_index]


def evaluate_policy_v2(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    policy: str,
    models: FittedTransferModelsV2 | None,
    guard: GuardConfigV2,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    tripinfo_output: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one policy; every non-native policy uses safe phase execution."""

    _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=tripinfo_output)
    queues: list[float] = []
    departed = 0
    arrived = 0
    executors: dict[str, SafePhaseExecutor] = {}
    guard_audit = GuardAuditV2()
    closed = False
    try:
        infos = _tls_phase_infos(sumo_api)
        lanes = _controlled_lane_set(infos)
        context = _scenario_summary(sumocfg=sumocfg, infos=infos, control_interval_sec=control_interval_sec)
        if policy != "fixed_program":
            executors = build_safe_phase_executors(sumo_api, infos)
        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + float(duration_sec)
        while float(sumo_api.simulation.getTime()) < end_time and int(sumo_api.simulation.getMinExpectedNumber()) > 0:
            if policy != "fixed_program":
                states = {
                    tls_id: _read_local_state(sumo_api, tls_id, info, context)
                    for tls_id, info in infos.items()
                }
                for tls_id, state in states.items():
                    executor = executors[tls_id]
                    if policy in {"max_pressure", "phase_pressure", "spillback_pressure"}:
                        candidate = _rule_candidate(state, executor, policy)
                    elif policy in {"simulator_mpc", "dense_mpc", "cfcmt_mpc"}:
                        if models is None:
                            raise ValueError(f"{policy} requires fitted models")
                        model = {
                            "simulator_mpc": models.simulator,
                            "dense_mpc": models.dense,
                            "cfcmt_mpc": models.cfcmt,
                        }[policy]
                        candidate = _model_candidate(
                            model,
                            state,
                            executor,
                            control_interval_sec=control_interval_sec,
                        )
                    elif policy in {"simulator_guard", "dense_guard", "cfcmt_guard"}:
                        if models is None:
                            raise ValueError(f"{policy} requires fitted models")
                        model = {
                            "simulator_guard": models.simulator,
                            "dense_guard": models.dense,
                            "cfcmt_guard": models.cfcmt,
                        }[policy]
                        candidate = _guarded_candidate(
                            model,
                            state,
                            executor,
                            control_interval_sec=control_interval_sec,
                            guard=guard,
                            audit=guard_audit,
                        )
                    else:
                        raise ValueError(f"unknown V2 policy: {policy}")
                    executor.request(candidate.state)

            for _ in range(int(control_interval_sec)):
                before = float(sumo_api.simulation.getTime())
                sumo_api.simulationStep()
                now = float(sumo_api.simulation.getTime())
                elapsed = max(now - before, 0.0)
                for executor in executors.values():
                    executor.advance(elapsed)
                departed += int(sumo_api.simulation.getDepartedNumber())
                arrived += int(sumo_api.simulation.getArrivedNumber())
                if now >= begin_time + float(warmup_sec):
                    queues.append(float(sum(_lane_queue(sumo_api, lane) for lane in lanes)))
                if now >= end_time:
                    break
        metrics = {
            "ok": True,
            "policy": policy,
            "mean_queue": float(np.mean(queues)) if queues else float("nan"),
            "p90_queue": float(np.quantile(queues, 0.90)) if queues else float("nan"),
            "departed": int(departed),
            "arrived": int(arrived),
            "throughput_ratio": float(arrived / max(departed, 1)),
            "phase_execution_audit": aggregate_phase_audits(executors) if executors else {},
            "guard_audit": guard_audit.to_dict() if policy.endswith("_guard") else {},
            "protocol": "safe_min_green_yellow_all_red_v2" if executors else "native_sumo_program",
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


def transition_diagnostics(dataset: MechanismDataset) -> dict[str, Any]:
    outputs = {}
    for name in OUTPUT_NAMES_V2:
        target = np.asarray(dataset.targets[name], dtype=float)
        prior = np.asarray(dataset.priors[name], dtype=float)
        outputs[name] = {
            "target_mean": float(np.mean(target)) if target.size else float("nan"),
            "target_std": float(np.std(target)) if target.size else float("nan"),
            "prior_mae": float(np.mean(np.abs(target - prior))) if target.size else float("nan"),
            "prior_rmse": float(np.sqrt(np.mean((target - prior) ** 2))) if target.size else float("nan"),
        }
    return {
        "rows": dataset.size,
        "domains": sorted(str(item) for item in np.unique(dataset.domains)),
        "outputs": outputs,
        "phase_execution_audit": dict(dataset.metadata.get("phase_execution_audit", {})),
    }


def _runtime_metadata() -> dict[str, Any]:
    return runtime_metadata()


def run_collection_smoke(
    *,
    env_root: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
) -> dict[str, Any]:
    sumo_api = _load_libsumo()
    sumocfg = _scenario_sumocfg(env_root, scenario)
    dataset = collect_mechanism_transitions(
        sumo_api=sumo_api,
        sumocfg=sumocfg,
        scenario=scenario,
        duration_sec=duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        seed=seed,
    )
    return {
        "experiment": "traffic_signal_resco_cfcmt_v2_collection_smoke",
        "runtime": _runtime_metadata(),
        "setting": {
            "scenario": scenario,
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
            "information_budget": "source_transition_labels",
        },
        "diagnostics": transition_diagnostics(dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scenario", default="cologne1", choices=EXTENDED_SCENARIOS)
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run_collection_smoke(
        env_root=args.env_root,
        scenario=args.scenario,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
