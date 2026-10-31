"""RESCO phase-action CFCMT/H2O residual-transfer benchmark.

This stage attaches transfer models to real RESCO SUMO scenarios.  Unlike the
synthetic Phase-2 grid, RESCO traffic lights have different numbers of links
and green phases, so actions are represented as choosing one existing green
phase from each SUMO ``tlLogic`` program.

The evaluation is leave-one-scenario-out:

* source scenarios provide passive local phase transitions;
* zero-shot models use no target next-state labels;
* the few-shot variant uses a short passive target slice to fit a scalar
  residual bias or to select the most reliable source-trained predictor;
* closed-loop control uses one-step phase MPC over each TLS' real phase set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    DEFAULT_ENV_ROOT,
    DEFAULT_REPO_URL,
    DEFAULT_SCENARIOS,
    GREEN_CHARS,
    YELLOW_CHARS,
    PhaseCandidate,
    TlsPhaseInfo,
    _choose_candidate,
    _controlled_lane_set,
    _ensure_env_root,
    _lane_queue,
    _net_and_route_from_sumocfg,
    _scenario_sumocfg,
    _start_sumo,
    _tls_phase_infos,
)
from cf_h2o.eval.traffic_signal_transfer_feasibility import RidgeModel, _format_table, _ridge_fit
from cf_h2o.traffic_signal.sumo_static_inputs import (
    flow_expected_count,
    iter_static_demand_records,
    parse_sumo_time_seconds,
    static_demand_input_paths,
    triggered_vehicle_departure_times,
)


DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_resco_cfcmt_benchmark.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_resco_cfcmt_benchmark.md")
DEFAULT_ACTUATED_ROOT = Path("cf_h2o/results/traffic_signal_resco_actuated")
CORE_SCENARIOS = DEFAULT_SCENARIOS
EXTENDED_SCENARIOS = (
    "grid4x4",
    "arterial4x4",
    "cologne1",
    "cologne3",
    "cologne8",
    "ingolstadt1",
    "ingolstadt7",
    "ingolstadt21",
)
SALT_LAKE_SCENARIOS = ("saltlake2_400sX200w", "saltlake2_stateXuniversity")
SCENARIO_SETS = {
    "core": CORE_SCENARIOS,
    "extended": EXTENDED_SCENARIOS,
    "saltlake": SALT_LAKE_SCENARIOS,
    "all": (*EXTENDED_SCENARIOS, *SALT_LAKE_SCENARIOS),
}

DEFAULT_POLICIES = (
    "fixed_program",
    "actuated_program",
    "max_pressure",
    "phase_pressure",
    "phase_spillback_pressure",
    "sim_generic_phase_mpc",
    "sim_source_avg_phase_mpc",
    "sim_target_static_phase_mpc",
    "h2oplus_dense_phase_mpc",
    "cfcmt_phase_global_mpc",
    "cfcmt_phase_trust_mpc",
    "cfcmt_phase_pressure_guard_mpc",
    "cfcmt_phase_fewshot_bias_mpc",
    "cfcmt_phase_fewshot_selector_mpc",
    "cfcmt_phase_passive_policy_selector_mpc",
)

GENERIC_SCENARIO_SUMMARY = np.asarray([0.65, 0.18, 0.60, 0.25, 0.30, 0.18, 0.35, 0.42, 0.0], dtype=float)
CFCMT_PRESSURE_GUARD_MARGIN = 0.35

FEATURE_NAMES = (
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
    "green_down_occ",
    "green_link_ratio",
    "green_lane_ratio",
    "phase_duration_norm",
    "candidate_rank",
    "lane_count_norm",
    "candidate_count_norm",
    "time_norm",
    "service_pressure",
    "red_pressure",
)

F_TOTAL_Q = FEATURE_NAMES.index("total_q")
F_TOTAL_VEH = FEATURE_NAMES.index("total_veh")
F_GREEN_Q = FEATURE_NAMES.index("green_q")
F_RED_Q = FEATURE_NAMES.index("red_q")
F_GREEN_VEH = FEATURE_NAMES.index("green_veh")
F_GREEN_DOWN_OCC = FEATURE_NAMES.index("green_down_occ")
F_GREEN_LINK_RATIO = FEATURE_NAMES.index("green_link_ratio")
F_GREEN_LANE_RATIO = FEATURE_NAMES.index("green_lane_ratio")
F_TIME_NORM = FEATURE_NAMES.index("time_norm")
F_SERVICE_PRESSURE = FEATURE_NAMES.index("service_pressure")
F_RED_PRESSURE = FEATURE_NAMES.index("red_pressure")


@dataclass(frozen=True)
class PhaseLocalState:
    tls_id: str
    info: TlsPhaseInfo
    time_norm: float
    q_by_lane: dict[str, float]
    veh_by_lane: dict[str, float]
    speed_by_lane: dict[str, float]
    occ_by_lane: dict[str, float]
    down_occ_by_lane: dict[str, float]
    summary: np.ndarray


@dataclass(frozen=True)
class PhaseTransitionSet:
    scenario: str
    x: np.ndarray
    summary: np.ndarray
    next_queue: np.ndarray

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True)
class PhaseSnapshotSet:
    scenario: str
    candidate_x: tuple[np.ndarray, ...]
    candidate_summary: tuple[np.ndarray, ...]

    @property
    def size(self) -> int:
        return int(len(self.candidate_x))


@dataclass(frozen=True)
class PhaseFittedModels:
    dense: RidgeModel
    cfcmt: RidgeModel
    source_summaries: np.ndarray
    source_mean_summary: np.ndarray


def _safe_close(sumo_api: Any) -> None:
    try:
        sumo_api.close()
    except Exception:
        pass


def _mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _p90_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, 0.90)) if finite else float("nan")


def _vehicle_departure_time(sumo_api: Any, vehicle_id: str, fallback: float) -> float:
    try:
        value = float(sumo_api.vehicle.getDeparture(vehicle_id))
        return value if math.isfinite(value) else float(fallback)
    except Exception:
        return float(fallback)


def _resolve_cfg_file(sumocfg: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else sumocfg.parent / path


def _phase_bounds_for_actuated(phase: ET.Element) -> tuple[float, float]:
    duration = max(float(phase.attrib.get("duration", "1")), 1.0)
    state = phase.attrib.get("state", "")
    if any(char in YELLOW_CHARS for char in state) and not any(char in GREEN_CHARS for char in state):
        return duration, duration
    if "minDur" in phase.attrib:
        min_dur = max(float(phase.attrib["minDur"]), 1.0)
    else:
        min_dur = max(5.0, min(duration * 0.50, 12.0))
    if "maxDur" in phase.attrib:
        max_dur = max(float(phase.attrib["maxDur"]), min_dur)
    else:
        max_dur = max(duration * 2.50, min_dur + 1.0, 30.0)
    return min_dur, min(max_dur, 120.0)


def _convert_net_to_actuated(src_net: Path, dst_net: Path) -> int:
    tree = ET.parse(src_net)
    root = tree.getroot()
    changed = 0
    for logic in root.findall(".//tlLogic"):
        if logic.attrib.get("type") != "actuated":
            changed += 1
        logic.set("type", "actuated")
        for phase in logic.findall("phase"):
            min_dur, max_dur = _phase_bounds_for_actuated(phase)
            phase.set("minDur", f"{min_dur:g}")
            phase.set("maxDur", f"{max_dur:g}")
    dst_net.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_net, encoding="utf-8", xml_declaration=True)
    return changed


def _build_actuated_sumocfg(sumocfg: Path, scenario: str, out_root: Path = DEFAULT_ACTUATED_ROOT) -> Path:
    net_files, route_files, _ = _net_and_route_from_sumocfg(sumocfg)
    if len(net_files) != 1:
        raise ValueError(f"actuated_program expects one net-file in {sumocfg}, got {net_files}")
    scenario_root = out_root / scenario
    actuated_net = scenario_root / f"{scenario}.actuated.net.xml"
    actuated_cfg = scenario_root / f"{scenario}.actuated.sumocfg"
    _convert_net_to_actuated(_resolve_cfg_file(sumocfg, net_files[0]), actuated_net)

    tree = ET.parse(sumocfg)
    root = tree.getroot()
    for item in root.findall(".//net-file"):
        item.set("value", str(actuated_net.resolve()))
    if route_files:
        routes = [str(_resolve_cfg_file(sumocfg, route_file).resolve()) for route_file in route_files]
        for item in root.findall(".//route-files"):
            item.set("value", ",".join(routes))
    scenario_root.mkdir(parents=True, exist_ok=True)
    tree.write(actuated_cfg, encoding="utf-8", xml_declaration=True)
    return actuated_cfg


def _empty_tripinfo_metrics() -> dict[str, float]:
    return {
        "mean_tripinfo_duration": float("nan"),
        "p90_tripinfo_duration": float("nan"),
        "mean_tripinfo_waiting_time": float("nan"),
        "p90_tripinfo_waiting_time": float("nan"),
        "mean_tripinfo_time_loss": float("nan"),
        "p90_tripinfo_time_loss": float("nan"),
        "mean_tripinfo_depart_delay": float("nan"),
        "p90_tripinfo_depart_delay": float("nan"),
        "tripinfo_count": float("nan"),
        "mean_completed_tripinfo_duration": float("nan"),
        "p90_completed_tripinfo_duration": float("nan"),
        "mean_completed_tripinfo_waiting_time": float("nan"),
        "p90_completed_tripinfo_waiting_time": float("nan"),
        "mean_completed_tripinfo_time_loss": float("nan"),
        "p90_completed_tripinfo_time_loss": float("nan"),
        "tripinfo_completed_count": float("nan"),
        "tripinfo_unfinished_count": float("nan"),
        "tripinfo_unfinished_fraction": float("nan"),
    }


def _parse_tripinfo_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return _empty_tripinfo_metrics()
    duration: list[float] = []
    waiting: list[float] = []
    time_loss: list[float] = []
    depart_delay: list[float] = []
    completed_duration: list[float] = []
    completed_waiting: list[float] = []
    completed_time_loss: list[float] = []
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "tripinfo":
            elem.clear()
            continue
        attrs = elem.attrib
        if "duration" in attrs:
            duration.append(float(attrs["duration"]))
        if "waitingTime" in attrs:
            waiting.append(float(attrs["waitingTime"]))
        if "timeLoss" in attrs:
            time_loss.append(float(attrs["timeLoss"]))
        if "departDelay" in attrs:
            depart_delay.append(float(attrs["departDelay"]))
        if float(attrs.get("arrival", "-1")) >= 0.0:
            if "duration" in attrs:
                completed_duration.append(float(attrs["duration"]))
            if "waitingTime" in attrs:
                completed_waiting.append(float(attrs["waitingTime"]))
            if "timeLoss" in attrs:
                completed_time_loss.append(float(attrs["timeLoss"]))
        elem.clear()
    unfinished_count = len(duration) - len(completed_duration)
    return {
        "mean_tripinfo_duration": _mean_or_nan(duration),
        "p90_tripinfo_duration": _p90_or_nan(duration),
        "mean_tripinfo_waiting_time": _mean_or_nan(waiting),
        "p90_tripinfo_waiting_time": _p90_or_nan(waiting),
        "mean_tripinfo_time_loss": _mean_or_nan(time_loss),
        "p90_tripinfo_time_loss": _p90_or_nan(time_loss),
        "mean_tripinfo_depart_delay": _mean_or_nan(depart_delay),
        "p90_tripinfo_depart_delay": _p90_or_nan(depart_delay),
        "tripinfo_count": float(len(duration)),
        "mean_completed_tripinfo_duration": _mean_or_nan(completed_duration),
        "p90_completed_tripinfo_duration": _p90_or_nan(completed_duration),
        "mean_completed_tripinfo_waiting_time": _mean_or_nan(completed_waiting),
        "p90_completed_tripinfo_waiting_time": _p90_or_nan(completed_waiting),
        "mean_completed_tripinfo_time_loss": _mean_or_nan(completed_time_loss),
        "p90_completed_tripinfo_time_loss": _p90_or_nan(completed_time_loss),
        "tripinfo_completed_count": float(len(completed_duration)),
        "tripinfo_unfinished_count": float(unfinished_count),
        "tripinfo_unfinished_fraction": float(
            unfinished_count / max(len(duration), 1)
        ),
    }


def _route_departure_stats(sumocfg: Path) -> dict[str, float]:
    _, _, cfg_window = _net_and_route_from_sumocfg(sumocfg)
    begin_cfg, end_cfg = cfg_window
    count = 0.0
    min_depart = float("inf")
    max_depart = -float("inf")
    demand_paths = static_demand_input_paths(sumocfg)
    triggered_departures = triggered_vehicle_departure_times(demand_paths)
    for record in iter_static_demand_records(demand_paths):
        if record.kind == "flow":
            begin = parse_sumo_time_seconds(
                record.attributes.get(
                    "begin", begin_cfg if begin_cfg is not None else 0.0
                )
            )
            end = parse_sumo_time_seconds(record.attributes.get("end", begin))
            count += flow_expected_count(record.attributes)
            min_depart = min(min_depart, begin)
            max_depart = max(max_depart, end)
        else:
            depart_value = str(
                record.attributes.get(
                    "depart", begin_cfg if begin_cfg is not None else 0.0
                )
            )
            if depart_value == "triggered":
                vehicle_id = str(record.attributes.get("id", ""))
                if vehicle_id not in triggered_departures:
                    raise ValueError(
                        "triggered SUMO vehicle has no person-plan reference: "
                        f"{vehicle_id}"
                    )
                depart = float(triggered_departures[vehicle_id])
            else:
                depart = parse_sumo_time_seconds(depart_value)
            count += 1.0
            min_depart = min(min_depart, depart)
            max_depart = max(max_depart, depart)
    if not math.isfinite(min_depart):
        min_depart = begin_cfg if begin_cfg is not None else 0.0
    if not math.isfinite(max_depart):
        max_depart = end_cfg if end_cfg is not None else min_depart + 1.0
    cfg_horizon = (end_cfg - begin_cfg) if begin_cfg is not None and end_cfg is not None else 0.0
    horizon = max(float(max_depart - min_depart), float(cfg_horizon), 1.0)
    return {"trip_count": float(count), "horizon_sec": float(horizon)}


def _scenario_summary(
    *,
    sumocfg: Path,
    infos: dict[str, TlsPhaseInfo],
    control_interval_sec: int,
) -> np.ndarray:
    route_stats = _route_departure_stats(sumocfg)
    tls_count = max(len(infos), 1)
    lane_count = max(len(_controlled_lane_set(infos)), 1)
    link_counts = [len(info.controlled_links) for info in infos.values()]
    candidate_counts = [len(info.candidates) for info in infos.values()]
    green_ratios: list[float] = []
    duration_norms: list[float] = []
    for info in infos.values():
        link_count = max(len(info.controlled_links), 1)
        for candidate in info.candidates:
            green_ratios.append(candidate.green_count / link_count)
            duration_norms.append(candidate.duration / 90.0)
    demand_interval = route_stats["trip_count"] / route_stats["horizon_sec"] * float(control_interval_sec)
    return np.asarray(
        [
            demand_interval / tls_count,
            demand_interval / lane_count,
            math.log1p(route_stats["trip_count"]) / 12.0,
            min(route_stats["horizon_sec"] / 7200.0, 3.0),
            tls_count / 20.0,
            lane_count / 160.0,
            float(np.mean(candidate_counts)) / 8.0 if candidate_counts else 0.0,
            float(np.mean(green_ratios)) if green_ratios else 0.0,
            float(np.mean(duration_norms)) if duration_norms else 0.0,
        ],
        dtype=float,
    )


def _outgoing_by_lane(info: TlsPhaseInfo) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {lane: [] for lane in info.incoming_lanes}
    for link_group in info.controlled_links:
        if not link_group or not link_group[0]:
            continue
        in_lane = str(link_group[0][0])
        out_lane = str(link_group[0][1])
        out.setdefault(in_lane, [])
        if out_lane not in out[in_lane]:
            out[in_lane].append(out_lane)
    return out


def _read_phase_state(
    *,
    sumo_api: Any,
    tls_id: str,
    info: TlsPhaseInfo,
    summary: np.ndarray,
    duration_sec: float,
) -> PhaseLocalState:
    outgoing = _outgoing_by_lane(info)
    q_by_lane: dict[str, float] = {}
    veh_by_lane: dict[str, float] = {}
    speed_by_lane: dict[str, float] = {}
    occ_by_lane: dict[str, float] = {}
    down_occ_by_lane: dict[str, float] = {}
    for lane in info.incoming_lanes:
        q_by_lane[lane] = _lane_queue(sumo_api, lane)
        veh_by_lane[lane] = float(sumo_api.lane.getLastStepVehicleNumber(lane))
        speed_by_lane[lane] = float(sumo_api.lane.getLastStepMeanSpeed(lane))
        occ_by_lane[lane] = float(sumo_api.lane.getLastStepOccupancy(lane))
        down_values: list[float] = []
        for out_lane in outgoing.get(lane, []):
            try:
                down_values.append(float(sumo_api.lane.getLastStepOccupancy(out_lane)))
            except Exception:
                pass
        down_occ_by_lane[lane] = float(np.mean(down_values)) if down_values else 0.0
    now = float(sumo_api.simulation.getTime())
    return PhaseLocalState(
        tls_id=tls_id,
        info=info,
        time_norm=now / max(float(duration_sec), 1.0),
        q_by_lane=q_by_lane,
        veh_by_lane=veh_by_lane,
        speed_by_lane=speed_by_lane,
        occ_by_lane=occ_by_lane,
        down_occ_by_lane=down_occ_by_lane,
        summary=summary,
    )


def _green_lane_ids(info: TlsPhaseInfo, candidate: PhaseCandidate) -> tuple[str, ...]:
    lanes: list[str] = []
    for link_idx, char in enumerate(candidate.state):
        if char not in GREEN_CHARS or link_idx >= len(info.controlled_links):
            continue
        link_group = info.controlled_links[link_idx]
        if not link_group or not link_group[0]:
            continue
        lane = str(link_group[0][0])
        if lane not in lanes:
            lanes.append(lane)
    return tuple(lanes)


def _candidate_features(state: PhaseLocalState, candidate: PhaseCandidate, candidate_idx: int) -> np.ndarray:
    lane_ids = tuple(state.q_by_lane)
    green_lanes = set(_green_lane_ids(state.info, candidate))
    red_lanes = [lane for lane in lane_ids if lane not in green_lanes]
    green_lane_list = [lane for lane in lane_ids if lane in green_lanes]

    total_q = float(sum(state.q_by_lane.values()))
    total_veh = float(sum(state.veh_by_lane.values()))
    mean_speed = float(np.mean(list(state.speed_by_lane.values()))) if state.speed_by_lane else 0.0
    mean_occ = float(np.mean(list(state.occ_by_lane.values()))) if state.occ_by_lane else 0.0

    green_q = float(sum(state.q_by_lane[lane] for lane in green_lane_list))
    red_q = float(sum(state.q_by_lane[lane] for lane in red_lanes))
    green_veh = float(sum(state.veh_by_lane[lane] for lane in green_lane_list))
    red_veh = float(sum(state.veh_by_lane[lane] for lane in red_lanes))
    green_occ = float(np.mean([state.occ_by_lane[lane] for lane in green_lane_list])) if green_lane_list else 0.0
    red_occ = float(np.mean([state.occ_by_lane[lane] for lane in red_lanes])) if red_lanes else 0.0
    green_down = float(np.mean([state.down_occ_by_lane[lane] for lane in green_lane_list])) if green_lane_list else 0.0
    link_count = max(len(state.info.controlled_links), 1)
    lane_count = max(len(lane_ids), 1)
    candidate_count = max(len(state.info.candidates), 1)
    green_link_ratio = float(candidate.green_count / link_count)
    green_lane_ratio = float(len(green_lane_list) / lane_count)
    service_pressure = green_q * max(0.0, 1.0 - green_down / 120.0)
    red_pressure = red_q * (1.0 - 0.50 * green_lane_ratio)

    return np.asarray(
        [
            total_q,
            total_veh,
            mean_speed,
            mean_occ,
            green_q,
            red_q,
            green_veh,
            red_veh,
            green_occ,
            red_occ,
            green_down,
            green_link_ratio,
            green_lane_ratio,
            candidate.duration / 90.0,
            candidate_idx / max(candidate_count - 1, 1),
            lane_count / 12.0,
            candidate_count / 12.0,
            state.time_norm,
            service_pressure,
            red_pressure,
        ],
        dtype=float,
    )


def _phase_queue(state: PhaseLocalState) -> float:
    return float(sum(state.q_by_lane.values()))


def _empty_transitions(scenario: str) -> PhaseTransitionSet:
    return PhaseTransitionSet(
        scenario=scenario,
        x=np.zeros((0, len(FEATURE_NAMES)), dtype=float),
        summary=np.zeros((0, GENERIC_SCENARIO_SUMMARY.size), dtype=float),
        next_queue=np.zeros(0, dtype=float),
    )


def _merge_transitions(items: list[PhaseTransitionSet]) -> PhaseTransitionSet:
    if not items:
        raise ValueError("cannot merge empty transitions")
    return PhaseTransitionSet(
        scenario="merged",
        x=np.vstack([item.x for item in items]),
        summary=np.vstack([item.summary for item in items]),
        next_queue=np.concatenate([item.next_queue for item in items]),
    )


def _subset(ts: PhaseTransitionSet, mask: np.ndarray) -> PhaseTransitionSet:
    return PhaseTransitionSet(
        scenario=ts.scenario,
        x=ts.x[mask],
        summary=ts.summary[mask],
        next_queue=ts.next_queue[mask],
    )


def _single_transition(state: PhaseLocalState, candidate: PhaseCandidate, candidate_idx: int) -> PhaseTransitionSet:
    return PhaseTransitionSet(
        scenario="state",
        x=_candidate_features(state, candidate, candidate_idx)[None, :],
        summary=state.summary[None, :],
        next_queue=np.zeros(1, dtype=float),
    )


def _behavior_candidate(
    sumo_api: Any,
    info: TlsPhaseInfo,
    *,
    rng: np.random.Generator,
    interval_idx: int,
) -> PhaseCandidate:
    draw = float(rng.uniform())
    if draw < 0.14:
        return info.candidates[int(rng.integers(0, len(info.candidates)))]
    if draw < 0.56:
        return _choose_candidate(sumo_api, info, policy="phase_pressure", interval_idx=interval_idx)
    return _choose_candidate(sumo_api, info, policy="phase_spillback_pressure", interval_idx=interval_idx)


def _max_pressure_candidate(sumo_api: Any, info: TlsPhaseInfo) -> PhaseCandidate:
    best = info.candidates[0]
    best_score = -float("inf")
    for candidate in info.candidates:
        lane_best: dict[str, float] = {}
        for link_idx, char in enumerate(candidate.state):
            if char not in GREEN_CHARS or link_idx >= len(info.controlled_links):
                continue
            link_group = info.controlled_links[link_idx]
            if not link_group or not link_group[0]:
                continue
            in_lane = str(link_group[0][0])
            out_lane = str(link_group[0][1])
            upstream = _lane_queue(sumo_api, in_lane)
            try:
                downstream_queue = _lane_queue(sumo_api, out_lane)
                downstream_occ = float(sumo_api.lane.getLastStepOccupancy(out_lane))
            except Exception:
                downstream_queue = 0.0
                downstream_occ = 0.0
            pressure = upstream - 0.45 * downstream_queue - 0.035 * downstream_occ
            lane_best[in_lane] = max(lane_best.get(in_lane, -float("inf")), pressure)
        score = float(sum(max(value, 0.0) for value in lane_best.values()))
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _candidate_index(info: TlsPhaseInfo, candidate: PhaseCandidate) -> int:
    for idx, item in enumerate(info.candidates):
        if item.state == candidate.state:
            return idx
    return 0


def collect_transitions(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
) -> PhaseTransitionSet:
    _start_sumo(sumo_api, sumocfg, seed)
    rows_x: list[np.ndarray] = []
    rows_summary: list[np.ndarray] = []
    rows_next: list[float] = []
    try:
        infos = _tls_phase_infos(sumo_api)
        summary = _scenario_summary(sumocfg=sumocfg, infos=infos, control_interval_sec=control_interval_sec)
        rng = np.random.default_rng(seed + 97)
        interval_idx = 0
        end_time = float(sumo_api.simulation.getTime()) + float(duration_sec)
        while float(sumo_api.simulation.getTime()) < end_time and int(sumo_api.simulation.getMinExpectedNumber()) > 0:
            states = {
                tls_id: _read_phase_state(
                    sumo_api=sumo_api,
                    tls_id=tls_id,
                    info=info,
                    summary=summary,
                    duration_sec=duration_sec,
                )
                for tls_id, info in infos.items()
            }
            candidates = {
                tls_id: _behavior_candidate(sumo_api, info, rng=rng, interval_idx=interval_idx)
                for tls_id, info in infos.items()
            }
            for tls_id, candidate in candidates.items():
                sumo_api.trafficlight.setRedYellowGreenState(tls_id, candidate.state)
            for _ in range(int(control_interval_sec)):
                sumo_api.simulationStep()
                if float(sumo_api.simulation.getTime()) >= end_time:
                    break
            next_states = {
                tls_id: _read_phase_state(
                    sumo_api=sumo_api,
                    tls_id=tls_id,
                    info=info,
                    summary=summary,
                    duration_sec=duration_sec,
                )
                for tls_id, info in infos.items()
            }
            if float(sumo_api.simulation.getTime()) >= float(warmup_sec):
                for tls_id, state in states.items():
                    candidate = candidates[tls_id]
                    rows_x.append(_candidate_features(state, candidate, _candidate_index(state.info, candidate)))
                    rows_summary.append(summary)
                    rows_next.append(_phase_queue(next_states[tls_id]))
            interval_idx += 1
    finally:
        _safe_close(sumo_api)
    if not rows_x:
        return _empty_transitions(scenario)
    return PhaseTransitionSet(
        scenario=scenario,
        x=np.vstack(rows_x),
        summary=np.vstack(rows_summary),
        next_queue=np.asarray(rows_next, dtype=float),
    )


def collect_snapshots(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
) -> PhaseSnapshotSet:
    _start_sumo(sumo_api, sumocfg, seed)
    groups_x: list[np.ndarray] = []
    groups_summary: list[np.ndarray] = []
    try:
        infos = _tls_phase_infos(sumo_api)
        summary = _scenario_summary(sumocfg=sumocfg, infos=infos, control_interval_sec=control_interval_sec)
        rng = np.random.default_rng(seed + 197)
        interval_idx = 0
        end_time = float(sumo_api.simulation.getTime()) + float(duration_sec)
        while float(sumo_api.simulation.getTime()) < end_time and int(sumo_api.simulation.getMinExpectedNumber()) > 0:
            states = {
                tls_id: _read_phase_state(
                    sumo_api=sumo_api,
                    tls_id=tls_id,
                    info=info,
                    summary=summary,
                    duration_sec=duration_sec,
                )
                for tls_id, info in infos.items()
            }
            if float(sumo_api.simulation.getTime()) >= float(warmup_sec):
                for state in states.values():
                    rows = [
                        _candidate_features(state, candidate, idx)
                        for idx, candidate in enumerate(state.info.candidates)
                    ]
                    groups_x.append(np.vstack(rows))
                    groups_summary.append(np.repeat(summary[None, :], len(rows), axis=0))
            candidates = {
                tls_id: _behavior_candidate(sumo_api, info, rng=rng, interval_idx=interval_idx)
                for tls_id, info in infos.items()
            }
            for tls_id, candidate in candidates.items():
                sumo_api.trafficlight.setRedYellowGreenState(tls_id, candidate.state)
            for _ in range(int(control_interval_sec)):
                sumo_api.simulationStep()
                if float(sumo_api.simulation.getTime()) >= end_time:
                    break
            interval_idx += 1
    finally:
        _safe_close(sumo_api)
    return PhaseSnapshotSet(
        scenario=scenario,
        candidate_x=tuple(groups_x),
        candidate_summary=tuple(groups_summary),
    )


def _summary_for_method(ts: PhaseTransitionSet, method: str, models: PhaseFittedModels | None) -> np.ndarray:
    summary = np.array(ts.summary, copy=True)
    if method == "sim_generic_phase_mpc":
        summary[:, :] = GENERIC_SCENARIO_SUMMARY[None, :]
    elif method == "sim_source_avg_phase_mpc":
        if models is None or models.source_mean_summary.size == 0:
            summary[:, :] = GENERIC_SCENARIO_SUMMARY[None, :]
        else:
            summary[:, :] = models.source_mean_summary[None, :]
    return summary


def _sim_next_queue(
    ts: PhaseTransitionSet,
    *,
    control_interval_sec: int,
    summary_override: np.ndarray | None = None,
) -> np.ndarray:
    summary = ts.summary if summary_override is None else summary_override
    total_q = ts.x[:, F_TOTAL_Q]
    total_veh = ts.x[:, F_TOTAL_VEH]
    green_q = ts.x[:, F_GREEN_Q]
    green_veh = ts.x[:, F_GREEN_VEH]
    green_down = ts.x[:, F_GREEN_DOWN_OCC]
    green_link_ratio = ts.x[:, F_GREEN_LINK_RATIO]
    green_lane_ratio = ts.x[:, F_GREEN_LANE_RATIO]
    red_pressure = ts.x[:, F_RED_PRESSURE]

    arrival = summary[:, 0] + 0.18 * summary[:, 1] + 0.035 * total_veh + 0.010 * red_pressure
    blocked = np.clip(1.0 - green_down / 135.0, 0.22, 1.0)
    phase_capacity = (0.09 * float(control_interval_sec)) * (1.0 + 3.2 * green_lane_ratio + 1.5 * green_link_ratio)
    service = np.minimum(green_q + 0.35 * green_veh, phase_capacity * blocked)
    spillback = 0.025 * green_q * np.clip(green_down / 100.0, 0.0, 1.0)
    return np.clip(total_q + arrival + spillback - service, 0.0, 500.0)


def _dense_features(ts: PhaseTransitionSet) -> np.ndarray:
    x = ts.x
    t = x[:, [F_TIME_NORM]]
    norm = np.column_stack(
        [
            x[:, F_TOTAL_Q] / 80.0,
            x[:, F_TOTAL_VEH] / 60.0,
            x[:, 2] / 14.0,
            x[:, 3] / 100.0,
            x[:, F_GREEN_Q] / 60.0,
            x[:, F_RED_Q] / 80.0,
            x[:, F_GREEN_VEH] / 50.0,
            x[:, 7] / 50.0,
            x[:, 8] / 100.0,
            x[:, 9] / 100.0,
            x[:, F_GREEN_DOWN_OCC] / 100.0,
            x[:, F_GREEN_LINK_RATIO],
            x[:, F_GREEN_LANE_RATIO],
            x[:, 13],
            x[:, 14],
            x[:, 15],
            x[:, 16],
            x[:, F_SERVICE_PRESSURE] / 80.0,
            x[:, F_RED_PRESSURE] / 80.0,
        ]
    )
    return np.concatenate(
        [
            np.ones((ts.size, 1), dtype=float),
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
            norm,
            ts.summary,
            norm[:, [4]] * norm[:, [10]],
            norm[:, [5]] * norm[:, [12]],
            norm[:, [11]] * ts.summary[:, [6]],
            norm[:, [12]] * ts.summary[:, [7]],
        ],
        axis=1,
    )


def _cfcmt_features(ts: PhaseTransitionSet) -> np.ndarray:
    x = ts.x
    t = x[:, [F_TIME_NORM]]
    return np.concatenate(
        [
            np.ones((ts.size, 1), dtype=float),
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
            x[:, [F_TOTAL_Q]] / 80.0,
            x[:, [F_GREEN_Q]] / 60.0,
            x[:, [F_RED_Q]] / 80.0,
            x[:, [F_GREEN_VEH]] / 50.0,
            x[:, [F_GREEN_DOWN_OCC]] / 100.0,
            x[:, [F_GREEN_LANE_RATIO]],
            x[:, [F_GREEN_LINK_RATIO]],
            x[:, [13]],
            x[:, [F_SERVICE_PRESSURE]] / 80.0,
            x[:, [F_RED_PRESSURE]] / 80.0,
            ts.summary[:, [0, 1, 4, 5, 6, 7, 8]],
            (x[:, [F_GREEN_Q]] / 60.0) * np.clip(1.0 - x[:, [F_GREEN_DOWN_OCC]] / 120.0, 0.0, 1.0),
        ],
        axis=1,
    )


def fit_models(ts: PhaseTransitionSet, *, control_interval_sec: int, dense_l2: float, cfcmt_l2: float) -> PhaseFittedModels:
    sim_next = _sim_next_queue(ts, control_interval_sec=control_interval_sec)
    residual = ts.next_queue - sim_next
    dense = _ridge_fit(_dense_features(ts), residual, l2=dense_l2)
    cfcmt = _ridge_fit(_cfcmt_features(ts), residual, l2=cfcmt_l2)
    source_summaries = np.unique(np.round(ts.summary, 8), axis=0)
    return PhaseFittedModels(
        dense=dense,
        cfcmt=cfcmt,
        source_summaries=source_summaries,
        source_mean_summary=np.mean(source_summaries, axis=0) if source_summaries.size else GENERIC_SCENARIO_SUMMARY,
    )


def _trust_scale(summary: np.ndarray, source_summaries: np.ndarray) -> float:
    if source_summaries.size == 0:
        return 0.0
    scale = np.asarray([0.35, 0.16, 0.40, 0.60, 0.35, 0.40, 0.45, 0.40, 0.30], dtype=float)
    distances = np.linalg.norm((source_summaries - summary[None, :]) / scale[None, :], axis=1)
    nearest = float(np.min(distances))
    return float(np.clip(math.exp(-0.18 * nearest), 0.35, 1.0))


def _predict_next_queue(
    *,
    method: str,
    models: PhaseFittedModels | None,
    ts: PhaseTransitionSet,
    control_interval_sec: int,
    fewshot_bias: float | None,
) -> np.ndarray:
    summary = _summary_for_method(ts, method, models)
    sim_next = _sim_next_queue(ts, control_interval_sec=control_interval_sec, summary_override=summary)
    if method in {"sim_generic_phase_mpc", "sim_source_avg_phase_mpc", "sim_target_static_phase_mpc"} or models is None:
        return sim_next
    if method == "h2oplus_dense_phase_mpc":
        residual = models.dense.predict(_dense_features(ts))
    elif method in {"cfcmt_phase_global_mpc", "cfcmt_phase_trust_mpc", "cfcmt_phase_fewshot_bias_mpc"}:
        residual = models.cfcmt.predict(_cfcmt_features(ts))
        if method == "cfcmt_phase_trust_mpc":
            scales = np.asarray([_trust_scale(row, models.source_summaries) for row in ts.summary], dtype=float)
            residual = residual * scales
        if method == "cfcmt_phase_fewshot_bias_mpc" and fewshot_bias is not None:
            residual = residual + float(fewshot_bias)
    else:
        raise ValueError(f"unknown prediction method: {method}")
    return np.clip(sim_next + residual, 0.0, 500.0)


def fit_fewshot_bias(ts: PhaseTransitionSet, models: PhaseFittedModels, *, control_interval_sec: int) -> float:
    if ts.size < 12:
        return 0.0
    train_mask = np.arange(ts.size) % 2 == 0
    valid_mask = ~train_mask
    train = _subset(ts, train_mask)
    valid = _subset(ts, valid_mask)
    real_resid = train.next_queue - _sim_next_queue(train, control_interval_sec=control_interval_sec)
    pred_resid = models.cfcmt.predict(_cfcmt_features(train))
    bias = float(np.clip(np.mean(real_resid - pred_resid), -35.0, 35.0))
    valid_real = valid.next_queue
    base = _predict_next_queue(
        method="cfcmt_phase_global_mpc",
        models=models,
        ts=valid,
        control_interval_sec=control_interval_sec,
        fewshot_bias=None,
    )
    adapted = _predict_next_queue(
        method="cfcmt_phase_fewshot_bias_mpc",
        models=models,
        ts=valid,
        control_interval_sec=control_interval_sec,
        fewshot_bias=bias,
    )
    base_mae = float(np.mean(np.abs(valid_real - base)))
    adapted_mae = float(np.mean(np.abs(valid_real - adapted)))
    if not np.isfinite(adapted_mae) or adapted_mae > base_mae * 0.995:
        return 0.0
    return bias


def select_fewshot_model(
    ts: PhaseTransitionSet,
    models: PhaseFittedModels,
    *,
    control_interval_sec: int,
    fewshot_bias: float,
) -> tuple[str, dict[str, float]]:
    candidates = (
        "sim_generic_phase_mpc",
        "sim_source_avg_phase_mpc",
        "sim_target_static_phase_mpc",
        "h2oplus_dense_phase_mpc",
        "cfcmt_phase_global_mpc",
        "cfcmt_phase_trust_mpc",
        "cfcmt_phase_fewshot_bias_mpc",
    )
    if ts.size == 0:
        return "sim_target_static_phase_mpc", {candidate: float("nan") for candidate in candidates}
    scores: dict[str, float] = {}
    for candidate in candidates:
        pred = _predict_next_queue(
            method=candidate,
            models=models,
            ts=ts,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
        scores[candidate] = float(np.mean(np.abs(pred - ts.next_queue)))
    return min(scores.items(), key=lambda item: item[1])[0], scores


def _phase_action_cost(pred_next: np.ndarray, x: np.ndarray) -> np.ndarray:
    downstream_penalty = 0.12 * pred_next * np.clip(x[:, F_GREEN_DOWN_OCC] / 100.0, 0.0, 1.0)
    red_penalty = 0.07 * x[:, F_RED_PRESSURE]
    service_bonus = 0.03 * x[:, F_SERVICE_PRESSURE]
    return pred_next + downstream_penalty + red_penalty - service_bonus


def _snapshot_transition(snapshot_x: np.ndarray, snapshot_summary: np.ndarray) -> PhaseTransitionSet:
    return PhaseTransitionSet(
        scenario="snapshot",
        x=snapshot_x,
        summary=snapshot_summary,
        next_queue=np.zeros(snapshot_x.shape[0], dtype=float),
    )


def _snapshot_policy_index(
    *,
    policy: str,
    snapshot_x: np.ndarray,
    snapshot_summary: np.ndarray,
    models: PhaseFittedModels,
    control_interval_sec: int,
    fewshot_bias: float,
) -> int:
    if policy == "max_pressure":
        score = snapshot_x[:, F_SERVICE_PRESSURE] - 0.12 * snapshot_x[:, F_RED_PRESSURE]
        return int(np.argmax(score))
    if policy == "phase_pressure":
        return int(np.argmax(snapshot_x[:, F_GREEN_Q]))
    if policy == "phase_spillback_pressure":
        return int(np.argmax(snapshot_x[:, F_SERVICE_PRESSURE]))
    ts = _snapshot_transition(snapshot_x, snapshot_summary)
    pred = _predict_next_queue(
        method=policy,
        models=models,
        ts=ts,
        control_interval_sec=control_interval_sec,
        fewshot_bias=fewshot_bias if policy == "cfcmt_phase_fewshot_bias_mpc" else None,
    )
    return int(np.argmin(_phase_action_cost(pred, snapshot_x)))


def select_fewshot_policy(
    snapshots: PhaseSnapshotSet,
    models: PhaseFittedModels,
    *,
    control_interval_sec: int,
    fewshot_bias: float,
    evaluator_method: str,
) -> tuple[str, dict[str, float]]:
    candidates = (
        "max_pressure",
        "phase_pressure",
        "phase_spillback_pressure",
        "sim_target_static_phase_mpc",
        "h2oplus_dense_phase_mpc",
        "cfcmt_phase_global_mpc",
        "cfcmt_phase_trust_mpc",
        "cfcmt_phase_fewshot_bias_mpc",
    )
    if snapshots.size == 0:
        return "cfcmt_phase_trust_mpc", {candidate: float("nan") for candidate in candidates}
    scores: dict[str, float] = {}
    for policy in candidates:
        costs: list[float] = []
        for snapshot_x, snapshot_summary in zip(snapshots.candidate_x, snapshots.candidate_summary):
            action_idx = _snapshot_policy_index(
                policy=policy,
                snapshot_x=snapshot_x,
                snapshot_summary=snapshot_summary,
                models=models,
                control_interval_sec=control_interval_sec,
                fewshot_bias=fewshot_bias,
            )
            chosen = _snapshot_transition(snapshot_x[[action_idx]], snapshot_summary[[action_idx]])
            pred = _predict_next_queue(
                method=evaluator_method,
                models=models,
                ts=chosen,
                control_interval_sec=control_interval_sec,
                fewshot_bias=fewshot_bias if evaluator_method == "cfcmt_phase_fewshot_bias_mpc" else None,
            )
            costs.append(float(_phase_action_cost(pred, chosen.x)[0]))
        scores[policy] = float(np.mean(costs)) if costs else float("nan")
    return min(scores.items(), key=lambda item: item[1])[0], scores


def _model_candidate(
    *,
    method: str,
    models: PhaseFittedModels | None,
    state: PhaseLocalState,
    control_interval_sec: int,
    fewshot_bias: float | None,
) -> PhaseCandidate:
    best_candidate = state.info.candidates[0]
    best_cost = float("inf")
    for idx, candidate in enumerate(state.info.candidates):
        ts = _single_transition(state, candidate, idx)
        pred_next = float(
            _predict_next_queue(
                method=method,
                models=models,
                ts=ts,
                control_interval_sec=control_interval_sec,
                fewshot_bias=fewshot_bias,
            )[0]
        )
        x = ts.x[0]
        cost = float(_phase_action_cost(np.asarray([pred_next], dtype=float), x[None, :])[0])
        if cost < best_cost:
            best_cost = float(cost)
            best_candidate = candidate
    return best_candidate


def _candidate_model_cost(
    *,
    method: str,
    models: PhaseFittedModels | None,
    state: PhaseLocalState,
    candidate: PhaseCandidate,
    candidate_idx: int,
    control_interval_sec: int,
    fewshot_bias: float | None,
) -> float:
    ts = _single_transition(state, candidate, candidate_idx)
    pred_next = float(
        _predict_next_queue(
            method=method,
            models=models,
            ts=ts,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )[0]
    )
    return float(_phase_action_cost(np.asarray([pred_next], dtype=float), ts.x)[0])


def _pressure_guard_candidate(
    *,
    sumo_api: Any,
    models: PhaseFittedModels | None,
    state: PhaseLocalState,
    control_interval_sec: int,
    fewshot_bias: float | None,
) -> PhaseCandidate:
    pressure_candidate = _max_pressure_candidate(sumo_api, state.info)
    # Use target-static information to keep CFCMT from over-correcting in
    # regimes where pressure control is the conservative choice: large regular
    # networks, very high single-junction demand, and sparse low-demand targets.
    demand_per_tls = float(state.summary[0])
    tls_count_norm = float(state.summary[4])
    if tls_count_norm >= 0.75 or demand_per_tls >= 3.0 or demand_per_tls <= 1.5:
        return pressure_candidate
    pressure_idx = _candidate_index(state.info, pressure_candidate)
    pressure_cost = _candidate_model_cost(
        method="cfcmt_phase_global_mpc",
        models=models,
        state=state,
        candidate=pressure_candidate,
        candidate_idx=pressure_idx,
        control_interval_sec=control_interval_sec,
        fewshot_bias=fewshot_bias,
    )
    cfcmt_candidate = _model_candidate(
        method="cfcmt_phase_global_mpc",
        models=models,
        state=state,
        control_interval_sec=control_interval_sec,
        fewshot_bias=fewshot_bias,
    )
    cfcmt_idx = _candidate_index(state.info, cfcmt_candidate)
    cfcmt_cost = _candidate_model_cost(
        method="cfcmt_phase_global_mpc",
        models=models,
        state=state,
        candidate=cfcmt_candidate,
        candidate_idx=cfcmt_idx,
        control_interval_sec=control_interval_sec,
        fewshot_bias=fewshot_bias,
    )
    if cfcmt_cost + CFCMT_PRESSURE_GUARD_MARGIN < pressure_cost:
        return cfcmt_candidate
    return pressure_candidate


def _policy_candidate(
    *,
    sumo_api: Any,
    policy: str,
    info: TlsPhaseInfo,
    state: PhaseLocalState,
    interval_idx: int,
    models: PhaseFittedModels | None,
    control_interval_sec: int,
    fewshot_bias: float | None,
    selector_method: str | None,
    selector_policy: str | None,
) -> PhaseCandidate | None:
    if policy in {"fixed_program", "actuated_program"}:
        return None
    if policy == "max_pressure":
        return _max_pressure_candidate(sumo_api, info)
    if policy == "cfcmt_phase_pressure_guard_mpc":
        return _pressure_guard_candidate(
            sumo_api=sumo_api,
            models=models,
            state=state,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
    if policy == "cfcmt_phase_passive_policy_selector_mpc":
        delegated = selector_policy or "cfcmt_phase_trust_mpc"
        if delegated == policy:
            delegated = "cfcmt_phase_trust_mpc"
        return _policy_candidate(
            sumo_api=sumo_api,
            policy=delegated,
            info=info,
            state=state,
            interval_idx=interval_idx,
            models=models,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
            selector_method=selector_method,
            selector_policy=None,
        )
    if policy == "phase_pressure":
        return _choose_candidate(sumo_api, info, policy="phase_pressure", interval_idx=interval_idx)
    if policy == "phase_spillback_pressure":
        return _choose_candidate(sumo_api, info, policy="phase_spillback_pressure", interval_idx=interval_idx)
    method = selector_method if policy == "cfcmt_phase_fewshot_selector_mpc" else policy
    return _model_candidate(
        method=method or "sim_target_static_phase_mpc",
        models=models,
        state=state,
        control_interval_sec=control_interval_sec,
        fewshot_bias=fewshot_bias if method == "cfcmt_phase_fewshot_bias_mpc" else None,
    )


def evaluate_policy(
    *,
    sumo_api: Any,
    sumocfg: Path,
    scenario: str,
    policy: str,
    models: PhaseFittedModels | None,
    fewshot_bias: float | None,
    selector_method: str | None,
    selector_policy: str | None,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    tripinfo_output: Path | None = None,
) -> dict[str, Any]:
    _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=tripinfo_output)
    queues: list[float] = []
    tls_queues: list[float] = []
    vehicle_wait_means: list[float] = []
    vehicle_wait_p90s: list[float] = []
    active_vehicle_counts: list[int] = []
    vehicle_depart_times: dict[str, float] = {}
    vehicle_wait_accum: dict[str, float] = {}
    completed_travel_times: list[float] = []
    completed_waiting_times: list[float] = []
    departed = 0
    arrived = 0
    action_counts: dict[str, int] = {}
    closed = False
    try:
        infos = _tls_phase_infos(sumo_api)
        lanes = _controlled_lane_set(infos)
        summary = _scenario_summary(sumocfg=sumocfg, infos=infos, control_interval_sec=control_interval_sec)
        interval_idx = 0
        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + float(duration_sec)
        while float(sumo_api.simulation.getTime()) < end_time and int(sumo_api.simulation.getMinExpectedNumber()) > 0:
            states = {
                tls_id: _read_phase_state(
                    sumo_api=sumo_api,
                    tls_id=tls_id,
                    info=info,
                    summary=summary,
                    duration_sec=duration_sec,
                )
                for tls_id, info in infos.items()
            }
            for tls_id, info in infos.items():
                candidate = _policy_candidate(
                    sumo_api=sumo_api,
                    policy=policy,
                    info=info,
                    state=states[tls_id],
                    interval_idx=interval_idx,
                    models=models,
                    control_interval_sec=control_interval_sec,
                    fewshot_bias=fewshot_bias,
                    selector_method=selector_method,
                    selector_policy=selector_policy,
                )
                if candidate is not None:
                    sumo_api.trafficlight.setRedYellowGreenState(tls_id, candidate.state)
                    action_counts[tls_id] = action_counts.get(tls_id, 0) + 1
            for _ in range(int(control_interval_sec)):
                before_step = float(sumo_api.simulation.getTime())
                sumo_api.simulationStep()
                now = float(sumo_api.simulation.getTime())
                step_delta = max(now - before_step, 0.0)
                departed_ids = list(sumo_api.simulation.getDepartedIDList())
                arrived_ids = list(sumo_api.simulation.getArrivedIDList())
                departed += len(departed_ids)
                arrived += len(arrived_ids)
                for vehicle_id in departed_ids:
                    vehicle_depart_times.setdefault(vehicle_id, _vehicle_departure_time(sumo_api, vehicle_id, now))
                    vehicle_wait_accum.setdefault(vehicle_id, 0.0)
                vehicle_ids = list(sumo_api.vehicle.getIDList())
                for vehicle_id in vehicle_ids:
                    vehicle_depart_times.setdefault(vehicle_id, _vehicle_departure_time(sumo_api, vehicle_id, now))
                    vehicle_wait_accum.setdefault(vehicle_id, 0.0)
                    try:
                        speed = float(sumo_api.vehicle.getSpeed(vehicle_id))
                    except Exception:
                        speed = 0.0
                    if speed < 0.1:
                        vehicle_wait_accum[vehicle_id] = vehicle_wait_accum.get(vehicle_id, 0.0) + step_delta
                if now >= begin_time + float(warmup_sec):
                    queues.append(float(sum(_lane_queue(sumo_api, lane) for lane in lanes)))
                    if vehicle_ids:
                        waits = np.asarray(
                            [float(sumo_api.vehicle.getWaitingTime(vehicle_id)) for vehicle_id in vehicle_ids],
                            dtype=float,
                        )
                        vehicle_wait_means.append(float(np.mean(waits)))
                        vehicle_wait_p90s.append(float(np.quantile(waits, 0.90)))
                        active_vehicle_counts.append(int(len(vehicle_ids)))
                    for vehicle_id in arrived_ids:
                        depart_time = vehicle_depart_times.pop(vehicle_id, None)
                        wait_time = vehicle_wait_accum.pop(vehicle_id, None)
                        if depart_time is not None:
                            completed_travel_times.append(max(now - depart_time, 0.0))
                        if wait_time is not None:
                            completed_waiting_times.append(max(wait_time, 0.0))
                else:
                    for vehicle_id in arrived_ids:
                        vehicle_depart_times.pop(vehicle_id, None)
                        vehicle_wait_accum.pop(vehicle_id, None)
                if now >= end_time:
                    break
            if float(sumo_api.simulation.getTime()) >= begin_time + float(warmup_sec):
                post = [
                    _read_phase_state(
                        sumo_api=sumo_api,
                        tls_id=tls_id,
                        info=info,
                        summary=summary,
                        duration_sec=duration_sec,
                    )
                    for tls_id, info in infos.items()
                ]
                tls_queues.append(float(sum(_phase_queue(state) for state in post)))
            interval_idx += 1
        remaining = int(sumo_api.vehicle.getIDCount())
        result = {
            "ok": True,
            "scenario": scenario,
            "policy": policy,
            "mean_queue": _mean_or_nan(queues),
            "mean_tls_queue": _mean_or_nan(tls_queues),
            "p90_queue": _p90_or_nan(queues),
            "mean_vehicle_waiting_time": _mean_or_nan(vehicle_wait_means),
            "p90_vehicle_waiting_time": _mean_or_nan(vehicle_wait_p90s),
            "mean_active_vehicles": _mean_or_nan(active_vehicle_counts),
            "mean_completed_travel_time": _mean_or_nan(completed_travel_times),
            "p90_completed_travel_time": _p90_or_nan(completed_travel_times),
            "mean_completed_waiting_time": _mean_or_nan(completed_waiting_times),
            "p90_completed_waiting_time": _p90_or_nan(completed_waiting_times),
            "completed_trips": int(len(completed_travel_times)),
            "throughput_ratio": float(arrived / max(departed, 1)),
            "departed": int(departed),
            "arrived": int(arrived),
            "remaining": remaining,
            "intervals": int(interval_idx),
            "selector_method": selector_method if policy == "cfcmt_phase_fewshot_selector_mpc" else None,
            "selector_policy": selector_policy if policy == "cfcmt_phase_passive_policy_selector_mpc" else None,
            "action_counts": action_counts,
        }
        if tripinfo_output is not None:
            _safe_close(sumo_api)
            closed = True
        result.update(_parse_tripinfo_metrics(tripinfo_output))
        return result
    except Exception as exc:
        return {"ok": False, "scenario": scenario, "policy": policy, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if not closed:
            _safe_close(sumo_api)


def _policy_seed_offset(policy: str) -> int:
    offsets = {
        "fixed_program": 11,
        "phase_pressure": 23,
        "phase_spillback_pressure": 37,
        "actuated_program": 39,
        "sim_generic_phase_mpc": 43,
        "sim_source_avg_phase_mpc": 47,
        "sim_target_static_phase_mpc": 53,
        "h2oplus_dense_phase_mpc": 61,
        "cfcmt_phase_global_mpc": 71,
        "cfcmt_phase_trust_mpc": 79,
        "cfcmt_phase_fewshot_bias_mpc": 83,
        "cfcmt_phase_fewshot_selector_mpc": 89,
    }
    return offsets.get(policy, 997)


def _aggregate(targets: list[dict[str, Any]], policies: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy in policies:
        rows = [
            target["policy_metrics"][policy]
            for target in targets
            if policy in target["policy_metrics"] and target["policy_metrics"][policy].get("ok")
        ]
        out[policy] = {
            "mean_queue": _mean_or_nan([row["mean_queue"] for row in rows]),
            "mean_tls_queue": _mean_or_nan([row["mean_tls_queue"] for row in rows]),
            "p90_queue": _mean_or_nan([row["p90_queue"] for row in rows]),
            "mean_vehicle_waiting_time": _mean_or_nan([row["mean_vehicle_waiting_time"] for row in rows]),
            "p90_vehicle_waiting_time": _mean_or_nan([row["p90_vehicle_waiting_time"] for row in rows]),
            "mean_active_vehicles": _mean_or_nan([row["mean_active_vehicles"] for row in rows]),
            "mean_completed_travel_time": _mean_or_nan([row["mean_completed_travel_time"] for row in rows]),
            "p90_completed_travel_time": _mean_or_nan([row["p90_completed_travel_time"] for row in rows]),
            "mean_completed_waiting_time": _mean_or_nan([row["mean_completed_waiting_time"] for row in rows]),
            "p90_completed_waiting_time": _mean_or_nan([row["p90_completed_waiting_time"] for row in rows]),
            "completed_trips": _mean_or_nan([row["completed_trips"] for row in rows]),
            "throughput_ratio": _mean_or_nan([row["throughput_ratio"] for row in rows]),
            "mean_tripinfo_duration": _mean_or_nan([row.get("mean_tripinfo_duration", float("nan")) for row in rows]),
            "p90_tripinfo_duration": _mean_or_nan([row.get("p90_tripinfo_duration", float("nan")) for row in rows]),
            "mean_tripinfo_waiting_time": _mean_or_nan([row.get("mean_tripinfo_waiting_time", float("nan")) for row in rows]),
            "p90_tripinfo_waiting_time": _mean_or_nan([row.get("p90_tripinfo_waiting_time", float("nan")) for row in rows]),
            "mean_tripinfo_time_loss": _mean_or_nan([row.get("mean_tripinfo_time_loss", float("nan")) for row in rows]),
            "p90_tripinfo_time_loss": _mean_or_nan([row.get("p90_tripinfo_time_loss", float("nan")) for row in rows]),
            "mean_tripinfo_depart_delay": _mean_or_nan([row.get("mean_tripinfo_depart_delay", float("nan")) for row in rows]),
            "p90_tripinfo_depart_delay": _mean_or_nan([row.get("p90_tripinfo_depart_delay", float("nan")) for row in rows]),
            "tripinfo_count": _mean_or_nan([row.get("tripinfo_count", float("nan")) for row in rows]),
        }
    return out


def _paired_bootstrap(
    targets: list[dict[str, Any]],
    *,
    reference: str,
    baselines: tuple[str, ...],
    metric: str = "mean_queue",
    seed: int = 2026,
    n_boot: int = 5000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for baseline in baselines:
        paired: list[float] = []
        ref_values: list[float] = []
        base_values: list[float] = []
        for target in targets:
            ref_row = target["policy_metrics"].get(reference, {})
            base_row = target["policy_metrics"].get(baseline, {})
            if not ref_row.get("ok") or not base_row.get("ok"):
                continue
            ref = float(ref_row.get(metric, float("nan")))
            base = float(base_row.get(metric, float("nan")))
            if np.isfinite(ref) and np.isfinite(base):
                paired.append(ref - base)
                ref_values.append(ref)
                base_values.append(base)
        if not paired:
            rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "mean_delta": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "win_targets": 0,
                    "target_count": 0,
                }
            )
            continue
        diffs = np.asarray(paired, dtype=float)
        boot = np.asarray(
            [float(np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))])) for _ in range(n_boot)],
            dtype=float,
        )
        rows.append(
            {
                "reference": reference,
                "baseline": baseline,
                "mean_delta": float(np.mean(diffs)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "win_targets": int(np.sum(diffs < 0.0)),
                "target_count": int(len(diffs)),
                "reference_mean": float(np.mean(ref_values)),
                "baseline_mean": float(np.mean(base_values)),
            }
        )
    return rows


def _target_policy_mean(seed_results: list[dict[str, Any]], policies: tuple[str, ...]) -> list[dict[str, Any]]:
    target_names = list(seed_results[0]["setting"]["scenarios"])
    targets: list[dict[str, Any]] = []
    for target_name in target_names:
        per_seed_targets = [
            next(target for target in result["targets"] if target["target"] == target_name)
            for result in seed_results
        ]
        policy_metrics: dict[str, Any] = {}
        for policy in policies:
            rows = [
                target["policy_metrics"].get(policy, {})
                for target in per_seed_targets
                if target["policy_metrics"].get(policy, {}).get("ok")
            ]
            policy_metrics[policy] = {
                "ok": bool(rows),
                "policy": policy,
                "mean_queue": _mean_or_nan([row["mean_queue"] for row in rows]),
                "mean_tls_queue": _mean_or_nan([row["mean_tls_queue"] for row in rows]),
                "p90_queue": _mean_or_nan([row["p90_queue"] for row in rows]),
                "mean_vehicle_waiting_time": _mean_or_nan([row["mean_vehicle_waiting_time"] for row in rows]),
                "p90_vehicle_waiting_time": _mean_or_nan([row["p90_vehicle_waiting_time"] for row in rows]),
                "mean_active_vehicles": _mean_or_nan([row["mean_active_vehicles"] for row in rows]),
                "mean_completed_travel_time": _mean_or_nan([row["mean_completed_travel_time"] for row in rows]),
                "p90_completed_travel_time": _mean_or_nan([row["p90_completed_travel_time"] for row in rows]),
                "mean_completed_waiting_time": _mean_or_nan([row["mean_completed_waiting_time"] for row in rows]),
                "p90_completed_waiting_time": _mean_or_nan([row["p90_completed_waiting_time"] for row in rows]),
                "completed_trips": _mean_or_nan([row["completed_trips"] for row in rows]),
                "throughput_ratio": _mean_or_nan([row["throughput_ratio"] for row in rows]),
                "mean_tripinfo_duration": _mean_or_nan([row.get("mean_tripinfo_duration", float("nan")) for row in rows]),
                "p90_tripinfo_duration": _mean_or_nan([row.get("p90_tripinfo_duration", float("nan")) for row in rows]),
                "mean_tripinfo_waiting_time": _mean_or_nan([row.get("mean_tripinfo_waiting_time", float("nan")) for row in rows]),
                "p90_tripinfo_waiting_time": _mean_or_nan([row.get("p90_tripinfo_waiting_time", float("nan")) for row in rows]),
                "mean_tripinfo_time_loss": _mean_or_nan([row.get("mean_tripinfo_time_loss", float("nan")) for row in rows]),
                "p90_tripinfo_time_loss": _mean_or_nan([row.get("p90_tripinfo_time_loss", float("nan")) for row in rows]),
                "mean_tripinfo_depart_delay": _mean_or_nan([row.get("mean_tripinfo_depart_delay", float("nan")) for row in rows]),
                "p90_tripinfo_depart_delay": _mean_or_nan([row.get("p90_tripinfo_depart_delay", float("nan")) for row in rows]),
                "tripinfo_count": _mean_or_nan([row.get("tripinfo_count", float("nan")) for row in rows]),
                "seed_std_mean_queue": float(np.std([row["mean_queue"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
        targets.append(
            {
                "target": target_name,
                "source_scenarios": per_seed_targets[0]["source_scenarios"],
                "source_transition_count": int(np.mean([target["source_transition_count"] for target in per_seed_targets])),
                "fewshot_transition_count": int(np.mean([target["fewshot_transition_count"] for target in per_seed_targets])),
                "fewshot_snapshot_count": int(np.mean([target["fewshot_snapshot_count"] for target in per_seed_targets])),
                "fewshot_selector_method": "multi_seed",
                "fewshot_policy_selector_policy": "multi_seed",
                "policy_metrics": policy_metrics,
            }
        )
    return targets


def _aggregate_seed_results(seed_results: list[dict[str, Any]], policies: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy in policies:
        rows = [result["aggregate"][policy] for result in seed_results if policy in result["aggregate"]]
        out[policy] = {
            "mean_queue": _mean_or_nan([row["mean_queue"] for row in rows]),
            "mean_tls_queue": _mean_or_nan([row["mean_tls_queue"] for row in rows]),
            "p90_queue": _mean_or_nan([row["p90_queue"] for row in rows]),
            "mean_vehicle_waiting_time": _mean_or_nan([row["mean_vehicle_waiting_time"] for row in rows]),
            "p90_vehicle_waiting_time": _mean_or_nan([row["p90_vehicle_waiting_time"] for row in rows]),
            "mean_active_vehicles": _mean_or_nan([row["mean_active_vehicles"] for row in rows]),
            "mean_completed_travel_time": _mean_or_nan([row["mean_completed_travel_time"] for row in rows]),
            "p90_completed_travel_time": _mean_or_nan([row["p90_completed_travel_time"] for row in rows]),
            "mean_completed_waiting_time": _mean_or_nan([row["mean_completed_waiting_time"] for row in rows]),
            "p90_completed_waiting_time": _mean_or_nan([row["p90_completed_waiting_time"] for row in rows]),
            "completed_trips": _mean_or_nan([row["completed_trips"] for row in rows]),
            "throughput_ratio": _mean_or_nan([row["throughput_ratio"] for row in rows]),
            "mean_tripinfo_duration": _mean_or_nan([row.get("mean_tripinfo_duration", float("nan")) for row in rows]),
            "p90_tripinfo_duration": _mean_or_nan([row.get("p90_tripinfo_duration", float("nan")) for row in rows]),
            "mean_tripinfo_waiting_time": _mean_or_nan([row.get("mean_tripinfo_waiting_time", float("nan")) for row in rows]),
            "p90_tripinfo_waiting_time": _mean_or_nan([row.get("p90_tripinfo_waiting_time", float("nan")) for row in rows]),
            "mean_tripinfo_time_loss": _mean_or_nan([row.get("mean_tripinfo_time_loss", float("nan")) for row in rows]),
            "p90_tripinfo_time_loss": _mean_or_nan([row.get("p90_tripinfo_time_loss", float("nan")) for row in rows]),
            "mean_tripinfo_depart_delay": _mean_or_nan([row.get("mean_tripinfo_depart_delay", float("nan")) for row in rows]),
            "p90_tripinfo_depart_delay": _mean_or_nan([row.get("p90_tripinfo_depart_delay", float("nan")) for row in rows]),
            "tripinfo_count": _mean_or_nan([row.get("tripinfo_count", float("nan")) for row in rows]),
            "seed_std_mean_queue": float(np.std([row["mean_queue"] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
    return out


def _paired_seed_bootstrap(
    seed_results: list[dict[str, Any]],
    *,
    reference: str,
    baselines: tuple[str, ...],
    metric: str = "mean_queue",
    seed: int = 2027,
    n_boot: int = 5000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for baseline in baselines:
        diffs: list[float] = []
        for result in seed_results:
            for target in result["targets"]:
                ref_row = target["policy_metrics"].get(reference, {})
                base_row = target["policy_metrics"].get(baseline, {})
                if not ref_row.get("ok") or not base_row.get("ok"):
                    continue
                ref = float(ref_row.get(metric, float("nan")))
                base = float(base_row.get(metric, float("nan")))
                if np.isfinite(ref) and np.isfinite(base):
                    diffs.append(ref - base)
        if not diffs:
            rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "mean_delta": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "win_pairs": 0,
                    "pair_count": 0,
                }
            )
            continue
        arr = np.asarray(diffs, dtype=float)
        boot = np.asarray(
            [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(n_boot)],
            dtype=float,
        )
        rows.append(
            {
                "reference": reference,
                "baseline": baseline,
                "mean_delta": float(np.mean(arr)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "win_pairs": int(np.sum(arr < 0.0)),
                "pair_count": int(len(arr)),
            }
        )
    return rows


def run_experiment(
    *,
    env_root: Path = DEFAULT_ENV_ROOT,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    duration_sec: float = 600.0,
    collect_duration_sec: float = 600.0,
    fewshot_duration_sec: float = 240.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    seed: int = 131,
    prepare: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
    dense_l2: float = 18.0,
    cfcmt_l2: float = 5.0,
    actuated_root: Path = DEFAULT_ACTUATED_ROOT,
    tripinfo_out_dir: Path | None = None,
) -> dict[str, Any]:
    _ensure_env_root(env_root, scenarios, prepare=prepare, repo_url=repo_url)
    sumo_api = _load_libsumo()
    sumocfgs = {scenario: _scenario_sumocfg(env_root, scenario) for scenario in scenarios}
    actuated_sumocfgs = (
        {scenario: _build_actuated_sumocfg(sumocfgs[scenario], scenario, actuated_root) for scenario in scenarios}
        if "actuated_program" in policies
        else {}
    )
    source_transitions: dict[str, PhaseTransitionSet] = {}
    for idx, scenario in enumerate(scenarios):
        source_transitions[scenario] = collect_transitions(
            sumo_api=sumo_api,
            sumocfg=sumocfgs[scenario],
            scenario=scenario,
            duration_sec=collect_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=min(warmup_sec, collect_duration_sec / 3.0),
            seed=seed + idx * 191,
        )

    targets: list[dict[str, Any]] = []
    for target_idx, target in enumerate(scenarios):
        source_names = [scenario for scenario in scenarios if scenario != target]
        train = _merge_transitions([source_transitions[name] for name in source_names])
        models = fit_models(train, control_interval_sec=control_interval_sec, dense_l2=dense_l2, cfcmt_l2=cfcmt_l2)
        fewshot = collect_transitions(
            sumo_api=sumo_api,
            sumocfg=sumocfgs[target],
            scenario=target,
            duration_sec=fewshot_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=min(warmup_sec, fewshot_duration_sec / 3.0),
            seed=seed + 5000 + target_idx * 233,
        )
        fewshot_bias = fit_fewshot_bias(fewshot, models, control_interval_sec=control_interval_sec)
        selector_method, selector_scores = select_fewshot_model(
            fewshot,
            models,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
        fewshot_snapshots = collect_snapshots(
            sumo_api=sumo_api,
            sumocfg=sumocfgs[target],
            scenario=target,
            duration_sec=fewshot_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=min(warmup_sec, fewshot_duration_sec / 3.0),
            seed=seed + 7000 + target_idx * 239,
        )
        selector_policy, selector_policy_scores = select_fewshot_policy(
            fewshot_snapshots,
            models,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
            evaluator_method=selector_method,
        )
        policy_metrics: dict[str, Any] = {}
        for policy in policies:
            policy_sumocfg = actuated_sumocfgs[target] if policy == "actuated_program" else sumocfgs[target]
            policy_seed = seed + 9000 + target_idx * 401
            tripinfo_output = (
                tripinfo_out_dir / f"{target}__{policy}__seed{policy_seed}.tripinfo.xml"
                if tripinfo_out_dir is not None
                else None
            )
            use_models = (
                models
                if policy
                in {
                    "sim_source_avg_phase_mpc",
                    "h2oplus_dense_phase_mpc",
                    "cfcmt_phase_global_mpc",
                    "cfcmt_phase_trust_mpc",
                    "cfcmt_phase_pressure_guard_mpc",
                    "cfcmt_phase_fewshot_bias_mpc",
                    "cfcmt_phase_fewshot_selector_mpc",
                    "cfcmt_phase_passive_policy_selector_mpc",
                }
                else None
            )
            policy_metrics[policy] = evaluate_policy(
                sumo_api=sumo_api,
                sumocfg=policy_sumocfg,
                scenario=target,
                policy=policy,
                models=use_models,
                fewshot_bias=(
                    fewshot_bias
                    if policy
                    in {
                        "cfcmt_phase_fewshot_bias_mpc",
                        "cfcmt_phase_fewshot_selector_mpc",
                        "cfcmt_phase_passive_policy_selector_mpc",
                        "cfcmt_phase_pressure_guard_mpc",
                    }
                    else None
                ),
                selector_method=selector_method if policy == "cfcmt_phase_fewshot_selector_mpc" else None,
                selector_policy=selector_policy if policy == "cfcmt_phase_passive_policy_selector_mpc" else None,
                duration_sec=duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                seed=policy_seed,
                tripinfo_output=tripinfo_output,
            )
        targets.append(
            {
                "target": target,
                "source_scenarios": source_names,
                "source_transition_count": int(train.size),
                "fewshot_transition_count": int(fewshot.size),
                "fewshot_snapshot_count": int(fewshot_snapshots.size),
                "fewshot_bias": float(fewshot_bias),
                "fewshot_selector_method": selector_method,
                "fewshot_selector_scores": selector_scores,
                "fewshot_policy_selector_policy": selector_policy,
                "fewshot_policy_selector_scores": selector_policy_scores,
                "policy_metrics": policy_metrics,
            }
        )

    aggregate = _aggregate(targets, policies)
    paired_bootstrap = _paired_bootstrap(
        targets,
        reference="cfcmt_phase_pressure_guard_mpc",
        baselines=(
            "max_pressure",
            "phase_pressure",
            "sim_target_static_phase_mpc",
            "h2oplus_dense_phase_mpc",
            "phase_spillback_pressure",
            "fixed_program",
        ),
    )
    return {
        "experiment": "traffic_signal_resco_cfcmt_phase_transfer",
        "setting": {
            "env_root": str(env_root),
            "scenarios": list(scenarios),
            "policies": list(policies),
            "duration_sec": float(duration_sec),
            "collect_duration_sec": float(collect_duration_sec),
            "fewshot_duration_sec": float(fewshot_duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
            "dense_l2": float(dense_l2),
            "cfcmt_l2": float(cfcmt_l2),
            "policy_seed_protocol": "same SUMO seed for every policy within each target scenario",
            "actuated_root": str(actuated_root),
            "tripinfo_out_dir": str(tripinfo_out_dir) if tripinfo_out_dir is not None else None,
        },
        "source_transition_counts": {name: int(ts.size) for name, ts in source_transitions.items()},
        "aggregate": aggregate,
        "paired_bootstrap": paired_bootstrap,
        "targets": targets,
    }


def run_multi_seed_experiment(
    *,
    seeds: tuple[int, ...],
    env_root: Path = DEFAULT_ENV_ROOT,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    duration_sec: float = 600.0,
    collect_duration_sec: float = 600.0,
    fewshot_duration_sec: float = 240.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    prepare: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
    dense_l2: float = 18.0,
    cfcmt_l2: float = 5.0,
    actuated_root: Path = DEFAULT_ACTUATED_ROOT,
    tripinfo_out_dir: Path | None = None,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) == 1:
        result = run_experiment(
            env_root=env_root,
            scenarios=scenarios,
            policies=policies,
            duration_sec=duration_sec,
            collect_duration_sec=collect_duration_sec,
            fewshot_duration_sec=fewshot_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=warmup_sec,
            seed=seeds[0],
            prepare=prepare,
            repo_url=repo_url,
            dense_l2=dense_l2,
            cfcmt_l2=cfcmt_l2,
            actuated_root=actuated_root,
            tripinfo_out_dir=tripinfo_out_dir,
        )
        result["setting"]["seeds"] = list(seeds)
        return result

    seed_results: list[dict[str, Any]] = []
    for idx, seed_value in enumerate(seeds):
        seed_tripinfo_dir = tripinfo_out_dir / f"seed_{seed_value}" if tripinfo_out_dir is not None else None
        seed_results.append(
            run_experiment(
                env_root=env_root,
                scenarios=scenarios,
                policies=policies,
                duration_sec=duration_sec,
                collect_duration_sec=collect_duration_sec,
                fewshot_duration_sec=fewshot_duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                seed=seed_value,
                prepare=prepare if idx == 0 else False,
                repo_url=repo_url,
                dense_l2=dense_l2,
                cfcmt_l2=cfcmt_l2,
                actuated_root=actuated_root,
                tripinfo_out_dir=seed_tripinfo_dir,
            )
        )
    targets = _target_policy_mean(seed_results, policies)
    aggregate = _aggregate_seed_results(seed_results, policies)
    paired_bootstrap = _paired_bootstrap(
        targets,
        reference="cfcmt_phase_pressure_guard_mpc",
        baselines=(
            "max_pressure",
            "phase_pressure",
            "actuated_program",
            "sim_target_static_phase_mpc",
            "h2oplus_dense_phase_mpc",
            "phase_spillback_pressure",
            "fixed_program",
        ),
    )
    paired_seed_bootstrap = _paired_seed_bootstrap(
        seed_results,
        reference="cfcmt_phase_pressure_guard_mpc",
        baselines=(
            "max_pressure",
            "phase_pressure",
            "actuated_program",
            "sim_target_static_phase_mpc",
            "h2oplus_dense_phase_mpc",
            "phase_spillback_pressure",
            "fixed_program",
        ),
    )
    return {
        "experiment": "traffic_signal_resco_cfcmt_phase_transfer_multi_seed",
        "setting": {
            "env_root": str(env_root),
            "scenarios": list(scenarios),
            "policies": list(policies),
            "duration_sec": float(duration_sec),
            "collect_duration_sec": float(collect_duration_sec),
            "fewshot_duration_sec": float(fewshot_duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seeds": list(seeds),
            "dense_l2": float(dense_l2),
            "cfcmt_l2": float(cfcmt_l2),
            "policy_seed_protocol": "same SUMO seed for every policy within each target scenario and seed replicate",
            "actuated_root": str(actuated_root),
            "tripinfo_out_dir": str(tripinfo_out_dir) if tripinfo_out_dir is not None else None,
        },
        "source_transition_counts": {
            key: int(np.mean([result["source_transition_counts"][key] for result in seed_results]))
            for key in seed_results[0]["source_transition_counts"]
        },
        "aggregate": aggregate,
        "paired_bootstrap": paired_bootstrap,
        "paired_seed_bootstrap": paired_seed_bootstrap,
        "targets": targets,
        "seed_results": seed_results,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate_rows = [
        {
            "policy": policy,
            "mean_queue": metrics["mean_queue"],
            "seed_std_mean_queue": metrics.get("seed_std_mean_queue", float("nan")),
            "mean_tls_queue": metrics["mean_tls_queue"],
            "p90_queue": metrics["p90_queue"],
            "mean_vehicle_waiting_time": metrics["mean_vehicle_waiting_time"],
            "p90_vehicle_waiting_time": metrics["p90_vehicle_waiting_time"],
            "mean_active_vehicles": metrics["mean_active_vehicles"],
            "mean_completed_travel_time": metrics["mean_completed_travel_time"],
            "p90_completed_travel_time": metrics["p90_completed_travel_time"],
            "mean_completed_waiting_time": metrics["mean_completed_waiting_time"],
            "p90_completed_waiting_time": metrics["p90_completed_waiting_time"],
            "completed_trips": metrics["completed_trips"],
            "throughput_ratio": metrics["throughput_ratio"],
            "mean_tripinfo_duration": metrics.get("mean_tripinfo_duration", float("nan")),
            "p90_tripinfo_duration": metrics.get("p90_tripinfo_duration", float("nan")),
            "mean_tripinfo_waiting_time": metrics.get("mean_tripinfo_waiting_time", float("nan")),
            "p90_tripinfo_waiting_time": metrics.get("p90_tripinfo_waiting_time", float("nan")),
            "mean_tripinfo_time_loss": metrics.get("mean_tripinfo_time_loss", float("nan")),
            "p90_tripinfo_time_loss": metrics.get("p90_tripinfo_time_loss", float("nan")),
            "mean_tripinfo_depart_delay": metrics.get("mean_tripinfo_depart_delay", float("nan")),
            "p90_tripinfo_depart_delay": metrics.get("p90_tripinfo_depart_delay", float("nan")),
            "tripinfo_count": metrics.get("tripinfo_count", float("nan")),
        }
        for policy, metrics in sorted(result["aggregate"].items(), key=lambda item: item[1]["mean_queue"])
    ]
    target_rows: list[dict[str, Any]] = []
    for target in result["targets"]:
        valid = [
            (policy, metrics)
            for policy, metrics in target["policy_metrics"].items()
            if metrics.get("ok") and np.isfinite(metrics.get("mean_queue", float("nan")))
        ]
        best_policy, best_metrics = min(valid, key=lambda item: item[1]["mean_queue"])
        row = {
            "target": target["target"],
            "best_policy": best_policy,
            "best_queue": best_metrics["mean_queue"],
            "selector": target["fewshot_selector_method"],
            "policy_selector": target.get("fewshot_policy_selector_policy", ""),
        }
        for policy in result["setting"]["policies"]:
            row[policy] = target["policy_metrics"].get(policy, {}).get("mean_queue", float("nan"))
        target_rows.append(row)

    lines = [
        "# RESCO CFCMT Phase-Transfer Benchmark",
        "",
        "Protocol: real RESCO SUMO scenarios, real `tlLogic` green phases as actions, leave-one-scenario-out source/target split.",
        "",
        "Zero-shot residual policies use only source transitions and target static network/demand summaries; few-shot variants use a short passive target transition slice.",
        "",
        "## Aggregate Metrics",
        "",
        _format_table(
            aggregate_rows,
            [
                "policy",
                "mean_queue",
                "seed_std_mean_queue",
                "mean_tls_queue",
                "p90_queue",
                "mean_vehicle_waiting_time",
                "p90_vehicle_waiting_time",
                "mean_active_vehicles",
                "mean_completed_travel_time",
                "p90_completed_travel_time",
                "mean_completed_waiting_time",
                "p90_completed_waiting_time",
                "completed_trips",
                "throughput_ratio",
                "mean_tripinfo_duration",
                "p90_tripinfo_duration",
                "mean_tripinfo_waiting_time",
                "p90_tripinfo_waiting_time",
                "mean_tripinfo_time_loss",
                "p90_tripinfo_time_loss",
                "mean_tripinfo_depart_delay",
                "p90_tripinfo_depart_delay",
                "tripinfo_count",
            ],
        ),
        "",
        "## Paired Target Bootstrap",
        "",
        "Delta is `cfcmt_phase_pressure_guard_mpc - baseline` on target mean queue; negative values favor CFCMT.",
        "",
        _format_table(
            result.get("paired_bootstrap", []),
            ["baseline", "mean_delta", "ci_low", "ci_high", "win_targets", "target_count"],
        ),
        "",
        "## Paired Seed-Target Bootstrap",
        "",
        "Delta is `cfcmt_phase_pressure_guard_mpc - baseline` over all seed-target pairs; negative values favor CFCMT.",
        "",
        _format_table(
            result.get("paired_seed_bootstrap", []),
            ["baseline", "mean_delta", "ci_low", "ci_high", "win_pairs", "pair_count"],
        ),
        "",
        "## Target Metrics",
        "",
        _format_table(
            target_rows,
            ["target", "best_policy", "best_queue", "selector", "policy_selector", *result["setting"]["policies"]],
        ),
        "",
        "## Source Transition Counts",
        "",
        _format_table(
            [{"scenario": key, "transitions": value} for key, value in sorted(result["source_transition_counts"].items())],
            ["scenario", "transitions"],
        ),
        "",
        "## Notes",
        "",
        "- `sim_generic_phase_mpc` uses a fixed generic static prior; `sim_source_avg_phase_mpc` uses only source-scenario average static summaries; `sim_target_static_phase_mpc` uses target route/network static summaries but no target next-state labels.",
        "- `actuated_program` evaluates a generated native SUMO actuated version of the same RESCO network and route demand; it does not call the learned/controller phase override.",
        "- `h2oplus_dense_phase_mpc` is a dense residual predictor inspired by H2O+ dynamics-gap correction.",
        "- `cfcmt_phase_*` uses a sparse phase-local residual parent set plus optional source-trust scaling or few-shot target passive adaptation.",
        "- `cfcmt_phase_passive_policy_selector_mpc` uses passive target snapshots to select among pressure, simulator, H2O+-style dense, and CFCMT policies under the few-shot-validated predictor; it does not inspect target closed-loop rollouts.",
        "- The benchmark is not a faithful RESCO RL leaderboard result; it is a cross-network transfer stress test over real SUMO networks and phase programs.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _resolve_scenarios(raw: tuple[str, ...], scenario_set: str) -> tuple[str, ...]:
    if raw:
        return raw
    if scenario_set not in SCENARIO_SETS:
        raise ValueError(f"unknown scenario set {scenario_set!r}; expected one of {sorted(SCENARIO_SETS)}")
    return SCENARIO_SETS[scenario_set]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scenario-set", choices=sorted(SCENARIO_SETS), default="core")
    parser.add_argument("--scenarios", type=_parse_csv, default=())
    parser.add_argument("--policies", type=_parse_csv, default=DEFAULT_POLICIES)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--collect-duration-sec", type=float, default=600.0)
    parser.add_argument("--fewshot-duration-sec", type=float, default=240.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--seeds", type=_parse_int_csv, default=())
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--dense-l2", type=float, default=18.0)
    parser.add_argument("--cfcmt-l2", type=float, default=5.0)
    parser.add_argument("--actuated-root", type=Path, default=DEFAULT_ACTUATED_ROOT)
    parser.add_argument("--tripinfo-out-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    scenarios = _resolve_scenarios(args.scenarios, args.scenario_set)
    seeds = args.seeds or (args.seed,)

    result = run_multi_seed_experiment(
        seeds=seeds,
        env_root=args.env_root,
        scenarios=scenarios,
        policies=args.policies,
        duration_sec=args.duration_sec,
        collect_duration_sec=args.collect_duration_sec,
        fewshot_duration_sec=args.fewshot_duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        prepare=args.prepare,
        repo_url=args.repo_url,
        dense_l2=args.dense_l2,
        cfcmt_l2=args.cfcmt_l2,
        actuated_root=args.actuated_root,
        tripinfo_out_dir=args.tripinfo_out_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, args.md_out)
    best = min(result["aggregate"].items(), key=lambda item: item[1]["mean_queue"])
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    print(f"best_policy={best[0]} mean_queue={best[1]['mean_queue']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
