"""SUMO Phase-1 feasibility benchmark for traffic-signal transfer.

This is the first microscopic follow-up to
``traffic_signal_transfer_feasibility.py``.  It keeps the benchmark deliberately
small: a generated 3x3 SUMO grid, one controlled center intersection, several
synthetic demand/bottleneck scenarios, and leave-one-scenario-out transfer.

The goal is not to claim a full traffic-signal-control result.  The goal is to
check whether the CFCMT mechanism-residual idea still beats simple rules when
the queue dynamics come from SUMO rather than from the analytic Phase-0 model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_transfer_feasibility import RidgeModel, _format_table, _ridge_fit


DEFAULT_OUT_ROOT = Path("H2Oplus/downloads/traffic_signal_phase1")
DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_sumo_phase1.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_sumo_phase1.md")

CENTER_TLS = "B1"
INCOMING_EDGES = ("B0B1", "B2B1", "A1B1", "C1B1")
INCOMING_LANES = tuple(f"{edge}_0" for edge in INCOMING_EDGES)
OUTGOING_BY_INCOMING = (
    ("B1B2_0", "B1C1_0"),
    ("B1B0_0", "B1A1_0"),
    ("B1C1_0", "B1B2_0"),
    ("B1A1_0", "B1B0_0"),
)
NS_INCOMING = {"B0B1", "B2B1"}
EW_INCOMING = {"A1B1", "C1B1"}


@dataclass(frozen=True)
class RouteDef:
    route_id: str
    edges: str
    incoming_idx: int
    route_class: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    ns_base_vph: float
    ew_base_vph: float
    ns_peak_vph: float
    ew_peak_vph: float
    turn_vph: float
    reversal: bool = False
    slow_edges: dict[str, float] | None = None


@dataclass(frozen=True)
class ScenarioBuild:
    spec: ScenarioSpec
    scenario_dir: Path
    net_path: Path
    route_path: Path
    config_path: Path
    summary: np.ndarray
    vehicles: int


@dataclass(frozen=True)
class State:
    time_norm: float
    q: np.ndarray
    veh: np.ndarray
    speed: np.ndarray
    occ: np.ndarray
    downstream_occ: np.ndarray
    summary: np.ndarray


@dataclass(frozen=True)
class TransitionSet:
    scenario: str
    q: np.ndarray
    veh: np.ndarray
    speed: np.ndarray
    occ: np.ndarray
    downstream_occ: np.ndarray
    time_norm: np.ndarray
    summary: np.ndarray
    action: np.ndarray
    next_q: np.ndarray

    @property
    def size(self) -> int:
        return int(self.q.shape[0])


@dataclass(frozen=True)
class MechanismModels:
    queue_models: tuple[RidgeModel, RidgeModel, RidgeModel, RidgeModel]


@dataclass(frozen=True)
class FittedSignalModels:
    dense: RidgeModel
    cfcmt: MechanismModels
    source_summaries: np.ndarray


def _route_defs() -> list[RouteDef]:
    return [
        RouteDef("s_to_n", "B0B1 B1B2", 0, "ns_main"),
        RouteDef("n_to_s", "B2B1 B1B0", 1, "ns_main"),
        RouteDef("w_to_e", "A1B1 B1C1", 2, "ew_main"),
        RouteDef("e_to_w", "C1B1 B1A1", 3, "ew_main"),
        RouteDef("s_to_e", "B0B1 B1C1", 0, "turn"),
        RouteDef("n_to_w", "B2B1 B1A1", 1, "turn"),
        RouteDef("w_to_n", "A1B1 B1B2", 2, "turn"),
        RouteDef("e_to_s", "C1B1 B1B0", 3, "turn"),
    ]


def _scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("balanced_grid", 410.0, 390.0, 560.0, 530.0, 55.0),
        ScenarioSpec("ns_arterial_peak", 620.0, 220.0, 980.0, 330.0, 70.0),
        ScenarioSpec("ew_arterial_peak", 240.0, 620.0, 360.0, 980.0, 70.0),
        ScenarioSpec(
            "downtown_spillback",
            470.0,
            500.0,
            790.0,
            840.0,
            95.0,
            slow_edges={"B1C1": 5.0, "B1A1": 4.8},
        ),
        ScenarioSpec("event_reversal", 330.0, 330.0, 880.0, 900.0, 80.0, reversal=True),
    ]


def _safe_close(sumo_api: Any) -> None:
    try:
        sumo_api.close()
    except Exception:
        pass


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _rate_for_route(spec: ScenarioSpec, route: RouteDef, t: float, duration_sec: float) -> float:
    if route.route_class == "turn":
        return float(spec.turn_vph)

    progress = 0.0 if duration_sec <= 0 else float(t / duration_sec)
    if spec.reversal:
        ns_rate = spec.ns_peak_vph if progress >= 0.5 else spec.ns_base_vph
        ew_rate = spec.ew_peak_vph if progress < 0.5 else spec.ew_base_vph
    else:
        wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * (progress - 0.25))
        ns_rate = spec.ns_base_vph + (spec.ns_peak_vph - spec.ns_base_vph) * wave
        ew_rate = spec.ew_base_vph + (spec.ew_peak_vph - spec.ew_base_vph) * wave
    return float(ns_rate if route.route_class == "ns_main" else ew_rate)


def _scenario_summary(spec: ScenarioSpec, duration_sec: float, control_interval_sec: int) -> np.ndarray:
    routes = _route_defs()
    bins = np.linspace(0.0, max(duration_sec, 1.0), num=49)
    movement_rate = np.zeros(4, dtype=float)
    for t in bins:
        for route in routes:
            movement_rate[route.incoming_idx] += _rate_for_route(spec, route, float(t), duration_sec) / len(bins)
    per_interval = movement_rate * float(control_interval_sec) / 3600.0
    ns = float(np.sum(per_interval[:2]))
    ew = float(np.sum(per_interval[2:]))
    return np.asarray(
        [
            *per_interval.tolist(),
            (ns - ew) / max(ns + ew, 1e-6),
            (ns + ew) / 8.0,
            float(bool(spec.slow_edges)),
            float(spec.reversal),
        ],
        dtype=float,
    )


def _patch_net_speeds(net_path: Path, speed_by_edge: dict[str, float]) -> None:
    if not speed_by_edge:
        return
    tree = ET.parse(net_path)
    root = tree.getroot()
    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id", "")
        if edge_id not in speed_by_edge:
            continue
        edge.set("speed", f"{speed_by_edge[edge_id]:.3f}")
        for lane in edge.findall("lane"):
            lane.set("speed", f"{speed_by_edge[edge_id]:.3f}")
    ET.indent(root, space="  ")
    tree.write(net_path, encoding="utf-8", xml_declaration=True)


def build_scenario(
    *,
    spec: ScenarioSpec,
    out_root: Path,
    duration_sec: float,
    control_interval_sec: int,
    seed: int,
) -> ScenarioBuild:
    scenario_dir = out_root / spec.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    net_path = scenario_dir / "net.net.xml"
    route_path = scenario_dir / "routes.rou.xml"
    config_path = scenario_dir / "run.sumocfg"

    subprocess.run(
        [
            "netgenerate",
            "--grid",
            "--grid.x-number",
            "3",
            "--grid.y-number",
            "3",
            "--grid.x-length",
            "220",
            "--grid.y-length",
            "220",
            "--default-junction-type",
            "traffic_light",
            "--no-turnarounds",
            "--output-file",
            str(net_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _patch_net_speeds(net_path, spec.slow_edges or {})

    rng = np.random.default_rng(seed)
    route_root = ET.Element("routes")
    ET.SubElement(
        route_root,
        "vType",
        id="car",
        vClass="passenger",
        length="5.0",
        accel="2.0",
        decel="4.5",
        sigma="0.25",
        maxSpeed="13.9",
    )
    routes = _route_defs()
    for route in routes:
        ET.SubElement(route_root, "route", id=route.route_id, edges=route.edges)

    vehicle_id = 0
    depart_step = 20.0
    for route in routes:
        t = 0.0
        while t < duration_sec:
            rate = _rate_for_route(spec, route, t, duration_sec)
            expected = max(rate, 0.0) * depart_step / 3600.0
            count = int(rng.poisson(expected))
            for _ in range(count):
                depart = min(t + float(rng.uniform(0.0, depart_step)), duration_sec - 0.1)
                ET.SubElement(
                    route_root,
                    "vehicle",
                    id=f"{spec.name}_{route.route_id}_{vehicle_id:06d}",
                    type="car",
                    route=route.route_id,
                    depart=f"{depart:.2f}",
                    departLane="best",
                    departSpeed="max",
                )
                vehicle_id += 1
            t += depart_step
    # SUMO requires vehicles sorted by departure time.
    vehicles = list(route_root.findall("vehicle"))
    for vehicle in vehicles:
        route_root.remove(vehicle)
    for vehicle in sorted(vehicles, key=lambda item: float(item.attrib["depart"])):
        route_root.append(vehicle)
    _write_xml(route_path, route_root)

    config_root = ET.Element("configuration")
    input_node = ET.SubElement(config_root, "input")
    ET.SubElement(input_node, "net-file", value=str(net_path.resolve()))
    ET.SubElement(input_node, "route-files", value=str(route_path.resolve()))
    time_node = ET.SubElement(config_root, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value=f"{duration_sec:.0f}")
    report_node = ET.SubElement(config_root, "report")
    ET.SubElement(report_node, "no-step-log", value="true")
    ET.SubElement(report_node, "no-warnings", value="true")
    processing_node = ET.SubElement(config_root, "processing")
    ET.SubElement(processing_node, "time-to-teleport", value="300")
    _write_xml(config_path, config_root)

    return ScenarioBuild(
        spec=spec,
        scenario_dir=scenario_dir,
        net_path=net_path,
        route_path=route_path,
        config_path=config_path,
        summary=_scenario_summary(spec, duration_sec, control_interval_sec),
        vehicles=vehicle_id,
    )


def _start_sumo(sumo_api: Any, config_path: Path, seed: int) -> None:
    _safe_close(sumo_api)
    sumo_api.start(
        [
            "sumo",
            "-c",
            str(config_path),
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--duration-log.disable",
            "true",
        ]
    )


def _phase_state(controlled_links: Any, action: int) -> str:
    chars: list[str] = []
    green_edges = NS_INCOMING if action == 0 else EW_INCOMING
    for link_group in controlled_links:
        in_lane = ""
        if link_group and link_group[0]:
            in_lane = str(link_group[0][0])
        edge = in_lane.rsplit("_", 1)[0]
        chars.append("G" if edge in green_edges else "r")
    return "".join(chars)


def _read_state(sumo_api: Any, *, summary: np.ndarray, duration_sec: float) -> State:
    now = float(sumo_api.simulation.getTime())
    q_values: list[float] = []
    veh_values: list[float] = []
    speed_values: list[float] = []
    occ_values: list[float] = []
    down_values: list[float] = []
    for lane, downstream_lanes in zip(INCOMING_LANES, OUTGOING_BY_INCOMING):
        halting = float(sumo_api.lane.getLastStepHaltingNumber(lane))
        veh = float(sumo_api.lane.getLastStepVehicleNumber(lane))
        speed = float(sumo_api.lane.getLastStepMeanSpeed(lane))
        occ = float(sumo_api.lane.getLastStepOccupancy(lane))
        downstream_occ = float(np.mean([sumo_api.lane.getLastStepOccupancy(item) for item in downstream_lanes]))
        q_values.append(halting + 0.35 * veh + 0.025 * occ)
        veh_values.append(veh)
        speed_values.append(speed)
        occ_values.append(occ)
        down_values.append(downstream_occ)
    return State(
        time_norm=now / max(duration_sec, 1.0),
        q=np.asarray(q_values, dtype=float),
        veh=np.asarray(veh_values, dtype=float),
        speed=np.asarray(speed_values, dtype=float),
        occ=np.asarray(occ_values, dtype=float),
        downstream_occ=np.asarray(down_values, dtype=float),
        summary=np.asarray(summary, dtype=float),
    )


def _state_cost(state: State) -> float:
    ns = float(np.sum(state.q[:2]))
    ew = float(np.sum(state.q[2:]))
    spillback = float(np.sum(state.q * np.clip(state.downstream_occ / 100.0, 0.0, 1.0)))
    return float(np.sum(state.q) + 0.20 * abs(ns - ew) + 0.55 * spillback)


def _pressure_action(state: State, *, spillback: bool) -> int:
    weights = np.ones(4, dtype=float)
    if spillback:
        weights = np.clip(1.0 - state.downstream_occ / 120.0, 0.20, 1.0)
    pressure = (state.q + 0.20 * state.veh) * weights
    return 0 if float(np.sum(pressure[:2])) >= float(np.sum(pressure[2:])) else 1


def _behavior_action(state: State, rng: np.random.Generator) -> int:
    draw = float(rng.uniform())
    if draw < 0.18:
        return int(rng.integers(0, 2))
    if draw < 0.58:
        return _pressure_action(state, spillback=False)
    return _pressure_action(state, spillback=True)


def _advance_interval(
    *,
    sumo_api: Any,
    controlled_links: Any,
    action: int,
    control_interval_sec: int,
) -> tuple[float, int, int]:
    phase = _phase_state(controlled_links, action)
    cost_sum = 0.0
    departed = 0
    arrived = 0
    for _ in range(control_interval_sec):
        sumo_api.trafficlight.setRedYellowGreenState(CENTER_TLS, phase)
        sumo_api.simulationStep()
        departed += int(sumo_api.simulation.getDepartedNumber())
        arrived += int(sumo_api.simulation.getArrivedNumber())
    return cost_sum, departed, arrived


def collect_transitions(
    *,
    sumo_api: Any,
    build: ScenarioBuild,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
) -> TransitionSet:
    _start_sumo(sumo_api, build.config_path, seed)
    controlled_links = sumo_api.trafficlight.getControlledLinks(CENTER_TLS)
    rng = np.random.default_rng(seed + 17)
    q_rows: list[np.ndarray] = []
    veh_rows: list[np.ndarray] = []
    speed_rows: list[np.ndarray] = []
    occ_rows: list[np.ndarray] = []
    down_rows: list[np.ndarray] = []
    time_rows: list[float] = []
    summary_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    next_q_rows: list[np.ndarray] = []
    try:
        while float(sumo_api.simulation.getTime()) < duration_sec:
            state = _read_state(sumo_api, summary=build.summary, duration_sec=duration_sec)
            action = _behavior_action(state, rng)
            _advance_interval(
                sumo_api=sumo_api,
                controlled_links=controlled_links,
                action=action,
                control_interval_sec=control_interval_sec,
            )
            next_state = _read_state(sumo_api, summary=build.summary, duration_sec=duration_sec)
            if float(sumo_api.simulation.getTime()) >= warmup_sec:
                q_rows.append(state.q)
                veh_rows.append(state.veh)
                speed_rows.append(state.speed)
                occ_rows.append(state.occ)
                down_rows.append(state.downstream_occ)
                time_rows.append(state.time_norm)
                summary_rows.append(state.summary)
                action_rows.append(action)
                next_q_rows.append(next_state.q)
    finally:
        _safe_close(sumo_api)

    if not q_rows:
        empty = np.zeros((0, 4), dtype=float)
        return TransitionSet(
            scenario=build.spec.name,
            q=empty,
            veh=empty,
            speed=empty,
            occ=empty,
            downstream_occ=empty,
            time_norm=np.zeros(0, dtype=float),
            summary=np.zeros((0, len(build.summary)), dtype=float),
            action=np.zeros(0, dtype=int),
            next_q=empty,
        )
    return TransitionSet(
        scenario=build.spec.name,
        q=np.vstack(q_rows),
        veh=np.vstack(veh_rows),
        speed=np.vstack(speed_rows),
        occ=np.vstack(occ_rows),
        downstream_occ=np.vstack(down_rows),
        time_norm=np.asarray(time_rows, dtype=float),
        summary=np.vstack(summary_rows),
        action=np.asarray(action_rows, dtype=int),
        next_q=np.vstack(next_q_rows),
    )


def _merge_transitions(items: list[TransitionSet]) -> TransitionSet:
    if not items:
        raise ValueError("cannot merge an empty transition list")
    return TransitionSet(
        scenario="merged",
        q=np.vstack([item.q for item in items]),
        veh=np.vstack([item.veh for item in items]),
        speed=np.vstack([item.speed for item in items]),
        occ=np.vstack([item.occ for item in items]),
        downstream_occ=np.vstack([item.downstream_occ for item in items]),
        time_norm=np.concatenate([item.time_norm for item in items]),
        summary=np.vstack([item.summary for item in items]),
        action=np.concatenate([item.action for item in items]),
        next_q=np.vstack([item.next_q for item in items]),
    )


def _subset_transitions(ts: TransitionSet, idx: np.ndarray) -> TransitionSet:
    return TransitionSet(
        scenario=ts.scenario,
        q=ts.q[idx],
        veh=ts.veh[idx],
        speed=ts.speed[idx],
        occ=ts.occ[idx],
        downstream_occ=ts.downstream_occ[idx],
        time_norm=ts.time_norm[idx],
        summary=ts.summary[idx],
        action=ts.action[idx],
        next_q=ts.next_q[idx],
    )


def _sim_next_q_arrays(
    *,
    q: np.ndarray,
    veh: np.ndarray,
    downstream_occ: np.ndarray,
    summary: np.ndarray,
    action: np.ndarray,
    control_interval_sec: int,
) -> np.ndarray:
    green = np.column_stack([action == 0, action == 0, action == 1, action == 1]).astype(float)
    arrival = summary[:, :4] + 0.05 * veh
    blocked = np.clip(1.0 - downstream_occ / 130.0, 0.25, 1.0)
    service = (0.34 * float(control_interval_sec)) * green * blocked
    return np.clip(q + arrival - service, 0.0, 80.0)


def _sim_next_q_state(state: State, action: int, control_interval_sec: int) -> np.ndarray:
    return _sim_next_q_arrays(
        q=state.q[None, :],
        veh=state.veh[None, :],
        downstream_occ=state.downstream_occ[None, :],
        summary=state.summary[None, :],
        action=np.asarray([action], dtype=int),
        control_interval_sec=control_interval_sec,
    )[0]


def _dense_features_arrays(ts: TransitionSet, action: np.ndarray) -> np.ndarray:
    t = ts.time_norm[:, None]
    t_sin = np.sin(2.0 * math.pi * t)
    t_cos = np.cos(2.0 * math.pi * t)
    a = action[:, None].astype(float)
    green = np.column_stack([action == 0, action == 0, action == 1, action == 1]).astype(float)
    q = ts.q / 20.0
    veh = ts.veh / 18.0
    speed = ts.speed / 14.0
    occ = ts.occ / 100.0
    down = ts.downstream_occ / 100.0
    return np.concatenate(
        [
            np.ones((ts.size, 1), dtype=float),
            t_sin,
            t_cos,
            q,
            veh,
            speed,
            occ,
            down,
            ts.summary,
            a,
            green,
            q * green,
            veh * green,
            down * green,
            np.sum(q[:, :2], axis=1, keepdims=True) - np.sum(q[:, 2:], axis=1, keepdims=True),
        ],
        axis=1,
    )


def _state_as_transition(state: State) -> TransitionSet:
    return TransitionSet(
        scenario="state",
        q=state.q[None, :],
        veh=state.veh[None, :],
        speed=state.speed[None, :],
        occ=state.occ[None, :],
        downstream_occ=state.downstream_occ[None, :],
        time_norm=np.asarray([state.time_norm], dtype=float),
        summary=state.summary[None, :],
        action=np.zeros(1, dtype=int),
        next_q=np.zeros((1, 4), dtype=float),
    )


def _mechanism_features_arrays(ts: TransitionSet, action: np.ndarray, movement_idx: int) -> np.ndarray:
    t = ts.time_norm[:, None]
    t_sin = np.sin(2.0 * math.pi * t)
    t_cos = np.cos(2.0 * math.pi * t)
    green = ((action == 0) if movement_idx < 2 else (action == 1)).astype(float)[:, None]
    q = ts.q / 20.0
    veh = ts.veh / 18.0
    speed = ts.speed / 14.0
    occ = ts.occ / 100.0
    down = ts.downstream_occ / 100.0
    paired_idx = 1 - movement_idx if movement_idx < 2 else 5 - movement_idx
    other_phase = (2, 3) if movement_idx < 2 else (0, 1)
    return np.concatenate(
        [
            np.ones((ts.size, 1), dtype=float),
            t_sin,
            t_cos,
            q[:, [movement_idx]],
            veh[:, [movement_idx]],
            speed[:, [movement_idx]],
            occ[:, [movement_idx]],
            down[:, [movement_idx]],
            q[:, [paired_idx]],
            veh[:, [paired_idx]],
            np.sum(q[:, other_phase], axis=1, keepdims=True),
            np.sum(veh[:, other_phase], axis=1, keepdims=True),
            green,
            green * q[:, [movement_idx]],
            green * veh[:, [movement_idx]],
            green * down[:, [movement_idx]],
            ts.summary[:, [movement_idx, 4, 5, 6, 7]],
        ],
        axis=1,
    )


def fit_models(transitions: TransitionSet, *, control_interval_sec: int, dense_l2: float, mechanism_l2: float) -> FittedSignalModels:
    sim_next = _sim_next_q_arrays(
        q=transitions.q,
        veh=transitions.veh,
        downstream_occ=transitions.downstream_occ,
        summary=transitions.summary,
        action=transitions.action,
        control_interval_sec=control_interval_sec,
    )
    residual = transitions.next_q - sim_next
    dense = _ridge_fit(_dense_features_arrays(transitions, transitions.action), residual, l2=dense_l2)
    queue_models: list[RidgeModel] = []
    for movement_idx in range(4):
        x = _mechanism_features_arrays(transitions, transitions.action, movement_idx)
        y = residual[:, movement_idx]
        queue_models.append(_ridge_fit(x, y, l2=mechanism_l2))
    return FittedSignalModels(
        dense=dense,
        cfcmt=MechanismModels(queue_models=tuple(queue_models)),  # type: ignore[arg-type]
        source_summaries=np.unique(np.round(transitions.summary, 8), axis=0),
    )


def _predict_residual_cfcmt(model: MechanismModels, ts: TransitionSet, action: np.ndarray) -> np.ndarray:
    cols = [
        model.queue_models[movement_idx].predict(_mechanism_features_arrays(ts, action, movement_idx))
        for movement_idx in range(4)
    ]
    return np.column_stack(cols)


def _residual_trust_scale(summary: np.ndarray, source_summaries: np.ndarray) -> float:
    """Shrink residuals for target static regimes not represented by sources."""

    if source_summaries.size == 0:
        return 0.0
    target_has_bottleneck = bool(summary[6] > 0.5)
    source_has_bottleneck = bool(np.any(source_summaries[:, 6] > 0.5))
    if target_has_bottleneck and not source_has_bottleneck:
        return 0.25
    scale = np.asarray([1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0], dtype=float)
    distances = np.linalg.norm((source_summaries - summary[None, :]) / scale[None, :], axis=1)
    nearest = float(np.min(distances))
    return float(np.clip(math.exp(-0.10 * nearest), 0.45, 1.0))


def _predict_next_q(
    *,
    method: str,
    models: FittedSignalModels | None,
    state: State,
    action: int,
    control_interval_sec: int,
    fewshot_bias: np.ndarray | None,
) -> np.ndarray:
    sim_q = _sim_next_q_state(state, action, control_interval_sec)
    if method == "sim_mpc" or models is None:
        return sim_q
    ts = _state_as_transition(state)
    action_arr = np.asarray([action], dtype=int)
    if method == "h2oplus_dense_mpc":
        residual = models.dense.predict(_dense_features_arrays(ts, action_arr))[0]
    elif method in {"cfcmt_global_mpc", "cfcmt_fewshot_bias_mpc", "cfcmt_trust_mpc"}:
        residual = _predict_residual_cfcmt(models.cfcmt, ts, action_arr)[0]
        if method == "cfcmt_trust_mpc":
            residual = residual * _residual_trust_scale(state.summary, models.source_summaries)
        if method == "cfcmt_fewshot_bias_mpc" and fewshot_bias is not None:
            residual = residual + fewshot_bias
    else:
        raise ValueError(f"unknown model policy: {method}")
    return np.clip(sim_q + residual, 0.0, 80.0)


def _predict_next_q_transitions(
    *,
    method: str,
    models: FittedSignalModels,
    transitions: TransitionSet,
    control_interval_sec: int,
    fewshot_bias: np.ndarray | None,
) -> np.ndarray:
    sim_q = _sim_next_q_arrays(
        q=transitions.q,
        veh=transitions.veh,
        downstream_occ=transitions.downstream_occ,
        summary=transitions.summary,
        action=transitions.action,
        control_interval_sec=control_interval_sec,
    )
    if method == "sim_mpc":
        return sim_q
    if method == "h2oplus_dense_mpc":
        residual = models.dense.predict(_dense_features_arrays(transitions, transitions.action))
    elif method in {"cfcmt_global_mpc", "cfcmt_trust_mpc", "cfcmt_fewshot_bias_mpc"}:
        residual = _predict_residual_cfcmt(models.cfcmt, transitions, transitions.action)
        if method == "cfcmt_trust_mpc":
            scales = np.asarray(
                [_residual_trust_scale(summary, models.source_summaries) for summary in transitions.summary],
                dtype=float,
            )
            residual = residual * scales[:, None]
        if method == "cfcmt_fewshot_bias_mpc" and fewshot_bias is not None:
            residual = residual + fewshot_bias[None, :]
    else:
        raise ValueError(f"unknown prediction method: {method}")
    return np.clip(sim_q + residual, 0.0, 80.0)


def select_fewshot_model(
    *,
    transitions: TransitionSet,
    models: FittedSignalModels,
    control_interval_sec: int,
    fewshot_bias: np.ndarray,
) -> tuple[str, dict[str, float]]:
    candidates = (
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
    )
    if transitions.size == 0:
        return "sim_mpc", {candidate: float("nan") for candidate in candidates}
    scores: dict[str, float] = {}
    for candidate in candidates:
        pred = _predict_next_q_transitions(
            method=candidate,
            models=models,
            transitions=transitions,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
        scores[candidate] = float(np.mean(np.abs(pred - transitions.next_q)))
    selected = min(scores.items(), key=lambda item: item[1])[0]
    return selected, scores


def _predicted_cost(state: State, q_next: np.ndarray) -> float:
    ns = float(np.sum(q_next[:2]))
    ew = float(np.sum(q_next[2:]))
    spill = float(np.sum(q_next * np.clip(state.downstream_occ / 100.0, 0.0, 1.0)))
    return float(np.sum(q_next) + 0.20 * abs(ns - ew) + 0.55 * spill)


def _model_action(
    *,
    method: str,
    models: FittedSignalModels | None,
    state: State,
    control_interval_sec: int,
    fewshot_bias: np.ndarray | None,
) -> int:
    costs = [
        _predicted_cost(
            state,
            _predict_next_q(
                method=method,
                models=models,
                state=state,
                action=action,
                control_interval_sec=control_interval_sec,
                fewshot_bias=fewshot_bias,
            ),
        )
        for action in (0, 1)
    ]
    return int(np.argmin(costs))


def fit_fewshot_bias(
    *,
    transitions: TransitionSet,
    models: FittedSignalModels,
    control_interval_sec: int,
) -> np.ndarray:
    if transitions.size < 4:
        return np.zeros(4, dtype=float)
    train_idx = np.arange(transitions.size) % 2 == 0
    valid_idx = ~train_idx
    train = _subset_transitions(transitions, train_idx)
    valid = _subset_transitions(transitions, valid_idx)

    def _real_and_pred(ts: TransitionSet) -> tuple[np.ndarray, np.ndarray]:
        sim_next = _sim_next_q_arrays(
            q=ts.q,
            veh=ts.veh,
            downstream_occ=ts.downstream_occ,
            summary=ts.summary,
            action=ts.action,
            control_interval_sec=control_interval_sec,
        )
        real_residual = ts.next_q - sim_next
        pred_residual = _predict_residual_cfcmt(models.cfcmt, ts, ts.action)
        return real_residual, pred_residual

    train_real, train_pred = _real_and_pred(train)
    bias = np.clip(np.mean(train_real - train_pred, axis=0), -15.0, 15.0)
    valid_real, valid_pred = _real_and_pred(valid)
    base_mae = float(np.mean(np.abs(valid_real - valid_pred)))
    bias_mae = float(np.mean(np.abs(valid_real - (valid_pred + bias[None, :]))))
    if not np.isfinite(bias_mae) or bias_mae > base_mae * 0.995:
        return np.zeros(4, dtype=float)
    return bias


def evaluate_policy(
    *,
    sumo_api: Any,
    build: ScenarioBuild,
    policy: str,
    models: FittedSignalModels | None,
    fewshot_bias: np.ndarray | None,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    selector_method: str | None = None,
) -> dict[str, Any]:
    _start_sumo(sumo_api, build.config_path, seed)
    controlled_links = sumo_api.trafficlight.getControlledLinks(CENTER_TLS)
    interval_idx = 0
    costs: list[float] = []
    q_totals: list[float] = []
    actions: list[int] = []
    arrived_total = 0
    departed_total = 0
    try:
        while float(sumo_api.simulation.getTime()) < duration_sec:
            state = _read_state(sumo_api, summary=build.summary, duration_sec=duration_sec)
            if policy == "fixed_time":
                action = (interval_idx // 2) % 2
            elif policy == "queue_pressure":
                action = _pressure_action(state, spillback=False)
            elif policy == "spillback_pressure":
                action = _pressure_action(state, spillback=True)
            elif policy in {
                "sim_mpc",
                "h2oplus_dense_mpc",
                "cfcmt_global_mpc",
                "cfcmt_fewshot_bias_mpc",
                "cfcmt_trust_mpc",
                "cfcmt_fewshot_selector_mpc",
            }:
                method = selector_method if policy == "cfcmt_fewshot_selector_mpc" else policy
                if method is None:
                    method = "sim_mpc"
                action = _model_action(
                    method=method,
                    models=models,
                    state=state,
                    control_interval_sec=control_interval_sec,
                    fewshot_bias=fewshot_bias,
                )
            else:
                raise ValueError(f"unknown policy: {policy}")

            _, departed, arrived = _advance_interval(
                sumo_api=sumo_api,
                controlled_links=controlled_links,
                action=action,
                control_interval_sec=control_interval_sec,
            )
            departed_total += departed
            arrived_total += arrived
            if float(sumo_api.simulation.getTime()) >= warmup_sec:
                post_state = _read_state(sumo_api, summary=build.summary, duration_sec=duration_sec)
                costs.append(_state_cost(post_state))
                q_totals.append(float(np.sum(post_state.q)))
                actions.append(action)
            interval_idx += 1
        remaining = int(sumo_api.vehicle.getIDCount())
    finally:
        _safe_close(sumo_api)

    action_arr = np.asarray(actions, dtype=int) if actions else np.zeros(0, dtype=int)
    return {
        "ok": True,
        "policy": policy,
        "mean_cost": float(np.mean(costs)) if costs else float("nan"),
        "mean_queue": float(np.mean(q_totals)) if q_totals else float("nan"),
        "p90_queue": float(np.quantile(q_totals, 0.90)) if q_totals else float("nan"),
        "arrived": int(arrived_total),
        "departed": int(departed_total),
        "remaining": int(remaining),
        "throughput_ratio": float(arrived_total / max(departed_total, 1)),
        "ns_action_share": float(np.mean(action_arr == 0)) if action_arr.size else float("nan"),
        "intervals": int(len(costs)),
        "selector_method": selector_method if policy == "cfcmt_fewshot_selector_mpc" else None,
    }


def _aggregate_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted(targets[0]["policy_metrics"].keys())
    aggregate: dict[str, Any] = {}
    for policy in policies:
        rows = [target["policy_metrics"][policy] for target in targets]
        aggregate[policy] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in ("mean_cost", "mean_queue", "p90_queue", "throughput_ratio", "ns_action_share")
        }
    return aggregate


def run_experiment(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    duration_sec: float = 900.0,
    collect_duration_sec: float = 900.0,
    fewshot_duration_sec: float = 450.0,
    control_interval_sec: int = 15,
    warmup_sec: float = 90.0,
    seed: int = 19,
    max_scenarios: int = 0,
    policies: tuple[str, ...] = (
        "fixed_time",
        "queue_pressure",
        "spillback_pressure",
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
        "cfcmt_fewshot_selector_mpc",
    ),
    dense_l2: float = 18.0,
    mechanism_l2: float = 5.0,
) -> dict[str, Any]:
    if not shutil.which("netgenerate") or not shutil.which("sumo"):
        raise RuntimeError("SUMO binaries are required for Phase 1")
    specs = _scenario_specs()
    if max_scenarios > 0:
        specs = specs[:max_scenarios]
    if len(specs) < 2:
        raise ValueError("at least two scenarios are required for leave-one-out transfer")
    build_duration = max(duration_sec, collect_duration_sec, fewshot_duration_sec)
    builds = [
        build_scenario(
            spec=spec,
            out_root=out_root,
            duration_sec=build_duration,
            control_interval_sec=control_interval_sec,
            seed=seed + idx * 101,
        )
        for idx, spec in enumerate(specs)
    ]
    sumo_api = _load_libsumo()
    source_transitions = {
        build.spec.name: collect_transitions(
            sumo_api=sumo_api,
            build=build,
            duration_sec=collect_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=min(warmup_sec, collect_duration_sec / 3.0),
            seed=seed + idx * 211,
        )
        for idx, build in enumerate(builds)
    }

    targets: list[dict[str, Any]] = []
    for target_idx, target in enumerate(builds):
        source_names = [build.spec.name for build in builds if build.spec.name != target.spec.name]
        train = _merge_transitions([source_transitions[name] for name in source_names])
        models = fit_models(
            train,
            control_interval_sec=control_interval_sec,
            dense_l2=dense_l2,
            mechanism_l2=mechanism_l2,
        )
        fewshot_transitions = collect_transitions(
            sumo_api=sumo_api,
            build=target,
            duration_sec=fewshot_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=min(warmup_sec, fewshot_duration_sec / 3.0),
            seed=seed + 5000 + target_idx * 97,
        )
        fewshot_bias = fit_fewshot_bias(
            transitions=fewshot_transitions,
            models=models,
            control_interval_sec=control_interval_sec,
        )
        selector_method, selector_scores = select_fewshot_model(
            transitions=fewshot_transitions,
            models=models,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
        policy_metrics: dict[str, Any] = {}
        for policy_idx, policy in enumerate(policies):
            use_models = (
                models
                if policy
                in {
                    "h2oplus_dense_mpc",
                    "cfcmt_global_mpc",
                    "cfcmt_trust_mpc",
                    "cfcmt_fewshot_bias_mpc",
                    "cfcmt_fewshot_selector_mpc",
                }
                else None
            )
            policy_metrics[policy] = evaluate_policy(
                sumo_api=sumo_api,
                build=target,
                policy=policy,
                models=use_models,
                fewshot_bias=fewshot_bias if policy == "cfcmt_fewshot_bias_mpc" else None,
                duration_sec=duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                seed=seed + 9000 + target_idx * 401 + policy_idx,
                selector_method=selector_method if policy == "cfcmt_fewshot_selector_mpc" else None,
            )
        targets.append(
            {
                "target": target.spec.name,
                "source_scenarios": source_names,
                "vehicles": target.vehicles,
                "source_transition_count": int(train.size),
                "fewshot_transition_count": int(fewshot_transitions.size),
                "fewshot_bias": fewshot_bias.tolist(),
                "fewshot_selector_method": selector_method,
                "fewshot_selector_scores": selector_scores,
                "policy_metrics": policy_metrics,
            }
        )

    return {
        "experiment": "traffic_signal_sumo_phase1",
        "setting": {
            "simulator": "SUMO/libsumo",
            "network": "generated_3x3_grid_center_tls",
            "controlled_tls": CENTER_TLS,
            "transfer_split": "leave_one_scenario_out",
            "duration_sec": float(duration_sec),
            "collect_duration_sec": float(collect_duration_sec),
            "fewshot_duration_sec": float(fewshot_duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
            "dense_l2": float(dense_l2),
            "mechanism_l2": float(mechanism_l2),
            "policies": list(policies),
        },
        "scenario_builds": [
            {
                "name": build.spec.name,
                "scenario_dir": str(build.scenario_dir),
                "vehicles": int(build.vehicles),
                "summary": build.summary.tolist(),
            }
            for build in builds
        ],
        "source_transition_counts": {name: int(ts.size) for name, ts in source_transitions.items()},
        "aggregate": _aggregate_targets(targets),
        "targets": targets,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate_rows = [
        {
            "policy": policy,
            "mean_cost": metrics["mean_cost"],
            "mean_queue": metrics["mean_queue"],
            "p90_queue": metrics["p90_queue"],
            "throughput_ratio": metrics["throughput_ratio"],
            "ns_action_share": metrics["ns_action_share"],
        }
        for policy, metrics in sorted(result["aggregate"].items(), key=lambda item: item[1]["mean_cost"])
    ]
    target_rows = []
    for target in result["targets"]:
        best = min(target["policy_metrics"].items(), key=lambda item: item[1]["mean_cost"])
        target_rows.append(
            {
                "target": target["target"],
                "best_policy": best[0],
                "best_cost": best[1]["mean_cost"],
                "fixed_time": target["policy_metrics"].get("fixed_time", {}).get("mean_cost", float("nan")),
                "spillback_pressure": target["policy_metrics"].get("spillback_pressure", {}).get("mean_cost", float("nan")),
                "h2oplus_dense_mpc": target["policy_metrics"].get("h2oplus_dense_mpc", {}).get("mean_cost", float("nan")),
                "cfcmt_global_mpc": target["policy_metrics"].get("cfcmt_global_mpc", {}).get("mean_cost", float("nan")),
                "cfcmt_trust_mpc": target["policy_metrics"].get("cfcmt_trust_mpc", {}).get("mean_cost", float("nan")),
                "cfcmt_fewshot_bias_mpc": target["policy_metrics"].get("cfcmt_fewshot_bias_mpc", {}).get("mean_cost", float("nan")),
                "cfcmt_fewshot_selector_mpc": target["policy_metrics"].get("cfcmt_fewshot_selector_mpc", {}).get("mean_cost", float("nan")),
                "selector": target.get("fewshot_selector_method", ""),
            }
        )

    lines = [
        "# Traffic Signal SUMO Phase 1",
        "",
        "This is a microscopic SUMO feasibility benchmark. It is not yet a full TSC paper experiment.",
        "",
        "Protocol: generated 3x3 grid, center intersection B1 controlled by libsumo, leave-one-scenario-out transfer. Zero-shot CFCMT and H2O+ use source SUMO transitions only; the few-shot CFCMT variant uses passive target transitions from a short behavior-policy run.",
        "",
        "## Aggregate Policy Metrics",
        "",
        _format_table(
            aggregate_rows,
            ["policy", "mean_cost", "mean_queue", "p90_queue", "throughput_ratio", "ns_action_share"],
        ),
        "",
        "## Target-Level Results",
        "",
        _format_table(
            target_rows,
            [
                "target",
                "best_policy",
                "best_cost",
                "fixed_time",
                "spillback_pressure",
                "h2oplus_dense_mpc",
                "cfcmt_global_mpc",
                "cfcmt_trust_mpc",
                "cfcmt_fewshot_bias_mpc",
                "cfcmt_fewshot_selector_mpc",
                "selector",
            ],
        ),
        "",
        "## Notes",
        "",
        "- This Phase 1 benchmark uses a small generated grid, not real city topology.",
        "- The few-shot row is target offline adaptation, not zero-shot.",
        "- There is no SUMO oracle row because cloning the simulator for per-state counterfactual actions is not part of this Phase 1 script.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_policies(raw: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in raw.split(",") if item.strip())
    valid = {
        "fixed_time",
        "queue_pressure",
        "spillback_pressure",
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
        "cfcmt_fewshot_selector_mpc",
    }
    unknown = sorted(set(policies) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown policies: {', '.join(unknown)}")
    return policies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--duration-sec", type=float, default=900.0)
    parser.add_argument("--collect-duration-sec", type=float, default=900.0)
    parser.add_argument("--fewshot-duration-sec", type=float, default=450.0)
    parser.add_argument("--control-interval-sec", type=int, default=15)
    parser.add_argument("--warmup-sec", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--policies", type=_parse_policies, default=None)
    parser.add_argument("--dense-l2", type=float, default=18.0)
    parser.add_argument("--mechanism-l2", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    result = run_experiment(
        out_root=args.out_root,
        duration_sec=args.duration_sec,
        collect_duration_sec=args.collect_duration_sec,
        fewshot_duration_sec=args.fewshot_duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        policies=args.policies
        or (
            "fixed_time",
            "queue_pressure",
            "spillback_pressure",
            "sim_mpc",
            "h2oplus_dense_mpc",
            "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
        "cfcmt_fewshot_selector_mpc",
        ),
        dense_l2=args.dense_l2,
        mechanism_l2=args.mechanism_l2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, args.md_out)
    best = min(result["aggregate"].items(), key=lambda item: item[1]["mean_cost"])
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    print(f"best_policy={best[0]} mean_cost={best[1]['mean_cost']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
