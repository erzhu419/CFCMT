"""SUMO Phase-2 multi-intersection traffic-signal transfer benchmark.

This extends ``traffic_signal_sumo_phase1.py`` in three ways:

* 4x4 SUMO grid with four controlled interior intersections.
* OD-style corridor and turning routes that cross multiple controlled signals.
* Optional SUMO one-step counterfactual oracle via ``saveState/loadState``.

The benchmark is still synthetic, but it is no longer a single-intersection
toy.  Source models are trained on local transitions from held-in scenarios and
evaluated as closed-loop multi-intersection controllers on the held-out target.
"""

from __future__ import annotations

import argparse
import itertools
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


DEFAULT_OUT_ROOT = Path("H2Oplus/downloads/traffic_signal_phase2")
DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_sumo_phase2.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_sumo_phase2.md")

CONTROLLED_TLS = ("B1", "B2", "C1", "C2")
GENERIC_SCENARIO_SUMMARY = np.asarray([0.70, 0.70, 0.11, 0.14, 0.0, 0.56, 0.0, 0.0], dtype=float)


@dataclass(frozen=True)
class RouteDef:
    route_id: str
    edges: str
    route_class: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    ns_base_vph: float
    ew_base_vph: float
    ns_peak_vph: float
    ew_peak_vph: float
    turn_vph: float
    diagonal_vph: float
    reversal: bool = False
    slow_edges: dict[str, float] | None = None


@dataclass(frozen=True)
class ScenarioBuild:
    spec: ScenarioSpec
    scenario_dir: Path
    net_path: Path
    route_path: Path
    config_path: Path
    scenario_summary: np.ndarray
    vehicles: int


@dataclass(frozen=True)
class TlsInfo:
    tls_id: str
    incoming_lanes: tuple[str, str, str, str]
    outgoing_by_lane: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    controlled_links: Any
    local_summary: np.ndarray


@dataclass(frozen=True)
class LocalState:
    tls_id: str
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
class FittedModels:
    dense: RidgeModel
    cfcmt: MechanismModels
    source_summaries: np.ndarray
    source_mean_summary: np.ndarray


def _route_defs() -> list[RouteDef]:
    return [
        RouteDef("west_east_row1", "A1B1 B1C1 C1D1", "ew_main"),
        RouteDef("east_west_row1", "D1C1 C1B1 B1A1", "ew_main"),
        RouteDef("west_east_row2", "A2B2 B2C2 C2D2", "ew_main"),
        RouteDef("east_west_row2", "D2C2 C2B2 B2A2", "ew_main"),
        RouteDef("south_north_colB", "B0B1 B1B2 B2B3", "ns_main"),
        RouteDef("north_south_colB", "B3B2 B2B1 B1B0", "ns_main"),
        RouteDef("south_north_colC", "C0C1 C1C2 C2C3", "ns_main"),
        RouteDef("north_south_colC", "C3C2 C2C1 C1C0", "ns_main"),
        RouteDef("sw_to_east", "A0B0 B0B1 B1C1 C1D1", "diagonal"),
        RouteDef("se_to_west", "D0C0 C0C1 C1B1 B1A1", "diagonal"),
        RouteDef("nw_to_east", "A3B3 B3B2 B2C2 C2D2", "diagonal"),
        RouteDef("ne_to_west", "D3C3 C3C2 C2B2 B2A2", "diagonal"),
        RouteDef("south_to_east_B", "B0B1 B1C1 C1D1", "turn"),
        RouteDef("north_to_west_B", "B3B2 B2A2", "turn"),
        RouteDef("west_to_north_C", "A1B1 B1C1 C1C2 C2C3", "turn"),
        RouteDef("east_to_south_C", "D2C2 C2C1 C1C0", "turn"),
    ]


def _scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("balanced_corridors", 520.0, 520.0, 760.0, 750.0, 90.0, 120.0),
        ScenarioSpec("ns_commute", 760.0, 330.0, 1180.0, 460.0, 100.0, 120.0),
        ScenarioSpec("ew_commute", 330.0, 760.0, 470.0, 1190.0, 100.0, 120.0),
        ScenarioSpec(
            "downtown_bottleneck",
            650.0,
            650.0,
            980.0,
            1000.0,
            140.0,
            180.0,
            slow_edges={"C1D1": 4.8, "C2D2": 4.5, "B2B3": 5.0, "C2C3": 5.0},
        ),
        ScenarioSpec("event_reversal", 430.0, 430.0, 1150.0, 1120.0, 130.0, 190.0, reversal=True),
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


def _parse_edge(edge_id: str) -> tuple[str, str]:
    if len(edge_id) < 4:
        return edge_id[:2], edge_id[2:]
    return edge_id[:2], edge_id[2:4]


def _edge_axis(edge_id: str) -> int:
    from_node, to_node = _parse_edge(edge_id)
    return 0 if from_node[:1] == to_node[:1] else 1


def _tls_xy(tls_id: str) -> tuple[float, float]:
    letter = tls_id[:1].upper()
    number = int(tls_id[1:])
    x = (ord(letter) - ord("A")) / 3.0
    y = number / 3.0
    return float(x), float(y)


def _rate_for_route(spec: ScenarioSpec, route: RouteDef, t: float, duration_sec: float) -> float:
    progress = 0.0 if duration_sec <= 0 else float(t / duration_sec)
    if route.route_class == "turn":
        return float(spec.turn_vph)
    if route.route_class == "diagonal":
        return float(spec.diagonal_vph)
    if spec.reversal:
        ns_rate = spec.ns_peak_vph if progress >= 0.5 else spec.ns_base_vph
        ew_rate = spec.ew_peak_vph if progress < 0.5 else spec.ew_base_vph
    else:
        wave = 0.5 + 0.5 * math.sin(2.0 * math.pi * (progress - 0.25))
        ns_rate = spec.ns_base_vph + (spec.ns_peak_vph - spec.ns_base_vph) * wave
        ew_rate = spec.ew_base_vph + (spec.ew_peak_vph - spec.ew_base_vph) * wave
    return float(ns_rate if route.route_class == "ns_main" else ew_rate)


def _scenario_summary(spec: ScenarioSpec, duration_sec: float, control_interval_sec: int) -> np.ndarray:
    bins = np.linspace(0.0, max(duration_sec, 1.0), num=61)
    ns = np.mean([_rate_for_route(spec, RouteDef("tmp", "", "ns_main"), float(t), duration_sec) for t in bins])
    ew = np.mean([_rate_for_route(spec, RouteDef("tmp", "", "ew_main"), float(t), duration_sec) for t in bins])
    turn = float(spec.turn_vph)
    diag = float(spec.diagonal_vph)
    per_interval_scale = float(control_interval_sec) / 3600.0
    total = ns + ew + turn + diag
    return np.asarray(
        [
            ns * per_interval_scale / 4.0,
            ew * per_interval_scale / 4.0,
            turn * per_interval_scale / 4.0,
            diag * per_interval_scale / 4.0,
            (ns - ew) / max(ns + ew, 1e-6),
            total * per_interval_scale / 12.0,
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
            "4",
            "--grid.y-number",
            "4",
            "--grid.x-length",
            "180",
            "--grid.y-length",
            "180",
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
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        vClass="passenger",
        length="5.0",
        accel="2.0",
        decel="4.5",
        sigma="0.30",
        maxSpeed="13.9",
    )
    routes = _route_defs()
    for route in routes:
        ET.SubElement(root, "route", id=route.route_id, edges=route.edges)

    vehicle_id = 0
    depart_step = 15.0
    for route in routes:
        t = 0.0
        while t < duration_sec:
            rate = max(_rate_for_route(spec, route, t, duration_sec), 0.0)
            expected = rate * depart_step / 3600.0
            count = int(rng.poisson(expected))
            for _ in range(count):
                depart = min(t + float(rng.uniform(0.0, depart_step)), duration_sec - 0.1)
                ET.SubElement(
                    root,
                    "vehicle",
                    id=f"{spec.name}_{route.route_id}_{vehicle_id:07d}",
                    type="car",
                    route=route.route_id,
                    depart=f"{depart:.2f}",
                    departLane="best",
                    departSpeed="max",
                )
                vehicle_id += 1
            t += depart_step

    vehicles = list(root.findall("vehicle"))
    for vehicle in vehicles:
        root.remove(vehicle)
    for vehicle in sorted(vehicles, key=lambda item: float(item.attrib["depart"])):
        root.append(vehicle)
    _write_xml(route_path, root)

    config = ET.Element("configuration")
    input_node = ET.SubElement(config, "input")
    ET.SubElement(input_node, "net-file", value=str(net_path.resolve()))
    ET.SubElement(input_node, "route-files", value=str(route_path.resolve()))
    time_node = ET.SubElement(config, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value=f"{duration_sec:.0f}")
    report_node = ET.SubElement(config, "report")
    ET.SubElement(report_node, "no-step-log", value="true")
    ET.SubElement(report_node, "no-warnings", value="true")
    processing_node = ET.SubElement(config, "processing")
    ET.SubElement(processing_node, "time-to-teleport", value="300")
    _write_xml(config_path, config)

    return ScenarioBuild(
        spec=spec,
        scenario_dir=scenario_dir,
        net_path=net_path,
        route_path=route_path,
        config_path=config_path,
        scenario_summary=_scenario_summary(spec, duration_sec, control_interval_sec),
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


def _tls_info(sumo_api: Any, tls_id: str, build: ScenarioBuild) -> TlsInfo:
    controlled_links = sumo_api.trafficlight.getControlledLinks(tls_id)
    outgoing: dict[str, list[str]] = {}
    for link_group in controlled_links:
        if not link_group or not link_group[0]:
            continue
        in_lane = str(link_group[0][0])
        out_lane = str(link_group[0][1])
        outgoing.setdefault(in_lane, [])
        if out_lane not in outgoing[in_lane]:
            outgoing[in_lane].append(out_lane)
    lanes = sorted(outgoing, key=lambda lane: (_edge_axis(lane.rsplit("_", 1)[0]), lane))
    if len(lanes) != 4:
        raise RuntimeError(f"{tls_id} expected 4 incoming lanes, got {len(lanes)}: {lanes}")
    x, y = _tls_xy(tls_id)
    lane_edges = [lane.rsplit("_", 1)[0] for lane in lanes]
    local_bottleneck = float(any(edge in (build.spec.slow_edges or {}) for outs in outgoing.values() for edge in [item.rsplit("_", 1)[0] for item in outs]))
    local_summary = np.concatenate(
        [
            build.scenario_summary,
            np.asarray([x, y, local_bottleneck], dtype=float),
        ]
    )
    _ = lane_edges
    return TlsInfo(
        tls_id=tls_id,
        incoming_lanes=tuple(lanes),  # type: ignore[arg-type]
        outgoing_by_lane=tuple(tuple(outgoing[lane]) for lane in lanes),  # type: ignore[arg-type]
        controlled_links=controlled_links,
        local_summary=local_summary,
    )


def _phase_state(info: TlsInfo, action: int) -> str:
    chars: list[str] = []
    for link_group in info.controlled_links:
        in_lane = ""
        if link_group and link_group[0]:
            in_lane = str(link_group[0][0])
        edge = in_lane.rsplit("_", 1)[0]
        axis = _edge_axis(edge)
        chars.append("G" if axis == action else "r")
    return "".join(chars)


def _read_local_state(sumo_api: Any, info: TlsInfo, *, duration_sec: float) -> LocalState:
    now = float(sumo_api.simulation.getTime())
    q_values: list[float] = []
    veh_values: list[float] = []
    speed_values: list[float] = []
    occ_values: list[float] = []
    down_values: list[float] = []
    for lane, downstream_lanes in zip(info.incoming_lanes, info.outgoing_by_lane):
        halting = float(sumo_api.lane.getLastStepHaltingNumber(lane))
        veh = float(sumo_api.lane.getLastStepVehicleNumber(lane))
        speed = float(sumo_api.lane.getLastStepMeanSpeed(lane))
        occ = float(sumo_api.lane.getLastStepOccupancy(lane))
        down = float(np.mean([sumo_api.lane.getLastStepOccupancy(item) for item in downstream_lanes])) if downstream_lanes else 0.0
        q_values.append(halting + 0.35 * veh + 0.025 * occ)
        veh_values.append(veh)
        speed_values.append(speed)
        occ_values.append(occ)
        down_values.append(down)
    return LocalState(
        tls_id=info.tls_id,
        time_norm=now / max(duration_sec, 1.0),
        q=np.asarray(q_values, dtype=float),
        veh=np.asarray(veh_values, dtype=float),
        speed=np.asarray(speed_values, dtype=float),
        occ=np.asarray(occ_values, dtype=float),
        downstream_occ=np.asarray(down_values, dtype=float),
        summary=info.local_summary,
    )


def _local_cost(state: LocalState) -> float:
    ns = float(np.sum(state.q[:2]))
    ew = float(np.sum(state.q[2:]))
    spillback = float(np.sum(state.q * np.clip(state.downstream_occ / 100.0, 0.0, 1.0)))
    return float(np.sum(state.q) + 0.20 * abs(ns - ew) + 0.60 * spillback)


def _system_cost(states: dict[str, LocalState]) -> float:
    return float(sum(_local_cost(state) for state in states.values()))


def _read_all_states(sumo_api: Any, infos: dict[str, TlsInfo], *, duration_sec: float) -> dict[str, LocalState]:
    return {
        tls_id: _read_local_state(sumo_api, info, duration_sec=duration_sec)
        for tls_id, info in infos.items()
    }


def _pressure_action(state: LocalState, *, spillback: bool) -> int:
    weights = np.ones(4, dtype=float)
    if spillback:
        weights = np.clip(1.0 - state.downstream_occ / 120.0, 0.20, 1.0)
    pressure = (state.q + 0.20 * state.veh) * weights
    return 0 if float(np.sum(pressure[:2])) >= float(np.sum(pressure[2:])) else 1


def _behavior_action(state: LocalState, rng: np.random.Generator) -> int:
    draw = float(rng.uniform())
    if draw < 0.16:
        return int(rng.integers(0, 2))
    if draw < 0.58:
        return _pressure_action(state, spillback=False)
    return _pressure_action(state, spillback=True)


def _fixed_action(tls_id: str, interval_idx: int) -> int:
    x, y = _tls_xy(tls_id)
    offset = int(round(x * 3 + y * 3)) % 2
    return (interval_idx // 2 + offset) % 2


def _policy_seed_offset(policy: str) -> int:
    offsets = {
        "fixed_offset": 11,
        "local_pressure": 23,
        "spillback_pressure": 37,
        "sim_generic_mpc": 43,
        "sim_source_avg_mpc": 47,
        "sim_mpc": 53,
        "h2oplus_dense_mpc": 61,
        "cfcmt_global_mpc": 71,
        "cfcmt_trust_mpc": 79,
        "cfcmt_fewshot_bias_mpc": 83,
        "cfcmt_fewshot_selector_mpc": 89,
        "oracle_mpc": 97,
    }
    return offsets.get(policy, 997)


def _advance_joint_interval(
    *,
    sumo_api: Any,
    infos: dict[str, TlsInfo],
    actions: dict[str, int],
    control_interval_sec: int,
) -> tuple[int, int]:
    departed = 0
    arrived = 0
    states = {tls_id: _phase_state(info, actions[tls_id]) for tls_id, info in infos.items()}
    for _ in range(control_interval_sec):
        for tls_id, state in states.items():
            sumo_api.trafficlight.setRedYellowGreenState(tls_id, state)
        sumo_api.simulationStep()
        departed += int(sumo_api.simulation.getDepartedNumber())
        arrived += int(sumo_api.simulation.getArrivedNumber())
    return departed, arrived


def _empty_transitions(name: str, summary_dim: int) -> TransitionSet:
    empty = np.zeros((0, 4), dtype=float)
    return TransitionSet(
        scenario=name,
        q=empty,
        veh=empty,
        speed=empty,
        occ=empty,
        downstream_occ=empty,
        time_norm=np.zeros(0, dtype=float),
        summary=np.zeros((0, summary_dim), dtype=float),
        action=np.zeros(0, dtype=int),
        next_q=empty,
    )


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
    infos = {tls_id: _tls_info(sumo_api, tls_id, build) for tls_id in CONTROLLED_TLS}
    rng = np.random.default_rng(seed + 31)
    rows: dict[str, list[Any]] = {
        "q": [],
        "veh": [],
        "speed": [],
        "occ": [],
        "downstream_occ": [],
        "time_norm": [],
        "summary": [],
        "action": [],
        "next_q": [],
    }
    try:
        while float(sumo_api.simulation.getTime()) < duration_sec:
            states = _read_all_states(sumo_api, infos, duration_sec=duration_sec)
            actions = {tls_id: _behavior_action(state, rng) for tls_id, state in states.items()}
            _advance_joint_interval(
                sumo_api=sumo_api,
                infos=infos,
                actions=actions,
                control_interval_sec=control_interval_sec,
            )
            next_states = _read_all_states(sumo_api, infos, duration_sec=duration_sec)
            if float(sumo_api.simulation.getTime()) >= warmup_sec:
                for tls_id, state in states.items():
                    rows["q"].append(state.q)
                    rows["veh"].append(state.veh)
                    rows["speed"].append(state.speed)
                    rows["occ"].append(state.occ)
                    rows["downstream_occ"].append(state.downstream_occ)
                    rows["time_norm"].append(state.time_norm)
                    rows["summary"].append(state.summary)
                    rows["action"].append(actions[tls_id])
                    rows["next_q"].append(next_states[tls_id].q)
    finally:
        _safe_close(sumo_api)

    summary_dim = len(next(iter(infos.values())).local_summary)
    if not rows["q"]:
        return _empty_transitions(build.spec.name, summary_dim)
    return TransitionSet(
        scenario=build.spec.name,
        q=np.vstack(rows["q"]),
        veh=np.vstack(rows["veh"]),
        speed=np.vstack(rows["speed"]),
        occ=np.vstack(rows["occ"]),
        downstream_occ=np.vstack(rows["downstream_occ"]),
        time_norm=np.asarray(rows["time_norm"], dtype=float),
        summary=np.vstack(rows["summary"]),
        action=np.asarray(rows["action"], dtype=int),
        next_q=np.vstack(rows["next_q"]),
    )


def _merge_transitions(items: list[TransitionSet]) -> TransitionSet:
    if not items:
        raise ValueError("cannot merge empty transitions")
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


def _local_state_transition(state: LocalState) -> TransitionSet:
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


def _sim_summary_for_budget(ts: TransitionSet, method: str, models: FittedModels | None) -> np.ndarray:
    summary = np.array(ts.summary, copy=True)
    if method == "sim_generic_mpc":
        summary[:, :8] = GENERIC_SCENARIO_SUMMARY[None, :]
        summary[:, 10] = 0.0
    elif method == "sim_source_avg_mpc":
        if models is None or models.source_mean_summary.size == 0:
            summary[:, :8] = GENERIC_SCENARIO_SUMMARY[None, :]
            summary[:, 10] = 0.0
        else:
            source = models.source_mean_summary
            summary[:, :8] = source[None, :8]
            summary[:, 10] = source[10]
    return summary


def _sim_next_arrays(
    ts: TransitionSet,
    action: np.ndarray,
    control_interval_sec: int,
    *,
    summary_override: np.ndarray | None = None,
) -> np.ndarray:
    summary = ts.summary if summary_override is None else summary_override
    green = np.column_stack([action == 0, action == 0, action == 1, action == 1]).astype(float)
    scenario_arrival = np.column_stack(
        [
            summary[:, 0] + 0.35 * summary[:, 2] + 0.25 * summary[:, 3],
            summary[:, 0] + 0.35 * summary[:, 2] + 0.25 * summary[:, 3],
            summary[:, 1] + 0.35 * summary[:, 2] + 0.25 * summary[:, 3],
            summary[:, 1] + 0.35 * summary[:, 2] + 0.25 * summary[:, 3],
        ]
    )
    arrival = scenario_arrival + 0.04 * ts.veh
    blocked = np.clip(1.0 - ts.downstream_occ / 125.0, 0.20, 1.0)
    service = (0.32 * float(control_interval_sec)) * green * blocked
    return np.clip(ts.q + arrival - service, 0.0, 100.0)


def _dense_features(ts: TransitionSet, action: np.ndarray) -> np.ndarray:
    t = ts.time_norm[:, None]
    a = action[:, None].astype(float)
    green = np.column_stack([action == 0, action == 0, action == 1, action == 1]).astype(float)
    q = ts.q / 24.0
    veh = ts.veh / 20.0
    speed = ts.speed / 14.0
    occ = ts.occ / 100.0
    down = ts.downstream_occ / 100.0
    return np.concatenate(
        [
            np.ones((ts.size, 1), dtype=float),
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
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


def _mechanism_features(ts: TransitionSet, action: np.ndarray, movement_idx: int) -> np.ndarray:
    t = ts.time_norm[:, None]
    green = ((action == 0) if movement_idx < 2 else (action == 1)).astype(float)[:, None]
    q = ts.q / 24.0
    veh = ts.veh / 20.0
    speed = ts.speed / 14.0
    occ = ts.occ / 100.0
    down = ts.downstream_occ / 100.0
    paired_idx = 1 - movement_idx if movement_idx < 2 else 5 - movement_idx
    other_phase = (2, 3) if movement_idx < 2 else (0, 1)
    return np.concatenate(
        [
            np.ones((ts.size, 1), dtype=float),
            np.sin(2.0 * math.pi * t),
            np.cos(2.0 * math.pi * t),
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
            ts.summary[:, [movement_idx if movement_idx < 2 else 1, 4, 5, 6, 7, 8, 9, 10]],
        ],
        axis=1,
    )


def fit_models(ts: TransitionSet, *, control_interval_sec: int, dense_l2: float, mechanism_l2: float) -> FittedModels:
    sim_next = _sim_next_arrays(ts, ts.action, control_interval_sec)
    residual = ts.next_q - sim_next
    dense = _ridge_fit(_dense_features(ts, ts.action), residual, l2=dense_l2)
    queue_models: list[RidgeModel] = []
    for movement_idx in range(4):
        queue_models.append(_ridge_fit(_mechanism_features(ts, ts.action, movement_idx), residual[:, movement_idx], l2=mechanism_l2))
    source_summaries = np.unique(np.round(ts.summary, 8), axis=0)
    return FittedModels(
        dense=dense,
        cfcmt=MechanismModels(queue_models=tuple(queue_models)),  # type: ignore[arg-type]
        source_summaries=source_summaries,
        source_mean_summary=np.mean(source_summaries, axis=0) if source_summaries.size else np.zeros(ts.summary.shape[1], dtype=float),
    )


def _predict_cfcmt_residual(models: FittedModels, ts: TransitionSet, action: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            models.cfcmt.queue_models[movement_idx].predict(_mechanism_features(ts, action, movement_idx))
            for movement_idx in range(4)
        ]
    )


def _trust_scale(summary: np.ndarray, source_summaries: np.ndarray) -> float:
    if source_summaries.size == 0:
        return 0.0
    scale = np.ones(summary.shape[0], dtype=float)
    scale[:8] = np.asarray([1.0, 1.0, 0.4, 0.5, 0.8, 0.9, 1.0, 1.0], dtype=float)
    scale[8:] = np.asarray([0.5, 0.5, 1.0], dtype=float)
    distances = np.linalg.norm((source_summaries - summary[None, :]) / scale[None, :], axis=1)
    nearest = float(np.min(distances))
    return float(np.clip(math.exp(-0.10 * nearest), 0.45, 1.0))


def _predict_next_arrays(
    *,
    method: str,
    models: FittedModels | None,
    ts: TransitionSet,
    action: np.ndarray,
    control_interval_sec: int,
    fewshot_bias: np.ndarray | None,
) -> np.ndarray:
    sim_summary = _sim_summary_for_budget(ts, method, models)
    sim_next = _sim_next_arrays(ts, action, control_interval_sec, summary_override=sim_summary)
    if method in {"sim_generic_mpc", "sim_source_avg_mpc", "sim_mpc"} or models is None:
        return sim_next
    if method == "h2oplus_dense_mpc":
        residual = models.dense.predict(_dense_features(ts, action))
    elif method in {"cfcmt_global_mpc", "cfcmt_trust_mpc", "cfcmt_fewshot_bias_mpc"}:
        residual = _predict_cfcmt_residual(models, ts, action)
        if method == "cfcmt_trust_mpc":
            scales = np.asarray([_trust_scale(summary, models.source_summaries) for summary in ts.summary], dtype=float)
            residual = residual * scales[:, None]
        if method == "cfcmt_fewshot_bias_mpc" and fewshot_bias is not None:
            residual = residual + fewshot_bias[None, :]
    else:
        raise ValueError(f"unknown prediction method: {method}")
    return np.clip(sim_next + residual, 0.0, 100.0)


def _state_action_cost(state: LocalState, q_next: np.ndarray) -> float:
    ns = float(np.sum(q_next[:2]))
    ew = float(np.sum(q_next[2:]))
    spillback = float(np.sum(q_next * np.clip(state.downstream_occ / 100.0, 0.0, 1.0)))
    return float(np.sum(q_next) + 0.20 * abs(ns - ew) + 0.60 * spillback)


def _model_action(
    *,
    method: str,
    models: FittedModels | None,
    state: LocalState,
    control_interval_sec: int,
    fewshot_bias: np.ndarray | None,
) -> int:
    ts = _local_state_transition(state)
    costs = []
    for action in (0, 1):
        pred = _predict_next_arrays(
            method=method,
            models=models,
            ts=ts,
            action=np.asarray([action], dtype=int),
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )[0]
        costs.append(_state_action_cost(state, pred))
    return int(np.argmin(costs))


def fit_fewshot_bias(ts: TransitionSet, models: FittedModels, *, control_interval_sec: int) -> np.ndarray:
    if ts.size < 8:
        return np.zeros(4, dtype=float)
    train_idx = np.arange(ts.size) % 2 == 0
    valid_idx = ~train_idx
    train = _subset_transitions(ts, train_idx)
    valid = _subset_transitions(ts, valid_idx)
    train_real = train.next_q - _sim_next_arrays(train, train.action, control_interval_sec)
    train_pred = _predict_cfcmt_residual(models, train, train.action)
    bias = np.clip(np.mean(train_real - train_pred, axis=0), -15.0, 15.0)
    valid_real = valid.next_q - _sim_next_arrays(valid, valid.action, control_interval_sec)
    valid_pred = _predict_cfcmt_residual(models, valid, valid.action)
    base_mae = float(np.mean(np.abs(valid_real - valid_pred)))
    bias_mae = float(np.mean(np.abs(valid_real - (valid_pred + bias[None, :]))))
    if not np.isfinite(bias_mae) or bias_mae > base_mae * 0.995:
        return np.zeros(4, dtype=float)
    return bias


def select_fewshot_model(
    ts: TransitionSet,
    models: FittedModels,
    *,
    control_interval_sec: int,
    fewshot_bias: np.ndarray,
) -> tuple[str, dict[str, float]]:
    candidates = (
        "sim_generic_mpc",
        "sim_source_avg_mpc",
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
    )
    if ts.size == 0:
        return "sim_mpc", {candidate: float("nan") for candidate in candidates}
    scores: dict[str, float] = {}
    for candidate in candidates:
        pred = _predict_next_arrays(
            method=candidate,
            models=models,
            ts=ts,
            action=ts.action,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
        scores[candidate] = float(np.mean(np.abs(pred - ts.next_q)))
    return min(scores.items(), key=lambda item: item[1])[0], scores


def _policy_actions(
    *,
    policy: str,
    states: dict[str, LocalState],
    interval_idx: int,
    models: FittedModels | None,
    control_interval_sec: int,
    fewshot_bias: np.ndarray | None,
    selector_method: str | None,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for tls_id, state in states.items():
        if policy == "fixed_offset":
            actions[tls_id] = _fixed_action(tls_id, interval_idx)
        elif policy == "local_pressure":
            actions[tls_id] = _pressure_action(state, spillback=False)
        elif policy == "spillback_pressure":
            actions[tls_id] = _pressure_action(state, spillback=True)
        elif policy in {
            "sim_generic_mpc",
            "sim_source_avg_mpc",
            "sim_mpc",
            "h2oplus_dense_mpc",
            "cfcmt_global_mpc",
            "cfcmt_trust_mpc",
            "cfcmt_fewshot_bias_mpc",
            "cfcmt_fewshot_selector_mpc",
        }:
            method = selector_method if policy == "cfcmt_fewshot_selector_mpc" else policy
            actions[tls_id] = _model_action(
                method=method or "sim_mpc",
                models=models,
                state=state,
                control_interval_sec=control_interval_sec,
                fewshot_bias=fewshot_bias if method == "cfcmt_fewshot_bias_mpc" else None,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
    return actions


def _oracle_actions(
    *,
    sumo_api: Any,
    infos: dict[str, TlsInfo],
    state_path: Path,
    duration_sec: float,
    control_interval_sec: int,
    oracle_horizon: int,
) -> dict[str, int]:
    sumo_api.simulation.saveState(str(state_path))
    best_cost = float("inf")
    best_actions: dict[str, int] | None = None
    tls_ids = list(infos)
    joint_actions = [
        {tls_id: int(action) for tls_id, action in zip(tls_ids, combo)}
        for combo in itertools.product((0, 1), repeat=len(tls_ids))
    ]
    horizon = max(1, int(oracle_horizon))
    try:
        for sequence in itertools.product(joint_actions, repeat=horizon):
            sumo_api.simulation.loadState(str(state_path))
            total_arrived = 0
            for candidate in sequence:
                _, arrived = _advance_joint_interval(
                    sumo_api=sumo_api,
                    infos=infos,
                    actions=candidate,
                    control_interval_sec=control_interval_sec,
                )
                total_arrived += arrived
            cost = _system_cost(_read_all_states(sumo_api, infos, duration_sec=duration_sec))
            cost -= 0.40 * float(total_arrived)
            if cost < best_cost:
                best_cost = cost
                best_actions = dict(sequence[0])
    finally:
        sumo_api.simulation.loadState(str(state_path))
    if best_actions is None:
        return {tls_id: 0 for tls_id in tls_ids}
    return best_actions


def evaluate_policy(
    *,
    sumo_api: Any,
    build: ScenarioBuild,
    policy: str,
    models: FittedModels | None,
    fewshot_bias: np.ndarray | None,
    selector_method: str | None,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    oracle_horizon: int,
) -> dict[str, Any]:
    _start_sumo(sumo_api, build.config_path, seed)
    infos = {tls_id: _tls_info(sumo_api, tls_id, build) for tls_id in CONTROLLED_TLS}
    interval_idx = 0
    costs: list[float] = []
    queues: list[float] = []
    ns_shares: list[float] = []
    departed_total = 0
    arrived_total = 0
    state_path = build.scenario_dir / f"oracle_state_{policy}.xml"
    try:
        while float(sumo_api.simulation.getTime()) < duration_sec:
            states = _read_all_states(sumo_api, infos, duration_sec=duration_sec)
            if policy == "oracle_mpc":
                actions = _oracle_actions(
                    sumo_api=sumo_api,
                    infos=infos,
                    state_path=state_path,
                    duration_sec=duration_sec,
                    control_interval_sec=control_interval_sec,
                    oracle_horizon=oracle_horizon,
                )
            else:
                actions = _policy_actions(
                    policy=policy,
                    states=states,
                    interval_idx=interval_idx,
                    models=models,
                    control_interval_sec=control_interval_sec,
                    fewshot_bias=fewshot_bias,
                    selector_method=selector_method,
                )
            departed, arrived = _advance_joint_interval(
                sumo_api=sumo_api,
                infos=infos,
                actions=actions,
                control_interval_sec=control_interval_sec,
            )
            departed_total += departed
            arrived_total += arrived
            if float(sumo_api.simulation.getTime()) >= warmup_sec:
                post = _read_all_states(sumo_api, infos, duration_sec=duration_sec)
                costs.append(_system_cost(post))
                queues.append(float(sum(np.sum(state.q) for state in post.values())))
                ns_shares.append(float(np.mean([action == 0 for action in actions.values()])))
            interval_idx += 1
        remaining = int(sumo_api.vehicle.getIDCount())
    finally:
        _safe_close(sumo_api)
    return {
        "ok": True,
        "policy": policy,
        "mean_cost": float(np.mean(costs)) if costs else float("nan"),
        "mean_queue": float(np.mean(queues)) if queues else float("nan"),
        "p90_queue": float(np.quantile(queues, 0.90)) if queues else float("nan"),
        "throughput_ratio": float(arrived_total / max(departed_total, 1)),
        "departed": int(departed_total),
        "arrived": int(arrived_total),
        "remaining": int(remaining),
        "ns_action_share": float(np.mean(ns_shares)) if ns_shares else float("nan"),
        "intervals": int(len(costs)),
        "selector_method": selector_method if policy == "cfcmt_fewshot_selector_mpc" else None,
    }


def _aggregate(targets: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted(set.intersection(*(set(target["policy_metrics"]) for target in targets)))
    out: dict[str, Any] = {}
    for policy in policies:
        rows = [target["policy_metrics"][policy] for target in targets if policy in target["policy_metrics"]]
        out[policy] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in ("mean_cost", "mean_queue", "p90_queue", "throughput_ratio", "ns_action_share")
        }
    return out


def run_experiment(
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    duration_sec: float = 900.0,
    collect_duration_sec: float = 900.0,
    fewshot_duration_sec: float = 450.0,
    control_interval_sec: int = 15,
    warmup_sec: float = 90.0,
    seed: int = 41,
    max_scenarios: int = 0,
    policies: tuple[str, ...] = (
        "fixed_offset",
        "local_pressure",
        "spillback_pressure",
        "sim_generic_mpc",
        "sim_source_avg_mpc",
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
        "cfcmt_fewshot_selector_mpc",
    ),
    oracle_targets: int = 1,
    oracle_horizon: int = 1,
    dense_l2: float = 22.0,
    mechanism_l2: float = 6.0,
) -> dict[str, Any]:
    if not shutil.which("sumo") or not shutil.which("netgenerate"):
        raise RuntimeError("SUMO binaries are required")
    specs = _scenario_specs()
    if max_scenarios > 0:
        specs = specs[:max_scenarios]
    if len(specs) < 2:
        raise ValueError("at least two scenarios are required")
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
        fewshot = collect_transitions(
            sumo_api=sumo_api,
            build=target,
            duration_sec=fewshot_duration_sec,
            control_interval_sec=control_interval_sec,
            warmup_sec=min(warmup_sec, fewshot_duration_sec / 3.0),
            seed=seed + 5000 + target_idx * 97,
        )
        fewshot_bias = fit_fewshot_bias(fewshot, models, control_interval_sec=control_interval_sec)
        selector_method, selector_scores = select_fewshot_model(
            fewshot,
            models,
            control_interval_sec=control_interval_sec,
            fewshot_bias=fewshot_bias,
        )
        target_policies = list(policies)
        if target_idx < oracle_targets and "oracle_mpc" not in target_policies:
            target_policies.append("oracle_mpc")
        policy_metrics: dict[str, Any] = {}
        for policy in target_policies:
            use_models = (
                models
                if policy
                in {
                    "sim_source_avg_mpc",
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
                fewshot_bias=fewshot_bias if policy in {"cfcmt_fewshot_bias_mpc", "cfcmt_fewshot_selector_mpc"} else None,
                selector_method=selector_method if policy == "cfcmt_fewshot_selector_mpc" else None,
                duration_sec=duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                seed=seed + 9000 + target_idx * 401 + _policy_seed_offset(policy),
                oracle_horizon=oracle_horizon,
            )
        targets.append(
            {
                "target": target.spec.name,
                "source_scenarios": source_names,
                "vehicles": target.vehicles,
                "source_transition_count": int(train.size),
                "fewshot_transition_count": int(fewshot.size),
                "fewshot_bias": fewshot_bias.tolist(),
                "fewshot_selector_method": selector_method,
                "fewshot_selector_scores": selector_scores,
                "policy_metrics": policy_metrics,
            }
        )

    return {
        "experiment": "traffic_signal_sumo_phase2_multi_intersection",
        "setting": {
            "simulator": "SUMO/libsumo",
            "network": "generated_4x4_grid_four_center_tls",
            "controlled_tls": list(CONTROLLED_TLS),
            "transfer_split": "leave_one_scenario_out",
            "duration_sec": float(duration_sec),
            "collect_duration_sec": float(collect_duration_sec),
            "fewshot_duration_sec": float(fewshot_duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
            "oracle_targets": int(oracle_targets),
            "oracle_horizon": int(oracle_horizon),
            "dense_l2": float(dense_l2),
            "mechanism_l2": float(mechanism_l2),
            "policies": list(policies),
        },
        "scenario_builds": [
            {
                "name": build.spec.name,
                "scenario_dir": str(build.scenario_dir),
                "vehicles": int(build.vehicles),
                "scenario_summary": build.scenario_summary.tolist(),
            }
            for build in builds
        ],
        "source_transition_counts": {name: int(ts.size) for name, ts in source_transitions.items()},
        "aggregate": _aggregate(targets),
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
        row = {
            "target": target["target"],
            "best_policy": best[0],
            "best_cost": best[1]["mean_cost"],
            "fixed_offset": target["policy_metrics"].get("fixed_offset", {}).get("mean_cost", float("nan")),
            "spillback_pressure": target["policy_metrics"].get("spillback_pressure", {}).get("mean_cost", float("nan")),
            "sim_generic_mpc": target["policy_metrics"].get("sim_generic_mpc", {}).get("mean_cost", float("nan")),
            "sim_source_avg_mpc": target["policy_metrics"].get("sim_source_avg_mpc", {}).get("mean_cost", float("nan")),
            "sim_mpc": target["policy_metrics"].get("sim_mpc", {}).get("mean_cost", float("nan")),
            "h2oplus_dense_mpc": target["policy_metrics"].get("h2oplus_dense_mpc", {}).get("mean_cost", float("nan")),
            "cfcmt_global_mpc": target["policy_metrics"].get("cfcmt_global_mpc", {}).get("mean_cost", float("nan")),
            "cfcmt_trust_mpc": target["policy_metrics"].get("cfcmt_trust_mpc", {}).get("mean_cost", float("nan")),
            "cfcmt_fewshot_selector_mpc": target["policy_metrics"].get("cfcmt_fewshot_selector_mpc", {}).get("mean_cost", float("nan")),
            "oracle_mpc": target["policy_metrics"].get("oracle_mpc", {}).get("mean_cost", float("nan")),
            "selector": target.get("fewshot_selector_method", ""),
        }
        target_rows.append(row)
    lines = [
        "# Traffic Signal SUMO Phase 2",
        "",
        "This is a multi-intersection microscopic SUMO benchmark. It is still synthetic, but it uses OD-style routes that cross multiple controlled signals.",
        "",
        "Protocol: generated 4x4 grid, four center intersections controlled by libsumo, leave-one-scenario-out transfer. The optional oracle row is a short-horizon SUMO counterfactual MPC computed with saveState/loadState.",
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
                "fixed_offset",
                "spillback_pressure",
                "sim_generic_mpc",
                "sim_source_avg_mpc",
                "sim_mpc",
                "h2oplus_dense_mpc",
                "cfcmt_global_mpc",
                "cfcmt_trust_mpc",
                "cfcmt_fewshot_selector_mpc",
                "oracle_mpc",
                "selector",
            ],
        ),
        "",
        "## Notes",
        "",
        "- Simulator information budget: `sim_generic_mpc` uses a fixed generic demand prior, `sim_source_avg_mpc` uses source-domain average demand summaries, and `sim_mpc` uses the target static demand/bottleneck summary.",
        "- The aggregate excludes oracle unless it was evaluated for every target; by default oracle is run on the first target only as a counterfactual sanity check.",
        "- The oracle row is a short-horizon myopic counterfactual controller, not a full-horizon performance upper bound.",
        "- Few-shot selector uses passive target transitions only; it does not inspect target policy rollouts.",
        "- This benchmark tests network-level closed-loop behavior but does not yet use a real-world TSC benchmark topology.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_policies(raw: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in raw.split(",") if item.strip())
    valid = {
        "fixed_offset",
        "local_pressure",
        "spillback_pressure",
        "sim_generic_mpc",
        "sim_source_avg_mpc",
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
        "cfcmt_fewshot_selector_mpc",
        "oracle_mpc",
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
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--policies", type=_parse_policies, default=None)
    parser.add_argument("--oracle-targets", type=int, default=1)
    parser.add_argument("--oracle-horizon", type=int, default=1)
    parser.add_argument("--dense-l2", type=float, default=22.0)
    parser.add_argument("--mechanism-l2", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    default_policies = (
        "fixed_offset",
        "local_pressure",
        "spillback_pressure",
        "sim_generic_mpc",
        "sim_source_avg_mpc",
        "sim_mpc",
        "h2oplus_dense_mpc",
        "cfcmt_global_mpc",
        "cfcmt_trust_mpc",
        "cfcmt_fewshot_bias_mpc",
        "cfcmt_fewshot_selector_mpc",
    )
    result = run_experiment(
        out_root=args.out_root,
        duration_sec=args.duration_sec,
        collect_duration_sec=args.collect_duration_sec,
        fewshot_duration_sec=args.fewshot_duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        policies=args.policies or default_policies,
        oracle_targets=args.oracle_targets,
        oracle_horizon=args.oracle_horizon,
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
