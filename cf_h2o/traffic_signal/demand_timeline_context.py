"""Label-free temporal demand and local TLS topology context."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import (
    _flow_rate,
    _resolve_cfg_paths,
    demand_input_sha256,
)
from cf_h2o.traffic_signal.sumo_static_inputs import parse_sumo_time_seconds


TEMPORAL_DEMAND_PROTOCOL = "sumo-static-demand-timeline-context-v1"
TEMPORAL_DEMAND_FEATURE_NAMES = (
    "scheduled_prev_300_per_tls",
    "scheduled_next_300_per_tls",
    "scheduled_next_450_per_tls",
    "scheduled_next_900_per_tls",
    "scheduled_next_prev_300_balance",
    "scheduled_next_450_to_global_ratio",
    "scheduled_next_900_300bin_cv",
    "scheduled_remaining_fraction",
)
TLS_TOPOLOGY_PROTOCOL = "sumo-static-tls-local-topology-context-v1"
TLS_TOPOLOGY_FEATURE_NAMES = (
    "tls_incoming_edge_count_norm",
    "tls_outgoing_edge_count_norm",
    "tls_controlled_link_count_norm",
    "tls_incoming_lane_count_norm",
    "tls_incoming_speed_norm",
    "tls_incoming_length_norm",
    "tls_graph_degree_norm",
    "tls_graph_closeness",
    "tls_phase_count_norm",
    "tls_mean_green_fraction",
)


def _window_sum(values: np.ndarray, *, start: float, stop: float, bin_sec: int) -> float:
    if stop <= start or values.size == 0:
        return 0.0
    horizon = values.size * int(bin_sec)
    clipped_start = max(float(start), 0.0)
    clipped_stop = min(float(stop), float(horizon))
    if clipped_stop <= clipped_start:
        return 0.0
    first = max(int(math.floor(clipped_start / bin_sec)), 0)
    last = min(int(math.ceil(clipped_stop / bin_sec)), values.size)
    total = 0.0
    for index in range(first, last):
        left = max(clipped_start, index * bin_sec)
        right = min(clipped_stop, (index + 1) * bin_sec)
        if right > left:
            total += float(values[index]) * (right - left) / float(bin_sec)
    return total


@dataclass(frozen=True)
class TemporalDemandContext:
    protocol: str
    horizon_sec: int
    bin_sec: int
    tls_count: int
    input_sha256: str
    expected_departures: tuple[float, ...]
    feature_names: tuple[str, ...] = TEMPORAL_DEMAND_FEATURE_NAMES

    def __post_init__(self) -> None:
        expected_bins = int(math.ceil(self.horizon_sec / self.bin_sec))
        values = np.asarray(self.expected_departures, dtype=float)
        if (
            self.protocol != TEMPORAL_DEMAND_PROTOCOL
            or self.horizon_sec <= 0
            or self.bin_sec <= 0
            or self.tls_count <= 0
            or len(self.input_sha256) != 64
            or self.feature_names != TEMPORAL_DEMAND_FEATURE_NAMES
            or values.shape != (expected_bins,)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or float(np.sum(values)) <= 0.0
        ):
            raise ValueError("invalid temporal demand context")

    def vector_at(self, time_sec: float) -> np.ndarray:
        values = np.asarray(self.expected_departures, dtype=float)
        time_value = float(time_sec)
        if not np.isfinite(time_value):
            raise ValueError("temporal demand time must be finite")
        prev_300 = _window_sum(
            values, start=time_value - 300.0, stop=time_value, bin_sec=self.bin_sec
        )
        next_300 = _window_sum(
            values, start=time_value, stop=time_value + 300.0, bin_sec=self.bin_sec
        )
        next_450 = _window_sum(
            values, start=time_value, stop=time_value + 450.0, bin_sec=self.bin_sec
        )
        next_900 = _window_sum(
            values, start=time_value, stop=time_value + 900.0, bin_sec=self.bin_sec
        )
        total = float(np.sum(values))
        global_450 = total * 450.0 / float(self.horizon_sec)
        segments = np.asarray(
            [
                _window_sum(
                    values,
                    start=time_value + offset,
                    stop=time_value + offset + 300.0,
                    bin_sec=self.bin_sec,
                )
                for offset in (0.0, 300.0, 600.0)
            ],
            dtype=float,
        )
        segment_mean = float(np.mean(segments))
        remaining = _window_sum(
            values,
            start=max(time_value, 0.0),
            stop=float(self.horizon_sec),
            bin_sec=self.bin_sec,
        )
        vector = np.asarray(
            [
                prev_300 / self.tls_count,
                next_300 / self.tls_count,
                next_450 / self.tls_count,
                next_900 / self.tls_count,
                (next_300 - prev_300) / max(next_300 + prev_300, 1.0),
                next_450 / max(global_450, 1.0),
                float(np.std(segments) / segment_mean)
                if segment_mean > 0.0
                else 0.0,
                remaining / total,
            ],
            dtype=float,
        )
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("temporal demand feature vector changed")
        return vector

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_temporal_demand_context(
    sumocfg: Path, *, horizon_sec: int = 3600, bin_sec: int = 10
) -> TemporalDemandContext:
    sumocfg = Path(sumocfg).resolve()
    if int(horizon_sec) <= 0 or int(bin_sec) <= 0:
        raise ValueError("temporal demand horizon and bin must be positive")
    cfg_root = ET.parse(sumocfg).getroot()
    begin_element = cfg_root.find(".//time/begin")
    window_start = (
        float(begin_element.attrib.get("value", 0.0))
        if begin_element is not None
        else 0.0
    )
    window_stop = window_start + int(horizon_sec)
    net_files = _resolve_cfg_paths(sumocfg, "net-file")
    if len(net_files) != 1:
        raise ValueError("temporal demand context requires one net file")
    net_root = ET.parse(net_files[0]).getroot()
    tls_ids = {str(element.attrib["id"]) for element in net_root.findall(".//tlLogic")}
    if not tls_ids:
        tls_ids = {
            str(element.attrib["id"])
            for element in net_root.findall(".//junction")
            if str(element.attrib.get("type", "")).startswith("traffic_light")
        }
    bins = np.zeros(int(math.ceil(horizon_sec / bin_sec)), dtype=float)
    for route_file in _resolve_cfg_paths(sumocfg, "route-files"):
        route_root = ET.parse(route_file).getroot()
        for element in route_root:
            if element.tag == "flow":
                begin = parse_sumo_time_seconds(
                    element.attrib.get("begin", window_start)
                )
                end = parse_sumo_time_seconds(element.attrib.get("end", begin))
                start = max(begin, window_start)
                stop = min(end, window_stop)
                rate = _flow_rate(element, begin, end)
                first = max(int(math.floor((start - window_start) / bin_sec)), 0)
                last = min(
                    int(math.ceil((stop - window_start) / bin_sec)), bins.size
                )
                for index in range(first, last):
                    bin_start = window_start + index * bin_sec
                    bin_stop = bin_start + bin_sec
                    overlap = max(
                        min(stop, bin_stop) - max(start, bin_start), 0.0
                    )
                    bins[index] += overlap * rate
            elif element.tag in {"vehicle", "trip"}:
                depart = parse_sumo_time_seconds(
                    element.attrib.get("depart", window_start)
                )
                if window_start <= depart < window_stop:
                    index = min(
                        int((depart - window_start) // bin_sec), bins.size - 1
                    )
                    bins[index] += 1.0
    return TemporalDemandContext(
        protocol=TEMPORAL_DEMAND_PROTOCOL,
        horizon_sec=int(horizon_sec),
        bin_sec=int(bin_sec),
        tls_count=len(tls_ids),
        input_sha256=demand_input_sha256(sumocfg),
        expected_departures=tuple(float(value) for value in bins),
    )


@dataclass(frozen=True)
class TLSLocalTopologyContext:
    protocol: str
    input_sha256: str
    feature_names: tuple[str, ...]
    tls_features: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if (
            self.protocol != TLS_TOPOLOGY_PROTOCOL
            or len(self.input_sha256) != 64
            or self.feature_names != TLS_TOPOLOGY_FEATURE_NAMES
            or not self.tls_features
        ):
            raise ValueError("invalid TLS topology context")
        for tls_id, values in self.tls_features.items():
            vector = np.asarray(values, dtype=float)
            if (
                not str(tls_id)
                or vector.shape != (len(self.feature_names),)
                or not np.isfinite(vector).all()
            ):
                raise ValueError("invalid TLS topology feature row")

    def vector_for(self, tls_id: str) -> np.ndarray:
        if str(tls_id) not in self.tls_features:
            raise KeyError(f"unknown TLS topology id: {tls_id}")
        return np.asarray(self.tls_features[str(tls_id)], dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "input_sha256": self.input_sha256,
            "feature_names": list(self.feature_names),
            "tls_features": {
                key: list(values) for key, values in self.tls_features.items()
            },
        }


def _closeness(node: str, adjacency: Mapping[str, set[str]]) -> float:
    distance = {node: 0}
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    reachable = len(distance) - 1
    total_distance = sum(distance.values())
    population = len(adjacency)
    if reachable <= 0 or total_distance <= 0 or population <= 1:
        return 0.0
    return (reachable / total_distance) * (reachable / (population - 1))


def build_tls_local_topology_context(sumocfg: Path) -> TLSLocalTopologyContext:
    sumocfg = Path(sumocfg).resolve()
    net_files = _resolve_cfg_paths(sumocfg, "net-file")
    if len(net_files) != 1:
        raise ValueError("TLS topology context requires one net file")
    root = ET.parse(net_files[0]).getroot()
    tls_logic = {
        str(element.attrib["id"]): element for element in root.findall(".//tlLogic")
    }
    if not tls_logic:
        raise ValueError("TLS topology context found no signal programs")
    tls_ids = set(tls_logic)
    edges = {
        str(edge.attrib["id"]): edge
        for edge in root.findall(".//edge")
        if not str(edge.attrib.get("function", ""))
        and "from" in edge.attrib
        and "to" in edge.attrib
    }
    incoming = {
        tls: {edge_id for edge_id, edge in edges.items() if edge.attrib["to"] == tls}
        for tls in tls_ids
    }
    outgoing = {
        tls: {edge_id for edge_id, edge in edges.items() if edge.attrib["from"] == tls}
        for tls in tls_ids
    }
    adjacency = {tls: set() for tls in tls_ids}
    for edge in edges.values():
        left = str(edge.attrib["from"])
        right = str(edge.attrib["to"])
        if left in tls_ids and right in tls_ids and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    connections = {tls: [] for tls in tls_ids}
    for connection in root.findall(".//connection"):
        tls = str(connection.attrib.get("tl", ""))
        if tls in connections:
            connections[tls].append(connection)
    max_degree = max(max((len(values) for values in adjacency.values()), default=1), 1)
    features = {}
    for tls in sorted(tls_ids):
        incoming_edges = incoming[tls]
        outgoing_edges = outgoing[tls]
        lane_count = 0
        lane_speeds = []
        lane_lengths = []
        for edge_id in incoming_edges:
            for lane in edges[edge_id].findall("lane"):
                lane_count += 1
                lane_speeds.append(float(lane.attrib.get("speed", 0.0)))
                lane_lengths.append(float(lane.attrib.get("length", 0.0)))
        phases = list(tls_logic[tls].findall("phase"))
        green_fractions = []
        for phase in phases:
            state = str(phase.attrib.get("state", ""))
            green_fractions.append(
                sum(character in {"g", "G"} for character in state)
                / max(len(state), 1)
            )
        features[tls] = (
            len(incoming_edges) / 8.0,
            len(outgoing_edges) / 8.0,
            len(connections[tls]) / 16.0,
            lane_count / 16.0,
            float(np.mean(lane_speeds)) / 20.0 if lane_speeds else 0.0,
            float(np.mean(lane_lengths)) / 300.0 if lane_lengths else 0.0,
            len(adjacency[tls]) / max_degree,
            _closeness(tls, adjacency),
            len(phases) / 12.0,
            float(np.mean(green_fractions)) if green_fractions else 0.0,
        )
    return TLSLocalTopologyContext(
        protocol=TLS_TOPOLOGY_PROTOCOL,
        input_sha256=demand_input_sha256(sumocfg),
        feature_names=TLS_TOPOLOGY_FEATURE_NAMES,
        tls_features=features,
    )
