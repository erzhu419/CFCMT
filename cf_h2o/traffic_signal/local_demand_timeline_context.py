"""Route-projected, label-free local demand arrivals for each traffic signal."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import (
    _flow_rate,
    _resolve_cfg_paths,
    demand_input_sha256,
)
from cf_h2o.traffic_signal.sumo_static_inputs import parse_sumo_time_seconds


LOCAL_DEMAND_PROTOCOL = "sumo-route-projected-local-demand-timeline-v1"
LOCAL_DEMAND_FEATURE_NAMES = (
    "local_arrivals_prev_300_per_in_lane",
    "local_arrivals_next_300_per_in_lane",
    "local_arrivals_next_450_per_in_lane",
    "local_arrivals_next_900_per_in_lane",
    "local_arrivals_next_prev_300_balance",
    "local_arrivals_next_450_to_tls_mean_ratio",
    "local_arrivals_next_900_300bin_cv",
    "local_arrivals_share_network_next_450",
    "neighbor_arrivals_next_450_ratio",
    "local_arrivals_remaining_fraction",
)


def _add_event(
    values: np.ndarray, *, time_sec: float, begin_sec: float, bin_sec: int
) -> None:
    offset = float(time_sec) - float(begin_sec)
    if 0.0 <= offset < values.size * int(bin_sec):
        values[min(int(offset // bin_sec), values.size - 1)] += 1.0


def _add_uniform(
    values: np.ndarray,
    *,
    start_sec: float,
    stop_sec: float,
    rate_per_sec: float,
    begin_sec: float,
    bin_sec: int,
) -> None:
    horizon_stop = begin_sec + values.size * int(bin_sec)
    start = max(float(start_sec), float(begin_sec))
    stop = min(float(stop_sec), float(horizon_stop))
    if stop <= start or rate_per_sec <= 0.0:
        return
    first = max(int(math.floor((start - begin_sec) / bin_sec)), 0)
    last = min(int(math.ceil((stop - begin_sec) / bin_sec)), values.size)
    for index in range(first, last):
        left = max(start, begin_sec + index * bin_sec)
        right = min(stop, begin_sec + (index + 1) * bin_sec)
        values[index] += max(right - left, 0.0) * rate_per_sec


def _window_sum(
    values: np.ndarray,
    *,
    start_sec: float,
    stop_sec: float,
    begin_sec: float,
    bin_sec: int,
) -> float:
    if stop_sec <= start_sec:
        return 0.0
    horizon_stop = begin_sec + values.size * int(bin_sec)
    start = max(float(start_sec), float(begin_sec))
    stop = min(float(stop_sec), float(horizon_stop))
    if stop <= start:
        return 0.0
    first = max(int(math.floor((start - begin_sec) / bin_sec)), 0)
    last = min(int(math.ceil((stop - begin_sec) / bin_sec)), values.size)
    total = 0.0
    for index in range(first, last):
        left = max(start, begin_sec + index * bin_sec)
        right = min(stop, begin_sec + (index + 1) * bin_sec)
        total += float(values[index]) * max(right - left, 0.0) / float(bin_sec)
    return total


def _route_edges(
    element: ET.Element, route_definitions: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    route_id = str(element.attrib.get("route", ""))
    if route_id in route_definitions:
        return route_definitions[route_id]
    inline = element.find("route")
    if inline is not None:
        return tuple(str(inline.attrib.get("edges", "")).split())
    return ()


@dataclass(frozen=True)
class LocalDemandTimelineContext:
    protocol: str
    horizon_sec: int
    begin_sec: float
    bin_sec: int
    input_sha256: str
    feature_names: tuple[str, ...]
    tls_arrivals: Mapping[str, tuple[float, ...]]
    incoming_lane_count: Mapping[str, int]
    tls_neighbors: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        expected_bins = int(math.ceil(self.horizon_sec / self.bin_sec))
        tls_ids = set(self.tls_arrivals)
        if (
            self.protocol != LOCAL_DEMAND_PROTOCOL
            or self.horizon_sec <= 0
            or self.bin_sec <= 0
            or len(self.input_sha256) != 64
            or self.feature_names != LOCAL_DEMAND_FEATURE_NAMES
            or not tls_ids
            or set(self.incoming_lane_count) != tls_ids
            or set(self.tls_neighbors) != tls_ids
        ):
            raise ValueError("invalid local demand timeline context")
        for tls, raw in self.tls_arrivals.items():
            values = np.asarray(raw, dtype=float)
            if (
                values.shape != (expected_bins,)
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
                or int(self.incoming_lane_count[tls]) <= 0
                or any(neighbor not in tls_ids for neighbor in self.tls_neighbors[tls])
            ):
                raise ValueError("invalid local demand timeline row")
        if sum(sum(values) for values in self.tls_arrivals.values()) <= 0.0:
            raise ValueError("local demand timeline has no route-projected arrivals")

    def vector_at(self, tls_id: str, time_sec: float) -> np.ndarray:
        tls = str(tls_id)
        if tls not in self.tls_arrivals:
            raise KeyError(f"unknown local demand TLS: {tls}")
        time_value = float(time_sec)
        if not np.isfinite(time_value):
            raise ValueError("local demand time must be finite")
        values = np.asarray(self.tls_arrivals[tls], dtype=float)
        lane_count = max(int(self.incoming_lane_count[tls]), 1)

        def local(start: float, stop: float) -> float:
            return _window_sum(
                values,
                start_sec=start,
                stop_sec=stop,
                begin_sec=self.begin_sec,
                bin_sec=self.bin_sec,
            )

        prev_300 = local(time_value - 300.0, time_value)
        next_300 = local(time_value, time_value + 300.0)
        next_450 = local(time_value, time_value + 450.0)
        next_900 = local(time_value, time_value + 900.0)
        total_local = float(np.sum(values))
        mean_450 = total_local * 450.0 / float(self.horizon_sec)
        segments = np.asarray(
            [
                local(time_value + offset, time_value + offset + 300.0)
                for offset in (0.0, 300.0, 600.0)
            ],
            dtype=float,
        )
        segment_mean = float(np.mean(segments))
        network_next_450 = sum(
            _window_sum(
                np.asarray(raw, dtype=float),
                start_sec=time_value,
                stop_sec=time_value + 450.0,
                begin_sec=self.begin_sec,
                bin_sec=self.bin_sec,
            )
            for raw in self.tls_arrivals.values()
        )
        neighbor_next_450 = sum(
            _window_sum(
                np.asarray(self.tls_arrivals[neighbor], dtype=float),
                start_sec=time_value,
                stop_sec=time_value + 450.0,
                begin_sec=self.begin_sec,
                bin_sec=self.bin_sec,
            )
            for neighbor in self.tls_neighbors[tls]
        )
        remaining = local(time_value, self.begin_sec + self.horizon_sec)
        vector = np.asarray(
            [
                prev_300 / lane_count,
                next_300 / lane_count,
                next_450 / lane_count,
                next_900 / lane_count,
                (next_300 - prev_300) / max(next_300 + prev_300, 1.0),
                next_450 / max(mean_450, 1.0),
                float(np.std(segments) / segment_mean)
                if segment_mean > 0.0
                else 0.0,
                next_450 / max(network_next_450, 1.0),
                neighbor_next_450 / max(next_450, 1.0),
                remaining / max(total_local, 1.0),
            ],
            dtype=float,
        )
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("local demand feature vector changed")
        return vector

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "horizon_sec": self.horizon_sec,
            "begin_sec": self.begin_sec,
            "bin_sec": self.bin_sec,
            "input_sha256": self.input_sha256,
            "feature_names": list(self.feature_names),
            "tls_arrivals": {
                tls: list(values) for tls, values in self.tls_arrivals.items()
            },
            "incoming_lane_count": dict(self.incoming_lane_count),
            "tls_neighbors": {
                tls: list(values) for tls, values in self.tls_neighbors.items()
            },
        }


def build_local_demand_timeline_context(
    sumocfg: Path, *, horizon_sec: int = 3600, bin_sec: int = 10
) -> LocalDemandTimelineContext:
    sumocfg = Path(sumocfg).resolve()
    cfg = ET.parse(sumocfg).getroot()
    begin_element = cfg.find(".//time/begin")
    begin_sec = (
        float(begin_element.attrib.get("value", 0.0))
        if begin_element is not None
        else 0.0
    )
    net_files = _resolve_cfg_paths(sumocfg, "net-file")
    if len(net_files) != 1:
        raise ValueError("local demand context requires one net file")
    net = ET.parse(net_files[0]).getroot()
    tls_ids = {str(element.attrib["id"]) for element in net.findall(".//tlLogic")}
    edges = {
        str(edge.attrib["id"]): edge
        for edge in net.findall(".//edge")
        if not str(edge.attrib.get("function", ""))
        and "from" in edge.attrib
        and "to" in edge.attrib
    }
    travel_time = {}
    for edge_id, edge in edges.items():
        lanes = list(edge.findall("lane"))
        if lanes:
            travel_time[edge_id] = float(
                np.mean(
                    [
                        float(lane.attrib.get("length", 0.0))
                        / max(float(lane.attrib.get("speed", 0.0)), 0.1)
                        for lane in lanes
                    ]
                )
            )
        else:
            travel_time[edge_id] = 0.0
    incoming_lane_count = {
        tls: max(
            sum(
                len(list(edge.findall("lane")))
                for edge in edges.values()
                if str(edge.attrib["to"]) == tls
            ),
            1,
        )
        for tls in tls_ids
    }
    neighbors = {tls: set() for tls in tls_ids}
    for edge in edges.values():
        left = str(edge.attrib["from"])
        right = str(edge.attrib["to"])
        if left in tls_ids and right in tls_ids and left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    bins = int(math.ceil(horizon_sec / bin_sec))
    arrivals = {tls: np.zeros(bins, dtype=float) for tls in tls_ids}

    def route_arrivals(route: Sequence[str]) -> list[tuple[str, float]]:
        elapsed = 0.0
        rows = []
        for edge_id in route:
            if edge_id not in edges:
                continue
            elapsed += travel_time.get(edge_id, 0.0)
            destination = str(edges[edge_id].attrib["to"])
            if destination in tls_ids:
                rows.append((destination, elapsed))
        return rows

    for route_file in _resolve_cfg_paths(sumocfg, "route-files"):
        root = ET.parse(route_file).getroot()
        definitions = {
            str(route.attrib["id"]): tuple(str(route.attrib.get("edges", "")).split())
            for route in root.findall("route")
            if "id" in route.attrib
        }
        for element in root:
            if element.tag not in {"vehicle", "trip", "flow"}:
                continue
            route = _route_edges(element, definitions)
            projected = route_arrivals(route)
            if not projected:
                continue
            if element.tag == "flow":
                start = parse_sumo_time_seconds(
                    element.attrib.get("begin", begin_sec)
                )
                stop = parse_sumo_time_seconds(element.attrib.get("end", start))
                rate = _flow_rate(element, start, stop)
                for tls, offset in projected:
                    _add_uniform(
                        arrivals[tls],
                        start_sec=start + offset,
                        stop_sec=stop + offset,
                        rate_per_sec=rate,
                        begin_sec=begin_sec,
                        bin_sec=bin_sec,
                    )
            else:
                depart = parse_sumo_time_seconds(
                    element.attrib.get("depart", begin_sec)
                )
                for tls, offset in projected:
                    _add_event(
                        arrivals[tls],
                        time_sec=depart + offset,
                        begin_sec=begin_sec,
                        bin_sec=bin_sec,
                    )
    return LocalDemandTimelineContext(
        protocol=LOCAL_DEMAND_PROTOCOL,
        horizon_sec=int(horizon_sec),
        begin_sec=float(begin_sec),
        bin_sec=int(bin_sec),
        input_sha256=demand_input_sha256(sumocfg),
        feature_names=LOCAL_DEMAND_FEATURE_NAMES,
        tls_arrivals={
            tls: tuple(float(value) for value in values)
            for tls, values in sorted(arrivals.items())
        },
        incoming_lane_count=incoming_lane_count,
        tls_neighbors={
            tls: tuple(sorted(values)) for tls, values in sorted(neighbors.items())
        },
    )
