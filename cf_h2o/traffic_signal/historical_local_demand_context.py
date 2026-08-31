"""Causal-prefix local demand features for traffic-signal transfer."""

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
from cf_h2o.traffic_signal.local_demand_timeline_context import (
    _add_event,
    _add_uniform,
    _route_edges,
    _window_sum,
)
from cf_h2o.traffic_signal.sumo_static_inputs import parse_sumo_time_seconds


HISTORICAL_LOCAL_DEMAND_PROTOCOL = "sumo-route-projected-historical-prefix-v1"
HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES = (
    "local_arrivals_prev_60_per_in_lane",
    "local_arrivals_prev_180_per_in_lane",
    "local_arrivals_prev_300_per_in_lane",
    "local_arrivals_prev_600_per_in_lane",
    "local_arrivals_prev_vs_prior_300_balance",
    "local_arrivals_prev_900_300bin_cv",
    "local_arrivals_prev_300_to_prefix_mean_ratio",
    "local_arrivals_share_network_prev_300",
    "neighbor_arrivals_prev_300_ratio",
    "local_arrivals_cumulative_per_in_lane",
)


@dataclass(frozen=True)
class HistoricalLocalDemandContext:
    """Route-projected arrivals summarized only through the decision time."""

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
            self.protocol != HISTORICAL_LOCAL_DEMAND_PROTOCOL
            or self.horizon_sec <= 0
            or self.bin_sec <= 0
            or len(self.input_sha256) != 64
            or self.feature_names != HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES
            or not tls_ids
            or set(self.incoming_lane_count) != tls_ids
            or set(self.tls_neighbors) != tls_ids
        ):
            raise ValueError("invalid historical local-demand context")
        for tls_id, raw in self.tls_arrivals.items():
            values = np.asarray(raw, dtype=float)
            if (
                values.shape != (expected_bins,)
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
                or int(self.incoming_lane_count[tls_id]) <= 0
                or any(neighbor not in tls_ids for neighbor in self.tls_neighbors[tls_id])
            ):
                raise ValueError("invalid historical local-demand row")
        if sum(sum(values) for values in self.tls_arrivals.values()) <= 0.0:
            raise ValueError("historical local-demand context has no arrivals")

    def vector_at(self, tls_id: str, time_sec: float) -> np.ndarray:
        tls = str(tls_id)
        if tls not in self.tls_arrivals:
            raise KeyError(f"unknown historical local-demand TLS: {tls}")
        decision_time = float(time_sec)
        if not np.isfinite(decision_time):
            raise ValueError("historical local-demand time must be finite")
        values = np.asarray(self.tls_arrivals[tls], dtype=float)
        lane_count = max(int(self.incoming_lane_count[tls]), 1)

        def past(window_sec: float, *, end_sec: float = decision_time) -> float:
            return _window_sum(
                values,
                start_sec=end_sec - float(window_sec),
                stop_sec=end_sec,
                begin_sec=self.begin_sec,
                bin_sec=self.bin_sec,
            )

        prev_60 = past(60.0)
        prev_180 = past(180.0)
        prev_300 = past(300.0)
        prev_600 = past(600.0)
        prior_300 = past(300.0, end_sec=decision_time - 300.0)
        segments = np.asarray(
            [
                past(300.0, end_sec=decision_time - offset)
                for offset in (0.0, 300.0, 600.0)
            ],
            dtype=float,
        )
        segment_mean = float(np.mean(segments))
        elapsed = max(decision_time - self.begin_sec, 0.0)
        cumulative = _window_sum(
            values,
            start_sec=self.begin_sec,
            stop_sec=decision_time,
            begin_sec=self.begin_sec,
            bin_sec=self.bin_sec,
        )
        prefix_mean_300 = cumulative * 300.0 / max(elapsed, 300.0)
        network_prev_300 = sum(
            _window_sum(
                np.asarray(raw, dtype=float),
                start_sec=decision_time - 300.0,
                stop_sec=decision_time,
                begin_sec=self.begin_sec,
                bin_sec=self.bin_sec,
            )
            for raw in self.tls_arrivals.values()
        )
        neighbor_prev_300 = sum(
            _window_sum(
                np.asarray(self.tls_arrivals[neighbor], dtype=float),
                start_sec=decision_time - 300.0,
                stop_sec=decision_time,
                begin_sec=self.begin_sec,
                bin_sec=self.bin_sec,
            )
            for neighbor in self.tls_neighbors[tls]
        )
        vector = np.asarray(
            [
                prev_60 / lane_count,
                prev_180 / lane_count,
                prev_300 / lane_count,
                prev_600 / lane_count,
                (prev_300 - prior_300) / max(prev_300 + prior_300, 1.0),
                float(np.std(segments) / segment_mean)
                if segment_mean > 0.0
                else 0.0,
                prev_300 / max(prefix_mean_300, 1.0),
                prev_300 / max(network_prev_300, 1.0),
                neighbor_prev_300 / max(prev_300, 1.0),
                cumulative / lane_count,
            ],
            dtype=float,
        )
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("historical local-demand feature vector changed")
        return vector

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "horizon_sec": self.horizon_sec,
            "begin_sec": self.begin_sec,
            "bin_sec": self.bin_sec,
            "input_sha256": self.input_sha256,
            "feature_names": list(self.feature_names),
            "information_cutoff": "strictly_no_later_than_runtime_time_sec",
            "incoming_lane_count": dict(self.incoming_lane_count),
            "tls_neighbors": {
                tls_id: list(values)
                for tls_id, values in self.tls_neighbors.items()
            },
        }


def build_historical_local_demand_context(
    sumocfg: Path, *, horizon_sec: int = 3600, bin_sec: int = 10
) -> HistoricalLocalDemandContext:
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
        raise ValueError("historical local demand requires one net file")
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
        travel_time[edge_id] = (
            float(
                np.mean(
                    [
                        float(lane.attrib.get("length", 0.0))
                        / max(float(lane.attrib.get("speed", 0.0)), 0.1)
                        for lane in lanes
                    ]
                )
            )
            if lanes
            else 0.0
        )
    incoming_lane_count = {
        tls_id: max(
            sum(
                len(list(edge.findall("lane")))
                for edge in edges.values()
                if str(edge.attrib["to"]) == tls_id
            ),
            1,
        )
        for tls_id in tls_ids
    }
    neighbors = {tls_id: set() for tls_id in tls_ids}
    for edge in edges.values():
        left = str(edge.attrib["from"])
        right = str(edge.attrib["to"])
        if left in tls_ids and right in tls_ids and left != right:
            neighbors[left].add(right)
            neighbors[right].add(left)
    bins = int(math.ceil(horizon_sec / bin_sec))
    arrivals = {tls_id: np.zeros(bins, dtype=float) for tls_id in tls_ids}

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
            projected = route_arrivals(_route_edges(element, definitions))
            if not projected:
                continue
            if element.tag == "flow":
                start = parse_sumo_time_seconds(
                    element.attrib.get("begin", begin_sec)
                )
                stop = parse_sumo_time_seconds(element.attrib.get("end", start))
                rate = _flow_rate(element, start, stop)
                for tls_id, offset in projected:
                    _add_uniform(
                        arrivals[tls_id],
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
                for tls_id, offset in projected:
                    _add_event(
                        arrivals[tls_id],
                        time_sec=depart + offset,
                        begin_sec=begin_sec,
                        bin_sec=bin_sec,
                    )
    return HistoricalLocalDemandContext(
        protocol=HISTORICAL_LOCAL_DEMAND_PROTOCOL,
        horizon_sec=int(horizon_sec),
        begin_sec=float(begin_sec),
        bin_sec=int(bin_sec),
        input_sha256=demand_input_sha256(sumocfg),
        feature_names=HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES,
        tls_arrivals={
            tls_id: tuple(float(value) for value in values)
            for tls_id, values in sorted(arrivals.items())
        },
        incoming_lane_count=incoming_lane_count,
        tls_neighbors={
            tls_id: tuple(sorted(values))
            for tls_id, values in sorted(neighbors.items())
        },
    )
