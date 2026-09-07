"""Candidate-specific arrival waves from static SUMO routes and topology."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.sumo_static_inputs import parse_sumo_time_seconds


MOVEMENT_ARRIVAL_PROTOCOL = "sumo-route-projected-movement-arrival-timeline-v1"
MOVEMENT_ARRIVAL_FEATURE_NAMES = (
    "served_arrivals_next_30_per_movement",
    "unserved_arrivals_next_30_per_movement",
    "served_arrivals_next_60_per_movement",
    "unserved_arrivals_next_60_per_movement",
    "served_arrivals_next_120_per_movement",
    "unserved_arrivals_next_120_per_movement",
    "served_arrival_share_next_30",
    "served_arrival_share_next_60",
    "served_arrival_share_next_120",
    "served_arrival_advantage_next_60",
    "served_arrival_advantage_next_120",
    "served_arrival_wave_growth_30_to_60",
)
GREEN_STATES = frozenset({"g", "G"})
Movement = tuple[str, str]


def _add_event(
    values: np.ndarray, *, time_sec: float, begin_sec: float, bin_sec: int
) -> bool:
    offset = float(time_sec) - float(begin_sec)
    if not 0.0 <= offset < values.size * int(bin_sec):
        return False
    values[min(int(offset // bin_sec), values.size - 1)] += 1.0
    return True


def _add_uniform(
    values: np.ndarray,
    *,
    start_sec: float,
    stop_sec: float,
    rate_per_sec: float,
    begin_sec: float,
    bin_sec: int,
) -> float:
    horizon_stop = begin_sec + values.size * int(bin_sec)
    start = max(float(start_sec), float(begin_sec))
    stop = min(float(stop_sec), float(horizon_stop))
    if stop <= start or rate_per_sec <= 0.0:
        return 0.0
    first = max(int(math.floor((start - begin_sec) / bin_sec)), 0)
    last = min(int(math.ceil((stop - begin_sec) / bin_sec)), values.size)
    added = 0.0
    for index in range(first, last):
        left = max(start, begin_sec + index * bin_sec)
        right = min(stop, begin_sec + (index + 1) * bin_sec)
        value = max(right - left, 0.0) * rate_per_sec
        values[index] += value
        added += value
    return added


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
class MovementArrivalTimelineContext:
    protocol: str
    horizon_sec: int
    begin_sec: float
    bin_sec: int
    input_sha256: str
    feature_names: tuple[str, ...]
    movement_arrivals: Mapping[str, Mapping[Movement, tuple[float, ...]]]
    movement_link_indices: Mapping[str, Mapping[Movement, tuple[int, ...]]]
    scheduled_elements: int
    projected_elements: int
    projected_arrival_mass: float
    _movement_order: Mapping[str, tuple[Movement, ...]] = field(
        init=False, repr=False, compare=False
    )
    _arrival_matrix: Mapping[str, np.ndarray] = field(
        init=False, repr=False, compare=False
    )
    _arrival_prefix: Mapping[str, np.ndarray] = field(
        init=False, repr=False, compare=False
    )
    _served_mask_cache: dict[tuple[str, str], np.ndarray] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        expected_bins = int(math.ceil(self.horizon_sec / self.bin_sec))
        if (
            self.protocol != MOVEMENT_ARRIVAL_PROTOCOL
            or self.horizon_sec <= 0
            or self.bin_sec <= 0
            or len(self.input_sha256) != 64
            or self.feature_names != MOVEMENT_ARRIVAL_FEATURE_NAMES
            or set(self.movement_arrivals) != set(self.movement_link_indices)
            or self.scheduled_elements < self.projected_elements
            or self.projected_elements <= 0
            or self.projected_arrival_mass <= 0.0
        ):
            raise ValueError("invalid movement arrival timeline context")
        for tls_id, arrivals in self.movement_arrivals.items():
            links = self.movement_link_indices[tls_id]
            if not arrivals or set(arrivals) != set(links):
                raise ValueError("movement arrival topology is inconsistent")
            for movement, raw in arrivals.items():
                values = np.asarray(raw, dtype=float)
                indices = tuple(int(value) for value in links[movement])
                if (
                    values.shape != (expected_bins,)
                    or not np.isfinite(values).all()
                    or np.any(values < 0.0)
                    or not indices
                    or min(indices) < 0
                    or len(indices) != len(set(indices))
                ):
                    raise ValueError("invalid movement arrival timeline row")
        movement_order = {
            tls_id: tuple(arrivals) for tls_id, arrivals in self.movement_arrivals.items()
        }
        arrival_matrix = {
            tls_id: np.vstack(
                [np.asarray(arrivals[movement], dtype=float) for movement in movement_order[tls_id]]
            )
            for tls_id, arrivals in self.movement_arrivals.items()
        }
        object.__setattr__(self, "_movement_order", movement_order)
        object.__setattr__(self, "_arrival_matrix", arrival_matrix)
        object.__setattr__(
            self,
            "_arrival_prefix",
            {
                tls_id: np.hstack(
                    [
                        np.zeros((values.shape[0], 1), dtype=float),
                        np.cumsum(values, axis=1),
                    ]
                )
                for tls_id, values in arrival_matrix.items()
            },
        )
        object.__setattr__(self, "_served_mask_cache", {})

    def _movement_is_served(
        self, tls_id: str, movement: Movement, candidate_state: str
    ) -> bool:
        indices = self.movement_link_indices[tls_id][movement]
        if max(indices) >= len(candidate_state):
            raise ValueError(
                f"candidate state is shorter than controlled links for {tls_id}"
            )
        return any(candidate_state[index] in GREEN_STATES for index in indices)

    def _served_mask(self, tls_id: str, candidate_state: str) -> np.ndarray:
        key = (str(tls_id), str(candidate_state))
        cached = self._served_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = np.asarray(
            [
                self._movement_is_served(key[0], movement, key[1])
                for movement in self._movement_order[key[0]]
            ],
            dtype=bool,
        )
        self._served_mask_cache[key] = mask
        return mask

    def _cumulative_arrivals(self, tls_id: str, time_sec: float) -> np.ndarray:
        values = self._arrival_matrix[tls_id]
        prefix = self._arrival_prefix[tls_id]
        position = (float(time_sec) - self.begin_sec) / float(self.bin_sec)
        if position <= 0.0:
            return np.zeros(values.shape[0], dtype=float)
        if position >= values.shape[1]:
            return prefix[:, -1].copy()
        index = int(math.floor(position))
        fraction = position - index
        return prefix[:, index] + values[:, index] * fraction

    def _window_arrivals(
        self, tls_id: str, start_sec: float, stop_sec: float
    ) -> np.ndarray:
        if stop_sec <= start_sec:
            return np.zeros(self._arrival_matrix[tls_id].shape[0], dtype=float)
        return self._cumulative_arrivals(tls_id, stop_sec) - self._cumulative_arrivals(
            tls_id, start_sec
        )

    def candidate_window_totals(
        self,
        tls_id: str,
        time_sec: float,
        candidate_state: str,
        horizon_sec: int,
    ) -> tuple[float, float, int, int]:
        tls = str(tls_id)
        if tls not in self.movement_arrivals:
            raise KeyError(f"unknown movement-arrival TLS: {tls}")
        if int(horizon_sec) <= 0 or not np.isfinite(float(time_sec)):
            raise ValueError("arrival window and time must be valid")
        values = self._window_arrivals(
            tls, float(time_sec), float(time_sec) + int(horizon_sec)
        )
        served_mask = self._served_mask(tls, str(candidate_state))
        served = float(np.sum(values[served_mask]))
        unserved = float(np.sum(values[~served_mask]))
        served_count = int(np.count_nonzero(served_mask))
        unserved_count = int(served_mask.size - served_count)
        return served, unserved, served_count, unserved_count

    def arrival_pressure_score(
        self,
        tls_id: str,
        time_sec: float,
        candidate_state: str,
        *,
        horizon_sec: int = 60,
    ) -> float:
        served, unserved, served_count, unserved_count = self.candidate_window_totals(
            tls_id, time_sec, candidate_state, horizon_sec
        )
        return served / max(served_count, 1) - unserved / max(unserved_count, 1)

    def vector_at(
        self, tls_id: str, time_sec: float, candidate_state: str
    ) -> np.ndarray:
        totals = {
            horizon: self.candidate_window_totals(
                tls_id, time_sec, candidate_state, horizon
            )
            for horizon in (30, 60, 120)
        }
        per_movement: dict[int, tuple[float, float]] = {}
        shares: dict[int, float] = {}
        for horizon, (served, unserved, served_count, unserved_count) in totals.items():
            per_movement[horizon] = (
                served / max(served_count, 1),
                unserved / max(unserved_count, 1),
            )
            shares[horizon] = served / max(served + unserved, 1.0)
        first_30 = per_movement[30][0]
        second_30 = max(
            totals[60][0] - totals[30][0], 0.0
        ) / max(totals[60][2], 1)
        vector = np.asarray(
            [
                *per_movement[30],
                *per_movement[60],
                *per_movement[120],
                shares[30],
                shares[60],
                shares[120],
                per_movement[60][0] - per_movement[60][1],
                per_movement[120][0] - per_movement[120][1],
                (second_30 - first_30) / max(second_30 + first_30, 1.0),
            ],
            dtype=float,
        )
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("movement-arrival feature vector changed")
        return vector

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "horizon_sec": self.horizon_sec,
            "begin_sec": self.begin_sec,
            "bin_sec": self.bin_sec,
            "input_sha256": self.input_sha256,
            "feature_names": list(self.feature_names),
            "scheduled_elements": self.scheduled_elements,
            "projected_elements": self.projected_elements,
            "projected_arrival_mass": self.projected_arrival_mass,
            "tls_count": len(self.movement_arrivals),
            "movement_count": sum(
                len(values) for values in self.movement_arrivals.values()
            ),
        }


def build_movement_arrival_timeline_context(
    sumocfg: Path, *, horizon_sec: int = 3600, bin_sec: int = 5
) -> MovementArrivalTimelineContext:
    sumocfg = Path(sumocfg).resolve()
    if int(horizon_sec) <= 0 or int(bin_sec) <= 0:
        raise ValueError("movement-arrival horizon and bin must be positive")
    cfg = ET.parse(sumocfg).getroot()
    begin_element = cfg.find(".//time/begin")
    begin_sec = (
        parse_sumo_time_seconds(begin_element.attrib.get("value", 0.0))
        if begin_element is not None
        else 0.0
    )
    net_files = _resolve_cfg_paths(sumocfg, "net-file")
    if len(net_files) != 1:
        raise ValueError("movement arrivals require exactly one net file")
    net = ET.parse(net_files[0]).getroot()
    edges = {
        str(edge.attrib["id"]): edge
        for edge in net.findall(".//edge")
        if not str(edge.attrib.get("function", ""))
        and "from" in edge.attrib
        and "to" in edge.attrib
    }
    travel_time: dict[str, float] = {}
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

    movement_links: dict[str, dict[Movement, set[int]]] = {}
    for connection in net.findall(".//connection"):
        if not {"from", "to", "tl", "linkIndex"} <= set(connection.attrib):
            continue
        movement = (
            str(connection.attrib["from"]),
            str(connection.attrib["to"]),
        )
        if movement[0] not in edges or movement[1] not in edges:
            continue
        movement_links.setdefault(str(connection.attrib["tl"]), {}).setdefault(
            movement, set()
        ).add(int(connection.attrib["linkIndex"]))
    if not movement_links:
        raise ValueError("SUMO network has no controlled edge-pair movements")
    pair_to_tls: dict[Movement, list[str]] = {}
    for tls_id, movements in movement_links.items():
        for movement in movements:
            pair_to_tls.setdefault(movement, []).append(tls_id)

    bins = int(math.ceil(int(horizon_sec) / int(bin_sec)))
    arrivals = {
        tls_id: {
            movement: np.zeros(bins, dtype=float) for movement in movements
        }
        for tls_id, movements in movement_links.items()
    }
    scheduled_elements = 0
    projected_elements = 0
    projected_arrival_mass = 0.0

    def route_movements(route: Sequence[str]) -> list[tuple[str, Movement, float]]:
        elapsed = 0.0
        rows: list[tuple[str, Movement, float]] = []
        for index, edge_id in enumerate(route[:-1]):
            elapsed += travel_time.get(edge_id, 0.0)
            movement = (str(edge_id), str(route[index + 1]))
            for tls_id in pair_to_tls.get(movement, ()):
                rows.append((tls_id, movement, elapsed))
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
            scheduled_elements += 1
            projected = route_movements(_route_edges(element, definitions))
            if not projected:
                continue
            projected_elements += 1
            if element.tag == "flow":
                start = parse_sumo_time_seconds(element.attrib.get("begin", begin_sec))
                stop = parse_sumo_time_seconds(element.attrib.get("end", start))
                rate = _flow_rate(element, start, stop)
                for tls_id, movement, offset in projected:
                    projected_arrival_mass += _add_uniform(
                        arrivals[tls_id][movement],
                        start_sec=start + offset,
                        stop_sec=stop + offset,
                        rate_per_sec=rate,
                        begin_sec=begin_sec,
                        bin_sec=int(bin_sec),
                    )
            else:
                depart = parse_sumo_time_seconds(element.attrib.get("depart", begin_sec))
                for tls_id, movement, offset in projected:
                    if _add_event(
                        arrivals[tls_id][movement],
                        time_sec=depart + offset,
                        begin_sec=begin_sec,
                        bin_sec=int(bin_sec),
                    ):
                        projected_arrival_mass += 1.0

    return MovementArrivalTimelineContext(
        protocol=MOVEMENT_ARRIVAL_PROTOCOL,
        horizon_sec=int(horizon_sec),
        begin_sec=float(begin_sec),
        bin_sec=int(bin_sec),
        input_sha256=demand_input_sha256(sumocfg),
        feature_names=MOVEMENT_ARRIVAL_FEATURE_NAMES,
        movement_arrivals={
            tls_id: {
                movement: tuple(float(value) for value in values)
                for movement, values in sorted(movements.items())
            }
            for tls_id, movements in sorted(arrivals.items())
        },
        movement_link_indices={
            tls_id: {
                movement: tuple(sorted(indices))
                for movement, indices in sorted(movements.items())
            }
            for tls_id, movements in sorted(movement_links.items())
        },
        scheduled_elements=scheduled_elements,
        projected_elements=projected_elements,
        projected_arrival_mass=float(projected_arrival_mass),
    )


def augment_dataset_with_movement_arrivals(
    dataset: MechanismDataset, context: MovementArrivalTimelineContext
) -> MechanismDataset:
    row_tls = list(dataset.metadata.get("row_tls", ()))
    row_times = list(dataset.metadata.get("row_times", ()))
    candidate_states = list(dataset.metadata.get("candidate_states", ()))
    if not (
        len(row_tls) == len(row_times) == len(candidate_states) == dataset.size
    ):
        raise ValueError("dataset lacks aligned candidate movement metadata")
    arrival_features = np.vstack(
        [
            context.vector_at(tls_id, float(time_sec), str(candidate_state))
            for tls_id, time_sec, candidate_state in zip(
                row_tls, row_times, candidate_states, strict=True
            )
        ]
    ) if dataset.size else np.zeros((0, len(context.feature_names)), dtype=float)
    metadata = dict(dataset.metadata)
    metadata.update(
        {
            "movement_arrival_protocol": context.protocol,
            "movement_arrival_input_sha256": context.input_sha256,
            "movement_arrival_feature_names": list(context.feature_names),
            "movement_arrival_information_budget": (
                "target_static_network_route_schedule_plus_candidate_state_no_labels"
            ),
        }
    )
    return MechanismDataset(
        feature_names=(*dataset.feature_names, *context.feature_names),
        features=np.hstack([dataset.features, arrival_features]),
        context_names=dataset.context_names,
        context=dataset.context,
        priors={name: np.asarray(values).copy() for name, values in dataset.priors.items()},
        targets={name: np.asarray(values).copy() for name, values in dataset.targets.items()},
        domains=dataset.domains.copy(),
        metadata=metadata,
    )
