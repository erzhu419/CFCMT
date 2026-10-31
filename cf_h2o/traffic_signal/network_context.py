"""Label-free topology descriptors for cross-network signal transfer."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.eval.traffic_signal_resco_phase_benchmark import _net_and_route_from_sumocfg
from cf_h2o.traffic_signal.sumo_static_inputs import (
    demand_route_mixture,
    flow_expected_count,
    iterparse_xml,
    iter_static_demand_records,
    load_static_route_catalog,
    parse_xml,
    static_demand_input_paths,
)


TOPOLOGY_CONTEXT_NAMES = (
    "incoming_lane_cv",
    "controlled_link_cv",
    "candidate_count_cv",
    "green_ratio_std",
    "phase_duration_cv",
    "signal_degree_mean_norm",
    "signal_degree_cv",
    "signal_graph_density",
    "signal_graph_reciprocity",
    "signal_layout_anisotropy",
    "signal_spacing_cv",
    "route_edge_count_mean_norm",
    "route_edge_count_cv",
    "route_entropy_norm",
    "route_max_share",
)


def static_topology_context(sumocfg: Path, infos: Mapping[str, Any]) -> np.ndarray:
    """Build portable network descriptors without simulation outcomes."""

    sumocfg = Path(sumocfg)
    incoming_counts = np.asarray([len(info.incoming_lanes) for info in infos.values()], dtype=float)
    link_counts = np.asarray([len(info.controlled_links) for info in infos.values()], dtype=float)
    candidate_counts = np.asarray([len(info.candidates) for info in infos.values()], dtype=float)
    green_ratios = []
    phase_durations = []
    for info in infos.values():
        link_count = max(len(info.controlled_links), 1)
        for candidate in info.candidates:
            green_ratios.append(float(candidate.green_count) / link_count)
            phase_durations.append(float(candidate.duration))

    adjacency = _signal_adjacency(infos, sumocfg=sumocfg)
    degrees = np.asarray(
        [len(adjacency[name] | {source for source, targets in adjacency.items() if name in targets}) for name in infos],
        dtype=float,
    )
    directed_edges = sum(len(targets) for targets in adjacency.values())
    reciprocal_edges = sum(
        int(source in adjacency.get(target, set()))
        for source, targets in adjacency.items()
        for target in targets
    )
    tls_count = max(len(infos), 1)
    possible_edges = max(tls_count * (tls_count - 1), 1)

    coordinates = _traffic_light_coordinates(sumocfg, set(infos))
    anisotropy, spacing_cv = _layout_descriptors(coordinates)
    route_lengths, route_weights = _route_descriptors(sumocfg)
    route_mean, route_cv = _weighted_mean_cv(route_lengths, route_weights)
    if route_weights.size:
        probabilities = route_weights / max(float(np.sum(route_weights)), 1e-12)
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        entropy_norm = entropy / max(math.log(max(probabilities.size, 2)), 1e-12)
        max_share = float(np.max(probabilities))
    else:
        entropy_norm = 0.0
        max_share = 0.0

    values = np.asarray(
        [
            _coefficient_of_variation(incoming_counts),
            _coefficient_of_variation(link_counts),
            _coefficient_of_variation(candidate_counts),
            float(np.std(green_ratios)) if green_ratios else 0.0,
            _coefficient_of_variation(np.asarray(phase_durations, dtype=float)),
            float(np.mean(degrees) / 4.0) if degrees.size else 0.0,
            _coefficient_of_variation(degrees),
            float(directed_edges / possible_edges),
            float(reciprocal_edges / max(directed_edges, 1)),
            anisotropy,
            spacing_cv,
            float(route_mean / 20.0),
            route_cv,
            entropy_norm,
            max_share,
        ],
        dtype=float,
    )
    if values.shape != (len(TOPOLOGY_CONTEXT_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid static topology context")
    return values


def _signal_adjacency(
    infos: Mapping[str, Any],
    *,
    sumocfg: Path | None = None,
    max_hops: int = 24,
) -> dict[str, set[str]]:
    """Trace each controlled movement to the nearest downstream signal.

    Converted networks can contain ordinary junctions or split edges between
    controlled intersections.  Matching only the immediate outgoing lane then
    reports a disconnected signal graph even though vehicles continue to a
    downstream signal.  The optional network trace mirrors the runtime graph
    semantics without requiring a live SUMO instance.
    """

    incoming_owner = {
        str(lane): str(tls_id)
        for tls_id, info in infos.items()
        for lane in info.incoming_lanes
    }
    adjacency = {str(tls_id): set() for tls_id in infos}
    successors = (
        _network_lane_successors(Path(sumocfg))
        if sumocfg is not None
        else None
    )
    for tls_id, info in infos.items():
        source = str(tls_id)
        for link_group in info.controlled_links:
            for connection in link_group or ():
                if not connection or len(connection) < 2 or not connection[1]:
                    continue
                outgoing_lane = str(connection[1])
                if successors is None:
                    target = incoming_owner.get(outgoing_lane)
                    if target is not None and target != source:
                        adjacency[source].add(target)
                    continue
                adjacency[source].update(
                    _nearest_downstream_signals(
                        outgoing_lane,
                        source_tls=source,
                        incoming_owner=incoming_owner,
                        successors=successors,
                        max_hops=max_hops,
                    )
                )
    return adjacency


def _network_lane_successors(sumocfg: Path) -> dict[str, set[str]]:
    net_files, _, _ = _net_and_route_from_sumocfg(sumocfg)
    successors: dict[str, set[str]] = {}
    for name in net_files:
        path = sumocfg.parent / name
        if not path.is_file():
            continue
        root = parse_xml(path).getroot()
        lanes: dict[tuple[str, str], str] = {}
        for edge in root.findall(".//edge"):
            edge_id = str(edge.attrib.get("id", ""))
            for ordinal, lane in enumerate(edge.findall("lane")):
                lane_index = str(lane.attrib.get("index", ordinal))
                lane_id = str(lane.attrib.get("id", ""))
                if edge_id and lane_id:
                    lanes[(edge_id, lane_index)] = lane_id
                    successors.setdefault(lane_id, set())
        for connection in root.findall(".//connection"):
            source = lanes.get(
                (
                    str(connection.attrib.get("from", "")),
                    str(connection.attrib.get("fromLane", "")),
                )
            )
            target = lanes.get(
                (
                    str(connection.attrib.get("to", "")),
                    str(connection.attrib.get("toLane", "")),
                )
            )
            if source is not None and target is not None:
                successors.setdefault(source, set()).add(target)
    return successors


def _nearest_downstream_signals(
    start_lane: str,
    *,
    source_tls: str,
    incoming_owner: Mapping[str, str],
    successors: Mapping[str, set[str]],
    max_hops: int,
) -> set[str]:
    queue = deque([(str(start_lane), 0)])
    visited: set[str] = set()
    targets: set[str] = set()
    while queue and len(visited) < 4096:
        lane, depth = queue.popleft()
        if lane in visited or depth > int(max_hops):
            continue
        visited.add(lane)
        owner = incoming_owner.get(lane)
        if owner is not None and owner != source_tls:
            targets.add(owner)
            continue
        if depth == int(max_hops):
            continue
        queue.extend(
            (next_lane, depth + 1)
            for next_lane in sorted(successors.get(lane, ()))
        )
    return targets


def _traffic_light_coordinates(sumocfg: Path, tls_ids: set[str]) -> np.ndarray:
    net_files, _, _ = _net_and_route_from_sumocfg(sumocfg)
    coordinates = []
    for name in net_files:
        path = sumocfg.parent / name
        if not path.exists():
            continue
        for _, element in iterparse_xml(path, events=("end",)):
            if element.tag == "junction" and element.attrib.get("id") in tls_ids:
                try:
                    coordinates.append((float(element.attrib["x"]), float(element.attrib["y"])))
                except (KeyError, ValueError):
                    pass
            element.clear()
    return np.asarray(coordinates, dtype=float).reshape(-1, 2)


def _layout_descriptors(coordinates: np.ndarray) -> tuple[float, float]:
    if coordinates.shape[0] < 2:
        return 0.0, 0.0
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(coordinates.shape[0] - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    anisotropy = 1.0 - float(eigenvalues[-1] / max(eigenvalues[0], 1e-9))
    distances = np.sqrt(np.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2))
    distances[distances <= 1e-9] = np.inf
    nearest = np.min(distances, axis=1)
    spacing_cv = _coefficient_of_variation(nearest[np.isfinite(nearest)])
    return float(np.clip(anisotropy, 0.0, 1.0)), spacing_cv


def _route_descriptors(sumocfg: Path) -> tuple[np.ndarray, np.ndarray]:
    demand_paths = static_demand_input_paths(sumocfg)
    catalog = load_static_route_catalog(demand_paths)
    path_weight: dict[tuple[str, ...], float] = {}
    for record in iter_static_demand_records(demand_paths):
        weight = (
            flow_expected_count(record.attributes)
            if record.kind == "flow"
            else 1.0
        )
        for edges, probability in demand_route_mixture(record, catalog):
            path_weight[edges] = (
                path_weight.get(edges, 0.0) + weight * probability
            )
    lengths = [float(len(edges)) for edges in path_weight]
    weights = [float(weight) for weight in path_weight.values()]
    return np.asarray(lengths, dtype=float), np.asarray(weights, dtype=float)


def _weighted_mean_cv(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if values.size == 0 or weights.size != values.size or float(np.sum(weights)) <= 0.0:
        return 0.0, 0.0
    normalized = weights / float(np.sum(weights))
    mean = float(np.sum(normalized * values))
    variance = float(np.sum(normalized * (values - mean) ** 2))
    return mean, math.sqrt(max(variance, 0.0)) / max(abs(mean), 1e-9)


def _coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.std(values) / max(abs(float(np.mean(values))), 1e-9))
