"""Deploy-observable right-of-way and neighboring signal features.

The historical V3 feature schema treats protected (``G``) and permissive
(``g``) service identically.  This module adds a separate, opt-in feature
family that can be reconstructed from a SUMO network and aligned row metadata
without reading any transition outcome.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


RIGHT_OF_WAY_ACTION_FEATURES = (
    "protected_green_ratio",
    "permissive_green_ratio",
    "green_shared_receiver_ratio",
    "green_uncontrolled_major_merge_ratio",
    "protected_uncontrolled_major_merge_ratio",
    "permissive_uncontrolled_major_merge_ratio",
    "green_receiver_competition_mean",
)

NEIGHBOR_EXECUTION_STATE_FEATURES = (
    "neighbor_phase_observation_ratio",
    "neighbor_current_protected_ratio_mean",
    "neighbor_current_permissive_ratio_mean",
    "neighbor_current_green_elapsed_norm_mean",
    "neighbor_switching_fraction",
    "neighbor_clearance_fraction_mean",
)

RIGHT_OF_WAY_CONTEXT_FEATURES = (
    *RIGHT_OF_WAY_ACTION_FEATURES,
    *NEIGHBOR_EXECUTION_STATE_FEATURES,
)


@dataclass(frozen=True)
class LinkRightOfWay:
    receiving_lane: str
    external_competitor_count: int
    uncontrolled_major_competitor_count: int


@dataclass(frozen=True)
class NetworkRightOfWayContext:
    network: str
    links_by_tls: Mapping[str, Mapping[int, tuple[LinkRightOfWay, ...]]]


def net_file_from_sumocfg(sumocfg: Path) -> Path:
    """Resolve the sole network file declared by a SUMO configuration."""

    sumocfg = Path(sumocfg).resolve()
    root = ET.parse(sumocfg).getroot()
    values: list[str] = []
    for item in root.findall(".//net-file"):
        values.extend(
            part.strip()
            for part in str(item.attrib.get("value", "")).split(",")
            if part.strip()
        )
    if len(values) != 1:
        raise ValueError(f"expected one net-file in {sumocfg}, found {values}")
    network = (sumocfg.parent / values[0]).resolve()
    if not network.is_file():
        raise FileNotFoundError(network)
    return network


def read_network_right_of_way_context(network: Path) -> NetworkRightOfWayContext:
    """Parse controlled links and competing external receiving movements."""

    network = Path(network).resolve()
    if not network.is_file():
        raise FileNotFoundError(network)
    connections: list[dict[str, str]] = []
    for _, element in ET.iterparse(network, events=("end",)):
        if str(element.tag).rsplit("}", 1)[-1] == "connection":
            attrs = {str(key): str(value) for key, value in element.attrib.items()}
            if attrs.get("from") and attrs.get("to"):
                connections.append(attrs)
        element.clear()

    by_receiver: dict[str, list[dict[str, str]]] = defaultdict(list)
    for connection in connections:
        receiver = _receiving_lane(connection)
        if receiver:
            by_receiver[receiver].append(connection)

    links: dict[str, dict[int, list[LinkRightOfWay]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for connection in connections:
        tls_id = str(connection.get("tl", ""))
        raw_index = connection.get("linkIndex")
        receiver = _receiving_lane(connection)
        if not tls_id or raw_index in {None, ""} or not receiver:
            continue
        try:
            link_index = int(str(raw_index))
        except ValueError as exc:
            raise ValueError(
                f"invalid linkIndex {raw_index!r} for TLS {tls_id!r}"
            ) from exc
        identity = _movement_identity(connection)
        competitors = [
            row
            for row in by_receiver[receiver]
            if _movement_identity(row) != identity and not _is_internal(row)
        ]
        uncontrolled_major = [
            row for row in competitors if _is_uncontrolled_major(row)
        ]
        links[tls_id][link_index].append(
            LinkRightOfWay(
                receiving_lane=receiver,
                external_competitor_count=len(competitors),
                uncontrolled_major_competitor_count=len(uncontrolled_major),
            )
        )
    return NetworkRightOfWayContext(
        network=str(network),
        links_by_tls={
            tls_id: {
                index: tuple(values) for index, values in sorted(by_index.items())
            }
            for tls_id, by_index in sorted(links.items())
        },
    )


def candidate_right_of_way_features(
    context: NetworkRightOfWayContext,
    tls_id: str,
    candidate_state: str,
) -> np.ndarray:
    """Return action features weighted by concrete controlled movements."""

    movements: list[tuple[str, LinkRightOfWay]] = []
    by_index = context.links_by_tls.get(str(tls_id), {})
    for index, signal in enumerate(str(candidate_state)):
        if signal not in {"G", "g"}:
            continue
        for movement in by_index.get(index, ()):
            movements.append((signal, movement))
    denominator = max(len(movements), 1)
    competition = [
        min(float(movement.external_competitor_count), 4.0) / 4.0
        for _, movement in movements
    ]
    values = {
        "protected_green_ratio": sum(
            signal == "G" for signal, _ in movements
        )
        / denominator,
        "permissive_green_ratio": sum(
            signal == "g" for signal, _ in movements
        )
        / denominator,
        "green_shared_receiver_ratio": sum(
            movement.external_competitor_count > 0 for _, movement in movements
        )
        / denominator,
        "green_uncontrolled_major_merge_ratio": sum(
            movement.uncontrolled_major_competitor_count > 0
            for _, movement in movements
        )
        / denominator,
        "protected_uncontrolled_major_merge_ratio": sum(
            signal == "G" and movement.uncontrolled_major_competitor_count > 0
            for signal, movement in movements
        )
        / denominator,
        "permissive_uncontrolled_major_merge_ratio": sum(
            signal == "g" and movement.uncontrolled_major_competitor_count > 0
            for signal, movement in movements
        )
        / denominator,
        "green_receiver_competition_mean": float(np.mean(competition))
        if competition
        else 0.0,
    }
    return np.asarray(
        [values[name] for name in RIGHT_OF_WAY_ACTION_FEATURES], dtype=float
    )


def augment_dataset_with_right_of_way_context(
    dataset: MechanismDataset,
    contexts_by_scenario: Mapping[str, NetworkRightOfWayContext],
) -> MechanismDataset:
    """Append right-of-way and recoverable neighbor execution features.

    Neighbor state is inferred only from action groups sampled at the exact
    same scenario, seed, and simulation time.  The observation-ratio feature
    keeps missing neighbor snapshots distinct from observed zeros.
    """

    if any(name in dataset.feature_names for name in RIGHT_OF_WAY_CONTEXT_FEATURES):
        raise ValueError("right-of-way context is already present")
    row_tls = np.asarray(dataset.metadata.get("row_tls", ()), dtype=str)
    candidate_states = np.asarray(
        dataset.metadata.get("candidate_states", ()), dtype=str
    )
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
    if any(values.shape != (dataset.size,) for values in (row_tls, candidate_states, groups)):
        raise ValueError("right-of-way augmentation requires aligned row metadata")
    group_scenarios, adjacency = _scenario_metadata(dataset.metadata)
    missing_groups = sorted(set(groups) - set(group_scenarios))
    if missing_groups:
        raise ValueError(
            f"scenario metadata missing for action groups: {missing_groups[:3]}"
        )

    topology = np.vstack(
        [
            candidate_right_of_way_features(
                contexts_by_scenario[group_scenarios[group]], tls_id, state
            )
            for group, tls_id, state in zip(
                groups, row_tls, candidate_states, strict=True
            )
        ]
    )
    group_rows: dict[str, np.ndarray] = {
        group: np.flatnonzero(groups == group) for group in sorted(set(groups))
    }
    feature_index = {
        str(name): index for index, name in enumerate(dataset.feature_names)
    }
    required = (
        "current_phase_overlap",
        "current_green_elapsed_norm",
        "switch_indicator",
        "clearance_fraction",
    )
    missing = [name for name in required if name not in feature_index]
    if missing:
        raise KeyError(f"missing execution features: {missing}")
    topo_index = {
        name: index for index, name in enumerate(RIGHT_OF_WAY_ACTION_FEATURES)
    }
    summaries: dict[str, dict[str, float]] = {}
    for group, rows in group_rows.items():
        overlaps = dataset.features[rows, feature_index["current_phase_overlap"]]
        best_local = int(np.argmax(overlaps))
        selected_row = int(rows[best_local])
        switch_values = dataset.features[rows, feature_index["switch_indicator"]]
        summaries[group] = {
            "protected": float(
                topology[selected_row, topo_index["protected_green_ratio"]]
            ),
            "permissive": float(
                topology[selected_row, topo_index["permissive_green_ratio"]]
            ),
            "elapsed": float(
                dataset.features[
                    selected_row, feature_index["current_green_elapsed_norm"]
                ]
            ),
            "switching": float(np.min(switch_values) > 0.5),
            "clearance": float(
                dataset.features[selected_row, feature_index["clearance_fraction"]]
            ),
        }

    neighbor_by_group: dict[str, np.ndarray] = {}
    for group in group_rows:
        scenario = group_scenarios[group]
        tls_id = str(row_tls[group_rows[group][0]])
        scenario_graph = adjacency.get(scenario, {})
        downstream = set(scenario_graph.get(tls_id, ()))
        upstream = {
            source
            for source, targets in scenario_graph.items()
            if tls_id in set(targets)
        }
        neighbors = tuple(sorted((downstream | upstream) - {tls_id}))
        prefix = group.rsplit(":", 1)[0]
        observed = [
            summaries[f"{prefix}:{neighbor}"]
            for neighbor in neighbors
            if f"{prefix}:{neighbor}" in summaries
        ]
        if neighbors:
            observation_ratio = len(observed) / len(neighbors)
        else:
            observation_ratio = 1.0
        neighbor_by_group[group] = np.asarray(
            [
                observation_ratio,
                _mean(summary["protected"] for summary in observed),
                _mean(summary["permissive"] for summary in observed),
                _mean(summary["elapsed"] for summary in observed),
                _mean(summary["switching"] for summary in observed),
                _mean(summary["clearance"] for summary in observed),
            ],
            dtype=float,
        )
    neighbor = np.vstack([neighbor_by_group[group] for group in groups])
    augmented = np.column_stack((dataset.features, topology, neighbor))
    if not np.all(np.isfinite(augmented)):
        raise ValueError("right-of-way context contains non-finite values")
    return MechanismDataset(
        feature_names=(*dataset.feature_names, *RIGHT_OF_WAY_CONTEXT_FEATURES),
        features=augmented,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata={
            **dict(dataset.metadata),
            "right_of_way_context_protocol": (
                "deploy-observable-right-of-way-and-matched-neighbor-state-v1"
            ),
            "right_of_way_feature_names": list(RIGHT_OF_WAY_CONTEXT_FEATURES),
            "right_of_way_networks": {
                scenario: context.network
                for scenario, context in sorted(contexts_by_scenario.items())
            },
        },
    )


def _scenario_metadata(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, tuple[str, ...]]]]:
    group_scenarios: dict[str, str] = {}
    adjacency: dict[str, dict[str, tuple[str, ...]]] = {}

    def visit(node: Mapping[str, Any]) -> None:
        children = node.get("source_datasets", ())
        if children:
            for child in children:
                visit(child)
            return
        scenario = str(node.get("scenario", ""))
        groups = tuple(str(value) for value in node.get("action_group_ids", ()))
        if not scenario or not groups:
            return
        for group in groups:
            existing = group_scenarios.setdefault(group, scenario)
            if existing != scenario:
                raise ValueError(f"action group belongs to two scenarios: {group}")
        raw_graph = node.get("signal_routing_graph", {}).get("adjacency", {})
        normalized = {
            str(source): tuple(sorted(str(value) for value in targets))
            for source, targets in raw_graph.items()
        }
        existing_graph = adjacency.setdefault(scenario, normalized)
        if existing_graph != normalized:
            raise ValueError(f"inconsistent signal graph for scenario {scenario}")

    visit(metadata)
    return group_scenarios, adjacency


def _receiving_lane(connection: Mapping[str, str]) -> str:
    edge = str(connection.get("to", ""))
    lane = str(connection.get("toLane", ""))
    return f"{edge}_{lane}" if edge and lane else ""


def _movement_identity(connection: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        str(connection.get(name, ""))
        for name in ("from", "fromLane", "to", "toLane")
    )


def _is_internal(connection: Mapping[str, str]) -> bool:
    return str(connection.get("from", "")).startswith(":")


def _is_uncontrolled_major(connection: Mapping[str, str]) -> bool:
    return bool(
        not connection.get("tl")
        and not _is_internal(connection)
        and (
            str(connection.get("uncontrolled", "")) == "1"
            or str(connection.get("state", "")) == "M"
        )
    )


def _mean(values: Any) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if array.size else 0.0
