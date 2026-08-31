"""Topology-invariant dynamic graph features for signal control transfer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SIGNAL_GRAPH_STATE_FEATURES = (
    "graph_upstream_q_mean",
    "graph_downstream_q_mean",
    "graph_neighbor_q_max",
    "graph_two_hop_q_mean",
    "graph_neighbor_occ_mean",
    "graph_neighbor_pressure_mean",
    "graph_global_q_per_tls",
    "graph_global_q_p90",
    "graph_focal_q_percentile",
    "graph_focal_to_neighbor_q_ratio",
    "graph_in_degree_norm",
    "graph_out_degree_norm",
    "network_active_per_tls",
)

SIGNAL_GRAPH_ACTION_FEATURES = (
    "green_signal_down_q_mean",
    "green_signal_down_q_max",
    "green_signal_down_occ_mean",
    "green_signal_down_occ_max",
    "green_signal_reach_ratio",
    "green_corridor_pressure",
)

SIGNAL_GRAPH_FEATURES = (*SIGNAL_GRAPH_STATE_FEATURES, *SIGNAL_GRAPH_ACTION_FEATURES)


@dataclass(frozen=True)
class DynamicSignalContext:
    values: Mapping[str, float]
    downstream_tls_by_lane: Mapping[str, tuple[str, ...]]
    signal_queue: Mapping[str, float]
    signal_occupancy: Mapping[str, float]


@dataclass(frozen=True)
class SignalRoutingGraph:
    adjacency: Mapping[str, tuple[str, ...]]
    downstream_tls_by_lane: Mapping[str, Mapping[str, tuple[str, ...]]]


def build_signal_routing_graph(
    sumo_api: Any,
    infos: Mapping[str, Any],
    *,
    max_hops: int = 24,
) -> SignalRoutingGraph:
    """Trace controlled outgoing lanes to the nearest downstream signals."""

    incoming_owner = {
        str(lane): str(tls_id)
        for tls_id, info in infos.items()
        for lane in getattr(info, "incoming_lanes", ())
    }
    downstream: dict[str, dict[str, tuple[str, ...]]] = {}
    adjacency: dict[str, set[str]] = {str(tls_id): set() for tls_id in infos}
    for raw_tls_id, info in infos.items():
        tls_id = str(raw_tls_id)
        outgoing_lanes = {
            str(connection[1])
            for link_group in getattr(info, "controlled_links", ())
            for connection in (link_group or ())
            if connection and len(connection) >= 2 and connection[1]
        }
        downstream[tls_id] = {}
        for outgoing_lane in sorted(outgoing_lanes):
            targets = _nearest_downstream_signals(
                sumo_api,
                outgoing_lane,
                source_tls=tls_id,
                incoming_owner=incoming_owner,
                max_hops=max_hops,
            )
            downstream[tls_id][outgoing_lane] = targets
            adjacency[tls_id].update(targets)
    return SignalRoutingGraph(
        adjacency={name: tuple(sorted(values)) for name, values in adjacency.items()},
        downstream_tls_by_lane=downstream,
    )


def build_dynamic_signal_contexts(
    states: Mapping[str, Any],
    infos: Mapping[str, Any],
    *,
    active_vehicles: float,
    routing_graph: SignalRoutingGraph | None = None,
) -> dict[str, DynamicSignalContext]:
    """Aggregate current traffic in portable local graph coordinates."""

    tls_ids = tuple(sorted(str(tls_id) for tls_id in infos))
    if routing_graph is None:
        incoming_owner = {
            str(lane): str(tls_id)
            for tls_id, info in infos.items()
            for lane in getattr(info, "incoming_lanes", ())
        }
        adjacency_sets: dict[str, set[str]] = {tls_id: set() for tls_id in tls_ids}
        downstream_tls_by_lane: dict[str, dict[str, tuple[str, ...]]] = {
            tls_id: {} for tls_id in tls_ids
        }
        for raw_tls_id, info in infos.items():
            tls_id = str(raw_tls_id)
            for link_group in getattr(info, "controlled_links", ()):
                for connection in link_group or ():
                    if not connection or len(connection) < 2:
                        continue
                    out_lane = str(connection[1])
                    target = incoming_owner.get(out_lane)
                    if target is None or target == tls_id:
                        continue
                    adjacency_sets[tls_id].add(target)
                    downstream_tls_by_lane[tls_id][out_lane] = (target,)
        adjacency = {
            name: set(values) for name, values in adjacency_sets.items()
        }
    else:
        adjacency = {
            tls_id: set(routing_graph.adjacency.get(tls_id, ()))
            for tls_id in tls_ids
        }
        downstream_tls_by_lane = {
            tls_id: dict(routing_graph.downstream_tls_by_lane.get(tls_id, {}))
            for tls_id in tls_ids
        }
    upstream: dict[str, set[str]] = {tls_id: set() for tls_id in tls_ids}
    for source, targets in adjacency.items():
        for target in targets:
            upstream[target].add(source)

    signal_queue = {
        tls_id: float(
            sum(
                float(states[tls_id].q_by_lane[lane])
                for lane in sorted(states[tls_id].q_by_lane)
            )
        )
        for tls_id in tls_ids
    }
    signal_occupancy = {
        tls_id: _safe_mean(
            states[tls_id].occ_by_lane[lane]
            for lane in sorted(states[tls_id].occ_by_lane)
        )
        for tls_id in tls_ids
    }
    signal_pressure = {
        tls_id: float(
            signal_queue[tls_id]
            - sum(
                float(states[tls_id].down_q_by_lane[lane])
                for lane in sorted(states[tls_id].down_q_by_lane)
            )
        )
        for tls_id in tls_ids
    }
    all_queue = np.asarray([signal_queue[tls_id] for tls_id in tls_ids], dtype=float)
    tls_count = max(len(tls_ids), 1)
    degree_scale = max(tls_count - 1, 1)
    result = {}
    for tls_id in tls_ids:
        direct = upstream[tls_id] | adjacency[tls_id]
        two_hop = set()
        for neighbor in direct:
            two_hop.update(upstream[neighbor])
            two_hop.update(adjacency[neighbor])
        two_hop.difference_update(direct | {tls_id})
        ordered_upstream = tuple(sorted(upstream[tls_id]))
        ordered_downstream = tuple(sorted(adjacency[tls_id]))
        ordered_direct = tuple(sorted(direct))
        ordered_two_hop = tuple(sorted(two_hop))
        neighbor_q = [signal_queue[name] for name in ordered_direct]
        values = {
            "graph_upstream_q_mean": _safe_mean(
                signal_queue[name] for name in ordered_upstream
            ),
            "graph_downstream_q_mean": _safe_mean(
                signal_queue[name] for name in ordered_downstream
            ),
            "graph_neighbor_q_max": max(neighbor_q, default=0.0),
            "graph_two_hop_q_mean": _safe_mean(
                signal_queue[name] for name in ordered_two_hop
            ),
            "graph_neighbor_occ_mean": _safe_mean(
                signal_occupancy[name] for name in ordered_direct
            ),
            "graph_neighbor_pressure_mean": _safe_mean(
                signal_pressure[name] for name in ordered_direct
            ),
            "graph_global_q_per_tls": float(np.mean(all_queue)) if all_queue.size else 0.0,
            "graph_global_q_p90": float(np.quantile(all_queue, 0.90))
            if all_queue.size
            else 0.0,
            "graph_focal_q_percentile": float(
                np.mean(all_queue <= signal_queue[tls_id])
            )
            if all_queue.size
            else 0.0,
            "graph_focal_to_neighbor_q_ratio": float(
                signal_queue[tls_id] / max(_safe_mean(neighbor_q), 1.0)
            ),
            "graph_in_degree_norm": float(len(upstream[tls_id]) / degree_scale),
            "graph_out_degree_norm": float(len(adjacency[tls_id]) / degree_scale),
            "network_active_per_tls": float(active_vehicles / tls_count),
        }
        result[tls_id] = DynamicSignalContext(
            values=values,
            downstream_tls_by_lane=downstream_tls_by_lane[tls_id],
            signal_queue=signal_queue,
            signal_occupancy=signal_occupancy,
        )
    return result


def candidate_graph_features(
    context: DynamicSignalContext | None,
    info: Any,
    candidate_state: str,
    *,
    green_queue: float,
) -> np.ndarray:
    if context is None:
        return np.zeros(len(SIGNAL_GRAPH_FEATURES), dtype=float)
    target_ids = []
    green_movements = 0
    for link_index, char in enumerate(candidate_state):
        if char not in {"G", "g"} or link_index >= len(info.controlled_links):
            continue
        for connection in info.controlled_links[link_index] or ():
            if not connection or len(connection) < 2:
                continue
            green_movements += 1
            targets = context.downstream_tls_by_lane.get(str(connection[1]), ())
            for target in targets:
                if target not in target_ids:
                    target_ids.append(target)
    down_q = [context.signal_queue[target] for target in target_ids]
    down_occ = [context.signal_occupancy[target] for target in target_ids]
    action_values = {
        "green_signal_down_q_mean": _safe_mean(down_q),
        "green_signal_down_q_max": max(down_q, default=0.0),
        "green_signal_down_occ_mean": _safe_mean(down_occ),
        "green_signal_down_occ_max": max(down_occ, default=0.0),
        "green_signal_reach_ratio": float(
            len(target_ids) / max(green_movements, 1)
        ),
        "green_corridor_pressure": float(
            green_queue - _safe_mean(down_q)
        ),
    }
    values = {**context.values, **action_values}
    return np.asarray([values[name] for name in SIGNAL_GRAPH_FEATURES], dtype=float)


def _safe_mean(values: Any) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if array.size else 0.0


def _nearest_downstream_signals(
    sumo_api: Any,
    start_lane: str,
    *,
    source_tls: str,
    incoming_owner: Mapping[str, str],
    max_hops: int,
) -> tuple[str, ...]:
    queue = deque([(str(start_lane), 0)])
    visited = set()
    targets = set()
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
        try:
            next_lanes = {
                str(link[0])
                for link in sumo_api.lane.getLinks(lane)
                if link and link[0]
            }
        except Exception:
            next_lanes = set()
        queue.extend((next_lane, depth + 1) for next_lane in sorted(next_lanes))
    return tuple(sorted(targets))
