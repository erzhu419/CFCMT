"""Local-interference graph for scalable traffic-signal interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TlsInterventionGraph:
    """One-hop lane neighborhoods and their induced TLS conflict graph."""

    neighborhoods: Mapping[str, tuple[str, ...]]
    conflicts: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "neighborhoods": {
                tls_id: list(lanes) for tls_id, lanes in sorted(self.neighborhoods.items())
            },
            "conflicts": {
                tls_id: list(neighbors) for tls_id, neighbors in sorted(self.conflicts.items())
            },
            "tls_count": len(self.neighborhoods),
            "conflict_edge_count": sum(len(values) for values in self.conflicts.values()) // 2,
        }


def build_tls_intervention_graph(infos: Mapping[str, Any]) -> TlsInterventionGraph:
    """Build local cost neighborhoods from incoming and immediate outgoing lanes.

    Two signal interventions conflict when their one-hop lane neighborhoods
    overlap. This encodes a partial-interference assumption: counterfactual
    effects learned for a focal TLS may be composed only for spatially
    separated TLSs.
    """

    neighborhoods: dict[str, tuple[str, ...]] = {}
    for raw_tls_id, info in infos.items():
        tls_id = str(raw_tls_id)
        lanes = {str(lane) for lane in getattr(info, "incoming_lanes", ())}
        for link_group in getattr(info, "controlled_links", ()):
            for connection in link_group or ():
                if not connection:
                    continue
                if len(connection) >= 1 and connection[0]:
                    lanes.add(str(connection[0]))
                if len(connection) >= 2 and connection[1]:
                    lanes.add(str(connection[1]))
        neighborhoods[tls_id] = tuple(sorted(lanes))

    lane_sets = {tls_id: set(lanes) for tls_id, lanes in neighborhoods.items()}
    conflicts: dict[str, set[str]] = {tls_id: set() for tls_id in neighborhoods}
    tls_ids = sorted(neighborhoods)
    for left_idx, left in enumerate(tls_ids):
        for right in tls_ids[left_idx + 1 :]:
            if lane_sets[left] & lane_sets[right]:
                conflicts[left].add(right)
                conflicts[right].add(left)
    return TlsInterventionGraph(
        neighborhoods=neighborhoods,
        conflicts={tls_id: tuple(sorted(values)) for tls_id, values in conflicts.items()},
    )


def add_routing_conflicts(
    graph: TlsInterventionGraph,
    adjacency: Mapping[str, Sequence[str]],
) -> TlsInterventionGraph:
    """Add directed next-signal links as undirected intervention conflicts."""

    conflicts = {
        str(tls_id): {str(value) for value in values}
        for tls_id, values in graph.conflicts.items()
    }
    for raw_source, raw_targets in adjacency.items():
        source = str(raw_source)
        conflicts.setdefault(source, set())
        for raw_target in raw_targets:
            target = str(raw_target)
            if target == source:
                continue
            conflicts.setdefault(target, set())
            conflicts[source].add(target)
            conflicts[target].add(source)
    return TlsInterventionGraph(
        neighborhoods=graph.neighborhoods,
        conflicts={name: tuple(sorted(values)) for name, values in conflicts.items()},
    )


def greedy_independent_interventions(
    priorities: Mapping[str, float],
    conflicts: Mapping[str, Sequence[str]],
    *,
    blocked: Sequence[str] = (),
    max_selected: int | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select a deterministic maximum-priority greedy independent set.

    Returns selected IDs and IDs rejected because they conflict with a selected
    intervention or exceed the optional global cap. IDs blocked by an active
    temporal cooldown should be removed by the caller before invoking this
    spatial selector.
    """

    blocked_set = {str(value) for value in blocked}
    ordered = sorted(
        (
            (float(priority), str(tls_id))
            for tls_id, priority in priorities.items()
            if str(tls_id) not in blocked_set
        ),
        key=lambda item: (-item[0], item[1]),
    )
    selected: list[str] = []
    rejected: list[str] = []
    limit = None if max_selected is None else max(int(max_selected), 0)
    for _, tls_id in ordered:
        if limit is not None and len(selected) >= limit:
            rejected.append(tls_id)
            continue
        tls_conflicts = {str(value) for value in conflicts.get(tls_id, ())}
        if any(selected_id in tls_conflicts for selected_id in selected):
            rejected.append(tls_id)
            continue
        selected.append(tls_id)
    return tuple(selected), tuple(rejected)
