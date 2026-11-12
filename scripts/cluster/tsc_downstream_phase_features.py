"""Six frozen action-linked downstream features from pre-action snapshots only."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


FEATURE_NAMES = (
    "ds_mapped_fraction", "ds_protected_fraction", "ds_permissive_fraction",
    "ds_clearance_fraction", "ds_open_green_age_sec", "ds_clearance_remaining_sec",
)


@dataclass
class NetworkTopology:
    successors: dict[str, tuple[str, ...]]
    terminal_lanes: frozenset[str]
    incoming_owner: dict[str, str]
    receiving_indices: dict[tuple[str, str], tuple[int, ...]]
    tls_connections: dict[str, tuple[tuple[str, str, int], ...]]
    tls_width: dict[str, int]

    @classmethod
    def from_net(cls, path: Path) -> "NetworkTopology":
        root = ET.parse(path).getroot()
        lanes, lane_edge, edge_nodes = {}, {}, {}
        outgoing_edges = defaultdict(set)
        for edge in root.findall("edge"):
            edge_id = edge.attrib["id"]
            if "from" in edge.attrib and "to" in edge.attrib:
                edge_nodes[edge_id] = (edge.attrib["from"], edge.attrib["to"])
                outgoing_edges[edge.attrib["from"]].add(edge_id)
            for lane in edge.findall("lane"):
                lane_id = lane.attrib["id"]
                lanes[(edge_id, int(lane.attrib["index"]))] = lane_id
                lane_edge[lane_id] = edge_id
        junction_types = {j.attrib["id"]: j.attrib["type"] for j in root.findall("junction")}
        terminal = frozenset(
            lane for lane, edge in lane_edge.items()
            if edge in edge_nodes and (
                junction_types.get(edge_nodes[edge][1]) == "dead_end"
                or not outgoing_edges[edge_nodes[edge][1]]
            )
        )
        successors = {lane: set() for lane in lane_edge}
        owners, indices, connections = {}, defaultdict(set), defaultdict(set)
        for connection in root.findall("connection"):
            data = connection.attrib
            try:
                incoming = lanes[(data["from"], int(data["fromLane"]))]
                outgoing = lanes[(data["to"], int(data["toLane"]))]
            except KeyError as exc:
                raise ValueError(f"connection references an absent edge/lane: {data}") from exc
            # lane.getLinks()[0] follows the approached lane, not the via lane.
            successors[incoming].add(outgoing)
            if "tl" not in data:
                continue
            tls, index = data["tl"], int(data["linkIndex"])
            if incoming in owners and owners[incoming] != tls:
                raise ValueError(f"receiving lane {incoming} has multiple TLS owners")
            owners[incoming] = tls
            indices[(tls, incoming)].add(index)
            connections[tls].add((incoming, outgoing, index))
        widths = {tls: 1 + max(i for _, _, i in links) for tls, links in connections.items()}
        for logic in root.findall("tlLogic"):
            tls = logic.attrib["id"]
            if tls not in widths:
                continue
            sizes = {len(phase.attrib["state"]) for phase in logic.findall("phase")}
            if sizes != {widths[tls]}:
                raise ValueError(f"TLS {tls}: phase strings and linkIndex width disagree")
        return cls(
            {lane: tuple(sorted(values)) for lane, values in successors.items()}, terminal, owners,
            {key: tuple(sorted(values)) for key, values in indices.items()},
            {tls: tuple(sorted(values)) for tls, values in connections.items()}, widths,
        )

    def receiving_entries(self, start_lane: str, source_tls: str) -> tuple[tuple[str, str], ...]:
        def visit(lane, depth, ancestors):
            owner = self.incoming_owner.get(lane)
            if owner is not None and owner != source_tls:
                return {(owner, lane)}
            if lane in ancestors:
                raise ValueError(f"unresolved nonterminal cycle from {start_lane} through {lane}")
            following = self.successors[lane]
            if not following:
                if lane in self.terminal_lanes:
                    return set()
                raise ValueError(f"unresolved nonterminal lane {lane}")
            if depth >= 24:
                raise ValueError(f"unresolved nonterminal path beyond 24 hops from {start_lane}")
            targets = set()
            for next_lane in following:
                targets.update(visit(next_lane, depth + 1, ancestors | {lane}))
            return targets

        return tuple(sorted(visit(start_lane, 0, frozenset())))


def features_for_group(topology, snapshot, tls_id, candidate_states):
    """Return candidate rows and mapping counts; no post-action fields are read."""
    executors = snapshot["executors"]
    checked = {}

    def executor(tls):
        if tls not in checked:
            if tls not in executors:
                raise ValueError(f"missing pre-action executor {tls}")
            value = executors[tls]
            if len(value["current_state"]) != topology.tls_width[tls]:
                raise ValueError(f"TLS {tls}: snapshot state/linkIndex width mismatch")
            if value["mode"] not in {"green", "yellow", "all_red"}:
                raise ValueError(f"TLS {tls}: unknown executor mode")
            checked[tls] = value
        return checked[tls]

    executor(tls_id)
    mappings, rows, counts = {}, [], []
    for state in candidate_states:
        if len(state) != topology.tls_width[tls_id]:
            raise ValueError(f"TLS {tls_id}: candidate state/linkIndex width mismatch")
        active = sorted({(incoming, outgoing) for incoming, outgoing, index
                         in topology.tls_connections[tls_id] if state[index] in "Gg"})
        if not active:
            raise ValueError(f"TLS {tls_id}: candidate has no enabled connection")
        row = np.zeros(len(FEATURE_NAMES), dtype=float)
        mapped_count, receiving_count = 0, 0
        for _, outgoing in active:
            if outgoing not in mappings:
                mappings[outgoing] = topology.receiving_entries(outgoing, tls_id)
            targets = mappings[outgoing]
            if not targets:
                continue
            mapped_count += 1
            receiving_count += len(targets)
            weight = 1.0 / (len(active) * len(targets))
            for target_tls, lane in targets:
                value = executor(target_tls)
                signal = value["current_state"]
                links = topology.receiving_indices[(target_tls, lane)]
                protected = sum(signal[index] == "G" for index in links) / len(links)
                permissive = sum(signal[index] == "g" for index in links) / len(links)
                clearing = float(value["mode"] != "green")
                remaining = float(value["remaining_sec"]) if clearing else 0.0
                if value["mode"] == "yellow":
                    remaining += float(value["pending_all_red_sec"])
                row += weight * np.array([
                    1.0, protected, permissive, clearing,
                    (protected + permissive) * float(value["green_elapsed_sec"]),
                    clearing * remaining,
                ])
        rows.append(row)
        counts.append({"enabled_connections": len(active), "mapped_connections": mapped_count,
                       "exit_connections": len(active) - mapped_count,
                       "receiving_entries_with_connection_multiplicity": receiving_count})
    features = np.asarray(rows)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite pre-action downstream phase features")
    return features, {"candidates": counts, "used_downstream_tls": sorted(set(checked) - {tls_id})}
