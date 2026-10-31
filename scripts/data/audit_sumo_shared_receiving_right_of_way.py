#!/usr/bin/env python3
"""Audit executable priority for controlled/uncontrolled receiving-lane merges."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "sumo-shared-receiving-right-of-way-audit-v1"


def _lane_id(connection: Mapping[str, str], prefix: str) -> str:
    return f"{connection.get(prefix, '')}_{connection.get(prefix + 'Lane', '')}"


def _bit(value: str, junction_index: int) -> str | None:
    text = str(value)
    offset = len(text) - 1 - int(junction_index)
    return text[offset] if 0 <= offset < len(text) else None


def _connection_identity(element: ET.Element) -> dict[str, str]:
    return {
        key: str(element.attrib[key])
        for key in (
            "from",
            "fromLane",
            "to",
            "toLane",
            "via",
            "tl",
            "linkIndex",
            "state",
            "dir",
            "uncontrolled",
        )
        if element.attrib.get(key) not in {None, ""}
    }


def _junction_indices(
    root: ET.Element,
    *,
    edges_from: Mapping[str, str],
    connections: Sequence[ET.Element],
) -> tuple[dict[ET.Element, int], dict[str, dict[str, Any]]]:
    by_junction: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        if junction_id:
            by_junction[junction_id].append(connection)

    indices: dict[ET.Element, int] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for junction in root.iter("junction"):
        junction_id = str(junction.get("id", ""))
        rows = by_junction.get(junction_id, [])
        if not rows:
            continue
        incoming_lanes = tuple(
            value for value in str(junction.get("incLanes", "")).split() if value
        )
        ordered: list[ET.Element] = []
        used: set[ET.Element] = set()
        for lane in incoming_lanes:
            for connection in rows:
                if connection not in used and _lane_id(connection.attrib, "from") == lane:
                    ordered.append(connection)
                    used.add(connection)
        unresolved = [connection for connection in rows if connection not in used]
        requests = list(junction.findall("request"))
        exact = not unresolved and len(ordered) == len(requests)
        if exact:
            indices.update({connection: index for index, connection in enumerate(ordered)})
        diagnostics[junction_id] = {
            "junction_type": str(junction.get("type", "")),
            "incoming_lane_count": len(incoming_lanes),
            "internal_lane_count": len(
                [value for value in str(junction.get("intLanes", "")).split() if value]
            ),
            "connection_count": len(rows),
            "request_count": len(requests),
            "unresolved_connection_count": len(unresolved),
            "index_mapping_exact": exact,
            "requests": requests,
        }
    return indices, diagnostics


def audit_shared_receiving_right_of_way(
    network: Path,
    *,
    focus_receiving_lanes: Sequence[str] = (),
    maximum_examples: int = 20,
) -> dict[str, Any]:
    network = Path(network)
    if not network.is_file():
        raise FileNotFoundError(network)
    if int(maximum_examples) < 0:
        raise ValueError("maximum_examples must be nonnegative")

    tree = ET.parse(network)
    root = tree.getroot()
    edges_from = {
        str(edge.get("id")): str(edge.get("from"))
        for edge in root.iter("edge")
        if edge.get("id") and edge.get("from")
    }
    junctions = {
        str(junction.get("id")): junction for junction in root.iter("junction")
    }
    connections = list(root.iter("connection"))
    junction_indices, junction_diagnostics = _junction_indices(
        root,
        edges_from=edges_from,
        connections=connections,
    )
    programs: dict[str, list[ET.Element]] = defaultdict(list)
    for program in root.iter("tlLogic"):
        programs[str(program.get("id", ""))].append(program)

    receiving_groups: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        receiving_groups[_lane_id(connection.attrib, "to")].append(connection)

    pair_count = 0
    exact_pair_count = 0
    unresolved_pair_count = 0
    controlled_missing_yield_count = 0
    major_wrongly_yields_count = 0
    missing_symmetric_foe_count = 0
    protected_pair_count = 0
    relation_change_pairs: set[tuple[str, int, int]] = set()
    phase_change_keys: set[tuple[str, str, int, int]] = set()
    candidate_junctions: set[str] = set()
    candidate_receivers: set[str] = set()
    focus = {str(value) for value in focus_receiving_lanes}
    examples: list[dict[str, Any]] = []

    for receiving_lane, rows in sorted(receiving_groups.items()):
        controlled = [row for row in rows if row.get("tl") and row.get("linkIndex")]
        major = [
            row
            for row in rows
            if not row.get("tl")
            and not str(row.get("from", "")).startswith(":")
            and (row.get("state") == "M" or row.get("uncontrolled") == "1")
        ]
        if not controlled or not major:
            continue
        junction_id = str(edges_from.get(str(rows[0].get("to", "")), ""))
        candidate_junctions.add(junction_id)
        candidate_receivers.add(receiving_lane)
        diagnostic = junction_diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        for controlled_connection in controlled:
            for major_connection in major:
                pair_count += 1
                controlled_index = junction_indices.get(controlled_connection)
                major_index = junction_indices.get(major_connection)
                if (
                    controlled_index is None
                    or major_index is None
                    or controlled_index >= len(requests)
                    or major_index >= len(requests)
                ):
                    unresolved_pair_count += 1
                    relation = {
                        "controlled_yields_to_major": None,
                        "major_yields_to_controlled": None,
                        "symmetric_foe_declared": None,
                    }
                    protected = None
                else:
                    controlled_request = requests[controlled_index]
                    major_request = requests[major_index]
                    controlled_yields = (
                        _bit(str(controlled_request.get("response", "")), major_index)
                        == "1"
                    )
                    major_yields = (
                        _bit(str(major_request.get("response", "")), controlled_index)
                        == "1"
                    )
                    symmetric_foe = (
                        _bit(str(controlled_request.get("foes", "")), major_index)
                        == "1"
                        and _bit(str(major_request.get("foes", "")), controlled_index)
                        == "1"
                    )
                    controlled_missing_yield_count += int(not controlled_yields)
                    major_wrongly_yields_count += int(major_yields)
                    missing_symmetric_foe_count += int(not symmetric_foe)
                    relation_change_pairs.add(
                        (junction_id, int(controlled_index), int(major_index))
                    )
                    relation = {
                        "controlled_yields_to_major": controlled_yields,
                        "major_yields_to_controlled": major_yields,
                        "symmetric_foe_declared": symmetric_foe,
                    }

                    protected = False
                    tls_id = str(controlled_connection.get("tl", ""))
                    link_index = int(str(controlled_connection.get("linkIndex")))
                    for program in programs.get(tls_id, ()):
                        program_id = str(program.get("programID", ""))
                        for phase_index, phase in enumerate(program.findall("phase")):
                            state = str(phase.get("state", ""))
                            if link_index < len(state) and state[link_index] == "G":
                                protected = True
                                phase_change_keys.add(
                                    (tls_id, program_id, phase_index, link_index)
                                )
                    protected_pair_count += int(protected)
                    exact_pair_count += int(
                        controlled_yields and not major_yields and symmetric_foe and not protected
                    )

                needs_example = (
                    receiving_lane in focus
                    or relation["controlled_yields_to_major"] is not True
                    or relation["major_yields_to_controlled"] is not False
                    or relation["symmetric_foe_declared"] is not True
                    or protected is not False
                )
                if needs_example and len(examples) < int(maximum_examples):
                    examples.append(
                        {
                            "receiving_lane": receiving_lane,
                            "junction_id": junction_id,
                            "junction_index_mapping_exact": bool(
                                diagnostic.get("index_mapping_exact", False)
                            ),
                            "internal_lane_count": int(
                                diagnostic.get("internal_lane_count", 0)
                            ),
                            "controlled_junction_index": controlled_index,
                            "major_junction_index": major_index,
                            "controlled_connection": _connection_identity(
                                controlled_connection
                            ),
                            "uncontrolled_major_connection": _connection_identity(
                                major_connection
                            ),
                            "relation": relation,
                            "controlled_has_protected_service": protected,
                        }
                    )

    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "counts": {
            "connection_count": len(connections),
            "candidate_receiving_lane_count": len(candidate_receivers),
            "candidate_junction_count": len(candidate_junctions),
            "controlled_uncontrolled_major_pair_count": pair_count,
            "already_exact_pair_count": exact_pair_count,
            "unresolved_junction_index_pair_count": unresolved_pair_count,
            "controlled_missing_yield_pair_count": controlled_missing_yield_count,
            "major_wrongly_yields_pair_count": major_wrongly_yields_count,
            "missing_symmetric_foe_pair_count": missing_symmetric_foe_count,
            "protected_service_pair_count": protected_pair_count,
            "unique_request_relation_change_count": len(relation_change_pairs),
            "unique_tls_phase_character_change_count": len(phase_change_keys),
        },
        "examples": examples,
        "audit_boundary": (
            "The audit checks executable junction response bits and TLS service. "
            "It does not modify the network or infer controller efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--focus-receiving-lane", action="append", default=[])
    parser.add_argument("--maximum-examples", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit_shared_receiving_right_of_way(
        args.network,
        focus_receiving_lanes=args.focus_receiving_lane,
        maximum_examples=args.maximum_examples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "DONE", "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
