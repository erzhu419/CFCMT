#!/usr/bin/env python3
"""Audit duplicate uncontrolled major movements sharing a receiving lane."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    _bit,
    _connection_identity,
    _junction_indices,
    _lane_id,
)


PROTOCOL = "sumo-uncontrolled-duplicate-major-right-of-way-audit-v1"
IMPLICIT_NO_TLS_PROTOCOL = (
    "sumo-implicit-no-tls-duplicate-major-right-of-way-audit-v1"
)


def _external_uncontrolled(connection: ET.Element) -> bool:
    return bool(
        not connection.get("tl")
        and not str(connection.get("from", "")).startswith(":")
        and connection.get("uncontrolled") == "1"
    )


def _external_no_tls(connection: ET.Element) -> bool:
    """Return external movements that SUMO executes without a TLS link."""

    return bool(
        not connection.get("tl")
        and not str(connection.get("from", "")).startswith(":")
    )


def audit_uncontrolled_major_right_of_way(
    network: Path,
    *,
    focus_receiving_lanes: Sequence[str] = (),
    maximum_examples: int = 20,
    include_implicit_no_tls: bool = False,
) -> dict[str, Any]:
    """Find pairs that are both marked major despite an existing yield relation.

    The legacy mode only audits connections carrying ``uncontrolled=\"1\"``.
    The implicit mode also covers external connections with no ``tl`` attribute,
    which is how the Boston v15 collision movements are encoded.
    """

    network = Path(network)
    if not network.is_file():
        raise FileNotFoundError(network)
    if int(maximum_examples) < 0:
        raise ValueError("maximum_examples must be nonnegative")

    root = ET.parse(network).getroot()
    edges_from = {
        str(edge.get("id")): str(edge.get("from"))
        for edge in root.iter("edge")
        if edge.get("id") and edge.get("from")
    }
    connections = list(root.iter("connection"))
    indices, diagnostics = _junction_indices(
        root,
        edges_from=edges_from,
        connections=connections,
    )
    external_predicate = (
        _external_no_tls if include_implicit_no_tls else _external_uncontrolled
    )
    receiving: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        if external_predicate(connection):
            receiving[_lane_id(connection.attrib, "to")].append(connection)

    connections_by_junction: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        if junction_id:
            connections_by_junction[junction_id].append(connection)

    focus = {str(value) for value in focus_receiving_lanes}
    candidate_groups = 0
    candidate_pairs = 0
    unresolved_pairs = 0
    missing_symmetric_foes = 0
    ambiguous_response_pairs = 0
    actionable_pairs = 0
    implicit_attribute_pairs = 0
    traffic_light_junction_pairs = 0
    mixed_tls_junction_pairs = 0
    major_multiplicity: Counter[int] = Counter()
    unique_priority_groups = 0
    ambiguous_priority_groups = 0
    planned_state_changes = 0
    candidate_junctions: set[str] = set()
    examples: list[dict[str, Any]] = []
    for receiving_lane, rows in sorted(receiving.items()):
        majors = [row for row in rows if row.get("state") == "M"]
        if len(majors) < 2:
            continue
        candidate_groups += 1
        major_multiplicity[len(majors)] += 1
        junction_id = str(edges_from.get(str(rows[0].get("to", "")), ""))
        candidate_junctions.add(junction_id)
        diagnostic = diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        yielding_to_peer_count = {row: 0 for row in majors}
        group_actionable = True
        for left_offset, left in enumerate(majors):
            for right in majors[left_offset + 1 :]:
                candidate_pairs += 1
                implicit_attribute_pair = bool(
                    left.get("uncontrolled") != "1"
                    or right.get("uncontrolled") != "1"
                )
                implicit_attribute_pairs += int(implicit_attribute_pair)
                junction_type = str(diagnostic.get("junction_type", ""))
                traffic_light_junction_pairs += int(
                    junction_type.startswith("traffic_light")
                )
                mixed_tls_junction = any(
                    row.get("tl")
                    for row in connections_by_junction.get(junction_id, ())
                )
                mixed_tls_junction_pairs += int(mixed_tls_junction)
                left_index = indices.get(left)
                right_index = indices.get(right)
                relation: dict[str, Any]
                if (
                    left_index is None
                    or right_index is None
                    or left_index >= len(requests)
                    or right_index >= len(requests)
                ):
                    unresolved_pairs += 1
                    relation = {
                        "symmetric_foes": None,
                        "left_yields_to_right": None,
                        "right_yields_to_left": None,
                        "actionable": False,
                    }
                else:
                    left_request = requests[left_index]
                    right_request = requests[right_index]
                    symmetric_foes = bool(
                        _bit(str(left_request.get("foes", "")), right_index) == "1"
                        and _bit(str(right_request.get("foes", "")), left_index)
                        == "1"
                    )
                    left_yields = bool(
                        _bit(str(left_request.get("response", "")), right_index)
                        == "1"
                    )
                    right_yields = bool(
                        _bit(str(right_request.get("response", "")), left_index)
                        == "1"
                    )
                    one_way_response = left_yields != right_yields
                    missing_symmetric_foes += int(not symmetric_foes)
                    ambiguous_response_pairs += int(not one_way_response)
                    actionable = symmetric_foes and one_way_response
                    actionable_pairs += int(actionable)
                    if actionable:
                        yielding_to_peer_count[left if left_yields else right] += 1
                    else:
                        group_actionable = False
                    relation = {
                        "symmetric_foes": symmetric_foes,
                        "left_yields_to_right": left_yields,
                        "right_yields_to_left": right_yields,
                        "actionable": actionable,
                    }
                if (
                    receiving_lane in focus
                    or not relation["actionable"]
                    or len(examples) < int(maximum_examples)
                ) and len(examples) < int(maximum_examples):
                    examples.append(
                        {
                            "receiving_lane": receiving_lane,
                            "junction_id": junction_id,
                            "junction_type": diagnostic.get("junction_type"),
                            "implicit_uncontrolled_attribute_pair": (
                                implicit_attribute_pair
                            ),
                            "junction_has_other_tls_connections": bool(
                                mixed_tls_junction
                            ),
                            "junction_index_mapping_exact": bool(
                                diagnostic.get("index_mapping_exact", False)
                            ),
                            "left_junction_index": left_index,
                            "right_junction_index": right_index,
                            "left_connection": _connection_identity(left),
                            "right_connection": _connection_identity(right),
                            "relation": relation,
                        }
                    )
        priority_rows = [
            row for row, count in yielding_to_peer_count.items() if count == 0
        ]
        if group_actionable and len(priority_rows) == 1:
            unique_priority_groups += 1
            planned_state_changes += len(majors) - 1
        else:
            ambiguous_priority_groups += 1

    counts = {
        "connection_count": len(connections),
        "duplicate_major_receiving_lane_count": candidate_groups,
        "duplicate_major_junction_count": len(candidate_junctions),
        "duplicate_major_pair_count": candidate_pairs,
        "unresolved_junction_index_pair_count": unresolved_pairs,
        "missing_symmetric_foe_pair_count": missing_symmetric_foes,
        "ambiguous_response_pair_count": ambiguous_response_pairs,
        "actionable_pair_count": actionable_pairs,
    }
    if include_implicit_no_tls:
        no_tls_rows = [row for row in connections if _external_no_tls(row)]
        counts.update(
            {
                "external_no_tls_connection_count": len(no_tls_rows),
                "explicit_uncontrolled_connection_count": sum(
                    row.get("uncontrolled") == "1" for row in no_tls_rows
                ),
                "implicit_uncontrolled_connection_count": sum(
                    row.get("uncontrolled") != "1" for row in no_tls_rows
                ),
                "pair_with_implicit_uncontrolled_attribute_count": (
                    implicit_attribute_pairs
                ),
                "traffic_light_junction_pair_count": (
                    traffic_light_junction_pairs
                ),
                "mixed_tls_junction_pair_count": mixed_tls_junction_pairs,
                "major_multiplicity_histogram": {
                    str(key): int(value)
                    for key, value in sorted(major_multiplicity.items())
                },
                "unique_priority_group_count": unique_priority_groups,
                "ambiguous_priority_group_count": ambiguous_priority_groups,
                "planned_connection_state_change_count": planned_state_changes,
            }
        )

    return {
        "protocol": (
            IMPLICIT_NO_TLS_PROTOCOL if include_implicit_no_tls else PROTOCOL
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "counts": counts,
        "examples": examples,
        "audit_boundary": (
            "The audit reports external no-TLS connections that share a receiving "
            "lane and are simultaneously marked state=M. Legacy mode requires an "
            "explicit uncontrolled=1 marker; implicit mode does not. It uses the "
            "existing junction foe/response relation and never edits the network."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--focus-receiving-lane", action="append", default=[])
    parser.add_argument("--maximum-examples", type=int, default=20)
    parser.add_argument("--include-implicit-no-tl", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit_uncontrolled_major_right_of_way(
        args.network,
        focus_receiving_lanes=args.focus_receiving_lane,
        maximum_examples=args.maximum_examples,
        include_implicit_no_tls=args.include_implicit_no_tl,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "DONE", "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
