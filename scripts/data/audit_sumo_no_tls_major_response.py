#!/usr/bin/env python3
"""Audit no-TLS major links that nevertheless declare yielding responses."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    _bit,
    _connection_identity,
    _junction_indices,
    _lane_id,
)
from scripts.data.audit_sumo_uncontrolled_major_right_of_way import (
    _external_no_tls,
)


PROTOCOL = "sumo-no-tls-major-response-consistency-audit-v1"


def _set_bits(value: str) -> list[int]:
    return [index for index in range(len(value)) if _bit(value, index) == "1"]


def audit_no_tls_major_response_consistency(
    network: Path,
    *,
    focus_junctions: Sequence[str] = (),
    maximum_examples: int = 20,
) -> dict[str, Any]:
    """Find executable major links that bypass declared right-of-way responses.

    SUMO treats an upper-case connection state as having priority and returns
    from ``MSLink::opened`` before checking ordinary foe links. A no-TLS
    connection with state ``M`` and a non-zero request ``response`` therefore
    contains contradictory executable instructions.
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
    by_junction_index: dict[tuple[str, int], ET.Element] = {}
    for connection, index in indices.items():
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        by_junction_index[(junction_id, int(index))] = connection

    focus = {str(value) for value in focus_junctions}
    junction_types: Counter[str] = Counter()
    candidate_junctions: set[str] = set()
    examples: list[dict[str, Any]] = []
    no_tls_count = 0
    major_count = 0
    unresolved_count = 0
    response_bit_count = 0
    response_without_foe_count = 0
    response_target_kinds: Counter[str] = Counter()
    response_target_states: Counter[str] = Counter()
    no_tls_response_bit_count = 0
    same_lane_no_tls_response_bit_count = 0
    asymmetric_no_tls_response_bit_count = 0
    mutual_no_tls_response_bit_count = 0
    no_tls_response_without_symmetric_foe_bit_count = 0
    candidate_with_no_tls_response_count = 0
    candidate_with_asymmetric_no_tls_response_count = 0
    candidate_with_mutual_no_tls_response_count = 0
    candidate_with_same_lane_no_tls_response_count = 0
    candidate_with_only_tls_response_count = 0
    candidate_with_mixed_response_kinds_count = 0
    mixed_tls_count = 0
    candidate_count = 0
    nonfocus_example_count = 0
    connections_by_junction: Counter[str] = Counter()
    tls_connections_by_junction: Counter[str] = Counter()
    for connection in connections:
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        if junction_id:
            connections_by_junction[junction_id] += 1
            tls_connections_by_junction[junction_id] += int(bool(connection.get("tl")))

    for connection in connections:
        if not _external_no_tls(connection):
            continue
        no_tls_count += 1
        if connection.get("state") != "M":
            continue
        major_count += 1
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        diagnostic = diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        index = indices.get(connection)
        if (
            not diagnostic.get("index_mapping_exact", False)
            or index is None
            or int(index) >= len(requests)
        ):
            unresolved_count += 1
            continue
        request = requests[int(index)]
        response = str(request.get("response", ""))
        response_indices = _set_bits(response)
        if not response_indices:
            continue
        foes = str(request.get("foes", ""))
        missing_foes = [
            value for value in response_indices if _bit(foes, value) != "1"
        ]
        candidate_count += 1
        response_bit_count += len(response_indices)
        response_without_foe_count += len(missing_foes)
        candidate_junctions.add(junction_id)
        junction_type = str(diagnostic.get("junction_type", ""))
        junction_types[junction_type] += 1
        has_tls = tls_connections_by_junction[junction_id] > 0
        mixed_tls_count += int(has_tls)
        response_relations: list[dict[str, Any]] = []
        candidate_has_no_tls_response = False
        candidate_has_asymmetric_no_tls_response = False
        candidate_has_mutual_no_tls_response = False
        candidate_has_same_lane_no_tls_response = False
        candidate_response_kinds: set[str] = set()
        receiving_lane = _lane_id(connection.attrib, "to")
        for response_index in response_indices:
            target = by_junction_index.get((junction_id, response_index))
            target_request = (
                requests[response_index]
                if 0 <= response_index < len(requests)
                else None
            )
            if target is None:
                kind = "unresolved"
            elif _external_no_tls(target):
                kind = "external_no_tls"
            elif target.get("tl"):
                kind = "tls_controlled"
            else:
                kind = "other"
            candidate_response_kinds.add(kind)
            response_target_kinds[kind] += 1
            if target is not None:
                response_target_states[str(target.get("state", ""))] += 1
            target_responds_back = bool(
                target_request is not None
                and _bit(str(target_request.get("response", "")), int(index))
                == "1"
            )
            target_declares_foe = bool(
                target_request is not None
                and _bit(str(target_request.get("foes", "")), int(index)) == "1"
            )
            same_receiving_lane = bool(
                target is not None
                and _lane_id(target.attrib, "to") == receiving_lane
            )
            if kind == "external_no_tls":
                no_tls_response_bit_count += 1
                candidate_has_no_tls_response = True
                no_tls_response_without_symmetric_foe_bit_count += int(
                    not target_declares_foe
                )
                if same_receiving_lane:
                    same_lane_no_tls_response_bit_count += 1
                    candidate_has_same_lane_no_tls_response = True
                if target_responds_back:
                    mutual_no_tls_response_bit_count += 1
                    candidate_has_mutual_no_tls_response = True
                else:
                    asymmetric_no_tls_response_bit_count += 1
                    candidate_has_asymmetric_no_tls_response = True
            response_relations.append(
                {
                    "junction_index": int(response_index),
                    "kind": kind,
                    "connection": (
                        _connection_identity(target) if target is not None else None
                    ),
                    "same_receiving_lane": same_receiving_lane,
                    "target_responds_back": target_responds_back,
                    "target_declares_foe": target_declares_foe,
                }
            )
        candidate_with_no_tls_response_count += int(
            candidate_has_no_tls_response
        )
        candidate_with_asymmetric_no_tls_response_count += int(
            candidate_has_asymmetric_no_tls_response
        )
        candidate_with_mutual_no_tls_response_count += int(
            candidate_has_mutual_no_tls_response
        )
        candidate_with_same_lane_no_tls_response_count += int(
            candidate_has_same_lane_no_tls_response
        )
        candidate_with_only_tls_response_count += int(
            candidate_response_kinds == {"tls_controlled"}
        )
        candidate_with_mixed_response_kinds_count += int(
            len(candidate_response_kinds) > 1
        )
        include_example = junction_id in focus or (
            nonfocus_example_count < int(maximum_examples)
        )
        if include_example:
            examples.append(
                {
                    "junction_id": junction_id,
                    "junction_type": junction_type,
                    "junction_index_mapping_exact": True,
                    "junction_index": int(index),
                    "junction_connection_count": int(
                        connections_by_junction[junction_id]
                    ),
                    "junction_has_tls_connections": has_tls,
                    "connection": _connection_identity(connection),
                    "response_indices": response_indices,
                    "response_without_foe_indices": missing_foes,
                    "response_connections": [
                        (
                            _connection_identity(target)
                            if (
                                target := by_junction_index.get(
                                    (junction_id, response_index)
                                )
                            )
                            is not None
                            else None
                        )
                        for response_index in response_indices
                    ],
                    "response_relations": response_relations,
                }
            )
            nonfocus_example_count += int(junction_id not in focus)

    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "counts": {
            "connection_count": len(connections),
            "external_no_tls_connection_count": no_tls_count,
            "external_no_tls_major_connection_count": major_count,
            "unresolved_major_connection_count": unresolved_count,
            "major_with_declared_response_connection_count": candidate_count,
            "candidate_junction_count": len(candidate_junctions),
            "declared_response_bit_count": response_bit_count,
            "response_without_foe_bit_count": response_without_foe_count,
            "response_target_kind_histogram": {
                key: int(value)
                for key, value in sorted(response_target_kinds.items())
            },
            "response_target_state_histogram": {
                key: int(value)
                for key, value in sorted(response_target_states.items())
            },
            "response_to_external_no_tls_bit_count": no_tls_response_bit_count,
            "response_to_same_receiving_lane_external_no_tls_bit_count": (
                same_lane_no_tls_response_bit_count
            ),
            "asymmetric_external_no_tls_response_bit_count": (
                asymmetric_no_tls_response_bit_count
            ),
            "mutual_external_no_tls_response_bit_count": (
                mutual_no_tls_response_bit_count
            ),
            "external_no_tls_response_without_symmetric_foe_bit_count": (
                no_tls_response_without_symmetric_foe_bit_count
            ),
            "candidate_with_external_no_tls_response_count": (
                candidate_with_no_tls_response_count
            ),
            "candidate_with_asymmetric_external_no_tls_response_count": (
                candidate_with_asymmetric_no_tls_response_count
            ),
            "candidate_with_mutual_external_no_tls_response_count": (
                candidate_with_mutual_no_tls_response_count
            ),
            "candidate_with_same_receiving_lane_external_no_tls_response_count": (
                candidate_with_same_lane_no_tls_response_count
            ),
            "candidate_with_only_tls_controlled_response_count": (
                candidate_with_only_tls_response_count
            ),
            "candidate_with_mixed_response_kinds_count": (
                candidate_with_mixed_response_kinds_count
            ),
            "candidate_at_mixed_tls_junction_count": mixed_tls_count,
            "candidate_junction_type_histogram": {
                key: int(value) for key, value in sorted(junction_types.items())
            },
        },
        "examples": examples,
        "audit_boundary": (
            "The audit reports external no-TLS state=M connections whose mapped "
            "junction request has at least one response bit. It does not infer "
            "controller efficacy or edit the network."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--focus-junction", action="append", default=[])
    parser.add_argument("--maximum-examples", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit_no_tls_major_response_consistency(
        args.network,
        focus_junctions=args.focus_junction,
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
