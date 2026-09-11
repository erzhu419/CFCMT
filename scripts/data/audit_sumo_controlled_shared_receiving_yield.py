#!/usr/bin/env python3
"""Audit TLS yield relations for co-active movements sharing a receiving lane."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


PROTOCOL = "sumo-controlled-shared-receiving-yield-audit-v1"
YIELDING_SIGNALS = frozenset({"g", "s"})
PROTECTED_SIGNAL = "G"


def _programs(root: ET.Element) -> dict[str, tuple[ET.Element, ...]]:
    grouped: dict[str, list[ET.Element]] = defaultdict(list)
    for program in root.iter("tlLogic"):
        tls_id = str(program.get("id", ""))
        if tls_id:
            grouped[tls_id].append(program)
    return {key: tuple(values) for key, values in grouped.items()}


def _coactive_requirements(
    left: ET.Element,
    right: ET.Element,
    programs: Mapping[str, Sequence[ET.Element]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_tls = str(left.get("tl", ""))
    right_tls = str(right.get("tl", ""))
    if not left_tls or left_tls != right_tls:
        return [], []
    left_link = int(str(left.get("linkIndex")))
    right_link = int(str(right.get("linkIndex")))
    requirements: list[dict[str, Any]] = []
    simultaneous_protected: list[dict[str, Any]] = []
    for program in programs.get(left_tls, ()):
        program_id = str(program.get("programID", ""))
        for phase_index, phase in enumerate(program.findall("phase")):
            state = str(phase.get("state", ""))
            if max(left_link, right_link) >= len(state):
                raise ValueError(
                    f"TLS state is too short: {left_tls}/{program_id}/{phase_index}"
                )
            left_signal = state[left_link]
            right_signal = state[right_link]
            phase_row = {
                "tls_id": left_tls,
                "program_id": program_id,
                "phase_index": phase_index,
                "state": state,
                "left_signal": left_signal,
                "right_signal": right_signal,
            }
            if left_signal == PROTECTED_SIGNAL and right_signal in YIELDING_SIGNALS:
                requirements.append({**phase_row, "yielding_side": "right"})
            elif right_signal == PROTECTED_SIGNAL and left_signal in YIELDING_SIGNALS:
                requirements.append({**phase_row, "yielding_side": "left"})
            elif left_signal == right_signal == PROTECTED_SIGNAL:
                simultaneous_protected.append(phase_row)
    return requirements, simultaneous_protected


def audit_controlled_shared_receiving_yield(
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
    programs = _programs(root)
    receiving: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        if (
            connection.get("tl")
            and connection.get("linkIndex") not in {None, ""}
            and not str(connection.get("from", "")).startswith(":")
        ):
            receiving[_lane_id(connection.attrib, "to")].append(connection)

    focus = {str(value) for value in focus_receiving_lanes}
    shared_pair_count = 0
    cross_tls_pair_count = 0
    coactive_pair_count = 0
    directed_requirement_count = 0
    missing_response_count = 0
    missing_symmetric_foe_count = 0
    simultaneous_protected_phase_count = 0
    unresolved_pair_count = 0
    affected_receivers: set[str] = set()
    affected_junctions: set[str] = set()
    examples: list[dict[str, Any]] = []
    directed_requirements: set[tuple[str, int, int]] = set()

    for receiving_lane, rows in sorted(receiving.items()):
        if len(rows) < 2:
            continue
        for left_offset, left in enumerate(rows):
            for right in rows[left_offset + 1 :]:
                shared_pair_count += 1
                if str(left.get("tl")) != str(right.get("tl")):
                    cross_tls_pair_count += 1
                    continue
                requirements, simultaneous = _coactive_requirements(
                    left,
                    right,
                    programs,
                )
                simultaneous_protected_phase_count += len(simultaneous)
                if not requirements and not simultaneous:
                    continue
                coactive_pair_count += 1
                junction_id = str(edges_from.get(str(left.get("to", "")), ""))
                diagnostic = diagnostics.get(junction_id, {})
                requests = list(diagnostic.get("requests", ()))
                left_index = indices.get(left)
                right_index = indices.get(right)
                if (
                    not diagnostic.get("index_mapping_exact", False)
                    or left_index is None
                    or right_index is None
                    or max(left_index, right_index) >= len(requests)
                ):
                    unresolved_pair_count += 1
                    continue
                pair_missing_response = 0
                pair_missing_foe = 0
                public_requirements = []
                for row in requirements:
                    if row["yielding_side"] == "left":
                        yielding_index, protected_index = left_index, right_index
                    else:
                        yielding_index, protected_index = right_index, left_index
                    key = (junction_id, int(yielding_index), int(protected_index))
                    if key in directed_requirements:
                        continue
                    directed_requirements.add(key)
                    directed_requirement_count += 1
                    yielding_request = requests[yielding_index]
                    protected_request = requests[protected_index]
                    response_present = (
                        _bit(
                            str(yielding_request.get("response", "")),
                            protected_index,
                        )
                        == "1"
                    )
                    symmetric_foe = (
                        _bit(str(yielding_request.get("foes", "")), protected_index)
                        == "1"
                        and _bit(
                            str(protected_request.get("foes", "")), yielding_index
                        )
                        == "1"
                    )
                    missing_response_count += int(not response_present)
                    missing_symmetric_foe_count += int(not symmetric_foe)
                    pair_missing_response += int(not response_present)
                    pair_missing_foe += int(not symmetric_foe)
                    public_requirements.append(
                        {
                            **row,
                            "yielding_junction_index": int(yielding_index),
                            "protected_junction_index": int(protected_index),
                            "yield_response_present": response_present,
                            "symmetric_foe_declared": symmetric_foe,
                        }
                    )
                if pair_missing_response or pair_missing_foe or simultaneous:
                    affected_receivers.add(receiving_lane)
                    affected_junctions.add(junction_id)
                if (
                    receiving_lane in focus
                    or pair_missing_response
                    or pair_missing_foe
                    or simultaneous
                ) and len(examples) < int(maximum_examples):
                    examples.append(
                        {
                            "receiving_lane": receiving_lane,
                            "junction_id": junction_id,
                            "left_junction_index": int(left_index),
                            "right_junction_index": int(right_index),
                            "left_connection": _connection_identity(left),
                            "right_connection": _connection_identity(right),
                            "directed_requirements": public_requirements,
                            "simultaneous_protected_phases": simultaneous,
                        }
                    )

    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "counts": {
            "controlled_shared_receiving_pair_count": shared_pair_count,
            "cross_tls_shared_receiving_pair_count": cross_tls_pair_count,
            "coactive_same_tls_pair_count": coactive_pair_count,
            "directed_yield_requirement_count": directed_requirement_count,
            "missing_yield_response_count": missing_response_count,
            "missing_symmetric_foe_count": missing_symmetric_foe_count,
            "simultaneous_protected_phase_count": (
                simultaneous_protected_phase_count
            ),
            "unresolved_junction_index_pair_count": unresolved_pair_count,
            "affected_receiving_lane_count": len(affected_receivers),
            "affected_junction_count": len(affected_junctions),
        },
        "examples": examples,
        "audit_boundary": (
            "The audit checks whether g/s links that share a receiving lane "
            "with a concurrently protected G link declare the required SUMO "
            "foe and response relations. It is read-only and does not infer "
            "controller efficacy."
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
    result = audit_controlled_shared_receiving_yield(
        args.network,
        focus_receiving_lanes=args.focus_receiving_lane,
        maximum_examples=args.maximum_examples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "DONE", "counts": result["counts"], "result": str(args.out)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
