#!/usr/bin/env python3
"""Audit SUMO connections that share a receiving lane across priority regimes.

The audit is read-only.  It identifies topology candidates that require
microscopic review; it does not infer a safe repair or modify the network.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "sumo-shared-receiving-priority-conflict-audit-v1"


def _tag(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _bit_count(value: str) -> int:
    return sum(char == "1" for char in str(value))


def _connection_identity(attributes: Mapping[str, str]) -> dict[str, str]:
    keys = (
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
    return {
        key: str(attributes[key])
        for key in keys
        if attributes.get(key) not in {None, ""}
    }


def _read_network(
    network: Path,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[dict[str, str]],
    set[str],
]:
    edges: dict[str, dict[str, str]] = {}
    programs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    junctions: dict[str, dict[str, Any]] = {}
    connections: list[dict[str, str]] = []
    lane_ids: set[str] = set()
    current_program: dict[str, Any] | None = None
    current_junction: dict[str, Any] | None = None

    for event, element in ET.iterparse(network, events=("start", "end")):
        name = _tag(element)
        if event == "start":
            if name == "edge":
                edge_id = str(element.get("id", ""))
                if edge_id:
                    edges[edge_id] = {
                        key: str(element.get(key, ""))
                        for key in ("id", "from", "to", "function")
                    }
            elif name == "lane":
                lane_id = str(element.get("id", ""))
                if lane_id:
                    lane_ids.add(lane_id)
            elif name == "tlLogic":
                current_program = {
                    "tls_id": str(element.get("id", "")),
                    "program_id": str(element.get("programID", "")),
                    "type": str(element.get("type", "")),
                    "phases": [],
                }
            elif name == "junction":
                current_junction = {
                    "id": str(element.get("id", "")),
                    "type": str(element.get("type", "")),
                    "inc_lanes": tuple(
                        value
                        for value in str(element.get("incLanes", "")).split()
                        if value
                    ),
                    "internal_lanes": tuple(
                        value
                        for value in str(element.get("intLanes", "")).split()
                        if value
                    ),
                    "requests": [],
                }
            elif name == "connection":
                connections.append(_connection_identity(element.attrib))
        else:
            if name == "phase" and current_program is not None:
                current_program["phases"].append(
                    {
                        "phase_index": len(current_program["phases"]),
                        "state": str(element.get("state", "")),
                        "duration": float(element.get("duration", 0.0) or 0.0),
                    }
                )
            elif name == "tlLogic" and current_program is not None:
                tls_id = str(current_program["tls_id"])
                if tls_id:
                    programs[tls_id].append(current_program)
                current_program = None
            elif name == "request" and current_junction is not None:
                current_junction["requests"].append(
                    {
                        "index": str(element.get("index", "")),
                        "response": str(element.get("response", "")),
                        "foes": str(element.get("foes", "")),
                        "cont": str(element.get("cont", "")),
                    }
                )
            elif name == "junction" and current_junction is not None:
                junction_id = str(current_junction["id"])
                if junction_id:
                    junctions[junction_id] = current_junction
                current_junction = None
            element.clear()
    return edges, dict(programs), junctions, connections, lane_ids


def _service_profile(
    connection: Mapping[str, str],
    programs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tls_id = str(connection.get("tl", ""))
    raw_index = connection.get("linkIndex")
    if not tls_id or raw_index in {None, ""}:
        return {
            "controlled": False,
            "protected_phase_count": 0,
            "yielding_phase_count": 0,
            "active_phase_count": 0,
            "service_phases": [],
        }, []
    anomalies = []
    try:
        link_index = int(str(raw_index))
    except ValueError:
        return {
            "controlled": True,
            "protected_phase_count": 0,
            "yielding_phase_count": 0,
            "active_phase_count": 0,
            "service_phases": [],
        }, [
            {
                "category": "invalid_tls_link_index",
                "connection": _connection_identity(connection),
            }
        ]
    tls_programs = list(programs.get(tls_id, ()))
    if not tls_programs:
        anomalies.append(
            {
                "category": "missing_tls_program",
                "connection": _connection_identity(connection),
            }
        )
    phases = []
    width_failures = 0
    for program in tls_programs:
        for phase in program.get("phases", ()):
            state = str(phase.get("state", ""))
            if link_index >= len(state):
                width_failures += 1
                continue
            signal = state[link_index]
            if signal in {"G", "g"}:
                phases.append(
                    {
                        "program_id": str(program.get("program_id", "")),
                        "phase_index": int(phase["phase_index"]),
                        "signal": signal,
                    }
                )
    if width_failures:
        anomalies.append(
            {
                "category": "tls_phase_state_too_short",
                "connection": _connection_identity(connection),
                "affected_phase_count": int(width_failures),
            }
        )
    return {
        "controlled": True,
        "protected_phase_count": sum(
            row["signal"] == "G" for row in phases
        ),
        "yielding_phase_count": sum(
            row["signal"] == "g" for row in phases
        ),
        "active_phase_count": len(phases),
        "service_phases": phases,
    }, anomalies


def _junction_summary(
    receiving_edge: str,
    edges: Mapping[str, Mapping[str, str]],
    junctions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    junction_id = str(edges.get(receiving_edge, {}).get("from", ""))
    junction = dict(junctions.get(junction_id, {}))
    requests = list(junction.get("requests", ()))
    return {
        "junction_id": junction_id or None,
        "junction_type": junction.get("type"),
        "incoming_lane_count": len(junction.get("inc_lanes", ())),
        "internal_lane_count": len(junction.get("internal_lanes", ())),
        "request_count": len(requests),
        "maximum_response_foe_count": max(
            (_bit_count(row.get("response", "")) for row in requests),
            default=0,
        ),
        "maximum_foe_count": max(
            (_bit_count(row.get("foes", "")) for row in requests),
            default=0,
        ),
    }


def audit_shared_receiving_conflicts(
    network: Path,
    *,
    focus_receiving_lanes: Sequence[str] = (),
    focus_tls_ids: Sequence[str] = (),
) -> dict[str, Any]:
    network = Path(network)
    if not network.is_file():
        raise FileNotFoundError(network)
    edges, programs, junctions, connections, lane_ids = _read_network(network)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anomalies: list[dict[str, Any]] = []
    for connection in connections:
        to_edge = str(connection.get("to", ""))
        to_lane_index = str(connection.get("toLane", ""))
        if not to_edge or not to_lane_index:
            anomalies.append(
                {
                    "category": "missing_receiving_lane_identity",
                    "connection": _connection_identity(connection),
                }
            )
            continue
        receiving_lane = f"{to_edge}_{to_lane_index}"
        service, service_anomalies = _service_profile(connection, programs)
        anomalies.extend(service_anomalies)
        controlled = bool(service["controlled"])
        state = str(connection.get("state", ""))
        from_edge = str(connection.get("from", ""))
        uncontrolled_major = bool(
            not controlled
            and not from_edge.startswith(":")
            and (
                connection.get("uncontrolled") == "1"
                or state == "M"
            )
        )
        groups[receiving_lane].append(
            {
                "connection": _connection_identity(connection),
                "service": service,
                "uncontrolled_major": uncontrolled_major,
            }
        )

    risk_groups = []
    shared_group_count = 0
    yielding_shared_group_count = 0
    protected_shared_group_count = 0
    uncontrolled_major_shared_group_count = 0
    for receiving_lane, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        shared_group_count += 1
        controlled = [row for row in rows if row["service"]["controlled"]]
        protected = [
            row
            for row in controlled
            if int(row["service"]["protected_phase_count"]) > 0
        ]
        yielding = [
            row
            for row in controlled
            if int(row["service"]["yielding_phase_count"]) > 0
        ]
        uncontrolled_major = [
            row for row in rows if row["uncontrolled_major"]
        ]
        if yielding and uncontrolled_major:
            yielding_shared_group_count += 1
        if protected and uncontrolled_major:
            protected_shared_group_count += 1
            risk_groups.append(
                {
                    "category": (
                        "protected_controlled_with_uncontrolled_major_"
                        "shared_receiving_lane"
                    ),
                    "receiving_lane": receiving_lane,
                    "receiving_lane_declared": receiving_lane in lane_ids,
                    "junction": _junction_summary(
                        str(rows[0]["connection"].get("to", "")),
                        edges,
                        junctions,
                    ),
                    "protected_controlled_connections": protected,
                    "uncontrolled_major_connections": uncontrolled_major,
                    "all_connection_count": len(rows),
                }
            )
        if len(uncontrolled_major) >= 2:
            uncontrolled_major_shared_group_count += 1
            risk_groups.append(
                {
                    "category": (
                        "multiple_uncontrolled_major_movements_"
                        "shared_receiving_lane"
                    ),
                    "receiving_lane": receiving_lane,
                    "receiving_lane_declared": receiving_lane in lane_ids,
                    "junction": _junction_summary(
                        str(rows[0]["connection"].get("to", "")),
                        edges,
                        junctions,
                    ),
                    "uncontrolled_major_connections": uncontrolled_major,
                    "all_connection_count": len(rows),
                }
            )

    focus_lane_set = {str(value) for value in focus_receiving_lanes}
    focus_tls_set = {str(value) for value in focus_tls_ids}
    focus_groups = {}
    for receiving_lane, rows in sorted(groups.items()):
        has_focus_tls = any(
            str(row["connection"].get("tl", "")) in focus_tls_set
            for row in rows
        )
        if receiving_lane in focus_lane_set or has_focus_tls:
            focus_groups[receiving_lane] = {
                "junction": _junction_summary(
                    str(rows[0]["connection"].get("to", "")),
                    edges,
                    junctions,
                ),
                "connections": rows,
            }

    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "counts": {
            "edge_count": len(edges),
            "lane_count": len(lane_ids),
            "junction_count": len(junctions),
            "tls_count": len(programs),
            "connection_count": len(connections),
            "receiving_lane_group_count": len(groups),
            "shared_receiving_lane_group_count": shared_group_count,
            "yielding_controlled_with_uncontrolled_major_group_count": (
                yielding_shared_group_count
            ),
            "protected_controlled_with_uncontrolled_major_group_count": (
                protected_shared_group_count
            ),
            "multiple_uncontrolled_major_group_count": (
                uncontrolled_major_shared_group_count
            ),
            "risk_record_count": len(risk_groups),
            "static_anomaly_count": len(anomalies),
        },
        "risk_groups": risk_groups,
        "static_anomalies": anomalies,
        "focus_groups": focus_groups,
        "audit_boundary": (
            "These are static topology candidates, not proven collisions. "
            "Any repair requires focused right-of-way and microscopic replay "
            "validation; this audit never modifies the network."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--focus-receiving-lane", action="append", default=[])
    parser.add_argument("--focus-tls", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit_shared_receiving_conflicts(
        args.network,
        focus_receiving_lanes=args.focus_receiving_lane,
        focus_tls_ids=args.focus_tls,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "counts": result["counts"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
