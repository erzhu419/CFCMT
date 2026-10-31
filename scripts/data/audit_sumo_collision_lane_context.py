#!/usr/bin/env python3
"""Audit the static lane context for one observed SUMO collision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO
import xml.etree.ElementTree as ET


PROTOCOL = "sumo-collision-lane-context-audit-v2"
DEFAULT_VEHICLE_TYPE = "DEFAULT_VEHTYPE"
SUMO_DEFAULT_VEHICLE_PARAMETERS: Mapping[str, str] = {
    "accel": "2.6",
    "carFollowModel": "Krauss",
    "decel": "4.5",
    "emergencyDecel": "9.0",
    "laneChangeModel": "LC2013",
    "length": "5.0",
    "minGap": "2.5",
    "sigma": "0.5",
    "tau": "1.0",
    "vClass": "passenger",
}
RELEVANT_CONFIG_OPTIONS = frozenset(
    {
        "begin",
        "collision.action",
        "collision.check-junctions",
        "collision.check-junctions.mingap",
        "collision.mingap-factor",
        "emergency-insert",
        "end",
        "lateral-resolution",
        "lanechange.duration",
        "seed",
        "step-length",
    }
)


def _attrs(element: ET.Element) -> dict[str, str]:
    return {str(key): str(value) for key, value in sorted(element.attrib.items())}


def _lane_rows(edge: ET.Element) -> list[dict[str, Any]]:
    return [
        {
            "attributes": _attrs(lane),
            "id": str(lane.get("id", "")),
            "index": int(str(lane.get("index", offset))),
        }
        for offset, lane in enumerate(edge.findall("lane"))
    ]


def _edge_row(edge: ET.Element) -> dict[str, Any]:
    return {
        "attributes": _attrs(edge),
        "id": str(edge.get("id", "")),
        "lanes": _lane_rows(edge),
    }


def _connection_row(connection: ET.Element) -> dict[str, str]:
    return _attrs(connection)


def _split_files(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _config_summary(config: Path | None) -> dict[str, Any]:
    if config is None:
        return {"provided": False}
    config = Path(config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    root = ET.parse(config).getroot()
    values: dict[str, str] = {}
    for element in root.iter():
        tag = str(element.tag)
        value = element.get("value")
        if value is not None:
            values[tag] = str(value)
    input_files: dict[str, list[str]] = {}
    for key in ("additional-files", "route-files"):
        if key not in values:
            continue
        input_files[key] = [
            str((config.parent / relative).resolve())
            for relative in _split_files(values[key])
        ]
    return {
        "provided": True,
        "path": str(config),
        "input_files": input_files,
        "relevant_options": {
            key: values[key] for key in sorted(RELEVANT_CONFIG_OPTIONS & values.keys())
        },
    }


def _open_xml(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def _prefix_vehicle_types(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read declarations before the first demand object without loading route data."""

    rows: list[dict[str, Any]] = []
    stopped_at = "end_of_file"
    with _open_xml(path) as source:
        for _, element in ET.iterparse(source, events=("start",)):
            tag = str(element.tag)
            if tag == "vType":
                rows.append({"attributes": _attrs(element), "source": str(path)})
            elif tag in {"flow", "person", "trip", "vehicle"}:
                stopped_at = tag
                break
    return rows, stopped_at


def _vehicle_type_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    declarations: list[dict[str, Any]] = []
    scans: list[dict[str, Any]] = []
    input_files = dict(config.get("input_files", {}))
    paths = [
        Path(value)
        for key in ("additional-files", "route-files")
        for value in input_files.get(key, [])
    ]
    for path in paths:
        if not path.is_file():
            scans.append({"path": str(path), "status": "missing"})
            continue
        rows, stopped_at = _prefix_vehicle_types(path)
        declarations.extend(rows)
        scans.append(
            {
                "path": str(path),
                "status": "scanned",
                "stopped_at": stopped_at,
                "vehicle_type_declaration_count": len(rows),
            }
        )
    explicit = next(
        (
            row
            for row in declarations
            if row["attributes"].get("id") == DEFAULT_VEHICLE_TYPE
        ),
        None,
    )
    effective = dict(SUMO_DEFAULT_VEHICLE_PARAMETERS)
    if explicit is not None:
        effective.update(dict(explicit["attributes"]))
    return {
        "requested_type_id": DEFAULT_VEHICLE_TYPE,
        "definition_source": (
            "input_declaration" if explicit is not None else "sumo_builtin_default"
        ),
        "effective_parameters": effective,
        "explicit_definition": explicit,
        "input_prefix_scans": scans,
        "all_prefix_declarations": declarations,
        "scan_boundary": (
            "Route-like files are streamed only through the first demand object; "
            "this resolves declarations available before the first vehicle is loaded."
        ),
    }


def _junction_row(junction: ET.Element | None) -> dict[str, Any] | None:
    if junction is None:
        return None
    return {
        "attributes": _attrs(junction),
        "requests": [_attrs(request) for request in junction.findall("request")],
    }


def _junction_request_context(
    root: ET.Element,
    *,
    junction_id: str,
    connections: Sequence[ET.Element],
    edges_from: Mapping[str, str],
) -> dict[str, Any]:
    from scripts.data.audit_sumo_shared_receiving_right_of_way import (
        _bit,
        _junction_indices,
    )

    indices, diagnostics = _junction_indices(
        root,
        edges_from=edges_from,
        connections=connections,
    )
    diagnostic = dict(diagnostics.get(junction_id, {}))
    requests = list(diagnostic.get("requests", ()))
    rows = [
        connection
        for connection in connections
        if edges_from.get(str(connection.get("to", ""))) == junction_id
    ]
    mapped: list[dict[str, Any]] = []
    by_index = {
        int(indices[connection]): connection
        for connection in rows
        if connection in indices
    }
    for connection in rows:
        index = indices.get(connection)
        request = (
            requests[int(index)]
            if index is not None and int(index) < len(requests)
            else None
        )
        response_to = []
        foes = []
        if request is not None:
            response_to = [
                other_index
                for other_index in sorted(by_index)
                if _bit(str(request.get("response", "")), other_index) == "1"
            ]
            foes = [
                other_index
                for other_index in sorted(by_index)
                if _bit(str(request.get("foes", "")), other_index) == "1"
            ]
        mapped.append(
            {
                "connection": _connection_row(connection),
                "junction_request_index": int(index) if index is not None else None,
                "request": _attrs(request) if request is not None else None,
                "response_to_indices": response_to,
                "foe_indices": foes,
                "response_to_connections": [
                    _connection_row(by_index[other_index])
                    for other_index in response_to
                ],
            }
        )
    public_diagnostic = {
        key: value
        for key, value in diagnostic.items()
        if key != "requests"
    }
    return {
        "junction_id": junction_id,
        "mapping": public_diagnostic,
        "connections": mapped,
    }


def _program_rows(
    root: ET.Element, tls_ids: Iterable[str]
) -> list[dict[str, Any]]:
    selected = set(tls_ids)
    rows: list[dict[str, Any]] = []
    for logic in root.iter("tlLogic"):
        if str(logic.get("id", "")) not in selected:
            continue
        rows.append(
            {
                "attributes": _attrs(logic),
                "phases": [_attrs(phase) for phase in logic.findall("phase")],
            }
        )
    return rows


def _geometry_summary(
    *,
    lane_length: float,
    collider_front_position: float | None,
    victim_front_position: float | None,
    vehicle_type: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if collider_front_position is None or victim_front_position is None:
        return {"positions_provided": False}
    parameters = dict(vehicle_type["effective_parameters"])
    leader_length = float(parameters["length"])
    minimum_gap = float(parameters["minGap"])
    options = dict(config.get("relevant_options", {}))
    factor_value = options.get("collision.mingap-factor")
    collision_factor = 1.0 if factor_value in {None, "-1"} else float(factor_value)
    front_distance = float(victim_front_position - collider_front_position)
    bumper_gap = float(front_distance - leader_length)
    threshold = float(minimum_gap * collision_factor)
    return {
        "positions_provided": True,
        "collider_front_position_m": float(collider_front_position),
        "victim_front_position_m": float(victim_front_position),
        "front_to_front_distance_m": front_distance,
        "assumed_leader_length_m": leader_length,
        "bumper_gap_m": bumper_gap,
        "collision_detection_gap_threshold_m": threshold,
        "below_detection_gap_threshold": bumper_gap < threshold,
        "physical_body_overlap": bumper_gap < 0.0,
        "collider_distance_to_lane_end_m": float(
            lane_length - collider_front_position
        ),
        "victim_distance_to_lane_end_m": float(lane_length - victim_front_position),
        "collision_mingap_factor_source": (
            "configuration"
            if factor_value not in {None, "-1"}
            else "car_follow_model_default"
        ),
    }


def audit_collision_lane_context(
    network: Path,
    *,
    lane_id: str,
    previous_edge_id: str,
    next_edge_id: str,
    config: Path | None = None,
    collider_front_position: float | None = None,
    victim_front_position: float | None = None,
) -> dict[str, Any]:
    network = Path(network).resolve()
    if not network.is_file():
        raise FileNotFoundError(network)
    root = ET.parse(network).getroot()
    edge_elements = {
        str(edge.get("id")): edge
        for edge in root.iter("edge")
        if edge.get("id") is not None
    }
    lane_to_edge: dict[str, tuple[ET.Element, ET.Element]] = {}
    for edge in edge_elements.values():
        for lane in edge.findall("lane"):
            if lane.get("id") is not None:
                lane_to_edge[str(lane.get("id"))] = (edge, lane)
    if lane_id not in lane_to_edge:
        raise ValueError(f"unknown collision lane: {lane_id}")
    current_edge, collision_lane = lane_to_edge[lane_id]
    current_edge_id = str(current_edge.get("id", ""))
    for required in (previous_edge_id, next_edge_id):
        if required not in edge_elements:
            raise ValueError(f"unknown route edge: {required}")

    lane_index = str(collision_lane.get("index", ""))
    connections = list(root.iter("connection"))
    edges_from = {
        edge_id: str(edge.get("from", ""))
        for edge_id, edge in edge_elements.items()
    }
    incoming = [
        row
        for row in connections
        if str(row.get("to", "")) == current_edge_id
        and str(row.get("toLane", "")) == lane_index
    ]
    outgoing = [
        row
        for row in connections
        if str(row.get("from", "")) == current_edge_id
        and str(row.get("fromLane", "")) == lane_index
    ]
    previous_to_current = [
        row
        for row in incoming
        if str(row.get("from", "")) == previous_edge_id
    ]
    current_to_next = [
        row
        for row in outgoing
        if str(row.get("to", "")) == next_edge_id
    ]
    all_current_to_next = [
        row
        for row in connections
        if str(row.get("from", "")) == current_edge_id
        and str(row.get("to", "")) == next_edge_id
    ]
    all_previous_to_current = [
        row
        for row in connections
        if str(row.get("from", "")) == previous_edge_id
        and str(row.get("to", "")) == current_edge_id
    ]
    tls_ids = {
        str(row.get("tl"))
        for row in (*incoming, *outgoing)
        if row.get("tl") not in {None, ""}
    }
    junctions = {
        str(junction.get("id")): junction
        for junction in root.iter("junction")
        if junction.get("id") is not None
    }
    config_row = _config_summary(config)
    vehicle_type = _vehicle_type_summary(config_row)
    lane_length = float(str(collision_lane.get("length", "nan")))
    geometry = _geometry_summary(
        lane_length=lane_length,
        collider_front_position=collider_front_position,
        victim_front_position=victim_front_position,
        vehicle_type=vehicle_type,
        config=config_row,
    )
    near_downstream = bool(
        geometry.get("positions_provided")
        and float(geometry["victim_distance_to_lane_end_m"])
        <= max(10.0, 0.2 * lane_length)
    )
    next_lane_indices = {
        str(row.get("toLane", "")) for row in all_current_to_next
    }
    route_chain_unique = len(previous_to_current) == len(current_to_next) == 1
    downstream_junction_id = str(current_edge.get("to", ""))
    downstream_request_context = _junction_request_context(
        root,
        junction_id=downstream_junction_id,
        connections=connections,
        edges_from=edges_from,
    )
    route_connection_request = next(
        (
            row
            for row in downstream_request_context["connections"]
            if row["connection"] == _connection_row(current_to_next[0])
        ),
        None,
    ) if len(current_to_next) == 1 else None

    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "collision_lane_id": lane_id,
        "route_chain": {
            "previous_edge_id": previous_edge_id,
            "current_edge_id": current_edge_id,
            "next_edge_id": next_edge_id,
            "previous_to_collision_lane": [
                _connection_row(row) for row in previous_to_current
            ],
            "collision_lane_to_next_edge": [
                _connection_row(row) for row in current_to_next
            ],
            "unique_on_both_sides": route_chain_unique,
        },
        "collision_lane": {
            "attributes": _attrs(collision_lane),
            "incoming_connections": [_connection_row(row) for row in incoming],
            "outgoing_connections": [_connection_row(row) for row in outgoing],
        },
        "edges": {
            "previous": _edge_row(edge_elements[previous_edge_id]),
            "current": _edge_row(current_edge),
            "next": _edge_row(edge_elements[next_edge_id]),
        },
        "junctions": {
            "upstream": _junction_row(junctions.get(str(current_edge.get("from", "")))),
            "downstream": _junction_row(junctions.get(str(current_edge.get("to", "")))),
        },
        "downstream_request_context": downstream_request_context,
        "traffic_light_programs": _program_rows(root, tls_ids),
        "lane_mapping": {
            "all_previous_to_current": [
                _connection_row(row) for row in all_previous_to_current
            ],
            "all_current_to_next": [
                _connection_row(row) for row in all_current_to_next
            ],
            "current_lane_count": len(current_edge.findall("lane")),
            "next_edge_lane_count": len(edge_elements[next_edge_id].findall("lane")),
            "distinct_next_lane_count": len(next_lane_indices),
            "current_to_next_lane_merge": (
                len({str(row.get("fromLane", "")) for row in all_current_to_next})
                > len(next_lane_indices)
            ),
        },
        "configuration": config_row,
        "vehicle_type": vehicle_type,
        "longitudinal_geometry": geometry,
        "classification": {
            "observed_route_chain_resolves_uniquely": route_chain_unique,
            "victim_near_downstream_boundary": near_downstream,
            "static_current_to_next_lane_merge": (
                len({str(row.get("fromLane", "")) for row in all_current_to_next})
                > len(next_lane_indices)
            ),
            "min_gap_violation_without_body_overlap": bool(
                geometry.get("below_detection_gap_threshold")
                and not geometry.get("physical_body_overlap")
            ),
            "downstream_route_connection_request_index": (
                route_connection_request["junction_request_index"]
                if route_connection_request is not None
                else None
            ),
            "downstream_route_yields_to_indices": (
                route_connection_request["response_to_indices"]
                if route_connection_request is not None
                else []
            ),
            "downstream_route_yields_to_connections": (
                route_connection_request["response_to_connections"]
                if route_connection_request is not None
                else []
            ),
            "scope": (
                "This classifies static topology and longitudinal geometry only; "
                "it does not infer the dynamic cause of the leader's deceleration."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--previous-edge-id", required=True)
    parser.add_argument("--next-edge-id", required=True)
    parser.add_argument("--collider-front-position", type=float)
    parser.add_argument("--victim-front-position", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit_collision_lane_context(
        args.network,
        lane_id=args.lane_id,
        previous_edge_id=args.previous_edge_id,
        next_edge_id=args.next_edge_id,
        config=args.config,
        collider_front_position=args.collider_front_position,
        victim_front_position=args.victim_front_position,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "classification": result["classification"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
