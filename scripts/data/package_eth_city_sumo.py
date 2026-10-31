#!/usr/bin/env python3
"""Package one complete ETH five-city unit for strict microscopic SUMO."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.sumo_static_inputs import (  # noqa: E402
    iterparse_xml,
    parse_sumo_time_seconds,
    parse_xml,
)
from scripts.data.acquire_eth_five_city_sumo import (  # noqa: E402
    ARCHIVE_NAME,
    DOI,
    EXPECTED_MD5,
    EXPECTED_SIZE_BYTES,
    PROTOCOL as ACQUISITION_PROTOCOL,
)


PROTOCOL = "cfcmt-eth-five-city-microscopic-complete-published-package-v5"
INPUT_AUDIT_PROTOCOL = "eth-five-city-complete-published-static-input-audit-v2"
MICRO_CONFIG_PROTOCOL = "eth-five-city-derived-strict-microscopic-sumo122-v5"
ZIPPER_REPAIR_PROTOCOL = "sumo122-unsupported-multi-foe-zipper-repair-v3"
ROUTE_CANONICALIZATION_PROTOCOL = (
    "eth-five-city-trip-destination-taz-schema-canonicalization-v1"
)
DRAIN_AFTER_LAST_DEPART_SEC = 6 * 3600
MINIMUM_PUBLISHED_SPAN_SEC = 11 * 3600
MAXIMUM_PUBLISHED_SPAN_SEC = 13 * 3600
DEPARTURE_BOUND_TOLERANCE_SEC = 0.011

CITY_NAMES = {
    "BOS": "boston",
    "LIS": "lisbon",
    "LAX": "los_angeles",
    "RIO": "rio_de_janeiro",
    "SFO": "san_francisco",
}

# These bounds were recorded by the v102 streaming screen before v118 was
# specified. They are input-identity checks, not values learned from a run.
EXPECTED_DEPARTURE_BOUNDS = {
    "BOS": (7201.78, 50395.74),
    "LIS": (7214.42, 50393.47),
    "LAX": (7208.84, 50393.08),
    "RIO": (7225.11, 50391.75),
    "SFO": (7213.23, 50399.69),
}


def expected_city_entries(city_code: str) -> tuple[str, ...]:
    code = city_code.upper()
    if code not in CITY_NAMES:
        raise ValueError(f"unsupported ETH city code: {city_code}")
    return (
        f"{code}/",
        f"{code}/additional.add.xml",
        f"{code}/{code}.net.xml",
        f"{code}/meso.sumo.cfg",
        f"{code}/taz.xml",
        f"{code}/trips24h_smoothed.rou.xml",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_manifest(
    acquisition_root: Path, *, city_code: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    path = acquisition_root / "acquisition_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": ACQUISITION_PROTOCOL,
        "doi": DOI,
        "archive_name": ARCHIVE_NAME,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "md5": EXPECTED_MD5,
        "archive_entry_count": 30,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"ETH acquisition manifest is not frozen: {observed}")
    archive = acquisition_root / ARCHIVE_NAME
    if archive.stat().st_size != EXPECTED_SIZE_BYTES:
        raise ValueError("ETH archive size changed after acquisition")
    entries = tuple(
        (acquisition_root / "archive_entries.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    expected_entries = expected_city_entries(city_code)
    city_entries = tuple(
        row for row in entries if row.startswith(f"{city_code}/")
    )
    if city_entries != expected_entries:
        raise ValueError(
            f"ETH {city_code} archive entries changed: {city_entries}"
        )
    return payload, expected_entries


def _network_inventory(path: Path) -> tuple[dict[str, Any], set[str]]:
    edge_ids: set[str] = set()
    junction_ids: set[str] = set()
    tls_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        tag = element.tag
        counts[tag] += 1
        if tag == "edge":
            identity = str(element.attrib.get("id", ""))
            if not identity:
                raise ValueError("ETH network edge has no id")
            if not identity.startswith(":"):
                edge_ids.add(identity)
        elif tag == "junction":
            identity = str(element.attrib.get("id", ""))
            if identity:
                junction_ids.add(identity)
        elif tag == "tlLogic":
            identity = str(element.attrib.get("id", ""))
            if identity:
                tls_ids.add(identity)
        element.clear()
    return (
        {
            "edge_count": counts["edge"],
            "noninternal_edge_count": len(edge_ids),
            "lane_count": counts["lane"],
            "junction_count": len(junction_ids),
            "connection_count": counts["connection"],
            "traffic_light_count": len(tls_ids),
            "traffic_light_program_count": counts["tlLogic"],
            "traffic_light_phase_count": counts["phase"],
        },
        edge_ids,
    )


def _unsupported_zipper_junctions(path: Path) -> dict[str, Any]:
    """Find zipper requests that SUMO 1.22 cannot evaluate microscopically."""
    zipper_count = 0
    response_foe_histogram: Counter[int] = Counter()
    candidates: list[dict[str, Any]] = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "junction":
            if element.attrib.get("type") == "zipper":
                zipper_count += 1
                response_foe_counts = [
                    str(request.attrib.get("response", "")).count("1")
                    for request in element.findall("request")
                ]
                maximum = max(response_foe_counts, default=0)
                response_foe_histogram[maximum] += 1
                if maximum > 1:
                    candidates.append(
                        {
                            "id": str(element.attrib["id"]),
                            "incoming_lane_count": len(
                                str(element.attrib.get("incLanes", "")).split()
                            ),
                            "request_count": len(response_foe_counts),
                            "max_response_foe_count": maximum,
                            "violating_request_indices": [
                                index
                                for index, count in enumerate(response_foe_counts)
                                if count > 1
                            ],
                        }
                    )
            element.clear()
        elif element.tag in {"edge", "tlLogic", "roundabout"}:
            element.clear()
    candidates.sort(key=lambda row: row["id"])
    return {
        "sumo122_failure_condition": "zipper_link_response_foe_count_gt_1",
        "zipper_junction_count": zipper_count,
        "max_response_foe_count_histogram": {
            str(key): value
            for key, value in sorted(response_foe_histogram.items())
        },
        "unsupported_candidate_count": len(candidates),
        "unsupported_candidates": candidates,
    }


def _write_zipper_compatibility_repair(
    source: Path,
    destination: Path,
    *,
    junction_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[tuple[str, ...], tuple[str, str]]]:
    expected = set(str(value) for value in junction_ids)
    tree = ET.parse(source)
    root = tree.getroot()
    changed = set()
    incoming_lane_to_junction: dict[str, str] = {}
    requests_by_junction: dict[str, list[dict[str, str]]] = {}
    for junction in root.iter("junction"):
        identity = str(junction.attrib.get("id", ""))
        if identity not in expected:
            continue
        if junction.attrib.get("type") != "zipper":
            raise ValueError(
                f"declared zipper repair has unexpected type: {identity}"
            )
        junction.set("type", "priority")
        changed.add(identity)
        requests = sorted(
            (dict(request.attrib) for request in junction.findall("request")),
            key=lambda row: int(row.get("index", "-1")),
        )
        if [int(row.get("index", "-1")) for row in requests] != list(
            range(len(requests))
        ):
            raise ValueError(
                f"declared zipper repair has invalid request indices: {identity}"
            )
        requests_by_junction[identity] = requests
        for lane_id in str(junction.attrib.get("incLanes", "")).split():
            previous = incoming_lane_to_junction.setdefault(lane_id, identity)
            if previous != identity:
                raise ValueError(
                    f"incoming lane belongs to multiple repaired junctions: {lane_id}"
                )
    if changed != expected:
        missing = sorted(expected - changed)
        raise ValueError(f"declared zipper junctions are missing: {missing}")

    request_offsets: Counter[str] = Counter()
    connection_changes: dict[tuple[str, ...], tuple[str, str]] = {}
    state_change_histogram: Counter[str] = Counter()
    for connection in root.iter("connection"):
        incoming_lane = (
            f"{connection.attrib.get('from', '')}_"
            f"{connection.attrib.get('fromLane', '')}"
        )
        junction_id = incoming_lane_to_junction.get(incoming_lane)
        if junction_id is None:
            continue
        request_index = request_offsets[junction_id]
        requests = requests_by_junction[junction_id]
        if request_index >= len(requests):
            raise ValueError(
                f"more connections than requests at repaired junction: {junction_id}"
            )
        request_offsets[junction_id] += 1
        before_state = str(connection.attrib.get("state", ""))
        if before_state != "Z":
            raise ValueError(
                "declared zipper connection has unexpected state: "
                f"{junction_id} index={request_index} state={before_state!r}"
            )
        after_state = (
            "m" if "1" in str(requests[request_index].get("response", "")) else "M"
        )
        identity = tuple(
            str(connection.attrib.get(key, ""))
            for key in (
                "from",
                "to",
                "fromLane",
                "toLane",
                "via",
                "tl",
                "linkIndex",
            )
        )
        if identity in connection_changes:
            raise ValueError(
                f"duplicate repaired connection identity: {identity}"
            )
        connection.set("state", after_state)
        connection_changes[identity] = (before_state, after_state)
        state_change_histogram[f"{before_state}->{after_state}"] += 1

    count_mismatches = {
        junction_id: {
            "requests": len(requests_by_junction[junction_id]),
            "connections": request_offsets[junction_id],
        }
        for junction_id in sorted(expected)
        if request_offsets[junction_id] != len(requests_by_junction[junction_id])
    }
    if count_mismatches:
        raise ValueError(
            f"repaired junction request/connection count mismatch: {count_mismatches}"
        )
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return (
        {
            "method": (
                "structured_xml_declared_junction_type_and_link_state_substitution"
            ),
            "changed_junction_count": len(changed),
            "changed_connection_state_count": len(connection_changes),
            "connection_state_change_histogram": dict(
                sorted(state_change_histogram.items())
            ),
        },
        connection_changes,
    )


def _network_structure_exact_except_declared_zipper_changes(
    source: Path,
    repaired: Path,
    *,
    repaired_junction_ids: Sequence[str],
    repaired_connection_states: dict[tuple[str, ...], tuple[str, str]],
) -> bool:
    repaired_ids = set(str(value) for value in repaired_junction_ids)
    remaining_connection_states = dict(repaired_connection_states)
    before = ET.iterparse(source, events=("start", "end"))
    after = ET.iterparse(repaired, events=("start", "end"))
    sentinel = object()
    while True:
        left = next(before, sentinel)
        right = next(after, sentinel)
        if left is sentinel or right is sentinel:
            return (
                left is sentinel
                and right is sentinel
                and not remaining_connection_states
            )
        left_event, left_element = left
        right_event, right_element = right
        if left_event != right_event or left_element.tag != right_element.tag:
            return False
        if left_event == "start":
            left_attributes = dict(left_element.attrib)
            right_attributes = dict(right_element.attrib)
            identity = str(left_attributes.get("id", ""))
            if left_element.tag == "junction" and identity in repaired_ids:
                if (
                    left_attributes.get("type") != "zipper"
                    or right_attributes.get("type") != "priority"
                ):
                    return False
                left_attributes["type"] = "priority"
            elif left_element.tag == "connection":
                connection_identity = tuple(
                    str(left_attributes.get(key, ""))
                    for key in (
                        "from",
                        "to",
                        "fromLane",
                        "toLane",
                        "via",
                        "tl",
                        "linkIndex",
                    )
                )
                expected_states = remaining_connection_states.pop(
                    connection_identity, None
                )
                if expected_states is not None:
                    before_state, after_state = expected_states
                    if (
                        left_attributes.get("state") != before_state
                        or right_attributes.get("state") != after_state
                    ):
                        return False
                    left_attributes["state"] = after_state
            if left_attributes != right_attributes:
                return False
        else:
            left_element.clear()
            right_element.clear()


def _network_semantics(path: Path) -> dict[str, Any]:
    edges: set[tuple[str, ...]] = set()
    lanes: set[tuple[str, ...]] = set()
    movements: set[tuple[str, ...]] = set()
    junction_types: dict[str, str] = {}
    tls_programs: set[tuple[Any, ...]] = set()
    current_external_edge = False
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if element.tag == "edge":
                edge_id = str(element.attrib.get("id", ""))
                current_external_edge = bool(edge_id and not edge_id.startswith(":"))
                if current_external_edge:
                    edges.add(
                        (
                            edge_id,
                            str(element.attrib.get("from", "")),
                            str(element.attrib.get("to", "")),
                            str(element.attrib.get("priority", "")),
                            str(element.attrib.get("type", "")),
                        )
                    )
            elif element.tag == "lane" and current_external_edge:
                lanes.add(
                    tuple(
                        str(element.attrib.get(key, ""))
                        for key in (
                            "id",
                            "index",
                            "speed",
                            "length",
                            "allow",
                            "disallow",
                            "width",
                        )
                    )
                )
            elif element.tag == "connection":
                source = str(element.attrib.get("from", ""))
                destination = str(element.attrib.get("to", ""))
                if source and destination and not source.startswith(":"):
                    movements.add(
                        (
                            source,
                            destination,
                            str(element.attrib.get("fromLane", "")),
                            str(element.attrib.get("toLane", "")),
                        )
                    )
            elif element.tag == "junction":
                junction_id = str(element.attrib.get("id", ""))
                if junction_id and not junction_id.startswith(":"):
                    junction_types[junction_id] = str(
                        element.attrib.get("type", "")
                    )
        elif element.tag == "tlLogic":
            phases = tuple(
                tuple(sorted(phase.attrib.items()))
                for phase in element.findall("phase")
            )
            tls_programs.add(
                (
                    str(element.attrib.get("id", "")),
                    str(element.attrib.get("programID", "")),
                    str(element.attrib.get("type", "")),
                    str(element.attrib.get("offset", "")),
                    phases,
                )
            )
            element.clear()
        elif element.tag == "edge":
            current_external_edge = False
            element.clear()
        elif element.tag in {"junction", "connection", "roundabout"}:
            element.clear()
    return {
        "edges": edges,
        "lanes": lanes,
        "movements": movements,
        "junction_types": junction_types,
        "tls_programs": tls_programs,
    }


def _network_compatibility_audit(
    source: Path,
    repaired: Path,
    *,
    repaired_junction_ids: Sequence[str],
    repaired_connection_states: dict[tuple[str, ...], tuple[str, str]],
) -> dict[str, Any]:
    before = _network_semantics(source)
    after = _network_semantics(repaired)
    expected_changes = {
        junction_id: ("zipper", "priority")
        for junction_id in sorted(set(repaired_junction_ids))
    }
    observed_changes = {
        junction_id: (
            before["junction_types"].get(junction_id, ""),
            after["junction_types"].get(junction_id, ""),
        )
        for junction_id in sorted(
            set(before["junction_types"]) | set(after["junction_types"])
        )
        if before["junction_types"].get(junction_id)
        != after["junction_types"].get(junction_id)
    }
    checks = {
        "network_xml_exact_except_declared_zipper_compatibility_changes": (
            _network_structure_exact_except_declared_zipper_changes(
                source,
                repaired,
                repaired_junction_ids=repaired_junction_ids,
                repaired_connection_states=repaired_connection_states,
            )
        ),
        "external_edges_exact": before["edges"] == after["edges"],
        "external_lanes_exact": before["lanes"] == after["lanes"],
        "external_movements_exact": (
            before["movements"] == after["movements"]
        ),
        "traffic_light_programs_exact": (
            before["tls_programs"] == after["tls_programs"]
        ),
        "junction_type_changes_exact": observed_changes == expected_changes,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "external_edges": len(after["edges"]),
            "external_lanes": len(after["lanes"]),
            "external_movements": len(after["movements"]),
            "traffic_light_programs": len(after["tls_programs"]),
            "junction_type_changes": len(observed_changes),
            "connection_state_changes": len(repaired_connection_states),
        },
        "expected_junction_type_changes": len(expected_changes),
        "expected_connection_state_changes": len(repaired_connection_states),
        "unexpected_junction_type_change_count": sum(
            observed_changes.get(junction_id) != expected_changes.get(junction_id)
            for junction_id in set(observed_changes) | set(expected_changes)
        ),
    }


def _repair_unsupported_zipper_junctions(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    before = _unsupported_zipper_junctions(source)
    junction_ids = [
        row["id"] for row in before["unsupported_candidates"]
    ]
    if not junction_ids:
        shutil.copy2(source, destination)
        return {
            "protocol": ZIPPER_REPAIR_PROTOCOL,
            "applied": False,
            "before": before,
            "after": before,
            "compatibility_audit": {
                "checks": {"source_network_reused_exactly": True},
                "passed": True,
            },
            "passed": True,
        }
    repair, repaired_connection_states = _write_zipper_compatibility_repair(
        source,
        destination,
        junction_ids=junction_ids,
    )
    after = _unsupported_zipper_junctions(destination)
    compatibility = _network_compatibility_audit(
        source,
        destination,
        repaired_junction_ids=junction_ids,
        repaired_connection_states=repaired_connection_states,
    )
    checks = {
        "all_declared_candidates_repaired": (
            after["unsupported_candidate_count"] == 0
        ),
        "network_semantics_preserved_except_declared_zipper_changes": compatibility[
            "passed"
        ],
    }
    return {
        "protocol": ZIPPER_REPAIR_PROTOCOL,
        "applied": True,
        "source_network": str(source.name),
        "derived_network": str(destination.name),
        "repair": repair,
        "before": before,
        "after": after,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _taz_inventory(path: Path) -> tuple[dict[str, Any], set[str]]:
    taz_ids: set[str] = set()
    duplicate_ids = 0
    taz_source_count = 0
    taz_sink_count = 0
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "taz":
            identity = str(element.attrib.get("id", ""))
            if not identity:
                raise ValueError("ETH TAZ has no id")
            duplicate_ids += int(identity in taz_ids)
            taz_ids.add(identity)
        elif element.tag == "tazSource":
            taz_source_count += 1
        elif element.tag == "tazSink":
            taz_sink_count += 1
        element.clear()
    return (
        {
            "taz_count": len(taz_ids),
            "duplicate_taz_id_count": duplicate_ids,
            "taz_source_count": taz_source_count,
            "taz_sink_count": taz_sink_count,
        },
        taz_ids,
    )


def _route_inventory(
    path: Path, *, edge_ids: set[str], taz_ids: set[str]
) -> dict[str, Any]:
    trip_ids: set[str] = set()
    duplicate_ids = 0
    missing_edge_anchors = 0
    missing_taz_anchors = 0
    missing_attributes = 0
    nonmonotonic_departures = 0
    minimum: float | None = None
    maximum: float | None = None
    previous: float | None = None
    tags: Counter[str] = Counter()
    custom_destination_taz_attributes = 0
    canonical_destination_taz_attributes = 0
    conflicting_destination_taz_attributes = 0
    for _, element in iterparse_xml(path, events=("end",)):
        tags[element.tag] += 1
        if element.tag != "trip":
            element.clear()
            continue
        identity = str(element.attrib.get("id", ""))
        source = str(element.attrib.get("from", ""))
        destination = str(element.attrib.get("to", ""))
        depart_text = element.attrib.get("depart")
        if not identity or not source or not destination or depart_text is None:
            missing_attributes += 1
            element.clear()
            continue
        duplicate_ids += int(identity in trip_ids)
        trip_ids.add(identity)
        missing_edge_anchors += int(
            source not in edge_ids or destination not in edge_ids
        )
        from_taz = element.attrib.get("fromTaz")
        custom_to_taz = element.attrib.get("trip_toTaz")
        canonical_to_taz = element.attrib.get("toTaz")
        custom_destination_taz_attributes += int(custom_to_taz is not None)
        canonical_destination_taz_attributes += int(canonical_to_taz is not None)
        conflicting_destination_taz_attributes += int(
            custom_to_taz is not None
            and canonical_to_taz is not None
            and custom_to_taz != canonical_to_taz
        )
        to_taz = custom_to_taz or canonical_to_taz
        missing_taz_anchors += int(
            from_taz is None
            or to_taz is None
            or from_taz not in taz_ids
            or to_taz not in taz_ids
        )
        depart = parse_sumo_time_seconds(depart_text)
        minimum = depart if minimum is None else min(minimum, depart)
        maximum = depart if maximum is None else max(maximum, depart)
        if previous is not None and depart < previous:
            nonmonotonic_departures += 1
        previous = depart
        element.clear()
    span = None if minimum is None or maximum is None else maximum - minimum
    return {
        "trip_count": tags["trip"],
        "unique_trip_id_count": len(trip_ids),
        "duplicate_trip_id_count": duplicate_ids,
        "missing_required_attribute_count": missing_attributes,
        "missing_edge_anchor_count": missing_edge_anchors,
        "missing_taz_anchor_count": missing_taz_anchors,
        "custom_trip_to_taz_attribute_count": (
            custom_destination_taz_attributes
        ),
        "canonical_to_taz_attribute_count": (
            canonical_destination_taz_attributes
        ),
        "conflicting_destination_taz_attribute_count": (
            conflicting_destination_taz_attributes
        ),
        "minimum_depart_sec": minimum,
        "maximum_depart_sec": maximum,
        "depart_span_sec": span,
        "nonmonotonic_departure_count": nonmonotonic_departures,
        "other_route_element_counts": {
            key: value for key, value in sorted(tags.items()) if key != "trip"
        },
    }


def _native_config_inventory(path: Path) -> dict[str, Any]:
    root = parse_xml(path).getroot()

    def value(section: str, tag: str) -> str | None:
        element = root.find(f"./{section}/{tag}")
        return None if element is None else element.attrib.get("value")

    return {
        "net_file": value("input", "net-file"),
        "route_files": value("input", "route-files"),
        "additional_files": value("input", "additional-files"),
        "begin_sec": value("time", "begin"),
        "end_sec": value("time", "end"),
        "step_length_sec": value("time", "step-length"),
        "mesosim": value("mesoscopic", "mesosim"),
        "ignore_route_errors": value("processing", "ignore-route-errors"),
        "collision_action": value("processing", "collision.action"),
        "time_to_teleport_sec": value("processing", "time-to-teleport"),
        "max_depart_delay_sec": value("processing", "max-depart-delay"),
        "random_depart_offset_sec": value(
            "processing", "random-depart-offset"
        ),
        "rerouting_probability": value(
            "routing", "device.rerouting.probability"
        ),
        "rerouting_period_sec": value(
            "routing", "device.rerouting.period"
        ),
        "seed": value("random_number", "seed"),
    }


def _write_microscopic_additional(
    source: Path, destination: Path
) -> dict[str, Any]:
    source_root = parse_xml(source).getroot()
    output_tags = {"edgeData", "laneData", "inductionLoop", "laneAreaDetector"}
    removed = []
    for child in list(source_root):
        if child.tag in output_tags:
            removed.append({"tag": child.tag, "id": child.attrib.get("id")})
            source_root.remove(child)
    ET.indent(source_root, space="    ")
    ET.ElementTree(source_root).write(
        destination, encoding="utf-8", xml_declaration=True
    )
    return {
        "protocol": "eth-five-city-passive-output-removal-v1",
        "removed_passive_output_elements": removed,
        "remaining_element_count": len(source_root),
        "demand_or_network_elements_removed": 0,
    }


def _write_microscopic_routes(source: Path, destination: Path) -> dict[str, Any]:
    """Canonicalize the published destination-TAZ field without changing trips."""

    counts: Counter[str] = Counter()
    depth = 0
    root_tag: str | None = None
    with destination.open("wb") as output:
        output.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        for event, element in ET.iterparse(source, events=("start", "end")):
            if event == "start":
                depth += 1
                if depth == 1:
                    root_tag = str(element.tag)
                    if root_tag != "routes":
                        raise ValueError(
                            f"ETH route root changed: {root_tag} != routes"
                        )
                    attributes = "".join(
                        f" {key}={quoteattr(str(value))}"
                        for key, value in element.attrib.items()
                    )
                    output.write(f"<routes{attributes}>\n".encode("utf-8"))
                continue

            if depth == 2:
                counts["direct_children"] += 1
                if element.tag == "trip":
                    counts["trips"] += 1
                    custom = element.attrib.get("trip_toTaz")
                    canonical = element.attrib.get("toTaz")
                    if custom is None and canonical is None:
                        raise ValueError(
                            f"trip {element.attrib.get('id')} has no destination TAZ"
                        )
                    if (
                        custom is not None
                        and canonical is not None
                        and custom != canonical
                    ):
                        raise ValueError(
                            "trip has conflicting destination TAZ attributes: "
                            f"{element.attrib.get('id')}"
                        )
                    if custom is not None:
                        element.attrib.pop("trip_toTaz")
                        element.set("toTaz", custom)
                        counts["renamed_trip_to_taz"] += 1
                    else:
                        counts["preexisting_to_taz"] += 1
                else:
                    counts[f"non_trip:{element.tag}"] += 1
                element.tail = None
                output.write(ET.tostring(element, encoding="utf-8"))
                output.write(b"\n")
                element.clear()
            if depth == 1 and element.tag == root_tag:
                output.write(b"</routes>\n")
            depth -= 1
    if depth != 0 or root_tag != "routes":
        raise ValueError("ETH route XML ended with an invalid structure")
    return {
        "protocol": ROUTE_CANONICALIZATION_PROTOCOL,
        "source": str(source),
        "destination": str(destination),
        "trip_count": counts["trips"],
        "renamed_trip_to_taz_count": counts["renamed_trip_to_taz"],
        "preexisting_to_taz_count": counts["preexisting_to_taz"],
        "direct_child_count": counts["direct_children"],
        "non_trip_element_counts": {
            key.split(":", 1)[1]: value
            for key, value in sorted(counts.items())
            if key.startswith("non_trip:")
        },
        "trip_anchor_edges_changed": 0,
        "trips_removed": 0,
        "route_repair_applied": False,
    }


def _set_option(root: ET.Element, section: str, tag: str, value: str) -> None:
    parent = root.find(f"./{section}")
    if parent is None:
        parent = ET.SubElement(root, section)
    option = parent.find(f"./{tag}")
    if option is None:
        option = ET.SubElement(parent, tag)
    option.attrib.clear()
    option.set("value", value)


def _write_microscopic_config(
    *,
    source: Path,
    destination: Path,
    city_code: str,
    begin_sec: float,
    end_sec: float,
    seed: int,
    network_file: str | None = None,
) -> dict[str, Any]:
    root = parse_xml(source).getroot()
    for mesoscopic in list(root.findall("./mesoscopic")):
        root.remove(mesoscopic)
    removed_output_options = []
    for output in list(root.findall("./output")):
        removed_output_options.extend(
            {
                "tag": child.tag,
                "value": child.attrib.get("value"),
            }
            for child in list(output)
        )
        root.remove(output)
    _set_option(
        root,
        "input",
        "net-file",
        network_file or f"source/{city_code}/{city_code}.net.xml",
    )
    _set_option(
        root,
        "input",
        "route-files",
        "microscopic_trips.rou.xml",
    )
    _set_option(
        root,
        "input",
        "additional-files",
        f"microscopic_inputs.add.xml,source/{city_code}/taz.xml",
    )
    _set_option(root, "time", "begin", str(int(math.floor(begin_sec))))
    _set_option(root, "time", "end", str(int(math.ceil(end_sec))))
    _set_option(root, "time", "step-length", "1")
    _set_option(root, "processing", "ignore-route-errors", "false")
    _set_option(root, "processing", "collision.action", "warn")
    _set_option(root, "processing", "collision.check-junctions", "true")
    _set_option(root, "processing", "time-to-teleport", "-1")
    _set_option(root, "processing", "max-depart-delay", "-1")
    _set_option(root, "random_number", "random", "false")
    _set_option(root, "random_number", "seed", str(int(seed)))
    _set_option(root, "report", "xml-validation", "never")
    _set_option(root, "report", "xml-validation.routes", "never")
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        destination, encoding="utf-8", xml_declaration=True
    )
    return {
        "protocol": "eth-five-city-config-passive-output-removal-v1",
        "removed_output_options": removed_output_options,
        "removed_output_option_count": len(removed_output_options),
        "demand_or_network_elements_removed": 0,
    }


def package_remote_source(
    *,
    acquisition_root: Path,
    output_root: Path,
    city_code: str,
    seed: int,
) -> dict[str, Any]:
    code = city_code.upper()
    if code not in CITY_NAMES:
        raise ValueError(f"unsupported ETH city code: {city_code}")
    source_manifest, expected_entries = _source_manifest(
        acquisition_root, city_code=code
    )
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        extraction = subprocess.run(
            [
                "unzip",
                "-q",
                str(acquisition_root / ARCHIVE_NAME),
                f"{code}/*",
                "-d",
                str(staging / "source"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if extraction.returncode != 0:
            raise RuntimeError(f"{code} extraction failed: {extraction.stderr}")
        city_root = staging / f"source/{code}"
        expected_files = {Path(value).name for value in expected_entries[1:]}
        observed_files = {
            path.name for path in city_root.iterdir() if path.is_file()
        }
        if observed_files != expected_files:
            raise ValueError(
                f"extracted {code} files changed: {sorted(observed_files)}"
            )

        source_network_path = city_root / f"{code}.net.xml"
        network, edge_ids = _network_inventory(source_network_path)
        microscopic_network_path = staging / "microscopic_network.net.xml"
        network_repair = _repair_unsupported_zipper_junctions(
            source_network_path,
            microscopic_network_path,
        )
        microscopic_network, microscopic_edge_ids = _network_inventory(
            microscopic_network_path
        )
        taz, taz_ids = _taz_inventory(city_root / "taz.xml")
        demand = _route_inventory(
            city_root / "trips24h_smoothed.rou.xml",
            edge_ids=edge_ids,
            taz_ids=taz_ids,
        )
        route_canonicalization = _write_microscopic_routes(
            city_root / "trips24h_smoothed.rou.xml",
            staging / "microscopic_trips.rou.xml",
        )
        route_canonicalization["source"] = (
            f"source/{code}/trips24h_smoothed.rou.xml"
        )
        route_canonicalization["destination"] = "microscopic_trips.rou.xml"
        microscopic_demand = _route_inventory(
            staging / "microscopic_trips.rou.xml",
            edge_ids=microscopic_edge_ids,
            taz_ids=taz_ids,
        )
        native_path = city_root / "meso.sumo.cfg"
        native = _native_config_inventory(native_path)
        additional = _write_microscopic_additional(
            city_root / "additional.add.xml",
            staging / "microscopic_inputs.add.xml",
        )
        minimum_depart = float(demand["minimum_depart_sec"])
        maximum_depart = float(demand["maximum_depart_sec"])
        simulation_end = maximum_depart + DRAIN_AFTER_LAST_DEPART_SEC
        config_output_change = _write_microscopic_config(
            source=native_path,
            destination=staging / "microscopic.sumo.cfg",
            city_code=code,
            begin_sec=minimum_depart,
            end_sec=simulation_end,
            seed=int(seed),
            network_file="microscopic_network.net.xml",
        )
        expected_minimum, expected_maximum = EXPECTED_DEPARTURE_BOUNDS[code]
        gates = {
            "source_archive_identity": True,
            "complete_city_archive_entries": True,
            "nonempty_unique_demand": (
                demand["trip_count"] > 0
                and demand["trip_count"] == demand["unique_trip_id_count"]
                and demand["duplicate_trip_id_count"] == 0
            ),
            "complete_trip_attributes": (
                demand["missing_required_attribute_count"] == 0
            ),
            "all_trip_edges_exist": demand["missing_edge_anchor_count"] == 0,
            "all_trip_taz_exist": demand["missing_taz_anchor_count"] == 0,
            "published_destination_taz_field_is_unambiguous": (
                demand["custom_trip_to_taz_attribute_count"]
                == demand["trip_count"]
                and demand["canonical_to_taz_attribute_count"] == 0
                and demand["conflicting_destination_taz_attribute_count"] == 0
            ),
            "destination_taz_schema_canonicalized_without_demand_loss": (
                route_canonicalization["trip_count"] == demand["trip_count"]
                and route_canonicalization["renamed_trip_to_taz_count"]
                == demand["trip_count"]
                and route_canonicalization["preexisting_to_taz_count"] == 0
                and route_canonicalization["trip_anchor_edges_changed"] == 0
                and route_canonicalization["trips_removed"] == 0
                and route_canonicalization["route_repair_applied"] is False
                and microscopic_demand["trip_count"] == demand["trip_count"]
                and microscopic_demand["unique_trip_id_count"]
                == demand["unique_trip_id_count"]
                and microscopic_demand["custom_trip_to_taz_attribute_count"]
                == 0
                and microscopic_demand["canonical_to_taz_attribute_count"]
                == demand["trip_count"]
                and microscopic_demand["missing_required_attribute_count"] == 0
                and microscopic_demand["missing_edge_anchor_count"] == 0
                and microscopic_demand["missing_taz_anchor_count"] == 0
                and microscopic_demand["minimum_depart_sec"]
                == demand["minimum_depart_sec"]
                and microscopic_demand["maximum_depart_sec"]
                == demand["maximum_depart_sec"]
                and microscopic_demand["nonmonotonic_departure_count"] == 0
            ),
            "monotonic_departures": (
                demand["nonmonotonic_departure_count"] == 0
            ),
            "previously_screened_departure_bounds_match": (
                abs(minimum_depart - expected_minimum)
                <= DEPARTURE_BOUND_TOLERANCE_SEC
                and abs(maximum_depart - expected_maximum)
                <= DEPARTURE_BOUND_TOLERANCE_SEC
            ),
            "published_approximately_12h_span": (
                MINIMUM_PUBLISHED_SPAN_SEC
                <= float(demand["depart_span_sec"])
                <= MAXIMUM_PUBLISHED_SPAN_SEC
            ),
            "controllable_traffic_lights_present": (
                microscopic_network["traffic_light_count"] > 0
                and microscopic_network["traffic_light_phase_count"] > 0
            ),
            "sumo122_zipper_compatibility_repaired": network_repair["passed"],
            "network_edge_identity_preserved": (
                edge_ids == microscopic_edge_ids
            ),
            "native_config_input_boundary_verified": native["net_file"]
            == f"{code}.net.xml"
            and native["route_files"] == "trips24h_smoothed.rou.xml"
            and native["additional_files"] == "additional.add.xml,taz.xml"
            and native["mesosim"] == "true",
            "only_passive_output_removed": (
                additional["demand_or_network_elements_removed"] == 0
                and config_output_change[
                    "demand_or_network_elements_removed"
                ]
                == 0
            ),
        }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "pre-simulation-input-admission",
            "scenario": f"eth_{CITY_NAMES[code]}_complete_published_micro",
            "city_code": code,
            "source": {
                "doi": source_manifest["doi"],
                "archive_name": source_manifest["archive_name"],
                "archive_size_bytes": source_manifest["size_bytes"],
                "archive_md5": source_manifest["md5"],
                "license": source_manifest["license"],
                "city_entries": list(expected_entries),
            },
            "input_audit_protocol": INPUT_AUDIT_PROTOCOL,
            "network_inventory": network,
            "microscopic_network_inventory": microscopic_network,
            "microscopic_network_compatibility": network_repair,
            "taz_inventory": taz,
            "demand_inventory": demand,
            "microscopic_demand_inventory": microscopic_demand,
            "destination_taz_schema_canonicalization": route_canonicalization,
            "native_meso_config_inventory": native,
            "microscopic_instantiation": {
                "protocol": MICRO_CONFIG_PROTOCOL,
                "config": "microscopic.sumo.cfg",
                "seed": int(seed),
                "begin_sec": int(math.floor(minimum_depart)),
                "last_depart_sec": maximum_depart,
                "drain_after_last_depart_sec": DRAIN_AFTER_LAST_DEPART_SEC,
                "end_sec": int(math.ceil(simulation_end)),
                "mode": "microscopic",
                "backend_required_by_protocol": "libsumo",
                "step_length_sec": 1,
                "time_to_teleport_sec": -1,
                "collision_check_junctions": True,
                "collision_action": "warn",
                "ignore_route_errors": False,
                "max_depart_delay_sec": -1,
                "route_file": "microscopic_trips.rou.xml",
                "network_file": "microscopic_network.net.xml",
                "passive_output_change": {
                    "additional_file": additional,
                    "configuration": config_output_change,
                },
                "transformation": (
                    "structured_destination_taz_schema_canonicalization_plus_"
                    "declared_sumo122_zipper_compatibility_normalization_plus_"
                    "native_config_copy_with_explicit_microscopic_and_strict_"
                    "safety_overrides"
                ),
            },
            "gates": gates,
            "passed": all(gates.values()),
        }
        _write_json(staging / "package_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--city-code", choices=sorted(CITY_NAMES), required=True)
    parser.add_argument("--seed", type=int, default=5057)
    args = parser.parse_args(argv)
    payload = package_remote_source(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        city_code=args.city_code,
        seed=int(args.seed),
    )
    print(json.dumps({"status": "PASS" if payload["passed"] else "REJECT"}))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
