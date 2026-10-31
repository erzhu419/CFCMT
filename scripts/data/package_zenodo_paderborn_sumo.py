#!/usr/bin/env python3
"""Package and audit the complete Paderborn full-day SUMO scenario."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.sumo_static_inputs import (  # noqa: E402
    iterparse_xml,
    parse_sumo_time_seconds,
    parse_xml,
)
from scripts.data.acquire_zenodo_paderborn import (  # noqa: E402
    CONCEPT_DOI,
    FILES,
    PROTOCOL as ACQUISITION_PROTOCOL,
    RECORD_ID,
    VERSION_DOI,
)


PROTOCOL = "cfcmt-zenodo-paderborn-full-day-sumo122-package-v1"
INPUT_AUDIT_PROTOCOL = "paderborn-trip-dua-route-tls-static-audit-v1"
MICRO_CONFIG_PROTOCOL = "paderborn-native-micro-sumo122-strict-safety-config-v1"
SCENARIO = "paderborn_full_day"
EXPECTED_TRIP_COUNT = 203_387
EXPECTED_MINIMUM_DEPART_SEC = 0.0
EXPECTED_MAXIMUM_DEPART_SEC = 86_400.0
DRAIN_AFTER_LAST_DEPART_SEC = 6 * 3600


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_manifest(acquisition_root: Path) -> dict[str, Any]:
    payload = json.loads(
        (acquisition_root / "acquisition_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_files = {
        name: {"size_bytes": size, "md5": md5}
        for name, size, md5 in FILES
    }
    observed_files = {
        name: {
            "size_bytes": row.get("size_bytes"),
            "md5": row.get("md5"),
        }
        for name, row in payload.get("files", {}).items()
    }
    expected = {
        "protocol": ACQUISITION_PROTOCOL,
        "record_id": RECORD_ID,
        "version_doi": VERSION_DOI,
        "concept_doi": CONCEPT_DOI,
        "file_count": len(FILES),
        "total_size_bytes": sum(size for _, size, _ in FILES),
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected or observed_files != expected_files:
        raise ValueError("Paderborn acquisition manifest is not frozen")
    for name, size, _ in FILES:
        path = acquisition_root / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Paderborn acquired source changed: {name}")
    return payload


def _network_inventory(
    path: Path,
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    edge_ids: set[str] = set()
    lane_ids: set[str] = set()
    junction_ids: set[str] = set()
    embedded_tls_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        counts[element.tag] += 1
        identity = str(element.attrib.get("id", ""))
        if element.tag == "edge" and identity:
            edge_ids.add(identity)
        elif element.tag == "lane" and identity:
            lane_ids.add(identity)
        elif element.tag == "junction" and identity:
            junction_ids.add(identity)
        elif element.tag == "tlLogic" and identity:
            embedded_tls_ids.add(identity)
        element.clear()
    noninternal_edges = {value for value in edge_ids if not value.startswith(":")}
    return (
        {
            "edge_count": len(edge_ids),
            "noninternal_edge_count": len(noninternal_edges),
            "lane_count": len(lane_ids),
            "junction_count": len(junction_ids),
            "connection_count": counts["connection"],
            "embedded_traffic_light_count": len(embedded_tls_ids),
            "embedded_traffic_light_program_count": counts["tlLogic"],
            "embedded_traffic_light_phase_count": counts["phase"],
        },
        edge_ids,
        lane_ids,
        junction_ids,
    )


def _trip_inventory(
    path: Path, *, edge_ids: set[str], lane_ids: set[str]
) -> tuple[dict[str, Any], set[str]]:
    trip_ids: set[str] = set()
    duplicate_ids = 0
    missing_attributes = 0
    missing_edge_anchors = 0
    missing_stop_lanes = 0
    minimum: float | None = None
    maximum: float | None = None
    previous: float | None = None
    nonmonotonic = 0
    type_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        counts[element.tag] += 1
        if element.tag == "stop":
            lane = str(element.attrib.get("lane", ""))
            missing_stop_lanes += int(not lane or lane not in lane_ids)
        elif element.tag == "trip":
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
            type_counts[str(element.attrib.get("type", ""))] += 1
            depart = parse_sumo_time_seconds(depart_text)
            minimum = depart if minimum is None else min(minimum, depart)
            maximum = depart if maximum is None else max(maximum, depart)
            if previous is not None and depart < previous:
                nonmonotonic += 1
            previous = depart
        element.clear()
    return (
        {
            "trip_count": counts["trip"],
            "unique_trip_id_count": len(trip_ids),
            "duplicate_trip_id_count": duplicate_ids,
            "missing_required_attribute_count": missing_attributes,
            "missing_edge_anchor_count": missing_edge_anchors,
            "stop_count": counts["stop"],
            "missing_stop_lane_count": missing_stop_lanes,
            "minimum_depart_sec": minimum,
            "maximum_depart_sec": maximum,
            "depart_span_sec": (
                None if minimum is None or maximum is None else maximum - minimum
            ),
            "nonmonotonic_departure_count": nonmonotonic,
            "vehicle_types": dict(sorted(type_counts.items())),
        },
        trip_ids,
    )


def _route_inventory(
    path: Path, *, edge_ids: set[str], lane_ids: set[str]
) -> tuple[dict[str, Any], set[str]]:
    vehicle_ids: set[str] = set()
    duplicate_ids = 0
    missing_attributes = 0
    missing_route_edges = 0
    empty_routes = 0
    missing_stop_lanes = 0
    route_edge_references = 0
    minimum: float | None = None
    maximum: float | None = None
    previous: float | None = None
    nonmonotonic = 0
    type_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        counts[element.tag] += 1
        if element.tag == "route":
            route_edges = str(element.attrib.get("edges", "")).split()
            empty_routes += int(not route_edges)
            route_edge_references += len(route_edges)
            missing_route_edges += sum(edge not in edge_ids for edge in route_edges)
        elif element.tag == "stop":
            lane = str(element.attrib.get("lane", ""))
            missing_stop_lanes += int(not lane or lane not in lane_ids)
        elif element.tag == "vehicle":
            identity = str(element.attrib.get("id", ""))
            depart_text = element.attrib.get("depart")
            if not identity or depart_text is None:
                missing_attributes += 1
                element.clear()
                continue
            duplicate_ids += int(identity in vehicle_ids)
            vehicle_ids.add(identity)
            type_counts[str(element.attrib.get("type", ""))] += 1
            depart = parse_sumo_time_seconds(depart_text)
            minimum = depart if minimum is None else min(minimum, depart)
            maximum = depart if maximum is None else max(maximum, depart)
            if previous is not None and depart < previous:
                nonmonotonic += 1
            previous = depart
        element.clear()
    return (
        {
            "vehicle_count": counts["vehicle"],
            "unique_vehicle_id_count": len(vehicle_ids),
            "duplicate_vehicle_id_count": duplicate_ids,
            "missing_required_attribute_count": missing_attributes,
            "route_count": counts["route"],
            "route_edge_reference_count": route_edge_references,
            "empty_route_count": empty_routes,
            "missing_route_edge_reference_count": missing_route_edges,
            "stop_count": counts["stop"],
            "missing_stop_lane_count": missing_stop_lanes,
            "minimum_depart_sec": minimum,
            "maximum_depart_sec": maximum,
            "depart_span_sec": (
                None if minimum is None or maximum is None else maximum - minimum
            ),
            "nonmonotonic_departure_count": nonmonotonic,
            "vehicle_types": dict(sorted(type_counts.items())),
        },
        vehicle_ids,
    )


def _vtype_inventory(path: Path) -> tuple[dict[str, Any], set[str]]:
    identifiers: set[str] = set()
    duplicate_ids = 0
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "vType":
            identity = str(element.attrib.get("id", ""))
            duplicate_ids += int(identity in identifiers)
            if identity:
                identifiers.add(identity)
        element.clear()
    return (
        {
            "vehicle_type_count": len(identifiers),
            "vehicle_type_ids": sorted(identifiers),
            "duplicate_vehicle_type_id_count": duplicate_ids,
        },
        identifiers,
    )


def _external_tls_inventory(
    path: Path, *, junction_ids: set[str]
) -> dict[str, Any]:
    identifiers: set[str] = set()
    duplicate_program_ids = 0
    missing_junction_ids = 0
    program_count = 0
    phase_count = 0
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "phase":
            phase_count += 1
        elif element.tag == "tlLogic":
            program_count += 1
            identity = str(element.attrib.get("id", ""))
            program = str(element.attrib.get("programID", ""))
            key = f"{identity}:{program}"
            duplicate_program_ids += int(key in identifiers)
            identifiers.add(key)
            missing_junction_ids += int(identity not in junction_ids)
        element.clear()
    return {
        "controlled_intersection_count": len(
            {value.split(":", 1)[0] for value in identifiers}
        ),
        "program_count": program_count,
        "phase_count": phase_count,
        "duplicate_program_id_count": duplicate_program_ids,
        "missing_network_junction_count": missing_junction_ids,
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
        "step_length_sec": value("time", "step-length"),
        "ignore_route_errors": value("processing", "ignore-route-errors"),
        "scale": value("processing", "scale"),
        "seed": value("random_number", "seed"),
    }


def _write_strict_config(
    *, destination: Path, begin_sec: float, end_sec: float, seed: int
) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value="source/network.net.xml")
    ET.SubElement(inputs, "route-files", value="source/allroutes.rou.xml")
    ET.SubElement(
        inputs,
        "additional-files",
        value="source/trafficlightlogic.tll.xml,source/vtypes.add.xml",
    )
    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value=str(int(math.floor(begin_sec))))
    ET.SubElement(time, "end", value=str(int(math.ceil(end_sec))))
    ET.SubElement(time, "step-length", value="1")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    ET.SubElement(processing, "scale", value="1")
    ET.SubElement(processing, "max-depart-delay", value="-1")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.SubElement(processing, "collision.check-junctions", value="true")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    random_number = ET.SubElement(root, "random_number")
    ET.SubElement(random_number, "random", value="false")
    ET.SubElement(random_number, "seed", value=str(int(seed)))
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "xml-validation", value="never")
    ET.SubElement(report, "xml-validation.routes", value="never")
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        destination, encoding="utf-8", xml_declaration=True
    )


def package_source(
    *, acquisition_root: Path, output_root: Path, seed: int
) -> dict[str, Any]:
    acquisition_root = acquisition_root.resolve()
    source_manifest = _source_manifest(acquisition_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        os.symlink(acquisition_root, staging / "source", target_is_directory=True)
        network, edge_ids, lane_ids, junction_ids = _network_inventory(
            acquisition_root / "network.net.xml"
        )
        trips, trip_ids = _trip_inventory(
            acquisition_root / "alltrips.trp.xml",
            edge_ids=edge_ids,
            lane_ids=lane_ids,
        )
        routes, route_ids = _route_inventory(
            acquisition_root / "allroutes.rou.xml",
            edge_ids=edge_ids,
            lane_ids=lane_ids,
        )
        vtypes, vtype_ids = _vtype_inventory(acquisition_root / "vtypes.add.xml")
        tls = _external_tls_inventory(
            acquisition_root / "trafficlightlogic.tll.xml",
            junction_ids=junction_ids,
        )
        native = _native_config_inventory(acquisition_root / "full-day.sumo.cfg")
        license_head = (acquisition_root / "LICENSE.md").read_text(
            encoding="utf-8"
        )[:300]
        minimum_depart = float(routes["minimum_depart_sec"])
        maximum_depart = float(routes["maximum_depart_sec"])
        simulation_end = maximum_depart + DRAIN_AFTER_LAST_DEPART_SEC
        _write_strict_config(
            destination=staging / "strict_full_day.sumo.cfg",
            begin_sec=minimum_depart,
            end_sec=simulation_end,
            seed=int(seed),
        )
        trip_only = sorted(trip_ids - route_ids)
        route_only = sorted(route_ids - trip_ids)
        observed_route_types = set(routes["vehicle_types"])
        observed_trip_types = set(trips["vehicle_types"])
        gates = {
            "source_record_identity": True,
            "published_trip_count": (
                trips["trip_count"] == EXPECTED_TRIP_COUNT
                and routes["vehicle_count"] == EXPECTED_TRIP_COUNT
            ),
            "unique_trip_and_route_ids": (
                trips["duplicate_trip_id_count"] == 0
                and routes["duplicate_vehicle_id_count"] == 0
                and trips["unique_trip_id_count"] == EXPECTED_TRIP_COUNT
                and routes["unique_vehicle_id_count"] == EXPECTED_TRIP_COUNT
            ),
            "trip_route_identity_match": not trip_only and not route_only,
            "complete_trip_attributes": (
                trips["missing_required_attribute_count"] == 0
                and routes["missing_required_attribute_count"] == 0
            ),
            "all_trip_edges_exist": trips["missing_edge_anchor_count"] == 0,
            "all_route_edges_exist": (
                routes["missing_route_edge_reference_count"] == 0
                and routes["empty_route_count"] == 0
                and routes["route_count"] == routes["vehicle_count"]
            ),
            "all_stop_lanes_exist": (
                trips["missing_stop_lane_count"] == 0
                and routes["missing_stop_lane_count"] == 0
            ),
            "all_vehicle_types_declared": (
                observed_route_types <= vtype_ids and observed_trip_types <= vtype_ids
            ),
            "full_day_departure_span": (
                trips["minimum_depart_sec"] == EXPECTED_MINIMUM_DEPART_SEC
                and trips["maximum_depart_sec"] == EXPECTED_MAXIMUM_DEPART_SEC
                and routes["minimum_depart_sec"] == EXPECTED_MINIMUM_DEPART_SEC
                and routes["maximum_depart_sec"] == EXPECTED_MAXIMUM_DEPART_SEC
            ),
            "monotonic_departures": (
                trips["nonmonotonic_departure_count"] == 0
                and routes["nonmonotonic_departure_count"] == 0
            ),
            "external_tls_complete": (
                tls["controlled_intersection_count"] > 0
                and tls["phase_count"] > 0
                and tls["duplicate_program_id_count"] == 0
                and tls["missing_network_junction_count"] == 0
            ),
            "native_config_boundary_verified": native
            == {
                "net_file": "network.net.xml",
                "route_files": "allroutes.rou.xml",
                "additional_files": (
                    "trafficlightlogic.tll.xml,polygons.poly.xml,vtypes.add.xml"
                ),
                "step_length_sec": "1.0",
                "ignore_route_errors": "true",
                "scale": "1.00",
                "seed": "23456",
            },
            "published_license_is_gplv3": (
                "GNU GENERAL PUBLIC LICENSE" in license_head
                and "Version 3, 29 June 2007" in license_head
            ),
        }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "pre-simulation-input-admission",
            "scenario": SCENARIO,
            "source": {
                "record_id": source_manifest["record_id"],
                "version_doi": source_manifest["version_doi"],
                "concept_doi": source_manifest["concept_doi"],
                "file_count": source_manifest["file_count"],
                "total_size_bytes": source_manifest["total_size_bytes"],
                "license": "GPL-3.0",
                "zenodo_license_metadata": source_manifest["license_metadata"],
                "source_link": str((staging / "source").resolve()),
            },
            "input_audit_protocol": INPUT_AUDIT_PROTOCOL,
            "network_inventory": network,
            "trip_inventory": trips,
            "route_inventory": routes,
            "vehicle_type_inventory": vtypes,
            "external_tls_inventory": tls,
            "native_config_inventory": native,
            "trip_route_identity_difference": {
                "trip_only_count": len(trip_only),
                "route_only_count": len(route_only),
                "trip_only_examples": trip_only[:20],
                "route_only_examples": route_only[:20],
            },
            "microscopic_instantiation": {
                "protocol": MICRO_CONFIG_PROTOCOL,
                "config": "strict_full_day.sumo.cfg",
                "mode": "microscopic",
                "seed": int(seed),
                "begin_sec": int(math.floor(minimum_depart)),
                "last_depart_sec": maximum_depart,
                "drain_after_last_depart_sec": DRAIN_AFTER_LAST_DEPART_SEC,
                "end_sec": int(math.ceil(simulation_end)),
                "step_length_sec": 1,
                "time_to_teleport_sec": -1,
                "collision_check_junctions": True,
                "collision_action": "warn",
                "ignore_route_errors": False,
                "max_depart_delay_sec": -1,
                "demand_scale": 1.0,
                "passive_polygon_input_omitted": True,
            },
            "source_reported_runtime": {
                "sumo_version": "1.6.0",
                "simulated_time_steps": 89_015,
                "teleports": 9,
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
    parser.add_argument("--seed", type=int, default=5057)
    args = parser.parse_args(argv)
    payload = package_source(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        seed=int(args.seed),
    )
    print(json.dumps({"status": "PASS" if payload["passed"] else "REJECT"}))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
