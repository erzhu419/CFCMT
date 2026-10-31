#!/usr/bin/env python3
"""Audit and package the complete fixed InTAS scenario for SUMO 1.22."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "cfcmt-intas-full-day-sumo122-package-v1"
ACQUISITION_PROTOCOL = "cfcmt-intas-fixed-git-acquisition-v1"
COMMIT = "0f7951ba01dda8483f0a852f2c3e4ff0d8a1c0ee"
EXPECTED_BLOB_COUNT = 35
EXPECTED_BLOB_BYTES = 983_190_873
CAR_ROUTE_NAMES = tuple(
    f"routes/InTAS_{index:03d}.rou.xml" for index in range(1, 23)
)
ROUTE_NAMES = ("routes/ped.rou.xml", "routes/BusRoutes.flow.xml", *CAR_ROUTE_NAMES)
SOURCE_ADDITIONAL_NAMES = (
    "BusStations.add.xml",
    "InTAS_E1.add.xml",
    "buildings.poly.xml",
)
SEED = 5107
STEP_LENGTH_SEC = 0.1
END_SEC = 108_000
WORKERS = 8


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _iter_xml(path: Path) -> Iterable[ET.Element]:
    for _, element in ET.iterparse(path, events=("end",)):
        yield element
        element.clear()


def _acquisition_manifest(acquisition_root: Path) -> dict[str, Any]:
    path = acquisition_root / "acquisition_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": ACQUISITION_PROTOCOL,
        "repository": "silaslobo/InTAS",
        "commit": COMMIT,
        "blob_count": EXPECTED_BLOB_COUNT,
        "blob_bytes": EXPECTED_BLOB_BYTES,
        "source_subdirectory": "source",
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"InTAS acquisition manifest is not frozen: {observed} != {expected}"
        )
    blobs = payload.get("blobs", [])
    if len(blobs) != EXPECTED_BLOB_COUNT:
        raise ValueError("InTAS acquisition blob inventory is incomplete")
    source = acquisition_root / "source"
    for row in blobs:
        file_path = source / str(row["path"])
        expected_size = int(row["size_bytes"])
        if not file_path.is_file() or file_path.stat().st_size != expected_size:
            raise ValueError(f"InTAS acquired blob changed: {row['path']}")
    return payload


def _network_inventory(
    path: Path,
) -> tuple[dict[str, Any], set[str], set[str]]:
    edge_ids: set[str] = set()
    lane_ids: set[str] = set()
    junction_ids: set[str] = set()
    tls_ids: set[str] = set()
    tls_programs: set[tuple[str, str]] = set()
    duplicate_edges = 0
    duplicate_lanes = 0
    duplicate_junctions = 0
    duplicate_tls_programs = 0
    counts: Counter[str] = Counter()
    for element in _iter_xml(path):
        counts[element.tag] += 1
        identity = str(element.attrib.get("id", ""))
        if element.tag == "edge" and identity:
            duplicate_edges += int(identity in edge_ids)
            edge_ids.add(identity)
        elif element.tag == "lane" and identity:
            duplicate_lanes += int(identity in lane_ids)
            lane_ids.add(identity)
        elif element.tag == "junction" and identity:
            duplicate_junctions += int(identity in junction_ids)
            junction_ids.add(identity)
        elif element.tag == "tlLogic" and identity:
            tls_ids.add(identity)
            program = (identity, str(element.attrib.get("programID", "")))
            duplicate_tls_programs += int(program in tls_programs)
            tls_programs.add(program)
    return (
        {
            "edge_count": len(edge_ids),
            "noninternal_edge_count": sum(
                not identity.startswith(":") for identity in edge_ids
            ),
            "lane_count": len(lane_ids),
            "junction_count": len(junction_ids),
            "connection_count": counts["connection"],
            "traffic_light_count": len(tls_ids),
            "traffic_light_program_count": len(tls_programs),
            "traffic_light_phase_count": counts["phase"],
            "duplicate_edge_id_count": duplicate_edges,
            "duplicate_lane_id_count": duplicate_lanes,
            "duplicate_junction_id_count": duplicate_junctions,
            "duplicate_tls_program_count": duplicate_tls_programs,
        },
        edge_ids,
        lane_ids,
    )


def _audit_additional_file(
    path: Path, *, lane_ids: set[str]
) -> tuple[dict[str, Any], set[str]]:
    counts: Counter[str] = Counter()
    bus_stop_ids: set[str] = set()
    duplicate_bus_stops = 0
    missing_lane_references = 0
    missing_lane_examples: list[str] = []
    lane_tags = {"busStop", "parkingArea", "inductionLoop", "laneAreaDetector"}
    for element in _iter_xml(path):
        counts[element.tag] += 1
        if element.tag == "busStop":
            identity = str(element.attrib.get("id", ""))
            duplicate_bus_stops += int(identity in bus_stop_ids)
            bus_stop_ids.add(identity)
        if element.tag in lane_tags:
            lane = str(element.attrib.get("lane", ""))
            if not lane or lane not in lane_ids:
                missing_lane_references += 1
                if len(missing_lane_examples) < 20:
                    missing_lane_examples.append(lane)
    return (
        {
            "file": path.name,
            "tag_counts": dict(sorted(counts.items())),
            "bus_stop_count": len(bus_stop_ids),
            "duplicate_bus_stop_id_count": duplicate_bus_stops,
            "missing_lane_reference_count": missing_lane_references,
            "missing_lane_examples": missing_lane_examples,
        },
        bus_stop_ids,
    )


def _audit_car_route_file(
    path_text: str, edge_ids: set[str]
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    path = Path(path_text)
    vehicle_ids: set[str] = set()
    vtype_ids: set[str] = set()
    referenced_types: set[str] = set()
    duplicate_vehicle_ids = 0
    duplicate_vtypes = 0
    missing_required = 0
    missing_route_edges = 0
    missing_edge_examples: list[str] = []
    empty_routes = 0
    route_edge_references = 0
    minimum_depart: float | None = None
    maximum_depart: float | None = None
    previous_depart: float | None = None
    nonmonotonic_departures = 0
    counts: Counter[str] = Counter()
    for element in _iter_xml(path):
        counts[element.tag] += 1
        if element.tag == "vType":
            identity = str(element.attrib.get("id", ""))
            duplicate_vtypes += int(identity in vtype_ids)
            vtype_ids.add(identity)
        elif element.tag == "route":
            edges = str(element.attrib.get("edges", "")).split()
            empty_routes += int(not edges)
            route_edge_references += len(edges)
            for edge in edges:
                if edge not in edge_ids:
                    missing_route_edges += 1
                    if len(missing_edge_examples) < 20:
                        missing_edge_examples.append(edge)
        elif element.tag == "vehicle":
            identity = str(element.attrib.get("id", ""))
            depart_text = element.attrib.get("depart")
            vehicle_type = str(element.attrib.get("type", ""))
            if not identity or depart_text is None or not vehicle_type:
                missing_required += 1
                continue
            duplicate_vehicle_ids += int(identity in vehicle_ids)
            vehicle_ids.add(identity)
            referenced_types.add(vehicle_type)
            depart = float(depart_text)
            minimum_depart = depart if minimum_depart is None else min(minimum_depart, depart)
            maximum_depart = depart if maximum_depart is None else max(maximum_depart, depart)
            if previous_depart is not None and depart < previous_depart:
                nonmonotonic_departures += 1
            previous_depart = depart
    stats = {
        "file": path.name,
        "vehicle_count": counts["vehicle"],
        "unique_vehicle_id_count": len(vehicle_ids),
        "duplicate_vehicle_id_count": duplicate_vehicle_ids,
        "vtype_count": counts["vType"],
        "unique_vtype_id_count": len(vtype_ids),
        "duplicate_vtype_id_count": duplicate_vtypes,
        "route_distribution_count": counts["routeDistribution"],
        "route_count": counts["route"],
        "empty_route_count": empty_routes,
        "route_edge_reference_count": route_edge_references,
        "missing_route_edge_count": missing_route_edges,
        "missing_route_edge_examples": missing_edge_examples,
        "missing_required_attribute_count": missing_required,
        "minimum_depart_sec": minimum_depart,
        "maximum_depart_sec": maximum_depart,
        "nonmonotonic_departure_count": nonmonotonic_departures,
        "undeclared_type_references": sorted(referenced_types - vtype_ids),
    }
    return stats, vehicle_ids, vtype_ids, referenced_types


def _flow_count(attributes: Mapping[str, str]) -> int:
    begin = float(attributes["begin"])
    end = float(attributes["end"])
    if end <= begin:
        return 0
    if "number" in attributes:
        return int(attributes["number"])
    if "period" in attributes:
        period = float(attributes["period"])
        if period <= 0:
            raise ValueError("nonpositive flow period")
        return int(math.ceil((end - begin) / period - 1e-12))
    if "vehsPerHour" in attributes:
        return int(math.ceil((end - begin) * float(attributes["vehsPerHour"]) / 3600))
    raise ValueError("flow lacks number, period, and vehsPerHour")


def _audit_bus_routes(
    path_text: str, edge_ids: set[str], bus_stop_ids: set[str]
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    path = Path(path_text)
    flow_ids: set[str] = set()
    vtype_ids: set[str] = set()
    referenced_types: set[str] = set()
    duplicate_flow_ids = 0
    duplicate_vtypes = 0
    expanded_vehicles = 0
    missing_route_edges = 0
    missing_edge_examples: list[str] = []
    missing_bus_stops = 0
    missing_bus_stop_examples: list[str] = []
    route_edge_references = 0
    minimum_begin: float | None = None
    maximum_end: float | None = None
    counts: Counter[str] = Counter()
    invalid_flows = 0
    for element in _iter_xml(path):
        counts[element.tag] += 1
        if element.tag == "vType":
            identity = str(element.attrib.get("id", ""))
            duplicate_vtypes += int(identity in vtype_ids)
            vtype_ids.add(identity)
        elif element.tag == "route":
            edges = str(element.attrib.get("edges", "")).split()
            route_edge_references += len(edges)
            for edge in edges:
                if edge not in edge_ids:
                    missing_route_edges += 1
                    if len(missing_edge_examples) < 20:
                        missing_edge_examples.append(edge)
        elif element.tag == "stop":
            stop = str(element.attrib.get("busStop", ""))
            if not stop or stop not in bus_stop_ids:
                missing_bus_stops += 1
                if len(missing_bus_stop_examples) < 20:
                    missing_bus_stop_examples.append(stop)
        elif element.tag == "flow":
            identity = str(element.attrib.get("id", ""))
            vehicle_type = str(element.attrib.get("type", ""))
            if not identity or not vehicle_type:
                invalid_flows += 1
                continue
            duplicate_flow_ids += int(identity in flow_ids)
            flow_ids.add(identity)
            referenced_types.add(vehicle_type)
            try:
                expanded_vehicles += _flow_count(element.attrib)
                begin = float(element.attrib["begin"])
                end = float(element.attrib["end"])
            except (KeyError, TypeError, ValueError):
                invalid_flows += 1
                continue
            minimum_begin = begin if minimum_begin is None else min(minimum_begin, begin)
            maximum_end = end if maximum_end is None else max(maximum_end, end)
    stats = {
        "file": path.name,
        "flow_count": counts["flow"],
        "unique_flow_id_count": len(flow_ids),
        "duplicate_flow_id_count": duplicate_flow_ids,
        "expanded_vehicle_count": expanded_vehicles,
        "vtype_count": counts["vType"],
        "unique_vtype_id_count": len(vtype_ids),
        "duplicate_vtype_id_count": duplicate_vtypes,
        "route_count": counts["route"],
        "route_edge_reference_count": route_edge_references,
        "missing_route_edge_count": missing_route_edges,
        "missing_route_edge_examples": missing_edge_examples,
        "stop_count": counts["stop"],
        "missing_bus_stop_count": missing_bus_stops,
        "missing_bus_stop_examples": missing_bus_stop_examples,
        "invalid_flow_count": invalid_flows,
        "minimum_begin_sec": minimum_begin,
        "maximum_end_sec": maximum_end,
        "undeclared_type_references": sorted(referenced_types - vtype_ids),
    }
    return stats, flow_ids, vtype_ids, referenced_types


def _audit_pedestrian_routes(
    path_text: str, edge_ids: set[str]
) -> tuple[dict[str, Any], set[str]]:
    path = Path(path_text)
    person_ids: set[str] = set()
    duplicate_person_ids = 0
    missing_walk_edges = 0
    missing_edge_examples: list[str] = []
    walk_edge_references = 0
    minimum_depart: float | None = None
    maximum_depart: float | None = None
    previous_depart: float | None = None
    nonmonotonic_departures = 0
    invalid_persons = 0
    counts: Counter[str] = Counter()
    for element in _iter_xml(path):
        counts[element.tag] += 1
        if element.tag == "walk":
            edges = str(element.attrib.get("edges", "")).split()
            walk_edge_references += len(edges)
            for edge in edges:
                if edge not in edge_ids:
                    missing_walk_edges += 1
                    if len(missing_edge_examples) < 20:
                        missing_edge_examples.append(edge)
        elif element.tag == "person":
            identity = str(element.attrib.get("id", ""))
            depart_text = element.attrib.get("depart")
            if not identity or depart_text is None:
                invalid_persons += 1
                continue
            duplicate_person_ids += int(identity in person_ids)
            person_ids.add(identity)
            depart = float(depart_text)
            minimum_depart = depart if minimum_depart is None else min(minimum_depart, depart)
            maximum_depart = depart if maximum_depart is None else max(maximum_depart, depart)
            if previous_depart is not None and depart < previous_depart:
                nonmonotonic_departures += 1
            previous_depart = depart
    stats = {
        "file": path.name,
        "person_count": counts["person"],
        "unique_person_id_count": len(person_ids),
        "duplicate_person_id_count": duplicate_person_ids,
        "walk_count": counts["walk"],
        "walk_edge_reference_count": walk_edge_references,
        "missing_walk_edge_count": missing_walk_edges,
        "missing_walk_edge_examples": missing_edge_examples,
        "invalid_person_count": invalid_persons,
        "minimum_depart_sec": minimum_depart,
        "maximum_depart_sec": maximum_depart,
        "nonmonotonic_departure_count": nonmonotonic_departures,
    }
    return stats, person_ids


def _source_config(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()

    def value(section: str, option: str) -> str | None:
        node = root.find(f"./{section}/{option}")
        return None if node is None else node.attrib.get("value")

    return {
        "net_file": value("input", "net-file"),
        "route_files": value("input", "route-files"),
        "additional_files": value("input", "additional-files"),
        "begin_sec": value("time", "begin"),
        "end_sec": value("time", "end"),
        "step_length_sec": value("time", "step-length"),
        "route_steps_sec": value("processing", "route-steps"),
        "max_depart_delay_sec": value("processing", "max-depart-delay"),
        "time_to_teleport_sec": value("processing", "time-to-teleport"),
        "rerouting_probability": value("routing", "device.rerouting.probability"),
        "rerouting_period_sec": value("routing", "device.rerouting.period"),
    }


def _write_strict_config(path: Path) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value="source/ingolstadt.net.xml")
    ET.SubElement(
        inputs,
        "route-files",
        value=",".join(f"source/{name}" for name in ROUTE_NAMES),
    )
    ET.SubElement(
        inputs,
        "additional-files",
        value="source/BusStations.add.xml",
    )
    time_section = ET.SubElement(root, "time")
    ET.SubElement(time_section, "begin", value="0")
    ET.SubElement(time_section, "step-length", value=str(STEP_LENGTH_SEC))
    ET.SubElement(time_section, "end", value=str(END_SEC))
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "route-steps", value="200")
    ET.SubElement(processing, "threads", value="1")
    ET.SubElement(processing, "ignore-junction-blocker", value="15")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    ET.SubElement(processing, "scale", value="1")
    ET.SubElement(processing, "max-depart-delay", value="-1")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.SubElement(processing, "collision.check-junctions", value="true")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    ET.SubElement(processing, "time-to-teleport.highways", value="-1")
    ET.SubElement(processing, "default.carfollowmodel", value="Krauss")
    ET.SubElement(processing, "parking.maneuver", value="true")
    ET.SubElement(processing, "pedestrian.model", value="striping")
    ET.SubElement(processing, "pedestrian.striping.stripe-width", value="0.55")
    ET.SubElement(processing, "pedestrian.striping.jamtime", value="30")
    routing = ET.SubElement(root, "routing")
    ET.SubElement(routing, "routing-algorithm", value="dijkstra")
    explicit_types = [
        *(f"default_{index:03d}" for index in range(1, 23)),
        *(f"random_{index:03d}" for index in range(1, 23)),
    ]
    ET.SubElement(
        routing,
        "device.rerouting.explicit",
        value=",".join(explicit_types),
    )
    ET.SubElement(routing, "device.rerouting.probability", value="0.82")
    ET.SubElement(routing, "device.rerouting.period", value="300")
    ET.SubElement(routing, "device.rerouting.threads", value="1")
    ET.SubElement(routing, "device.rerouting.synchronize", value="true")
    ET.SubElement(routing, "persontrip.transfer.car-walk", value="allJunctions")
    random_number = ET.SubElement(root, "random_number")
    ET.SubElement(random_number, "random", value="false")
    ET.SubElement(random_number, "seed", value=str(SEED))
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", value="false")
    ET.SubElement(report, "no-warnings", value="false")
    ET.SubElement(report, "duration-log.statistics", value="true")
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _link_source(source_root: Path, staging: Path) -> None:
    destination = staging / "source"
    (destination / "routes").mkdir(parents=True)
    names = (
        "ingolstadt.net.xml",
        "BusStations.add.xml",
        *ROUTE_NAMES,
    )
    for name in names:
        source = source_root / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)


def package_source(
    *, acquisition_root: Path, output_root: Path, workers: int
) -> dict[str, Any]:
    if workers != WORKERS:
        raise ValueError(f"InTAS package audit is frozen at {WORKERS} workers")
    acquisition_root = acquisition_root.resolve()
    output_root = output_root.resolve()
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"InTAS package destination already exists: {output_root}")
    source_manifest = _acquisition_manifest(acquisition_root)
    scenario = acquisition_root / "source" / "scenario"
    network, edge_ids, lane_ids = _network_inventory(scenario / "ingolstadt.net.xml")
    bus_stations, bus_stop_ids = _audit_additional_file(
        scenario / "BusStations.add.xml",
        lane_ids=lane_ids,
    )
    e1_inventory, _ = _audit_additional_file(
        scenario / "InTAS_E1.add.xml",
        lane_ids=lane_ids,
    )
    native_config = _source_config(scenario / "InTAS_buildings.sumocfg")

    car_results: dict[str, tuple[dict[str, Any], set[str], set[str], set[str]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_audit_car_route_file, str(scenario / name), edge_ids): name
            for name in CAR_ROUTE_NAMES
        }
        for future in as_completed(futures):
            car_results[futures[future]] = future.result()
    ordered_car_results = [car_results[name] for name in CAR_ROUTE_NAMES]
    car_file_stats = [row[0] for row in ordered_car_results]

    all_vehicle_ids: set[str] = set()
    duplicate_global_vehicle_ids = 0
    all_vtypes: set[str] = set()
    all_referenced_types: set[str] = set()
    for _, vehicle_ids, vtypes, referenced_types in ordered_car_results:
        duplicate_global_vehicle_ids += len(all_vehicle_ids & vehicle_ids)
        all_vehicle_ids.update(vehicle_ids)
        all_vtypes.update(vtypes)
        all_referenced_types.update(referenced_types)

    bus_stats, flow_ids, bus_vtypes, bus_referenced_types = _audit_bus_routes(
        str(scenario / "routes" / "BusRoutes.flow.xml"),
        edge_ids,
        bus_stop_ids,
    )
    pedestrian_stats, person_ids = _audit_pedestrian_routes(
        str(scenario / "routes" / "ped.rou.xml"),
        edge_ids,
    )

    expected_native_routes = ",".join(ROUTE_NAMES)
    expected_native_additional = ",".join(SOURCE_ADDITIONAL_NAMES)
    car_depart_min = min(
        float(row["minimum_depart_sec"])
        for row in car_file_stats
        if row["minimum_depart_sec"] is not None
    )
    car_depart_max = max(
        float(row["maximum_depart_sec"])
        for row in car_file_stats
        if row["maximum_depart_sec"] is not None
    )
    static_failures = []
    if any(
        network[key]
        for key in (
            "duplicate_edge_id_count",
            "duplicate_lane_id_count",
            "duplicate_junction_id_count",
            "duplicate_tls_program_count",
        )
    ):
        static_failures.append("network identity duplication")
    if network["traffic_light_count"] <= 0 or network["traffic_light_phase_count"] <= 0:
        static_failures.append("network has no controllable traffic lights")
    if bus_stations["duplicate_bus_stop_id_count"] or bus_stations["missing_lane_reference_count"]:
        static_failures.append("bus-stop inventory is invalid")
    if e1_inventory["missing_lane_reference_count"]:
        static_failures.append("E1 detector inventory has missing lane references")
    for row in car_file_stats:
        if any(
            row[key]
            for key in (
                "duplicate_vehicle_id_count",
                "duplicate_vtype_id_count",
                "empty_route_count",
                "missing_route_edge_count",
                "missing_required_attribute_count",
                "nonmonotonic_departure_count",
            )
        ) or row["undeclared_type_references"]:
            static_failures.append(f"invalid car route file: {row['file']}")
    if duplicate_global_vehicle_ids:
        static_failures.append(
            f"duplicate global vehicle identities={duplicate_global_vehicle_ids}"
        )
    if all_referenced_types - all_vtypes:
        static_failures.append("global car type references are undeclared")
    if any(
        bus_stats[key]
        for key in (
            "duplicate_flow_id_count",
            "duplicate_vtype_id_count",
            "missing_route_edge_count",
            "missing_bus_stop_count",
            "invalid_flow_count",
        )
    ) or bus_stats["undeclared_type_references"]:
        static_failures.append("bus route inventory is invalid")
    if bus_referenced_types - bus_vtypes:
        static_failures.append("global bus type references are undeclared")
    if any(
        pedestrian_stats[key]
        for key in (
            "duplicate_person_id_count",
            "missing_walk_edge_count",
            "invalid_person_count",
            "nonmonotonic_departure_count",
        )
    ):
        static_failures.append("pedestrian route inventory is invalid")
    if not (
        car_depart_min == 0.0
        and 86_390.0 <= car_depart_max <= 86_400.0
        and pedestrian_stats["minimum_depart_sec"] == 0.0
    ):
        static_failures.append("route sources do not span the frozen full day")
    if native_config != {
        "net_file": "ingolstadt.net.xml",
        "route_files": expected_native_routes,
        "additional_files": expected_native_additional,
        "begin_sec": "0",
        "end_sec": "86400",
        "step_length_sec": "0.1",
        "route_steps_sec": "200",
        "max_depart_delay_sec": "300",
        "time_to_teleport_sec": "300",
        "rerouting_probability": "0.82",
        "rerouting_period_sec": "300",
    }:
        static_failures.append("published native configuration boundary changed")

    staging.mkdir(parents=True)
    try:
        _link_source(scenario, staging)
        _write_strict_config(staging / "strict_full_day.sumocfg")
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "pre-simulation-input-admission",
            "scenario": "intas_ingolstadt_full_day",
            "workers": workers,
            "source": {
                "acquisition_protocol": source_manifest["protocol"],
                "repository": source_manifest["repository"],
                "commit": source_manifest["commit"],
                "archive_sha256": source_manifest["archive_sha256"],
                "archive_size_bytes": source_manifest["archive_size_bytes"],
                "blob_count": source_manifest["blob_count"],
                "blob_bytes": source_manifest["blob_bytes"],
                "license": "GPL-3.0",
            },
            "native_config_inventory": native_config,
            "network_inventory": network,
            "bus_station_inventory": bus_stations,
            "e1_detector_inventory": e1_inventory,
            "car_route_files": car_file_stats,
            "car_demand_inventory": {
                "route_file_count": len(car_file_stats),
                "vehicle_count": sum(row["vehicle_count"] for row in car_file_stats),
                "unique_vehicle_id_count": len(all_vehicle_ids),
                "duplicate_global_vehicle_id_count": duplicate_global_vehicle_ids,
                "vtype_count": len(all_vtypes),
                "minimum_depart_sec": car_depart_min,
                "maximum_depart_sec": car_depart_max,
            },
            "bus_demand_inventory": bus_stats,
            "pedestrian_demand_inventory": pedestrian_stats,
            "entity_identity_inventory": {
                "vehicle_id_count": len(all_vehicle_ids),
                "flow_id_count": len(flow_ids),
                "person_id_count": len(person_ids),
                "vehicle_flow_id_overlap_count": len(all_vehicle_ids & flow_ids),
                "vehicle_person_id_overlap_count": len(all_vehicle_ids & person_ids),
                "flow_person_id_overlap_count": len(flow_ids & person_ids),
            },
            "strict_execution": {
                "config": "strict_full_day.sumocfg",
                "backend": "libsumo",
                "sumo_version": "1.22.0",
                "seed": SEED,
                "begin_sec": 0,
                "end_sec": END_SEC,
                "step_length_sec": STEP_LENGTH_SEC,
                "time_to_teleport_sec": -1,
                "max_depart_delay_sec": -1,
                "demand_scale": 1.0,
                "collision_check_junctions": True,
                "collision_action": "warn",
                "native_rerouting_probability": 0.82,
                "native_rerouting_period_sec": 300,
                "passive_outputs_omitted": True,
                "hard_linked_immutable_source_inputs": True,
            },
            "static_failures": static_failures,
            "passed": not static_failures,
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
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    payload = package_source(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": "PASS" if payload["passed"] else "REJECT",
                "protocol": payload["protocol"],
                "static_failures": payload["static_failures"],
                "vehicle_count": payload["car_demand_inventory"]["vehicle_count"],
                "expanded_bus_vehicle_count": payload["bus_demand_inventory"][
                    "expanded_vehicle_count"
                ],
                "person_count": payload["pedestrian_demand_inventory"][
                    "person_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
