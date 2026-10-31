#!/usr/bin/env python3
"""Package the complete LuST DUE-static variant for read-only libsumo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.data.acquire_lust_scenario import (
    COMMIT,
    DEMAND_FILES,
    FILES,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
)


PROTOCOL = "cfcmt-lust-native-sumo-due-static-package-v1"
NETWORK_BUILD_PROTOCOL = "lust-native-sumo-read-only-due-static-v1"
DEMAND_PROTOCOL = "lust-due-static-complete-demand-inventory-v1"
TLS_PROTOCOL = "lust-native-static-tls-preserved-v1"
SCENARIO = "luxembourg_lust_due_static"
HORIZON_SEC = 86_400.0


def _network_inventory(net_path: Path) -> dict[str, Any]:
    root = ET.parse(net_path).getroot()
    edges = {
        str(edge.attrib["id"])
        for edge in root.findall("edge")
        if not edge.attrib.get("function")
    }
    connections = {
        (str(connection.attrib["from"]), str(connection.attrib["to"]))
        for connection in root.findall("connection")
        if connection.attrib.get("from") in edges
        and connection.attrib.get("to") in edges
    }
    programs = {
        str(logic.attrib["id"]): len(logic.findall("phase"))
        for logic in root.findall("tlLogic")
    }
    if not edges or not programs or any(count <= 0 for count in programs.values()):
        raise ValueError("LuST native SUMO network inventory is invalid")
    return {
        "sumo_net_version": str(root.attrib.get("version", "")),
        "edges": edges,
        "connections": connections,
        "programs": programs,
    }


def _static_tls_inventory(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    programs = {
        str(logic.attrib["id"]): len(logic.findall("phase"))
        for logic in root.iter("tlLogic")
    }
    if not programs or any(count <= 0 for count in programs.values()):
        raise ValueError("LuST static TLS file is invalid")
    return programs


def _demand_inventory(
    *, scenario_root: Path, network: dict[str, Any]
) -> dict[str, Any]:
    route_count = 0
    vehicle_count = 0
    minimum_departure: float | None = None
    maximum_departure: float | None = None
    missing_edges: set[str] = set()
    missing_pairs: set[tuple[str, str]] = set()
    vehicle_ids: set[str] = set()
    duplicate_vehicle_ids: set[str] = set()
    file_rows: dict[str, dict[str, Any]] = {}
    for relative in DEMAND_FILES:
        path = scenario_root / Path(relative).relative_to("scenario")
        file_route_count = 0
        file_vehicle_count = 0
        file_minimum: float | None = None
        file_maximum: float | None = None
        for _, element in ET.iterparse(path, events=("end",)):
            if element.tag == "route" and element.attrib.get("edges"):
                edges = tuple(str(element.attrib["edges"]).split())
                file_route_count += 1
                route_count += 1
                missing_edges.update(set(edges) - network["edges"])
                missing_pairs.update(
                    pair
                    for pair in zip(edges, edges[1:])
                    if pair not in network["connections"]
                )
            elif element.tag == "vehicle":
                identity = str(element.attrib.get("id", ""))
                if not identity:
                    raise ValueError(f"LuST vehicle has no ID in {relative}")
                if identity in vehicle_ids:
                    duplicate_vehicle_ids.add(identity)
                vehicle_ids.add(identity)
                departure = float(element.attrib["depart"])
                file_vehicle_count += 1
                vehicle_count += 1
                file_minimum = (
                    departure if file_minimum is None else min(file_minimum, departure)
                )
                file_maximum = (
                    departure if file_maximum is None else max(file_maximum, departure)
                )
                minimum_departure = (
                    departure
                    if minimum_departure is None
                    else min(minimum_departure, departure)
                )
                maximum_departure = (
                    departure
                    if maximum_departure is None
                    else max(maximum_departure, departure)
                )
            element.clear()
        file_rows[relative] = {
            "route_count": file_route_count,
            "vehicle_count": file_vehicle_count,
            "minimum_departure_sec": file_minimum,
            "maximum_departure_sec": file_maximum,
        }
    if duplicate_vehicle_ids:
        raise ValueError(
            f"LuST demand contains duplicate vehicle IDs: {sorted(duplicate_vehicle_ids)[:5]}"
        )
    if route_count != vehicle_count:
        raise ValueError(
            f"LuST embedded-route count changed: {route_count} != {vehicle_count}"
        )
    if missing_edges or missing_pairs:
        raise ValueError(
            "LuST source routes are invalid in the source network: "
            f"edges={sorted(missing_edges)[:5]} pairs={sorted(missing_pairs)[:5]}"
        )
    if (
        vehicle_count <= 0
        or minimum_departure != 0.0
        or maximum_departure is None
        or maximum_departure >= HORIZON_SEC
    ):
        raise ValueError("LuST source demand does not cover one valid full day")
    return {
        "protocol": DEMAND_PROTOCOL,
        "total": vehicle_count,
        "before_horizon": vehicle_count,
        "at_or_after_horizon": 0,
        "minimum_departure_sec": minimum_departure,
        "maximum_departure_sec": maximum_departure,
        "embedded_route_count": route_count,
        "duplicate_vehicle_id_count": 0,
        "missing_route_edge_count": 0,
        "missing_route_connection_pair_count": 0,
        "all_source_demand_files_loaded": True,
        "source_network_and_demand_files_preserved_byte_for_byte": True,
        "files": file_rows,
    }


def _write_config(path: Path) -> None:
    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", value="lust.net.xml")
    ET.SubElement(
        input_node,
        "route-files",
        value=(
            "buslines.rou.xml,DUERoutes/local.static.0.rou.xml,"
            "DUERoutes/local.static.1.rou.xml,DUERoutes/local.static.2.rou.xml,"
            "transit.rou.xml"
        ),
    )
    ET.SubElement(
        input_node,
        "additional-files",
        value="vtypes.add.xml,busstops.add.xml,tll.static.xml",
    )
    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value="86400")
    ET.SubElement(time_node, "step-length", value="1")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-junction-blocker", value="20")
    ET.SubElement(processing, "max-depart-delay", value="600")
    ET.SubElement(processing, "routing-algorithm", value="dijkstra")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def package_lust(*, acquisition_root: Path, output_root: Path) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite LuST package: {output_root}")
    acquisition = json.loads(
        (acquisition_root / "acquisition_manifest.json").read_text(encoding="utf-8")
    )
    if (
        acquisition.get("protocol") != ACQUISITION_PROTOCOL
        or acquisition.get("commit") != COMMIT
        or set(dict(acquisition.get("files", {}))) != set(FILES)
        or tuple(acquisition.get("demand_files", ())) != DEMAND_FILES
    ):
        raise ValueError("LuST acquisition identity or coverage changed")
    for relative, row in dict(acquisition["files"]).items():
        path = acquisition_root / relative
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"LuST acquired file size changed: {relative}")

    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    scenario_root = staging / SCENARIO
    scenario_root.mkdir(parents=True)
    try:
        source_root = acquisition_root / "scenario"
        retained = (
            "lust.net.xml",
            "buslines.rou.xml",
            "DUERoutes/local.static.0.rou.xml",
            "DUERoutes/local.static.1.rou.xml",
            "DUERoutes/local.static.2.rou.xml",
            "transit.rou.xml",
            "vtypes.add.xml",
            "busstops.add.xml",
            "tll.static.xml",
        )
        for relative in retained:
            destination = scenario_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / relative, destination)
        config_name = f"{SCENARIO}.sumocfg"
        _write_config(scenario_root / config_name)
        network = _network_inventory(scenario_root / "lust.net.xml")
        static_programs = _static_tls_inventory(scenario_root / "tll.static.xml")
        if static_programs != network["programs"]:
            raise ValueError("LuST static TLS IDs or phase counts differ from native net")
        demand = _demand_inventory(scenario_root=scenario_root, network=network)
        network_row = {
            "sumocfg": f"{SCENARIO}/{config_name}",
            "horizon_sec": HORIZON_SEC,
            "vehicle_count": int(demand["total"]),
            "person_count": 0,
            "controlled_intersection_count": len(static_programs),
            "route_audit": {
                "route_count": int(demand["embedded_route_count"]),
                "missing_edge_route_count": 0,
                "disconnected_route_count": 0,
            },
            "connection_audit": {"missing_route_edge_pair_count": 0},
            "demand_audit": demand,
            "network_build": {
                "protocol": NETWORK_BUILD_PROTOCOL,
                "mode": "source_native_sumo_due_static_read_only_inputs",
                "source_network_and_demand_files_preserved_byte_for_byte": True,
                "passive_output_detector_and_polygon_files_removed": True,
                "traffic_semantics_rebuilt_or_calibrated": False,
                "source_sumo_version": "0.26",
                "packaged_net_version": network["sumo_net_version"],
                "modern_sumo_real_data_validation_claimed": False,
            },
            "tls_semantic_audit": {
                "protocol": TLS_PROTOCOL,
                "controlled_intersection_count": len(static_programs),
                "program_count": len(static_programs),
                "phase_count": sum(static_programs.values()),
                "source_native_net_and_static_tls_preserved_byte_for_byte": True,
            },
        }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY,
            "commit": COMMIT,
            "variant": "DUE static",
            "networks": {SCENARIO: network_row},
        }
        atomic_write_json(staging / "conversion_manifest.json", payload)
        shutil.copy2(
            acquisition_root / "acquisition_manifest.json",
            staging / "source_acquisition_manifest.json",
        )
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
    args = parser.parse_args(argv)
    payload = package_lust(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
    )
    network = payload["networks"][SCENARIO]
    print(
        json.dumps(
            {
                "status": "PASS",
                "scenario": SCENARIO,
                "vehicle_count": network["vehicle_count"],
                "controlled_intersection_count": network[
                    "controlled_intersection_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
