#!/usr/bin/env python3
"""Package Dublin Urban A-F as native, full-day, read-only SUMO cases."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from scripts.data.acquire_dublin_urban import (
    COMMIT,
    FILES,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
    SCENARIOS,
    _git_blob_sha1,
)
from scripts.data.convert_libsignal_external_to_sumo import (
    _atomic_json,
    _validate_edge_routes,
    _write_xml,
)


PROTOCOL = "cfcmt-dublin-urban-native-sumo-package-v2"
NETWORK_BUILD_PROTOCOL = "dublin-urban-native-read-only-package-v2"
DEMAND_AUDIT_PROTOCOL = "dublin-urban-full-day-explicit-demand-v1"
TLS_AUDIT_PROTOCOL = "dublin-urban-native-sumo-tls-preserved-v1"
HORIZON_SEC = 86_400.0
SOURCE_ADDITIONAL_FILES = (
    "DCC_trafficlights.add.xml",
    "vtypes.add.xml",
    "DCC_routes.rou.xml",
    "DCC_emitters.emi.xml",
    "DCC_detectors.poi.xml",
)
PACKAGED_ADDITIONAL_FILES = (
    "../common/DCC_trafficlights.add.xml",
    "vtypes.add.xml",
    "../common/DCC_routes.rou.xml",
    "../common/DCC_emitters.emi.xml",
)
COMMON_FILES = (
    "DCC.net.xml",
    "DCC_routes.rou.xml",
    "DCC_emitters.emi.xml",
    "DCC_trafficlights.add.xml",
)


def _validate_acquisition(root: Path) -> dict[str, Any]:
    manifest = json.loads(
        (root / "acquisition_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("protocol") != ACQUISITION_PROTOCOL
        or manifest.get("commit") != COMMIT
        or set(dict(manifest.get("files", {}))) != set(FILES)
    ):
        raise ValueError("Dublin acquisition contract changed")
    for relative, (expected_size, expected_blob) in FILES.items():
        path = root / relative
        row = dict(manifest["files"][relative])
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or int(row.get("size_bytes", -1)) != expected_size
            or row.get("git_blob_sha1") != expected_blob
            or _git_blob_sha1(path) != expected_blob
        ):
            raise ValueError(f"Dublin acquisition identity changed: {relative}")
    return manifest


def _network_edges(path: Path) -> dict[str, tuple[str, str]]:
    root = ET.parse(path).getroot()
    return {
        str(edge.attrib["id"]): (
            str(edge.attrib["from"]),
            str(edge.attrib["to"]),
        )
        for edge in root.findall("edge")
        if "from" in edge.attrib and "to" in edge.attrib
    }


def _route_definitions(path: Path) -> dict[str, tuple[str, ...]]:
    definitions: dict[str, tuple[str, ...]] = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "route" and "id" in element.attrib:
            identity = str(element.attrib["id"])
            edges = tuple(str(element.attrib.get("edges", "")).split())
            if identity in definitions or not edges:
                raise ValueError(f"invalid Dublin route definition: {identity}")
            definitions[identity] = edges
        element.clear()
    if not definitions:
        raise ValueError("Dublin route file contains no route definitions")
    return definitions


def _demand_audit(
    path: Path, *, route_definitions: Mapping[str, tuple[str, ...]]
) -> dict[str, Any]:
    vehicle_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    vehicle_route_refs: set[str] = set()
    distribution_ids: set[str] = set()
    distribution_route_refs: set[str] = set()
    minimum_departure: float | None = None
    maximum_departure: float | None = None
    trip_count = 0
    flow_count = 0
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "routeDistribution":
            identity = str(element.attrib.get("id", ""))
            if not identity or identity in distribution_ids:
                raise ValueError(f"invalid Dublin route distribution: {identity}")
            distribution_ids.add(identity)
        elif element.tag == "route" and "refId" in element.attrib:
            distribution_route_refs.add(str(element.attrib["refId"]))
        elif element.tag == "vehicle":
            identity = str(element.attrib.get("id", ""))
            if not identity or identity in vehicle_ids:
                duplicate_ids.add(identity)
            vehicle_ids.add(identity)
            route = str(element.attrib.get("route", ""))
            if route:
                vehicle_route_refs.add(route)
            depart = float(element.attrib["depart"])
            minimum_departure = (
                depart
                if minimum_departure is None
                else min(minimum_departure, depart)
            )
            maximum_departure = (
                depart
                if maximum_departure is None
                else max(maximum_departure, depart)
            )
        elif element.tag == "trip":
            trip_count += 1
        elif element.tag == "flow":
            flow_count += 1
        element.clear()
    known_routes = set(route_definitions) | distribution_ids
    unknown_vehicle_routes = vehicle_route_refs - known_routes
    unknown_distribution_routes = distribution_route_refs - set(route_definitions)
    if (
        not vehicle_ids
        or duplicate_ids
        or trip_count
        or flow_count
        or unknown_vehicle_routes
        or unknown_distribution_routes
        or minimum_departure is None
        or minimum_departure < 0.0
        or maximum_departure is None
        or maximum_departure >= HORIZON_SEC
    ):
        raise ValueError(
            "Dublin full-day demand audit failed: "
            f"vehicles={len(vehicle_ids)} duplicates={len(duplicate_ids)} "
            f"trips={trip_count} flows={flow_count} "
            f"unknown_vehicle_routes={len(unknown_vehicle_routes)} "
            f"unknown_distribution_routes={len(unknown_distribution_routes)} "
            f"window=({minimum_departure},{maximum_departure})"
        )
    return {
        "protocol": DEMAND_AUDIT_PROTOCOL,
        "all_source_demand_files_loaded": True,
        "source_network_route_emitter_tls_files_preserved_byte_for_byte": True,
        "total": len(vehicle_ids),
        "explicit_vehicle_count": len(vehicle_ids),
        "trip_count": trip_count,
        "flow_count": flow_count,
        "duplicate_vehicle_id_count": len(duplicate_ids),
        "route_distribution_count": len(distribution_ids),
        "vehicle_route_reference_count": len(vehicle_route_refs),
        "unknown_vehicle_route_reference_count": len(unknown_vehicle_routes),
        "unknown_distribution_route_reference_count": len(
            unknown_distribution_routes
        ),
        "minimum_departure_sec": minimum_departure,
        "maximum_departure_sec": maximum_departure,
        "full_day_horizon_sec": HORIZON_SEC,
    }


def _tls_audit(net_path: Path, tls_path: Path) -> dict[str, Any]:
    programs: list[tuple[str, str, int]] = []
    for path in (net_path, tls_path):
        root = ET.parse(path).getroot()
        programs.extend(
            (
                str(logic.attrib["id"]),
                str(logic.attrib.get("programID", "0")),
                len(logic.findall("phase")),
            )
            for logic in root.iter("tlLogic")
        )
    tls_ids = {identity for identity, _, _ in programs}
    if not tls_ids or any(count <= 0 for _, _, count in programs):
        raise ValueError("Dublin native TLS inventory is invalid")
    return {
        "protocol": TLS_AUDIT_PROTOCOL,
        "controlled_intersection_count": len(tls_ids),
        "program_count": len(programs),
        "phase_count": sum(count for _, _, count in programs),
        "source_native_net_and_tls_preserved_byte_for_byte": True,
    }


def _vtype_audit(path: Path, *, scenario: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    probabilities = {
        str(element.attrib["id"]): float(element.attrib.get("probability", 0.0))
        for element in root.iter("vType")
    }
    distributions = {
        str(element.attrib["id"]): tuple(
            str(element.attrib.get("vTypes", "")).split()
        )
        for element in root.iter("vTypeDistribution")
    }
    if (
        set(probabilities) != {"CAT4", "CAT2", "HDT", "CAV4", "CAV2", "HDC"}
        or set(distributions) != {"PKW", "LKW"}
        or set(distributions["PKW"]) != {"HDC", "CAV2", "CAV4"}
        or set(distributions["LKW"]) != {"HDT", "CAT2", "CAT4"}
    ):
        raise ValueError(f"Dublin scenario {scenario} vType contract changed")
    return {
        "scenario": scenario,
        "vtype_probabilities": probabilities,
        "distributions": {
            key: list(value) for key, value in distributions.items()
        },
        "source_vtype_file_preserved_byte_for_byte": True,
    }


def _write_config(*, source_config: Path, destination: Path) -> None:
    source = ET.parse(source_config).getroot()
    source_input = source.find("input")
    if source_input is None:
        raise ValueError("Dublin source config contains no input block")
    source_net = source_input.find("net-file")
    source_routes = source_input.find("route-files")
    source_additional = source_input.find("additional-files")
    source_additional_values = tuple(
        value.strip()
        for value in str(
            (source_additional.attrib if source_additional is not None else {}).get(
                "value", ""
            )
        ).split(",")
        if value.strip()
    )
    if (
        source_net is None
        or source_net.attrib.get("value") != "DCC.net.xml"
        or source_routes is not None
        or source_additional_values != SOURCE_ADDITIONAL_FILES
    ):
        raise ValueError(
            "Dublin source input ordering changed: "
            f"net={None if source_net is None else source_net.attrib.get('value')} "
            f"route_files={source_routes is not None} "
            f"additional={source_additional_values}"
        )
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value="../common/DCC.net.xml")
    ET.SubElement(
        inputs,
        "additional-files",
        value=",".join(PACKAGED_ADDITIONAL_FILES),
    )
    for tag in ("time", "output"):
        element = source.find(tag)
        if element is not None:
            root.append(copy.deepcopy(element))
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(destination, root)


def package_dublin(*, acquisition_root: Path, output_root: Path) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root)
    acquisition = _validate_acquisition(acquisition_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Dublin package: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        base = acquisition_root / "Urban/Simulations/Base"
        common = staging / "common"
        common.mkdir()
        for filename in COMMON_FILES:
            shutil.copy2(base / filename, common / filename)
        shutil.copy2(
            acquisition_root / "acquisition_manifest.json",
            staging / "source_acquisition_manifest.json",
        )

        route_definitions = _route_definitions(common / "DCC_routes.rou.xml")
        route_audit = _validate_edge_routes(
            _network_edges(common / "DCC.net.xml"),
            route_definitions.values(),
        )
        demand_audit = _demand_audit(
            common / "DCC_emitters.emi.xml",
            route_definitions=route_definitions,
        )
        tls_audit = _tls_audit(
            common / "DCC.net.xml",
            common / "DCC_trafficlights.add.xml",
        )
        networks = {}
        for source_scenario in SCENARIOS:
            scenario = f"dublin_urban_{source_scenario.lower()}"
            directory = staging / scenario
            directory.mkdir()
            vtype_source = (
                acquisition_root
                / f"Urban/Simulations/Scenario {source_scenario}/vtypes.add.xml"
            )
            shutil.copy2(vtype_source, directory / "vtypes.add.xml")
            config = directory / f"{scenario}.sumocfg"
            _write_config(
                source_config=base / "DCC_simulation.sumo.cfg",
                destination=config,
            )
            scenario_demand = {
                **demand_audit,
                "source_scenario": source_scenario,
                "vehicle_type_audit": _vtype_audit(
                    directory / "vtypes.add.xml", scenario=source_scenario
                ),
            }
            networks[scenario] = {
                "sumocfg": str(config.relative_to(staging)),
                "vehicle_count": int(demand_audit["total"]),
                "person_count": 0,
                "controlled_intersection_count": int(
                    tls_audit["controlled_intersection_count"]
                ),
                "route_audit": route_audit,
                "demand_audit": scenario_demand,
                "tls_semantic_audit": tls_audit,
                "network_build": {
                    "protocol": NETWORK_BUILD_PROTOCOL,
                    "mode": "source_native_sumo_full_day_read_only_inputs",
                    "source_network_route_emitter_tls_files_preserved_byte_for_byte": True,
                    "source_config_input_order_preserved_except_passive_detector": True,
                    "packaged_additional_file_order": list(
                        PACKAGED_ADDITIONAL_FILES
                    ),
                    "all_six_source_vehicle_composition_scenarios_packaged": True,
                    "passive_detector_and_poi_definition_removed": True,
                    "traffic_semantics_rebuilt_or_calibrated": False,
                    "source_step_length_sec": 0.5,
                    "source_horizon_sec": HORIZON_SEC,
                },
            }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY,
            "commit": COMMIT,
            "city": "Dublin",
            "case_study": "Urban",
            "source_scenarios": list(SCENARIOS),
            "source_acquisition": {
                "protocol": acquisition["protocol"],
                "file_count": acquisition["file_count"],
                "total_size_bytes": acquisition["total_size_bytes"],
            },
            "networks": networks,
        }
        _atomic_json(staging / "conversion_manifest.json", payload)
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
    payload = package_dublin(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scenario_count": len(payload["networks"]),
                "scenarios": sorted(payload["networks"]),
                "vehicle_count": next(
                    iter(payload["networks"].values())
                )["vehicle_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
