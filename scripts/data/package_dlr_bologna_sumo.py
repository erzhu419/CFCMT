#!/usr/bin/env python3
"""Package the complete frozen DLR Bologna scenarios for read-only libsumo."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from cf_h2o.eval.traffic_signal_external_network_admission import (
    _explicit_vehicle_inventory,
)
from scripts.data.acquire_dlr_bologna import (
    COMMIT,
    FILES,
    GIT_BLOBS,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
    SCENARIOS,
    _git_blob_sha1,
)
from scripts.data.convert_libsignal_external_to_sumo import (
    _atomic_json,
    _manifest_path,
    _validate_edge_routes,
    _write_xml,
)


PROTOCOL = "cfcmt-dlr-bologna-native-sumo-package-v1"
NETWORK_BUILD_PROTOCOL = "dlr-bologna-native-read-only-package-v1"
TLS_AUDIT_PROTOCOL = "dlr-bologna-native-sumo-tls-preserved-v1"
HORIZON_SEC = 3600.0


def _csv_values(root: ET.Element, tag: str) -> list[str]:
    values: list[str] = []
    for element in root.findall(f".//{tag}"):
        values.extend(
            value.strip()
            for value in str(element.attrib.get("value", "")).split(",")
            if value.strip()
        )
    return values


def _is_passive_detector_file(path: Path) -> bool:
    root = ET.parse(path).getroot()
    active = {
        "busStop",
        "chargingStation",
        "containerStop",
        "rerouter",
        "routeProbe",
        "taz",
        "tlLogic",
        "trainStop",
        "vType",
    }
    return not any(element.tag in active for element in root)


def canonical_config_inputs(
    *, source_directory: Path, source_config: Path
) -> dict[str, list[str]]:
    root = ET.parse(source_config).getroot()
    net_files = _csv_values(root, "net-file")
    route_files = _csv_values(root, "route-files")
    additional_files = _csv_values(root, "additional-files")
    if len(net_files) != 1:
        raise ValueError(f"Bologna scenario must use one network: {source_config}")
    retained_additional: list[str] = []
    dropped_passive: list[str] = []
    for filename in additional_files:
        path = source_directory / filename
        if filename.endswith(".rou.xml"):
            route_files.append(filename)
        elif _is_passive_detector_file(path):
            dropped_passive.append(filename)
        else:
            retained_additional.append(filename)
    return {
        "net_files": net_files,
        "route_files": list(dict.fromkeys(route_files)),
        "additional_files": list(dict.fromkeys(retained_additional)),
        "dropped_passive_output_files": dropped_passive,
    }


def _write_canonical_config(
    *, source_config: Path, destination: Path, inputs: dict[str, list[str]]
) -> None:
    source_root = ET.parse(source_config).getroot()
    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", value=",".join(inputs["net_files"]))
    ET.SubElement(
        input_node,
        "route-files",
        value=",".join(inputs["route_files"]),
    )
    if inputs["additional_files"]:
        ET.SubElement(
            input_node,
            "additional-files",
            value=",".join(inputs["additional_files"]),
        )
    for processing in source_root.findall("processing"):
        root.append(copy.deepcopy(processing))
    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value=f"{HORIZON_SEC:g}")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(destination, root)


def _network_edges(net_path: Path) -> dict[str, tuple[str, str]]:
    root = ET.parse(net_path).getroot()
    return {
        str(edge.attrib["id"]): (
            str(edge.attrib["from"]),
            str(edge.attrib["to"]),
        )
        for edge in root.findall("edge")
        if "from" in edge.attrib and "to" in edge.attrib
    }


def _route_audit(
    *, directory: Path, route_files: Sequence[str], net_path: Path
) -> dict[str, Any]:
    routes: list[tuple[str, ...]] = []
    road_ids: set[str] = set()
    person_ids: set[str] = set()
    duplicate_road_ids: set[str] = set()
    duplicate_person_ids: set[str] = set()
    for filename in route_files:
        root = ET.parse(directory / filename).getroot()
        routes.extend(
            tuple(str(route.attrib.get("edges", "")).split())
            for route in root.iter("route")
            if route.attrib.get("edges")
        )
        for tag in ("vehicle", "trip", "flow"):
            for element in root.iter(tag):
                identity = str(element.attrib.get("id", ""))
                if identity in road_ids:
                    duplicate_road_ids.add(identity)
                road_ids.add(identity)
        for tag in ("person", "personFlow"):
            for element in root.iter(tag):
                identity = str(element.attrib.get("id", ""))
                if identity in person_ids:
                    duplicate_person_ids.add(identity)
                person_ids.add(identity)
    if duplicate_road_ids or duplicate_person_ids:
        raise ValueError(
            "Bologna demand contains duplicate IDs: "
            f"road={sorted(duplicate_road_ids)[:5]} "
            f"person={sorted(duplicate_person_ids)[:5]}"
        )
    audit = _validate_edge_routes(_network_edges(net_path), routes)
    audit.update(
        {
            "road_demand_id_count": len(road_ids),
            "person_demand_id_count": len(person_ids),
            "duplicate_road_demand_id_count": 0,
            "duplicate_person_demand_id_count": 0,
        }
    )
    return audit


def _tls_audit(
    *, directory: Path, net_file: str, additional_files: Sequence[str]
) -> dict[str, Any]:
    source_files = [net_file, *additional_files]
    programs: list[tuple[str, str, int]] = []
    for filename in source_files:
        root = ET.parse(directory / filename).getroot()
        programs.extend(
            (
                str(logic.attrib["id"]),
                str(logic.attrib.get("programID", "0")),
                len(logic.findall("phase")),
            )
            for logic in root.iter("tlLogic")
        )
    tls_ids = {identity for identity, _, _ in programs}
    if not tls_ids or any(phase_count <= 0 for _, _, phase_count in programs):
        raise ValueError(f"Bologna scenario has invalid TLS programs: {directory}")
    return {
        "protocol": TLS_AUDIT_PROTOCOL,
        "controlled_intersection_count": len(tls_ids),
        "program_count": len(programs),
        "phase_count": sum(phase_count for _, _, phase_count in programs),
        "source_net_and_tls_files_preserved_byte_for_byte": True,
    }


def _validate_acquisition(acquisition_root: Path) -> dict[str, Any]:
    manifest_path = acquisition_root / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != ACQUISITION_PROTOCOL:
        raise ValueError("DLR Bologna acquisition protocol changed")
    if manifest.get("commit") != COMMIT:
        raise ValueError("DLR Bologna acquisition commit changed")
    if set(dict(manifest.get("files", {}))) != set(FILES):
        raise ValueError("DLR Bologna acquisition file inventory changed")
    for relative, expected_size in FILES.items():
        path = acquisition_root / relative
        row = dict(manifest["files"][relative])
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or int(row.get("size_bytes", -1)) != expected_size
            or str(row.get("git_blob_sha1")) != GIT_BLOBS[relative]
            or _git_blob_sha1(path) != GIT_BLOBS[relative]
        ):
            raise ValueError(f"DLR Bologna acquisition identity changed: {relative}")
    return manifest


def package_bologna(*, acquisition_root: Path, output_root: Path) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root)
    acquisition = _validate_acquisition(acquisition_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Bologna package: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(
            acquisition_root / "acquisition_manifest.json",
            staging / "source_acquisition_manifest.json",
        )
        networks: dict[str, Any] = {}
        for source_scenario in SCENARIOS:
            scenario = f"bologna_{source_scenario}"
            source_directory = acquisition_root / "bologna" / source_scenario
            destination = staging / scenario
            shutil.copytree(source_directory, destination)
            source_config = source_directory / "run.sumocfg"
            inputs = canonical_config_inputs(
                source_directory=source_directory,
                source_config=source_config,
            )
            config_path = destination / f"{scenario}.sumocfg"
            _write_canonical_config(
                source_config=source_config,
                destination=config_path,
                inputs=inputs,
            )
            net_path = destination / inputs["net_files"][0]
            route_audit = _route_audit(
                directory=destination,
                route_files=inputs["route_files"],
                net_path=net_path,
            )
            inventory = _explicit_vehicle_inventory(
                config_path,
                horizon_sec=HORIZON_SEC,
            )
            tls_audit = _tls_audit(
                directory=destination,
                net_file=inputs["net_files"][0],
                additional_files=inputs["additional_files"],
            )
            networks[scenario] = {
                "sumocfg": _manifest_path(config_path, artifact_root=staging),
                "source_scenario": source_scenario,
                "source_config": f"bologna/{source_scenario}/run.sumocfg",
                "net_file": inputs["net_files"][0],
                "route_files": inputs["route_files"],
                "additional_files": inputs["additional_files"],
                "dropped_passive_output_files": inputs[
                    "dropped_passive_output_files"
                ],
                "horizon_sec": HORIZON_SEC,
                "vehicle_count": int(inventory["total"]),
                "person_count": int(inventory.get("person_total", 0)),
                "controlled_intersection_count": int(
                    tls_audit["controlled_intersection_count"]
                ),
                "route_audit": route_audit,
                "demand_audit": {
                    **inventory,
                    "protocol": "dlr-bologna-deterministic-demand-inventory-v1",
                    "all_source_demand_files_loaded": True,
                },
                "network_build": {
                    "protocol": NETWORK_BUILD_PROTOCOL,
                    "mode": "source_native_sumo_read_only_inputs",
                    "source_network_and_demand_files_preserved_byte_for_byte": True,
                    "passive_output_definitions_removed": True,
                    "traffic_semantics_rebuilt_or_calibrated": False,
                },
                "tls_semantic_audit": tls_audit,
            }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repository": REPOSITORY,
                "commit": COMMIT,
                "acquisition_protocol": ACQUISITION_PROTOCOL,
                "acquisition_file_count": int(acquisition["file_count"]),
                "coverage": "all four Bologna repository subscenarios",
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
    payload = package_bologna(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": payload["protocol"],
                "scenarios": sorted(payload["networks"]),
                "vehicle_counts": {
                    scenario: row["vehicle_count"]
                    for scenario, row in payload["networks"].items()
                },
                "person_counts": {
                    scenario: row["person_count"]
                    for scenario, row in payload["networks"].items()
                },
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
