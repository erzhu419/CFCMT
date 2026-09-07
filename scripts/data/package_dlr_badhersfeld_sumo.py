#!/usr/bin/env python3
"""Package frozen full-day DLR Bad Hersfeld scenarios for libsumo."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_network_admission import (
    _explicit_vehicle_inventory,
)
from cf_h2o.traffic_signal.sumo_static_inputs import (
    DEMAND_ROOT_TAGS,
    demand_route_mixture,
    iterparse_xml,
    iter_static_demand_records,
    load_static_route_catalog,
    parse_xml,
    static_demand_input_paths,
)
from scripts.data.acquire_dlr_badhersfeld import (
    COMMIT,
    FILES,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
    SCENARIOS,
    _git_blob_sha1,
)
from scripts.data.convert_libsignal_external_to_sumo import (
    _atomic_json,
    _manifest_path,
    _write_xml,
)


PROTOCOL = "cfcmt-dlr-badhersfeld-native-sumo-package-v1"
NETWORK_BUILD_PROTOCOL = "dlr-badhersfeld-native-read-only-package-v1"
TLS_AUDIT_PROTOCOL = "dlr-badhersfeld-native-sumo-tls-preserved-v1"
DEMAND_AUDIT_PROTOCOL = "dlr-badhersfeld-complete-rich-demand-inventory-v1"
HORIZON_SEC = 86_400.0
SOURCE_ADDITIONAL_FILES = (
    "../osm/pt_vtypes.xml",
    "../osm/gtfs_publictransport.add.xml",
    "../osm/gtfs_publictransport.rou.xml",
    "../osm/osm_polygons.add.xml",
    "../osm/defaults/basic.vType.xml",
    "../osm/osm_complete_parking_areas.add.xml",
    "../osm/osm_parking_rerouters.add.xml",
    "../osm/obstacle_seilerweg.add.xml",
    "../demand/osm_activitygen.lkw.rou.xml",
    "../osm/calib.add.xml",
)
PASSIVE_ADDITIONAL_FILE = "../osm/osm_polygons.add.xml"
SOURCE_ROUTE_FILES = {
    "present": "../demand/osm_activitygen_present_no_prt.rou.xml.gz",
    "future_no_prt": "../demand/osm_activitygen_future_no_prt.rou.xml.gz",
}
PASSIVE_OUTPUT_TAGS = frozenset(
    {
        "device.taxi.dispatch-algorithm.output",
        "device.taxi.idle-algorithm.output",
    }
)


def _csv_values(root: ET.Element, tag: str) -> list[str]:
    values: list[str] = []
    for element in root.findall(f".//{tag}"):
        values.extend(
            value.strip()
            for value in str(element.attrib.get("value", "")).split(",")
            if value.strip()
        )
    return values


def canonical_config_inputs(
    *, source_config: Path, scenario: str
) -> dict[str, list[str]]:
    root = parse_xml(source_config).getroot()
    net_files = _csv_values(root, "net-file")
    route_files = _csv_values(root, "route-files")
    additional_files = _csv_values(root, "additional-files")
    if net_files != ["../osm/osm_edited.net.xml.gz"]:
        raise ValueError(f"Bad Hersfeld network input changed: {net_files}")
    if route_files != [SOURCE_ROUTE_FILES[scenario]]:
        raise ValueError(
            f"Bad Hersfeld {scenario} demand input changed: {route_files}"
        )
    if tuple(additional_files) != SOURCE_ADDITIONAL_FILES:
        raise ValueError(
            f"Bad Hersfeld {scenario} additional input order changed"
        )
    retained = [
        value for value in additional_files if value != PASSIVE_ADDITIONAL_FILE
    ]
    return {
        "net_files": net_files,
        "route_files": route_files,
        "additional_files": retained,
        "dropped_passive_output_files": [PASSIVE_ADDITIONAL_FILE],
    }


def _remove_passive_outputs(root: ET.Element) -> list[str]:
    removed: list[str] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "gui_only" or child.tag in PASSIVE_OUTPUT_TAGS:
                removed.append(str(child.tag))
                parent.remove(child)
    return removed


def _write_canonical_config(
    *,
    source_config: Path,
    destination: Path,
    inputs: dict[str, list[str]],
) -> list[str]:
    root = copy.deepcopy(parse_xml(source_config).getroot())
    additional_nodes = root.findall(".//additional-files")
    if len(additional_nodes) != 1:
        raise ValueError("Bad Hersfeld config must have one additional-files node")
    additional_nodes[0].set("value", ",".join(inputs["additional_files"]))
    removed = _remove_passive_outputs(root)
    expected_removed = {
        "gui_only",
        "device.taxi.dispatch-algorithm.output",
        "device.taxi.idle-algorithm.output",
    }
    if set(removed) != expected_removed or len(removed) != len(expected_removed):
        raise ValueError(f"Bad Hersfeld passive output layout changed: {removed}")
    _write_xml(destination, root)
    return removed


def _dynamic_routing_audit(config_path: Path) -> dict[str, Any]:
    root = parse_xml(config_path).getroot()

    def value(tag: str) -> str | None:
        element = root.find(f".//{tag}")
        return None if element is None else element.attrib.get("value")

    audit = {
        "ignore_route_errors": value("ignore-route-errors"),
        "rerouting_probability": value("device.rerouting.probability"),
        "rerouting_period_sec": value("device.rerouting.period"),
        "rerouting_pre_period_sec": value("device.rerouting.pre-period"),
    }
    if audit != {
        "ignore_route_errors": "true",
        "rerouting_probability": "1",
        "rerouting_period_sec": "300",
        "rerouting_pre_period_sec": "300",
    }:
        raise ValueError(f"Bad Hersfeld dynamic rerouting changed: {audit}")
    return audit


def _network_edges(net_path: Path) -> dict[str, tuple[str, str]]:
    endpoints: dict[str, tuple[str, str]] = {}
    for _, element in iterparse_xml(net_path, events=("end",)):
        if element.tag == "edge" and {
            "id",
            "from",
            "to",
        }.issubset(element.attrib):
            endpoints[str(element.attrib["id"])] = (
                str(element.attrib["from"]),
                str(element.attrib["to"]),
            )
        element.clear()
    if not endpoints:
        raise ValueError(f"Bad Hersfeld network has no road edges: {net_path}")
    return endpoints


def _route_audit(sumocfg: Path) -> dict[str, Any]:
    demand_paths = static_demand_input_paths(sumocfg)
    catalog = load_static_route_catalog(demand_paths)
    config_root = parse_xml(sumocfg).getroot()
    net_files = _csv_values(config_root, "net-file")
    if len(net_files) != 1:
        raise ValueError("Bad Hersfeld package must use one network")
    endpoints = _network_edges((sumocfg.parent / net_files[0]).resolve())
    route_count = 0
    missing_edge_route_count = 0
    disconnected_route_count = 0
    minimum_route_edges: int | None = None
    maximum_route_edges = 0
    dynamic_skeleton_ids: set[str] = set()
    unclassified_disconnected: list[str] = []

    def inspect(
        edges: tuple[str, ...], *, identity: str, source_name: str
    ) -> None:
        nonlocal route_count, missing_edge_route_count
        nonlocal disconnected_route_count, minimum_route_edges
        nonlocal maximum_route_edges
        route_count += 1
        minimum_route_edges = (
            len(edges)
            if minimum_route_edges is None
            else min(minimum_route_edges, len(edges))
        )
        maximum_route_edges = max(maximum_route_edges, len(edges))
        if any(edge not in endpoints for edge in edges):
            missing_edge_route_count += 1
            return
        disconnected = any(
            endpoints[first][1] != endpoints[second][0]
            for first, second in zip(edges, edges[1:])
        )
        if not disconnected:
            return
        disconnected_route_count += 1
        is_published_lkw_skeleton = (
            source_name == "osm_activitygen.lkw.rou.xml"
            and identity.startswith("lkw_")
            and len(edges) == 3
            and edges[1] == "-902659463"
            and edges[2] == f"-{edges[0]}"
        )
        if is_published_lkw_skeleton:
            dynamic_skeleton_ids.add(identity)
        else:
            unclassified_disconnected.append(f"{source_name}:{identity}")

    for route_id, edges in catalog.routes.items():
        inspect(edges, identity=route_id, source_name="named_route_catalog")
    demand_record_count = 0
    for path in demand_paths:
        for record in iter_static_demand_records([path]):
            demand_record_count += 1
            for index, (edges, _) in enumerate(
                demand_route_mixture(record, catalog)
            ):
                inspect(
                    edges,
                    identity=(
                        f"{record.attributes.get('id', '')}:mixture{index}"
                        if index
                        else str(record.attributes.get("id", ""))
                    ),
                    source_name=path.name,
                )
    expected_lkw_ids = {f"lkw_{index}" for index in range(1, 30)}
    if (
        route_count <= 0
        or missing_edge_route_count
        or unclassified_disconnected
        or dynamic_skeleton_ids != expected_lkw_ids
        or disconnected_route_count != len(expected_lkw_ids)
    ):
        raise ValueError(
            "Bad Hersfeld route topology validation failed: "
            f"routes={route_count} missing={missing_edge_route_count} "
            f"disconnected={disconnected_route_count} "
            f"dynamic_skeletons={len(dynamic_skeleton_ids)} "
            f"unclassified={unclassified_disconnected[:5]}"
        )
    return {
        "route_count": route_count,
        "missing_edge_route_count": missing_edge_route_count,
        "disconnected_route_count": disconnected_route_count,
        "minimum_route_edges": int(minimum_route_edges or 0),
        "maximum_route_edges": maximum_route_edges,
        "global_named_route_count": len(catalog.routes),
        "route_distribution_count": len(catalog.distributions),
        "road_demand_record_count": demand_record_count,
        "source_declared_dynamic_reroute_skeleton_count": len(
            dynamic_skeleton_ids
        ),
        "unclassified_disconnected_route_count": 0,
        "dynamic_reroute_skeleton_ids": sorted(dynamic_skeleton_ids),
        "all_configured_route_and_additional_files_scanned": True,
    }


def _demand_id_audit(sumocfg: Path) -> dict[str, int]:
    road_ids: set[str] = set()
    person_ids: set[str] = set()
    duplicate_road_ids: set[str] = set()
    duplicate_person_ids: set[str] = set()
    calibrator_flow_count = 0
    for path in static_demand_input_paths(sumocfg):
        ancestors: list[str] = []
        for event, element in iterparse_xml(path, events=("start", "end")):
            if event == "start":
                ancestors.append(str(element.tag))
                continue
            parent = ancestors[-2] if len(ancestors) >= 2 else None
            if element.tag in {"vehicle", "trip", "flow"}:
                if parent in DEMAND_ROOT_TAGS:
                    identity = str(element.attrib.get("id", ""))
                    if identity:
                        if identity in road_ids:
                            duplicate_road_ids.add(identity)
                        road_ids.add(identity)
                elif element.tag == "flow" and parent == "calibrator":
                    calibrator_flow_count += 1
            elif element.tag in {"person", "personFlow"} and parent in DEMAND_ROOT_TAGS:
                identity = str(element.attrib.get("id", ""))
                if identity:
                    if identity in person_ids:
                        duplicate_person_ids.add(identity)
                    person_ids.add(identity)
            element.clear()
            if not ancestors or ancestors[-1] != element.tag:
                raise ValueError(f"invalid SUMO XML nesting in {path}")
            ancestors.pop()
    if duplicate_road_ids or duplicate_person_ids:
        raise ValueError(
            "Bad Hersfeld demand contains duplicate IDs: "
            f"road={sorted(duplicate_road_ids)[:5]} "
            f"person={sorted(duplicate_person_ids)[:5]}"
        )
    return {
        "road_demand_id_count": len(road_ids),
        "person_demand_id_count": len(person_ids),
        "duplicate_road_demand_id_count": 0,
        "duplicate_person_demand_id_count": 0,
        "nested_calibrator_flow_count": calibrator_flow_count,
    }


def _tls_audit(net_path: Path) -> dict[str, Any]:
    root = parse_xml(net_path).getroot()
    programs = [
        (
            str(logic.attrib["id"]),
            str(logic.attrib.get("programID", "0")),
            len(logic.findall("phase")),
        )
        for logic in root.iter("tlLogic")
    ]
    tls_ids = {identity for identity, _, _ in programs}
    if not tls_ids or any(count <= 0 for _, _, count in programs):
        raise ValueError("Bad Hersfeld network has invalid TLS programs")
    return {
        "protocol": TLS_AUDIT_PROTOCOL,
        "controlled_intersection_count": len(tls_ids),
        "program_count": len(programs),
        "phase_count": sum(count for _, _, count in programs),
        "source_native_net_and_tls_preserved_byte_for_byte": True,
    }


def _validate_acquisition(acquisition_root: Path) -> dict[str, Any]:
    manifest_path = acquisition_root / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != ACQUISITION_PROTOCOL:
        raise ValueError("DLR Bad Hersfeld acquisition protocol changed")
    if manifest.get("commit") != COMMIT:
        raise ValueError("DLR Bad Hersfeld acquisition commit changed")
    if set(dict(manifest.get("files", {}))) != set(FILES):
        raise ValueError("DLR Bad Hersfeld acquisition file inventory changed")
    for relative, (expected_size, expected_blob) in FILES.items():
        path = acquisition_root / relative
        row = dict(manifest["files"][relative])
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or int(row.get("size_bytes", -1)) != expected_size
            or str(row.get("git_blob_sha1")) != expected_blob
            or _git_blob_sha1(path) != expected_blob
        ):
            raise ValueError(
                f"DLR Bad Hersfeld acquisition identity changed: {relative}"
            )
    return manifest


def package_badhersfeld(
    *, acquisition_root: Path, output_root: Path
) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root)
    acquisition = _validate_acquisition(acquisition_root)
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite Bad Hersfeld package: {output_root}"
        )
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(
            acquisition_root / "acquisition_manifest.json",
            staging / "source_acquisition_manifest.json",
        )
        destination_city = staging / "BadHersfeld"
        shutil.copytree(acquisition_root / "BadHersfeld", destination_city)
        networks: dict[str, Any] = {}
        for source_scenario in SCENARIOS:
            scenario = f"badhersfeld_{source_scenario}"
            evaluation = destination_city / "evaluation"
            source_config = evaluation / f"{source_scenario}.sumocfg"
            inputs = canonical_config_inputs(
                source_config=source_config,
                scenario=source_scenario,
            )
            config_path = evaluation / f"cfcmt_{source_scenario}.sumocfg"
            removed_outputs = _write_canonical_config(
                source_config=source_config,
                destination=config_path,
                inputs=inputs,
            )
            inventory = _explicit_vehicle_inventory(
                config_path,
                horizon_sec=HORIZON_SEC,
            )
            route_audit = _route_audit(config_path)
            dynamic_routing_audit = _dynamic_routing_audit(config_path)
            id_audit = _demand_id_audit(config_path)
            net_path = (evaluation / inputs["net_files"][0]).resolve()
            tls_audit = _tls_audit(net_path)
            if int(id_audit["road_demand_id_count"]) != int(
                inventory["explicit_vehicle_count"]
            ):
                raise ValueError(
                    f"Bad Hersfeld road demand ID count changed: {scenario}"
                )
            if int(id_audit["person_demand_id_count"]) != int(
                inventory["person_total"]
            ):
                raise ValueError(
                    f"Bad Hersfeld person demand ID count changed: {scenario}"
                )
            networks[scenario] = {
                "sumocfg": _manifest_path(config_path, artifact_root=staging),
                "source_scenario": source_scenario,
                "source_config": (
                    f"BadHersfeld/evaluation/{source_scenario}.sumocfg"
                ),
                "net_file": inputs["net_files"][0],
                "route_files": inputs["route_files"],
                "additional_files": inputs["additional_files"],
                "dropped_passive_output_files": inputs[
                    "dropped_passive_output_files"
                ],
                "removed_passive_config_tags": removed_outputs,
                "horizon_sec": HORIZON_SEC,
                "vehicle_count": int(inventory["total"]),
                "person_count": int(inventory["person_total"]),
                "controlled_intersection_count": int(
                    tls_audit["controlled_intersection_count"]
                ),
                "route_audit": route_audit,
                "dynamic_routing_audit": dynamic_routing_audit,
                "demand_audit": {
                    **inventory,
                    **id_audit,
                    "protocol": DEMAND_AUDIT_PROTOCOL,
                    "all_source_demand_files_loaded": True,
                    "all_triggered_vehicles_resolved_to_person_plans": True,
                    "published_calibrator_input_preserved": True,
                    "source_network_and_demand_files_preserved_byte_for_byte": True,
                },
                "network_build": {
                    "protocol": NETWORK_BUILD_PROTOCOL,
                    "mode": "source_native_sumo_full_day_read_only_inputs",
                    "source_network_and_demand_files_preserved_byte_for_byte": True,
                    "complete_source_directory_preserved": True,
                    "source_input_order_preserved_except_passive_polygon": True,
                    "passive_polygon_gui_and_output_destinations_removed": True,
                    "published_calibrator_input_preserved": True,
                    "source_dynamic_route_repair_preserved": True,
                    "traffic_semantics_rebuilt_or_calibrated": False,
                    "source_horizon_sec": HORIZON_SEC,
                    "source_step_length_sec": 1.0,
                },
                "tls_semantic_audit": tls_audit,
            }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repository": REPOSITORY,
                "commit": COMMIT,
                "license": "EPL-2.0",
                "acquisition_protocol": ACQUISITION_PROTOCOL,
                "acquisition_file_count": int(acquisition["file_count"]),
                "acquisition_total_size_bytes": int(
                    acquisition["total_size_bytes"]
                ),
                "coverage": "complete tracked BadHersfeld directory",
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
    payload = package_badhersfeld(
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
                    key: int(value["vehicle_count"])
                    for key, value in payload["networks"].items()
                },
                "person_counts": {
                    key: int(value["person_count"])
                    for key, value in payload["networks"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
