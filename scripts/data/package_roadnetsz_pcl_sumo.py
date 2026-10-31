#!/usr/bin/env python3
"""Package frozen RoadnetSZ PCL windows as strict explicit SUMO routes."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from scripts.data.acquire_roadnetsz_pcl import (
    COMMIT,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
)
from scripts.data.convert_libsignal_external_to_sumo import (
    _atomic_json,
    _manifest_path,
    _sha256,
    _validate_edge_routes,
    _write_xml,
)


PROTOCOL = "cfcmt-roadnetsz-native-sumo122-window-package-v4"
NETWORK_BUILD_PROTOCOL = (
    "roadnetsz-source-components-sumo122-protected-incoming-window-duarouter-v4"
)
TLS_AUDIT_PROTOCOL = "roadnetsz-source-tll-sumo122-rebuild-v2"
NET_RELATIVE = "data_sumo/pcl.net.xml"
NODE_RELATIVE = "data_sumo/pcl.nod.xml"
EDGE_RELATIVE = "data_sumo/pcl.edg.xml"
CONNECTION_RELATIVE = "data_sumo/pcl.con.xml"
TLS_RELATIVE = "data_sumo/pcl.tll.xml"
TYPE_RELATIVE = "data_sumo/pcl.typ.xml"
TRIPS_RELATIVE = "data_sumo/pcl.trips.xml"
WINDOWS = tuple(
    (f"shenzhen_pcl_hour_{index:02d}", index * 3600.0, (index + 1) * 3600.0)
    for index in range(28)
)
MINIMUM_ROUTING_RETENTION = 0.90
AUTO_GENERATED_TLS_PHASE_LAYOUT = "incoming"


def _validate_manifest_file(
    *, acquisition_root: Path, manifest: dict[str, Any], relative: str
) -> Path:
    row = dict(dict(manifest.get("files", {})).get(relative, {}))
    path = acquisition_root / relative
    if not row or not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row.get("size_bytes", -1)):
        raise ValueError(f"RoadnetSZ PCL file size changed: {relative}")
    if _sha256(path) != str(row.get("sha256", "")):
        raise ValueError(f"RoadnetSZ PCL file identity changed: {relative}")
    return path


def _run_duarouter(
    *, duarouter: Path, net_path: Path, trips_path: Path, route_path: Path
) -> dict[str, Any]:
    command = [
        str(duarouter),
        "--net-file",
        str(net_path),
        "--route-files",
        str(trips_path),
        "--output-file",
        str(route_path),
        "--ignore-errors",
        "true",
        "--no-warnings",
        "true",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not route_path.is_file():
        raise RuntimeError(
            "duarouter failed\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _run_netconvert(
    *,
    netconvert: Path,
    node_path: Path,
    edge_path: Path,
    connection_path: Path,
    tls_path: Path,
    type_path: Path,
    net_path: Path,
) -> dict[str, Any]:
    command = [
        str(netconvert),
        "--node-files",
        str(node_path),
        "--edge-files",
        str(edge_path),
        "--connection-files",
        str(connection_path),
        "--tllogic-files",
        str(tls_path),
        "--type-files",
        str(type_path),
        "--output-file",
        str(net_path),
        "--no-turnarounds",
        "true",
        "--junctions.corner-detail",
        "5",
        "--tls.layout",
        AUTO_GENERATED_TLS_PHASE_LAYOUT,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not net_path.is_file():
        raise RuntimeError(
            "netconvert failed\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _tls_phase_counts(root: ET.Element) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for logic in root.findall(".//tlLogic"):
        identity = (
            str(logic.attrib["id"]),
            str(logic.attrib.get("programID", "0")),
        )
        if identity in counts:
            raise ValueError(f"duplicate TLS program identity: {identity}")
        counts[identity] = len(logic.findall("phase"))
    return counts


def _write_window_trips(
    *, source_root: ET.Element, start: float, end: float, scenario: str, path: Path
) -> tuple[str, ...]:
    root = ET.Element("routes")
    ids = []
    for row in source_root:
        if row.tag != "trip":
            continue
        departure = float(row.attrib["depart"])
        if start <= departure < end:
            item = copy.deepcopy(row)
            item.attrib["id"] = f"{scenario}_{row.attrib['id']}"
            item.attrib["depart"] = f"{departure - start:g}"
            root.append(item)
            ids.append(item.attrib["id"])
    if not ids:
        raise ValueError(f"RoadnetSZ PCL window is empty: {scenario}")
    _write_xml(path, root)
    return tuple(ids)


def _route_inventory(
    *, route_path: Path, edge_endpoints: dict[str, tuple[str, str]]
) -> tuple[dict[str, Any], tuple[str, ...], tuple[float, ...]]:
    root = ET.parse(route_path).getroot()
    vehicles = [row for row in root if row.tag == "vehicle"]
    ids = tuple(str(row.attrib["id"]) for row in vehicles)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duarouter emitted duplicate vehicle IDs: {route_path}")
    departures = tuple(float(row.attrib["depart"]) for row in vehicles)
    routes = []
    for vehicle in vehicles:
        route = vehicle.find("route")
        if route is None:
            raise ValueError(f"duarouter vehicle has no explicit route: {route_path}")
        routes.append(tuple(str(route.attrib.get("edges", "")).split()))
    audit = _validate_edge_routes(edge_endpoints, routes)
    return audit, ids, departures


def _write_sumocfg(
    path: Path, *, net_relative: str, route_filename: str
) -> None:
    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", value=net_relative)
    ET.SubElement(input_node, "route-files", value=route_filename)
    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value="3600")
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")
    _write_xml(path, root)


def package_pcl(
    *,
    acquisition_root: Path,
    output_root: Path,
    netconvert: Path,
    duarouter: Path,
) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root)
    netconvert = Path(netconvert).resolve()
    duarouter = Path(duarouter).resolve()
    if not netconvert.is_file():
        raise FileNotFoundError(netconvert)
    if not duarouter.is_file():
        raise FileNotFoundError(duarouter)
    manifest_path = acquisition_root / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != ACQUISITION_PROTOCOL:
        raise ValueError("RoadnetSZ PCL acquisition protocol changed")
    if manifest.get("commit") != COMMIT:
        raise ValueError("RoadnetSZ PCL acquisition commit changed")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite PCL package: {output_root}")
    source_components = {
        relative: _validate_manifest_file(
            acquisition_root=acquisition_root,
            manifest=manifest,
            relative=relative,
        )
        for relative in (
            NET_RELATIVE,
            NODE_RELATIVE,
            EDGE_RELATIVE,
            CONNECTION_RELATIVE,
            TLS_RELATIVE,
            TYPE_RELATIVE,
        )
    }
    source_net = source_components[NET_RELATIVE]
    source_trips = _validate_manifest_file(
        acquisition_root=acquisition_root,
        manifest=manifest,
        relative=TRIPS_RELATIVE,
    )
    legacy_net_root = ET.parse(source_net).getroot()
    source_node_root = ET.parse(source_components[NODE_RELATIVE]).getroot()
    source_tll_root = ET.parse(source_components[TLS_RELATIVE]).getroot()
    declared_tls_node_ids = {
        str(row.attrib["id"])
        for row in source_node_root.findall(".//node")
        if str(row.attrib.get("type", "")).startswith("traffic_light")
    }
    explicit_tls_phase_counts = _tls_phase_counts(source_tll_root)
    explicit_tls_ids = {identity[0] for identity in explicit_tls_phase_counts}
    if not declared_tls_node_ids or not explicit_tls_phase_counts:
        raise ValueError("RoadnetSZ PCL source components lack TLS declarations")
    if not explicit_tls_ids.issubset(declared_tls_node_ids):
        raise ValueError("RoadnetSZ PCL explicit TLS programs escape TLS nodes")
    source_trip_root = ET.parse(source_trips).getroot()
    if any(
        float(end) - float(start) != 3600.0
        for _, start, end in WINDOWS
    ) or any(
        float(first[2]) != float(second[1])
        for first, second in zip(WINDOWS, WINDOWS[1:])
    ):
        raise ValueError("RoadnetSZ PCL windows must be contiguous one-hour blocks")
    source_trip_count = sum(row.tag == "trip" for row in source_trip_root)

    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        packaged_net = staging / "pcl.net.xml"
        net_builder = _run_netconvert(
            netconvert=netconvert,
            node_path=source_components[NODE_RELATIVE],
            edge_path=source_components[EDGE_RELATIVE],
            connection_path=source_components[CONNECTION_RELATIVE],
            tls_path=source_components[TLS_RELATIVE],
            type_path=source_components[TYPE_RELATIVE],
            net_path=packaged_net,
        )
        net_builder["command"] = [
            _manifest_path(Path(value), artifact_root=staging)
            if value == str(packaged_net)
            else value
            for value in net_builder["command"]
        ]
        net_root = ET.parse(packaged_net).getroot()
        edge_endpoints = {
            str(row.attrib["id"]): (str(row.attrib["from"]), str(row.attrib["to"]))
            for row in net_root.findall("edge")
            if "from" in row.attrib and "to" in row.attrib
        }
        rebuilt_tls_phase_counts = _tls_phase_counts(net_root)
        rebuilt_tls_ids = {identity[0] for identity in rebuilt_tls_phase_counts}
        missing_declared_tls_ids = sorted(declared_tls_node_ids - rebuilt_tls_ids)
        unexpected_rebuilt_tls_ids = sorted(rebuilt_tls_ids - declared_tls_node_ids)
        missing_explicit_programs = sorted(
            explicit_tls_phase_counts.keys() - rebuilt_tls_phase_counts.keys()
        )
        explicit_phase_count_mismatches = sorted(
            identity
            for identity, count in explicit_tls_phase_counts.items()
            if rebuilt_tls_phase_counts.get(identity) != count
        )
        if not edge_endpoints:
            raise ValueError("SUMO 1.22 rebuilt PCL network has no directed edges")
        if (
            missing_declared_tls_ids
            or unexpected_rebuilt_tls_ids
            or missing_explicit_programs
            or explicit_phase_count_mismatches
        ):
            raise ValueError(
                "SUMO 1.22 PCL TLS reconstruction audit failed: "
                f"missing_nodes={missing_declared_tls_ids}, "
                f"unexpected_tls={unexpected_rebuilt_tls_ids}, "
                f"missing_programs={missing_explicit_programs}, "
                f"phase_mismatches={explicit_phase_count_mismatches}"
            )
        legacy_tls_phase_counts = _tls_phase_counts(legacy_net_root)
        auto_generated_tls_ids = rebuilt_tls_ids - explicit_tls_ids
        auto_generated_minor_green_signal_count = sum(
            str(phase.attrib.get("state", "")).count("g")
            for logic in net_root.findall(".//tlLogic")
            if str(logic.attrib["id"]) in auto_generated_tls_ids
            for phase in logic.findall("phase")
        )
        if auto_generated_minor_green_signal_count != 0:
            raise ValueError(
                "protected incoming TLS layout retained permissive minor greens"
            )
        tls_semantic_audit = {
            "protocol": TLS_AUDIT_PROTOCOL,
            "declared_tls_node_count": len(declared_tls_node_ids),
            "explicit_source_tls_program_count": len(explicit_tls_phase_counts),
            "explicit_source_tls_id_count": len(explicit_tls_ids),
            "legacy_net_tls_program_count": len(legacy_tls_phase_counts),
            "rebuilt_tls_program_count": len(rebuilt_tls_phase_counts),
            "rebuilt_tls_id_count": len(rebuilt_tls_ids),
            "auto_generated_tls_id_count": len(rebuilt_tls_ids - explicit_tls_ids),
            "auto_generated_phase_layout": AUTO_GENERATED_TLS_PHASE_LAYOUT,
            "auto_generated_minor_green_signal_count": (
                auto_generated_minor_green_signal_count
            ),
            "explicit_source_phase_count": sum(explicit_tls_phase_counts.values()),
            "rebuilt_phase_count": sum(rebuilt_tls_phase_counts.values()),
            "missing_declared_tls_id_count": len(missing_declared_tls_ids),
            "unexpected_rebuilt_tls_id_count": len(unexpected_rebuilt_tls_ids),
            "missing_explicit_tls_program_count": len(missing_explicit_programs),
            "explicit_phase_count_mismatch_count": len(
                explicit_phase_count_mismatches
            ),
            "all_declared_tls_nodes_rebuilt": True,
            "explicit_source_tls_program_phase_counts_preserved": True,
            "source_net_sha256": _sha256(source_net),
            "rebuilt_net_sha256": _sha256(packaged_net),
        }
        networks: dict[str, dict[str, Any]] = {}
        total_windowed_trips = 0
        total_routed_vehicles = 0
        for scenario, start, end in WINDOWS:
            scenario_dir = staging / scenario
            scenario_dir.mkdir()
            window_trips = scenario_dir / f"{scenario}.trips.xml"
            input_ids = _write_window_trips(
                source_root=source_trip_root,
                start=start,
                end=end,
                scenario=scenario,
                path=window_trips,
            )
            total_windowed_trips += len(input_ids)
            route_path = scenario_dir / f"{scenario}.rou.xml"
            router = _run_duarouter(
                duarouter=duarouter,
                net_path=packaged_net,
                trips_path=window_trips,
                route_path=route_path,
            )
            router["command"] = [
                _manifest_path(Path(value), artifact_root=staging)
                if value in {str(packaged_net), str(window_trips), str(route_path)}
                else value
                for value in router["command"]
            ]
            route_audit, routed_ids, departures = _route_inventory(
                route_path=route_path,
                edge_endpoints=edge_endpoints,
            )
            routed_set = set(routed_ids)
            dropped_ids = [value for value in input_ids if value not in routed_set]
            retention = len(routed_ids) / len(input_ids)
            total_routed_vehicles += len(routed_ids)
            if retention + 1e-12 < MINIMUM_ROUTING_RETENTION:
                raise ValueError(
                    f"PCL routing retention below {MINIMUM_ROUTING_RETENTION:.0%}: "
                    f"{scenario}={retention:.6f}"
                )
            if not departures or min(departures) < 0 or max(departures) >= 3600:
                raise ValueError(f"PCL routed departures escape the window: {scenario}")
            cfg_path = scenario_dir / f"{scenario}.sumocfg"
            _write_sumocfg(
                cfg_path,
                net_relative="../pcl.net.xml",
                route_filename=route_path.name,
            )
            networks[scenario] = {
                "city": "shenzhen",
                "network": scenario,
                "source_format": "RoadnetSZ native SUMO",
                "controlled_intersection_count": len(rebuilt_tls_ids),
                "directed_edge_count": len(edge_endpoints),
                "vehicle_count": len(routed_ids),
                "route_audit": route_audit,
                "source_window": {
                    "start_sec": start,
                    "end_sec": end,
                    "input_trip_count": len(input_ids),
                    "routed_vehicle_count": len(routed_ids),
                    "dropped_trip_count": len(dropped_ids),
                    "routing_retention": retention,
                    "minimum_required_retention": MINIMUM_ROUTING_RETENTION,
                    "dropped_trip_ids": dropped_ids,
                },
                "network_build": {
                    "protocol": NETWORK_BUILD_PROTOCOL,
                    "mode": (
                        "source_components_rebuilt_with_sumo122_protected_"
                        "incoming_and_strict_explicit_routing"
                    ),
                    "auto_generated_tls_phase_layout": (
                        AUTO_GENERATED_TLS_PHASE_LAYOUT
                    ),
                    "netconvert": net_builder,
                    "duarouter": router,
                },
                "tls_semantic_audit": dict(tls_semantic_audit),
                "sumocfg": _manifest_path(cfg_path, artifact_root=staging),
            }

        if total_windowed_trips != source_trip_count:
            raise ValueError(
                "full PCL window package did not cover every source trip: "
                f"{total_windowed_trips} != {source_trip_count}"
            )

        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_repositories": {
                "shenzhen": {"url": REPOSITORY, "commit": COMMIT}
            },
            "source_files": {
                relative: {
                    "path": str((acquisition_root / relative).resolve()),
                    "size_bytes": int(row["size_bytes"]),
                    "sha256": str(row["sha256"]),
                }
                for relative, row in manifest["files"].items()
            },
            "selection": {
                "rule": (
                    "all 28 contiguous one-hour windows cover the complete source "
                    "trip file before network admission or controller evaluation"
                ),
                "routing_retention_gate": MINIMUM_ROUTING_RETENTION,
                "source_runtime_ignore_route_errors_removed": True,
                "legacy_compiled_net_replaced_from_source_components": True,
                "auto_generated_tls_phase_layout": AUTO_GENERATED_TLS_PHASE_LAYOUT,
                "source_trip_count": source_trip_count,
                "windowed_input_trip_count": total_windowed_trips,
                "routed_vehicle_count": total_routed_vehicles,
                "all_source_trips_covered": total_windowed_trips
                == source_trip_count,
            },
            "city_groups": {
                "shenzhen": [scenario for scenario, _, _ in WINDOWS]
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
    parser.add_argument("--netconvert", type=Path, required=True)
    parser.add_argument("--duarouter", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = package_pcl(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        netconvert=args.netconvert,
        duarouter=args.duarouter,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": payload["protocol"],
                "network_count": len(payload["networks"]),
                "vehicle_count": sum(
                    int(row["vehicle_count"])
                    for row in payload["networks"].values()
                ),
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
