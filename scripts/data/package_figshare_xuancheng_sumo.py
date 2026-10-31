#!/usr/bin/env python3
"""Package Xuancheng anchor flows against the published native SUMO network."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.data.acquire_figshare_xuancheng import (
    ARTICLE_ID,
    ARTICLE_VERSION,
    CANDIDATE_DAY,
    DAILY_PATTERN,
    PROTOCOL as ACQUISITION_PROTOCOL,
)
from scripts.data.cityflow_anchor_router import (
    CITYFLOW_REFERENCE_COMMIT,
    NATIVE_SUMO_ROUTER_PROTOCOL,
    NativeSumoLengthRouter,
    anchors_are_subsequence,
)


PROTOCOL = "cfcmt-figshare-xuancheng-native-sumo-package-v1"
NETWORK_BUILD_PROTOCOL = "figshare-xuancheng-native-sumo-read-only-v1"
DEMAND_PROTOCOL = "figshare-xuancheng-cityflow-anchor-completion-v1"
TLS_PROTOCOL = "figshare-xuancheng-native-sumo-tls-preserved-v1"
HORIZON_SEC = 86_401.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.12g}"


def _expanded_departures(row: Mapping[str, Any]) -> tuple[float, ...]:
    start = float(row["startTime"])
    end = float(row["endTime"])
    interval = float(row["interval"])
    if interval <= 0.0 or end < start:
        raise ValueError(
            f"invalid CityFlow flow interval: start={start} end={end} interval={interval}"
        )
    count = int(math.floor((end - start) / interval + 1e-9)) + 1
    return tuple(start + index * interval for index in range(count))


def _vehicle_signature(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"))


def _vtype_attributes(signature: str, type_id: str) -> dict[str, str]:
    source = json.loads(signature)
    mapping = {
        "id": type_id,
        "length": _format_number(float(source["length"])),
        "width": _format_number(float(source["width"])),
        "minGap": _format_number(float(source["minGap"])),
        "maxSpeed": _format_number(float(source["maxSpeed"])),
        "accel": _format_number(float(source["maxPosAcc"])),
        "decel": _format_number(float(source["maxNegAcc"])),
        "tau": _format_number(float(source["headwayTime"])),
    }
    return mapping


def _write_tag(handle, tag: str, attributes: Mapping[str, str]) -> None:
    rendered = " ".join(
        f"{key}={quoteattr(str(value))}" for key, value in attributes.items()
    )
    handle.write(f"  <{tag} {rendered}/>\n")


def _network_inventory(net_path: Path) -> dict[str, Any]:
    root = ET.parse(net_path).getroot()
    edges = {
        str(edge.attrib["id"])
        for edge in root.findall("edge")
        if not edge.attrib.get("function")
    }
    connections = {
        (str(item.attrib["from"]), str(item.attrib["to"]))
        for item in root.findall("connection")
        if "from" in item.attrib and "to" in item.attrib
    }
    programs = [
        (str(item.attrib["id"]), len(item.findall("phase")))
        for item in root.findall("tlLogic")
    ]
    if not programs or any(count <= 0 for _, count in programs):
        raise ValueError("Xuancheng native SUMO network has invalid TLS programs")
    return {
        "edges": edges,
        "connections": connections,
        "sumo_net_version": str(root.attrib.get("version", "")),
        "tls_ids": {identity for identity, _ in programs},
        "tls_program_count": len(programs),
        "tls_phase_count": sum(count for _, count in programs),
    }


def _daily_files(manifest: Mapping[str, Any]) -> list[str]:
    names = set(dict(manifest.get("files", {})))
    daily = sorted(name for name in names if DAILY_PATTERN.fullmatch(name))
    coverage = str(manifest.get("coverage"))
    if coverage == "candidate_screen" and daily != [CANDIDATE_DAY]:
        raise ValueError(f"candidate day coverage changed: {daily}")
    if coverage == "full_month":
        expected = [f"data_2023_04_{day:02d}_type_filtered.json" for day in range(1, 31)]
        if daily != expected:
            raise ValueError("full Xuancheng month is incomplete")
    if coverage not in {"candidate_screen", "full_month"}:
        raise ValueError(f"unknown Xuancheng acquisition coverage: {coverage}")
    return daily


def _scenario_name(filename: str) -> str:
    match = DAILY_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"invalid Xuancheng day filename: {filename}")
    return f"xuancheng_202304{match.group(1)}"


def _normalized_anchors(anchors: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for anchor in anchors:
        if not normalized or normalized[-1] != str(anchor):
            normalized.append(str(anchor))
    return tuple(normalized)


def _write_day(
    *,
    source_path: Path,
    destination: Path,
    scenario: str,
    router: NativeSumoLengthRouter,
    network: Mapping[str, Any],
    global_completions: dict[tuple[str, str], tuple[str, ...]],
) -> dict[str, Any]:
    rows = list(json.loads(source_path.read_text(encoding="utf-8")))
    if not rows:
        raise ValueError(f"Xuancheng day contains no demand: {source_path.name}")
    anchor_routes = {
        _normalized_anchors(tuple(str(edge) for edge in row["route"]))
        for row in rows
    }
    if any(not route for route in anchor_routes):
        raise ValueError(f"Xuancheng day contains an empty route: {source_path.name}")
    needed_pairs = {
        pair
        for route in anchor_routes
        for pair in zip(route, route[1:])
        if pair not in router.links and pair not in global_completions
    }
    global_completions.update(router.shortest_paths(needed_pairs))
    expanded_routes = {
        anchors: router.expand(anchors, global_completions)
        for anchors in anchor_routes
    }
    if any(
        not anchors_are_subsequence(anchors, expanded)
        for anchors, expanded in expanded_routes.items()
    ):
        raise ValueError("Xuancheng route completion changed anchor order")
    missing_edges = {
        edge
        for route in expanded_routes.values()
        for edge in route
        if edge not in network["edges"]
    }
    missing_pairs = {
        pair
        for route in expanded_routes.values()
        for pair in zip(route, route[1:])
        if pair not in network["connections"]
    }
    if missing_edges or missing_pairs:
        raise ValueError(
            "completed Xuancheng routes are invalid in native SUMO: "
            f"edges={sorted(missing_edges)[:5]} pairs={sorted(missing_pairs)[:5]}"
        )

    signatures = sorted({_vehicle_signature(row["vehicle"]) for row in rows})
    type_ids = {signature: f"xc_type_{index}" for index, signature in enumerate(signatures)}
    ordered_routes = sorted(expanded_routes.items())
    route_ids = {
        anchors: f"{scenario}_route_{index:06d}"
        for index, (anchors, _) in enumerate(ordered_routes)
    }
    vehicle_rows: list[tuple[float, int, int, dict[str, Any]]] = []
    for source_index, row in enumerate(rows):
        for departure_index, departure in enumerate(_expanded_departures(row)):
            vehicle_rows.append((departure, source_index, departure_index, row))
    vehicle_rows.sort(key=lambda item: (item[0], item[1], item[2]))

    destination.mkdir(parents=True)
    route_name = f"{scenario}.rou.xml"
    route_path = destination / route_name
    weighted_inserted_edges = 0
    with route_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        for signature in signatures:
            _write_tag(
                handle,
                "vType",
                _vtype_attributes(signature, type_ids[signature]),
            )
        for anchors, expanded in ordered_routes:
            _write_tag(
                handle,
                "route",
                {"id": route_ids[anchors], "edges": " ".join(expanded)},
            )
        for vehicle_index, (departure, _, _, row) in enumerate(vehicle_rows):
            anchors = _normalized_anchors(tuple(str(edge) for edge in row["route"]))
            inserted = len(expanded_routes[anchors]) - len(anchors)
            weighted_inserted_edges += inserted
            _write_tag(
                handle,
                "vehicle",
                {
                    "id": f"{scenario}_vehicle_{vehicle_index:07d}",
                    "type": type_ids[_vehicle_signature(row["vehicle"])],
                    "route": route_ids[anchors],
                    "depart": _format_number(departure),
                    "departLane": "best",
                    "departSpeed": "max",
                },
            )
        handle.write("</routes>\n")

    config_name = f"{scenario}.sumocfg"
    config = ET.Element("configuration")
    input_node = ET.SubElement(config, "input")
    ET.SubElement(input_node, "net-file", value="../xuancheng.net.xml")
    ET.SubElement(input_node, "route-files", value=route_name)
    time_node = ET.SubElement(config, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value=_format_number(HORIZON_SEC))
    report = ET.SubElement(config, "report")
    ET.SubElement(report, "no-step-log", value="true")
    ET.indent(config, space="  ")
    ET.ElementTree(config).write(
        destination / config_name,
        encoding="utf-8",
        xml_declaration=True,
    )

    departure_counter = Counter(int(row[0] // 3600) for row in vehicle_rows)
    source_anchor_edge_count = sum(len(route) for route in anchor_routes)
    completed_edge_count = sum(len(route) for route in expanded_routes.values())
    return {
        "sumocfg": f"{scenario}/{config_name}",
        "route_file": f"{scenario}/{route_name}",
        "horizon_sec": HORIZON_SEC,
        "vehicle_count": len(vehicle_rows),
        "person_count": 0,
        "controlled_intersection_count": len(network["tls_ids"]),
        "route_audit": {
            "route_count": len(expanded_routes),
            "missing_edge_route_count": 0,
            "disconnected_route_count": 0,
        },
        "connection_audit": {"missing_route_edge_pair_count": 0},
        "demand_audit": {
            "protocol": DEMAND_PROTOCOL,
            "source_flow_record_count": len(rows),
            "total": len(vehicle_rows),
            "before_horizon": len(vehicle_rows),
            "at_or_after_horizon": 0,
            "minimum_departure_sec": vehicle_rows[0][0],
            "maximum_departure_sec": vehicle_rows[-1][0],
            "source_anchor_route_count": len(anchor_routes),
            "source_anchor_edge_count": source_anchor_edge_count,
            "completed_explicit_route_count": len(expanded_routes),
            "completed_explicit_edge_count": completed_edge_count,
            "unique_inserted_edge_count": completed_edge_count - source_anchor_edge_count,
            "vehicle_weighted_inserted_edge_count": weighted_inserted_edges,
            "all_source_flow_records_loaded": True,
            "all_anchor_edges_preserved_in_order": True,
            "all_completed_route_pairs_native_sumo_connected": True,
            "departure_count_by_hour": {
                str(hour): departure_counter[hour]
                for hour in sorted(departure_counter)
            },
            "vehicle_parameter_signature_count": len(signatures),
        },
        "network_build": {
            "protocol": NETWORK_BUILD_PROTOCOL,
            "mode": "source_native_sumo_read_only_with_cityflow_anchor_completion",
            "source_network_preserved_byte_for_byte": True,
            "traffic_signal_programs_preserved_byte_for_byte": True,
            "traffic_semantics_rebuilt_or_calibrated": False,
            "route_completion_protocol": NATIVE_SUMO_ROUTER_PROTOCOL,
            "cityflow_reference_commit": CITYFLOW_REFERENCE_COMMIT,
        },
        "tls_semantic_audit": {
            "protocol": TLS_PROTOCOL,
            "controlled_intersection_count": len(network["tls_ids"]),
            "program_count": int(network["tls_program_count"]),
            "phase_count": int(network["tls_phase_count"]),
            "source_native_sumo_tls_preserved_byte_for_byte": True,
        },
        "source_day_file": source_path.name,
        "source_day_md5": None,
    }


def package_xuancheng(*, acquisition_root: Path, output_root: Path) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Xuancheng package: {output_root}")
    manifest = json.loads(
        (acquisition_root / "acquisition_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("protocol") != ACQUISITION_PROTOCOL
        or int(manifest.get("article_id", -1)) != ARTICLE_ID
        or int(manifest.get("article_version", -1)) != ARTICLE_VERSION
    ):
        raise ValueError("Xuancheng acquisition identity changed")
    daily_files = _daily_files(manifest)
    for name, row in dict(manifest["files"]).items():
        path = acquisition_root / name
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Xuancheng acquisition file size changed: {name}")

    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(acquisition_root / "xuancheng.net.xml", staging / "xuancheng.net.xml")
        roadnet = json.loads(
            (acquisition_root / "roadnet_xuancheng250319.json").read_text(
                encoding="utf-8"
            )
        )
        network = _network_inventory(staging / "xuancheng.net.xml")
        router = NativeSumoLengthRouter(staging / "xuancheng.net.xml")
        cityflow_road_ids = {str(row["id"]) for row in roadnet["roads"]}
        if cityflow_road_ids != set(network["edges"]):
            raise ValueError("CityFlow and native SUMO road IDs differ")
        completions: dict[tuple[str, str], tuple[str, ...]] = {}
        networks = {}
        for filename in daily_files:
            scenario = _scenario_name(filename)
            row = _write_day(
                source_path=acquisition_root / filename,
                destination=staging / scenario,
                scenario=scenario,
                router=router,
                network=network,
                global_completions=completions,
            )
            row["source_day_md5"] = dict(manifest["files"])[filename]["md5"]
            networks[scenario] = row
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "article_id": ARTICLE_ID,
            "article_version": ARTICLE_VERSION,
            "coverage": manifest["coverage"],
            "network_source_md5": dict(manifest["files"])["xuancheng.net.xml"]["md5"],
            "roadnet_source_md5": dict(manifest["files"])[
                "roadnet_xuancheng250319.json"
            ]["md5"],
            "scenario_count": len(networks),
            "global_completed_anchor_pair_count": len(completions),
            "networks": networks,
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
    payload = package_xuancheng(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "coverage": payload["coverage"],
                "scenario_count": payload["scenario_count"],
                "global_completed_anchor_pair_count": payload[
                    "global_completed_anchor_pair_count"
                ],
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
