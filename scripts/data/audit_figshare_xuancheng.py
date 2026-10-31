#!/usr/bin/env python3
"""Audit native Xuancheng SUMO topology against one complete CityFlow day."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.data.acquire_figshare_xuancheng import (
    ARTICLE_ID,
    ARTICLE_VERSION,
    CANDIDATE_DAY,
    PROTOCOL as ACQUISITION_PROTOCOL,
)


PROTOCOL = "figshare-xuancheng-native-sumo-candidate-audit-v1"


def _quantiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def value_at(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "minimum": float(ordered[0]),
        "q25": value_at(0.25),
        "median": value_at(0.5),
        "q75": value_at(0.75),
        "maximum": float(ordered[-1]),
    }


def _expanded_departures(row: dict[str, Any]) -> list[float]:
    start = float(row["startTime"])
    end = float(row["endTime"])
    interval = float(row["interval"])
    if interval <= 0.0 or end < start:
        raise ValueError(
            f"invalid CityFlow interval: start={start} end={end} interval={interval}"
        )
    count = int(math.floor((end - start) / interval + 1e-9)) + 1
    return [start + index * interval for index in range(count)]


def audit_candidate(acquisition_root: Path) -> dict[str, Any]:
    root = Path(acquisition_root).resolve()
    manifest = json.loads(
        (root / "acquisition_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("protocol") != ACQUISITION_PROTOCOL
        or int(manifest.get("article_id", -1)) != ARTICLE_ID
        or int(manifest.get("article_version", -1)) != ARTICLE_VERSION
        or manifest.get("coverage") != "candidate_screen"
        or manifest.get("candidate_day") != CANDIDATE_DAY
    ):
        raise ValueError("Xuancheng candidate acquisition manifest changed")
    for name, row in dict(manifest.get("files", {})).items():
        path = root / name
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Xuancheng acquired file size changed: {name}")

    net_root = ET.parse(root / "xuancheng.net.xml").getroot()
    sumo_edges = {
        str(edge.attrib["id"])
        for edge in net_root.findall("edge")
        if not edge.attrib.get("function")
    }
    sumo_connections = {
        (str(connection.attrib["from"]), str(connection.attrib["to"]))
        for connection in net_root.findall("connection")
        if "from" in connection.attrib and "to" in connection.attrib
    }
    sumo_programs = [
        {
            "id": str(logic.attrib["id"]),
            "program_id": str(logic.attrib.get("programID", "0")),
            "phase_count": len(logic.findall("phase")),
        }
        for logic in net_root.findall("tlLogic")
    ]
    sumo_tls_ids = {row["id"] for row in sumo_programs}
    sumo_junctions = net_root.findall("junction")

    roadnet = json.loads(
        (root / "roadnet_xuancheng250319.json").read_text(encoding="utf-8")
    )
    roads = list(roadnet["roads"])
    intersections = list(roadnet["intersections"])
    cityflow_edges = {str(road["id"]) for road in roads}
    cityflow_connections = {
        (str(link["startRoad"]), str(link["endRoad"]))
        for intersection in intersections
        for link in intersection.get("roadLinks", [])
    }
    cityflow_tls_ids = {
        str(intersection["id"])
        for intersection in intersections
        if not bool(intersection.get("virtual", False))
        and len(
            dict(intersection.get("trafficLight", {})).get("lightphases", [])
        )
        > 1
    }

    flow_rows = json.loads((root / CANDIDATE_DAY).read_text(encoding="utf-8"))
    route_lengths: list[int] = []
    departure_hours: Counter[int] = Counter()
    unique_routes: set[tuple[str, ...]] = set()
    used_edges: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    missing_edges: set[str] = set()
    missing_pairs: set[tuple[str, str]] = set()
    parameter_signatures: Counter[str] = Counter()
    expanded_vehicle_count = 0
    minimum_departure = math.inf
    maximum_departure = -math.inf
    multi_departure_flow_count = 0
    empty_route_count = 0
    for row in flow_rows:
        route = tuple(str(edge) for edge in row.get("route", []))
        if not route:
            empty_route_count += 1
            continue
        departures = _expanded_departures(row)
        if len(departures) > 1:
            multi_departure_flow_count += 1
        expanded_vehicle_count += len(departures)
        minimum_departure = min(minimum_departure, departures[0])
        maximum_departure = max(maximum_departure, departures[-1])
        for departure in departures:
            departure_hours[int(departure // 3600)] += 1
        route_lengths.append(len(route))
        unique_routes.add(route)
        used_edges.update(route)
        missing_edges.update(set(route) - sumo_edges)
        for pair in zip(route, route[1:]):
            used_pairs.add(pair)
            if pair not in sumo_connections:
                missing_pairs.add(pair)
        parameter_signatures[
            json.dumps(row.get("vehicle", {}), sort_keys=True, separators=(",", ":"))
        ] += len(departures)

    checks = {
        "source_file_inventory": len(dict(manifest["files"])) == 4,
        "road_id_exact_match": cityflow_edges == sumo_edges,
        "cityflow_connections_present_in_sumo": not (
            cityflow_connections - sumo_connections
        ),
        "sumo_tls_match_multiphase_cityflow_intersections": (
            cityflow_tls_ids == sumo_tls_ids
        ),
        "positive_full_day_demand": expanded_vehicle_count > 0,
        "all_flow_route_edges_present": not missing_edges,
        "all_flow_route_pairs_connected": not missing_pairs,
        "no_empty_routes": empty_route_count == 0,
        "full_day_time_coverage": (
            minimum_departure == 0.0
            and maximum_departure >= 23.0 * 3600.0
            and set(departure_hours) == set(range(24))
        ),
    }
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "acquisition": {
            "article_id": ARTICLE_ID,
            "article_version": ARTICLE_VERSION,
            "candidate_day": CANDIDATE_DAY,
            "source_md5": dict(manifest["files"])[CANDIDATE_DAY]["md5"],
        },
        "network": {
            "sumo_net_version": str(net_root.attrib.get("version", "")),
            "sumo_edge_count": len(sumo_edges),
            "cityflow_road_count": len(cityflow_edges),
            "road_id_symmetric_difference_count": len(sumo_edges ^ cityflow_edges),
            "sumo_connection_pair_count": len(sumo_connections),
            "cityflow_roadlink_pair_count": len(cityflow_connections),
            "cityflow_connection_missing_from_sumo_count": len(
                cityflow_connections - sumo_connections
            ),
            "sumo_junction_count": len(sumo_junctions),
            "cityflow_intersection_count": len(intersections),
            "cityflow_virtual_intersection_count": sum(
                bool(row.get("virtual", False)) for row in intersections
            ),
            "sumo_tls_count": len(sumo_tls_ids),
            "sumo_tls_program_count": len(sumo_programs),
            "cityflow_multiphase_intersection_count": len(cityflow_tls_ids),
            "tls_id_symmetric_difference_count": len(sumo_tls_ids ^ cityflow_tls_ids),
            "sumo_tls_phase_count": sum(row["phase_count"] for row in sumo_programs),
        },
        "demand": {
            "flow_record_count": len(flow_rows),
            "expanded_vehicle_count": expanded_vehicle_count,
            "multi_departure_flow_count": multi_departure_flow_count,
            "minimum_departure_sec": minimum_departure,
            "maximum_departure_sec": maximum_departure,
            "departure_count_by_hour": {
                str(hour): departure_hours[hour] for hour in range(24)
            },
            "route_length_edges": _quantiles(route_lengths),
            "unique_route_count": len(unique_routes),
            "used_edge_count": len(used_edges),
            "used_connection_pair_count": len(used_pairs),
            "missing_route_edge_count": len(missing_edges),
            "missing_route_connection_pair_count": len(missing_pairs),
            "empty_route_count": empty_route_count,
            "vehicle_parameter_signature_count": len(parameter_signatures),
            "vehicle_parameter_signature_vehicle_counts": dict(parameter_signatures),
        },
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_candidate(args.acquisition_root)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Results saved to: {args.output.resolve()}")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
