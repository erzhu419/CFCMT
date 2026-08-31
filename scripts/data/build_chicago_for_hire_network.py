#!/usr/bin/env python3
"""Build and statically admit the frozen Chicago v107 OSM network."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "cfcmt-chicago-osm-sumo122-network-and-anchor-v1"
ACQUISITION_PROTOCOL = "cfcmt-chicago-full-week-for-hire-acquisition-v1"
EXPECTED_TNP_ROWS = 1_599_557
EXPECTED_TAXI_ROWS = 118_756
EXPECTED_SPEED_ROWS = 1_004_073
EXPECTED_DAILY_FILES = 21
EXPECTED_OSM_SIZE = 231_147_462
EXPECTED_OSM_SHA256 = "2ef110eff0932a3b34548b8138f2b3f315083e6dc88cbf5e4f97780a919f2cda"
EXPECTED_BOUNDARY_SIZE = 2_072_102
EXPECTED_BOUNDARY_SHA256 = "3468ac30bd813fa17d755b12a911a43c44f251465604bbe8e1ca1b1b540dca66"
ANCHORS_PER_AREA = 32
MIN_EDGE_LENGTH_M = 15.0
EXPECTED_SUMO_VERSION = "1.22.0"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_file(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"frozen Chicago source size changed: {path}")
    observed = _file_sha256(path)
    if observed != sha256:
        raise ValueError(f"frozen Chicago source digest changed: {path}")


def _acquisition_manifest(acquisition_root: Path) -> dict[str, Any]:
    path = acquisition_root / "acquisition_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": ACQUISITION_PROTOCOL,
        "workers": 8,
        "trip_source_totals": {
            "tnp": EXPECTED_TNP_ROWS,
            "taxi": EXPECTED_TAXI_ROWS,
        },
        "speed_row_total": EXPECTED_SPEED_ROWS,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Chicago acquisition manifest changed: {observed}")
    daily = payload.get("daily_files", [])
    if len(daily) != EXPECTED_DAILY_FILES:
        raise ValueError(f"Chicago acquisition daily inventory changed: {len(daily)}")
    osm = payload.get("osm", {})
    boundaries = payload.get("community_areas", {})
    if {
        "path": osm.get("path"),
        "size_bytes": osm.get("size_bytes"),
        "sha256": osm.get("sha256"),
    } != {
        "path": "Chicago.osm.gz",
        "size_bytes": EXPECTED_OSM_SIZE,
        "sha256": EXPECTED_OSM_SHA256,
    }:
        raise ValueError("Chicago OSM manifest identity changed")
    if {
        "path": boundaries.get("path"),
        "size_bytes": boundaries.get("size_bytes"),
        "sha256": boundaries.get("sha256"),
        "feature_count": boundaries.get("feature_count"),
    } != {
        "path": "community_areas.geojson",
        "size_bytes": EXPECTED_BOUNDARY_SIZE,
        "sha256": EXPECTED_BOUNDARY_SHA256,
        "feature_count": 77,
    }:
        raise ValueError("Chicago boundary manifest identity changed")
    _validate_file(
        acquisition_root / "Chicago.osm.gz",
        size=EXPECTED_OSM_SIZE,
        sha256=EXPECTED_OSM_SHA256,
    )
    _validate_file(
        acquisition_root / "community_areas.geojson",
        size=EXPECTED_BOUNDARY_SIZE,
        sha256=EXPECTED_BOUNDARY_SHA256,
    )
    return payload


def netconvert_command(
    *, netconvert: Path, osm: Path, output: Path, typemap: Path
) -> list[str]:
    return [
        str(netconvert),
        "--osm-files",
        str(osm),
        "--type-files",
        str(typemap),
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--tls.default-type",
        "actuated",
        "--output.original-names",
        "true",
        "--output-file",
        str(output),
    ]


def _run_command(
    *, name: str, command: Sequence[str], cwd: Path, log_root: Path
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result = {
        "command": [str(value) for value in command],
        "returncode": int(completed.returncode),
        "duration_sec": time.monotonic() - started,
        "stdout_log": stdout_path.relative_to(log_root.parent).as_posix(),
        "stderr_log": stderr_path.relative_to(log_root.parent).as_posix(),
        "stdout_tail": completed.stdout[-2_000:],
        "stderr_tail": completed.stderr[-4_000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed ({completed.returncode}): {completed.stderr[-4000:]}"
        )
    return result


def _sumo_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot query {executable} version")
    first = completed.stdout.splitlines()[0] if completed.stdout else ""
    version = first.rsplit(" ", 1)[-1]
    if version != EXPECTED_SUMO_VERSION:
        raise ValueError(f"SUMO version changed: {version}")
    return version


def _network_xml_inventory(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    edge_ids: set[str] = set()
    lane_ids: set[str] = set()
    junction_ids: set[str] = set()
    tls_ids: set[str] = set()
    tls_programs: set[tuple[str, str]] = set()
    location: dict[str, str] | None = None
    with gzip.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            counts[element.tag] += 1
            identity = str(element.attrib.get("id", ""))
            if element.tag == "edge" and identity:
                if identity in edge_ids:
                    raise ValueError(f"duplicate network edge: {identity}")
                edge_ids.add(identity)
            elif element.tag == "lane" and identity:
                if identity in lane_ids:
                    raise ValueError(f"duplicate network lane: {identity}")
                lane_ids.add(identity)
            elif element.tag == "junction" and identity:
                if identity in junction_ids:
                    raise ValueError(f"duplicate network junction: {identity}")
                junction_ids.add(identity)
            elif element.tag == "tlLogic" and identity:
                tls_ids.add(identity)
                key = (identity, str(element.attrib.get("programID", "")))
                if key in tls_programs:
                    raise ValueError(f"duplicate traffic-light program: {key}")
                tls_programs.add(key)
            elif element.tag == "location":
                location = dict(element.attrib)
            element.clear()
    inventory = {
        "edge_count": len(edge_ids),
        "noninternal_edge_count": sum(not value.startswith(":") for value in edge_ids),
        "lane_count": len(lane_ids),
        "junction_count": len(junction_ids),
        "connection_count": counts["connection"],
        "traffic_light_count": len(tls_ids),
        "traffic_light_program_count": len(tls_programs),
        "traffic_light_phase_count": counts["phase"],
        "location": location,
    }
    if not edge_ids or not lane_ids or not junction_ids or not tls_ids or location is None:
        raise ValueError(f"Chicago network inventory is incomplete: {inventory}")
    return inventory


def largest_strongly_connected_component(
    adjacency: Mapping[str, Sequence[str]],
) -> set[str]:
    nodes = sorted(adjacency)
    node_set = set(nodes)
    normalized = {
        node: tuple(sorted({value for value in adjacency[node] if value in node_set}))
        for node in nodes
    }
    reverse: dict[str, list[str]] = {node: [] for node in nodes}
    for node, neighbors in normalized.items():
        for neighbor in neighbors:
            reverse[neighbor].append(node)
    for neighbors in reverse.values():
        neighbors.sort()

    visited: set[str] = set()
    finish: list[str] = []
    for start in nodes:
        if start in visited:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for neighbor in reversed(normalized[node]):
                if neighbor not in visited:
                    stack.append((neighbor, False))

    best: set[str] = set()
    best_minimum = ""
    assigned: set[str] = set()
    for start in reversed(finish):
        if start in assigned:
            continue
        component: set[str] = set()
        stack = [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in reverse[node]:
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)
        minimum = min(component)
        if len(component) > len(best) or (
            len(component) == len(best) and (not best or minimum < best_minimum)
        ):
            best = component
            best_minimum = minimum
    return best


def _point_on_segment(
    x: float,
    y: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
    if abs(cross) > 1e-10:
        return False
    return (
        min(ax, bx) - 1e-12 <= x <= max(ax, bx) + 1e-12
        and min(ay, by) - 1e-12 <= y <= max(ay, by) + 1e-12
    )


def _point_in_ring(point: tuple[float, float], ring: Sequence[Sequence[float]]) -> bool:
    x, y = point
    inside = False
    if len(ring) < 4:
        return False
    previous = ring[-1]
    for current in ring:
        ax, ay = float(previous[0]), float(previous[1])
        bx, by = float(current[0]), float(current[1])
        if _point_on_segment(x, y, ax, ay, bx, by):
            return True
        if (ay > y) != (by > y):
            intersection = (bx - ax) * (y - ay) / (by - ay) + ax
            if x < intersection:
                inside = not inside
        previous = current
    return inside


def point_in_geometry(point: tuple[float, float], geometry: Mapping[str, Any]) -> bool:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if geometry_type == "Polygon" else coordinates
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"unsupported Chicago boundary geometry: {geometry_type}")
    for polygon in polygons:
        if not polygon or not _point_in_ring(point, polygon[0]):
            continue
        if any(_point_in_ring(point, hole) for hole in polygon[1:]):
            continue
        return True
    return False


def _geometry_bounds(geometry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if geometry.get("type") == "Polygon" else coordinates
    points = [point for polygon in polygons for ring in polygon for point in ring]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _community_areas(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        area = int(properties["area_numbe"])
        geometry = feature["geometry"]
        rows.append(
            {
                "area": area,
                "name": str(properties["community"]),
                "geometry": geometry,
                "bounds": _geometry_bounds(geometry),
            }
        )
    rows.sort(key=lambda row: int(row["area"]))
    if [row["area"] for row in rows] != list(range(1, 78)):
        raise ValueError("Chicago community-area identities are incomplete")
    return rows


def _polyline_midpoint(shape: Sequence[Sequence[float]]) -> tuple[float, float]:
    if not shape:
        raise ValueError("edge has no shape")
    if len(shape) == 1:
        return float(shape[0][0]), float(shape[0][1])
    lengths = [
        math.hypot(
            float(shape[index + 1][0]) - float(shape[index][0]),
            float(shape[index + 1][1]) - float(shape[index][1]),
        )
        for index in range(len(shape) - 1)
    ]
    target = sum(lengths) / 2.0
    traversed = 0.0
    for index, length in enumerate(lengths):
        if traversed + length >= target and length > 0:
            fraction = (target - traversed) / length
            return (
                float(shape[index][0])
                + fraction * (float(shape[index + 1][0]) - float(shape[index][0])),
                float(shape[index][1])
                + fraction * (float(shape[index + 1][1]) - float(shape[index][1])),
            )
        traversed += length
    return float(shape[-1][0]), float(shape[-1][1])


def select_anchor_rows(
    candidates: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} anchor candidates for {count} slots")
    remaining = {str(row["edge_id"]): dict(row) for row in candidates}
    first = sorted(
        remaining.values(),
        key=lambda row: (-float(row["capacity_score"]), str(row["edge_id"])),
    )[0]
    selected = [first]
    del remaining[str(first["edge_id"])]
    while len(selected) < count:
        ranked: list[tuple[float, float, str, dict[str, Any]]] = []
        for row in remaining.values():
            distance = min(
                (float(row["x"]) - float(chosen["x"])) ** 2
                + (float(row["y"]) - float(chosen["y"])) ** 2
                for chosen in selected
            )
            ranked.append(
                (
                    -distance,
                    -float(row["capacity_score"]),
                    str(row["edge_id"]),
                    row,
                )
            )
        chosen = min(ranked)[3]
        selected.append(chosen)
        del remaining[str(chosen["edge_id"])]
    return selected


def _area_for_lonlat(
    lon: float, lat: float, areas: Sequence[Mapping[str, Any]]
) -> int | None:
    for row in areas:
        west, south, east, north = row["bounds"]
        if west <= lon <= east and south <= lat <= north:
            if point_in_geometry((lon, lat), row["geometry"]):
                return int(row["area"])
    return None


def _build_anchors(net_path: Path, boundary_path: Path) -> dict[str, Any]:
    import sumolib

    net = sumolib.net.readNet(str(net_path), withInternal=True)
    eligible: dict[str, Any] = {}
    for edge in net.getEdges(withInternal=False):
        if (
            edge.allows("passenger")
            and float(edge.getLength()) >= MIN_EDGE_LENGTH_M
            and float(edge.getSpeed()) > 0.0
        ):
            eligible[edge.getID()] = edge
    adjacency = {
        identity: [
            outgoing.getID()
            for outgoing in edge.getOutgoing()
            if outgoing.getID() in eligible
        ]
        for identity, edge in eligible.items()
    }
    component = largest_strongly_connected_component(adjacency)
    if not component:
        raise ValueError("Chicago passenger graph has no strongly connected component")
    areas = _community_areas(boundary_path)
    candidates: dict[int, list[dict[str, Any]]] = {area: [] for area in range(1, 78)}
    unassigned = 0
    for identity in sorted(component):
        edge = eligible[identity]
        x, y = _polyline_midpoint(edge.getShape())
        lon, lat = net.convertXY2LonLat(x, y)
        area = _area_for_lonlat(float(lon), float(lat), areas)
        if area is None:
            unassigned += 1
            continue
        lane_count = int(edge.getLaneNumber())
        speed = float(edge.getSpeed())
        candidates[area].append(
            {
                "edge_id": identity,
                "x": float(x),
                "y": float(y),
                "lon": float(lon),
                "lat": float(lat),
                "lane_count": lane_count,
                "speed_mps": speed,
                "length_m": float(edge.getLength()),
                "capacity_score": lane_count * speed,
            }
        )
    anchor_rows: dict[str, Any] = {}
    for area in range(1, 78):
        selected = select_anchor_rows(candidates[area], ANCHORS_PER_AREA)
        anchor_rows[str(area)] = {
            "community": areas[area - 1]["name"],
            "candidate_count": len(candidates[area]),
            "anchors": selected,
        }
    return {
        "protocol": "cfcmt-chicago-capacity-farthest-passenger-scc-anchors-v1",
        "eligible_passenger_edge_count": len(eligible),
        "largest_strong_component_edge_count": len(component),
        "component_fraction_of_eligible": len(component) / len(eligible),
        "component_edge_id_min": min(component),
        "component_edge_id_max": max(component),
        "component_unassigned_to_chicago_area_count": unassigned,
        "anchors_per_area": ANCHORS_PER_AREA,
        "total_anchor_count": 77 * ANCHORS_PER_AREA,
        "areas": anchor_rows,
    }


def build_network(
    *, acquisition_root: Path, output_root: Path, netconvert: Path
) -> dict[str, Any]:
    existing_manifest = output_root / "network_manifest.json"
    if existing_manifest.is_file():
        payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("existing Chicago network uses a different protocol")
        return payload
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    rejected = output_root.parent / f"{output_root.name}.rejected"
    if staging.exists() or rejected.exists():
        raise FileExistsError(staging if staging.exists() else rejected)
    staging.mkdir(parents=True)
    logs = staging / "logs"
    logs.mkdir()
    try:
        acquisition = _acquisition_manifest(acquisition_root)
        resolved_netconvert = netconvert.resolve()
        version = _sumo_version(resolved_netconvert)
        sumo_home = Path(os.environ["SUMO_HOME"]).resolve()
        typemap = sumo_home / "data/typemap/osmNetconvert.typ.xml"
        if not typemap.is_file():
            raise FileNotFoundError(typemap)
        net_path = staging / "chicago_full.net.xml.gz"
        command = netconvert_command(
            netconvert=resolved_netconvert,
            osm=acquisition_root / "Chicago.osm.gz",
            output=net_path,
            typemap=typemap,
        )
        netconvert_result = _run_command(
            name="netconvert", command=command, cwd=staging, log_root=logs
        )
        inventory = _network_xml_inventory(net_path)
        anchors = _build_anchors(
            net_path, acquisition_root / "community_areas.geojson"
        )
        _write_json(staging / "anchors.json", anchors)
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "acquisition_protocol": acquisition["protocol"],
            "acquisition_root": str(acquisition_root.resolve()),
            "acquisition_manifest_sha256": _file_sha256(
                acquisition_root / "acquisition_manifest.json"
            ),
            "sumo_version": version,
            "netconvert": netconvert_result,
            "network_file": net_path.name,
            "network_size_bytes": net_path.stat().st_size,
            "network_sha256": _file_sha256(net_path),
            "network_inventory": inventory,
            "anchor_file": "anchors.json",
            "anchor_protocol": anchors["protocol"],
            "anchor_summary": {
                key: anchors[key]
                for key in (
                    "eligible_passenger_edge_count",
                    "largest_strong_component_edge_count",
                    "component_fraction_of_eligible",
                    "component_unassigned_to_chicago_area_count",
                    "anchors_per_area",
                    "total_anchor_count",
                )
            },
            "status": "PASS",
        }
        _write_json(staging / "network_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception as error:
        failure = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "REJECT",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(staging / "build_failure.json", failure)
        os.replace(staging, rejected)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--netconvert", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_network(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        netconvert=args.netconvert,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol": payload["protocol"],
                "sumo_version": payload["sumo_version"],
                "network_inventory": payload["network_inventory"],
                "anchor_summary": payload["anchor_summary"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
