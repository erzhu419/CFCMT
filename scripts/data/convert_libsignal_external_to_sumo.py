#!/usr/bin/env python3
"""Convert the full LibSignal LA and Nanchang benchmarks into SUMO inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


CityFlowConnectionKey = tuple[str, str, int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_path(path: Path, *, artifact_root: Path) -> str:
    """Return a relocation-safe path for a generated conversion artifact."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(artifact_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tree.write(handle, encoding="utf-8", xml_declaration=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _polyline_length(points: Sequence[dict[str, float]]) -> float:
    return float(
        sum(
            math.hypot(
                float(second["x"]) - float(first["x"]),
                float(second["y"]) - float(first["y"]),
            )
            for first, second in zip(points, points[1:])
        )
    )


def _validate_edge_routes(
    edge_endpoints: dict[str, tuple[str, str]],
    routes: Iterable[Sequence[str]],
) -> dict[str, Any]:
    route_count = 0
    missing_edge_routes = 0
    disconnected_routes = 0
    minimum_edges: int | None = None
    maximum_edges = 0
    for route in routes:
        edges = [str(value) for value in route]
        route_count += 1
        minimum_edges = len(edges) if minimum_edges is None else min(minimum_edges, len(edges))
        maximum_edges = max(maximum_edges, len(edges))
        if any(edge not in edge_endpoints for edge in edges):
            missing_edge_routes += 1
            continue
        if any(
            edge_endpoints[first][1] != edge_endpoints[second][0]
            for first, second in zip(edges, edges[1:])
        ):
            disconnected_routes += 1
    if route_count <= 0 or missing_edge_routes or disconnected_routes:
        raise ValueError(
            "external route topology validation failed: "
            f"routes={route_count} missing={missing_edge_routes} "
            f"disconnected={disconnected_routes}"
        )
    return {
        "route_count": route_count,
        "missing_edge_route_count": missing_edge_routes,
        "disconnected_route_count": disconnected_routes,
        "minimum_route_edges": int(minimum_edges or 0),
        "maximum_route_edges": maximum_edges,
    }


def _la_connection_key(
    *,
    road_link: dict[str, Any],
    lane_link: dict[str, Any],
    road_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str, int, int]:
    start_road = str(road_link["startRoad"])
    end_road = str(road_link["endRoad"])
    start_lanes = len(road_by_id[start_road]["lanes"])
    end_lanes = len(road_by_id[end_road]["lanes"])
    return (
        start_road,
        end_road,
        start_lanes - 1 - int(lane_link["startLaneIndex"]),
        end_lanes - 1 - int(lane_link["endLaneIndex"]),
    )


def _lane_pair_has_inversion(pairs: set[tuple[int, int]]) -> bool:
    """Return whether lane links cross under the source/target lane ordering."""

    return any(
        (first_source < second_source and first_target > second_target)
        or (first_source > second_source and first_target < second_target)
        for first_source, first_target in pairs
        for second_source, second_target in pairs
    )


def _nearest_rank(index: int, source_count: int, target_count: int) -> int:
    if source_count <= 0 or target_count <= 0:
        raise ValueError("lane-rank dimensions must be positive")
    if target_count == 1:
        return 0
    if source_count == 1:
        return (target_count - 1) // 2
    return (
        index * (target_count - 1) + (source_count - 1) // 2
    ) // (source_count - 1)


def _monotone_virtual_lane_pairs(
    pairs: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], bool]:
    """Collapse a virtual Cartesian lane fan-out to a covering monotone map.

    CityFlow routes identify roads rather than lanes. Some imported virtual
    junctions therefore enumerate every active incoming/outgoing lane pair to
    permit lane choice. Keeping that Cartesian product in SUMO creates crossing
    internal trajectories at an otherwise zero-width connector. A rank-aligned
    map preserves every active source and target lane without those crossings.
    """

    if not pairs or not _lane_pair_has_inversion(pairs):
        return set(pairs), False
    source_lanes = sorted({source for source, _ in pairs})
    target_lanes = sorted({target for _, target in pairs})
    cartesian = {
        (source, target)
        for source in source_lanes
        for target in target_lanes
    }
    if pairs != cartesian:
        raise ValueError(
            "crossing virtual lane links are not a Cartesian lane fan-out"
        )

    selected: set[tuple[int, int]] = set()
    for source_rank, source_lane in enumerate(source_lanes):
        target_rank = _nearest_rank(
            source_rank,
            len(source_lanes),
            len(target_lanes),
        )
        selected.add((source_lane, target_lanes[target_rank]))
    for target_rank, target_lane in enumerate(target_lanes):
        source_rank = _nearest_rank(
            target_rank,
            len(target_lanes),
            len(source_lanes),
        )
        selected.add((source_lanes[source_rank], target_lane))

    if {source for source, _ in selected} != set(source_lanes):
        raise AssertionError("virtual lane reduction lost an active source lane")
    if {target for _, target in selected} != set(target_lanes):
        raise AssertionError("virtual lane reduction lost an active target lane")
    if _lane_pair_has_inversion(selected):
        raise AssertionError("virtual lane reduction retained a crossing pair")
    return selected, True


def _sumo_connection_key(connection: ET.Element) -> CityFlowConnectionKey:
    return (
        str(connection.attrib["from"]),
        str(connection.attrib["to"]),
        int(connection.attrib["fromLane"]),
        int(connection.attrib["toLane"]),
    )


def _remap_cityflow_tls_states(
    net_path: Path,
    tls_semantics: dict[str, dict[str, Any]],
    *,
    yield_to_netconvert_priority: bool = False,
) -> dict[str, Any]:
    """Rewrite CityFlow phases against netconvert's final link ordering."""

    tree = ET.parse(net_path)
    root = tree.getroot()
    output_links: dict[str, dict[CityFlowConnectionKey, int]] = {}
    output_connection_attributes: dict[
        str,
        dict[CityFlowConnectionKey, dict[str, str]],
    ] = {}
    for connection in root.findall("connection"):
        tls_id = connection.attrib.get("tl")
        link_index = connection.attrib.get("linkIndex")
        if tls_id is None or link_index is None:
            continue
        key = _sumo_connection_key(connection)
        links = output_links.setdefault(str(tls_id), {})
        index = int(link_index)
        if key in links and links[key] != index:
            raise ValueError(
                "netconvert assigned one CityFlow lane connection to multiple "
                f"TLS indices: {tls_id}/{key}"
            )
        links[key] = index
        output_connection_attributes.setdefault(str(tls_id), {})[key] = dict(
            connection.attrib
        )

    logics: dict[str, ET.Element] = {}
    for logic in root.findall("tlLogic"):
        tls_id = str(logic.attrib.get("id", ""))
        if logic.attrib.get("programID", "0") != "0":
            continue
        if tls_id in logics:
            raise ValueError(f"duplicate SUMO TLS program 0: {tls_id}")
        logics[tls_id] = logic

    if set(output_links) != set(tls_semantics):
        raise ValueError(
            "CityFlow/SUMO controlled-intersection mismatch: "
            f"output_only={sorted(set(output_links) - set(tls_semantics))} "
            f"source_only={sorted(set(tls_semantics) - set(output_links))}"
        )
    if set(logics) != set(tls_semantics):
        raise ValueError(
            "CityFlow/SUMO TLS-program mismatch: "
            f"output_only={sorted(set(logics) - set(tls_semantics))} "
            f"source_only={sorted(set(tls_semantics) - set(logics))}"
        )

    per_tls: dict[str, dict[str, Any]] = {}
    total_before_mismatch = 0
    total_reassigned = 0
    total_phase_count = 0
    total_merge_yield_phase_links = 0
    total_conflict_yield_phase_links = 0
    for tls_id in sorted(tls_semantics):
        semantic = tls_semantics[tls_id]
        actual_by_key = output_links[tls_id]
        source_keys = set(semantic["provisional_indices"])
        actual_keys = set(actual_by_key)
        if source_keys != actual_keys:
            raise ValueError(
                f"CityFlow/SUMO TLS connection mismatch at {tls_id}: "
                f"output_only={sorted(actual_keys - source_keys)} "
                f"source_only={sorted(source_keys - actual_keys)}"
            )
        actual_indices = set(actual_by_key.values())
        if actual_indices != set(range(len(actual_indices))):
            raise ValueError(
                f"non-contiguous netconvert TLS indices at {tls_id}: "
                f"{sorted(actual_indices)}"
            )

        logic = logics[tls_id]
        old_phases = list(logic.findall("phase"))
        source_phases = list(semantic["phases"])
        append_intergreen = bool(semantic["append_intergreen"])
        expected_old_count = len(source_phases) * (2 if append_intergreen else 1)
        if len(old_phases) != expected_old_count:
            raise ValueError(
                f"unexpected pre-remap phase count at {tls_id}: "
                f"{len(old_phases)} != {expected_old_count}"
            )

        before_mismatch = 0
        for phase_index, source_phase in enumerate(source_phases):
            old_phase = old_phases[phase_index * (2 if append_intergreen else 1)]
            observed = {
                index
                for index, state in enumerate(str(old_phase.attrib["state"]))
                if state in "Gg"
            }
            expected = {
                actual_by_key[key] for key in source_phase["active_keys"]
            }
            before_mismatch += len(observed.symmetric_difference(expected))

        for phase in old_phases:
            logic.remove(phase)
        state_length = len(actual_indices)
        permissive_indices = {
            actual_by_key[key] for key in semantic["permissive_keys"]
        }
        if permissive_indices and any(
            not set(semantic["permissive_keys"]).issubset(
                set(source_phase["active_keys"])
            )
            for source_phase in source_phases
        ):
            raise ValueError(
                f"CityFlow phase-0 links are not always permissive at {tls_id}"
            )
        for source_phase in source_phases:
            active_keys = set(source_phase["active_keys"])
            active_indices = {
                actual_by_key[key] for key in active_keys
            }
            state = ["r"] * state_length
            for key in active_keys:
                index = actual_by_key[key]
                lower_priority_conflict = (
                    yield_to_netconvert_priority
                    and output_connection_attributes[tls_id][key].get("state")
                    == "o"
                )
                state[index] = (
                    "g"
                    if index in permissive_indices or lower_priority_conflict
                    else "G"
                )
                total_conflict_yield_phase_links += int(lower_priority_conflict)
            target_groups: dict[tuple[str, str], list[CityFlowConnectionKey]] = (
                defaultdict(list)
            )
            for key in active_keys:
                index = actual_by_key[key]
                if index in permissive_indices:
                    continue
                attributes = output_connection_attributes[tls_id][key]
                target_groups[(attributes["to"], attributes["toLane"])].append(
                    key
                )
            for keys in target_groups.values():
                if len(keys) <= 1:
                    continue
                protected = sorted(
                    key
                    for key in keys
                    if output_connection_attributes[tls_id][key].get("state")
                    == "O"
                )[:1]
                protected_set = set(protected)
                for key in keys:
                    index = actual_by_key[key]
                    if key not in protected_set and state[index] != "g":
                        state[index] = "g"
                        total_merge_yield_phase_links += 1
            protected_target_counts: dict[tuple[str, str], int] = defaultdict(int)
            for key in active_keys:
                index = actual_by_key[key]
                if state[index] != "G":
                    continue
                attributes = output_connection_attributes[tls_id][key]
                protected_target_counts[
                    (attributes["to"], attributes["toLane"])
                ] += 1
            if any(count > 1 for count in protected_target_counts.values()):
                raise AssertionError(
                    f"unsafe multi-G target lane remained at {tls_id}"
                )
            ET.SubElement(
                logic,
                "phase",
                duration=f"{float(source_phase['duration']):g}",
                state="".join(state),
            )
            if append_intergreen:
                intergreen = ["r"] * state_length
                for index in permissive_indices:
                    intergreen[index] = "g"
                for index in active_indices - permissive_indices:
                    intergreen[index] = "y"
                ET.SubElement(
                    logic,
                    "phase",
                    duration="5",
                    state="".join(intergreen),
                )

        after_mismatch = 0
        rewritten = list(logic.findall("phase"))
        for phase_index, source_phase in enumerate(source_phases):
            phase = rewritten[phase_index * (2 if append_intergreen else 1)]
            observed = {
                index
                for index, state in enumerate(str(phase.attrib["state"]))
                if state in "Gg"
            }
            expected = {
                actual_by_key[key] for key in source_phase["active_keys"]
            }
            after_mismatch += len(observed.symmetric_difference(expected))
        if after_mismatch:
            raise AssertionError(
                f"post-remap CityFlow TLS mismatch at {tls_id}: {after_mismatch}"
            )

        reassigned = sum(
            int(actual_by_key[key] != int(provisional_index))
            for key, provisional_index in semantic["provisional_indices"].items()
        )
        total_before_mismatch += before_mismatch
        total_reassigned += reassigned
        total_phase_count += len(source_phases)
        per_tls[tls_id] = {
            "connection_count": len(actual_by_key),
            "phase_count": len(source_phases),
            "pre_remap_phase_link_symmetric_difference": before_mismatch,
            "post_remap_phase_link_symmetric_difference": after_mismatch,
            "reassigned_connection_count": reassigned,
        }

    _write_xml(net_path, root)
    return {
        "protocol": (
            "cityflow-roadlink-to-netconvert-linkindex-remap-"
            + (
                "permissive-intergreen-merge-conflict-yield-v4"
                if yield_to_netconvert_priority
                else "permissive-intergreen-merge-yield-v3"
            )
        ),
        "controlled_intersection_count": len(per_tls),
        "phase_count": total_phase_count,
        "pre_remap_phase_link_symmetric_difference": total_before_mismatch,
        "post_remap_phase_link_symmetric_difference": 0,
        "reassigned_connection_count": total_reassigned,
        "merge_yield_phase_link_count": total_merge_yield_phase_links,
        "conflict_yield_phase_link_count": total_conflict_yield_phase_links,
        "per_tls": per_tls,
    }


def _run_netconvert(
    netconvert: Path,
    *,
    node_file: Path,
    edge_file: Path,
    output_file: Path,
    connection_file: Path | None = None,
    tls_file: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    command = [
        str(netconvert),
        "--node-files",
        str(node_file),
        "--edge-files",
        str(edge_file),
        "--output-file",
        str(output_file),
        "--no-turnarounds",
        "true",
        "--junctions.corner-detail",
        "5",
    ]
    if connection_file is not None:
        command.extend(["--connection-files", str(connection_file)])
    if tls_file is not None:
        command.extend(["--tllogic-files", str(tls_file)])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not output_file.is_file():
        raise RuntimeError(
            "netconvert failed\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    manifest_command = list(command)
    if artifact_root is not None:
        for generated_path in (
            node_file,
            edge_file,
            output_file,
            connection_file,
            tls_file,
        ):
            if generated_path is None:
                continue
            manifest_command = [
                _manifest_path(generated_path, artifact_root=artifact_root)
                if value == str(generated_path)
                else value
                for value in manifest_command
            ]
    return {
        "command": manifest_command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _write_sumocfg(path: Path, *, net_file: Path, route_file: Path, end: float) -> None:
    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", value=net_file.name)
    ET.SubElement(input_node, "route-files", value=route_file.name)
    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "end", value=f"{float(end):g}")
    _write_xml(path, root)


def convert_cityflow(
    *,
    roadnet_path: Path,
    flow_path: Path,
    output_dir: Path,
    netconvert: Path,
    city: str,
    network: str,
    prefix: str,
    yield_to_netconvert_priority: bool = False,
) -> dict[str, Any]:
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    flows = json.loads(flow_path.read_text(encoding="utf-8"))
    intersections = list(roadnet["intersections"])
    roads = list(roadnet["roads"])
    road_by_id = {str(road["id"]): road for road in roads}
    if len(road_by_id) != len(roads):
        raise ValueError(f"CityFlow road IDs are not unique: {network}")
    edge_endpoints = {
        road_id: (
            str(road["startIntersection"]),
            str(road["endIntersection"]),
        )
        for road_id, road in road_by_id.items()
    }
    route_audit = _validate_edge_routes(
        edge_endpoints,
        ([str(edge) for edge in row["route"]] for row in flows),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    node_path = output_dir / f"{prefix}.nod.xml"
    edge_path = output_dir / f"{prefix}.edg.xml"
    connection_path = output_dir / f"{prefix}.con.xml"
    tls_path = output_dir / f"{prefix}.tll.xml"
    net_path = output_dir / f"{prefix}.net.xml"
    route_path = output_dir / f"{prefix}.rou.xml"
    cfg_path = output_dir / f"{prefix}.sumocfg"

    nodes_root = ET.Element("nodes")
    controlled = 0
    for intersection in intersections:
        virtual = bool(intersection["virtual"])
        controlled += int(not virtual)
        ET.SubElement(
            nodes_root,
            "node",
            id=str(intersection["id"]),
            x=f"{float(intersection['point']['x']):.9f}",
            y=f"{float(intersection['point']['y']):.9f}",
            type="priority" if virtual else "traffic_light",
        )
    _write_xml(node_path, nodes_root)

    edges_root = ET.Element("edges")
    for road_id, road in road_by_id.items():
        lanes = list(road["lanes"])
        if not lanes:
            raise ValueError(f"CityFlow road has no lanes: {network}/{road_id}")
        speed = min(float(lane.get("maxSpeed", 11.111)) for lane in lanes)
        points = list(road.get("points", ()))
        attributes = {
            "id": road_id,
            "from": str(road["startIntersection"]),
            "to": str(road["endIntersection"]),
            "numLanes": str(len(lanes)),
            "speed": f"{speed:.9f}",
            "priority": "1",
        }
        length = _polyline_length(points) if len(points) >= 2 else 0.0
        if length > 0.0:
            attributes["length"] = f"{length:.9f}"
            attributes["shape"] = " ".join(
                f"{float(point['x']):.9f},{float(point['y']):.9f}"
                for point in points
            )
        ET.SubElement(edges_root, "edge", attributes)
    _write_xml(edge_path, edges_root)

    connections_root = ET.Element("connections")
    tls_root = ET.Element("tlLogics")
    tls_semantics: dict[str, dict[str, Any]] = {}
    connection_count = 0
    source_lane_link_count = 0
    duplicate_lane_link_count = 0
    virtual_lane_link_reduction = {
        "protocol": "cityflow-virtual-cartesian-lane-link-monotone-rank-v1",
        "reduced_road_link_count": 0,
        "source_unique_lane_pair_count": 0,
        "retained_unique_lane_pair_count": 0,
        "removed_unique_lane_pair_count": 0,
        "post_reduction_crossing_road_link_count": 0,
        "source_lane_coverage_failures": 0,
        "target_lane_coverage_failures": 0,
    }
    for intersection in intersections:
        link_indices: dict[int, list[int]] = {}
        road_link_keys: dict[int, set[CityFlowConnectionKey]] = {}
        provisional_indices: dict[CityFlowConnectionKey, int] = {}
        unique_connections: dict[
            CityFlowConnectionKey,
            tuple[int | None, tuple[tuple[float, float], ...]],
        ] = {}
        source_connection_geometries: dict[
            CityFlowConnectionKey,
            tuple[tuple[float, float], ...],
        ] = {}
        tls_link_index = 0
        for road_link_index, road_link in enumerate(intersection.get("roadLinks", ())):
            road_link_keys[road_link_index] = set()
            lane_links = list(road_link.get("laneLinks", ()))
            source_pairs = {
                (
                    int(lane_link["startLaneIndex"]),
                    int(lane_link["endLaneIndex"]),
                )
                for lane_link in lane_links
            }
            retained_pairs = set(source_pairs)
            if bool(intersection["virtual"]):
                retained_pairs, reduced = _monotone_virtual_lane_pairs(
                    source_pairs
                )
                if reduced:
                    virtual_lane_link_reduction["reduced_road_link_count"] += 1
                    virtual_lane_link_reduction[
                        "source_unique_lane_pair_count"
                    ] += len(source_pairs)
                    virtual_lane_link_reduction[
                        "retained_unique_lane_pair_count"
                    ] += len(retained_pairs)
                    virtual_lane_link_reduction[
                        "removed_unique_lane_pair_count"
                    ] += len(source_pairs) - len(retained_pairs)
                if _lane_pair_has_inversion(retained_pairs):
                    virtual_lane_link_reduction[
                        "post_reduction_crossing_road_link_count"
                    ] += 1
                if {
                    source for source, _ in retained_pairs
                } != {source for source, _ in source_pairs}:
                    virtual_lane_link_reduction[
                        "source_lane_coverage_failures"
                    ] += 1
                if {
                    target for _, target in retained_pairs
                } != {target for _, target in source_pairs}:
                    virtual_lane_link_reduction[
                        "target_lane_coverage_failures"
                    ] += 1
            for lane_link in lane_links:
                source_lane_link_count += 1
                key = _la_connection_key(
                    road_link=road_link,
                    lane_link=lane_link,
                    road_by_id=road_by_id,
                )
                geometry = tuple(
                    (float(point["x"]), float(point["y"]))
                    for point in lane_link.get("points", ())
                )
                if key in source_connection_geometries:
                    existing_geometry = source_connection_geometries[key]
                    if geometry != existing_geometry:
                        raise ValueError(
                            "CityFlow input has parallel lane links with one SUMO lane-pair "
                            f"identity but different geometry: {key}"
                        )
                    duplicate_lane_link_count += 1
                else:
                    source_connection_geometries[key] = geometry
                source_pair = (
                    int(lane_link["startLaneIndex"]),
                    int(lane_link["endLaneIndex"]),
                )
                if source_pair not in retained_pairs:
                    continue
                road_link_keys[road_link_index].add(key)
                if key in unique_connections:
                    existing_index, _ = unique_connections[key]
                    if existing_index is not None:
                        link_indices.setdefault(road_link_index, []).append(
                            existing_index
                        )
                    continue
                start_road, end_road, from_lane, to_lane = key
                attributes = {
                    "from": start_road,
                    "to": end_road,
                    "fromLane": str(from_lane),
                    "toLane": str(to_lane),
                }
                if not bool(intersection["virtual"]):
                    attributes["tl"] = str(intersection["id"])
                    attributes["linkIndex"] = str(tls_link_index)
                    link_indices.setdefault(road_link_index, []).append(
                        tls_link_index
                    )
                    unique_connections[key] = (tls_link_index, geometry)
                    provisional_indices[key] = tls_link_index
                    tls_link_index += 1
                else:
                    unique_connections[key] = (None, geometry)
                ET.SubElement(connections_root, "connection", attributes)
                connection_count += 1
        if bool(intersection["virtual"]):
            continue
        local_indices = sorted(
            {index for values in link_indices.values() for index in values}
        )
        if not local_indices:
            raise ValueError(
                "controlled CityFlow intersection has no lane links: "
                f"{network}/{intersection['id']}"
            )
        offset = 0
        if local_indices != list(range(len(local_indices))):
            raise ValueError("CityFlow TLS lane-link indices are not contiguous")
        logic = ET.SubElement(
            tls_root,
            "tlLogic",
            id=str(intersection["id"]),
            type="static",
            programID="0",
            offset="0",
        )
        phases = list(intersection.get("trafficLight", {}).get("lightphases", ()))
        append_intergreen = bool(phases)
        permissive_links: set[int] = set()
        if phases and float(phases[0].get("time", 0.0)) <= 5.0:
            permissive_links = {
                index
                for road_link_index in phases[0].get("availableRoadLinks", ())
                for index in link_indices.get(int(road_link_index), ())
            }
            phases = phases[1:]
        semantic_phases = [
            {
                "duration": float(phase.get("time", 30.0)),
                "active_keys": {
                    key
                    for road_link_index in phase.get("availableRoadLinks", ())
                    for key in road_link_keys.get(int(road_link_index), set())
                },
            }
            for phase in phases
        ]
        for phase in phases:
            active = {
                index
                for road_link_index in phase.get("availableRoadLinks", ())
                for index in link_indices.get(int(road_link_index), ())
            }
            state = ["r"] * len(local_indices)
            for index in active:
                state[index - offset] = "G"
            ET.SubElement(
                logic,
                "phase",
                duration=f"{float(phase.get('time', 30.0)):g}",
                state="".join(state),
            )
            yellow = ["r"] * len(local_indices)
            for index in permissive_links:
                yellow[index - offset] = "s"
            ET.SubElement(logic, "phase", duration="5", state="".join(yellow))
        if not phases:
            state = ["G"] * len(local_indices)
            ET.SubElement(logic, "phase", duration="30", state="".join(state))
            semantic_phases = [
                {
                    "duration": 30.0,
                    "active_keys": set(provisional_indices),
                }
            ]
            append_intergreen = False
        tls_semantics[str(intersection["id"])] = {
            "provisional_indices": dict(provisional_indices),
            "permissive_keys": {
                key
                for key, provisional_index in provisional_indices.items()
                if provisional_index in permissive_links
            },
            "phases": semantic_phases,
            "append_intergreen": append_intergreen,
        }
    _write_xml(connection_path, connections_root)
    _write_xml(tls_path, tls_root)

    failed_virtual_audits = {
        key: int(virtual_lane_link_reduction[key])
        for key in (
            "post_reduction_crossing_road_link_count",
            "source_lane_coverage_failures",
            "target_lane_coverage_failures",
        )
        if int(virtual_lane_link_reduction[key]) != 0
    }
    if failed_virtual_audits:
        raise AssertionError(
            "virtual lane-link reduction failed its topology audit: "
            f"{failed_virtual_audits}"
        )

    routes_root = ET.Element("routes")
    ET.SubElement(
        routes_root,
        "vType",
        id="passenger",
        vClass="passenger",
        length="5.0",
        width="2.0",
        minGap="2.5",
        maxSpeed="11.111",
        accel="2.0",
        decel="4.5",
        sigma="0.5",
    )
    for index, flow in enumerate(sorted(flows, key=lambda row: (float(row["startTime"]), str(row["route"])))):
        vehicle = ET.SubElement(
            routes_root,
            "vehicle",
            id=f"{prefix}_{index}",
            type="passenger",
            depart=f"{float(flow['startTime']):g}",
            departLane="best",
            departSpeed="max",
        )
        ET.SubElement(vehicle, "route", edges=" ".join(str(edge) for edge in flow["route"]))
    _write_xml(route_path, routes_root)
    netconvert_result = _run_netconvert(
        netconvert,
        node_file=node_path,
        edge_file=edge_path,
        connection_file=connection_path,
        tls_file=tls_path,
        output_file=net_path,
        artifact_root=output_dir.parent,
    )
    tls_semantic_audit = _remap_cityflow_tls_states(
        net_path,
        tls_semantics,
        yield_to_netconvert_priority=yield_to_netconvert_priority,
    )
    _write_sumocfg(cfg_path, net_file=net_path, route_file=route_path, end=3600.0)
    return {
        "city": str(city),
        "network": str(network),
        "source_format": "CityFlow JSON",
        "intersection_count": len(intersections),
        "controlled_intersection_count": controlled,
        "directed_edge_count": len(roads),
        "source_lane_link_count": source_lane_link_count,
        "connection_count": connection_count,
        "duplicate_lane_link_count": duplicate_lane_link_count,
        "virtual_lane_link_reduction": virtual_lane_link_reduction,
        "vehicle_count": len(flows),
        "route_audit": route_audit,
        "tls_semantic_audit": tls_semantic_audit,
        "netconvert": netconvert_result,
        "sumocfg": _manifest_path(cfg_path, artifact_root=output_dir.parent),
    }


def convert_la(
    *,
    roadnet_path: Path,
    flow_path: Path,
    output_dir: Path,
    netconvert: Path,
) -> dict[str, Any]:
    """Compatibility wrapper for the frozen LA/Nanchang v5 conversion."""

    return convert_cityflow(
        roadnet_path=roadnet_path,
        flow_path=flow_path,
        output_dir=output_dir,
        netconvert=netconvert,
        city="los_angeles",
        network="la_1x4",
        prefix="la_1x4",
    )


def _parse_nanchang_roadnet(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cursor = 0
    intersection_count = int(lines[cursor])
    cursor += 1
    intersections = {}
    for _ in range(intersection_count):
        latitude, longitude, node_id, signal_flag = lines[cursor].split()
        cursor += 1
        intersections[node_id] = {
            "id": node_id,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "controlled": int(signal_flag) == 1,
        }
    undirected_road_count = int(lines[cursor])
    cursor += 1
    edges = {}
    for _ in range(undirected_road_count):
        row = lines[cursor].split()
        cursor += 1
        if len(row) != 8:
            raise ValueError("invalid Nanchang road header")
        start, end, length, speed, forward_lanes, reverse_lanes, forward_id, reverse_id = row
        forward_turns = [int(value) for value in lines[cursor].split()]
        reverse_turns = [int(value) for value in lines[cursor + 1].split()]
        cursor += 2
        if len(forward_turns) != 3 * int(forward_lanes) or len(reverse_turns) != 3 * int(reverse_lanes):
            raise ValueError("invalid Nanchang lane-turn vector")
        edges[forward_id] = {
            "id": forward_id,
            "from": start,
            "to": end,
            "length": float(length),
            "speed": float(speed),
            "lanes": int(forward_lanes),
            "lane_turns": forward_turns,
            "inverse": reverse_id,
        }
        edges[reverse_id] = {
            "id": reverse_id,
            "from": end,
            "to": start,
            "length": float(length),
            "speed": float(speed),
            "lanes": int(reverse_lanes),
            "lane_turns": reverse_turns,
            "inverse": forward_id,
        }
    signal_count = int(lines[cursor])
    cursor += 1
    signal_rows = [lines[cursor + index].split() for index in range(signal_count)]
    cursor += signal_count
    if cursor != len(lines):
        raise ValueError("unparsed Nanchang roadnet rows remain")
    signal_nodes = {row[0] for row in signal_rows}
    controlled_nodes = {node for node, row in intersections.items() if row["controlled"]}
    if signal_nodes != controlled_nodes:
        raise ValueError("Nanchang signal section differs from intersection flags")
    return {
        "intersections": intersections,
        "edges": edges,
        "undirected_road_count": undirected_road_count,
        "signal_rows": signal_rows,
    }


def _parse_nanchang_flows(path: Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    flow_count = int(lines[0])
    if len(lines) != 1 + 3 * flow_count:
        raise ValueError("Nanchang flow row count changed")
    flows = []
    for index in range(flow_count):
        begin, end, period = [float(value) for value in lines[1 + 3 * index].split()]
        route_length = int(lines[2 + 3 * index])
        route = lines[3 + 3 * index].split()
        if len(route) != route_length or not (0.0 <= begin <= end) or period <= 0.0:
            raise ValueError(f"invalid Nanchang flow definition {index}")
        flows.append(
            {
                "begin": begin,
                "end": end,
                "period": period,
                "route": route,
            }
        )
    return flows


def _nanchang_explicit_departures(
    flows: list[dict[str, Any]],
) -> list[tuple[float, int, int]]:
    """Expand CBEngine's inclusive flow intervals into sorted SUMO departures."""

    departures: list[tuple[float, int, int]] = []
    for flow_index, flow in enumerate(flows):
        begin = float(flow["begin"])
        end = float(flow["end"])
        period = float(flow["period"])
        departure_count = int(math.floor((end - begin) / period + 1e-9)) + 1
        for departure_index in range(departure_count):
            departure = begin + departure_index * period
            if departure > end + 1e-7:
                raise ValueError(
                    "expanded Nanchang departure exceeds its CBEngine flow end: "
                    f"flow={flow_index} departure={departure} end={end}"
                )
            departures.append((departure, flow_index, departure_index))
    departures.sort(key=lambda row: (row[0], row[1], row[2]))
    return departures


def _nanchang_geometric_movement(
    *,
    incoming_edge: dict[str, Any],
    outgoing_edge: dict[str, Any],
    intersections: dict[str, dict[str, Any]],
) -> int:
    upstream = intersections[str(incoming_edge["from"])]
    junction = intersections[str(incoming_edge["to"])]
    downstream = intersections[str(outgoing_edge["to"])]
    longitude_scale = math.cos(math.radians(float(junction["latitude"])))
    incoming_heading = math.atan2(
        float(junction["latitude"]) - float(upstream["latitude"]),
        (float(junction["longitude"]) - float(upstream["longitude"]))
        * longitude_scale,
    )
    outgoing_heading = math.atan2(
        float(downstream["latitude"]) - float(junction["latitude"]),
        (float(downstream["longitude"]) - float(junction["longitude"]))
        * longitude_scale,
    )
    delta = (outgoing_heading - incoming_heading + math.pi) % (2.0 * math.pi) - math.pi
    if delta > math.pi / 4.0:
        return 0
    if delta < -math.pi / 4.0:
        return 2
    return 1


def _nanchang_connection_plan(
    *,
    roadnet: dict[str, Any],
    flows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    intersections = roadnet["intersections"]
    edges = roadnet["edges"]
    signal_approaches = {
        str(row[0]): tuple(str(value) for value in row[1:])
        for row in roadnet["signal_rows"]
    }
    incoming_by_node: dict[str, list[str]] = {}
    outgoing_by_node: dict[str, list[str]] = {}
    for edge_id, edge in edges.items():
        incoming_by_node.setdefault(str(edge["to"]), []).append(str(edge_id))
        outgoing_by_node.setdefault(str(edge["from"]), []).append(str(edge_id))

    connections: list[dict[str, str]] = []
    connection_identities: set[tuple[str, str, int, int]] = set()
    connected_edge_pairs: set[tuple[str, str]] = set()
    movement_counts = {"left": 0, "straight": 0, "right": 0}
    controlled_connection_count = 0
    terminal_incoming_edge_count = 0
    movement_names = ("left", "straight", "right")

    for node_id in sorted(intersections):
        controlled = node_id in signal_approaches
        approaches = signal_approaches.get(node_id)
        if approaches is not None:
            if len(approaches) != 4:
                raise ValueError(f"Nanchang signal {node_id} does not have four approach slots")
            for edge_id in approaches:
                if edge_id == "-1":
                    continue
                if edge_id not in edges or str(edges[edge_id]["from"]) != node_id:
                    raise ValueError(
                        f"Nanchang signal {node_id} has invalid outgoing approach {edge_id}"
                    )

        for incoming_id in sorted(incoming_by_node.get(node_id, ())):
            incoming = edges[incoming_id]
            targets: dict[int, list[str]] = {0: [], 1: [], 2: []}
            if approaches is not None:
                inverse_id = str(incoming["inverse"])
                if inverse_id not in approaches:
                    raise ValueError(
                        f"Nanchang signal {node_id} omits incoming inverse {inverse_id}"
                    )
                approach_index = approaches.index(inverse_id)
                for movement, target_index in (
                    (0, (approach_index + 1) % 4),
                    (1, (approach_index + 2) % 4),
                    (2, (approach_index - 1) % 4),
                ):
                    target_id = approaches[target_index]
                    if target_id != "-1":
                        targets[movement].append(target_id)
            else:
                for outgoing_id in sorted(outgoing_by_node.get(node_id, ())):
                    if outgoing_id == str(incoming["inverse"]):
                        continue
                    movement = _nanchang_geometric_movement(
                        incoming_edge=incoming,
                        outgoing_edge=edges[outgoing_id],
                        intersections=intersections,
                    )
                    targets[movement].append(outgoing_id)

            if not any(targets.values()):
                terminal_incoming_edge_count += 1
            lane_count = int(incoming["lanes"])
            turn_flags = list(incoming["lane_turns"])
            for cb_lane_index in range(lane_count):
                sumo_from_lane = lane_count - 1 - cb_lane_index
                flags = turn_flags[3 * cb_lane_index : 3 * cb_lane_index + 3]
                for movement, permitted in enumerate(flags):
                    if not permitted:
                        continue
                    for outgoing_id in targets[movement]:
                        outgoing_lanes = int(edges[outgoing_id]["lanes"])
                        if movement == 0:
                            sumo_to_lane = outgoing_lanes - 1
                        elif movement == 1:
                            sumo_to_lane = outgoing_lanes // 2
                        else:
                            sumo_to_lane = 0
                        identity = (
                            incoming_id,
                            outgoing_id,
                            sumo_from_lane,
                            sumo_to_lane,
                        )
                        if identity in connection_identities:
                            raise ValueError(f"duplicate Nanchang connection: {identity}")
                        connection_identities.add(identity)
                        connected_edge_pairs.add((incoming_id, outgoing_id))
                        connections.append(
                            {
                                "from": incoming_id,
                                "to": outgoing_id,
                                "fromLane": str(sumo_from_lane),
                                "toLane": str(sumo_to_lane),
                            }
                        )
                        movement_counts[movement_names[movement]] += 1
                        controlled_connection_count += int(controlled)

    route_transition_count = 0
    route_edge_pairs: set[tuple[str, str]] = set()
    missing_route_edge_pairs: set[tuple[str, str]] = set()
    for flow in flows:
        for first, second in zip(flow["route"], flow["route"][1:]):
            pair = (str(first), str(second))
            route_transition_count += 1
            route_edge_pairs.add(pair)
            if pair not in connected_edge_pairs:
                missing_route_edge_pairs.add(pair)
    if missing_route_edge_pairs:
        examples = sorted(missing_route_edge_pairs)[:12]
        raise ValueError(
            "Nanchang lane-turn conversion disconnects flow routes: "
            f"missing={len(missing_route_edge_pairs)} examples={examples}"
        )
    return connections, {
        "protocol": "cbengine-left-straight-right-lane-flags-plus-signal-approach-order-v1",
        "connection_count": len(connections),
        "controlled_connection_count": controlled_connection_count,
        "movement_connection_counts": movement_counts,
        "terminal_incoming_edge_count": terminal_incoming_edge_count,
        "route_transition_count": route_transition_count,
        "unique_route_edge_pair_count": len(route_edge_pairs),
        "missing_route_edge_pair_count": len(missing_route_edge_pairs),
    }


def convert_nanchang(
    *,
    roadnet_path: Path,
    flow_path: Path,
    output_dir: Path,
    netconvert: Path,
) -> dict[str, Any]:
    roadnet = _parse_nanchang_roadnet(roadnet_path)
    flows = _parse_nanchang_flows(flow_path)
    intersections = roadnet["intersections"]
    edges = roadnet["edges"]
    edge_endpoints = {
        edge_id: (str(edge["from"]), str(edge["to"]))
        for edge_id, edge in edges.items()
    }
    route_audit = _validate_edge_routes(
        edge_endpoints,
        (flow["route"] for flow in flows),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix = "nanchang_full"
    node_path = output_dir / f"{prefix}.nod.xml"
    edge_path = output_dir / f"{prefix}.edg.xml"
    connection_path = output_dir / f"{prefix}.con.xml"
    net_path = output_dir / f"{prefix}.net.xml"
    route_path = output_dir / f"{prefix}.rou.xml"
    cfg_path = output_dir / f"{prefix}.sumocfg"
    mean_latitude = float(
        sum(row["latitude"] for row in intersections.values()) / len(intersections)
    )
    mean_longitude = float(
        sum(row["longitude"] for row in intersections.values()) / len(intersections)
    )
    longitude_scale = 111320.0 * math.cos(math.radians(mean_latitude))
    latitude_scale = 110540.0
    projected = {
        node_id: (
            (row["longitude"] - mean_longitude) * longitude_scale,
            (row["latitude"] - mean_latitude) * latitude_scale,
        )
        for node_id, row in intersections.items()
    }
    nodes_root = ET.Element("nodes")
    for node_id, row in intersections.items():
        x, y = projected[node_id]
        ET.SubElement(
            nodes_root,
            "node",
            id=node_id,
            x=f"{x:.9f}",
            y=f"{y:.9f}",
            type="traffic_light" if row["controlled"] else "priority",
        )
    _write_xml(node_path, nodes_root)
    edges_root = ET.Element("edges")
    for edge_id, edge in edges.items():
        ET.SubElement(
            edges_root,
            "edge",
            id=edge_id,
            **{
                "from": str(edge["from"]),
                "to": str(edge["to"]),
                "numLanes": str(edge["lanes"]),
                "speed": f"{edge['speed']:.9f}",
                "length": f"{edge['length']:.9f}",
                "priority": "1",
            },
        )
    _write_xml(edge_path, edges_root)
    connection_plan, connection_audit = _nanchang_connection_plan(
        roadnet=roadnet,
        flows=flows,
    )
    connections_root = ET.Element("connections")
    for connection in connection_plan:
        ET.SubElement(connections_root, "connection", connection)
    _write_xml(connection_path, connections_root)
    routes_root = ET.Element("routes")
    ET.SubElement(
        routes_root,
        "vType",
        id="passenger",
        vClass="passenger",
        length="5.0",
        minGap="2.5",
        maxSpeed="22.222222222",
        accel="2.0",
        decel="4.5",
        sigma="0.5",
    )
    for index, flow in enumerate(flows):
        route_id = f"nanchang_route_{index}"
        ET.SubElement(routes_root, "route", id=route_id, edges=" ".join(flow["route"]))
    explicit_departures = _nanchang_explicit_departures(flows)
    for departure, flow_index, departure_index in explicit_departures:
        ET.SubElement(
            routes_root,
            "vehicle",
            id=f"nanchang_{flow_index}_{departure_index}",
            type="passenger",
            route=f"nanchang_route_{flow_index}",
            depart=f"{departure:.12g}",
            departLane="best",
            departSpeed="max",
        )
    _write_xml(route_path, routes_root)
    netconvert_result = _run_netconvert(
        netconvert,
        node_file=node_path,
        edge_file=edge_path,
        output_file=net_path,
        connection_file=connection_path,
        artifact_root=output_dir.parent,
    )
    _write_sumocfg(cfg_path, net_file=net_path, route_file=route_path, end=3600.0)
    return {
        "city": "nanchang",
        "network": "nanchang_full",
        "source_format": "CBEngine text",
        "intersection_count": len(intersections),
        "controlled_intersection_count": sum(
            int(row["controlled"]) for row in intersections.values()
        ),
        "undirected_road_count": roadnet["undirected_road_count"],
        "directed_edge_count": len(edges),
        "flow_definition_count": len(flows),
        "demand_encoding": "explicit_globally_departure_sorted_vehicles",
        "vehicle_count": len(explicit_departures),
        "first_departure_sec": explicit_departures[0][0],
        "last_departure_sec": explicit_departures[-1][0],
        "source_flow_begin_nondecreasing": all(
            float(first["begin"]) <= float(second["begin"])
            for first, second in zip(flows, flows[1:])
        ),
        "route_audit": route_audit,
        "connection_audit": connection_audit,
        "netconvert": netconvert_result,
        "projection": {
            "method": "local_equirectangular",
            "origin_latitude": mean_latitude,
            "origin_longitude": mean_longitude,
        },
        "sumocfg": _manifest_path(cfg_path, artifact_root=output_dir.parent),
    }


def convert_all(
    *,
    la_roadnet: Path,
    la_flow: Path,
    nanchang_roadnet: Path,
    nanchang_flow: Path,
    output_root: Path,
    netconvert: Path,
) -> dict[str, Any]:
    for path in (la_roadnet, la_flow, nanchang_roadnet, nanchang_flow, netconvert):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite external conversion root: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        la = convert_la(
            roadnet_path=la_roadnet,
            flow_path=la_flow,
            output_dir=staging / "la_1x4",
            netconvert=netconvert,
        )
        nanchang = convert_nanchang(
            roadnet_path=nanchang_roadnet,
            flow_path=nanchang_flow,
            output_dir=staging / "nanchang_full",
            netconvert=netconvert,
        )
        payload = {
            "protocol": "cfcmt-libsignal-full-external-sumo-conversion-v5",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "netconvert": str(netconvert.resolve()),
            "source_files": {
                "la_flow": {"path": str(la_flow.resolve()), "sha256": _sha256(la_flow)},
                "la_roadnet": {"path": str(la_roadnet.resolve()), "sha256": _sha256(la_roadnet)},
                "nanchang_flow": {"path": str(nanchang_flow.resolve()), "sha256": _sha256(nanchang_flow)},
                "nanchang_roadnet": {"path": str(nanchang_roadnet.resolve()), "sha256": _sha256(nanchang_roadnet)},
            },
            "networks": {"la_1x4": la, "nanchang_full": nanchang},
        }
        _atomic_json(staging / "conversion_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la-roadnet", type=Path, required=True)
    parser.add_argument("--la-flow", type=Path, required=True)
    parser.add_argument("--nanchang-roadnet", type=Path, required=True)
    parser.add_argument("--nanchang-flow", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--netconvert", type=Path, default=Path("/usr/bin/netconvert"))
    args = parser.parse_args()
    payload = convert_all(
        la_roadnet=args.la_roadnet,
        la_flow=args.la_flow,
        nanchang_roadnet=args.nanchang_roadnet,
        nanchang_flow=args.nanchang_flow,
        output_root=args.output_root,
        netconvert=args.netconvert,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Results saved to: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
