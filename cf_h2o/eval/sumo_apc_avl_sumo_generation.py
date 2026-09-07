"""Stage 2 SUMO network and route generation for APC/AVL benchmarks.

This stage consumes the Stage 1 line manifest, generates SUMO nodes, edges,
routes, explicit bus-stop dwell points, and a runnable configuration per city,
then optionally runs SUMO as a smoke validation. It does not generate APC/AVL
learning snapshots yet.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_LINE_MANIFEST = Path("H2Oplus/downloads/sumo_apc_avl_benchmark/inputs/sumo_apc_avl_line_inputs_full.jsonl")
DEFAULT_OUT_ROOT = Path("H2Oplus/downloads/sumo_apc_avl_benchmark/sumo")
DEFAULT_REPORT = Path("cf_h2o/results/sumo_apc_avl_sumo_generation_smoke.json")
DEFAULT_MD_REPORT = Path("cf_h2o/results/sumo_apc_avl_sumo_generation_smoke.md")
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class SegmentRecord:
    edge_id: str
    start_node: str
    end_node: str
    length_m: float
    speed_mps: float
    start_stop: str
    end_stop: str


@dataclass(frozen=True)
class LineBuild:
    line_key: str
    line_uid: str
    route_id: str
    edges: list[str]
    bus_stops: list[str]
    departures: list[float]
    stops: int
    segments: int
    distance_m: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_id(value: str, prefix: str = "id") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:120]


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _read_line_tables(line_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_excel(line_dir / "stop_news.xlsx"),
        pd.read_excel(line_dir / "route_news.xlsx"),
        pd.read_excel(line_dir / "time_table.xlsx"),
    )


def _city_origin(stops_by_line: list[pd.DataFrame]) -> tuple[float, float]:
    lat_values = []
    lon_values = []
    for stops in stops_by_line:
        lat = pd.to_numeric(stops.get("latitude"), errors="coerce")
        lon = pd.to_numeric(stops.get("longitude"), errors="coerce")
        valid = lat.between(-90.0, 90.0, inclusive="both") & lon.between(-180.0, 180.0, inclusive="both")
        lat_values.extend(lat[valid].astype(float).tolist())
        lon_values.extend(lon[valid].astype(float).tolist())
    if not lat_values or not lon_values:
        return 0.0, 0.0
    return float(np.median(lat_values)), float(np.median(lon_values))


def _latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    origin_lat_rad = math.radians(origin_lat)
    x = EARTH_RADIUS_M * math.radians(lon - origin_lon) * math.cos(origin_lat_rad)
    y = EARTH_RADIUS_M * (lat_rad - origin_lat_rad)
    return x, y


def _speed_for_segment(row: pd.Series) -> float:
    hourly_cols = [f"{hour:02d}:00:00" for hour in range(24) if f"{hour:02d}:00:00" in row.index]
    values = pd.to_numeric(row[hourly_cols], errors="coerce").to_numpy(dtype=np.float64) if hourly_cols else np.array([])
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values):
        return float(np.clip(np.median(values), 1.0, 25.0))
    fallback = pd.to_numeric(pd.Series([row.get("V_max", 8.0)]), errors="coerce").iloc[0]
    if not math.isfinite(float(fallback)) or float(fallback) <= 0.0:
        return 8.0
    return float(np.clip(float(fallback), 1.0, 25.0))


def _valid_departures(timetable: pd.DataFrame, max_departures: int) -> list[float]:
    if "launch_time" not in timetable.columns:
        return []
    launch = pd.to_numeric(timetable["launch_time"], errors="coerce")
    launch = launch[np.isfinite(launch) & (launch >= 0.0) & (launch < 86400.0)].astype(float)
    departures = sorted(float(value) for value in launch.tolist())
    if max_departures > 0:
        departures = departures[:max_departures]
    return departures


def _selected_lines(line_manifest: list[dict[str, Any]], city_key: str, max_lines: int) -> list[dict[str, Any]]:
    selected = [line for line in line_manifest if line.get("city_key") == city_key and line.get("ok", False)]
    selected = sorted(selected, key=lambda row: str(row["line_key"]))
    if max_lines > 0:
        return selected[:max_lines]
    return selected


def build_city_sumo(
    *,
    city_key: str,
    line_manifest: list[dict[str, Any]],
    out_dir: Path,
    max_lines: int,
    max_departures_per_line: int,
    duration_sec: float,
    include_bus_stops: bool = True,
    base_dwell_sec: float = 8.0,
) -> dict[str, Any]:
    """Generate plain XML, net.xml, routes, config, and metadata for one city."""

    selected = _selected_lines(line_manifest, city_key, max_lines)
    if not selected:
        raise RuntimeError(f"no ready lines selected for {city_key}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_tables: list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for line in selected:
        stops, route, timetable = _read_line_tables(Path(line["line_dir"]))
        raw_tables.append((line, stops, route, timetable))

    origin_lat, origin_lon = _city_origin([item[1] for item in raw_tables])
    nodes_root = ET.Element("nodes")
    edges_root = ET.Element("edges")
    connections_root = ET.Element("connections")
    routes_root = ET.Element("routes")
    additional_root = ET.Element("additional")
    ET.SubElement(
        routes_root,
        "vType",
        id="bus",
        vClass="bus",
        guiShape="bus",
        length="12.0",
        accel="1.0",
        decel="4.5",
        sigma="0.5",
        maxSpeed="25.0",
    )

    lines_built: list[LineBuild] = []
    segment_records: list[dict[str, Any]] = []
    vehicle_specs: list[dict[str, str | float]] = []
    all_departures: list[float] = []
    for line_idx, (line, stops, route, timetable) in enumerate(raw_tables):
        line_uid = _safe_id(f"{city_key}_{line_idx:05d}_{line['line_key']}", "line")
        stop_names = stops["stop_name"].astype(str).tolist()
        node_ids: list[str] = []
        for stop_idx, stop in stops.reset_index(drop=True).iterrows():
            node_id = f"n_{line_uid}_{int(stop_idx):04d}"
            lat = float(pd.to_numeric(pd.Series([stop.get("latitude")]), errors="coerce").fillna(origin_lat).iloc[0])
            lon = float(pd.to_numeric(pd.Series([stop.get("longitude")]), errors="coerce").fillna(origin_lon).iloc[0])
            x, y = _latlon_to_xy(lat, lon, origin_lat, origin_lon)
            ET.SubElement(nodes_root, "node", id=node_id, x=f"{x:.3f}", y=f"{y:.3f}", type="priority")
            node_ids.append(node_id)

        edge_ids: list[str] = []
        bus_stop_ids: list[str] = []
        distance_total = 0.0
        for seg_idx, row in route.reset_index(drop=True).iterrows():
            if int(seg_idx) + 1 >= len(node_ids):
                break
            edge_id = f"e_{line_uid}_{int(seg_idx):04d}"
            length_m = float(pd.to_numeric(pd.Series([row.get("distance")]), errors="coerce").fillna(0.0).iloc[0])
            length_m = float(np.clip(length_m, 1.0, 100_000.0))
            speed_mps = _speed_for_segment(row)
            ET.SubElement(
                edges_root,
                "edge",
                id=edge_id,
                **{
                    "from": node_ids[int(seg_idx)],
                    "to": node_ids[int(seg_idx) + 1],
                    "priority": "1",
                    "numLanes": "1",
                    "speed": f"{speed_mps:.3f}",
                    "length": f"{length_m:.3f}",
                    "allow": "bus",
                },
            )
            edge_ids.append(edge_id)
            distance_total += length_m
            if include_bus_stops:
                bus_stop_id = f"bs_{line_uid}_{int(seg_idx):04d}"
                stop_end = min(max(0.5, length_m - 0.1), 12.0)
                ET.SubElement(
                    additional_root,
                    "busStop",
                    id=bus_stop_id,
                    lane=f"{edge_id}_0",
                    startPos="0.0",
                    endPos=f"{stop_end:.3f}",
                    friendlyPos="true",
                )
                bus_stop_ids.append(bus_stop_id)
            segment_records.append(
                {
                    "city_key": city_key,
                    "line_key": line["line_key"],
                    "line_uid": line_uid,
                    "edge_id": edge_id,
                    "segment_idx": int(seg_idx),
                    "start_node": node_ids[int(seg_idx)],
                    "end_node": node_ids[int(seg_idx) + 1],
                    "start_stop": str(row.get("start_stop", stop_names[int(seg_idx)] if int(seg_idx) < len(stop_names) else "")),
                    "end_stop": str(row.get("end_stop", stop_names[int(seg_idx) + 1] if int(seg_idx) + 1 < len(stop_names) else "")),
                    "length_m": length_m,
                    "speed_mps": speed_mps,
                }
            )

        if not edge_ids:
            continue
        for from_edge, to_edge in zip(edge_ids[:-1], edge_ids[1:]):
            ET.SubElement(connections_root, "connection", **{"from": from_edge, "to": to_edge, "fromLane": "0", "toLane": "0"})
        route_id = f"r_{line_uid}"
        ET.SubElement(routes_root, "route", id=route_id, edges=" ".join(edge_ids))
        departures = _valid_departures(timetable, max_departures_per_line)
        for dep_idx, depart in enumerate(departures):
            vehicle_specs.append(
                {
                    "id": f"veh_{line_uid}_{dep_idx:06d}",
                    "type": "bus",
                    "route": route_id,
                    "depart": float(depart),
                    "departLane": "best",
                    "departSpeed": "max",
                    "bus_stops": bus_stop_ids,
                }
            )
        all_departures.extend(departures)
        lines_built.append(
            LineBuild(
                line_key=str(line["line_key"]),
                line_uid=line_uid,
                route_id=route_id,
                edges=edge_ids,
                bus_stops=bus_stop_ids,
                departures=departures,
                stops=len(stops),
                segments=len(edge_ids),
                distance_m=distance_total,
            )
        )

    if not lines_built:
        raise RuntimeError(f"no SUMO routes built for {city_key}")
    if not all_departures:
        raise RuntimeError(f"no SUMO vehicles built for {city_key}")

    for vehicle in sorted(vehicle_specs, key=lambda item: (float(item["depart"]), str(item["id"]))):
        vehicle_el = ET.SubElement(
            routes_root,
            "vehicle",
            id=str(vehicle["id"]),
            type=str(vehicle["type"]),
            route=str(vehicle["route"]),
            depart=f"{float(vehicle['depart']):.1f}",
            departLane=str(vehicle["departLane"]),
            departSpeed=str(vehicle["departSpeed"]),
        )
        for bus_stop_id in vehicle.get("bus_stops", []):
            ET.SubElement(
                vehicle_el,
                "stop",
                busStop=str(bus_stop_id),
                duration=f"{max(0.0, float(base_dwell_sec)):.1f}",
            )

    begin = float(min(all_departures))
    end = float(min(86400.0, begin + max(1.0, float(duration_sec))))
    nodes_path = out_dir / "nodes.nod.xml"
    edges_path = out_dir / "edges.edg.xml"
    connections_path = out_dir / "connections.con.xml"
    additional_path = out_dir / "stops.add.xml"
    routes_path = out_dir / "routes.rou.xml"
    net_path = out_dir / "net.net.xml"
    sumocfg_path = out_dir / "simulation.sumocfg"
    summary_path = out_dir / "summary.xml"
    tripinfo_path = out_dir / "tripinfo.xml"
    segment_map_path = out_dir / "segment_map.jsonl"
    line_map_path = out_dir / "line_map.json"

    _write_xml(nodes_path, nodes_root)
    _write_xml(edges_path, edges_root)
    _write_xml(connections_path, connections_root)
    if include_bus_stops:
        _write_xml(additional_path, additional_root)
    _write_xml(routes_path, routes_root)
    segment_map_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in segment_records),
        encoding="utf-8",
    )
    line_map = [
        {
            "line_key": item.line_key,
            "line_uid": item.line_uid,
            "route_id": item.route_id,
            "edges": item.edges,
            "bus_stops": item.bus_stops,
            "departures": item.departures,
            "stops": item.stops,
            "segments": item.segments,
            "distance_m": item.distance_m,
        }
        for item in lines_built
    ]
    line_map_path.write_text(json.dumps(line_map, indent=2) + "\n", encoding="utf-8")

    netconvert = shutil.which("netconvert")
    if not netconvert:
        raise RuntimeError("netconvert not found on PATH")
    netconvert_cmd = [
        netconvert,
        "--node-files",
        str(nodes_path),
        "--edge-files",
        str(edges_path),
        "--connection-files",
        str(connections_path),
        "--output-file",
        str(net_path),
        "--no-turnarounds",
        "true",
        "--geometry.remove",
        "false",
        "--junctions.join",
        "false",
    ]
    netconvert_run = subprocess.run(netconvert_cmd, cwd=out_dir, text=True, capture_output=True, check=False)
    if netconvert_run.returncode != 0:
        raise RuntimeError(
            f"netconvert failed for {city_key}: {netconvert_run.stderr[-2000:] or netconvert_run.stdout[-2000:]}"
        )

    cfg_root = ET.Element("configuration")
    input_el = ET.SubElement(cfg_root, "input")
    ET.SubElement(input_el, "net-file", value=net_path.name)
    ET.SubElement(input_el, "route-files", value=routes_path.name)
    if include_bus_stops:
        ET.SubElement(input_el, "additional-files", value=additional_path.name)
    time_el = ET.SubElement(cfg_root, "time")
    ET.SubElement(time_el, "begin", value=f"{begin:.1f}")
    ET.SubElement(time_el, "end", value=f"{end:.1f}")
    output_el = ET.SubElement(cfg_root, "output")
    ET.SubElement(output_el, "summary-output", value=summary_path.name)
    ET.SubElement(output_el, "tripinfo-output", value=tripinfo_path.name)
    ET.SubElement(output_el, "tripinfo-output.write-unfinished", value="true")
    report_el = ET.SubElement(cfg_root, "report")
    ET.SubElement(report_el, "no-step-log", value="true")
    ET.SubElement(report_el, "duration-log.disable", value="true")
    _write_xml(sumocfg_path, cfg_root)

    vehicles_in_window = sum(1 for depart in all_departures if begin <= depart < end)
    return {
        "city_key": city_key,
        "city": selected[0].get("city", city_key),
        "out_dir": str(out_dir),
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "connections_path": str(connections_path),
        "additional_path": str(additional_path) if include_bus_stops else None,
        "routes_path": str(routes_path),
        "net_path": str(net_path),
        "sumocfg_path": str(sumocfg_path),
        "summary_path": str(summary_path),
        "tripinfo_path": str(tripinfo_path),
        "segment_map_path": str(segment_map_path),
        "line_map_path": str(line_map_path),
        "lines_selected": len(selected),
        "lines_built": len(lines_built),
        "nodes": sum(item.stops for item in lines_built),
        "edges": sum(item.segments for item in lines_built),
        "bus_stops": sum(len(item.bus_stops) for item in lines_built),
        "vehicles": len(all_departures),
        "vehicles_in_smoke_window": int(vehicles_in_window),
        "begin": begin,
        "end": end,
        "duration_sec": end - begin,
        "include_bus_stops": bool(include_bus_stops),
        "base_dwell_sec": float(base_dwell_sec),
        "distance_km": float(sum(item.distance_m for item in lines_built) / 1000.0),
        "netconvert": {
            "returncode": netconvert_run.returncode,
            "stdout_tail": netconvert_run.stdout[-2000:],
            "stderr_tail": netconvert_run.stderr[-2000:],
        },
    }


def run_sumo_smoke(city_result: dict[str, Any]) -> dict[str, Any]:
    sumo = shutil.which("sumo")
    if not sumo:
        return {"ok": False, "error": "sumo not found on PATH"}
    cmd = [sumo, "-c", city_result["sumocfg_path"]]
    started = time.time()
    proc = subprocess.run(cmd, cwd=city_result["out_dir"], text=True, capture_output=True, check=False)
    elapsed = time.time() - started
    trip_count = 0
    tripinfo_path = Path(city_result["tripinfo_path"])
    if tripinfo_path.exists():
        try:
            trip_root = ET.parse(tripinfo_path).getroot()
            trip_count = len(trip_root.findall("tripinfo"))
        except ET.ParseError:
            trip_count = 0
    return {
        "ok": proc.returncode == 0 and trip_count > 0,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "tripinfo_count": trip_count,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _markdown_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["ok"] else "FAIL"
    lines = [
        "# SUMO APC/AVL Stage 2 SUMO Generation",
        "",
        f"Verdict: **{verdict}**",
        "",
        "This stage generates SUMO nodes, edges, routes, explicit bus-stop dwell points, and config files from the validated Stage 1 manifest.",
        "",
        "| City | Lines | Edges | Bus stops | Vehicles | Smoke trips | Window | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for city in result["cities"].values():
        smoke = city.get("sumo_smoke", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(city["city"]),
                    str(city["lines_built"]),
                    str(city["edges"]),
                    str(city.get("bus_stops", 0)),
                    str(city["vehicles"]),
                    str(smoke.get("tripinfo_count", "n/a")),
                    f"{city['begin']:.0f}-{city['end']:.0f}",
                    "PASS" if city["ok"] else "FAIL",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Artifacts", ""])
    for city in result["cities"].values():
        lines.append(f"- {city['city']}: `{city['out_dir']}`")
    return "\n".join(lines).rstrip() + "\n"


def _write_report(result: dict[str, Any], out: Path, md_out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    result["result_path"] = str(out)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_markdown_report(result), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    manifest_path = _resolve_path(root, args.line_manifest)
    line_manifest = _read_jsonl(manifest_path)
    city_keys = list(args.cities) if args.cities else sorted({str(line["city_key"]) for line in line_manifest if line.get("ok")})
    suffix = "full" if int(args.max_lines_per_city) <= 0 else f"max{int(args.max_lines_per_city)}"
    out_root = _resolve_path(root, args.out_root)
    cities: dict[str, Any] = {}
    for city_key in city_keys:
        city_out = out_root / city_key / suffix
        print(f"[stage2] generating SUMO for {city_key} -> {city_out}", flush=True)
        city_result = build_city_sumo(
            city_key=city_key,
            line_manifest=line_manifest,
            out_dir=city_out,
            max_lines=int(args.max_lines_per_city),
            max_departures_per_line=int(args.max_departures_per_line),
            duration_sec=float(args.duration_sec),
            include_bus_stops=bool(args.include_bus_stops),
            base_dwell_sec=float(args.base_dwell_sec),
        )
        smoke = {"ok": True, "skipped": True}
        if args.run_sumo:
            smoke = run_sumo_smoke(city_result)
            print(
                f"[stage2] {city_key} smoke ok={smoke['ok']} trips={smoke.get('tripinfo_count', 0)}",
                flush=True,
            )
        city_result["sumo_smoke"] = smoke
        city_result["ok"] = bool(city_result["vehicles_in_smoke_window"] > 0 and smoke.get("ok", False))
        cities[city_key] = city_result
    result = {
        "ok": all(city["ok"] for city in cities.values()),
        "stage": "sumo_apc_avl_stage2_sumo_generation",
        "performance_validation": False,
        "sumo_generation": True,
        "line_manifest_path": str(manifest_path),
        "out_root": str(out_root),
        "max_lines_per_city": int(args.max_lines_per_city),
        "max_departures_per_line": int(args.max_departures_per_line),
        "include_bus_stops": bool(args.include_bus_stops),
        "base_dwell_sec": float(args.base_dwell_sec),
        "run_sumo": bool(args.run_sumo),
        "cities": cities,
    }
    _write_report(result, _resolve_path(root, args.out), _resolve_path(root, args.md_out))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-manifest", type=Path, default=DEFAULT_LINE_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--max-lines-per-city", type=int, default=2)
    parser.add_argument("--max-departures-per-line", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=1800.0)
    parser.add_argument("--include-bus-stops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-dwell-sec", type=float, default=8.0)
    parser.add_argument("--run-sumo", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
