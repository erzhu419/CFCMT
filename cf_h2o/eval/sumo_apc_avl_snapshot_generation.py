"""Stage 3 SUMO-driven AVL/APC snapshot generation.

This stage runs the Stage 2 SUMO scenarios with libsumo, records vehicle-level
AVL snapshots, and overlays deterministic APC board/alight/occupancy events
from the original line-level OD matrices. It is a controlled benchmark
generator, not a calibrated passenger simulation.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures as futures
import json
import math
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_STAGE2_REPORT = Path("cf_h2o/results/sumo_apc_avl_sumo_generation_smoke.json")
DEFAULT_LINE_MANIFEST = Path("H2Oplus/downloads/sumo_apc_avl_benchmark/inputs/sumo_apc_avl_line_inputs_full.jsonl")
DEFAULT_OUT = Path("cf_h2o/results/sumo_apc_avl_snapshot_generation_smoke.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/sumo_apc_avl_snapshot_generation_smoke.md")
CONTROL_COLUMNS = {"time_period", "stop_name"}


@dataclass
class LineDemand:
    line_key: str
    line_uid: str
    stops: list[str]
    hourly_od: dict[int, pd.DataFrame]
    departures_by_hour: dict[int, int]
    boarding_plan: dict[int, dict[int, list[tuple[str, float]]]]
    final_stop: str


@dataclass
class VehicleAPCState:
    vehicle_id: str
    line_key: str
    line_uid: str
    depart: float
    demand_hour: int
    last_edge: str | None = None
    onboard_by_dest: dict[str, float] = field(default_factory=dict)
    cumulative_boardings: float = 0.0
    cumulative_alightings: float = 0.0
    last_boardings: float = 0.0
    last_alightings: float = 0.0

    @property
    def occupancy(self) -> float:
        return float(sum(self.onboard_by_dest.values()))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _load_libsumo():
    try:
        import libsumo  # type: ignore

        return libsumo
    except ImportError:
        sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
        tools_path = Path(sumo_home) / "tools"
        sys.path.insert(0, str(tools_path))
        import libsumo  # type: ignore

        return libsumo


def _hour_from_time_period(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "hour"):
        try:
            hour = int(value.hour)
            return hour if 0 <= hour <= 23 else None
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    head = text.split(":", 1)[0]
    if not head.lstrip("-").isdigit():
        return None
    hour = int(head)
    return hour if 0 <= hour <= 23 else None


def _vehicle_line_uid(vehicle_id: str) -> str:
    match = re.match(r"veh_(.+)_\d{6}$", vehicle_id)
    if not match:
        return ""
    return match.group(1)


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _load_segment_map(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_edge: dict[str, dict[str, Any]] = {}
    by_line: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        by_edge[str(row["edge_id"])] = row
        by_line.setdefault(str(row["line_uid"]), []).append(row)
    for rows in by_line.values():
        rows.sort(key=lambda item: int(item["segment_idx"]))
    return by_edge, by_line


def _load_vehicle_departures(line_map: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    vehicles: dict[str, dict[str, Any]] = {}
    for line in line_map:
        line_uid = str(line["line_uid"])
        for dep_idx, depart in enumerate(line.get("departures", [])):
            vehicle_id = f"veh_{line_uid}_{dep_idx:06d}"
            vehicles[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "line_uid": line_uid,
                "line_key": str(line["line_key"]),
                "depart": float(depart),
                "demand_hour": int(float(depart) // 3600) % 24,
            }
    return vehicles


def _line_manifest_lookup(line_manifest: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(line["city_key"]), str(line["line_key"])): line
        for line in line_manifest
        if line.get("ok", False)
    }


def _hourly_od(passenger_od: pd.DataFrame, ordered_stops: list[str], required_hours: set[int] | None = None) -> dict[int, pd.DataFrame]:
    value_cols = [column for column in passenger_od.columns if column not in CONTROL_COLUMNS]
    if not value_cols:
        return {}
    unique_stops = _unique_preserve_order([str(stop) for stop in ordered_stops])
    od = passenger_od.copy()
    od["_hour"] = od["time_period"].map(_hour_from_time_period) if "time_period" in od.columns else None
    if required_hours is not None:
        od = od[od["_hour"].isin(required_hours)]
        if od.empty:
            return {}
    od["stop_name"] = od["stop_name"].astype(str)
    value_cols = _unique_preserve_order([str(column) for column in value_cols if str(column) in set(unique_stops)])
    out: dict[int, pd.DataFrame] = {}
    for hour, group in od.dropna(subset=["_hour"]).groupby("_hour"):
        matrix = group[["stop_name"] + value_cols].copy()
        matrix[value_cols] = matrix[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        matrix = matrix.groupby("stop_name")[value_cols].sum()
        matrix = matrix.reindex(index=unique_stops, columns=unique_stops, fill_value=0.0)
        out[int(hour)] = matrix
    return out


def _matrix_value(matrix: pd.DataFrame, origin: str, dest: str) -> float:
    if origin not in matrix.index or dest not in matrix.columns:
        return 0.0
    value = matrix.loc[origin, dest]
    if isinstance(value, pd.Series):
        return float(pd.to_numeric(value, errors="coerce").fillna(0.0).sum())
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])


def _build_boarding_plan(
    hourly_od: dict[int, pd.DataFrame],
    ordered_stops: list[str],
    departures_by_hour: dict[int, int],
) -> dict[int, dict[int, list[tuple[str, float]]]]:
    plan: dict[int, dict[int, list[tuple[str, float]]]] = {}
    later_unique: dict[int, list[str]] = {
        stop_idx: _unique_preserve_order(ordered_stops[stop_idx + 1 :])
        for stop_idx in range(len(ordered_stops))
    }
    for hour, matrix in hourly_od.items():
        denom = float(max(1, int(departures_by_hour.get(hour, 1))))
        hour_plan: dict[int, list[tuple[str, float]]] = {}
        for stop_idx, stop_name in enumerate(ordered_stops):
            if stop_name not in matrix.index:
                continue
            entries: list[tuple[str, float]] = []
            for dest in later_unique[stop_idx]:
                value = _matrix_value(matrix, stop_name, dest) / denom
                if value > 0.0:
                    entries.append((dest, value))
            if entries:
                hour_plan[stop_idx] = entries
        plan[int(hour)] = hour_plan
    return plan


def _build_line_demands(
    *,
    city_key: str,
    city_result: dict[str, Any],
    line_manifest: list[dict[str, Any]],
) -> tuple[dict[str, LineDemand], dict[str, dict[str, Any]], dict[str, Any]]:
    _, by_line_segments = _load_segment_map(Path(city_result["segment_map_path"]))
    line_lookup = _line_manifest_lookup(line_manifest)
    line_map = _read_json(Path(city_result["line_map_path"]))
    demand_by_uid: dict[str, LineDemand] = {}
    line_map_by_uid = {str(line["line_uid"]): line for line in line_map}
    warnings: list[str] = []
    skipped_inactive_lines = 0
    begin = float(city_result["begin"])
    end = float(city_result["end"])

    for line_uid, line in line_map_by_uid.items():
        line_key = str(line["line_key"])
        departures = [float(value) for value in line.get("departures", [])]
        relevant_departures = [depart for depart in departures if depart <= end]
        if not relevant_departures:
            skipped_inactive_lines += 1
            continue
        manifest = line_lookup.get((city_key, line_key))
        if manifest is None:
            warnings.append(f"{line_key}: no Stage 1 manifest row")
            continue
        segments = by_line_segments.get(line_uid, [])
        if not segments:
            warnings.append(f"{line_key}: no segment map rows")
            continue
        ordered_stops = [str(segments[0]["start_stop"])] + [str(item["end_stop"]) for item in segments]
        passenger_od_path = Path(manifest["line_dir"]) / "passenger_OD.xlsx"
        passenger_od = pd.read_excel(passenger_od_path)
        departures_by_hour: dict[int, int] = {}
        for depart in relevant_departures:
            hour = int(depart // 3600) % 24
            departures_by_hour[hour] = departures_by_hour.get(hour, 0) + 1
        hourly_od = _hourly_od(passenger_od, ordered_stops, set(departures_by_hour))
        demand_by_uid[line_uid] = LineDemand(
            line_key=line_key,
            line_uid=line_uid,
            stops=ordered_stops,
            hourly_od={},
            departures_by_hour=departures_by_hour,
            boarding_plan=_build_boarding_plan(hourly_od, ordered_stops, departures_by_hour),
            final_stop=ordered_stops[-1],
        )
    vehicle_departures = _load_vehicle_departures(line_map)
    return demand_by_uid, vehicle_departures, {"warnings": warnings, "skipped_inactive_lines": skipped_inactive_lines}


def _demand_cache_path(city_result: dict[str, Any]) -> Path:
    return Path(city_result["out_dir"]) / "demand_cache.pkl"


def _serialize_line_demands(line_demands: dict[str, LineDemand]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "line_key": value.line_key,
            "line_uid": value.line_uid,
            "stops": value.stops,
            "departures_by_hour": value.departures_by_hour,
            "boarding_plan": value.boarding_plan,
            "final_stop": value.final_stop,
        }
        for key, value in line_demands.items()
    }


def _deserialize_line_demands(payload: dict[str, dict[str, Any]]) -> dict[str, LineDemand]:
    return {
        key: LineDemand(
            line_key=str(value["line_key"]),
            line_uid=str(value["line_uid"]),
            stops=[str(item) for item in value["stops"]],
            hourly_od={},
            departures_by_hour={int(hour): int(count) for hour, count in value["departures_by_hour"].items()},
            boarding_plan={
                int(hour): {
                    int(stop_idx): [(str(dest), float(amount)) for dest, amount in entries]
                    for stop_idx, entries in stops.items()
                }
                for hour, stops in value["boarding_plan"].items()
            },
            final_stop=str(value["final_stop"]),
        )
        for key, value in payload.items()
    }


def _cache_signature(city_key: str, city_result: dict[str, Any], line_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    city_lines = [line for line in line_manifest if line.get("city_key") == city_key and line.get("ok", False)]
    return {
        "version": 2,
        "city_key": city_key,
        "line_map_path": str(city_result["line_map_path"]),
        "segment_map_path": str(city_result["segment_map_path"]),
        "line_map_mtime": Path(city_result["line_map_path"]).stat().st_mtime,
        "segment_map_mtime": Path(city_result["segment_map_path"]).stat().st_mtime,
        "begin": float(city_result["begin"]),
        "end": float(city_result["end"]),
        "line_count": len(city_lines),
        "line_keys": sorted(str(line["line_key"]) for line in city_lines),
    }


def _load_or_build_line_demands(
    *,
    city_key: str,
    city_result: dict[str, Any],
    line_manifest: list[dict[str, Any]],
    rebuild_cache: bool,
) -> tuple[dict[str, LineDemand], dict[str, dict[str, Any]], dict[str, Any]]:
    signature = _cache_signature(city_key, city_result, line_manifest)
    cache_path = _demand_cache_path(city_result)
    if not rebuild_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("signature") == signature:
                meta = dict(payload.get("meta", {}))
                meta["cache_hit"] = True
                meta["cache_path"] = str(cache_path)
                return _deserialize_line_demands(payload["line_demands"]), payload["vehicle_departures"], meta
        except Exception:
            pass

    started = time.time()
    line_demands, vehicle_departures, meta = _build_line_demands(
        city_key=city_key,
        city_result=city_result,
        line_manifest=line_manifest,
    )
    meta = dict(meta)
    meta["cache_hit"] = False
    meta["cache_path"] = str(cache_path)
    meta["cache_build_sec"] = time.time() - started
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "line_demands": _serialize_line_demands(line_demands),
                "vehicle_departures": vehicle_departures,
                "meta": meta,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return line_demands, vehicle_departures, meta


def _process_stop_event(
    *,
    state: VehicleAPCState,
    demand: LineDemand,
    stop_name: str,
    stop_index: int,
    timestamp: float,
    vehicle_capacity: float,
) -> dict[str, Any]:
    alightings = float(state.onboard_by_dest.pop(stop_name, 0.0))
    requested_items = demand.boarding_plan.get(state.demand_hour, {}).get(stop_index, [])
    requested_boardings = float(sum(value for _, value in requested_items))
    available_capacity = max(0.0, float(vehicle_capacity) - state.occupancy)
    scale = min(1.0, available_capacity / requested_boardings) if requested_boardings > 0.0 else 0.0
    boardings_by_dest = {dest: value * scale for dest, value in requested_items if value * scale > 0.0}
    for dest, value in boardings_by_dest.items():
        state.onboard_by_dest[dest] = state.onboard_by_dest.get(dest, 0.0) + value
    boardings = float(sum(boardings_by_dest.values()))
    state.cumulative_boardings += boardings
    state.cumulative_alightings += alightings
    state.last_boardings = boardings
    state.last_alightings = alightings
    return {
        "timestamp_sec": timestamp,
        "vehicle_id": state.vehicle_id,
        "line_key": state.line_key,
        "line_uid": state.line_uid,
        "stop_name": stop_name,
        "stop_index": int(stop_index),
        "demand_hour": int(state.demand_hour),
        "boardings": boardings,
        "requested_boardings": requested_boardings,
        "denied_boardings": requested_boardings - boardings,
        "alightings": alightings,
        "occupancy_after": state.occupancy,
        "cumulative_boardings": state.cumulative_boardings,
        "cumulative_alightings": state.cumulative_alightings,
    }


def _segment_travel_time(segment: dict[str, Any]) -> float:
    length_m = float(segment.get("length_m", 0.0) or 0.0)
    speed_mps = float(segment.get("speed_mps", 1.0) or 1.0)
    return max(1.0, length_m / max(speed_mps, 0.1))


def _generate_scheduled_apc(
    *,
    city_result: dict[str, Any],
    line_demands: dict[str, LineDemand],
    vehicle_departures: dict[str, dict[str, Any]],
    by_line_segments: dict[str, list[dict[str, Any]]],
    vehicle_capacity: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    begin = float(city_result["begin"])
    end = float(city_result["end"])
    apc_rows: list[dict[str, Any]] = []
    timelines: dict[str, dict[str, list[float]]] = {}
    for vehicle_id, spec in sorted(vehicle_departures.items(), key=lambda item: (float(item[1]["depart"]), item[0])):
        depart = float(spec["depart"])
        if depart > end:
            continue
        line_uid = str(spec["line_uid"])
        demand = line_demands.get(line_uid)
        segments = by_line_segments.get(line_uid, [])
        if demand is None or not segments:
            continue
        state = VehicleAPCState(
            vehicle_id=vehicle_id,
            line_key=str(spec["line_key"]),
            line_uid=line_uid,
            depart=depart,
            demand_hour=int(spec["demand_hour"]),
        )
        timeline = {
            "time": [],
            "occupancy": [],
            "last_boardings": [],
            "last_alightings": [],
            "cumulative_boardings": [],
            "cumulative_alightings": [],
        }
        timestamp = depart
        for segment in segments:
            row = _process_stop_event(
                state=state,
                demand=demand,
                stop_name=str(segment["start_stop"]),
                stop_index=int(segment["segment_idx"]),
                timestamp=timestamp,
                vehicle_capacity=vehicle_capacity,
            )
            timeline["time"].append(timestamp)
            timeline["occupancy"].append(float(row["occupancy_after"]))
            timeline["last_boardings"].append(float(row["boardings"]))
            timeline["last_alightings"].append(float(row["alightings"]))
            timeline["cumulative_boardings"].append(float(row["cumulative_boardings"]))
            timeline["cumulative_alightings"].append(float(row["cumulative_alightings"]))
            if begin <= timestamp <= end:
                apc_rows.append(row)
            timestamp += _segment_travel_time(segment)

        if state.occupancy > 0.0:
            final_alighting = state.occupancy
            state.onboard_by_dest.clear()
            state.cumulative_alightings += final_alighting
            row = {
                "timestamp_sec": timestamp,
                "vehicle_id": state.vehicle_id,
                "line_key": state.line_key,
                "line_uid": state.line_uid,
                "stop_name": demand.final_stop,
                "stop_index": len(demand.stops) - 1,
                "demand_hour": int(state.demand_hour),
                "boardings": 0.0,
                "requested_boardings": 0.0,
                "denied_boardings": 0.0,
                "alightings": final_alighting,
                "occupancy_after": 0.0,
                "cumulative_boardings": state.cumulative_boardings,
                "cumulative_alightings": state.cumulative_alightings,
            }
            timeline["time"].append(timestamp)
            timeline["occupancy"].append(0.0)
            timeline["last_boardings"].append(0.0)
            timeline["last_alightings"].append(final_alighting)
            timeline["cumulative_boardings"].append(float(state.cumulative_boardings))
            timeline["cumulative_alightings"].append(float(state.cumulative_alightings))
            if begin <= timestamp <= end:
                apc_rows.append(row)
        timelines[vehicle_id] = timeline
    return apc_rows, timelines


def _timeline_at(timeline: dict[str, list[float]] | None, timestamp: float) -> dict[str, float]:
    empty = {
        "occupancy": 0.0,
        "last_boardings": 0.0,
        "last_alightings": 0.0,
        "cumulative_boardings": 0.0,
        "cumulative_alightings": 0.0,
    }
    if not timeline or not timeline.get("time"):
        return empty
    idx = bisect.bisect_right(timeline["time"], timestamp) - 1
    if idx < 0:
        return empty
    return {
        "occupancy": float(timeline["occupancy"][idx]),
        "last_boardings": float(timeline["last_boardings"][idx]),
        "last_alightings": float(timeline["last_alightings"][idx]),
        "cumulative_boardings": float(timeline["cumulative_boardings"][idx]),
        "cumulative_alightings": float(timeline["cumulative_alightings"][idx]),
    }


def _snapshot_sumocfg(city_result: dict[str, Any], *, keep_sumo_output: bool) -> Path:
    source = Path(city_result["sumocfg_path"])
    if keep_sumo_output:
        return source
    target = Path(city_result["out_dir"]) / "simulation.snapshot.sumocfg"
    root = ET.parse(source).getroot()
    output_el = root.find("output")
    if output_el is not None:
        for child in list(output_el):
            if child.tag in {"summary-output", "tripinfo-output", "tripinfo-output.write-unfinished"}:
                output_el.remove(child)
        if len(list(output_el)) == 0 and not output_el.attrib and (output_el.text is None or not output_el.text.strip()):
            root.remove(output_el)
    report_el = root.find("report")
    if report_el is None:
        report_el = ET.SubElement(root, "report")
    existing = {child.tag: child for child in list(report_el)}
    if "no-step-log" not in existing:
        ET.SubElement(report_el, "no-step-log", value="true")
    else:
        existing["no-step-log"].set("value", "true")
    if "duration-log.disable" not in existing:
        ET.SubElement(report_el, "duration-log.disable", value="true")
    else:
        existing["duration-log.disable"].set("value", "true")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def generate_city_snapshots(
    *,
    city_key: str,
    city_result: dict[str, Any],
    line_manifest: list[dict[str, Any]],
    snapshot_period: float,
    vehicle_capacity: float,
    rebuild_demand_cache: bool = False,
    keep_sumo_output: bool = False,
) -> dict[str, Any]:
    sumolib_api = _load_libsumo()
    edge_info, by_line_segments = _load_segment_map(Path(city_result["segment_map_path"]))
    line_demands, vehicle_departures, demand_meta = _load_or_build_line_demands(
        city_key=city_key,
        city_result=city_result,
        line_manifest=line_manifest,
        rebuild_cache=rebuild_demand_cache,
    )
    apc_rows, occupancy_timelines = _generate_scheduled_apc(
        city_result=city_result,
        line_demands=line_demands,
        vehicle_departures=vehicle_departures,
        by_line_segments=by_line_segments,
        vehicle_capacity=vehicle_capacity,
    )
    out_dir = Path(city_result["out_dir"]) / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    avl_path = out_dir / "avl_snapshots.csv"
    apc_path = out_dir / "apc_events.csv"

    avl_rows: list[dict[str, Any]] = []
    run_sumocfg_path = _snapshot_sumocfg(city_result, keep_sumo_output=keep_sumo_output)
    cmd = [
        "sumo",
        "-c",
        str(run_sumocfg_path),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    started = time.time()
    sumolib_api.start(cmd)
    try:
        begin = float(city_result["begin"])
        end = float(city_result["end"])
        next_snapshot = begin
        while next_snapshot <= end:
            sumolib_api.simulationStep(float(next_snapshot))
            timestamp = float(sumolib_api.simulation.getTime())
            for vehicle_id in sumolib_api.vehicle.getIDList():
                spec = vehicle_departures.get(vehicle_id)
                if spec is None:
                    continue
                edge_id = sumolib_api.vehicle.getRoadID(vehicle_id)
                info = edge_info.get(edge_id, {})
                x, y = sumolib_api.vehicle.getPosition(vehicle_id)
                apc_state = _timeline_at(occupancy_timelines.get(vehicle_id), timestamp)
                avl_rows.append(
                    {
                        "timestamp_sec": timestamp,
                        "city_key": city_key,
                        "city": city_result.get("city", city_key),
                        "vehicle_id": vehicle_id,
                        "line_key": str(spec["line_key"]),
                        "line_uid": str(spec["line_uid"]),
                        "edge_id": edge_id,
                        "segment_idx": int(info.get("segment_idx", -1)),
                        "next_stop": info.get("end_stop"),
                        "x": float(x),
                        "y": float(y),
                        "speed_mps": float(sumolib_api.vehicle.getSpeed(vehicle_id)),
                        "occupancy": apc_state["occupancy"],
                        "last_boardings": apc_state["last_boardings"],
                        "last_alightings": apc_state["last_alightings"],
                        "cumulative_boardings": apc_state["cumulative_boardings"],
                        "cumulative_alightings": apc_state["cumulative_alightings"],
                    }
                )
            next_snapshot += max(1.0, float(snapshot_period))
    finally:
        sumolib_api.close()

    avl = pd.DataFrame(avl_rows)
    apc = pd.DataFrame(apc_rows)
    avl.to_csv(avl_path, index=False)
    apc.to_csv(apc_path, index=False)

    occupancy_min = float(avl["occupancy"].min()) if len(avl) else 0.0
    occupancy_max = float(avl["occupancy"].max()) if len(avl) else 0.0
    requested_boardings_total = float(apc["requested_boardings"].sum()) if len(apc) and "requested_boardings" in apc else 0.0
    denied_boardings_total = float(apc["denied_boardings"].sum()) if len(apc) and "denied_boardings" in apc else 0.0
    summary = {
        "ok": bool(
            len(avl) > 0
            and len(apc) > 0
            and occupancy_min >= -1e-9
            and occupancy_max <= float(vehicle_capacity) + 1e-6
        ),
        "city_key": city_key,
        "city": city_result.get("city", city_key),
        "out_dir": str(out_dir),
        "avl_path": str(avl_path),
        "apc_path": str(apc_path),
        "avl_rows": int(len(avl)),
        "apc_rows": int(len(apc)),
        "vehicles_with_apc_state": int(len(occupancy_timelines)),
        "unique_vehicles_in_avl": int(avl["vehicle_id"].nunique()) if len(avl) else 0,
        "unique_lines_in_avl": int(avl["line_key"].nunique()) if len(avl) else 0,
        "occupancy_min": occupancy_min,
        "occupancy_max": occupancy_max,
        "requested_boardings_total": requested_boardings_total,
        "boardings_total": float(apc["boardings"].sum()) if len(apc) else 0.0,
        "denied_boardings_total": denied_boardings_total,
        "alightings_total": float(apc["alightings"].sum()) if len(apc) else 0.0,
        "snapshot_period": float(snapshot_period),
        "vehicle_capacity": float(vehicle_capacity),
        "sumocfg_path": str(run_sumocfg_path),
        "original_sumocfg_path": str(city_result["sumocfg_path"]),
        "keep_sumo_output": bool(keep_sumo_output),
        "elapsed_sec": time.time() - started,
        "demand_cache_hit": bool(demand_meta.get("cache_hit", False)),
        "demand_cache_path": demand_meta.get("cache_path"),
        "demand_cache_build_sec": demand_meta.get("cache_build_sec"),
        "skipped_inactive_lines": int(demand_meta.get("skipped_inactive_lines", 0) or 0),
        "warnings": demand_meta.get("warnings", []),
    }
    (out_dir / "snapshot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _generate_city_worker(payload: tuple[str, dict[str, Any], list[dict[str, Any]], float, float, bool, bool]) -> dict[str, Any]:
    city_key, city_result, line_manifest, snapshot_period, vehicle_capacity, rebuild_demand_cache, keep_sumo_output = payload
    return generate_city_snapshots(
        city_key=city_key,
        city_result=city_result,
        line_manifest=line_manifest,
        snapshot_period=snapshot_period,
        vehicle_capacity=vehicle_capacity,
        rebuild_demand_cache=rebuild_demand_cache,
        keep_sumo_output=keep_sumo_output,
    )


def _markdown_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["ok"] else "FAIL"
    lines = [
        "# SUMO APC/AVL Stage 3 Snapshot Generation",
        "",
        f"Verdict: **{verdict}**",
        "",
        "This stage runs SUMO through libsumo and produces vehicle-level AVL snapshots plus OD-overlay APC events.",
        "",
        "| City | AVL rows | APC rows | Vehicles | Lines | Boardings | Denied | Max occ | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for city in result["cities"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(city["city"]),
                    str(city["avl_rows"]),
                    str(city["apc_rows"]),
                    str(city["unique_vehicles_in_avl"]),
                    str(city["unique_lines_in_avl"]),
                    f"{city['boardings_total']:.2f}",
                    f"{city.get('denied_boardings_total', 0.0):.2f}",
                    f"{city['occupancy_max']:.2f}",
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
    stage2_report = _read_json(_resolve_path(root, args.stage2_report))
    line_manifest = _read_jsonl(_resolve_path(root, args.line_manifest))
    city_keys = list(args.cities) if args.cities else list(stage2_report.get("cities", {}).keys())
    cities: dict[str, Any] = {}
    payloads = [
        (
            city_key,
            stage2_report["cities"][city_key],
            line_manifest,
            float(args.snapshot_period),
            float(args.vehicle_capacity),
            bool(args.rebuild_demand_cache),
            bool(args.keep_sumo_output),
        )
        for city_key in city_keys
    ]
    workers = max(1, min(int(args.workers), len(payloads)))
    if workers <= 1:
        for payload in payloads:
            city_key = payload[0]
            print(f"[stage3] generating AVL/APC snapshots for {city_key}", flush=True)
            summary = _generate_city_worker(payload)
            print(
                f"[stage3] {city_key} ok={summary['ok']} avl={summary['avl_rows']} apc={summary['apc_rows']}",
                flush=True,
            )
            cities[city_key] = summary
    else:
        print(f"[stage3] generating AVL/APC snapshots with workers={workers}", flush=True)
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_city = {executor.submit(_generate_city_worker, payload): payload[0] for payload in payloads}
            for future in futures.as_completed(future_to_city):
                city_key = future_to_city[future]
                summary = future.result()
                print(
                    f"[stage3] {city_key} ok={summary['ok']} avl={summary['avl_rows']} apc={summary['apc_rows']}",
                    flush=True,
                )
                cities[city_key] = summary
    result = {
        "ok": all(city["ok"] for city in cities.values()),
        "stage": "sumo_apc_avl_stage3_snapshot_generation",
        "performance_validation": False,
        "sumo_generation": True,
        "stage2_report": str(_resolve_path(root, args.stage2_report)),
        "line_manifest_path": str(_resolve_path(root, args.line_manifest)),
        "snapshot_period": float(args.snapshot_period),
        "vehicle_capacity": float(args.vehicle_capacity),
        "rebuild_demand_cache": bool(args.rebuild_demand_cache),
        "keep_sumo_output": bool(args.keep_sumo_output),
        "workers": int(args.workers),
        "cities": cities,
    }
    _write_report(result, _resolve_path(root, args.out), _resolve_path(root, args.md_out))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-report", type=Path, default=DEFAULT_STAGE2_REPORT)
    parser.add_argument("--line-manifest", type=Path, default=DEFAULT_LINE_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--snapshot-period", type=float, default=60.0)
    parser.add_argument("--vehicle-capacity", type=float, default=90.0)
    parser.add_argument("--rebuild-demand-cache", action="store_true")
    parser.add_argument("--keep-sumo-output", action="store_true", help="Keep SUMO summary/tripinfo outputs during Stage 3 debugging")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
