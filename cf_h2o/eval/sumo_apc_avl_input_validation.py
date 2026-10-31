"""Stage 1 validation for SUMO APC/AVL benchmark inputs.

This stage extracts and validates the existing H2O city-env tables before any
SUMO network, route, AVL, or APC files are generated. It is intentionally
full-route by default. Use ``--max-lines-per-city`` only for development smoke
checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REQUIRED_LINE_FILES = {"stop_news.xlsx", "route_news.xlsx", "time_table.xlsx", "passenger_OD.xlsx"}
CONTROL_COLUMNS = {"time_period", "stop_name"}
HOUR_COLUMNS = [f"{hour:02d}:00:00" for hour in range(24)]
DEFAULT_CONFIG = Path("cf_h2o/config/cross_city_open_transit.json")
DEFAULT_OUT = Path("cf_h2o/results/sumo_apc_avl_input_validation.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/sumo_apc_avl_input_validation.md")
DEFAULT_INPUT_DIR = Path("H2Oplus/downloads/sumo_apc_avl_benchmark/inputs")


@dataclass(frozen=True)
class LineTask:
    city_key: str
    city_name: str
    env_path: str
    line_key: str
    line_dir: str
    manifest_line: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value)


def _finite_stats(values: Iterable[float]) -> dict[str, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _line_dirs(env_path: Path) -> list[Path]:
    data_dir = env_path / "data"
    if not data_dir.exists():
        return []
    return [
        child
        for child in sorted(data_dir.iterdir())
        if child.is_dir() and REQUIRED_LINE_FILES.issubset({item.name for item in child.iterdir()})
    ]


def _city_manifest(env_path: Path) -> tuple[Path | None, dict[str, Any]]:
    for name in ("gtfs_city_manifest.json", "lta_city_manifest.json"):
        path = env_path / name
        if path.exists():
            return path, _read_json(path)
    return None, {}


def _manifest_by_line_key(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in manifest.get("lines", []) if isinstance(manifest.get("lines"), list) else []:
        line_key = item.get("line_key")
        if line_key:
            out[str(line_key)] = dict(item)
    return out


def _hour_from_time_period(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "hour"):
        try:
            hour = int(value.hour)
            return hour if 0 <= hour <= 23 else None
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            return None
        hour = int(float(value)) // 3600 if float(value) >= 24 else int(float(value))
        return hour if 0 <= hour <= 23 else None
    text = str(value).strip()
    if not text:
        return None
    head = text.split(":", 1)[0]
    if not head.lstrip("-").isdigit():
        return None
    hour = int(head)
    return hour if 0 <= hour <= 23 else None


def _series_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _read_line_tables(line_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_excel(line_dir / "stop_news.xlsx"),
        pd.read_excel(line_dir / "route_news.xlsx"),
        pd.read_excel(line_dir / "time_table.xlsx"),
        pd.read_excel(line_dir / "passenger_OD.xlsx"),
    )


def validate_line_task(task: LineTask) -> dict[str, Any]:
    """Validate one line directory and return a JSON-serializable summary."""

    line_dir = Path(task.line_dir)
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "city_key": task.city_key,
        "city": task.city_name,
        "env_path": task.env_path,
        "line_key": task.line_key,
        "line_dir": task.line_dir,
        "manifest": {
            key: task.manifest_line.get(key)
            for key in (
                "route_id",
                "route_short_name",
                "route_long_name",
                "service",
                "direction",
                "pattern_index",
                "pattern_share",
                "demand_source",
                "line_headway",
                "stops",
                "segments",
                "timetable_rows",
            )
            if key in task.manifest_line
        },
    }

    missing_files = sorted(name for name in REQUIRED_LINE_FILES if not (line_dir / name).exists())
    if missing_files:
        errors.append(f"missing line files: {missing_files}")
        summary.update({"ok": False, "errors": errors, "warnings": warnings})
        return summary

    try:
        stops, route, timetable, passenger_od = _read_line_tables(line_dir)
    except Exception as exc:  # pragma: no cover - exact engine errors vary by pandas/openpyxl version.
        errors.append(f"failed to read line tables: {type(exc).__name__}: {exc}")
        summary.update({"ok": False, "errors": errors, "warnings": warnings})
        return summary

    summary.update(
        {
            "stops": int(len(stops)),
            "segments": int(len(route)),
            "timetable_rows": int(len(timetable)),
            "od_rows": int(len(passenger_od)),
        }
    )

    if len(stops) < 2:
        errors.append("fewer than 2 stops")
    if len(route) < 1:
        errors.append("no route segments")
    if len(route) and len(stops) and len(route) != len(stops) - 1:
        errors.append(f"route segment count {len(route)} != stops - 1 ({len(stops) - 1})")

    stop_names = set(stops.get("stop_name", pd.Series(dtype=str)).astype(str))
    if not stop_names:
        errors.append("missing stop_name column or empty stop names")

    lat = _series_numeric(stops, "latitude", np.nan)
    lon = _series_numeric(stops, "longitude", np.nan)
    valid_lat = lat.between(-90.0, 90.0, inclusive="both")
    valid_lon = lon.between(-180.0, 180.0, inclusive="both")
    valid_geo = valid_lat & valid_lon
    invalid_geo_count = int((~valid_geo).sum())
    if int(valid_geo.sum()) != len(stops):
        message = f"{invalid_geo_count} stops have invalid lat/lon"
        if int(valid_geo.sum()) == 0:
            errors.append(message)
        else:
            warnings.append(message)
    if int(valid_geo.sum()):
        summary["latlon_bounds"] = {
            "min_latitude": float(lat[valid_geo].min()),
            "max_latitude": float(lat[valid_geo].max()),
            "min_longitude": float(lon[valid_geo].min()),
            "max_longitude": float(lon[valid_geo].max()),
        }

    if "stop_name" in stops.columns:
        duplicate_stop_names = int(stops["stop_name"].astype(str).duplicated().sum())
        if duplicate_stop_names:
            warnings.append(f"{duplicate_stop_names} duplicated stop_name values in stop table")
        summary["duplicate_stop_names"] = duplicate_stop_names

    required_route_columns = {"start_stop", "end_stop", "distance"}
    missing_route_columns = sorted(required_route_columns - set(map(str, route.columns)))
    if missing_route_columns:
        errors.append(f"missing route columns: {missing_route_columns}")
    else:
        route_stop_refs = set(route["start_stop"].astype(str)) | set(route["end_stop"].astype(str))
        missing_route_stop_refs = sorted(route_stop_refs - stop_names)
        if missing_route_stop_refs:
            errors.append(f"{len(missing_route_stop_refs)} route stop refs are absent from stop table")
        continuity_breaks = int((route["end_stop"].astype(str).iloc[:-1].to_numpy() != route["start_stop"].astype(str).iloc[1:].to_numpy()).sum()) if len(route) > 1 else 0
        if continuity_breaks:
            warnings.append(f"{continuity_breaks} route continuity breaks between adjacent segments")
        summary["route_continuity_breaks"] = continuity_breaks

    distance = _series_numeric(route, "distance", np.nan)
    positive_distance = distance[np.isfinite(distance) & (distance > 0.0)]
    bad_distance_count = int((~np.isfinite(distance) | (distance <= 0.0)).sum())
    if bad_distance_count:
        errors.append(f"{bad_distance_count} route segments have nonpositive/invalid distance")
    summary["distance_m"] = {
        **_finite_stats(positive_distance),
        "total": float(positive_distance.sum()) if len(positive_distance) else 0.0,
        "invalid_count": bad_distance_count,
    }

    speed_columns = [column for column in HOUR_COLUMNS if column in route.columns]
    v_max = _series_numeric(route, "V_max", np.nan)
    bad_vmax_count = int((~np.isfinite(v_max) | (v_max <= 0.0)).sum())
    if bad_vmax_count:
        errors.append(f"{bad_vmax_count} route segments have nonpositive/invalid V_max")
    speed_arrays: list[np.ndarray] = []
    bad_hourly_speed_cells = 0
    for column in speed_columns:
        values = pd.to_numeric(route[column], errors="coerce").to_numpy(dtype=np.float64)
        bad_hourly_speed_cells += int((~np.isfinite(values) | (values <= 0.0)).sum())
        speed_arrays.append(values[np.isfinite(values) & (values > 0.0)])
    if len(speed_columns) != 24:
        warnings.append(f"{len(speed_columns)}/24 hourly speed columns present")
    if bad_hourly_speed_cells:
        errors.append(f"{bad_hourly_speed_cells} hourly speed cells are nonpositive/invalid")
    all_hourly_speeds = np.concatenate(speed_arrays) if speed_arrays else np.array([], dtype=np.float64)
    summary["speed_mps"] = {
        "hourly_columns": len(speed_columns),
        "v_max": _finite_stats(v_max[np.isfinite(v_max) & (v_max > 0.0)]),
        "hourly": _finite_stats(all_hourly_speeds),
        "invalid_v_max_count": bad_vmax_count,
        "invalid_hourly_speed_cells": bad_hourly_speed_cells,
    }

    launch = _series_numeric(timetable, "launch_time", np.nan)
    valid_launch = launch[np.isfinite(launch) & (launch >= 0.0) & (launch < 86400.0)]
    invalid_launch_count = int(len(launch) - len(valid_launch))
    if len(valid_launch) == 0:
        errors.append("no valid launch_time rows in [0, 86400)")
    elif invalid_launch_count:
        warnings.append(f"{invalid_launch_count} launch_time rows are invalid/outside service day")
    sorted_launch = np.sort(valid_launch.to_numpy(dtype=np.float64))
    positive_headways = np.diff(sorted_launch)
    positive_headways = positive_headways[positive_headways > 0.0]
    duplicate_departures = int(len(sorted_launch) - len(np.unique(sorted_launch))) if len(sorted_launch) else 0
    if duplicate_departures:
        warnings.append(f"{duplicate_departures} duplicate launch_time rows")
    summary["schedule"] = {
        "valid_departures": int(len(valid_launch)),
        "invalid_departures": invalid_launch_count,
        "duplicate_departures": duplicate_departures,
        "service_start_sec": float(sorted_launch[0]) if len(sorted_launch) else None,
        "service_end_sec": float(sorted_launch[-1]) if len(sorted_launch) else None,
        "headway_sec": _finite_stats(positive_headways),
    }

    value_cols = [column for column in passenger_od.columns if column not in CONTROL_COLUMNS]
    stop_name_rows = passenger_od.get("stop_name", pd.Series(dtype=str)).astype(str)
    unmatched_origin_names = sorted(set(stop_name_rows) - stop_names)
    unmatched_dest_names = sorted(set(map(str, value_cols)) - stop_names)
    matched_origin_ratio = 1.0 - (len(unmatched_origin_names) / max(1, len(set(stop_name_rows))))
    matched_dest_ratio = 1.0 - (len(unmatched_dest_names) / max(1, len(value_cols)))
    if not value_cols:
        errors.append("passenger_OD has no destination columns")
    elif matched_dest_ratio <= 0.0:
        errors.append("passenger_OD destination columns do not match stop table")
    elif unmatched_dest_names:
        warnings.append(f"{len(unmatched_dest_names)} passenger_OD destination columns do not match stop table")
    if len(passenger_od) == 0:
        errors.append("passenger_OD is empty")
    elif matched_origin_ratio <= 0.0:
        errors.append("passenger_OD stop_name rows do not match stop table")
    elif unmatched_origin_names:
        warnings.append(f"{len(unmatched_origin_names)} passenger_OD origin rows do not match stop table")

    if value_cols:
        od_values = passenger_od[value_cols].apply(pd.to_numeric, errors="coerce")
        negative_od_cells = int((od_values < 0.0).sum().sum())
        if negative_od_cells:
            errors.append(f"{negative_od_cells} passenger_OD cells are negative")
        od_values = od_values.fillna(0.0)
        row_demand = od_values.clip(lower=0.0).sum(axis=1)
        hours = passenger_od.get("time_period", pd.Series([None] * len(passenger_od))).map(_hour_from_time_period)
        invalid_hour_rows = int(hours.isna().sum())
        if invalid_hour_rows:
            warnings.append(f"{invalid_hour_rows} passenger_OD rows have invalid time_period")
        hourly_origin = row_demand.groupby(hours).sum()
        active_hours = [int(hour) for hour, value in hourly_origin.items() if hour is not None and not pd.isna(hour) and float(value) > 0.0]
        expected_od_rows = len(stops) * 24 if len(stops) else 0
        if expected_od_rows and len(passenger_od) != expected_od_rows:
            warnings.append(f"passenger_OD rows {len(passenger_od)} != stops * 24 ({expected_od_rows})")
        summary["demand"] = {
            "destination_columns": int(len(value_cols)),
            "matched_origin_ratio": float(matched_origin_ratio),
            "matched_destination_ratio": float(matched_dest_ratio),
            "total_od_demand": float(row_demand.sum()),
            "positive_od_cells": int((od_values > 0.0).sum().sum()),
            "negative_od_cells": negative_od_cells,
            "max_origin_hour_demand": float(row_demand.max()) if len(row_demand) else 0.0,
            "active_hours": active_hours,
            "invalid_time_period_rows": invalid_hour_rows,
        }
        if float(row_demand.sum()) <= 0.0:
            warnings.append("total passenger_OD demand is zero")
    else:
        summary["demand"] = {
            "destination_columns": 0,
            "matched_origin_ratio": 0.0,
            "matched_destination_ratio": 0.0,
            "total_od_demand": 0.0,
            "positive_od_cells": 0,
            "negative_od_cells": 0,
            "max_origin_hour_demand": 0.0,
            "active_hours": [],
            "invalid_time_period_rows": 0,
        }

    summary["sumo_input_contract"] = {
        "stops_have_coordinates": bool(int(valid_geo.sum()) == len(stops) and len(stops) >= 2),
        "segments_are_chain": bool(len(route) == max(len(stops) - 1, 0)),
        "segments_have_distance_and_speed": bool(bad_distance_count == 0 and bad_vmax_count == 0 and bad_hourly_speed_cells == 0),
        "departures_available": bool(len(valid_launch) > 0),
        "od_matrix_available": bool(len(value_cols) > 0 and len(passenger_od) > 0),
    }
    summary["warnings"] = warnings
    summary["errors"] = errors
    summary["ok"] = len(errors) == 0
    return summary


def _build_tasks(root: Path, config: dict[str, Any], city_keys: list[str], max_lines_per_city: int) -> tuple[list[LineTask], dict[str, dict[str, Any]]]:
    tasks: list[LineTask] = []
    city_inputs: dict[str, dict[str, Any]] = {}
    generated_envs = config.get("generated_envs", {})
    for city_key in city_keys:
        spec = generated_envs[city_key]
        env_path = _resolve_path(root, spec["env_path"])
        manifest_path, manifest = _city_manifest(env_path)
        by_line = _manifest_by_line_key(manifest)
        line_dirs = _line_dirs(env_path)
        if max_lines_per_city > 0:
            line_dirs = line_dirs[:max_lines_per_city]
        city_inputs[city_key] = {
            "city": spec.get("city", city_key),
            "env_path": str(env_path),
            "expected_line_count": int(spec.get("line_count", manifest.get("line_count", 0) or 0)),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_line_count": int(manifest.get("line_count", 0) or 0),
            "manifest_failure_count": int(manifest.get("failure_count", 0) or 0),
            "selected_line_count": len(line_dirs),
            "max_lines_per_city": max_lines_per_city,
        }
        for line_dir in line_dirs:
            manifest_line = by_line.get(line_dir.name, {})
            tasks.append(
                LineTask(
                    city_key=city_key,
                    city_name=str(spec.get("city", city_key)),
                    env_path=str(env_path),
                    line_key=line_dir.name,
                    line_dir=str(line_dir),
                    manifest_line=manifest_line,
                )
            )
    return tasks, city_inputs


def _validate_tasks_parallel(tasks: list[LineTask], workers: int, progress_every: int) -> list[dict[str, Any]]:
    if workers <= 1:
        out = []
        for idx, task in enumerate(tasks, start=1):
            out.append(validate_line_task(task))
            if progress_every > 0 and idx % progress_every == 0:
                print(f"[stage1] validated {idx}/{len(tasks)} lines", flush=True)
        return out

    out: list[dict[str, Any]] = []
    done = 0
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(validate_line_task, task): task for task in tasks}
        for future in futures.as_completed(future_to_task):
            out.append(future.result())
            done += 1
            if progress_every > 0 and done % progress_every == 0:
                print(f"[stage1] validated {done}/{len(tasks)} lines", flush=True)
    return sorted(out, key=lambda row: (str(row["city_key"]), str(row["line_key"])))


def _aggregate_city(city_input: dict[str, Any], lines: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [line for line in lines if line.get("ok")]
    failed = [line for line in lines if not line.get("ok")]
    warning_lines = [line for line in lines if line.get("warnings")]

    def total(path: tuple[str, ...]) -> float:
        value = 0.0
        for line in lines:
            cursor: Any = line
            for key in path:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            value += _safe_float(cursor, 0.0)
        return float(value)

    speed_means = [
        _safe_float(line.get("speed_mps", {}).get("hourly", {}).get("mean"), np.nan)
        for line in ready
        if line.get("speed_mps", {}).get("hourly", {}).get("mean") is not None
    ]
    distance_totals = [
        _safe_float(line.get("distance_m", {}).get("total"), 0.0)
        for line in ready
        if _safe_float(line.get("distance_m", {}).get("total"), 0.0) > 0.0
    ]
    departures = [_safe_float(line.get("schedule", {}).get("valid_departures"), 0.0) for line in lines]
    errors = [f"{line['line_key']}: {err}" for line in failed for err in line.get("errors", [])]
    warnings_flat = [f"{line['line_key']}: {warn}" for line in warning_lines for warn in line.get("warnings", [])]

    selected = int(city_input["selected_line_count"])
    expected = int(city_input["expected_line_count"])
    full_selection = selected == expected if expected else True
    if int(city_input.get("max_lines_per_city") or 0) > 0:
        full_selection = False

    return {
        **city_input,
        "ok": len(failed) == 0 and selected > 0,
        "full_selection": full_selection,
        "ready_lines": len(ready),
        "failed_lines": len(failed),
        "warning_lines": len(warning_lines),
        "total_stops": int(total(("stops",))),
        "total_segments": int(total(("segments",))),
        "total_departures": int(total(("schedule", "valid_departures"))),
        "total_od_demand": total(("demand", "total_od_demand")),
        "total_distance_km": total(("distance_m", "total")) / 1000.0,
        "line_distance_km": _finite_stats(distance_totals),
        "hourly_speed_mps": _finite_stats(speed_means),
        "departures_per_line": _finite_stats(departures),
        "errors": errors[:50],
        "warnings": warnings_flat[:50],
        "truncated_error_count": max(0, len(errors) - 50),
        "truncated_warning_count": max(0, len(warnings_flat) - 50),
    }


def _quality_gates(result: dict[str, Any]) -> dict[str, bool]:
    cities = result["cities"]
    lines = result["line_counts"]
    return {
        "all_configured_lines_checked": all(city["full_selection"] for city in cities.values()),
        "all_selected_lines_ready": all(city["ok"] for city in cities.values()),
        "no_line_errors": lines["failed"] == 0,
        "all_cities_have_departures": all(int(city["total_departures"]) > 0 for city in cities.values()),
        "all_cities_have_positive_demand": all(float(city["total_od_demand"]) > 0.0 for city in cities.values()),
        "all_cities_have_positive_distance": all(float(city["total_distance_km"]) > 0.0 for city in cities.values()),
    }


def _write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False, default=_json_default, sort_keys=True) + "\n")


def _format_float(value: Any, digits: int = 2) -> str:
    number = _safe_float(value, float("nan"))
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _markdown_report(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["ok"] else "FAIL"
    lines = [
        "# SUMO APC/AVL Stage 1 Input Validation",
        "",
        f"Verdict: **{verdict}**",
        "",
        "This stage validates existing H2O city-env tables as inputs for a later SUMO APC/AVL generator. It does not generate SUMO files or run SUMO.",
        "",
        "## Quality Gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for gate, passed in result["quality_gates"].items():
        lines.append(f"| `{gate}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## City Summary",
            "",
            "| City | Lines | Ready | Failed | Warnings | Stops | Segments | Departures | OD demand | Distance km | Mean speed m/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for city in result["cities"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(city["city"]),
                    str(city["selected_line_count"]),
                    str(city["ready_lines"]),
                    str(city["failed_lines"]),
                    str(city["warning_lines"]),
                    str(city["total_stops"]),
                    str(city["total_segments"]),
                    str(city["total_departures"]),
                    _format_float(city["total_od_demand"], 1),
                    _format_float(city["total_distance_km"], 1),
                    _format_float(city["hourly_speed_mps"]["mean"], 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Line-level manifest: `{result['line_manifest_path']}`",
            f"- JSON report: `{result['result_path']}`",
            "",
            "## Next Step",
            "",
            "If this stage passes, the next stage may generate SUMO nodes/edges/routes from the validated line manifest and run a small SUMO smoke before full-city generation.",
            "",
        ]
    )
    if not result["ok"]:
        lines.extend(["## First Errors", ""])
        for city in result["cities"].values():
            for error in city.get("errors", [])[:10]:
                lines.append(f"- {city['city']}: {error}")
    return "\n".join(lines).rstrip() + "\n"


def _write_report(result: dict[str, Any], out: Path, md_out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    result["result_path"] = str(out)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_markdown_report(result), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    config_path = _resolve_path(root, args.config)
    config = _read_json(config_path)
    generated_envs = config.get("generated_envs", {})
    if args.cities:
        city_keys = list(args.cities)
    else:
        city_keys = list(generated_envs.keys())
    unknown = sorted(set(city_keys) - set(generated_envs))
    if unknown:
        raise KeyError(f"unknown city keys: {unknown}")

    tasks, city_inputs = _build_tasks(root, config, city_keys, args.max_lines_per_city)
    print(
        f"[stage1] validating {len(tasks)} line envs across {len(city_keys)} cities "
        f"with workers={args.workers}",
        flush=True,
    )
    started = time.time()
    line_summaries = _validate_tasks_parallel(tasks, max(1, int(args.workers)), int(args.progress_every))
    elapsed = time.time() - started

    input_dir = _resolve_path(root, args.input_dir)
    suffix = "full" if int(args.max_lines_per_city) <= 0 else f"max{int(args.max_lines_per_city)}"
    line_manifest_path = input_dir / f"sumo_apc_avl_line_inputs_{suffix}.jsonl"
    _write_jsonl(line_manifest_path, line_summaries)

    by_city: dict[str, list[dict[str, Any]]] = {key: [] for key in city_keys}
    for line in line_summaries:
        by_city.setdefault(str(line["city_key"]), []).append(line)

    cities = {
        key: _aggregate_city(city_inputs[key], by_city.get(key, []))
        for key in city_keys
    }
    failed = sum(city["failed_lines"] for city in cities.values())
    warning_lines = sum(city["warning_lines"] for city in cities.values())
    result: dict[str, Any] = {
        "ok": False,
        "stage": "sumo_apc_avl_stage1_input_extraction_validation",
        "performance_validation": False,
        "sumo_generation": False,
        "config": str(config_path),
        "workers": int(args.workers),
        "max_lines_per_city": int(args.max_lines_per_city),
        "elapsed_sec": elapsed,
        "line_manifest_path": str(line_manifest_path),
        "line_counts": {
            "selected": len(line_summaries),
            "ready": len([line for line in line_summaries if line.get("ok")]),
            "failed": int(failed),
            "with_warnings": int(warning_lines),
        },
        "cities": cities,
    }
    gates = _quality_gates(result)
    result["quality_gates"] = gates
    result["next_stage_allowed"] = bool(all(gates.values()))
    result["ok"] = bool(result["next_stage_allowed"])
    _write_report(result, _resolve_path(root, args.out), _resolve_path(root, args.md_out))
    print(
        f"[stage1] done: ok={result['ok']} ready={result['line_counts']['ready']}/"
        f"{result['line_counts']['selected']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--cities", nargs="*", default=None, help="Optional city keys from the config.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-lines-per-city", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
