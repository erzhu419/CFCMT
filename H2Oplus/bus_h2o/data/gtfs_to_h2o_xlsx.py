"""
Convert a GTFS static feed, plus optional public ridership data, into H2O+
BusSimEnv data.

The output directory contains:
    config.json
    data/stop_news.xlsx
    data/route_news.xlsx
    data/time_table.xlsx
    data/passenger_OD.xlsx

By default a single route/direction can still be exported for debugging.  For
validation data, use ``--all-routes``: it writes a city-level MultiLineEnv
bundle with one ``data/<line_key>/`` directory per route-direction-pattern.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HOURS = list(range(6, 20))
DEFAULT_DAY_TYPE = "weekday"
DEFAULT_MIN_SPEED = 2.0
DEFAULT_MAX_SPEED = 15.0
DEFAULT_PROXY_PASSENGERS_PER_TRIP = 32.0
_MBTA_GROUPED_CACHE: dict[tuple[str, str, str, tuple[int, ...]], pd.DataFrame] = {}


def parse_hours(spec: str) -> list[int]:
    values: list[int] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(item))
    result = sorted(set(values))
    if not result or any(hour < 0 or hour > 23 for hour in result):
        raise ValueError(f"Invalid hour spec: {spec}")
    return result


def safe_token(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "", str(value)).upper()
    return text or "ROUTE"


def read_csv(path: pathlib.Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def parse_gtfs_time(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    return hour * 3600 + minute * 60 + second


def seconds_to_hhmmss(seconds: int) -> str:
    seconds = int(seconds)
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def yyyymmdd(value: str) -> dt.date:
    return dt.datetime.strptime(str(value), "%Y%m%d").date()


def choose_service_date(calendar_df: pd.DataFrame, calendar_dates_df: pd.DataFrame, requested: str | None) -> dt.date | None:
    if requested:
        return yyyymmdd(requested.replace("-", ""))
    if calendar_df.empty:
        dates = calendar_dates_df.get("date", pd.Series(dtype=str))
        dates = [yyyymmdd(item) for item in dates if str(item)]
        weekdays = [item for item in dates if item.weekday() < 5]
        return max(weekdays or dates) if dates else None

    start = min(yyyymmdd(item) for item in calendar_df["start_date"] if str(item))
    end = max(yyyymmdd(item) for item in calendar_df["end_date"] if str(item))
    today = dt.date.today()
    candidate = today if start <= today <= end and today.weekday() < 5 else end
    while candidate.weekday() >= 5:
        candidate -= dt.timedelta(days=1)
    if candidate < start:
        candidate = start
        while candidate.weekday() >= 5 and candidate <= end:
            candidate += dt.timedelta(days=1)
    return candidate if start <= candidate <= end else None


def active_service_ids(calendar_df: pd.DataFrame, calendar_dates_df: pd.DataFrame, service_date: dt.date | None) -> set[str]:
    if service_date is None:
        return set()
    service_ids: set[str] = set()
    weekday_col = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][service_date.weekday()]
    date_text = service_date.strftime("%Y%m%d")

    if not calendar_df.empty:
        for row in calendar_df.itertuples(index=False):
            if not getattr(row, weekday_col, "0") == "1":
                continue
            if str(getattr(row, "start_date")) <= date_text <= str(getattr(row, "end_date")):
                service_ids.add(str(getattr(row, "service_id")))

    if not calendar_dates_df.empty:
        for row in calendar_dates_df.itertuples(index=False):
            if str(getattr(row, "date")) != date_text:
                continue
            service_id = str(getattr(row, "service_id"))
            exception_type = str(getattr(row, "exception_type"))
            if exception_type == "1":
                service_ids.add(service_id)
            elif exception_type == "2":
                service_ids.discard(service_id)
    return service_ids


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def stop_haversine_distances(stop_rows: pd.DataFrame) -> list[float]:
    dists = [0.0]
    for prev, cur in zip(stop_rows.iloc[:-1].itertuples(index=False), stop_rows.iloc[1:].itertuples(index=False)):
        dist = haversine_m(float(prev.stop_lat), float(prev.stop_lon), float(cur.stop_lat), float(cur.stop_lon))
        dists.append(max(30.0, dist))
    return dists


def infer_shape_dist_factor(shape_dist_values: list[float], haversine_total_m: float) -> float:
    if len(shape_dist_values) < 2:
        return 1.0
    span = max(shape_dist_values) - min(shape_dist_values)
    if span <= 0 or haversine_total_m <= 0:
        return 1.0
    candidates = [1.0, 1000.0, 1609.344]
    return min(candidates, key=lambda factor: abs(span * factor - haversine_total_m))


def load_gtfs(gtfs_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    return {
        "agency": read_csv(gtfs_dir / "agency.txt", required=False),
        "routes": read_csv(gtfs_dir / "routes.txt"),
        "trips": read_csv(gtfs_dir / "trips.txt"),
        "stop_times": read_csv(gtfs_dir / "stop_times.txt"),
        "stops": read_csv(gtfs_dir / "stops.txt"),
        "calendar": read_csv(gtfs_dir / "calendar.txt", required=False),
        "calendar_dates": read_csv(gtfs_dir / "calendar_dates.txt", required=False),
    }


def resolve_route_id(routes: pd.DataFrame, selector: str | None) -> str | None:
    if selector is None:
        return None
    selector = str(selector)
    if selector in set(routes["route_id"].astype(str)):
        return selector
    if "route_short_name" in routes.columns:
        matches = routes[routes["route_short_name"].astype(str) == selector]
        if not matches.empty:
            return str(matches.iloc[0]["route_id"])
    raise RuntimeError(f"Route selector {selector!r} did not match route_id or route_short_name")


def trip_signature(stop_times: pd.DataFrame, trip_id: str) -> tuple[str, ...]:
    rows = stop_times[stop_times["trip_id"] == trip_id].sort_values("stop_sequence")
    return tuple(rows["stop_id"].astype(str).tolist())


def auto_route_direction(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    service_ids: set[str],
    min_stops: int,
    max_stops: int,
) -> tuple[str, int]:
    bus_routes = routes.copy()
    if "route_type" in bus_routes.columns:
        bus_routes = bus_routes[bus_routes["route_type"].astype(str) == "3"]
    active_trips = trips[trips["service_id"].isin(service_ids)].copy() if service_ids else trips.copy()
    if "direction_id" not in active_trips.columns:
        active_trips["direction_id"] = "0"

    trip_stop_counts = stop_times.groupby("trip_id").size()
    candidates: list[tuple[int, int, str, int]] = []
    for route_id in bus_routes["route_id"].astype(str).unique():
        route_trips = active_trips[active_trips["route_id"].astype(str) == route_id]
        if route_trips.empty:
            continue
        for direction_text, group in route_trips.groupby("direction_id"):
            sample_ids = group["trip_id"].astype(str).head(40).tolist()
            stop_counts = [int(trip_stop_counts.get(trip_id, 0)) for trip_id in sample_ids]
            if not stop_counts:
                continue
            median_stops = int(np.median(stop_counts))
            if min_stops <= median_stops <= max_stops:
                candidates.append((len(group), median_stops, route_id, int(float(direction_text or 0))))
    if not candidates:
        raise RuntimeError("Could not auto-select a bus route/direction with the requested stop-count bounds")
    candidates.sort(reverse=True)
    _, _, route_id, direction = candidates[0]
    return route_id, direction


def choose_pattern_trips(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    route_id: str,
    direction: int,
    service_ids: set[str],
    min_stops: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    candidates = route_direction_patterns(
        trips,
        stop_times,
        route_id,
        direction,
        service_ids,
        min_stops=min_stops,
        max_stops=None,
    )
    if not candidates:
        raise RuntimeError(f"No pattern with at least {min_stops} stops for route_id={route_id}, direction={direction}")

    # Prefer the most frequent full pattern, then the longest.
    signature, trip_ids = candidates[0]
    active = trips[trips["route_id"].astype(str) == str(route_id)].copy()
    return active[active["trip_id"].astype(str).isin(trip_ids)].copy(), signature


def route_direction_patterns(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    route_id: str,
    direction: int,
    service_ids: set[str],
    min_stops: int,
    max_stops: int | None,
) -> list[tuple[tuple[str, ...], list[str]]]:
    active = trips[trips["route_id"].astype(str) == str(route_id)].copy()
    if service_ids:
        active = active[active["service_id"].isin(service_ids)].copy()
    if "direction_id" in active.columns:
        direction_num = pd.to_numeric(active["direction_id"].replace("", "0"), errors="coerce").fillna(0).astype(int)
        active = active[direction_num == int(direction)].copy()
    if active.empty:
        return []

    active_ids = set(active["trip_id"].astype(str).tolist())
    active_stop_times = stop_times[stop_times["trip_id"].astype(str).isin(active_ids)].copy()
    active_stop_times["_seq"] = pd.to_numeric(active_stop_times["stop_sequence"], errors="coerce").fillna(0)
    signatures: dict[tuple[str, ...], list[str]] = {}
    for trip_id, rows in active_stop_times.sort_values(["trip_id", "_seq"]).groupby("trip_id", sort=False):
        sig = tuple(rows.sort_values("_seq")["stop_id"].astype(str).tolist())
        if len(sig) < min_stops:
            continue
        if max_stops is not None and len(sig) > max_stops:
            continue
        signatures.setdefault(sig, []).append(str(trip_id))
    return sorted(signatures.items(), key=lambda item: (len(item[1]), len(item[0])), reverse=True)


def representative_stop_times(stop_times: pd.DataFrame, trip_ids: list[str], signature: tuple[str, ...]) -> pd.DataFrame:
    # Use the first matching trip with the chosen signature.
    for trip_id in trip_ids:
        rows = stop_times[stop_times["trip_id"] == trip_id].copy()
        rows["_seq"] = pd.to_numeric(rows["stop_sequence"], errors="coerce").fillna(0)
        rows = rows.sort_values("_seq")
        if tuple(rows["stop_id"].astype(str).tolist()) == signature:
            return rows.drop(columns=["_seq"])
    raise RuntimeError("Internal error: no representative trip matched selected signature")


def build_stop_and_segment_tables(
    rep_st: pd.DataFrame,
    stops: pd.DataFrame,
    city: str,
    route_token: str,
    direction: int,
    matching_stop_times: pd.DataFrame,
    hours: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    stops_by_id = stops.set_index("stop_id")
    ordered_stop_ids = rep_st["stop_id"].astype(str).tolist()
    prefix = f"{safe_token(city)}{route_token}D{direction}"
    internal_names = [f"{prefix}_{idx + 1:03d}" for idx in range(len(ordered_stop_ids))]
    stop_id_to_internal = dict(zip(ordered_stop_ids, internal_names))

    stop_records: list[dict[str, Any]] = []
    geo_rows = []
    for idx, (stop_id, internal) in enumerate(zip(ordered_stop_ids, internal_names)):
        meta = stops_by_id.loc[stop_id]
        geo_rows.append(meta)
        stop_records.append(
            {
                "stop_id": idx,
                "stop_name": internal,
                "gtfs_stop_id": stop_id,
                "gtfs_stop_name": meta.get("stop_name", ""),
                "latitude": float(meta.get("stop_lat", np.nan)),
                "longitude": float(meta.get("stop_lon", np.nan)),
            }
        )
    stop_news = pd.DataFrame(stop_records)
    geo_df = pd.DataFrame(geo_rows).reset_index(drop=True)
    haversine_segments = stop_haversine_distances(geo_df)

    shape_dist_numeric = pd.to_numeric(rep_st.get("shape_dist_traveled", pd.Series(dtype=str)), errors="coerce")
    shape_values = shape_dist_numeric.tolist() if shape_dist_numeric.notna().sum() >= 2 else []
    shape_factor = infer_shape_dist_factor(
        [float(x) for x in shape_values if pd.notna(x)],
        sum(haversine_segments[1:]),
    )

    segment_dists: list[float] = []
    for idx in range(len(ordered_stop_ids) - 1):
        if shape_dist_numeric.notna().sum() >= 2 and pd.notna(shape_dist_numeric.iloc[idx]) and pd.notna(shape_dist_numeric.iloc[idx + 1]):
            dist = (float(shape_dist_numeric.iloc[idx + 1]) - float(shape_dist_numeric.iloc[idx])) * shape_factor
            if not np.isfinite(dist) or dist <= 0:
                dist = haversine_segments[idx + 1]
        else:
            dist = haversine_segments[idx + 1]
        segment_dists.append(max(30.0, float(dist)))

    speed_by_segment_hour = compute_schedule_speeds(matching_stop_times, ordered_stop_ids, segment_dists, hours)
    route_records: list[dict[str, Any]] = []
    for idx, dist in enumerate(segment_dists):
        hourly = speed_by_segment_hour.get(idx, {})
        fallback_speed = float(np.nanmedian(list(hourly.values()))) if hourly else 8.0
        fallback_speed = float(np.clip(fallback_speed, DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED))
        row: dict[str, Any] = {
            "route_id": idx,
            "start_stop": internal_names[idx],
            "end_stop": internal_names[idx + 1],
            "distance": round(dist, 2),
            "V_max": round(fallback_speed, 2),
        }
        for hour in hours:
            row[f"{hour:02d}:00:00"] = round(float(hourly.get(hour, fallback_speed)), 2)
        row["gtfs_start_stop_id"] = ordered_stop_ids[idx]
        row["gtfs_end_stop_id"] = ordered_stop_ids[idx + 1]
        route_records.append(row)
    route_news = pd.DataFrame(route_records)
    return stop_news, route_news, stop_id_to_internal


def compute_schedule_speeds(
    stop_times: pd.DataFrame,
    ordered_stop_ids: list[str],
    segment_dists: list[float],
    hours: list[int],
) -> dict[int, dict[int, float]]:
    per_segment_hour: dict[int, dict[int, list[float]]] = {
        idx: {hour: [] for hour in hours} for idx in range(len(segment_dists))
    }
    wanted = set(ordered_stop_ids)
    for _, trip_rows in stop_times.groupby("trip_id"):
        rows = trip_rows.copy()
        rows["_seq"] = pd.to_numeric(rows["stop_sequence"], errors="coerce").fillna(0)
        rows = rows.sort_values("_seq")
        rows = rows[rows["stop_id"].astype(str).isin(wanted)]
        if list(rows["stop_id"].astype(str)) != ordered_stop_ids:
            continue
        dep = [parse_gtfs_time(x) for x in rows["departure_time"].tolist()]
        arr = [parse_gtfs_time(x) for x in rows["arrival_time"].tolist()]
        for idx in range(len(segment_dists)):
            start = dep[idx] if dep[idx] is not None else arr[idx]
            end = arr[idx + 1] if arr[idx + 1] is not None else dep[idx + 1]
            if start is None or end is None or end <= start:
                continue
            hour = int(start // 3600)
            if hour not in per_segment_hour[idx]:
                continue
            speed = segment_dists[idx] / float(end - start)
            if np.isfinite(speed):
                per_segment_hour[idx][hour].append(float(np.clip(speed, DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED)))

    result: dict[int, dict[int, float]] = {}
    for idx, by_hour in per_segment_hour.items():
        result[idx] = {}
        all_values = [value for values in by_hour.values() for value in values]
        global_median = float(np.median(all_values)) if all_values else 8.0
        for hour in hours:
            values = by_hour[hour]
            result[idx][hour] = float(np.median(values)) if values else global_median
    return result


def build_timetable(
    matching_stop_times: pd.DataFrame,
    ordered_stop_ids: list[str],
    hours: list[int],
) -> pd.DataFrame:
    sim_start = min(hours) * 3600
    sim_end = (max(hours) + 1) * 3600
    rows = []
    for trip_id, group in matching_stop_times.groupby("trip_id"):
        st = group.copy()
        st["_seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce").fillna(0)
        st = st.sort_values("_seq")
        if list(st["stop_id"].astype(str)) != ordered_stop_ids:
            continue
        first_time = parse_gtfs_time(st.iloc[0].get("departure_time")) or parse_gtfs_time(st.iloc[0].get("arrival_time"))
        if first_time is None or not (sim_start <= first_time < sim_end):
            continue
        rows.append({"launch_time": int(first_time - sim_start), "direction": 1, "gtfs_trip_id": str(trip_id)})
    if not rows:
        raise RuntimeError("No timetable rows fell within requested simulation hours")
    df = pd.DataFrame(rows).sort_values("launch_time").reset_index(drop=True)
    return df


def route_profile_od(
    hourly_totals: dict[int, float],
    internal_names: list[str],
    hours: list[int],
    stop_origin_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(internal_names)
    if n < 2:
        raise RuntimeError("At least two stops are required for OD generation")
    if stop_origin_weights is None:
        x = np.linspace(0.15, 0.95, n)
        stop_origin_weights = np.sin(np.pi * x)
        stop_origin_weights[-1] = 0.0
    stop_origin_weights = np.maximum(stop_origin_weights.astype(float), 0.0)
    if stop_origin_weights.sum() <= 0:
        stop_origin_weights = np.ones(n)
        stop_origin_weights[-1] = 0.0
    stop_origin_weights /= stop_origin_weights.sum()

    rows: list[dict[str, Any]] = []
    for hour in hours:
        total = float(hourly_totals.get(hour, 0.0))
        origin_boardings = total * stop_origin_weights
        for i, origin in enumerate(internal_names):
            row: dict[str, Any] = {"time_period": f"{hour:02d}:00:00", "stop_name": origin}
            downstream = np.zeros(n)
            if i + 1 < n:
                distances = np.arange(1, n - i, dtype=float)
                downstream[i + 1 :] = 1.0 / np.sqrt(distances)
                downstream[-1] += 0.2
            if downstream.sum() > 0:
                downstream /= downstream.sum()
            for j, dest in enumerate(internal_names):
                row[dest] = round(float(origin_boardings[i] * downstream[j]), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def mbta_hourly_boardings(
    ridership_path: pathlib.Path,
    route_id: str,
    direction: int,
    internal_names: list[str],
    stop_id_to_internal: dict[str, str],
    hours: list[int],
    day_type: str,
    season: str | None,
) -> pd.DataFrame | None:
    csv_paths = []
    if ridership_path.is_dir():
        csv_paths = sorted(ridership_path.rglob("*.csv"))
    elif ridership_path.suffix.lower() == ".csv":
        csv_paths = [ridership_path]
    if not csv_paths:
        return None

    if season:
        season_token = season.replace(" ", "_")
        matching = [path for path in csv_paths if season_token in path.name]
        path = matching[-1] if matching else csv_paths[-1]
    else:
        path = max(csv_paths, key=_mbta_season_sort_key)
        season = _season_from_mbta_filename(path.name)

    sample = pd.read_csv(path, nrows=5, dtype=str)
    cols = set(sample.columns)
    if {"route_id", "direction_id", "trip_start_time", "day_type_name", "stop_id"}.issubset(cols):
        colmap = {
            "season": "season",
            "route": "route_id",
            "direction": "direction_id",
            "time": "trip_start_time",
            "day": "day_type_name",
            "stop": "stop_id",
            "board": "average_boardings" if "average_boardings" in cols else "boardings",
            "alight": "average_alightings" if "average_alightings" in cols else "alightings",
        }
        weekday_values = {"weekday"}
    elif {"GTFS route_id", "GTFS direction_id", "trip start time", "Day Type", "GTFS stop_id"}.issubset(cols):
        colmap = {
            "season": "Year",
            "route": "GTFS route_id",
            "direction": "GTFS direction_id",
            "time": "trip start time",
            "day": "Day Type",
            "stop": "GTFS stop_id",
            "board": "Boardings",
            "alight": "Alightings",
        }
        weekday_values = {"weekday", "wkdy"}
    else:
        return None
    if colmap["board"] not in sample.columns or colmap["alight"] not in sample.columns:
        return None

    seasons: set[str] | None = {season} if season else None
    if seasons is None:
        season_values = pd.read_csv(path, usecols=["season"], dtype=str)["season"].dropna().unique().tolist()
        if season_values:
            seasons = {sorted(season_values)[-1]}

    cache_key = (str(path.resolve()), sorted(seasons)[-1] if seasons else "all", day_type.lower(), tuple(hours))
    if cache_key not in _MBTA_GROUPED_CACHE:
        grouped_parts = []
        chunksize = 500_000
        for chunk in pd.read_csv(path, dtype=str, chunksize=chunksize):
            if seasons is not None:
                chunk = chunk[chunk[colmap["season"]].isin(seasons)]
            if day_type.lower() == "weekday":
                chunk = chunk[chunk[colmap["day"]].str.lower().isin(weekday_values)].copy()
            else:
                chunk = chunk[chunk[colmap["day"]].str.lower() == day_type.lower()].copy()
            if chunk.empty:
                continue
            chunk["_hour"] = chunk[colmap["time"]].map(parse_gtfs_time).fillna(-1).astype(int) // 3600
            chunk = chunk[chunk["_hour"].isin(hours)]
            if chunk.empty:
                continue
            chunk["_route"] = chunk[colmap["route"]].astype(str)
            chunk["_direction"] = pd.to_numeric(chunk[colmap["direction"]], errors="coerce").fillna(-1).astype(int)
            chunk["_stop_id"] = chunk[colmap["stop"]].astype(str)
            chunk["_board"] = pd.to_numeric(chunk[colmap["board"]], errors="coerce").fillna(0.0)
            chunk["_alight"] = pd.to_numeric(chunk[colmap["alight"]], errors="coerce").fillna(0.0)
            grouped_parts.append(
                chunk.groupby(["_route", "_direction", "_hour", "_stop_id"], as_index=False)[["_board", "_alight"]].sum()
            )
        if grouped_parts:
            grouped_all = pd.concat(grouped_parts, ignore_index=True)
            grouped_all = grouped_all.groupby(["_route", "_direction", "_hour", "_stop_id"], as_index=False)[["_board", "_alight"]].sum()
        else:
            grouped_all = pd.DataFrame(columns=["_route", "_direction", "_hour", "_stop_id", "_board", "_alight"])
        _MBTA_GROUPED_CACHE[cache_key] = grouped_all

    grouped_all = _MBTA_GROUPED_CACHE[cache_key]
    sub = grouped_all[
        (grouped_all["_route"].astype(str) == str(route_id))
        & (grouped_all["_direction"].astype(int) == int(direction))
        & (grouped_all["_stop_id"].isin(stop_id_to_internal))
    ]

    board = {hour: np.zeros(len(internal_names), dtype=float) for hour in hours}
    alight = {hour: np.zeros(len(internal_names), dtype=float) for hour in hours}
    stop_rank = {name: idx for idx, name in enumerate(internal_names)}
    matched = len(sub)
    for _, rec in sub.iterrows():
        internal = stop_id_to_internal[str(rec["_stop_id"])]
        idx = stop_rank[internal]
        hour = int(rec["_hour"])
        board[hour][idx] += float(rec["_board"])
        alight[hour][idx] += float(rec["_alight"])

    if matched == 0:
        return None

    rows: list[dict[str, Any]] = []
    for hour in hours:
        for i, origin in enumerate(internal_names):
            row: dict[str, Any] = {"time_period": f"{hour:02d}:00:00", "stop_name": origin}
            downstream = np.zeros(len(internal_names), dtype=float)
            if i + 1 < len(internal_names):
                downstream[i + 1 :] = alight[hour][i + 1 :]
                if downstream.sum() <= 0:
                    downstream[i + 1 :] = 1.0
            if downstream.sum() > 0:
                downstream /= downstream.sum()
            for j, dest in enumerate(internal_names):
                row[dest] = round(float(board[hour][i] * downstream[j]), 3)
            rows.append(row)
    print(f"MBTA ridership matched grouped rows={matched}, season={sorted(seasons)[-1] if seasons else 'unknown'}")
    return pd.DataFrame(rows)


def _season_from_mbta_filename(name: str) -> str | None:
    match = re.search(r"_(Spring|Fall)_(\d{4})\.csv$", name)
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)}"


def _mbta_season_sort_key(path: pathlib.Path) -> tuple[int, int]:
    season = _season_from_mbta_filename(path.name)
    if season is None:
        return (0, 0)
    name, year_text = season.split()
    return (int(year_text), 1 if name.lower() == "spring" else 2)


def halifax_hourly_totals(ridership_csv: pathlib.Path, route_short_name: str, hours: list[int]) -> dict[int, float] | None:
    if not ridership_csv.exists():
        return None
    df = pd.read_csv(ridership_csv, dtype=str)
    required = {"Route_Number", "Ridership_Total", "Route_Hour", "Route_Date"}
    if not required.issubset(df.columns):
        return None
    sub = df[df["Route_Number"].astype(str) == str(route_short_name)].copy()
    if sub.empty:
        return None
    sub["Ridership_Total"] = pd.to_numeric(sub["Ridership_Total"], errors="coerce").fillna(0.0)
    sub["Route_Hour"] = pd.to_numeric(sub["Route_Hour"], errors="coerce")
    sub["Route_Date_dt"] = pd.to_datetime(pd.to_numeric(sub["Route_Date"], errors="coerce"), unit="ms", errors="coerce")
    sub = sub[sub["Route_Date_dt"].dt.weekday < 5]
    sub["_hour"] = np.floor(sub["Route_Hour"]).astype("Int64")
    sub = sub[sub["_hour"].isin(hours)]
    if sub.empty:
        return None
    by_date_hour = sub.groupby([sub["Route_Date_dt"].dt.date, "_hour"])["Ridership_Total"].sum().reset_index()
    hourly = by_date_hour.groupby("_hour")["Ridership_Total"].mean().to_dict()
    return {int(hour): float(value) for hour, value in hourly.items()}


def scheduled_proxy_totals(timetable: pd.DataFrame, hours: list[int], passengers_per_trip: float) -> dict[int, float]:
    totals = {hour: 0.0 for hour in hours}
    sim_start = min(hours) * 3600
    for launch_time in timetable["launch_time"]:
        hour = int((float(launch_time) + sim_start) // 3600)
        if hour in totals:
            totals[hour] += passengers_per_trip
    return totals


def scale_passenger_od(passenger_od: pd.DataFrame, internal_names: list[str], scale: float) -> pd.DataFrame:
    if abs(float(scale) - 1.0) < 1e-9:
        return passenger_od
    result = passenger_od.copy()
    for col in internal_names:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0) * float(scale)
    return result


def write_config(
    env_dir: pathlib.Path,
    line_idx: int,
    line_id_str: str,
    line_headway: float,
    max_time: int,
    sim_start_hour: int,
    sim_end_hour: int,
) -> None:
    config = {
        "time_step": 1,
        "route_state_update_freq": 300,
        "passenger_state_update_freq": 20,
        "max_time": int(max_time),
        "line_idx": int(line_idx),
        "line_id_str": line_id_str,
        "line_headway": float(line_headway),
        "sim_start_hour": int(sim_start_hour),
        "sim_end_hour": int(sim_end_hour),
        "use_virtual_colines": False,
    }
    (env_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def _service_filter(gtfs: dict[str, pd.DataFrame], args: argparse.Namespace) -> tuple[dt.date | None, set[str]]:
    if args.service_scope == "service-date" or args.service_date:
        service_date = choose_service_date(gtfs["calendar"], gtfs["calendar_dates"], args.service_date)
        return service_date, active_service_ids(gtfs["calendar"], gtfs["calendar_dates"], service_date)
    return None, set()


def _line_key(route_short: str, route_id: str, direction: int, pattern_index: int, pattern_count: int, signature: tuple[str, ...]) -> str:
    base = safe_token(route_short or route_id)
    if pattern_count <= 1:
        return f"{base}_D{int(direction)}"
    digest = hashlib.sha1("|".join(signature).encode("utf-8")).hexdigest()[:8].upper()
    return f"{base}_D{int(direction)}_P{pattern_index + 1:03d}_{digest}"


def _unique_line_key(line_key: str, route_id: str, used: set[str]) -> str:
    """Avoid overwriting lines when agencies reuse the same public short name."""

    if line_key not in used:
        used.add(line_key)
        return line_key
    route_digest = hashlib.sha1(str(route_id).encode("utf-8")).hexdigest()[:6].upper()
    candidate = f"{line_key}_R{route_digest}"
    suffix = 2
    while candidate in used:
        candidate = f"{line_key}_R{route_digest}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def build_env_for_pattern(
    args: argparse.Namespace,
    gtfs: dict[str, pd.DataFrame],
    route_id: str,
    direction: int,
    service_ids: set[str],
    service_date: dt.date | None,
    signature: tuple[str, ...],
    pattern_trip_ids: list[str],
    *,
    pattern_index: int = 0,
    pattern_count: int = 1,
    pattern_share: float = 1.0,
    env_dir: pathlib.Path | None = None,
    direct_data_dir: bool = False,
    write_config_file: bool = True,
    line_key_override: str | None = None,
) -> tuple[pathlib.Path, dict[str, Any]]:
    routes = gtfs["routes"]
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]
    stops = gtfs["stops"]
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"

    route_row = routes[routes["route_id"].astype(str) == str(route_id)].iloc[0]
    route_short = str(route_row.get("route_short_name", route_id)) or str(route_id)
    line_key = line_key_override or _line_key(route_short, route_id, int(direction), pattern_index, pattern_count, signature)
    route_token = safe_token(line_key if direct_data_dir else route_short)
    pattern_trips = trips[trips["trip_id"].astype(str).isin(pattern_trip_ids)].copy()
    matching_stop_times = stop_times[stop_times["trip_id"].isin(pattern_trips["trip_id"].astype(str))].copy()
    rep_st = representative_stop_times(stop_times, pattern_trips["trip_id"].astype(str).tolist(), signature)

    stop_news, route_news, stop_id_to_internal = build_stop_and_segment_tables(
        rep_st,
        stops,
        args.city,
        route_token,
        int(direction),
        matching_stop_times,
        args.hours,
    )
    internal_names = stop_news["stop_name"].astype(str).tolist()
    timetable_with_extra = build_timetable(matching_stop_times, list(signature), args.hours)
    timetable = timetable_with_extra[["launch_time", "direction"]].copy()

    passenger_od = None
    demand_source = "schedule_proxy"
    if args.ridership_kind == "mbta_bus" and args.ridership_path:
        passenger_od = mbta_hourly_boardings(
            args.ridership_path,
            route_id=str(route_id),
            direction=int(direction),
            internal_names=internal_names,
            stop_id_to_internal=stop_id_to_internal,
            hours=args.hours,
            day_type=args.day_type_name,
            season=args.ridership_season,
        )
        if passenger_od is not None:
            demand_source = "mbta_stop_level_boarding_alighting"
            passenger_od = scale_passenger_od(passenger_od, internal_names, pattern_share)
    elif args.ridership_kind == "halifax_apc" and args.ridership_path:
        hourly = halifax_hourly_totals(args.ridership_path, route_short, args.hours)
        if hourly is not None:
            hourly = {hour: value * float(pattern_share) for hour, value in hourly.items()}
            passenger_od = route_profile_od(hourly, internal_names, args.hours)
            demand_source = "halifax_route_level_half_hour_apc_proxy_od"

    if passenger_od is None:
        hourly = scheduled_proxy_totals(timetable, args.hours, args.proxy_passengers_per_trip)
        passenger_od = route_profile_od(hourly, internal_names, args.hours)
        demand_source = "schedule_headway_proxy_od"

    env_name = f"{safe_token(args.city)}_{route_token}_D{int(direction)}_{args.day_type_name.lower()}"
    if env_dir is None:
        env_dir = args.out.resolve() if args.out else args.gtfs_dir.resolve().parents[0] / "h2o_envs" / env_name
    data_dir = env_dir if direct_data_dir else env_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Keep H2O+-consumed columns first; metadata columns are harmless afterward.
    route_required = ["route_id", "start_stop", "end_stop", "distance", "V_max"] + [
        f"{hour:02d}:00:00" for hour in args.hours
    ]
    route_news = route_news[route_required + [col for col in route_news.columns if col not in route_required]]
    stop_news.to_excel(data_dir / "stop_news.xlsx", index=False)
    route_news.to_excel(data_dir / "route_news.xlsx", index=False)
    timetable.to_excel(data_dir / "time_table.xlsx", index=False)
    passenger_od.to_excel(data_dir / "passenger_OD.xlsx", index=False)

    launch_times = sorted(float(x) for x in timetable["launch_time"].tolist())
    diffs = [b - a for a, b in zip(launch_times[:-1], launch_times[1:]) if b > a]
    line_headway = float(np.median(diffs)) if diffs else 360.0
    fallback_idx = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(route_id))) % 10000
    line_idx = int(re.sub(r"\D", "", str(route_short)) or fallback_idx)
    line_id_str = f"{safe_token(args.city)}:{line_key}"
    max_time = int((max(args.hours) - min(args.hours) + 2) * 3600)
    if write_config_file:
        write_config(
            env_dir,
            line_idx,
            line_id_str,
            line_headway,
            max_time,
            min(args.hours),
            max(args.hours),
        )

    summary = {
        "source": "GTFS + optional public ridership",
        "city": args.city,
        "gtfs_dir": str(args.gtfs_dir.resolve()),
        "service_scope": args.service_scope,
        "service_date": service_date.isoformat() if service_date else None,
        "route_id": str(route_id),
        "route_short_name": route_short,
        "route_long_name": str(route_row.get("route_long_name", "")),
        "direction": int(direction),
        "line_key": line_key,
        "pattern_index": int(pattern_index),
        "pattern_count_for_route_direction": int(pattern_count),
        "pattern_share": float(pattern_share),
        "pattern_trips": len(pattern_trips),
        "timetable_rows": len(timetable),
        "stops": len(stop_news),
        "segments": len(route_news),
        "hours": args.hours,
        "demand_source": demand_source,
        "line_idx": line_idx,
        "line_headway": line_headway,
        "output": str(env_dir),
        "data_dir": str(data_dir),
    }
    if write_config_file:
        (env_dir / "gtfs_conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return env_dir, summary


def build_env(args: argparse.Namespace) -> pathlib.Path:
    gtfs = load_gtfs(args.gtfs_dir)
    routes = gtfs["routes"]
    trips = gtfs["trips"]
    stop_times = gtfs["stop_times"]
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"

    service_date, service_ids = _service_filter(gtfs, args)
    route_id = resolve_route_id(routes, args.route)
    direction = args.direction
    if route_id is None:
        route_id, direction = auto_route_direction(
            routes,
            trips,
            stop_times,
            service_ids,
            min_stops=args.min_stops,
            max_stops=args.max_stops,
        )
    if direction is None:
        direction = 0
    candidates = route_direction_patterns(
        trips,
        stop_times,
        str(route_id),
        int(direction),
        service_ids,
        min_stops=args.min_stops,
        max_stops=args.max_stops,
    )
    if not candidates:
        raise RuntimeError(f"No route pattern found for route_id={route_id}, direction={direction}")
    signature, pattern_trip_ids = candidates[0]
    env_dir, _summary = build_env_for_pattern(
        args,
        gtfs,
        str(route_id),
        int(direction),
        service_ids,
        service_date,
        signature,
        pattern_trip_ids,
    )
    return env_dir


def build_all_envs(args: argparse.Namespace) -> pathlib.Path:
    gtfs = load_gtfs(args.gtfs_dir)
    routes = gtfs["routes"]
    trips = gtfs["trips"].copy()
    stop_times = gtfs["stop_times"]
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"

    service_date, service_ids = _service_filter(gtfs, args)
    active_trips = trips[trips["service_id"].isin(service_ids)].copy() if service_ids else trips.copy()
    active_trips["_direction"] = pd.to_numeric(active_trips["direction_id"].replace("", "0"), errors="coerce").fillna(0).astype(int)

    bus_routes = routes.copy()
    if "route_type" in bus_routes.columns:
        bus_routes = bus_routes[bus_routes["route_type"].astype(str) == "3"]
    if args.route:
        wanted = resolve_route_id(routes, args.route)
        bus_routes = bus_routes[bus_routes["route_id"].astype(str) == str(wanted)]

    bundle_name = f"{safe_token(args.city)}_{args.day_type_name.lower()}_all_routes"
    bundle_dir = args.out.resolve() if args.out else args.gtfs_dir.resolve().parents[0] / "h2o_city_envs" / bundle_name
    (bundle_dir / "data").mkdir(parents=True, exist_ok=True)

    write_config(
        bundle_dir,
        line_idx=0,
        line_id_str=f"{safe_token(args.city)}:ALL",
        line_headway=360.0,
        max_time=int((max(args.hours) - min(args.hours) + 2) * 3600),
        sim_start_hour=min(args.hours),
        sim_end_hour=max(args.hours),
    )

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    used_line_keys: set[str] = set()
    for route_id in bus_routes["route_id"].astype(str).unique():
        route_trips = active_trips[active_trips["route_id"].astype(str) == str(route_id)]
        if route_trips.empty:
            continue
        directions = [int(args.direction)] if args.direction is not None else sorted(route_trips["_direction"].dropna().astype(int).unique().tolist())
        for direction in directions:
            try:
                patterns = route_direction_patterns(
                    trips,
                    stop_times,
                    route_id,
                    int(direction),
                    service_ids,
                    min_stops=args.min_stops,
                    max_stops=args.max_stops,
                )
                if args.max_patterns_per_route_direction > 0:
                    patterns = patterns[: args.max_patterns_per_route_direction]
                if not patterns:
                    continue
                total_pattern_trips = sum(len(trip_ids) for _sig, trip_ids in patterns)
                route_row = routes[routes["route_id"].astype(str) == str(route_id)].iloc[0]
                route_short = str(route_row.get("route_short_name", route_id)) or str(route_id)
                for pattern_index, (signature, pattern_trip_ids) in enumerate(patterns):
                    base_line_key = _line_key(route_short, route_id, int(direction), pattern_index, len(patterns), signature)
                    line_key = _unique_line_key(base_line_key, route_id, used_line_keys)
                    line_dir = bundle_dir / "data" / line_key
                    _env_dir, summary = build_env_for_pattern(
                        args,
                        gtfs,
                        route_id,
                        int(direction),
                        service_ids,
                        service_date,
                        signature,
                        pattern_trip_ids,
                        pattern_index=pattern_index,
                        pattern_count=len(patterns),
                        pattern_share=(len(pattern_trip_ids) / total_pattern_trips) if total_pattern_trips else 1.0,
                        env_dir=line_dir,
                        direct_data_dir=True,
                        write_config_file=False,
                        line_key_override=line_key,
                    )
                    if line_key != base_line_key:
                        summary["base_line_key"] = base_line_key
                    summaries.append(summary)
            except Exception as exc:
                failures.append({"route_id": str(route_id), "direction": int(direction), "error": str(exc)})

    manifest = {
        "source": "GTFS + optional public ridership",
        "city": args.city,
        "gtfs_dir": str(args.gtfs_dir.resolve()),
        "service_scope": args.service_scope,
        "service_date": service_date.isoformat() if service_date else None,
        "hours": args.hours,
        "day_type_name": args.day_type_name,
        "ridership_kind": args.ridership_kind,
        "line_count": len(summaries),
        "failure_count": len(failures),
        "lines": summaries,
        "failures": failures,
        "output": str(bundle_dir),
    }
    (bundle_dir / "gtfs_city_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["city", "service_scope", "line_count", "failure_count", "output"]}, indent=2))
    return bundle_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--gtfs-dir", type=pathlib.Path, required=True)
    parser.add_argument("--all-routes", action="store_true", help="Export every valid bus route-direction-pattern as a MultiLineEnv city bundle.")
    parser.add_argument("--route", default=None, help="GTFS route_id or route_short_name. Omit for auto-select.")
    parser.add_argument("--direction", type=int, default=None)
    parser.add_argument("--service-date", default=None, help="YYYYMMDD or YYYY-MM-DD. Defaults to a valid weekday in the feed.")
    parser.add_argument(
        "--service-scope",
        choices=["all-feed", "service-date"],
        default="all-feed",
        help="all-feed uses all GTFS trips in the feed; service-date restricts to one date.",
    )
    parser.add_argument("--hours", type=parse_hours, default=DEFAULT_HOURS)
    parser.add_argument("--day-type-name", default="Weekday")
    parser.add_argument("--min-stops", type=int, default=18)
    parser.add_argument("--max-stops", type=int, default=120)
    parser.add_argument("--max-patterns-per-route-direction", type=int, default=0, help="0 means keep every valid pattern.")
    parser.add_argument("--ridership-kind", choices=["none", "mbta_bus", "halifax_apc"], default="none")
    parser.add_argument("--ridership-path", type=pathlib.Path, default=None)
    parser.add_argument("--ridership-season", default=None)
    parser.add_argument("--proxy-passengers-per-trip", type=float, default=DEFAULT_PROXY_PASSENGERS_PER_TRIP)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.all_routes:
        build_all_envs(args)
    else:
        build_env(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
