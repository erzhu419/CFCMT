"""
Convert downloaded Singapore LTA DataMall bus data into H2O+ BusSim data.

This builds the four Excel files expected by sim_core/sim.py:
    data/stop_news.xlsx
    data/route_news.xlsx
    data/time_table.xlsx
    data/passenger_OD.xlsx

For validation data, use ``--all-services`` to write a city-level MultiLineEnv
bundle with one ``data/<service-direction>/`` directory per LTA bus direction.
The original LTA BusStopCode is kept in extra columns because sim_core slices
OD columns by stop-name labels and therefore needs labels whose lexicographic
order matches route order.

Typical use after lta_datamall_fetch.py:
    python H2Oplus/bus_h2o/data/lta_to_h2o_xlsx.py \\
        --raw-root H2Oplus/downloads/lta_datamall/202604 \\
        --service 187 --direction 1
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HOURS = list(range(6, 20))
DEFAULT_DAY_TYPE = "WEEKDAY"
DEFAULT_BASE_SPEED = 10.0
MIN_SEGMENT_METERS = 50.0
_LTA_OD_CACHE: dict[tuple[str, str, tuple[int, ...], bool], tuple[pd.DataFrame, int, int]] = {}


def parse_hours(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            values.extend(range(int(start_s), int(end_s) + 1))
        else:
            values.append(int(part))
    result = sorted(set(values))
    if any(h < 0 or h > 23 for h in result):
        raise ValueError(f"Invalid hour spec: {spec}")
    return result


def safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(text)).upper()


def load_json_value(path: pathlib.Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("value", [])
    if not isinstance(value, list):
        raise RuntimeError(f"{path} does not contain a list field named 'value'")
    return value


def parse_hhmm(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    text = text.zfill(4)
    hour = int(text[:2])
    minute = int(text[2:])
    return hour * 3600 + minute * 60


def parse_freq_minutes(value: Any, fallback: float = 10.0) -> float:
    if value is None or pd.isna(value):
        return fallback
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if not nums:
        return fallback
    return float(sum(nums) / len(nums))


def weekday_count(year_month: str) -> int:
    year, month = [int(x) for x in year_month.split("-")]
    _, ndays = calendar.monthrange(year, month)
    return sum(1 for day in range(1, ndays + 1) if dt.date(year, month, day).weekday() < 5)


def weekend_count(year_month: str) -> int:
    year, month = [int(x) for x in year_month.split("-")]
    _, ndays = calendar.monthrange(year, month)
    return sum(1 for day in range(1, ndays + 1) if dt.date(year, month, day).weekday() >= 5)


def day_type_divisor(year_month: str, day_type: str) -> int:
    if day_type.upper() == "WEEKDAY":
        return max(1, weekday_count(year_month))
    return max(1, weekend_count(year_month))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_traffic_speed(
    traffic: pd.DataFrame | None,
    lat: float,
    lon: float,
    default_speed: float,
) -> tuple[float, str | None, str | None, float | None]:
    if traffic is None or traffic.empty:
        return default_speed, None, None, None

    # Singapore is small, so this vectorized equirectangular approximation is
    # accurate enough for nearest-link matching at the bus-segment scale.
    lat_arr = traffic["_mid_lat"].to_numpy(dtype=float)
    lon_arr = traffic["_mid_lon"].to_numpy(dtype=float)
    x = (lon_arr - lon) * math.cos(math.radians(lat))
    y = lat_arr - lat
    dist_m = np.sqrt(x * x + y * y) * 111_320.0
    idx = int(np.nanargmin(dist_m))
    row = traffic.iloc[idx]
    speed = float(row["_speed_mps"]) if pd.notna(row["_speed_mps"]) else default_speed
    return (
        max(2.0, min(15.0, speed)),
        str(row.get("LinkID")) if "LinkID" in row else None,
        str(row.get("RoadName")) if "RoadName" in row else None,
        float(dist_m[idx]),
    )


def load_latest_traffic(raw_root: pathlib.Path) -> pd.DataFrame | None:
    paths = sorted((raw_root / "traffic").glob("*/TrafficSpeedBands.json"))
    if not paths:
        return None
    records = load_json_value(paths[-1])
    df = pd.DataFrame(records)
    required = {"StartLat", "StartLon", "EndLat", "EndLon", "MinimumSpeed", "MaximumSpeed"}
    if df.empty or not required.issubset(df.columns):
        return None
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_mid_lat"] = (df["StartLat"] + df["EndLat"]) / 2.0
    df["_mid_lon"] = (df["StartLon"] + df["EndLon"]) / 2.0
    df["_speed_mps"] = ((df["MinimumSpeed"] + df["MaximumSpeed"]) / 2.0) / 3.6
    return df.dropna(subset=["_mid_lat", "_mid_lon"])


def build_timetable(route_rows: pd.DataFrame, service_row: pd.Series | None, hours: list[int]) -> pd.DataFrame:
    first = parse_hhmm(route_rows.iloc[0].get("WD_FirstBus"))
    last = parse_hhmm(route_rows.iloc[0].get("WD_LastBus"))
    window_start = min(hours) * 3600
    window_end = (max(hours) + 1) * 3600
    start_abs = max(first if first is not None else window_start, window_start)
    end_abs = min(last if last is not None else window_end, window_end)
    if end_abs <= start_abs:
        end_abs = window_end

    if service_row is None:
        am_peak = offpeak = pm_peak = 10.0
    else:
        am_peak = parse_freq_minutes(service_row.get("AM_Peak_Freq"), 10.0)
        offpeak = parse_freq_minutes(service_row.get("AM_Offpeak_Freq"), 12.0)
        pm_peak = parse_freq_minutes(service_row.get("PM_Peak_Freq"), am_peak)

    rows: list[dict[str, int]] = []
    t = start_abs
    while t < end_abs:
        hour = int(t // 3600)
        if 6 <= hour <= 8:
            headway_min = am_peak
        elif 17 <= hour <= 19:
            headway_min = pm_peak
        else:
            headway_min = offpeak
        rows.append({"launch_time": int(t - window_start), "direction": 1})
        t += max(60.0, headway_min * 60.0)

    return pd.DataFrame(rows)


def find_csv(raw_root: pathlib.Path, stem: str) -> pathlib.Path:
    matches = sorted((raw_root / "passenger_volume").rglob(f"{stem}.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find {stem}.csv under {raw_root / 'passenger_volume'}")
    return matches[-1]


def build_od_matrix(
    csv_path: pathlib.Path,
    ordered_codes: list[str],
    code_to_name: dict[str, str],
    day_type: str,
    hours: list[int],
    divide_by_day_count: bool,
) -> pd.DataFrame:
    wanted_codes = set(ordered_codes)
    wanted_hours = set(hours)
    code_rank = {code: i for i, code in enumerate(ordered_codes)}
    matrices = {
        hour: pd.DataFrame(0.0, index=list(code_to_name.values()), columns=list(code_to_name.values()))
        for hour in hours
    }
    cache_key = (str(csv_path.resolve()), day_type.upper(), tuple(hours), bool(divide_by_day_count))
    if cache_key not in _LTA_OD_CACHE:
        chunksize = 1_000_000
        total_rows = 0
        year_month_seen: str | None = None
        grouped_parts = []

        dtype = {
            "YEAR_MONTH": "string",
            "DAY_TYPE": "string",
            "TIME_PER_HOUR": "Int64",
            "PT_TYPE": "string",
            "ORIGIN_PT_CODE": "string",
            "DESTINATION_PT_CODE": "string",
            "TOTAL_TRIPS": "float64",
        }

        for chunk in pd.read_csv(csv_path, dtype=dtype, chunksize=chunksize):
            total_rows += len(chunk)
            if year_month_seen is None and not chunk.empty:
                year_month_seen = str(chunk["YEAR_MONTH"].dropna().iloc[0])
            chunk = chunk[
                (chunk["DAY_TYPE"].str.upper() == day_type.upper())
                & (chunk["PT_TYPE"].str.upper() == "BUS")
                & (chunk["TIME_PER_HOUR"].isin(wanted_hours))
            ].copy()
            if chunk.empty:
                continue
            chunk["ORIGIN_PT_CODE"] = chunk["ORIGIN_PT_CODE"].str.zfill(5)
            chunk["DESTINATION_PT_CODE"] = chunk["DESTINATION_PT_CODE"].str.zfill(5)
            grouped_parts.append(
                chunk.groupby(["TIME_PER_HOUR", "ORIGIN_PT_CODE", "DESTINATION_PT_CODE"], as_index=False)[
                    "TOTAL_TRIPS"
                ].sum()
            )

        if grouped_parts:
            grouped_all = pd.concat(grouped_parts, ignore_index=True)
            grouped_all = grouped_all.groupby(["TIME_PER_HOUR", "ORIGIN_PT_CODE", "DESTINATION_PT_CODE"], as_index=False)[
                "TOTAL_TRIPS"
            ].sum()
        else:
            grouped_all = pd.DataFrame(columns=["TIME_PER_HOUR", "ORIGIN_PT_CODE", "DESTINATION_PT_CODE", "TOTAL_TRIPS"])
        divisor = day_type_divisor(year_month_seen, day_type) if divide_by_day_count and year_month_seen else 1
        if divisor != 1:
            grouped_all["TOTAL_TRIPS"] = pd.to_numeric(grouped_all["TOTAL_TRIPS"], errors="coerce").fillna(0.0) / divisor
        _LTA_OD_CACHE[cache_key] = (grouped_all, total_rows, divisor)

    grouped_all, total_rows, divisor = _LTA_OD_CACHE[cache_key]
    sub = grouped_all[
        grouped_all["ORIGIN_PT_CODE"].isin(wanted_codes)
        & grouped_all["DESTINATION_PT_CODE"].isin(wanted_codes)
    ].copy()
    sub = sub[
        sub["DESTINATION_PT_CODE"].map(code_rank).fillna(-1).astype(int)
        > sub["ORIGIN_PT_CODE"].map(code_rank).fillna(-1).astype(int)
    ]
    kept_rows = len(sub)
    for rec in sub.itertuples(index=False):
        hour = int(rec.TIME_PER_HOUR)
        origin = code_to_name[str(rec.ORIGIN_PT_CODE).zfill(5)]
        dest = code_to_name[str(rec.DESTINATION_PT_CODE).zfill(5)]
        matrices[hour].loc[origin, dest] += float(rec.TOTAL_TRIPS)

    rows: list[dict[str, Any]] = []
    internal_names = list(code_to_name.values())
    for hour in hours:
        period = f"{hour:02d}:00:00"
        matrix = matrices[hour]
        for origin in internal_names:
            row: dict[str, Any] = {"time_period": period, "stop_name": origin}
            for dest in internal_names:
                row[dest] = round(float(matrix.loc[origin, dest]), 3)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(
        f"OD rows scanned={total_rows}, matched_route_rows={kept_rows}, "
        f"day_divisor={divisor}, total_daily_route_trips={df[internal_names].to_numpy().sum():.1f}"
    )
    return df


def write_config(
    env_dir: pathlib.Path,
    max_time: int,
    line_idx: int,
    line_id_str: str,
    line_headway: float,
    sim_start_hour: int,
    sim_end_hour: int,
) -> None:
    config = {
        "time_step": 1,
        "route_state_update_freq": 300,
        "passenger_state_update_freq": 20,
        "max_time": max_time,
        "line_idx": line_idx,
        "line_id_str": line_id_str,
        "line_headway": line_headway,
        "sim_start_hour": int(sim_start_hour),
        "sim_end_hour": int(sim_end_hour),
        "use_virtual_colines": False,
    }
    (env_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_env_for_service(
    args: argparse.Namespace,
    raw_root: pathlib.Path,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
    services: pd.DataFrame,
    service_no: str,
    direction: int,
    *,
    env_dir: pathlib.Path | None = None,
    direct_data_dir: bool = False,
    write_config_file: bool = True,
) -> tuple[pathlib.Path, dict[str, Any]]:
    route_rows = routes[
        (routes["ServiceNo"].astype(str) == service_no)
        & (pd.to_numeric(routes["Direction"], errors="coerce").astype("Int64") == direction)
    ].copy()
    if route_rows.empty:
        raise RuntimeError(f"No BusRoutes rows found for service={service_no}, direction={direction}")
    route_rows["StopSequence"] = pd.to_numeric(route_rows["StopSequence"], errors="coerce")
    route_rows["Distance"] = pd.to_numeric(route_rows["Distance"], errors="coerce")
    route_rows = route_rows.sort_values("StopSequence").reset_index(drop=True)

    stops["BusStopCode"] = stops["BusStopCode"].astype(str).str.zfill(5)
    stop_meta = stops.set_index("BusStopCode").to_dict(orient="index")
    ordered_codes = route_rows["BusStopCode"].astype(str).str.zfill(5).tolist()
    prefix = f"SG{safe_token(service_no)}D{direction}"
    internal_names = [f"{prefix}_{i + 1:03d}" for i in range(len(ordered_codes))]
    code_to_name = dict(zip(ordered_codes, internal_names))

    service_rows = services[
        (services["ServiceNo"].astype(str) == service_no)
        & (pd.to_numeric(services["Direction"], errors="coerce").astype("Int64") == direction)
    ]
    service_row = service_rows.iloc[0] if not service_rows.empty else None

    if env_dir is None:
        env_dir = args.out.resolve() if args.out else raw_root / "h2o_envs" / f"{prefix}_{args.day_type.lower()}"
    data_dir = env_dir if direct_data_dir else env_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    stop_records: list[dict[str, Any]] = []
    for idx, (code, internal) in enumerate(zip(ordered_codes, internal_names)):
        meta = stop_meta.get(code, {})
        stop_records.append(
            {
                "stop_id": idx,
                "stop_name": internal,
                "bus_stop_code": code,
                "description": meta.get("Description"),
                "road_name": meta.get("RoadName"),
                "latitude": meta.get("Latitude"),
                "longitude": meta.get("Longitude"),
                "lta_distance_km": float(route_rows.iloc[idx]["Distance"]),
            }
        )
    stop_news = pd.DataFrame(stop_records)

    traffic = load_latest_traffic(raw_root)
    route_records: list[dict[str, Any]] = []
    for i in range(len(route_rows) - 1):
        cur = route_rows.iloc[i]
        nxt = route_rows.iloc[i + 1]
        start_code = str(cur["BusStopCode"]).zfill(5)
        end_code = str(nxt["BusStopCode"]).zfill(5)
        start_meta = stop_meta.get(start_code, {})
        end_meta = stop_meta.get(end_code, {})
        start_lat = float(start_meta.get("Latitude", np.nan))
        start_lon = float(start_meta.get("Longitude", np.nan))
        end_lat = float(end_meta.get("Latitude", np.nan))
        end_lon = float(end_meta.get("Longitude", np.nan))

        distance_m = (float(nxt["Distance"]) - float(cur["Distance"])) * 1000.0
        if not np.isfinite(distance_m) or distance_m <= 0:
            distance_m = haversine_m(start_lat, start_lon, end_lat, end_lon)
        distance_m = max(MIN_SEGMENT_METERS, distance_m)

        mid_lat = (start_lat + end_lat) / 2.0
        mid_lon = (start_lon + end_lon) / 2.0
        speed, link_id, road_name, match_dist = nearest_traffic_speed(traffic, mid_lat, mid_lon, DEFAULT_BASE_SPEED)
        row: dict[str, Any] = {
            "route_id": i,
            "start_stop": code_to_name[start_code],
            "end_stop": code_to_name[end_code],
            "distance": round(distance_m, 2),
            "V_max": round(speed, 2),
        }
        for hour in args.hours:
            row[f"{hour:02d}:00:00"] = round(speed, 2)
        row["lta_start_code"] = start_code
        row["lta_end_code"] = end_code
        row["matched_link_id"] = link_id
        row["matched_road_name"] = road_name
        row["matched_distance_m"] = None if match_dist is None else round(match_dist, 1)
        route_records.append(row)
    route_news = pd.DataFrame(route_records)

    timetable = build_timetable(route_rows, service_row, args.hours)
    od_csv = find_csv(raw_root, f"origin_destination_bus_{args.date}")
    passenger_od = build_od_matrix(
        csv_path=od_csv,
        ordered_codes=ordered_codes,
        code_to_name=code_to_name,
        day_type=args.day_type,
        hours=args.hours,
        divide_by_day_count=not args.keep_monthly_totals,
    )

    # Keep only the columns consumed by Route before extra metadata columns.
    route_required = ["route_id", "start_stop", "end_stop", "distance", "V_max"] + [
        f"{hour:02d}:00:00" for hour in args.hours
    ]
    route_extra = [c for c in route_news.columns if c not in route_required]
    route_news = route_news[route_required + route_extra]

    stop_news.to_excel(data_dir / "stop_news.xlsx", index=False)
    route_news.to_excel(data_dir / "route_news.xlsx", index=False)
    timetable.to_excel(data_dir / "time_table.xlsx", index=False)
    passenger_od.to_excel(data_dir / "passenger_OD.xlsx", index=False)
    max_time = int((max(args.hours) - min(args.hours) + 2) * 3600)
    launch_times = sorted(float(x) for x in timetable["launch_time"].tolist())
    diffs = [b - a for a, b in zip(launch_times[:-1], launch_times[1:]) if b > a]
    line_headway = float(np.median(diffs)) if diffs else 360.0
    line_idx = int(re.sub(r"\D", "", service_no) or 0)
    if write_config_file:
        write_config(
            env_dir,
            max_time=max_time,
            line_idx=line_idx,
            line_id_str=prefix,
            line_headway=line_headway,
            sim_start_hour=min(args.hours),
            sim_end_hour=max(args.hours),
        )

    summary = {
        "source": "Singapore LTA DataMall",
        "service": service_no,
        "direction": direction,
        "day_type": args.day_type,
        "date": args.date,
        "hours": args.hours,
        "stops": len(stop_news),
        "segments": len(route_news),
        "timetable_rows": len(timetable),
        "line_idx": line_idx,
        "line_headway": line_headway,
        "output": str(env_dir),
        "data_dir": str(data_dir),
        "note": "Traffic speeds come from one current TrafficSpeedBands snapshot, nearest-matched to bus-stop segment midpoints.",
    }
    if write_config_file:
        (env_dir / "lta_conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return env_dir, summary


def build_env(args: argparse.Namespace) -> pathlib.Path:
    raw_root = args.raw_root.resolve()
    routes = pd.DataFrame(load_json_value(raw_root / "metadata" / "BusRoutes.json"))
    stops = pd.DataFrame(load_json_value(raw_root / "metadata" / "BusStops.json"))
    services = pd.DataFrame(load_json_value(raw_root / "metadata" / "BusServices.json"))
    env_dir, _summary = build_env_for_service(
        args,
        raw_root,
        routes,
        stops,
        services,
        str(args.service),
        int(args.direction),
    )
    return env_dir


def build_all_envs(args: argparse.Namespace) -> pathlib.Path:
    raw_root = args.raw_root.resolve()
    routes = pd.DataFrame(load_json_value(raw_root / "metadata" / "BusRoutes.json"))
    stops = pd.DataFrame(load_json_value(raw_root / "metadata" / "BusStops.json"))
    services = pd.DataFrame(load_json_value(raw_root / "metadata" / "BusServices.json"))

    routes["_direction"] = pd.to_numeric(routes["Direction"], errors="coerce").astype("Int64")
    pairs = (
        routes[["ServiceNo", "_direction"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["ServiceNo", "_direction"])
    )

    bundle_name = f"SINGAPORE_LTA_{args.day_type.lower().replace('/', '_')}_all_services"
    bundle_dir = args.out.resolve() if args.out else raw_root / "h2o_city_envs" / bundle_name
    (bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    write_config(
        bundle_dir,
        max_time=int((max(args.hours) - min(args.hours) + 2) * 3600),
        line_idx=0,
        line_id_str="SG:ALL",
        line_headway=360.0,
        sim_start_hour=min(args.hours),
        sim_end_hour=max(args.hours),
    )

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, row in pairs.reset_index(drop=True).iterrows():
        if args.max_services > 0 and idx >= args.max_services:
            break
        service_no = str(row["ServiceNo"])
        direction = int(row["_direction"])
        line_key = f"SG{safe_token(service_no)}D{direction}"
        try:
            _env_dir, summary = build_env_for_service(
                args,
                raw_root,
                routes,
                stops,
                services,
                service_no,
                direction,
                env_dir=bundle_dir / "data" / line_key,
                direct_data_dir=True,
                write_config_file=False,
            )
            summary["line_key"] = line_key
            summaries.append(summary)
        except Exception as exc:
            failures.append({"service": service_no, "direction": direction, "error": str(exc)})

    manifest = {
        "source": "Singapore LTA DataMall",
        "raw_root": str(raw_root),
        "day_type": args.day_type,
        "date": args.date,
        "hours": args.hours,
        "line_count": len(summaries),
        "failure_count": len(failures),
        "lines": summaries,
        "failures": failures,
        "output": str(bundle_dir),
        "note": "Passenger OD is monthly DataMall PV/ODBus filtered to each service direction and divided into a typical day average unless --keep-monthly-totals is used.",
    }
    (bundle_dir / "lta_city_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["source", "line_count", "failure_count", "output"]}, indent=2))
    return bundle_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--all-services", action="store_true", help="Export every LTA service-direction as a MultiLineEnv city bundle.")
    parser.add_argument("--max-services", type=int, default=0, help="Debug cap for --all-services; 0 keeps all.")
    parser.add_argument("--service", default=None, help="LTA ServiceNo, e.g. 187")
    parser.add_argument("--direction", type=int, default=1)
    parser.add_argument("--date", default=None, help="YYYYMM; defaults to raw-root directory name")
    parser.add_argument("--day-type", default=DEFAULT_DAY_TYPE, choices=["WEEKDAY", "WEEKENDS/HOLIDAY"])
    parser.add_argument("--hours", type=parse_hours, default=DEFAULT_HOURS, help="Hour list/range, e.g. 6-19")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument(
        "--keep-monthly-totals",
        action="store_true",
        help="Do not divide passenger OD monthly totals into a typical day average.",
    )
    args = parser.parse_args(argv)
    if args.date is None:
        args.date = args.raw_root.name
    if not args.all_services and args.service is None:
        parser.error("--service is required unless --all-services is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.all_services:
        build_all_envs(args)
    else:
        build_env(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
