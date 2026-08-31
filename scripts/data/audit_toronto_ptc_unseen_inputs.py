#!/usr/bin/env python3
"""Audit frozen Toronto v105 inputs in parallel before SUMO construction."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import re
import tarfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile


PROTOCOL = "cfcmt-toronto-ptc-unseen-input-audit-v1"
EXPECTED_MISSING_DATES = {
    date(2025, 5, 26),
    date(2025, 5, 27),
    date(2025, 5, 28),
}
EXPECTED_PTC_DATES = {
    date(2025, 1, 1).fromordinal(ordinal)
    for ordinal in range(date(2025, 1, 1).toordinal(), date(2026, 1, 1).toordinal())
} - EXPECTED_MISSING_DATES
EXPECTED_PTC_ENTRIES = tuple(
    f"trips_2025{month:02d}.csv" for month in range(1, 13)
)
PTC_COLUMNS = (
    "dt",
    "pickup_hr",
    "pickup_municipality",
    "pickup_community_council",
    "pickup_ward",
    "dropoff_municipality",
    "dropoff_community_council",
    "dropoff_ward",
    "trips_total",
    "fare_avg",
    "distance_avg",
    "waittime_avg",
    "duration_avg",
)
SUMMARY_REQUIRED_COLUMNS = {
    "dt",
    "reported_trips_started",
    "active_vehicles",
    "dist_ontrip_routed",
}
MIDBLOCK_SUMMARY_REQUIRED_COLUMNS = {
    "count_id",
    "count_date_start",
    "count_date_end",
    "count_duration",
    "longitude",
    "latitude",
    "centreline_id",
    "avg_daily_vol",
    "avg_speed",
}
MIDBLOCK_SPEED_COLUMNS = (
    "id",
    "count_id",
    "location_name",
    "longitude",
    "latitude",
    "centreline_id",
    "time_start",
    "time_end",
    "direction",
    "vol_1_19kph",
    "vol_20_25kph",
    "vol_26_30kph",
    "vol_31_35kph",
    "vol_36_40kph",
    "vol_41_45kph",
    "vol_46_50kph",
    "vol_51_55kph",
    "vol_56_60kph",
    "vol_61_65kph",
    "vol_66_70kph",
    "vol_71_75kph",
    "vol_76_80kph",
    "vol_81_160kph",
)
TIMING_REQUIRED_COLUMNS = {
    "TCS",
    "PHASE",
    "LAST_UPDATED",
    "PHASE_STATUS",
    "PHASE_PED_STATUS",
}
DRIVABLE_CLASS_COUNTS = {
    201100: ("Expressway", 1_189),
    201101: ("Expressway Ramp", 1_134),
    201200: ("Major Arterial", 5_837),
    201201: ("Major Arterial Ramp", 171),
    201300: ("Minor Arterial", 3_550),
    201301: ("Minor Arterial Ramp", 1),
    201400: ("Collector", 6_357),
    201401: ("Collector Ramp", 16),
    201500: ("Local", 24_844),
    201600: ("Other", 2_102),
    201601: ("Other Ramp", 6),
    201700: ("Laneway", 4_146),
    201801: ("Busway", 18),
    201803: ("Access Road", 57),
}
EXPECTED_DRIVABLE_FEATURE_COUNT = 49_428
WEEKDAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def normalize_ward(value: str) -> str:
    text = " ".join(value.strip().split())
    return re.sub(r"^\d{1,2}\s*-\s*", "", text).casefold()


def normalize_council(value: str) -> str:
    text = " ".join(value.strip().split())
    text = re.sub(r"\s+community\s+council$", "", text, flags=re.IGNORECASE)
    return text.casefold()


def _is_specific(value: str) -> bool:
    text = value.strip().casefold()
    return bool(text) and text not in {"not included elsewhere", "none", "null"}


def _load_geojson(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"not a GeoJSON FeatureCollection: {path.name}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON features are not a list: {path.name}")
    return features


def _feature_properties(features: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for feature in features:
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("GeoJSON feature lacks properties")
        yield properties


def audit_network_geographies(root: str) -> dict[str, Any]:
    base = Path(root)
    centreline = _load_geojson(base / "toronto_centreline_4326.geojson")
    wards = _load_geojson(base / "city_wards_4326.geojson")
    councils = _load_geojson(base / "community_councils_4326.geojson")
    signals = _load_geojson(base / "traffic_signals_4326.geojson")

    class_counts: Counter[int] = Counter()
    class_names: dict[int, set[str]] = {}
    directions: Counter[str] = Counter()
    centreline_ids: set[int] = set()
    intersection_ids: set[int] = set()
    duplicate_centreline_ids = 0
    for row in _feature_properties(centreline):
        code = int(row["FEATURE_CODE"])
        class_counts[code] += 1
        class_names.setdefault(code, set()).add(str(row["FEATURE_CODE_DESC"]))
        directions[f"{row.get('ONEWAY_DIR_CODE')}|{row.get('ONEWAY_DIR_CODE_DESC')}"] += 1
        centreline_id = int(row["CENTRELINE_ID"])
        if centreline_id in centreline_ids:
            duplicate_centreline_ids += 1
        centreline_ids.add(centreline_id)
        intersection_ids.add(int(row["FROM_INTERSECTION_ID"]))
        intersection_ids.add(int(row["TO_INTERSECTION_ID"]))

    class_failures = []
    for code, (expected_name, expected_count) in DRIVABLE_CLASS_COUNTS.items():
        observed_names = class_names.get(code, set())
        observed_count = class_counts.get(code, 0)
        if observed_names != {expected_name} or observed_count != expected_count:
            class_failures.append(
                {
                    "code": code,
                    "expected_name": expected_name,
                    "observed_names": sorted(observed_names),
                    "expected_count": expected_count,
                    "observed_count": observed_count,
                }
            )
    drivable_total = sum(class_counts.get(code, 0) for code in DRIVABLE_CLASS_COUNTS)

    ward_names = {
        normalize_ward(str(row["AREA_NAME"]))
        for row in _feature_properties(wards)
    }
    council_names = {
        normalize_council(str(row["AREA_NAME"]))
        for row in _feature_properties(councils)
    }

    signal_node_ids: set[int] = set()
    duplicate_signal_nodes = 0
    missing_signal_nodes = 0
    for row in _feature_properties(signals):
        raw_node_id = row.get("NODE_ID")
        if raw_node_id is None:
            missing_signal_nodes += 1
            continue
        node_id = int(raw_node_id)
        if node_id in signal_node_ids:
            duplicate_signal_nodes += 1
        signal_node_ids.add(node_id)
    matched_signal_nodes = signal_node_ids & intersection_ids

    failures = []
    if len(centreline) != 64_341:
        failures.append(f"centreline_feature_count={len(centreline)} expected=64341")
    if class_failures:
        failures.append("drivable_class_identity_mismatch")
    if drivable_total != EXPECTED_DRIVABLE_FEATURE_COUNT:
        failures.append(
            f"drivable_feature_count={drivable_total} "
            f"expected={EXPECTED_DRIVABLE_FEATURE_COUNT}"
        )
    if duplicate_centreline_ids:
        failures.append(f"duplicate_centreline_ids={duplicate_centreline_ids}")
    if len(wards) != 25 or len(ward_names) != 25:
        failures.append(f"ward_inventory={len(wards)}/{len(ward_names)} expected=25/25")
    if len(councils) != 4 or len(council_names) != 4:
        failures.append(
            f"council_inventory={len(councils)}/{len(council_names)} expected=4/4"
        )
    if duplicate_signal_nodes:
        failures.append(f"duplicate_signal_nodes={duplicate_signal_nodes}")
    if not matched_signal_nodes:
        failures.append("no_signal_nodes_match_centreline_intersections")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "centreline_feature_count": len(centreline),
        "centreline_id_count": len(centreline_ids),
        "intersection_id_count": len(intersection_ids),
        "feature_code_counts": {
            str(code): count for code, count in sorted(class_counts.items())
        },
        "feature_code_names": {
            str(code): sorted(names) for code, names in sorted(class_names.items())
        },
        "drivable_feature_count": drivable_total,
        "drivable_class_failures": class_failures,
        "direction_inventory": dict(sorted(directions.items())),
        "ward_count": len(wards),
        "ward_names": sorted(ward_names),
        "community_council_count": len(councils),
        "community_council_names": sorted(council_names),
        "signal_feature_count": len(signals),
        "signal_node_count": len(signal_node_ids),
        "signal_without_node_id_count": missing_signal_nodes,
        "matched_signal_node_count": len(matched_signal_nodes),
        "unmatched_signal_node_count": len(signal_node_ids - intersection_ids),
        "total_unmatched_source_signal_count": (
            missing_signal_nodes + len(signal_node_ids - intersection_ids)
        ),
    }


def _resolved_geography(
    *,
    ward: str,
    council: str,
    ward_names: set[str],
    council_names: set[str],
) -> tuple[str, str] | None:
    if _is_specific(ward):
        normalized = normalize_ward(ward)
        if normalized in ward_names:
            return "ward", normalized
        return None
    if _is_specific(council):
        normalized = normalize_council(council)
        if normalized in council_names:
            return "community_council", normalized
    return None


def _parse_ptc_hour(value: str) -> tuple[date, int, str]:
    if len(value) < 22:
        raise ValueError(f"invalid pickup_hr={value!r}")
    local = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
    offset = value[19:]
    if offset not in {"-05", "-04", "-05:00", "-04:00"}:
        raise ValueError(f"unexpected Toronto UTC offset={offset!r}")
    return local.date(), local.hour, offset


def audit_ptc_trips(root: str) -> dict[str, Any]:
    base = Path(root)
    ward_features = _load_geojson(base / "city_wards_4326.geojson")
    council_features = _load_geojson(base / "community_councils_4326.geojson")
    ward_names = {
        normalize_ward(str(row["AREA_NAME"]))
        for row in _feature_properties(ward_features)
    }
    council_names = {
        normalize_council(str(row["AREA_NAME"]))
        for row in _feature_properties(council_features)
    }

    row_count = 0
    source_trip_total = 0
    eligible_row_count = 0
    eligible_trip_total = 0
    invalid_rows = 0
    invalid_examples: list[dict[str, Any]] = []
    unresolved_rows = 0
    unresolved_trip_total = 0
    unresolved_examples: list[dict[str, Any]] = []
    dates: set[date] = set()
    eligible_dates: set[date] = set()
    timezone_offsets: Counter[str] = Counter()
    pickup_municipalities: Counter[str] = Counter()
    dropoff_municipalities: Counter[str] = Counter()
    origin_levels: Counter[str] = Counter()
    destination_levels: Counter[str] = Counter()
    eligible_rows_by_weekday: Counter[str] = Counter()
    eligible_trips_by_weekday: Counter[str] = Counter()
    monthly_rows: dict[str, int] = {}
    monthly_trips: dict[str, int] = {}

    archive_path = base / "trips_2025.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = tuple(archive.namelist())
        entry_failure = entries != EXPECTED_PTC_ENTRIES
        for expected_month, name in enumerate(EXPECTED_PTC_ENTRIES, start=1):
            month_rows = 0
            month_trips = 0
            with archive.open(name) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    reader = csv.DictReader(text)
                    if tuple(reader.fieldnames or ()) != PTC_COLUMNS:
                        raise ValueError(
                            f"PTC header mismatch in {name}: {reader.fieldnames}"
                        )
                    for row in reader:
                        row_count += 1
                        month_rows += 1
                        try:
                            source_date = date.fromisoformat(row["dt"])
                            pickup_date, pickup_hour, offset = _parse_ptc_hour(
                                row["pickup_hr"]
                            )
                            trips = int(row["trips_total"])
                            if source_date.year != 2025:
                                raise ValueError("source date is outside 2025")
                            if source_date.month != expected_month:
                                raise ValueError("row is in the wrong monthly archive member")
                            if pickup_date != source_date:
                                raise ValueError("pickup_hr local date differs from dt")
                            if not 0 <= pickup_hour <= 23:
                                raise ValueError("pickup hour is outside 0..23")
                            if trips <= 0:
                                raise ValueError("trips_total is not positive")
                        except Exception as exc:
                            invalid_rows += 1
                            if len(invalid_examples) < 10:
                                invalid_examples.append(
                                    {
                                        "entry": name,
                                        "dt": row.get("dt"),
                                        "pickup_hr": row.get("pickup_hr"),
                                        "trips_total": row.get("trips_total"),
                                        "error": str(exc),
                                    }
                                )
                            continue

                        dates.add(source_date)
                        timezone_offsets[offset] += 1
                        source_trip_total += trips
                        month_trips += trips
                        pickup_municipalities[row["pickup_municipality"]] += trips
                        dropoff_municipalities[row["dropoff_municipality"]] += trips

                        if (
                            row["pickup_municipality"] != "Toronto"
                            or row["dropoff_municipality"] != "Toronto"
                        ):
                            continue
                        eligible_dates.add(source_date)
                        origin = _resolved_geography(
                            ward=row["pickup_ward"],
                            council=row["pickup_community_council"],
                            ward_names=ward_names,
                            council_names=council_names,
                        )
                        destination = _resolved_geography(
                            ward=row["dropoff_ward"],
                            council=row["dropoff_community_council"],
                            ward_names=ward_names,
                            council_names=council_names,
                        )
                        if origin is None or destination is None:
                            unresolved_rows += 1
                            unresolved_trip_total += trips
                            if len(unresolved_examples) < 10:
                                unresolved_examples.append(
                                    {
                                        "dt": row["dt"],
                                        "pickup_municipality": row[
                                            "pickup_municipality"
                                        ],
                                        "pickup_ward": row["pickup_ward"],
                                        "pickup_community_council": row[
                                            "pickup_community_council"
                                        ],
                                        "dropoff_ward": row["dropoff_ward"],
                                        "dropoff_municipality": row[
                                            "dropoff_municipality"
                                        ],
                                        "dropoff_community_council": row[
                                            "dropoff_community_council"
                                        ],
                                        "trips_total": trips,
                                    }
                                )
                            continue
                        eligible_row_count += 1
                        eligible_trip_total += trips
                        origin_levels[origin[0]] += trips
                        destination_levels[destination[0]] += trips
                        weekday = WEEKDAY_NAMES[source_date.weekday()]
                        eligible_rows_by_weekday[weekday] += 1
                        eligible_trips_by_weekday[weekday] += trips
            monthly_rows[name] = month_rows
            monthly_trips[name] = month_trips

    observed_missing_dates = sorted(EXPECTED_PTC_DATES - dates)
    unexpected_dates = sorted(dates - EXPECTED_PTC_DATES)
    failures = []
    if entry_failure:
        failures.append("PTC archive member inventory mismatch")
    if invalid_rows:
        failures.append(f"invalid_ptc_rows={invalid_rows}")
    if dates != EXPECTED_PTC_DATES:
        failures.append(
            f"PTC date inventory mismatch: missing={len(observed_missing_dates)} "
            f"unexpected={len(unexpected_dates)}"
        )
    if eligible_dates != EXPECTED_PTC_DATES:
        failures.append(
            f"eligible PTC date inventory={len(eligible_dates)} expected=362"
        )
    if unresolved_rows:
        failures.append(
            f"unresolved Toronto-to-Toronto geographies={unresolved_rows}"
        )
    if eligible_row_count <= 0 or eligible_trip_total <= 0:
        failures.append("no eligible Toronto-to-Toronto demand")
    if set(eligible_trips_by_weekday) != set(WEEKDAY_NAMES):
        failures.append("eligible PTC demand does not cover all weekdays")

    weekday_date_counts = Counter(
        WEEKDAY_NAMES[source_date.weekday()] for source_date in EXPECTED_PTC_DATES
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "archive_entries": list(entries),
        "row_count": row_count,
        "source_trip_total": source_trip_total,
        "date_count": len(dates),
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "expected_missing_dates": sorted(
            value.isoformat() for value in EXPECTED_MISSING_DATES
        ),
        "additional_missing_dates": [value.isoformat() for value in observed_missing_dates],
        "unexpected_dates": [value.isoformat() for value in unexpected_dates],
        "invalid_row_count": invalid_rows,
        "invalid_examples": invalid_examples,
        "eligible_row_count": eligible_row_count,
        "eligible_trip_total": eligible_trip_total,
        "eligible_date_count": len(eligible_dates),
        "unresolved_row_count": unresolved_rows,
        "unresolved_trip_total": unresolved_trip_total,
        "unresolved_examples": unresolved_examples,
        "weekday_observed_date_counts": dict(sorted(weekday_date_counts.items())),
        "eligible_rows_by_weekday": dict(sorted(eligible_rows_by_weekday.items())),
        "eligible_trips_by_weekday": dict(sorted(eligible_trips_by_weekday.items())),
        "origin_trip_totals_by_level": dict(sorted(origin_levels.items())),
        "destination_trip_totals_by_level": dict(sorted(destination_levels.items())),
        "timezone_row_counts": dict(sorted(timezone_offsets.items())),
        "pickup_trip_totals_by_municipality": dict(
            sorted(pickup_municipalities.items())
        ),
        "dropoff_trip_totals_by_municipality": dict(
            sorted(dropoff_municipalities.items())
        ),
        "monthly_rows": monthly_rows,
        "monthly_trips": monthly_trips,
    }


def audit_ptc_summary(root: str) -> dict[str, Any]:
    path = Path(root) / "ptc_summary_stats.csv"
    row_count = 0
    dates_2025: set[date] = set()
    reported_trips_2025 = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(SUMMARY_REQUIRED_COLUMNS - columns)
        for row in reader:
            row_count += 1
            source_date = date.fromisoformat(row["dt"])
            if source_date.year == 2025:
                dates_2025.add(source_date)
                reported_trips_2025 += int(row["reported_trips_started"])
    failures = []
    if missing_columns:
        failures.append(f"missing_columns={missing_columns}")
    additional_missing_dates = sorted(EXPECTED_PTC_DATES - dates_2025)
    unexpected_dates = sorted(dates_2025 - EXPECTED_PTC_DATES)
    if dates_2025 != EXPECTED_PTC_DATES:
        failures.append(
            f"summary 2025 date inventory mismatch: "
            f"missing={len(additional_missing_dates)} "
            f"unexpected={len(unexpected_dates)}"
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "row_count": row_count,
        "date_count_2025": len(dates_2025),
        "additional_missing_dates_2025": [
            value.isoformat() for value in additional_missing_dates
        ],
        "unexpected_dates_2025": [value.isoformat() for value in unexpected_dates],
        "reported_trips_started_2025": reported_trips_2025,
    }


def audit_midblock_summary(root: str) -> dict[str, Any]:
    path = Path(root) / "svc_summary_data.csv"
    row_count = 0
    count_ids: set[str] = set()
    dates: list[date] = []
    durations: Counter[str] = Counter()
    invalid_coordinates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(MIDBLOCK_SUMMARY_REQUIRED_COLUMNS - columns)
        for row in reader:
            row_count += 1
            count_ids.add(row["count_id"])
            dates.append(date.fromisoformat(row["count_date_start"]))
            durations[row["count_duration"]] += 1
            try:
                longitude = float(row["longitude"])
                latitude = float(row["latitude"])
                if not (-80.0 < longitude < -78.0 and 43.0 < latitude < 44.5):
                    invalid_coordinates += 1
            except ValueError:
                invalid_coordinates += 1
    failures = []
    if missing_columns:
        failures.append(f"missing_columns={missing_columns}")
    if row_count <= 0 or not count_ids:
        failures.append("empty midblock summary")
    if invalid_coordinates:
        failures.append(f"invalid_coordinates={invalid_coordinates}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "row_count": row_count,
        "count_id_count": len(count_ids),
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "duration_inventory": dict(sorted(durations.items())),
        "invalid_coordinate_count": invalid_coordinates,
    }


def audit_midblock_speed(root: str) -> dict[str, Any]:
    path = Path(root) / "svc_raw_data_speed_2025_2029.csv"
    row_count = 0
    rows_2025 = 0
    count_ids_2025: set[str] = set()
    dates_2025: set[date] = set()
    interval_minutes: Counter[int] = Counter()
    volume_total_2025 = 0
    invalid_rows = 0
    volume_columns = MIDBLOCK_SPEED_COLUMNS[9:]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        for row in reader:
            row_count += 1
            try:
                start = datetime.fromisoformat(row["time_start"])
                end = datetime.fromisoformat(row["time_end"])
                minutes = int((end - start).total_seconds() // 60)
                if minutes <= 0:
                    raise ValueError("nonpositive interval")
                volumes = [int(row[column]) for column in volume_columns]
                if any(value < 0 for value in volumes):
                    raise ValueError("negative speed-bin volume")
            except Exception:
                invalid_rows += 1
                continue
            if start.year == 2025:
                rows_2025 += 1
                count_ids_2025.add(row["count_id"])
                dates_2025.add(start.date())
                interval_minutes[minutes] += 1
                volume_total_2025 += sum(volumes)
    failures = []
    if header != MIDBLOCK_SPEED_COLUMNS:
        failures.append("midblock speed header mismatch")
    if invalid_rows:
        failures.append(f"invalid_speed_rows={invalid_rows}")
    if rows_2025 <= 0 or not count_ids_2025 or not dates_2025:
        failures.append("no valid 2025 midblock speed observations")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "row_count": row_count,
        "row_count_2025": rows_2025,
        "count_id_count_2025": len(count_ids_2025),
        "date_count_2025": len(dates_2025),
        "date_min_2025": min(dates_2025).isoformat() if dates_2025 else None,
        "date_max_2025": max(dates_2025).isoformat() if dates_2025 else None,
        "interval_minutes_2025": {
            str(key): value for key, value in sorted(interval_minutes.items())
        },
        "speed_bin_vehicle_total_2025": volume_total_2025,
        "invalid_row_count": invalid_rows,
    }


def audit_signal_timing(root: str) -> dict[str, Any]:
    path = Path(root) / "traffic_signal_timing.zip"
    row_count = 0
    tcs_ids: set[str] = set()
    phases: set[str] = set()
    invalid_rows = 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        with archive.open("traffic-signals-timing.csv") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                reader = csv.DictReader(text)
                columns = set(reader.fieldnames or ())
                missing_columns = sorted(TIMING_REQUIRED_COLUMNS - columns)
                for row in reader:
                    row_count += 1
                    try:
                        int(row["TCS"])
                        int(row["PHASE"])
                    except ValueError:
                        invalid_rows += 1
                        continue
                    tcs_ids.add(row["TCS"])
                    phases.add(row["PHASE"])
    failures = []
    if entries != ["traffic-signals-timing.csv"]:
        failures.append(f"timing archive entries={entries}")
    if missing_columns:
        failures.append(f"missing_columns={missing_columns}")
    if invalid_rows:
        failures.append(f"invalid_timing_rows={invalid_rows}")
    if row_count <= 0 or not tcs_ids:
        failures.append("empty signal timing inventory")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "archive_entries": entries,
        "row_count": row_count,
        "tcs_id_count": len(tcs_ids),
        "phase_inventory": sorted(phases, key=int),
        "invalid_row_count": invalid_rows,
    }


def audit_framework(root: str) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "acquisition_manifest.json").read_text())
    framework = manifest["framework"]
    path = base / framework["output_name"]
    with tarfile.open(path, "r:gz") as archive:
        entries = archive.getnames()
    expected_prefix = (
        "TorontoSUMONetworks-"
        + "7975f1aa01eeac62ba395e9ae66f59e5b0b1a5a9"
    )
    failures = []
    if len(entries) != framework["entry_count"]:
        failures.append(
            f"framework_entry_count={len(entries)} "
            f"expected={framework['entry_count']}"
        )
    if not entries or any(
        entry != expected_prefix and not entry.startswith(expected_prefix + "/")
        for entry in entries
    ):
        failures.append("framework archive commit prefix mismatch")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "entry_count": len(entries),
        "expected_prefix": expected_prefix,
    }


def audit_file_inventory(root: str) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "acquisition_manifest.json").read_text())
    expected = {
        manifest["framework"]["output_name"]: manifest["framework"]["size_bytes"]
    }
    expected.update(
        {
            row["output_name"]: row["size_bytes"]
            for row in manifest["resources"].values()
        }
    )
    missing = []
    mismatched = []
    for name, expected_size in expected.items():
        path = base / name
        if not path.is_file():
            missing.append(name)
            continue
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            mismatched.append(
                {
                    "name": name,
                    "expected_size": expected_size,
                    "observed_size": observed_size,
                }
            )
    observed_files = sorted(path.name for path in base.iterdir() if path.is_file())
    expected_files = sorted([*expected, "acquisition_manifest.json"])
    unexpected = sorted(set(observed_files) - set(expected_files))
    failures = []
    if missing:
        failures.append(f"missing_files={missing}")
    if mismatched:
        failures.append(f"size_mismatches={len(mismatched)}")
    if unexpected:
        failures.append(f"unexpected_files={unexpected}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "expected_file_count": len(expected_files),
        "observed_file_count": len(observed_files),
        "missing_files": missing,
        "size_mismatches": mismatched,
        "unexpected_files": unexpected,
    }


AUDITS: tuple[tuple[str, Callable[[str], dict[str, Any]]], ...] = (
    ("file_inventory", audit_file_inventory),
    ("framework", audit_framework),
    ("network_geographies", audit_network_geographies),
    ("ptc_trips", audit_ptc_trips),
    ("ptc_summary", audit_ptc_summary),
    ("midblock_summary", audit_midblock_summary),
    ("midblock_speed", audit_midblock_speed),
    ("signal_timing", audit_signal_timing),
)


def run_audit(*, input_root: Path, workers: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(function, str(input_root)): name
            for name, function in AUDITS
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {
                    "status": "FAIL",
                    "failures": [f"{type(exc).__name__}: {exc}"],
                }
    ordered_results = {name: results[name] for name, _ in AUDITS}
    failures = {
        name: result["failures"]
        for name, result in ordered_results.items()
        if result["status"] != "PASS"
    }
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "workers": workers,
        "status": "PASS" if not failures else "FAIL",
        "gate_failures": failures,
        "audits": ordered_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers != 8:
        raise ValueError("Toronto v105 input audit is frozen at exactly 8 workers")
    payload = run_audit(input_root=args.input_root, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol": payload["protocol"],
                "gate_failures": payload["gate_failures"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
