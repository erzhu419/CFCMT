#!/usr/bin/env python3
"""Expand every locatable Chicago v107 aggregate into deterministic SUMO trips."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import traceback
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import quoteattr


from scripts.data.acquire_chicago_for_hire_unseen import (
    DATES,
    DEFAULT_WORKERS,
    EXPECTED_TRIP_ROWS,
    PROTOCOL as ACQUISITION_PROTOCOL,
    TRIP_SOURCES,
)
from scripts.data.build_chicago_for_hire_network import (
    PROTOCOL as NETWORK_PROTOCOL,
    _area_for_lonlat,
    _community_areas,
    point_in_geometry,
)


PROTOCOL = "cfcmt-chicago-full-week-locatable-demand-v1"
EXPECTED_NETWORK_SHA256 = (
    "45250772dcff2cd79bd4472e0657d09c20eb32be280a1099d5ba93b831ac42ae"
)
EXPECTED_NETWORK_SIZE = 362_471_296
EXPECTED_ANCHOR_SHA256 = (
    "3c3ee079d62ebc0a4df0f7798b464b1934f962b8a74caaf17742059937f8c805"
)
EXPECTED_ANCHOR_SIZE = 857_511
ANCHORS_PER_AREA = 32
CENTROID_ANCHORS = 8
BIN_SECONDS = 900.0


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


def _validate_file(path: Path, row: Mapping[str, Any]) -> None:
    expected_size = int(row["size_bytes"])
    expected_sha = str(row["sha256"])
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"Chicago aggregate size changed: {path}")
    if _file_sha256(path) != expected_sha:
        raise ValueError(f"Chicago aggregate digest changed: {path}")


def _input_manifests(
    acquisition_root: Path, network_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    acquisition_path = acquisition_root / "acquisition_manifest.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("protocol") != ACQUISITION_PROTOCOL:
        raise ValueError("Chicago demand acquisition protocol changed")
    network_path = network_root / "network_manifest.json"
    network = json.loads(network_path.read_text(encoding="utf-8"))
    expected_network = {
        "protocol": NETWORK_PROTOCOL,
        "status": "PASS",
        "network_size_bytes": EXPECTED_NETWORK_SIZE,
        "network_sha256": EXPECTED_NETWORK_SHA256,
        "anchor_file": "anchors.json",
        "anchor_protocol": "cfcmt-chicago-capacity-farthest-passenger-scc-anchors-v3",
    }
    observed_network = {key: network.get(key) for key in expected_network}
    if observed_network != expected_network:
        raise ValueError(f"Chicago network manifest changed: {observed_network}")
    anchor_path = network_root / "anchors.json"
    if (
        not anchor_path.is_file()
        or anchor_path.stat().st_size != EXPECTED_ANCHOR_SIZE
        or _file_sha256(anchor_path) != EXPECTED_ANCHOR_SHA256
    ):
        raise ValueError("Chicago anchor identity changed")
    trip_rows = {
        (str(row["date"]), str(row["source"])): dict(row)
        for row in acquisition.get("daily_files", [])
        if row.get("kind") == "trip_aggregate"
    }
    expected_keys = {
        (day.isoformat(), source.key) for day in DATES for source in TRIP_SOURCES
    }
    if set(trip_rows) != expected_keys:
        raise ValueError("Chicago trip aggregate inventory changed")
    for key, row in trip_rows.items():
        path = acquisition_root / str(row["path"])
        _validate_file(path, row)
        expected_count = EXPECTED_TRIP_ROWS[key[1]][key[0]]
        if int(row["represented_trip_count"]) != expected_count:
            raise ValueError(f"Chicago trip manifest count changed: {key}")
    return acquisition, network, trip_rows


def _load_anchors(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != (
        "cfcmt-chicago-capacity-farthest-passenger-scc-anchors-v3"
    ):
        raise ValueError("Chicago anchor protocol changed")
    rows: dict[int, list[dict[str, Any]]] = {}
    edge_ids: set[str] = set()
    for area in range(1, 78):
        anchors = [dict(row) for row in payload["areas"][str(area)]["anchors"]]
        if len(anchors) != ANCHORS_PER_AREA:
            raise ValueError(f"Chicago area {area} anchor count changed")
        for row in anchors:
            identity = str(row["edge_id"])
            if identity in edge_ids:
                raise ValueError(f"Chicago anchor edge reused across areas: {identity}")
            edge_ids.add(identity)
        rows[area] = anchors
    if len(edge_ids) != 77 * ANCHORS_PER_AREA:
        raise ValueError("Chicago anchor identities are incomplete")
    return rows


def _valid_area(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = int(float(str(value)))
    except ValueError:
        return None
    return number if 1 <= number <= 77 else None


def _finite_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _nearest_centroid_anchors(
    *, lon: float, lat: float, anchors: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    longitude_scale = math.cos(math.radians(lat))
    ranked = sorted(
        anchors,
        key=lambda row: (
            ((float(row["lon"]) - lon) * longitude_scale) ** 2
            + (float(row["lat"]) - lat) ** 2,
            str(row["edge_id"]),
        ),
    )
    return tuple(str(row["edge_id"]) for row in ranked[:CENTROID_ANCHORS])


def resolve_endpoint(
    *,
    area_value: str | None,
    latitude_value: str | None,
    longitude_value: str | None,
    areas: Sequence[Mapping[str, Any]],
    anchors: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any] | None:
    released_area = _valid_area(area_value)
    latitude = _finite_float(latitude_value)
    longitude = _finite_float(longitude_value)
    if latitude is not None and longitude is not None:
        centroid_area = _area_for_lonlat(longitude, latitude, areas)
        if centroid_area is not None:
            return {
                "key": ("centroid", centroid_area, latitude, longitude),
                "area": centroid_area,
                "mode": "centroid",
                "released_area_mismatch": (
                    released_area is not None and released_area != centroid_area
                ),
                "anchors": _nearest_centroid_anchors(
                    lon=longitude,
                    lat=latitude,
                    anchors=anchors[centroid_area],
                ),
            }
    if released_area is not None:
        return {
            "key": ("area", released_area),
            "area": released_area,
            "mode": "community_area_fallback",
            "released_area_mismatch": False,
            "anchors": tuple(
                sorted(str(row["edge_id"]) for row in anchors[released_area])
            ),
        }
    return None


def feasible_anchor_pairs(
    origin: Mapping[str, Any], destination: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    pairs = tuple(
        (source, target)
        for source in sorted(set(origin["anchors"]))
        for target in sorted(set(destination["anchors"]))
        if source != target
    )
    if not pairs:
        raise ValueError(
            f"no distinct Chicago anchor pair for {origin['key']} -> {destination['key']}"
        )
    return pairs


def stratified_departure(
    *, bin_start_sec: float, rank: int, count: int, day_offset_sec: float
) -> float:
    if count <= 0 or not 0 <= rank < count:
        raise ValueError("invalid Chicago aggregate rank")
    return day_offset_sec + bin_start_sec + (rank + 0.5) * BIN_SECONDS / count


def _new_source_stats() -> dict[str, int]:
    return {
        "aggregate_group_count": 0,
        "published_trip_count": 0,
        "included_trip_count": 0,
        "pickup_missing_only_count": 0,
        "dropoff_missing_only_count": 0,
        "both_endpoints_missing_count": 0,
        "pickup_centroid_count": 0,
        "pickup_area_fallback_count": 0,
        "dropoff_centroid_count": 0,
        "dropoff_area_fallback_count": 0,
        "pickup_released_area_mismatch_count": 0,
        "dropoff_released_area_mismatch_count": 0,
    }


def _day_worker(
    *,
    day_text: str,
    day_index: int,
    acquisition_root_text: str,
    boundary_path_text: str,
    anchor_path_text: str,
    output_root_text: str,
    input_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    day = date.fromisoformat(day_text)
    acquisition_root = Path(acquisition_root_text)
    output_root = Path(output_root_text)
    areas = _community_areas(Path(boundary_path_text))
    anchors = _load_anchors(Path(anchor_path_text))
    endpoint_cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
    pair_cache: dict[tuple[Any, ...], tuple[tuple[str, str], ...]] = {}
    records: list[tuple[float, str, str, str]] = []
    source_stats: dict[str, dict[str, int]] = {}

    for source in TRIP_SOURCES:
        source_key = source.key
        row_manifest = input_rows[source_key]
        path = acquisition_root / str(row_manifest["path"])
        stats = _new_source_stats()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for group_index, row in enumerate(reader):
                count = int(row["trip_count"])
                if count <= 0:
                    raise ValueError(f"nonpositive Chicago group count in {path}")
                timestamp = datetime.fromisoformat(row["trip_start_timestamp"])
                if timestamp.date() != day:
                    raise ValueError(f"Chicago aggregate date escaped day: {timestamp}")
                bin_start = (
                    timestamp.hour * 3600
                    + timestamp.minute * 60
                    + timestamp.second
                    + timestamp.microsecond / 1_000_000
                )
                if bin_start < 0 or bin_start >= 86_400:
                    raise ValueError(f"invalid Chicago departure bin: {timestamp}")
                stats["aggregate_group_count"] += 1
                stats["published_trip_count"] += count

                endpoints: list[dict[str, Any] | None] = []
                for prefix in ("pickup", "dropoff"):
                    cache_key = (
                        row[f"{prefix}_community_area"],
                        row[f"{prefix}_centroid_latitude"],
                        row[f"{prefix}_centroid_longitude"],
                    )
                    if cache_key not in endpoint_cache:
                        endpoint_cache[cache_key] = resolve_endpoint(
                            area_value=cache_key[0],
                            latitude_value=cache_key[1],
                            longitude_value=cache_key[2],
                            areas=areas,
                            anchors=anchors,
                        )
                    endpoints.append(endpoint_cache[cache_key])
                origin, destination = endpoints
                if origin is None and destination is None:
                    stats["both_endpoints_missing_count"] += count
                    continue
                if origin is None:
                    stats["pickup_missing_only_count"] += count
                    continue
                if destination is None:
                    stats["dropoff_missing_only_count"] += count
                    continue

                stats["included_trip_count"] += count
                stats[f"pickup_{origin['mode'].replace('community_area_', 'area_')}_count"] += count
                stats[f"dropoff_{destination['mode'].replace('community_area_', 'area_')}_count"] += count
                stats["pickup_released_area_mismatch_count"] += (
                    count if origin["released_area_mismatch"] else 0
                )
                stats["dropoff_released_area_mismatch_count"] += (
                    count if destination["released_area_mismatch"] else 0
                )
                pair_key = (origin["key"], destination["key"])
                if pair_key not in pair_cache:
                    pair_cache[pair_key] = feasible_anchor_pairs(origin, destination)
                pairs = pair_cache[pair_key]
                for rank in range(count):
                    source_edge, target_edge = pairs[rank % len(pairs)]
                    depart = stratified_departure(
                        bin_start_sec=bin_start,
                        rank=rank,
                        count=count,
                        day_offset_sec=day_index * 86_400.0,
                    )
                    identity = (
                        f"d{day_index}_{source_key}_g{group_index:07d}_r{rank:05d}"
                    )
                    records.append((depart, identity, source_edge, target_edge))
        expected = EXPECTED_TRIP_ROWS[source_key][day_text]
        if stats["published_trip_count"] != expected:
            raise ValueError(
                f"Chicago source count changed for {source_key} {day}: "
                f"{stats['published_trip_count']} != {expected}"
            )
        classified = (
            stats["included_trip_count"]
            + stats["pickup_missing_only_count"]
            + stats["dropoff_missing_only_count"]
            + stats["both_endpoints_missing_count"]
        )
        if classified != stats["published_trip_count"]:
            raise ValueError(f"Chicago endpoint classification failed for {source_key} {day}")
        source_stats[source_key] = stats

    records.sort(key=lambda value: (value[0], value[1]))
    expected_included = sum(row["included_trip_count"] for row in source_stats.values())
    if len(records) != expected_included:
        raise ValueError(f"Chicago expanded trip count failed for {day}")
    identities = {row[1] for row in records}
    if len(identities) != len(records):
        raise ValueError(f"duplicate Chicago expanded trip IDs for {day}")
    destination = output_root / f"trips_{day_text}.xml.gz"
    with destination.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                text.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
                for depart, identity, source_edge, target_edge in records:
                    text.write(
                        "    <trip id="
                        + quoteattr(identity)
                        + f' depart="{depart:.6f}" from='
                        + quoteattr(source_edge)
                        + " to="
                        + quoteattr(target_edge)
                        + ' departLane="best" departSpeed="max"/>\n'
                    )
                text.write("</routes>\n")
    return {
        "date": day_text,
        "day_index": day_index,
        "source_stats": source_stats,
        "published_trip_count": sum(
            row["published_trip_count"] for row in source_stats.values()
        ),
        "included_trip_count": expected_included,
        "excluded_trip_count": sum(
            row["published_trip_count"] - row["included_trip_count"]
            for row in source_stats.values()
        ),
        "output_file": destination.name,
        "output_size_bytes": destination.stat().st_size,
        "output_sha256": _file_sha256(destination),
        "minimum_depart_sec": records[0][0] if records else None,
        "maximum_depart_sec": records[-1][0] if records else None,
        "unique_endpoint_key_count": len(endpoint_cache),
        "unique_endpoint_pair_count": len(pair_cache),
    }


def prepare_demand(
    *,
    acquisition_root: Path,
    network_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    if workers != DEFAULT_WORKERS:
        raise ValueError(f"frozen Chicago demand build requires {DEFAULT_WORKERS} workers")
    existing = output_root / "demand_manifest.json"
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("existing Chicago demand uses a different protocol")
        return payload
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    rejected = output_root.parent / f"{output_root.name}.rejected"
    if staging.exists() or rejected.exists():
        raise FileExistsError(staging if staging.exists() else rejected)
    staging.mkdir(parents=True)
    try:
        acquisition, network, trip_rows = _input_manifests(
            acquisition_root, network_root
        )
        futures = {}
        daily: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for day_index, day in enumerate(DATES):
                source_rows = {
                    source.key: trip_rows[(day.isoformat(), source.key)]
                    for source in TRIP_SOURCES
                }
                future = executor.submit(
                    _day_worker,
                    day_text=day.isoformat(),
                    day_index=day_index,
                    acquisition_root_text=str(acquisition_root.resolve()),
                    boundary_path_text=str(
                        (acquisition_root / "community_areas.geojson").resolve()
                    ),
                    anchor_path_text=str((network_root / "anchors.json").resolve()),
                    output_root_text=str(staging.resolve()),
                    input_rows=source_rows,
                )
                futures[future] = day
            for future in as_completed(futures):
                row = future.result()
                daily.append(row)
                print(
                    f"prepared {row['date']}: included={row['included_trip_count']} "
                    f"excluded={row['excluded_trip_count']}",
                    flush=True,
                )
        daily.sort(key=lambda row: str(row["date"]))
        if [row["date"] for row in daily] != [day.isoformat() for day in DATES]:
            raise ValueError("Chicago demand day inventory is incomplete")
        published_total = sum(int(row["published_trip_count"]) for row in daily)
        expected_published = sum(
            sum(EXPECTED_TRIP_ROWS[source.key].values()) for source in TRIP_SOURCES
        )
        if published_total != expected_published:
            raise ValueError("Chicago weekly published demand conservation failed")
        included_total = sum(int(row["included_trip_count"]) for row in daily)
        excluded_total = sum(int(row["excluded_trip_count"]) for row in daily)
        if included_total + excluded_total != published_total:
            raise ValueError("Chicago weekly endpoint classification failed")
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workers": workers,
            "date_interval": [
                DATES[0].isoformat(),
                (DATES[-1] + timedelta(days=1)).isoformat(),
            ],
            "acquisition_root": str(acquisition_root.resolve()),
            "acquisition_protocol": acquisition["protocol"],
            "acquisition_manifest_sha256": _file_sha256(
                acquisition_root / "acquisition_manifest.json"
            ),
            "network_root": str(network_root.resolve()),
            "network_protocol": network["protocol"],
            "network_sha256": EXPECTED_NETWORK_SHA256,
            "anchor_sha256": EXPECTED_ANCHOR_SHA256,
            "published_trip_count": published_total,
            "included_trip_count": included_total,
            "public_location_suppressed_or_outside_count": excluded_total,
            "included_fraction": included_total / published_total,
            "daily": daily,
            "status": "PASS",
        }
        _write_json(staging / "demand_manifest.json", payload)
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
        _write_json(staging / "demand_failure.json", failure)
        os.replace(staging, rejected)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--network-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    payload = prepare_demand(
        acquisition_root=args.acquisition_root,
        network_root=args.network_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol": payload["protocol"],
                "published_trip_count": payload["published_trip_count"],
                "included_trip_count": payload["included_trip_count"],
                "public_location_suppressed_or_outside_count": payload[
                    "public_location_suppressed_or_outside_count"
                ],
                "included_fraction": payload["included_fraction"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
