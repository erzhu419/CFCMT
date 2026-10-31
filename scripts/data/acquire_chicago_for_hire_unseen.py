#!/usr/bin/env python3
"""Acquire the frozen Chicago v107 aggregates directly on shared storage."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROTOCOL = "cfcmt-chicago-full-week-for-hire-acquisition-v1"
USER_AGENT = "CFCMT-research-acquisition/1.0"
START_DATE = date(2023, 8, 21)
DATES = tuple(START_DATE + timedelta(days=index) for index in range(7))
PAGE_SIZE = 50_000
DEFAULT_WORKERS = 8


@dataclass(frozen=True)
class SocrataSource:
    key: str
    dataset_id: str
    name: str
    rows_updated_at: int


TRIP_SOURCES = (
    SocrataSource(
        key="tnp",
        dataset_id="n26f-ihde",
        name="Transportation Network Providers - Trips (2023-2024)",
        rows_updated_at=1_740_694_101,
    ),
    SocrataSource(
        key="taxi",
        dataset_id="e55j-2ewb",
        name="Taxi Trips - 2023",
        rows_updated_at=1_707_338_412,
    ),
)
SPEED_SOURCE = SocrataSource(
    key="speed",
    dataset_id="sxs8-h27x",
    name=(
        "Chicago Traffic Tracker - Historical Congestion Estimates by Segment - "
        "2018-2023"
    ),
    rows_updated_at=1_747_348_090,
)
BOUNDARY_SOURCE = SocrataSource(
    key="community_areas",
    dataset_id="igwz-8jzy",
    name="Boundaries - Community Areas",
    rows_updated_at=1_745_363_197,
)
SOURCES = (*TRIP_SOURCES, SPEED_SOURCE, BOUNDARY_SOURCE)

EXPECTED_TRIP_ROWS = {
    "tnp": {
        "2023-08-21": 181_800,
        "2023-08-22": 189_822,
        "2023-08-23": 215_703,
        "2023-08-24": 240_691,
        "2023-08-25": 260_008,
        "2023-08-26": 290_310,
        "2023-08-27": 221_223,
    },
    "taxi": {
        "2023-08-21": 17_217,
        "2023-08-22": 17_870,
        "2023-08-23": 19_423,
        "2023-08-24": 20_508,
        "2023-08-25": 18_151,
        "2023-08-26": 13_107,
        "2023-08-27": 12_480,
    },
}
EXPECTED_SPEED_ROWS = {
    "2023-08-21": (135_063, 76_983, 1_047),
    "2023-08-22": (148_674, 87_830, 1_047),
    "2023-08-23": (150_768, 88_168, 1_047),
    "2023-08-24": (129_828, 76_506, 1_047),
    "2023-08-25": (149_721, 87_121, 1_047),
    "2023-08-26": (148_674, 78_350, 1_047),
    "2023-08-27": (141_345, 70_556, 1_047),
}

OSM_URL = "https://download.bbbike.org/osm/bbbike/Chicago/Chicago.osm.gz"
OSM_NAME = "Chicago.osm.gz"
OSM_SIZE_BYTES = 231_147_462
OSM_MD5 = "1ae6e180e1c996c5def4cbfb03684808"
OSM_LAST_MODIFIED = "Sun, 30 Aug 2026 03:39:55 GMT"

TRIP_GROUP_FIELDS = (
    "trip_start_timestamp",
    "pickup_community_area",
    "dropoff_community_area",
    "pickup_centroid_latitude",
    "pickup_centroid_longitude",
    "dropoff_centroid_latitude",
    "dropoff_centroid_longitude",
)
TRIP_AGGREGATES = (
    "count(*) as trip_count",
    "count(trip_seconds) as trip_seconds_count",
    "sum(trip_seconds) as trip_seconds_sum",
    "count(trip_miles) as trip_miles_count",
    "sum(trip_miles) as trip_miles_sum",
)
TRIP_COLUMNS = (*TRIP_GROUP_FIELDS, *(item.split(" as ")[-1] for item in TRIP_AGGREGATES))
SPEED_COLUMNS = (
    "time",
    "segment_id",
    "speed",
    "street",
    "direction",
    "from_street",
    "to_street",
    "length",
    "street_heading",
    "comments",
    "bus_count",
    "message_count",
    "record_id",
    "start_latitude",
    "start_longitude",
    "end_latitude",
    "end_longitude",
)


def _metadata_url(dataset_id: str) -> str:
    return f"https://data.cityofchicago.org/api/views/{dataset_id}"


def _resource_url(dataset_id: str, extension: str, params: Mapping[str, str]) -> str:
    return (
        f"https://data.cityofchicago.org/resource/{dataset_id}.{extension}?"
        f"{urlencode(params)}"
    )


def _day_where(field: str, day: date) -> str:
    following = day + timedelta(days=1)
    return (
        f"{field} >= '{day.isoformat()}T00:00:00.000' AND "
        f"{field} < '{following.isoformat()}T00:00:00.000'"
    )


def trip_page_url(source: SocrataSource, day: date, offset: int) -> str:
    fields = ",".join(TRIP_GROUP_FIELDS)
    return _resource_url(
        source.dataset_id,
        "csv",
        {
            "$select": ",".join((*TRIP_GROUP_FIELDS, *TRIP_AGGREGATES)),
            "$where": _day_where("trip_start_timestamp", day),
            "$group": fields,
            "$order": fields,
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        },
    )


def speed_page_url(day: date, offset: int) -> str:
    return _resource_url(
        SPEED_SOURCE.dataset_id,
        "csv",
        {
            "$select": ",".join(SPEED_COLUMNS),
            "$where": _day_where("time", day),
            "$order": "time,segment_id,record_id",
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        },
    )


def _request(url: str, *, accept: str = "*/*") -> Request:
    return Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})


def _fetch_json(url: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(_request(url, accept="application/json"), timeout=180) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < 5:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch JSON after retries: {url}: {last_error}")


def _source_metadata() -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        payload = _fetch_json(_metadata_url(source.dataset_id))
        identity = {
            "id": payload.get("id"),
            "name": payload.get("name"),
            "rowsUpdatedAt": payload.get("rowsUpdatedAt"),
        }
        expected = {
            "id": source.dataset_id,
            "name": source.name,
            "rowsUpdatedAt": source.rows_updated_at,
        }
        if identity != expected:
            raise ValueError(
                f"Chicago Socrata source changed for {source.key}: "
                f"{identity} != {expected}"
            )
        observed[source.key] = {
            **identity,
            "metadata_url": _metadata_url(source.dataset_id),
        }
    return observed


def _download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_last_modified: str | None = None,
) -> dict[str, Any]:
    temporary = destination.with_name(f".{destination.name}.part")
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    last_error: Exception | None = None
    for attempt in range(5):
        size = 0
        sha256 = hashlib.sha256()
        md5 = hashlib.md5()
        try:
            with urlopen(_request(url), timeout=900) as response, temporary.open("wb") as handle:
                last_modified = response.headers.get("Last-Modified")
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    handle.write(block)
                    size += len(block)
                    sha256.update(block)
                    md5.update(block)
            if expected_size is not None and size != expected_size:
                raise ValueError(f"size changed for {destination.name}: {size}")
            if expected_md5 is not None and md5.hexdigest() != expected_md5:
                raise ValueError(f"publisher MD5 changed for {destination.name}")
            if (
                expected_last_modified is not None
                and last_modified != expected_last_modified
            ):
                raise ValueError(
                    f"Last-Modified changed for {destination.name}: {last_modified}"
                )
            os.replace(temporary, destination)
            return {
                "url": url,
                "size_bytes": size,
                "sha256": sha256.hexdigest(),
                "md5": md5.hexdigest(),
                "last_modified": last_modified,
            }
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < 5:
                time.sleep(2 ** attempt)
        except ValueError:
            temporary.unlink(missing_ok=True)
            raise
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _fetch_csv_page(url: str, page_path: Path) -> None:
    _download(url, page_path)


def _append_page(
    *,
    page_path: Path,
    writer: csv.DictWriter,
    expected_columns: Sequence[str],
) -> list[dict[str, str]]:
    with page_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(
                f"unexpected columns in {page_path.name}: {reader.fieldnames}"
            )
        rows = list(reader)
    for row in rows:
        writer.writerow(row)
    page_path.unlink()
    return rows


def _acquire_trip_day(source: SocrataSource, day: date, root: Path) -> dict[str, Any]:
    destination = root / "trips" / f"{source.key}_{day.isoformat()}_aggregates.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    expected_rows = EXPECTED_TRIP_ROWS[source.key][day.isoformat()]
    aggregate_rows = 0
    represented_rows = 0
    duplicate_groups = 0
    groups_seen: set[tuple[str, ...]] = set()
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TRIP_COLUMNS)
        writer.writeheader()
        offset = 0
        while True:
            page_path = destination.with_name(f".{destination.name}.page-{offset}")
            _fetch_csv_page(trip_page_url(source, day, offset), page_path)
            rows = _append_page(
                page_path=page_path,
                writer=writer,
                expected_columns=TRIP_COLUMNS,
            )
            for row in rows:
                key = tuple(row[field] for field in TRIP_GROUP_FIELDS)
                duplicate_groups += int(key in groups_seen)
                groups_seen.add(key)
                represented_rows += int(row["trip_count"])
            aggregate_rows += len(rows)
            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    if not aggregate_rows:
        raise ValueError(f"no aggregate rows returned for {source.key} {day}")
    if duplicate_groups:
        raise ValueError(f"duplicate aggregate groups for {source.key} {day}")
    if represented_rows != expected_rows:
        raise ValueError(
            f"trip conservation failed for {source.key} {day}: "
            f"{represented_rows} != {expected_rows}"
        )
    os.replace(temporary, destination)
    return {
        "kind": "trip_aggregate",
        "source": source.key,
        "date": day.isoformat(),
        "path": str(destination.relative_to(root)),
        "aggregate_row_count": aggregate_rows,
        "represented_trip_count": represented_rows,
        "size_bytes": destination.stat().st_size,
        "sha256": _file_digest(destination),
    }


def _acquire_speed_day(day: date, root: Path) -> dict[str, Any]:
    destination = root / "speed" / f"speed_{day.isoformat()}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    expected_rows, expected_positive, expected_segments = EXPECTED_SPEED_ROWS[
        day.isoformat()
    ]
    row_count = 0
    positive_count = 0
    segment_ids: set[str] = set()
    record_ids: set[str] = set()
    duplicate_records = 0
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SPEED_COLUMNS)
        writer.writeheader()
        offset = 0
        while True:
            page_path = destination.with_name(f".{destination.name}.page-{offset}")
            _fetch_csv_page(speed_page_url(day, offset), page_path)
            rows = _append_page(
                page_path=page_path,
                writer=writer,
                expected_columns=SPEED_COLUMNS,
            )
            for row in rows:
                record_id = row["record_id"]
                duplicate_records += int(record_id in record_ids)
                record_ids.add(record_id)
                segment_ids.add(row["segment_id"])
                positive_count += int(float(row["speed"]) > 0.0)
            row_count += len(rows)
            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    observed = (row_count, positive_count, len(segment_ids))
    expected = (expected_rows, expected_positive, expected_segments)
    if observed != expected:
        raise ValueError(f"speed coverage changed for {day}: {observed} != {expected}")
    if duplicate_records:
        raise ValueError(f"duplicate speed record IDs for {day}: {duplicate_records}")
    os.replace(temporary, destination)
    return {
        "kind": "speed",
        "source": SPEED_SOURCE.key,
        "date": day.isoformat(),
        "path": str(destination.relative_to(root)),
        "row_count": row_count,
        "positive_speed_row_count": positive_count,
        "segment_count": len(segment_ids),
        "size_bytes": destination.stat().st_size,
        "sha256": _file_digest(destination),
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _acquire_boundaries(root: Path) -> dict[str, Any]:
    url = _resource_url(
        BOUNDARY_SOURCE.dataset_id,
        "geojson",
        {"$limit": "100", "$order": "area_numbe"},
    )
    destination = root / "community_areas.geojson"
    observation = _download(url, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    areas = {
        int(feature.get("properties", {}).get("area_numbe"))
        for feature in features
    }
    if len(features) != 77 or areas != set(range(1, 78)):
        raise ValueError(
            f"Chicago community-area inventory changed: {len(features)} features"
        )
    return {
        **observation,
        "path": destination.name,
        "feature_count": len(features),
        "area_numbers": sorted(areas),
    }


def _acquire_osm(root: Path) -> dict[str, Any]:
    return {
        **_download(
            OSM_URL,
            root / OSM_NAME,
            expected_size=OSM_SIZE_BYTES,
            expected_md5=OSM_MD5,
            expected_last_modified=OSM_LAST_MODIFIED,
        ),
        "path": OSM_NAME,
        "publisher_md5": OSM_MD5,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def acquire(output_root: Path, workers: int = DEFAULT_WORKERS) -> dict[str, Any]:
    if workers != DEFAULT_WORKERS:
        raise ValueError(f"frozen acquisition requires {DEFAULT_WORKERS} workers")
    if output_root.exists():
        manifest = output_root / "acquisition_manifest.json"
        if not manifest.is_file():
            raise FileExistsError(output_root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("existing Chicago acquisition has a different protocol")
        return payload
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        metadata_before = _source_metadata()
        boundaries = _acquire_boundaries(staging)
        osm = _acquire_osm(staging)
        jobs: list[tuple[str, SocrataSource | None, date]] = []
        for day in DATES:
            jobs.extend(("trip", source, day) for source in TRIP_SOURCES)
            jobs.append(("speed", None, day))
        observations: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _acquire_trip_day, source, day, staging
                )
                if kind == "trip" and source is not None
                else executor.submit(_acquire_speed_day, day, staging): (kind, source, day)
                for kind, source, day in jobs
            }
            for future in as_completed(futures):
                kind, source, day = futures[future]
                row = future.result()
                observations.append(row)
                label = source.key if source is not None else kind
                print(f"acquired {label} {day}: {row.get('row_count', row.get('represented_trip_count'))}", flush=True)
        metadata_after = _source_metadata()
        if metadata_after != metadata_before:
            raise ValueError("Chicago Socrata source metadata changed during acquisition")
        observations.sort(key=lambda row: (str(row["date"]), str(row["source"])))
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "date_interval": [DATES[0].isoformat(), (DATES[-1] + timedelta(days=1)).isoformat()],
            "coverage": (
                "all published TNP and taxi rows represented in grouped files; "
                "all historical speed rows for seven complete days"
            ),
            "transfer_mode": "public_sources_downloaded_directly_on_shared_cluster_storage",
            "workers": workers,
            "source_metadata": metadata_after,
            "osm": osm,
            "community_areas": boundaries,
            "daily_files": observations,
            "trip_source_totals": {
                source.key: sum(EXPECTED_TRIP_ROWS[source.key].values())
                for source in TRIP_SOURCES
            },
            "speed_row_total": sum(row[0] for row in EXPECTED_SPEED_ROWS.values()),
            "positive_speed_row_total": sum(
                row[1] for row in EXPECTED_SPEED_ROWS.values()
            ),
        }
        _write_json(staging / "acquisition_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    payload = acquire(args.output_root, workers=args.workers)
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": payload["protocol"],
                "workers": payload["workers"],
                "trip_source_totals": payload["trip_source_totals"],
                "speed_row_total": payload["speed_row_total"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
