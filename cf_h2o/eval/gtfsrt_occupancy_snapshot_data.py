"""Build AVL/occupancy snapshot samples from public GTFS-Realtime feeds.

The goal is narrow: collect synchronized vehicle-position snapshots that also
carry a vehicle occupancy signal. NYC MTA Bus and SEPTA Bus use the gtfsrt.io
historical Parquet archive; MBTA is collected live because a public historical
bulk archive is not available in the current gtfsrt.io inventory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from google.transit import gtfs_realtime_pb2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.cross_city_performance_validation import _repo_root, _resolve_path


INVENTORY_URL = "https://storage.googleapis.com/parquet.gtfsrt.io/inventory.json"
ARCHIVE_PARQUET_TEMPLATE = (
    "https://storage.googleapis.com/parquet.gtfsrt.io/"
    "vehicle_positions/date={date}/base64url={base64url}/data.parquet"
)
ARCHIVE_COLUMNS = [
    "feed_url",
    "feed_timestamp",
    "fetch_timestamp",
    "entity_id",
    "trip_id",
    "route_id",
    "direction_id",
    "start_time",
    "start_date",
    "schedule_relationship",
    "vehicle_id",
    "vehicle_label",
    "latitude",
    "longitude",
    "bearing",
    "speed",
    "current_stop_sequence",
    "stop_id",
    "current_status",
    "timestamp",
    "congestion_level",
    "occupancy_status",
    "occupancy_percentage",
]
OCCUPANCY_STATUS_NAMES = {
    0: "EMPTY",
    1: "MANY_SEATS_AVAILABLE",
    2: "FEW_SEATS_AVAILABLE",
    3: "STANDING_ROOM_ONLY",
    4: "CRUSHED_STANDING_ROOM_ONLY",
    5: "FULL",
    6: "NOT_ACCEPTING_PASSENGERS",
    7: "NO_DATA_AVAILABLE",
}


@dataclass(frozen=True)
class FeedSpec:
    key: str
    label: str
    mode: str
    archive_agency_id: str | None = None
    archive_system_name: str | None = None
    live_url: str | None = None


FEEDS = {
    "nyc_mta_bus": FeedSpec(
        key="nyc_mta_bus",
        label="NYC MTA Bus",
        mode="archive",
        archive_agency_id="mta",
        archive_system_name="NYC Bus",
        live_url="https://gtfsrt.prod.obanyc.com/vehiclePositions",
    ),
    "septa_bus": FeedSpec(
        key="septa_bus",
        label="SEPTA Bus",
        mode="archive",
        archive_agency_id="septa",
        archive_system_name="Bus",
        live_url="https://www3.septa.org/gtfsrt/septa-pa-us/Vehicle/rtVehiclePosition.pb",
    ),
    "mbta_live": FeedSpec(
        key="mbta_live",
        label="MBTA live",
        mode="live",
        live_url="https://cdn.mbta.com/realtime/VehiclePositions.pb",
    ),
}


def _url_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _url_json(url: str, timeout: int = 60) -> Any:
    return json.loads(_url_bytes(url, timeout=timeout).decode("utf-8"))


def _status_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    try:
        index = int(value)
    except (TypeError, ValueError):
        return str(value)
    try:
        return gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(index)
    except Exception:
        return OCCUPANCY_STATUS_NAMES.get(index, str(index))


def _iso_from_epoch(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return pd.to_datetime(int(value), unit="s", utc=True).isoformat()
    except Exception:
        return None


def _inventory_item(spec: FeedSpec, inventory: list[dict[str, Any]], archive_date: str | None) -> dict[str, Any]:
    candidates = [
        item
        for item in inventory
        if item.get("feed_type") == "vehicle_positions"
        and item.get("agency_id") == spec.archive_agency_id
        and (spec.archive_system_name is None or item.get("system_name") == spec.archive_system_name)
    ]
    if not candidates:
        raise RuntimeError(f"{spec.key}: no vehicle_positions feed found in gtfsrt.io inventory")
    item = sorted(candidates, key=lambda row: str(row.get("date_max") or ""))[-1]
    if archive_date:
        date_min = str(item.get("date_min"))
        date_max = str(item.get("date_max"))
        if not (date_min <= archive_date <= date_max):
            raise RuntimeError(f"{spec.key}: archive date {archive_date} outside inventory range {date_min}..{date_max}")
        item = dict(item)
        item["date_selected"] = archive_date
    else:
        item = dict(item)
        item["date_selected"] = item["date_max"]
    return item


def _row_group_indices(num_groups: int, max_snapshots: int) -> list[int]:
    if num_groups <= 0:
        return []
    if max_snapshots <= 0 or max_snapshots >= num_groups:
        return list(range(num_groups))
    return sorted(set(np.linspace(0, num_groups - 1, num=max_snapshots, dtype=np.int64).tolist()))


def _normalize_archive_batch(df: pd.DataFrame, spec: FeedSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    fetch_ts = pd.to_datetime(df["fetch_timestamp"], utc=True)
    out["source_key"] = spec.key
    out["source_label"] = spec.label
    out["source_kind"] = "gtfsrt_io_archive"
    out["snapshot_ts"] = fetch_ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out["feed_timestamp"] = df["feed_timestamp"].map(_iso_from_epoch)
    out["vehicle_timestamp"] = df["timestamp"].map(_iso_from_epoch)
    for name in [
        "entity_id",
        "trip_id",
        "route_id",
        "direction_id",
        "start_time",
        "start_date",
        "schedule_relationship",
        "vehicle_id",
        "vehicle_label",
        "latitude",
        "longitude",
        "bearing",
        "speed",
        "current_stop_sequence",
        "stop_id",
        "current_status",
        "congestion_level",
        "occupancy_percentage",
    ]:
        out[name] = df[name] if name in df.columns else None
    out["occupancy_status"] = df["occupancy_status"].map(_status_name)
    return out


def _collect_archive_snapshots(spec: FeedSpec, inventory: list[dict[str, Any]], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    item = _inventory_item(spec, inventory, args.archive_date)
    date = str(item["date_selected"])
    parquet_url = ARCHIVE_PARQUET_TEMPLATE.format(date=date, base64url=item["base64url"])
    fs = fsspec.filesystem("https")
    frames: list[pd.DataFrame] = []
    selected_groups: list[int] = []
    with fs.open(parquet_url, "rb") as fh:
        parquet = pq.ParquetFile(fh)
        for group_idx in _row_group_indices(parquet.metadata.num_row_groups, int(args.max_snapshots)):
            table = parquet.read_row_group(group_idx, columns=ARCHIVE_COLUMNS)
            df = table.to_pandas()
            if args.require_occupancy and int(df["occupancy_status"].notna().sum() + df["occupancy_percentage"].notna().sum()) == 0:
                continue
            frames.append(_normalize_archive_batch(df, spec))
            selected_groups.append(int(group_idx))
            if 0 < int(args.max_snapshots) <= len(frames):
                break
    if frames:
        rows = pd.concat(frames, ignore_index=True)
    else:
        rows = pd.DataFrame(columns=_snapshot_columns())
    meta = {
        "source_key": spec.key,
        "source_label": spec.label,
        "source_kind": "gtfsrt_io_archive",
        "archive_date": date,
        "inventory": item,
        "parquet_url": parquet_url,
        "selected_row_groups": selected_groups,
    }
    return rows, meta


def _snapshot_columns() -> list[str]:
    return [
        "source_key",
        "source_label",
        "source_kind",
        "snapshot_ts",
        "feed_timestamp",
        "vehicle_timestamp",
        "entity_id",
        "trip_id",
        "route_id",
        "direction_id",
        "start_time",
        "start_date",
        "schedule_relationship",
        "vehicle_id",
        "vehicle_label",
        "latitude",
        "longitude",
        "bearing",
        "speed",
        "current_stop_sequence",
        "stop_id",
        "current_status",
        "congestion_level",
        "occupancy_status",
        "occupancy_percentage",
    ]


def _vehicle_rows_from_feed(raw: bytes, spec: FeedSpec, fetched_at: pd.Timestamp) -> pd.DataFrame:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)
    feed_ts = _iso_from_epoch(feed.header.timestamp if feed.header.HasField("timestamp") else None)
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        vehicle = entity.vehicle
        position = vehicle.position if vehicle.HasField("position") else None
        trip = vehicle.trip if vehicle.HasField("trip") else None
        descriptor = vehicle.vehicle if vehicle.HasField("vehicle") else None
        rows.append(
            {
                "source_key": spec.key,
                "source_label": spec.label,
                "source_kind": "live_gtfsrt",
                "snapshot_ts": fetched_at.isoformat(),
                "feed_timestamp": feed_ts,
                "vehicle_timestamp": _iso_from_epoch(vehicle.timestamp if vehicle.HasField("timestamp") else None),
                "entity_id": entity.id,
                "trip_id": trip.trip_id if trip is not None and trip.trip_id else None,
                "route_id": trip.route_id if trip is not None and trip.route_id else None,
                "direction_id": trip.direction_id if trip is not None and trip.HasField("direction_id") else None,
                "start_time": trip.start_time if trip is not None and trip.start_time else None,
                "start_date": trip.start_date if trip is not None and trip.start_date else None,
                "schedule_relationship": trip.schedule_relationship if trip is not None and trip.HasField("schedule_relationship") else None,
                "vehicle_id": descriptor.id if descriptor is not None and descriptor.id else None,
                "vehicle_label": descriptor.label if descriptor is not None and descriptor.label else None,
                "latitude": position.latitude if position is not None and position.HasField("latitude") else None,
                "longitude": position.longitude if position is not None and position.HasField("longitude") else None,
                "bearing": position.bearing if position is not None and position.HasField("bearing") else None,
                "speed": position.speed if position is not None and position.HasField("speed") else None,
                "current_stop_sequence": vehicle.current_stop_sequence if vehicle.HasField("current_stop_sequence") else None,
                "stop_id": vehicle.stop_id if vehicle.stop_id else None,
                "current_status": vehicle.current_status if vehicle.HasField("current_status") else None,
                "congestion_level": vehicle.congestion_level if vehicle.HasField("congestion_level") else None,
                "occupancy_status": _status_name(vehicle.occupancy_status if vehicle.HasField("occupancy_status") else None),
                "occupancy_percentage": vehicle.occupancy_percentage if vehicle.HasField("occupancy_percentage") else None,
            }
        )
    return pd.DataFrame(rows, columns=_snapshot_columns())


def _collect_live_snapshots(spec: FeedSpec, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not spec.live_url:
        raise RuntimeError(f"{spec.key}: missing live_url")
    frames = []
    polls = max(1, int(args.live_polls))
    for poll_idx in range(polls):
        fetched_at = pd.Timestamp.now(tz="UTC")
        raw = _url_bytes(spec.live_url, timeout=int(args.http_timeout))
        frame = _vehicle_rows_from_feed(raw, spec, fetched_at)
        if args.require_occupancy and int(frame["occupancy_status"].notna().sum() + frame["occupancy_percentage"].notna().sum()) == 0:
            frame = frame.iloc[0:0]
        frames.append(frame)
        if poll_idx + 1 < polls:
            time.sleep(max(0.0, float(args.live_poll_interval_sec)))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_snapshot_columns())
    meta = {
        "source_key": spec.key,
        "source_label": spec.label,
        "source_kind": "live_gtfsrt",
        "live_url": spec.live_url,
        "polls": polls,
    }
    return rows, meta


def _summary_for(rows: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    with_position = rows["latitude"].notna() & rows["longitude"].notna() if len(rows) else pd.Series(dtype=bool)
    with_status = rows["occupancy_status"].notna() if len(rows) else pd.Series(dtype=bool)
    with_pct = rows["occupancy_percentage"].notna() if len(rows) else pd.Series(dtype=bool)
    status_counts = rows["occupancy_status"].dropna().value_counts().sort_index().to_dict() if len(rows) else {}
    return {
        **meta,
        "rows": int(len(rows)),
        "snapshots": int(rows["snapshot_ts"].nunique()) if len(rows) else 0,
        "vehicles": int(rows["vehicle_id"].nunique()) if len(rows) and "vehicle_id" in rows else 0,
        "routes": int(rows["route_id"].nunique()) if len(rows) and "route_id" in rows else 0,
        "rows_with_position": int(with_position.sum()) if len(rows) else 0,
        "rows_with_occupancy_status": int(with_status.sum()) if len(rows) else 0,
        "rows_with_occupancy_percentage": int(with_pct.sum()) if len(rows) else 0,
        "occupancy_status_rate": float(with_status.mean()) if len(rows) else 0.0,
        "occupancy_percentage_rate": float(with_pct.mean()) if len(rows) else 0.0,
        "occupancy_status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }


def _write_outputs(rows: pd.DataFrame, summary: dict[str, Any], out_dir: Path, source_key: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{source_key}_vehicle_snapshots.csv"
    parquet_path = out_dir / f"{source_key}_vehicle_snapshots.parquet"
    summary_path = out_dir / f"{source_key}_summary.json"
    rows.to_csv(csv_path, index=False)
    rows.to_parquet(parquet_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "parquet": str(parquet_path), "summary": str(summary_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    out_dir = _resolve_path(root, args.out_dir)
    selected = [item.strip() for item in args.feeds.split(",") if item.strip()]
    unknown = [item for item in selected if item not in FEEDS]
    if unknown:
        raise SystemExit(f"unknown feeds: {unknown}; available={sorted(FEEDS)}")
    inventory = _url_json(args.inventory_url, timeout=int(args.http_timeout))
    results = []
    for key in selected:
        spec = FEEDS[key]
        if spec.mode == "archive":
            rows, meta = _collect_archive_snapshots(spec, inventory, args)
        elif spec.mode == "live":
            rows, meta = _collect_live_snapshots(spec, args)
        else:
            raise RuntimeError(f"{key}: unknown mode {spec.mode}")
        summary = _summary_for(rows, meta)
        summary["outputs"] = _write_outputs(rows, summary, out_dir, key)
        results.append(summary)
        print(json.dumps({key: {k: summary[k] for k in ("rows", "snapshots", "rows_with_occupancy_status", "rows_with_occupancy_percentage", "occupancy_status_rate", "occupancy_percentage_rate")}}, indent=2), flush=True)

    result = {
        "ok": True,
        "validation_level": "gtfsrt_avl_occupancy_snapshot_probe",
        "out_dir": str(out_dir),
        "feeds": results,
        "notes": (
            "These are synchronized AVL/vehicle-occupancy snapshots. Occupancy is GTFS-RT crowding status/percentage where available, "
            "not exact APC boarding/alighting counts."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feeds", default="nyc_mta_bus,septa_bus,mbta_live")
    parser.add_argument("--out-dir", type=Path, default=Path("H2Oplus/downloads/open_transit/gtfsrt_occupancy_snapshots"))
    parser.add_argument("--inventory-url", default=INVENTORY_URL)
    parser.add_argument("--archive-date", default="")
    parser.add_argument("--max-snapshots", type=int, default=12)
    parser.add_argument("--require-occupancy", action="store_true", default=True)
    parser.add_argument("--allow-empty-occupancy", dest="require_occupancy", action="store_false")
    parser.add_argument("--live-polls", type=int, default=1)
    parser.add_argument("--live-poll-interval-sec", type=float, default=30.0)
    parser.add_argument("--http-timeout", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
