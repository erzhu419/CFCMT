"""
Download public bus datasets for cross-city H2O+/CFCMT experiments.

No API keys are required for the current presets:
    - austin_capmetro
    - halifax_transit
    - mbta

Outputs go to H2Oplus/downloads/open_transit by default. That tree is ignored
by H2Oplus/.gitignore.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "austin_capmetro": {
        "label": "Austin / CapMetro",
        "gtfs": {
            "url": "https://data.texas.gov/api/views/r4v4-vz24/files/8875c745-8922-486a-b44f-dcdbf0d2aea2?download=true&filename=capmetro.zip",
            "filename": "capmetro_gtfs.zip",
            "source": "https://data.texas.gov/dataset/CapMetro-GTFS/r4v4-vz24",
        },
        "notes": "CapMetro public source gives GTFS/static schedule. Public no-key raw APC stop-event data is handled by cf_h2o.eval.austin_apc_real_validation.",
    },
    "halifax_transit": {
        "label": "Halifax Transit",
        "gtfs": {
            "url": "https://gtfs.halifax.ca/static/google_transit.zip",
            "filename": "halifax_google_transit.zip",
            "source": "https://www.halifax.ca/transportation/halifax-transit/transit-technology/general-transit-feed-gtfs",
        },
        "arcgis_table": {
            "url": "https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/Transit_Automated_Passenger_Counts/FeatureServer/0",
            "filename": "halifax_transit_automated_passenger_counts.csv",
            "source": "https://data-hrm.hub.arcgis.com/search?q=Transit%20Automated%20Passenger%20Counts",
        },
        "notes": "Halifax APC is route-level half-hour boardings, not stop-level OD.",
    },
    "mbta": {
        "label": "MBTA",
        "gtfs": {
            "url": "https://cdn.mbta.com/MBTA_GTFS.zip",
            "filename": "mbta_gtfs.zip",
            "source": "https://github.com/mbta/gtfs-documentation",
        },
        "ridership_zip": {
            "url": "https://www.arcgis.com/sharing/rest/content/items/8daf4a33925a4df59183f860826d29ee/data",
            "filename": "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop.zip",
            "source": "https://mbta-massdot.opendata.arcgis.com/search?q=Bus%20Ridership",
        },
        "notes": "MBTA bus ridership has average boardings/alightings/load by season, route, direction, trip start, stop, and day type.",
    },
}


def repo_default_out_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] / "downloads" / "open_transit"


def discover_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_range = resp.headers.get("content-range", "")
            match = re.search(r"/(\d+)$", content_range)
            if match:
                return int(match.group(1))
            length = resp.headers.get("content-length")
            return int(length) if length else None
    except Exception:
        return None


def download_file(url: str, out_path: pathlib.Path, force: bool = False) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected = discover_size(url)
    if out_path.exists() and not force:
        actual = out_path.stat().st_size
        if expected is None or expected == actual:
            print(f"skip existing: {out_path} ({actual} bytes)")
            return {"path": str(out_path), "bytes": actual, "skipped": True, "source_url": url}

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            if expected:
                print(f"downloading {out_path.name}: {total / expected:.1%}", end="\r")
        print()
    tmp.replace(out_path)
    actual = out_path.stat().st_size
    if expected is not None and actual != expected:
        raise RuntimeError(f"Downloaded size mismatch for {out_path}: {actual} != {expected}")
    return {"path": str(out_path), "bytes": actual, "skipped": False, "source_url": url}


def extract_zip(zip_path: pathlib.Path, out_dir: pathlib.Path, force: bool = False) -> list[str]:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return [str(path) for path in sorted(out_dir.iterdir())]
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
        return [str(out_dir / name) for name in zf.namelist()]


def urlopen_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {})
    full_url = f"{url}?{query}" if query else url
    req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_arcgis_table(service_url: str, out_path: pathlib.Path, page_size: int = 2000, force: bool = False) -> dict[str, Any]:
    if out_path.exists() and not force:
        print(f"skip existing: {out_path} ({out_path.stat().st_size} bytes)")
        return {"path": str(out_path), "skipped": True}

    meta = urlopen_json(f"{service_url}", {"f": "json"})
    fields = [field["name"] for field in meta.get("fields", [])]
    if not fields:
        raise RuntimeError(f"No fields found for ArcGIS table {service_url}")

    count_data = urlopen_json(
        f"{service_url}/query",
        {"f": "json", "where": "1=1", "returnCountOnly": "true"},
    )
    total = int(count_data.get("count", 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        offset = 0
        while True:
            data = urlopen_json(
                f"{service_url}/query",
                {
                    "f": "json",
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "false",
                    "orderByFields": meta.get("objectIdField", "OBJECTID"),
                    "resultOffset": offset,
                    "resultRecordCount": page_size,
                },
            )
            features = data.get("features", [])
            if not features:
                break
            for feature in features:
                attrs = feature.get("attributes", {})
                writer.writerow({field: attrs.get(field) for field in fields})
            rows_written += len(features)
            print(f"{out_path.name}: {rows_written}/{total or '?'} rows")
            if len(features) < page_size:
                break
            offset += len(features)
            time.sleep(0.05)
    return {"path": str(out_path), "rows": rows_written, "skipped": False, "source_url": service_url}


def fetch_city(city: str, out_root: pathlib.Path, extract: bool, skip_ridership: bool, force: bool) -> dict[str, Any]:
    preset = PRESETS[city]
    city_dir = out_root / city
    manifest: dict[str, Any] = {
        "city": city,
        "label": preset["label"],
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": preset.get("notes", ""),
        "items": [],
    }

    gtfs = preset["gtfs"]
    gtfs_zip = city_dir / "raw" / gtfs["filename"]
    gtfs_info = download_file(gtfs["url"], gtfs_zip, force=force)
    gtfs_info["source_page"] = gtfs["source"]
    if extract:
        gtfs_info["extracted_files"] = extract_zip(gtfs_zip, city_dir / "gtfs", force=force)
    manifest["items"].append({"kind": "gtfs", **gtfs_info})

    if not skip_ridership and "arcgis_table" in preset:
        table = preset["arcgis_table"]
        table_info = fetch_arcgis_table(table["url"], city_dir / "ridership" / table["filename"], force=force)
        table_info["source_page"] = table["source"]
        manifest["items"].append({"kind": "ridership_arcgis_table", **table_info})

    if not skip_ridership and "ridership_zip" in preset:
        ridership = preset["ridership_zip"]
        ridership_zip = city_dir / "ridership" / ridership["filename"]
        ridership_info = download_file(ridership["url"], ridership_zip, force=force)
        ridership_info["source_page"] = ridership["source"]
        if extract:
            ridership_info["extracted_files"] = extract_zip(
                ridership_zip,
                city_dir / "ridership" / pathlib.Path(ridership["filename"]).stem,
                force=force,
            )
        manifest["items"].append({"kind": "ridership_zip", **ridership_info})

    (city_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {city_dir / 'manifest.json'}")
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=["all", *PRESETS.keys()], default="all")
    parser.add_argument("--out", type=pathlib.Path, default=repo_default_out_dir())
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zip files")
    parser.add_argument("--skip-ridership", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cities = list(PRESETS) if args.city == "all" else [args.city]
    all_manifest = {
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cities": [],
    }
    for city in cities:
        all_manifest["cities"].append(fetch_city(city, args.out, args.extract, args.skip_ridership, args.force))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
