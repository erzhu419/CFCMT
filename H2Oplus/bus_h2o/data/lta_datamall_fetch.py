"""
Fetch Singapore LTA DataMall bus data for CFCMT/H2O+ calibration.

The API key is read from LTA_DATAMALL_KEY. Do not hard-code it in this file.

Typical use:
    export LTA_DATAMALL_KEY='...'
    python H2Oplus/bus_h2o/data/lta_datamall_fetch.py --date 202604

Outputs are written under H2Oplus/downloads/lta_datamall by default, which is
ignored by the copied H2Oplus tree's .gitignore.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any


BASE_URL = "https://datamall2.mytransport.sg/ltaodataservice"
PAGE_SIZE = 500

PAGED_ENDPOINTS = {
    "BusStops": "/BusStops",
    "BusRoutes": "/BusRoutes",
    "BusServices": "/BusServices",
}

TRAFFIC_ENDPOINTS = {
    "TrafficSpeedBands": "/v4/TrafficSpeedBands",
}

PV_ENDPOINTS = {
    "ODBus": "/PV/ODBus",
    "Bus": "/PV/Bus",
}


def repo_default_out_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] / "downloads" / "lta_datamall"


def previous_month(today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    first = today.replace(day=1)
    prev = first - dt.timedelta(days=1)
    return f"{prev.year:04d}{prev.month:02d}"


def request_json(path: str, key: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {})
    url = f"{BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "AccountKey": key,
            "accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DataMall request failed: {exc.code} {exc.reason}: {detail}") from exc
    return json.loads(body.decode("utf-8"))


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_paged(name: str, path: str, key: str, out_dir: pathlib.Path, page_limit: int | None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    pages = 0
    skip = 0

    while True:
        data = request_json(path, key, {"$skip": skip} if skip else None)
        value = data.get("value", [])
        if not isinstance(value, list):
            raise RuntimeError(f"{name} response field 'value' is not a list")

        pages += 1
        records.extend(value)
        print(f"{name}: page {pages}, skip={skip}, rows={len(value)}, total={len(records)}")

        if len(value) < PAGE_SIZE:
            break
        if page_limit is not None and pages >= page_limit:
            break
        skip += PAGE_SIZE
        time.sleep(0.1)

    payload = {
        "endpoint": name,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "count": len(records),
        "value": records,
    }
    write_json(out_dir / f"{name}.json", payload)
    return {"name": name, "path": str(out_dir / f"{name}.json"), "count": len(records), "pages": pages}


def discover_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
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
    expected_size = discover_size(url)

    if out_path.exists() and not force:
        actual = out_path.stat().st_size
        if expected_size is None or actual == expected_size:
            print(f"skip existing: {out_path} ({actual} bytes)")
            return {"path": str(out_path), "bytes": actual, "skipped": True}

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=300) as resp, tmp_path.open("wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            if expected_size:
                print(f"downloading {out_path.name}: {total / expected_size:.1%}", end="\r")
        print()
    tmp_path.replace(out_path)

    actual = out_path.stat().st_size
    if expected_size is not None and actual != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {out_path}: {actual} != {expected_size}")
    return {"path": str(out_path), "bytes": actual, "skipped": False}


def infer_download_name(kind: str, month: str, url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = pathlib.PurePosixPath(parsed.path).name
    if base.endswith(".zip"):
        return base
    return f"{kind.lower()}_{month}.zip"


def extract_zip(zip_path: pathlib.Path, extract_dir: pathlib.Path) -> list[str]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
        return [str(extract_dir / name) for name in zf.namelist()]


def fetch_passenger_volume(
    kind: str,
    path: str,
    key: str,
    month: str,
    out_dir: pathlib.Path,
    download: bool,
    extract: bool,
    force: bool,
) -> dict[str, Any]:
    data = request_json(path, key, {"Date": month})
    value = data.get("value", [])
    if not value or not isinstance(value, list) or "Link" not in value[0]:
        raise RuntimeError(f"{kind} did not return a download Link for Date={month}: {data}")

    link = value[0]["Link"]
    link_path = out_dir / f"PV_{kind}_{month}_link.json"
    write_json(
        link_path,
        {
            "endpoint": kind,
            "date": month,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "link": link,
            "note": "This DataMall/S3 URL is temporary; re-fetch the link if it expires.",
        },
    )

    result: dict[str, Any] = {
        "name": f"PV/{kind}",
        "date": month,
        "link_metadata": str(link_path),
    }
    if not download:
        return result

    filename = infer_download_name(kind, month, link)
    zip_path = out_dir / filename
    size = discover_size(link)
    if size:
        print(f"PV/{kind}: remote size {size / (1024 * 1024):.1f} MiB")
    download_info = download_file(link, zip_path, force=force)
    result["download"] = download_info

    if extract:
        extract_dir = out_dir / zip_path.stem
        result["extracted_files"] = extract_zip(zip_path, extract_dir)
        print(f"PV/{kind}: extracted to {extract_dir}")
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=previous_month(), help="Passenger-volume month as YYYYMM")
    parser.add_argument("--out", type=pathlib.Path, default=repo_default_out_dir(), help="Output root")
    parser.add_argument("--page-limit", type=int, default=None, help="Optional page cap for smoke tests")
    parser.add_argument("--skip-static", action="store_true", help="Do not fetch BusStops/BusRoutes/BusServices")
    parser.add_argument("--skip-traffic", action="store_true", help="Do not fetch current traffic speed bands")
    parser.add_argument("--skip-pv", action="store_true", help="Do not fetch passenger-volume download links/files")
    parser.add_argument("--no-pv-download", action="store_true", help="Only save PV temporary links; do not download zips")
    parser.add_argument("--extract-pv", action="store_true", help="Extract downloaded passenger-volume zip files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downloaded files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    key = os.environ.get("LTA_DATAMALL_KEY")
    if not key:
        print("Missing LTA_DATAMALL_KEY in environment.", file=sys.stderr)
        return 2

    root = args.out / args.date
    metadata_dir = root / "metadata"
    pv_dir = root / "passenger_volume"
    traffic_dir = root / "traffic"

    manifest: dict[str, Any] = {
        "date": args.date,
        "out_root": str(root),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "items": [],
    }

    if not args.skip_static:
        for name, path in PAGED_ENDPOINTS.items():
            manifest["items"].append(fetch_paged(name, path, key, metadata_dir, args.page_limit))

    if not args.skip_traffic:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, path in TRAFFIC_ENDPOINTS.items():
            item = fetch_paged(name, path, key, traffic_dir / stamp, args.page_limit)
            manifest["items"].append(item)

    if not args.skip_pv:
        for kind, path in PV_ENDPOINTS.items():
            item = fetch_passenger_volume(
                kind=kind,
                path=path,
                key=key,
                month=args.date,
                out_dir=pv_dir,
                download=not args.no_pv_download,
                extract=args.extract_pv,
                force=args.force,
            )
            manifest["items"].append(item)

    write_json(root / "manifest.json", manifest)
    print(f"manifest: {root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
