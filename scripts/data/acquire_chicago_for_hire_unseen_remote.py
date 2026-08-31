#!/usr/bin/env python3
"""HTTPS-verify Chicago v107 locally while streaming files to shared storage."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.data.acquire_chicago_for_hire_unseen import (
    BOUNDARY_SOURCE,
    DATES,
    DEFAULT_WORKERS,
    EXPECTED_SPEED_ROWS,
    EXPECTED_TRIP_ROWS,
    OSM_LAST_MODIFIED,
    OSM_MD5,
    OSM_NAME,
    OSM_SIZE_BYTES,
    OSM_URL,
    PAGE_SIZE,
    PROTOCOL,
    SPEED_COLUMNS,
    SPEED_SOURCE,
    TRIP_COLUMNS,
    TRIP_GROUP_FIELDS,
    TRIP_SOURCES,
    _request,
    _resource_url,
    _source_metadata,
    speed_page_url,
    trip_page_url,
)
from scripts.data.acquire_toronto_ptc_unseen import (
    _check_remote,
    _cluster_connection,
    _remote_run,
)


TRANSFER_MODE = "https_verified_local_stream_to_shared_cluster_no_local_raw_file"


def remote_receive_command(partial: PurePosixPath) -> str:
    return (
        "set -e; "
        f"mkdir -p {shlex.quote(str(partial.parent))}; "
        f"test ! -e {shlex.quote(str(partial))}; "
        f"cat > {shlex.quote(str(partial))}; "
        f"sha256sum {shlex.quote(str(partial))}"
    )


class RemoteSink:
    def __init__(
        self,
        *,
        ssh: Sequence[str],
        target: str,
        destination: PurePosixPath,
    ) -> None:
        self.ssh = list(ssh)
        self.target = target
        self.destination = destination
        self.partial = destination.with_name(f".{destination.name}.part")
        self.digest = hashlib.sha256()
        self.size = 0
        self.process = subprocess.Popen(
            [*self.ssh, self.target, remote_receive_command(self.partial)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise RuntimeError("remote receiver has no stdin")

    def write(self, block: bytes) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(block)
        self.digest.update(block)
        self.size += len(block)

    def close_to_partial(self) -> tuple[int, str]:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stdout = self.process.stdout.read() if self.process.stdout is not None else b""
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        result = subprocess.CompletedProcess(
            self.process.args,
            self.process.wait(),
            stdout=stdout,
            stderr=stderr,
        )
        _check_remote(result, f"Chicago remote receive for {self.destination.name}")
        tokens = stdout.decode("ascii", errors="replace").split()
        remote_digest = tokens[0] if tokens else ""
        local_digest = self.digest.hexdigest()
        if remote_digest != local_digest:
            raise ValueError(
                f"stream digest mismatch for {self.destination.name}: "
                f"{remote_digest} != {local_digest}"
            )
        return self.size, local_digest

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.communicate(timeout=5)
        except Exception:
            pass
        _remote_run(
            self.ssh,
            self.target,
            f"rm -f -- {shlex.quote(str(self.partial))}",
        )


def _finalize_remote(
    *,
    ssh: Sequence[str],
    target: str,
    destination: PurePosixPath,
    size: int,
    sha256: str,
) -> None:
    partial = destination.with_name(f".{destination.name}.part")
    command = (
        "set -e; "
        f"test ! -e {shlex.quote(str(destination))}; "
        f"test $(wc -c < {shlex.quote(str(partial))}) -eq {size}; "
        f"test $(sha256sum {shlex.quote(str(partial))} | cut -d' ' -f1) = "
        f"{shlex.quote(sha256)}; "
        f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}"
    )
    result = _remote_run(ssh, target, command)
    _check_remote(result, f"Chicago remote finalize for {destination.name}")


def _remove_remote_partial(
    ssh: Sequence[str], target: str, destination: PurePosixPath
) -> None:
    partial = destination.with_name(f".{destination.name}.part")
    _remote_run(ssh, target, f"rm -f -- {shlex.quote(str(partial))}")


def _stream_bytes(
    *,
    ssh: Sequence[str],
    target: str,
    destination: PurePosixPath,
    producer: Callable[[RemoteSink], dict[str, Any]],
    attempts: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        _remove_remote_partial(ssh, target, destination)
        sink = RemoteSink(ssh=ssh, target=target, destination=destination)
        try:
            metadata = producer(sink)
            size, sha256 = sink.close_to_partial()
            _finalize_remote(
                ssh=ssh,
                target=target,
                destination=destination,
                size=size,
                sha256=sha256,
            )
            return {**metadata, "size_bytes": size, "sha256": sha256}
        except Exception as error:
            last_error = error
            sink.abort()
            _remove_remote_partial(ssh, target, destination)
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def _emit_csv_row(
    writer: csv.DictWriter,
    buffer: io.StringIO,
    sink: RemoteSink,
    row: Mapping[str, str] | None = None,
) -> None:
    buffer.seek(0)
    buffer.truncate(0)
    if row is None:
        writer.writeheader()
    else:
        writer.writerow(row)
    sink.write(buffer.getvalue().encode("utf-8"))


def _stream_csv_page(
    *,
    url: str,
    expected_columns: Sequence[str],
    consume: Callable[[dict[str, str]], None],
) -> int:
    with urlopen(_request(url, accept="text/csv"), timeout=300) as response:
        wrapper = io.TextIOWrapper(response, encoding="utf-8", newline="")
        reader = csv.DictReader(wrapper)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(f"unexpected Socrata columns: {reader.fieldnames}")
        count = 0
        for row in reader:
            consume(row)
            count += 1
        return count


def _trip_producer(source: Any, day: Any) -> Callable[[RemoteSink], dict[str, Any]]:
    def produce(sink: RemoteSink) -> dict[str, Any]:
        text_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(text_buffer, fieldnames=TRIP_COLUMNS)
        _emit_csv_row(writer, text_buffer, sink)
        aggregate_rows = 0
        represented_rows = 0
        duplicate_groups = 0
        groups_seen: set[tuple[str, ...]] = set()

        def consume(row: dict[str, str]) -> None:
            nonlocal represented_rows, duplicate_groups
            key = tuple(row[field] for field in TRIP_GROUP_FIELDS)
            duplicate_groups += int(key in groups_seen)
            groups_seen.add(key)
            represented_rows += int(row["trip_count"])
            _emit_csv_row(writer, text_buffer, sink, row)

        offset = 0
        while True:
            page_rows = _stream_csv_page(
                url=trip_page_url(source, day, offset),
                expected_columns=TRIP_COLUMNS,
                consume=consume,
            )
            aggregate_rows += page_rows
            if page_rows < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        expected = EXPECTED_TRIP_ROWS[source.key][day.isoformat()]
        if not aggregate_rows or represented_rows != expected or duplicate_groups:
            raise ValueError(
                f"trip conservation failed for {source.key} {day}: "
                f"groups={aggregate_rows} represented={represented_rows} "
                f"expected={expected} duplicate_groups={duplicate_groups}"
            )
        return {
            "kind": "trip_aggregate",
            "source": source.key,
            "date": day.isoformat(),
            "aggregate_row_count": aggregate_rows,
            "represented_trip_count": represented_rows,
        }

    return produce


def _speed_producer(day: Any) -> Callable[[RemoteSink], dict[str, Any]]:
    def produce(sink: RemoteSink) -> dict[str, Any]:
        text_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(text_buffer, fieldnames=SPEED_COLUMNS)
        _emit_csv_row(writer, text_buffer, sink)
        row_count = 0
        positive_count = 0
        segment_ids: set[str] = set()
        record_ids: set[str] = set()
        duplicate_records = 0

        def consume(row: dict[str, str]) -> None:
            nonlocal positive_count, duplicate_records
            record_id = row["record_id"]
            duplicate_records += int(record_id in record_ids)
            record_ids.add(record_id)
            segment_ids.add(row["segment_id"])
            positive_count += int(float(row["speed"]) > 0.0)
            _emit_csv_row(writer, text_buffer, sink, row)

        offset = 0
        while True:
            page_rows = _stream_csv_page(
                url=speed_page_url(day, offset),
                expected_columns=SPEED_COLUMNS,
                consume=consume,
            )
            row_count += page_rows
            if page_rows < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        expected = EXPECTED_SPEED_ROWS[day.isoformat()]
        observed = (row_count, positive_count, len(segment_ids))
        if observed != expected or duplicate_records:
            raise ValueError(
                f"speed coverage failed for {day}: observed={observed} "
                f"expected={expected} duplicates={duplicate_records}"
            )
        return {
            "kind": "speed",
            "source": SPEED_SOURCE.key,
            "date": day.isoformat(),
            "row_count": row_count,
            "positive_speed_row_count": positive_count,
            "segment_count": len(segment_ids),
        }

    return produce


def _binary_url_producer(
    url: str,
    *,
    validate: Callable[[int, str | None, str], None] | None = None,
) -> Callable[[RemoteSink], dict[str, Any]]:
    def produce(sink: RemoteSink) -> dict[str, Any]:
        md5 = hashlib.md5()
        with urlopen(_request(url), timeout=900) as response:
            last_modified = response.headers.get("Last-Modified")
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                md5.update(block)
                sink.write(block)
        observed_md5 = md5.hexdigest()
        if validate is not None:
            validate(sink.size, last_modified, observed_md5)
        return {"url": url, "last_modified": last_modified, "md5": observed_md5}

    return produce


def _osm_validation(size: int, last_modified: str | None, md5: str) -> None:
    observed = (size, last_modified, md5)
    expected = (OSM_SIZE_BYTES, OSM_LAST_MODIFIED, OSM_MD5)
    if observed != expected:
        raise ValueError(f"Chicago OSM identity changed: {observed} != {expected}")


def _boundary_producer() -> Callable[[RemoteSink], dict[str, Any]]:
    url = _resource_url(
        BOUNDARY_SOURCE.dataset_id,
        "geojson",
        {"$limit": "100", "$order": "area_numbe"},
    )

    def produce(sink: RemoteSink) -> dict[str, Any]:
        with urlopen(_request(url, accept="application/geo+json"), timeout=180) as response:
            payload_bytes = response.read()
        payload = json.loads(payload_bytes.decode("utf-8"))
        features = payload.get("features", [])
        areas = {
            int(feature.get("properties", {}).get("area_numbe"))
            for feature in features
        }
        if len(features) != 77 or areas != set(range(1, 78)):
            raise ValueError(f"community-area inventory changed: {len(features)}")
        sink.write(payload_bytes)
        return {
            "url": url,
            "feature_count": len(features),
            "area_numbers": sorted(areas),
        }

    return produce


def _acquire_job(
    *,
    ssh: Sequence[str],
    target: str,
    staging: PurePosixPath,
    kind: str,
    source: Any,
    day: Any,
) -> dict[str, Any]:
    if kind == "trip":
        relative = PurePosixPath("trips") / f"{source.key}_{day.isoformat()}_aggregates.csv"
        producer = _trip_producer(source, day)
    else:
        relative = PurePosixPath("speed") / f"speed_{day.isoformat()}.csv"
        producer = _speed_producer(day)
    row = _stream_bytes(
        ssh=ssh,
        target=target,
        destination=staging / relative,
        producer=producer,
    )
    return {**row, "path": str(relative)}


def acquire_remote(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if workers != DEFAULT_WORKERS:
        raise ValueError(f"frozen acquisition requires {DEFAULT_WORKERS} workers")
    ssh, target = _cluster_connection(node=node, scheduler_skill_dir=scheduler_skill_dir)
    existing = _remote_run(
        ssh,
        target,
        (
            f"if test -f {shlex.quote(str(output_root / 'acquisition_manifest.json'))}; "
            f"then cat {shlex.quote(str(output_root / 'acquisition_manifest.json'))}; fi"
        ),
    )
    _check_remote(existing, "Chicago existing acquisition lookup")
    if existing.stdout.strip():
        payload = json.loads(existing.stdout.decode("utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("existing Chicago acquisition has a different protocol")
        return payload
    staging = output_root.parent / f".{output_root.name}.staging-v2"
    initialize = _remote_run(
        ssh,
        target,
        (
            "set -e; "
            f"test ! -e {shlex.quote(str(output_root))}; "
            f"test ! -e {shlex.quote(str(staging))}; "
            f"mkdir -p {shlex.quote(str(staging))}"
        ),
    )
    _check_remote(initialize, "Chicago remote acquisition initialization")
    try:
        metadata_before = _source_metadata()
        boundaries = _stream_bytes(
            ssh=ssh,
            target=target,
            destination=staging / "community_areas.geojson",
            producer=_boundary_producer(),
        )
        boundaries["path"] = "community_areas.geojson"
        osm = _stream_bytes(
            ssh=ssh,
            target=target,
            destination=staging / OSM_NAME,
            producer=_binary_url_producer(OSM_URL, validate=_osm_validation),
        )
        osm.update({"path": OSM_NAME, "publisher_md5": OSM_MD5})

        jobs: list[tuple[str, Any, Any]] = []
        for day in DATES:
            jobs.extend(("trip", source, day) for source in TRIP_SOURCES)
            jobs.append(("speed", None, day))
        observations: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _acquire_job,
                    ssh=ssh,
                    target=target,
                    staging=staging,
                    kind=kind,
                    source=source,
                    day=day,
                ): (kind, source, day)
                for kind, source, day in jobs
            }
            for future in as_completed(futures):
                kind, source, day = futures[future]
                row = future.result()
                observations.append(row)
                label = source.key if source is not None else kind
                count = row.get("row_count", row.get("represented_trip_count"))
                print(f"acquired {label} {day}: {count}", flush=True)

        metadata_after = _source_metadata()
        if metadata_after != metadata_before:
            raise ValueError("Chicago Socrata metadata changed during acquisition")
        observations.sort(key=lambda row: (str(row["date"]), str(row["source"])))
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "date_interval": [
                DATES[0].isoformat(),
                (DATES[-1] + timedelta(days=1)).isoformat(),
            ],
            "coverage": (
                "all published TNP and taxi rows represented in grouped files; "
                "all historical speed rows for seven complete days"
            ),
            "transfer_mode": TRANSFER_MODE,
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
        upload = _remote_run(
            ssh,
            target,
            f"cat > {shlex.quote(str(staging / 'acquisition_manifest.json'))}",
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "Chicago acquisition manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Chicago acquisition finalization")
        return payload
    except Exception:
        _remote_run(ssh, target, f"rm -rf -- {shlex.quote(str(staging))}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--remote-node", default="node001")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--scheduler-skill-dir",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm/skill"),
    )
    args = parser.parse_args(argv)
    payload = acquire_remote(
        output_root=PurePosixPath(args.remote_output_root),
        node=args.remote_node,
        scheduler_skill_dir=args.scheduler_skill_dir,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": payload["protocol"],
                "workers": payload["workers"],
                "trip_source_totals": payload["trip_source_totals"],
                "speed_row_total": payload["speed_row_total"],
                "remote_output_root": args.remote_output_root,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
