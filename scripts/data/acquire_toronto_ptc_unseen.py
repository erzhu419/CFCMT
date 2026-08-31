#!/usr/bin/env python3
"""Acquire the frozen Toronto v105 sources directly to shared storage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROTOCOL = "cfcmt-toronto-ptc-unseen-acquisition-v1"
USER_AGENT = "Mozilla/5.0 CFCMT-research-acquisition/1.0"
CKAN_API = (
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
)
FRAMEWORK_COMMIT = "7975f1aa01eeac62ba395e9ae66f59e5b0b1a5a9"
FRAMEWORK_URL = (
    "https://codeload.github.com/Jahandad-Baloch/TorontoSUMONetworks/"
    f"tar.gz/{FRAMEWORK_COMMIT}"
)
FRAMEWORK_NAME = f"TorontoSUMONetworks-{FRAMEWORK_COMMIT}.tar.gz"


@dataclass(frozen=True)
class FrozenResource:
    key: str
    package: str
    resource_id: str
    source_name: str
    output_name: str
    format: str
    size_bytes: int
    last_modified: str
    url: str


RESOURCES = (
    FrozenResource(
        key="road_centreline",
        package="toronto-centreline-tcl",
        resource_id="7bc94ccf-7bcf-4a7d-88b1-bdfc8ec5aaf1",
        source_name="Centreline - Version 2 - 4326.geojson",
        output_name="toronto_centreline_4326.geojson",
        format="GeoJSON",
        size_bytes=93_264_749,
        last_modified="2026-08-28T18:15:15.257968",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "1d079757-377b-4564-82df-eb5638583bfb/resource/"
            "7bc94ccf-7bcf-4a7d-88b1-bdfc8ec5aaf1/download/"
            "centreline-version-2-4326.geojson"
        ),
    ),
    FrozenResource(
        key="ward_polygons",
        package="city-wards",
        resource_id="737b29e0-8329-4260-b6af-21555ab24f28",
        source_name="City Wards Data - 4326.geojson",
        output_name="city_wards_4326.geojson",
        format="GeoJSON",
        size_bytes=1_148_140,
        last_modified="2026-02-20T19:02:44.898789",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "5e7a8234-f805-43ac-820f-03d7c360b588/resource/"
            "737b29e0-8329-4260-b6af-21555ab24f28/download/"
            "city-wards-data-4326.geojson"
        ),
    ),
    FrozenResource(
        key="community_council_polygons",
        package="community-council-boundaries",
        resource_id="cc935c56-dbcd-4035-b156-a7f8f8eae68b",
        source_name="Community Council Boundaries Data - 4326.geojson",
        output_name="community_councils_4326.geojson",
        format="GeoJSON",
        size_bytes=749_891,
        last_modified="2026-02-20T21:23:10.809795",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "5709d6ff-75a3-493d-864d-ca1b49711074/resource/"
            "cc935c56-dbcd-4035-b156-a7f8f8eae68b/download/"
            "community-council-boundaries-data-4326.geojson"
        ),
    ),
    FrozenResource(
        key="signal_locations",
        package="traffic-signals-tabular",
        resource_id="e331c953-7e49-418b-9eab-594881c76f33",
        source_name="Traffic Signal - 4326.geojson",
        output_name="traffic_signals_4326.geojson",
        format="GeoJSON",
        size_bytes=2_646_676,
        last_modified="2026-08-29T00:09:26.973606",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "1a106e88-f734-4179-b3fe-d690a6187a71/resource/"
            "e331c953-7e49-418b-9eab-594881c76f33/download/"
            "traffic-signal-4326.geojson"
        ),
    ),
    FrozenResource(
        key="signal_timing_inventory",
        package="traffic-signal-timing",
        resource_id="02c90a3a-d754-4023-a283-ed5687e87f1f",
        source_name="Traffic Signal Timing",
        output_name="traffic_signal_timing.zip",
        format="ZIP",
        size_bytes=365_673,
        last_modified="2026-08-30T13:25:05",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "7dda2235-999e-4a17-b228-abd0961e045d/resource/"
            "02c90a3a-d754-4023-a283-ed5687e87f1f/download/"
            "traffic-signals-timing.zip"
        ),
    ),
    FrozenResource(
        key="ptc_documentation",
        package="private-transportation-companies-summary-and-trip-data",
        resource_id="6c7e183b-9df4-4179-b2d0-4929afd8e8d4",
        source_name="trips_and_summary_readme",
        output_name="trips_and_summary_readme.pdf",
        format="PDF",
        size_bytes=374_725,
        last_modified="2026-07-31T19:44:37",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "fbac8e59-4883-4252-b73c-4a5caf19126d/resource/"
            "6c7e183b-9df4-4179-b2d0-4929afd8e8d4/download/"
            "trips_and_summary_readme.pdf"
        ),
    ),
    FrozenResource(
        key="ptc_daily_summary",
        package="private-transportation-companies-summary-and-trip-data",
        resource_id="c2f8c4c8-c120-480b-8713-89186488f5f5",
        source_name="summary_stats.csv",
        output_name="ptc_summary_stats.csv",
        format="CSV",
        size_bytes=497_524,
        last_modified="2026-08-01T04:35:59.670168",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "fbac8e59-4883-4252-b73c-4a5caf19126d/resource/"
            "c2f8c4c8-c120-480b-8713-89186488f5f5/download/summary_stats.csv"
        ),
    ),
    FrozenResource(
        key="ptc_trip_aggregates",
        package="private-transportation-companies-summary-and-trip-data",
        resource_id="3aef7686-488c-4430-8bfe-1e4d44625fde",
        source_name="trips_2025.zip",
        output_name="trips_2025.zip",
        format="ZIP",
        size_bytes=55_410_662,
        last_modified="2026-03-24T14:34:10",
        url=(
            "https://opendata.toronto.ca/transportation.services/"
            "private-transportation-company-dev/trips/trips_2025.zip"
        ),
    ),
    FrozenResource(
        key="midblock_dictionary",
        package=(
            "traffic-volumes-midblock-vehicle-speed-volume-and-"
            "classification-counts"
        ),
        resource_id="bed17b1a-a425-4130-a49c-67174dcf0e50",
        source_name="svc_data_dictionary.xlsx",
        output_name="svc_data_dictionary.xlsx",
        format="XLSX",
        size_bytes=56_579,
        last_modified="2025-12-02T14:17:28.900901",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "7a0ac637-43da-4e42-a79c-6d8279e21d85/resource/"
            "bed17b1a-a425-4130-a49c-67174dcf0e50/download/"
            "svc_data_dictionary_dec2025_update.xlsx"
        ),
    ),
    FrozenResource(
        key="midblock_site_summary",
        package=(
            "traffic-volumes-midblock-vehicle-speed-volume-and-"
            "classification-counts"
        ),
        resource_id="af1ccce4-d978-4a6d-9dc3-3fb33b1cc349",
        source_name="svc_summary_data.csv",
        output_name="svc_summary_data.csv",
        format="CSV",
        size_bytes=8_286_214,
        last_modified="2026-08-30T04:36:36.545099",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "7a0ac637-43da-4e42-a79c-6d8279e21d85/resource/"
            "af1ccce4-d978-4a6d-9dc3-3fb33b1cc349/download/svc_summary_data.csv"
        ),
    ),
    FrozenResource(
        key="midblock_speed_observations",
        package=(
            "traffic-volumes-midblock-vehicle-speed-volume-and-"
            "classification-counts"
        ),
        resource_id="25a7459e-b3ac-46f1-9b76-393be209b02c",
        source_name="svc_raw_data_speed_2025_2029",
        output_name="svc_raw_data_speed_2025_2029.csv",
        format="CSV",
        size_bytes=116_677_560,
        last_modified="2026-08-30T06:00:46",
        url=(
            "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
            "7a0ac637-43da-4e42-a79c-6d8279e21d85/resource/"
            "25a7459e-b3ac-46f1-9b76-393be209b02c/download/"
            "svc_raw_data_speed_2025_2029.csv"
        ),
    ),
)


def _cluster_connection(
    *, node: str, scheduler_skill_dir: Path
) -> tuple[list[str], str]:
    sys.path.insert(0, str(scheduler_skill_dir.resolve()))
    import scheduler  # type: ignore[import-not-found]

    return (
        shlex.split(scheduler._ssh_rsync_shell_for_node(node)),
        str(scheduler._ssh_target_for_node(node)),
    )


def _remote_run(
    ssh: Sequence[str], target: str, command: str, *, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*ssh, target, command],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _check_remote(result: subprocess.CompletedProcess[bytes], context: str) -> None:
    if result.returncode == 0:
        return
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    raise RuntimeError(f"{context} failed ({result.returncode}): {stderr}")


def _fetch_package(package: str) -> Mapping[str, Any]:
    request = Request(
        f"{CKAN_API}?{urlencode({'id': package})}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not payload.get("success"):
        raise ValueError(f"Toronto CKAN package lookup failed: {package}")
    return payload["result"]


def _validate_resource_metadata(
    resource: FrozenResource, observed: Mapping[str, Any]
) -> None:
    expected = {
        "id": resource.resource_id,
        "name": resource.source_name,
        "format": resource.format,
        "size": resource.size_bytes,
        "last_modified": resource.last_modified,
        "url": resource.url,
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Toronto source metadata changed for {resource.key}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _resolve_metadata() -> dict[str, Mapping[str, Any]]:
    packages = {resource.package for resource in RESOURCES}
    package_payloads = {package: _fetch_package(package) for package in packages}
    selected: dict[str, Mapping[str, Any]] = {}
    for resource in RESOURCES:
        rows = {
            row["id"]: row
            for row in package_payloads[resource.package].get("resources", [])
        }
        if resource.resource_id not in rows:
            raise ValueError(
                f"Toronto source resource disappeared: {resource.resource_id}"
            )
        observed = rows[resource.resource_id]
        _validate_resource_metadata(resource, observed)
        selected[resource.key] = observed
    return selected


def _receive_command(
    *, staging: PurePosixPath, name: str, expected_size: int | None
) -> str:
    destination = staging / name
    partial = staging / f".{name}.part"
    checks = ["set -e;", f"cat > {shlex.quote(str(partial))};"]
    if expected_size is None:
        checks.append(f"test -s {shlex.quote(str(partial))};")
    else:
        checks.append(
            f"test $(wc -c < {shlex.quote(str(partial))}) -eq {expected_size};"
        )
    checks.extend(
        [
            f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))};",
            f"sha256sum {shlex.quote(str(destination))}",
        ]
    )
    return " ".join(checks)


def _stream_url(
    *,
    ssh: Sequence[str],
    target: str,
    staging: PurePosixPath,
    name: str,
    url: str,
    expected_size: int | None,
) -> dict[str, Any]:
    transfer = subprocess.Popen(
        [
            *ssh,
            target,
            _receive_command(
                staging=staging,
                name=name,
                expected_size=expected_size,
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.sha256()
    observed_size = 0
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urlopen(request, timeout=900) as response:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                observed_size += len(block)
                digest.update(block)
                transfer.stdin.write(block)
    finally:
        transfer.stdin.close()
    stdout = transfer.stdout.read() if transfer.stdout is not None else b""
    stderr = transfer.stderr.read() if transfer.stderr is not None else b""
    result = subprocess.CompletedProcess(
        transfer.args,
        transfer.wait(),
        stdout=stdout,
        stderr=stderr,
    )
    _check_remote(result, f"Toronto transfer for {name}")
    if expected_size is not None and observed_size != expected_size:
        raise ValueError(
            f"Toronto source size changed for {name}: "
            f"expected={expected_size} observed={observed_size}"
        )
    local_sha256 = digest.hexdigest()
    remote_tokens = stdout.decode("ascii", errors="replace").split()
    remote_sha256 = remote_tokens[0] if remote_tokens else ""
    if remote_sha256 != local_sha256:
        raise ValueError(
            f"Toronto transfer digest mismatch for {name}: "
            f"local={local_sha256} remote={remote_sha256}"
        )
    return {"size_bytes": observed_size, "sha256": local_sha256}


def _manifest(
    *,
    observations: Mapping[str, Mapping[str, Any]],
    framework: Mapping[str, Any],
    framework_entry_count: int,
) -> dict[str, Any]:
    resources = {}
    for resource in RESOURCES:
        row = asdict(resource)
        row.update(observations[resource.key])
        resources[resource.key] = row
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": "all frozen Toronto v105 source resources",
        "transfer_mode": (
            "https_stream_through_local_to_shared_cluster_no_local_raw_file"
        ),
        "framework": {
            "repository": "Jahandad-Baloch/TorontoSUMONetworks",
            "commit": FRAMEWORK_COMMIT,
            "url": FRAMEWORK_URL,
            "output_name": FRAMEWORK_NAME,
            "entry_count": int(framework_entry_count),
            **framework,
        },
        "resource_count": len(resources),
        "total_frozen_resource_bytes": sum(
            resource.size_bytes for resource in RESOURCES
        ),
        "resources": resources,
    }


def acquire_remote(
    *, output_root: PurePosixPath, node: str, scheduler_skill_dir: Path
) -> dict[str, Any]:
    selected_metadata = _resolve_metadata()
    ssh, target = _cluster_connection(
        node=node, scheduler_skill_dir=scheduler_skill_dir
    )
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    initialization = _remote_run(
        ssh,
        target,
        (
            "set -e; "
            f"test ! -e {shlex.quote(str(output_root))}; "
            f"test ! -e {shlex.quote(str(staging))}; "
            f"mkdir -p {shlex.quote(str(staging))}"
        ),
    )
    _check_remote(initialization, "Toronto acquisition initialization")
    try:
        print(f"streaming framework commit {FRAMEWORK_COMMIT}", flush=True)
        framework = _stream_url(
            ssh=ssh,
            target=target,
            staging=staging,
            name=FRAMEWORK_NAME,
            url=FRAMEWORK_URL,
            expected_size=None,
        )
        framework_count_result = _remote_run(
            ssh,
            target,
            (
                "set -o pipefail; "
                f"tar -tzf {shlex.quote(str(staging / FRAMEWORK_NAME))} "
                "| wc -l"
            ),
        )
        _check_remote(framework_count_result, "Toronto framework archive audit")
        framework_entry_count = int(
            framework_count_result.stdout.decode("ascii").strip()
        )
        if framework_entry_count <= 0:
            raise ValueError("Toronto framework archive contains no entries")

        observations: dict[str, Mapping[str, Any]] = {}
        for resource in RESOURCES:
            print(
                f"streaming {resource.output_name} ({resource.size_bytes} bytes)",
                flush=True,
            )
            transfer = _stream_url(
                ssh=ssh,
                target=target,
                staging=staging,
                name=resource.output_name,
                url=resource.url,
                expected_size=resource.size_bytes,
            )
            observations[resource.key] = {
                **transfer,
                "ckan_created": selected_metadata[resource.key].get("created"),
            }

        payload = _manifest(
            observations=observations,
            framework=framework,
            framework_entry_count=framework_entry_count,
        )
        upload = _remote_run(
            ssh,
            target,
            f"cat > {shlex.quote(str(staging / 'acquisition_manifest.json'))}",
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "Toronto acquisition manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Toronto acquisition finalization")
        return payload
    except Exception:
        _remote_run(ssh, target, f"rm -rf -- {shlex.quote(str(staging))}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--remote-node", default="node001")
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
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": payload["protocol"],
                "resource_count": payload["resource_count"],
                "total_frozen_resource_bytes": payload[
                    "total_frozen_resource_bytes"
                ],
                "framework_entry_count": payload["framework"]["entry_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
