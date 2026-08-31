#!/usr/bin/env python3
"""Route and split the complete frozen Chicago v107 locatable demand."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


from scripts.data.acquire_chicago_for_hire_unseen import DATES, DEFAULT_WORKERS
from scripts.data.build_chicago_for_hire_network import PROTOCOL as NETWORK_PROTOCOL
from scripts.data.prepare_chicago_for_hire_demand import (
    EXPECTED_ANCHOR_SHA256,
    EXPECTED_NETWORK_SHA256,
    PROTOCOL as DEMAND_PROTOCOL,
)


PROTOCOL = "cfcmt-chicago-full-week-duarouter-ch-package-v1"
EXPECTED_DEMAND_MANIFEST_SHA256 = (
    "835fa16ccc50395505c95fa2c894321b80069f2829348952e9fb3e4b0e87cff1"
)
EXPECTED_PUBLISHED_TRIPS = 1_718_313
EXPECTED_INCLUDED_TRIPS = 1_422_405
EXPECTED_EXCLUDED_TRIPS = 295_908
EXPECTED_SUMO_VERSION = "1.22.0"
ROUTING_ALGORITHM = "CH"
ROUTING_THREADS = 8
ROUTING_SEED = 5307
SAFETY_SEEDS = tuple(range(5201, 5208))
END_SEC = 108_000
STEP_LENGTH_SEC = 1.0


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


def _sumo_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot query {executable} version")
    first = completed.stdout.splitlines()[0] if completed.stdout else ""
    version = first.rsplit(" ", 1)[-1]
    if version != EXPECTED_SUMO_VERSION:
        raise ValueError(f"SUMO version changed: {version}")
    return version


def duarouter_command(
    *,
    duarouter: Path,
    network_file: Path,
    trip_files: Sequence[Path],
    output_file: Path,
) -> list[str]:
    return [
        str(duarouter),
        "--net-file",
        str(network_file),
        "--route-files",
        ",".join(str(path) for path in trip_files),
        "--output-file",
        str(output_file),
        "--routing-algorithm",
        ROUTING_ALGORITHM,
        "--bulk-routing",
        "--routing-threads",
        str(ROUTING_THREADS),
        "--seed",
        str(ROUTING_SEED),
        "--stats-period",
        "100000",
    ]


def _run_logged(
    *, name: str, command: Sequence[str], cwd: Path, log_root: Path
) -> dict[str, Any]:
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    result = {
        "command": [str(value) for value in command],
        "returncode": int(completed.returncode),
        "duration_sec": time.monotonic() - started,
        "stdout_log": stdout_path.relative_to(log_root.parent).as_posix(),
        "stderr_log": stderr_path.relative_to(log_root.parent).as_posix(),
        "stdout_tail": stdout_path.read_text(encoding="utf-8")[-2_000:],
        "stderr_tail": stderr_path.read_text(encoding="utf-8")[-4_000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed ({completed.returncode}): {result['stderr_tail']}"
        )
    return result


def _validate_inputs(
    demand_root: Path, network_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    demand_path = demand_root / "demand_manifest.json"
    if _file_sha256(demand_path) != EXPECTED_DEMAND_MANIFEST_SHA256:
        raise ValueError("Chicago demand manifest identity changed")
    demand = json.loads(demand_path.read_text(encoding="utf-8"))
    expected_demand = {
        "protocol": DEMAND_PROTOCOL,
        "status": "PASS",
        "published_trip_count": EXPECTED_PUBLISHED_TRIPS,
        "included_trip_count": EXPECTED_INCLUDED_TRIPS,
        "public_location_suppressed_or_outside_count": EXPECTED_EXCLUDED_TRIPS,
        "network_sha256": EXPECTED_NETWORK_SHA256,
        "anchor_sha256": EXPECTED_ANCHOR_SHA256,
    }
    observed_demand = {key: demand.get(key) for key in expected_demand}
    if observed_demand != expected_demand:
        raise ValueError(f"Chicago demand manifest changed: {observed_demand}")
    network = json.loads(
        (network_root / "network_manifest.json").read_text(encoding="utf-8")
    )
    if {
        "protocol": network.get("protocol"),
        "status": network.get("status"),
        "network_sha256": network.get("network_sha256"),
    } != {
        "protocol": NETWORK_PROTOCOL,
        "status": "PASS",
        "network_sha256": EXPECTED_NETWORK_SHA256,
    }:
        raise ValueError("Chicago network manifest changed before routing")
    trip_files: list[Path] = []
    for row, day in zip(demand["daily"], DATES, strict=True):
        if row["date"] != day.isoformat():
            raise ValueError("Chicago demand day order changed")
        path = demand_root / str(row["output_file"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["output_size_bytes"])
            or _file_sha256(path) != str(row["output_sha256"])
        ):
            raise ValueError(f"Chicago trip file changed: {path}")
        trip_files.append(path)
    return demand, network, trip_files


def _load_input_endpoints(
    trip_files: Sequence[Path], demand: Mapping[str, Any]
) -> dict[str, tuple[str, str, float, int, str]]:
    endpoints: dict[str, tuple[str, str, float, int, str]] = {}
    counts: Counter[tuple[int, str]] = Counter()
    for day_index, path in enumerate(trip_files):
        with gzip.open(path, "rb") as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag != "trip":
                    element.clear()
                    continue
                identity = str(element.attrib["id"])
                if identity in endpoints:
                    raise ValueError(f"duplicate Chicago input trip: {identity}")
                prefix = f"d{day_index}_"
                if not identity.startswith(prefix):
                    raise ValueError(f"Chicago trip day identity changed: {identity}")
                source = identity.split("_", 2)[1]
                endpoints[identity] = (
                    str(element.attrib["from"]),
                    str(element.attrib["to"]),
                    float(element.attrib["depart"]),
                    day_index,
                    source,
                )
                counts[(day_index, source)] += 1
                element.clear()
    if len(endpoints) != EXPECTED_INCLUDED_TRIPS:
        raise ValueError(
            f"Chicago input trip inventory changed: {len(endpoints)}"
        )
    for day_index, row in enumerate(demand["daily"]):
        for source, stats in row["source_stats"].items():
            if counts[(day_index, source)] != int(stats["included_trip_count"]):
                raise ValueError(f"Chicago input source count changed: {day_index} {source}")
    return endpoints


def route_matches_endpoints(
    edges: Sequence[str], source_edge: str, target_edge: str
) -> bool:
    return bool(edges) and edges[0] == source_edge and edges[-1] == target_edge


def parse_day_source(identity: str) -> tuple[int, str]:
    parts = identity.split("_", 2)
    if len(parts) != 3 or not parts[0].startswith("d"):
        raise ValueError(f"invalid Chicago vehicle identity: {identity}")
    return int(parts[0][1:]), parts[1]


def _open_deterministic_gzip(path: Path) -> tuple[Any, Any, io.TextIOWrapper]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _split_and_audit(
    *,
    combined_path: Path,
    input_endpoints: dict[str, tuple[str, str, float, int, str]],
    staging: Path,
) -> list[dict[str, Any]]:
    output_paths = [staging / f"routes_{day.isoformat()}.rou.xml.gz" for day in DATES]
    streams = [_open_deterministic_gzip(path) for path in output_paths]
    for _, _, text in streams:
        text.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
    counts = [0] * len(DATES)
    source_counts: list[Counter[str]] = [Counter() for _ in DATES]
    route_edge_counts = [0] * len(DATES)
    minimum_route_edges: list[int | None] = [None] * len(DATES)
    maximum_route_edges = [0] * len(DATES)
    previous_depart: list[float | None] = [None] * len(DATES)
    seen: set[str] = set()
    try:
        with gzip.open(combined_path, "rb") as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if element.tag == "route":
                    continue
                if element.tag != "vehicle":
                    element.clear()
                    continue
                identity = str(element.attrib.get("id", ""))
                if identity in seen or identity not in input_endpoints:
                    raise ValueError(f"unexpected or duplicate routed vehicle: {identity}")
                seen.add(identity)
                source_edge, target_edge, expected_depart, day_index, source = (
                    input_endpoints.pop(identity)
                )
                parsed_day, parsed_source = parse_day_source(identity)
                if (parsed_day, parsed_source) != (day_index, source):
                    raise ValueError(f"Chicago routed identity changed: {identity}")
                global_depart = float(element.attrib["depart"])
                if abs(global_depart - expected_depart) > 1e-5:
                    raise ValueError(f"Chicago routed departure changed: {identity}")
                local_depart = global_depart - day_index * 86_400.0
                if not 0 <= local_depart < 86_400:
                    raise ValueError(f"Chicago routed departure left day: {identity}")
                if (
                    previous_depart[day_index] is not None
                    and local_depart < float(previous_depart[day_index]) - 1e-9
                ):
                    raise ValueError(f"Chicago routed output is not sorted: day {day_index}")
                previous_depart[day_index] = local_depart
                route = element.find("route")
                edges = [] if route is None else str(route.attrib.get("edges", "")).split()
                if not route_matches_endpoints(edges, source_edge, target_edge):
                    raise ValueError(f"Chicago route endpoints changed: {identity}")
                edge_count = len(edges)
                counts[day_index] += 1
                source_counts[day_index][source] += 1
                route_edge_counts[day_index] += edge_count
                minimum_route_edges[day_index] = (
                    edge_count
                    if minimum_route_edges[day_index] is None
                    else min(int(minimum_route_edges[day_index]), edge_count)
                )
                maximum_route_edges[day_index] = max(
                    maximum_route_edges[day_index], edge_count
                )
                element.set("depart", f"{local_depart:.6f}")
                streams[day_index][2].write(
                    ET.tostring(element, encoding="unicode", short_empty_elements=True)
                    + "\n"
                )
                element.clear()
        if input_endpoints:
            raise ValueError(
                f"duarouter omitted {len(input_endpoints)} Chicago input trips"
            )
    finally:
        for raw, compressed, text in streams:
            try:
                text.write("</routes>\n")
                text.close()
            finally:
                try:
                    compressed.close()
                finally:
                    raw.close()
    rows: list[dict[str, Any]] = []
    for day_index, (day, path) in enumerate(zip(DATES, output_paths, strict=True)):
        rows.append(
            {
                "date": day.isoformat(),
                "day_index": day_index,
                "vehicle_count": counts[day_index],
                "source_counts": dict(sorted(source_counts[day_index].items())),
                "route_edge_reference_count": route_edge_counts[day_index],
                "minimum_route_edge_count": minimum_route_edges[day_index],
                "maximum_route_edge_count": maximum_route_edges[day_index],
                "route_file": path.name,
                "route_size_bytes": path.stat().st_size,
                "route_sha256": _file_sha256(path),
            }
        )
    if sum(row["vehicle_count"] for row in rows) != EXPECTED_INCLUDED_TRIPS:
        raise ValueError("Chicago routed vehicle conservation failed")
    return rows


def strict_config_tree(
    *, network_file: Path, route_file_name: str, seed: int
) -> ET.ElementTree:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=str(network_file.resolve()))
    ET.SubElement(inputs, "route-files", value=route_file_name)
    time_section = ET.SubElement(root, "time")
    ET.SubElement(time_section, "begin", value="0")
    ET.SubElement(time_section, "step-length", value=str(STEP_LENGTH_SEC))
    ET.SubElement(time_section, "end", value=str(END_SEC))
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "route-steps", value="200")
    ET.SubElement(processing, "threads", value="1")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    ET.SubElement(processing, "scale", value="1")
    ET.SubElement(processing, "max-depart-delay", value="-1")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.SubElement(processing, "collision.check-junctions", value="true")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    ET.SubElement(processing, "time-to-teleport.highways", value="-1")
    ET.SubElement(processing, "default.carfollowmodel", value="Krauss")
    random_number = ET.SubElement(root, "random_number")
    ET.SubElement(random_number, "seed", value=str(seed))
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-step-log", value="true")
    ET.SubElement(report, "duration-log.statistics", value="true")
    ET.indent(root, space="    ")
    return ET.ElementTree(root)


def route_demand(
    *,
    demand_root: Path,
    network_root: Path,
    output_root: Path,
    duarouter: Path,
) -> dict[str, Any]:
    existing = output_root / "route_package_manifest.json"
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError("existing Chicago route package uses a different protocol")
        return payload
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    rejected = output_root.parent / f"{output_root.name}.rejected"
    if staging.exists() or rejected.exists():
        raise FileExistsError(staging if staging.exists() else rejected)
    staging.mkdir(parents=True)
    logs = staging / "logs"
    logs.mkdir()
    try:
        demand, network, trip_files = _validate_inputs(demand_root, network_root)
        executable = duarouter.resolve()
        version = _sumo_version(executable)
        combined = staging / "combined_routes.rou.xml.gz"
        command = duarouter_command(
            duarouter=executable,
            network_file=network_root / str(network["network_file"]),
            trip_files=trip_files,
            output_file=combined,
        )
        routing = _run_logged(
            name="duarouter", command=command, cwd=staging, log_root=logs
        )
        input_endpoints = _load_input_endpoints(trip_files, demand)
        daily_routes = _split_and_audit(
            combined_path=combined,
            input_endpoints=input_endpoints,
            staging=staging,
        )
        for row, demand_row, seed in zip(
            daily_routes, demand["daily"], SAFETY_SEEDS, strict=True
        ):
            if row["vehicle_count"] != int(demand_row["included_trip_count"]):
                raise ValueError(f"Chicago routed day count changed: {row['date']}")
            expected_sources = {
                source: int(stats["included_trip_count"])
                for source, stats in demand_row["source_stats"].items()
            }
            if row["source_counts"] != expected_sources:
                raise ValueError(f"Chicago routed source count changed: {row['date']}")
            config_name = f"strict_{row['date']}.sumocfg"
            config_path = staging / config_name
            strict_config_tree(
                network_file=network_root / str(network["network_file"]),
                route_file_name=str(row["route_file"]),
                seed=seed,
            ).write(config_path, encoding="utf-8", xml_declaration=True)
            row["strict_config"] = config_name
            row["strict_seed"] = seed
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sumo_version": version,
            "demand_root": str(demand_root.resolve()),
            "demand_protocol": demand["protocol"],
            "demand_manifest_sha256": EXPECTED_DEMAND_MANIFEST_SHA256,
            "network_root": str(network_root.resolve()),
            "network_protocol": network["protocol"],
            "network_sha256": EXPECTED_NETWORK_SHA256,
            "routing_algorithm": ROUTING_ALGORITHM,
            "bulk_routing": True,
            "routing_threads": ROUTING_THREADS,
            "routing_seed": ROUTING_SEED,
            "repair": False,
            "ignore_errors": False,
            "duarouter": routing,
            "combined_route_file": combined.name,
            "combined_route_size_bytes": combined.stat().st_size,
            "combined_route_sha256": _file_sha256(combined),
            "published_trip_count": EXPECTED_PUBLISHED_TRIPS,
            "routed_vehicle_count": EXPECTED_INCLUDED_TRIPS,
            "public_location_suppressed_or_outside_count": EXPECTED_EXCLUDED_TRIPS,
            "daily_routes": daily_routes,
            "strict_execution": {
                "step_length_sec": STEP_LENGTH_SEC,
                "end_sec": END_SEC,
                "demand_scale": 1.0,
                "max_depart_delay_sec": -1,
                "time_to_teleport_sec": -1,
                "collision_action": "warn",
                "collision_check_junctions": True,
                "route_errors_fatal": True,
            },
            "status": "PASS",
        }
        _write_json(staging / "route_package_manifest.json", payload)
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
        _write_json(staging / "route_failure.json", failure)
        os.replace(staging, rejected)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-root", type=Path, required=True)
    parser.add_argument("--network-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duarouter", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = route_demand(
        demand_root=args.demand_root,
        network_root=args.network_root,
        output_root=args.output_root,
        duarouter=args.duarouter,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "protocol": payload["protocol"],
                "routed_vehicle_count": payload["routed_vehicle_count"],
                "routing_algorithm": payload["routing_algorithm"],
                "routing_threads": payload["routing_threads"],
                "daily_vehicle_counts": {
                    row["date"]: row["vehicle_count"]
                    for row in payload["daily_routes"]
                },
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
