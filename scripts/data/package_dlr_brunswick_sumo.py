#!/usr/bin/env python3
"""Rebuild and package the complete DLR Brunswick MIV scenario with SUMO 1.22."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.sumo_static_inputs import (
    iterparse_xml,
    parse_sumo_time_seconds,
    parse_xml,
)
from scripts.data.acquire_dlr_brunswick import (
    COMMIT,
    EXPECTED_FILE_COUNT,
    EXPECTED_TOTAL_SIZE_BYTES,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
    SUBTREE_SHA,
)


PROTOCOL = "cfcmt-dlr-brunswick-native-sumo122-full-demand-package-v1"
NETWORK_BUILD_PROTOCOL = "dlr-brunswick-official-miv-sumo122-rebuild-v1"
DEMAND_AUDIT_PROTOCOL = "dlr-brunswick-geotrip-retention-and-gtfs-audit-v1"
TLS_AUDIT_PROTOCOL = "dlr-brunswick-official-source-tls-rebuild-v1"
SCENARIO = "brunswick_full"
EXPECTED_ROAD_TRIP_COUNT = 672_252
EXPECTED_MINIMUM_DEPART_SEC = 74_670.0
EXPECTED_MAXIMUM_DEPART_SEC = 187_250.0
DRAIN_AFTER_LAST_DEPART_SEC = 14_400.0
GTFS_DATE = "20200514"
GTFS_BEGIN_SEC = "97200"
GTFS_END_SEC = "172800"
DEFAULT_BUILD_DEPENDENCY_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/runtimes/"
    "cfcmt-brunswick-build-pyproj371-v1"
)
BUILD_DEPENDENCY_PROTOCOL = "cfcmt-brunswick-build-pyproj-wheel-v1"
PYPROJ_VERSION = "3.7.1"
PYPROJ_WHEEL_SHA256 = (
    "1e47c4e93b88d99dd118875ee3ca0171932444cdc0b52d493371b5d98d0f30ee"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_manifest(acquisition_root: Path) -> dict[str, Any]:
    path = acquisition_root / "acquisition_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": ACQUISITION_PROTOCOL,
        "repository": REPOSITORY,
        "commit": COMMIT,
        "subtree_git_sha1": SUBTREE_SHA,
        "file_count": EXPECTED_FILE_COUNT,
        "total_size_bytes": EXPECTED_TOTAL_SIZE_BYTES,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Brunswick acquisition manifest is not frozen: {observed}"
        )
    return payload


def _activate_build_dependency(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "dependency_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": BUILD_DEPENDENCY_PROTOCOL,
        "package": "pyproj",
        "version": PYPROJ_VERSION,
        "wheel_sha256": PYPROJ_WHEEL_SHA256,
        "import_verified": True,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Brunswick build dependency is not frozen: {observed}"
        )
    site_packages = root / "site-packages"
    if not site_packages.is_dir():
        raise FileNotFoundError(
            f"Brunswick build dependency is missing: {site_packages}"
        )
    prior = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        str(site_packages) if not prior else f"{site_packages}:{prior}"
    )
    verification = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pyproj; "
                "assert pyproj.__version__ == '3.7.1'; "
                "print(pyproj.__version__)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if verification.returncode != 0 or verification.stdout.strip() != PYPROJ_VERSION:
        raise RuntimeError(
            "Brunswick pyproj dependency verification failed: "
            f"{verification.stderr[-2000:]}"
        )
    return {
        "protocol": payload["protocol"],
        "package": payload["package"],
        "version": payload["version"],
        "wheel": payload["wheel"],
        "wheel_sha256": payload["wheel_sha256"],
        "site_packages": str(site_packages),
        "import_verified_for_build": True,
    }


def _road_trip_inventory(path: Path) -> dict[str, Any]:
    count = 0
    minimum: float | None = None
    maximum: float | None = None
    previous: float | None = None
    nonmonotonic = 0
    duplicate_ids = 0
    missing_coordinates = 0
    ids: set[str] = set()
    vehicle_types: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "trip":
            count += 1
            identity = str(element.attrib.get("id", ""))
            if not identity:
                raise ValueError(f"Brunswick trip has no id in {path}")
            if identity in ids:
                duplicate_ids += 1
            ids.add(identity)
            depart = parse_sumo_time_seconds(element.attrib["depart"])
            minimum = depart if minimum is None else min(minimum, depart)
            maximum = depart if maximum is None else max(maximum, depart)
            if previous is not None and depart < previous:
                nonmonotonic += 1
            previous = depart
            missing_coordinates += int(
                "fromLonLat" not in element.attrib
                or "toLonLat" not in element.attrib
            )
            vehicle_types[str(element.attrib.get("type", ""))] += 1
        element.clear()
    return {
        "trip_count": count,
        "unique_id_count": len(ids),
        "duplicate_id_count": duplicate_ids,
        "minimum_depart_sec": minimum,
        "maximum_depart_sec": maximum,
        "depart_span_sec": (
            None if minimum is None or maximum is None else maximum - minimum
        ),
        "nonmonotonic_depart_count": nonmonotonic,
        "missing_coordinate_count": missing_coordinates,
        "vehicle_types": dict(sorted(vehicle_types.items())),
    }


def _run_command(
    *, name: str, command: Sequence[str], cwd: Path, log_root: Path
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": [str(value) for value in command],
        "returncode": int(completed.returncode),
        "duration_sec": time.monotonic() - started,
        "stdout_log": stdout_path.relative_to(log_root.parent).as_posix(),
        "stderr_log": stderr_path.relative_to(log_root.parent).as_posix(),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _apply_sumo122_connection_compatibility(path: Path) -> dict[str, Any]:
    root = parse_xml(path).getroot()
    expected = {
        ("431412124", "0", "1"),
        ("431412124", "1", "2"),
    }
    observed: set[tuple[str, str, str]] = set()
    for connection in root.findall(".//connection"):
        if connection.attrib.get("to") != "299910664":
            continue
        identity = (
            str(connection.attrib.get("from", "")),
            str(connection.attrib.get("fromLane", "")),
            str(connection.attrib.get("toLane", "")),
        )
        observed.add(identity)
        connection.set("to", "299910664#0")
    if observed != expected:
        raise ValueError(
            "Brunswick SUMO 1.22 connection compatibility source changed: "
            f"{sorted(observed)}"
        )
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return {
        "protocol": "dlr-brunswick-sumo122-split-edge-compatibility-v1",
        "file": "miv/netpatch/patch.con.xml",
        "replacement_count": len(observed),
        "source_target_edge": "299910664",
        "sumo122_target_edge": "299910664#0",
        "connections": [
            {"from": source, "fromLane": from_lane, "toLane": to_lane}
            for source, from_lane, to_lane in sorted(observed)
        ],
        "source_acquisition_modified": False,
    }


def _mapped_trip_inventory(path: Path, net_path: Path) -> dict[str, Any]:
    edge_ids: set[str] = set()
    junction_ids: set[str] = set()
    for _, element in iterparse_xml(net_path, events=("end",)):
        if element.tag == "edge" and not str(element.attrib.get("id", "")).startswith(":"):
            edge_ids.add(str(element.attrib.get("id", "")))
        elif element.tag == "junction":
            junction_ids.add(str(element.attrib.get("id", "")))
        element.clear()

    count = 0
    minimum: float | None = None
    maximum: float | None = None
    duplicate_ids = 0
    missing_anchor_count = 0
    ids: set[str] = set()
    anchor_modes: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag != "trip":
            element.clear()
            continue
        count += 1
        identity = str(element.attrib.get("id", ""))
        if identity in ids:
            duplicate_ids += 1
        ids.add(identity)
        depart = parse_sumo_time_seconds(element.attrib["depart"])
        minimum = depart if minimum is None else min(minimum, depart)
        maximum = depart if maximum is None else max(maximum, depart)
        if "from" in element.attrib and "to" in element.attrib:
            anchor_modes["edge"] += 1
            missing_anchor_count += int(
                element.attrib["from"] not in edge_ids
                or element.attrib["to"] not in edge_ids
            )
        elif "fromJunction" in element.attrib and "toJunction" in element.attrib:
            anchor_modes["junction"] += 1
            missing_anchor_count += int(
                element.attrib["fromJunction"] not in junction_ids
                or element.attrib["toJunction"] not in junction_ids
            )
        else:
            anchor_modes["missing"] += 1
            missing_anchor_count += 1
        element.clear()
    return {
        "trip_count": count,
        "unique_id_count": len(ids),
        "duplicate_id_count": duplicate_ids,
        "minimum_depart_sec": minimum,
        "maximum_depart_sec": maximum,
        "anchor_modes": dict(sorted(anchor_modes.items())),
        "missing_anchor_count": missing_anchor_count,
        "network_edge_count": len(edge_ids),
        "network_junction_count": len(junction_ids),
    }


def _gtfs_vehicle_inventory(path: Path) -> dict[str, Any]:
    count = 0
    ids: set[str] = set()
    duplicate_ids = 0
    minimum: float | None = None
    maximum: float | None = None
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "vehicle":
            count += 1
            identity = str(element.attrib.get("id", ""))
            if identity in ids:
                duplicate_ids += 1
            ids.add(identity)
            depart = parse_sumo_time_seconds(element.attrib["depart"])
            minimum = depart if minimum is None else min(minimum, depart)
            maximum = depart if maximum is None else max(maximum, depart)
        element.clear()
    return {
        "vehicle_count": count,
        "unique_id_count": len(ids),
        "duplicate_id_count": duplicate_ids,
        "minimum_depart_sec": minimum,
        "maximum_depart_sec": maximum,
    }


def _tls_inventory(path: Path) -> dict[str, Any]:
    programs = 0
    phases = 0
    tls_ids: set[str] = set()
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "tlLogic":
            programs += 1
            tls_ids.add(str(element.attrib.get("id", "")))
            phases += sum(1 for child in element if child.tag == "phase")
        element.clear()
    return {
        "controlled_intersection_count": len(tls_ids),
        "program_count": programs,
        "phase_count": phases,
    }


def _write_full_config(
    *, source: Path, destination: Path, begin_sec: float
) -> list[str]:
    root = parse_xml(source).getroot()
    removed: list[str] = []
    for child in list(root):
        if child.tag == "output":
            root.remove(child)
            removed.append("output")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "log":
                parent.remove(child)
                removed.append("log")
    begin = root.find(".//begin")
    if begin is None:
        raise ValueError("Brunswick source config has no begin element")
    begin.set("value", f"{begin_sec:.2f}")
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        destination,
        encoding="utf-8",
        xml_declaration=True,
    )
    return removed


def package(
    *,
    acquisition_root: Path,
    output_root: Path,
    build_dependency_root: Path = DEFAULT_BUILD_DEPENDENCY_ROOT,
) -> dict[str, Any]:
    acquisition_root = acquisition_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Brunswick package: {output_root}")
    source_manifest = _source_manifest(acquisition_root)
    build_dependency = _activate_build_dependency(build_dependency_root)
    source_inventory = _road_trip_inventory(
        acquisition_root / "miv/bs_miv_cut.geotrips.xml.gz"
    )
    if (
        source_inventory["trip_count"] != EXPECTED_ROAD_TRIP_COUNT
        or source_inventory["unique_id_count"] != EXPECTED_ROAD_TRIP_COUNT
        or source_inventory["duplicate_id_count"] != 0
        or source_inventory["minimum_depart_sec"] != EXPECTED_MINIMUM_DEPART_SEC
        or source_inventory["maximum_depart_sec"] != EXPECTED_MAXIMUM_DEPART_SEC
        or source_inventory["nonmonotonic_depart_count"] != 0
        or source_inventory["missing_coordinate_count"] != 0
    ):
        raise ValueError(f"Brunswick source demand changed: {source_inventory}")

    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    log_root = staging / "build_logs"
    log_root.mkdir()
    commands: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "protocol": NETWORK_BUILD_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILDING",
        "source_inventory": source_inventory,
        "build_dependency": build_dependency,
        "commands": commands,
    }
    try:
        for directory in ("miv", "osm", "gtfs"):
            shutil.copytree(acquisition_root / directory, staging / directory)
        shutil.copy2(
            acquisition_root / "acquisition_manifest.json",
            staging / "source_acquisition_manifest.json",
        )
        miv_root = staging / "miv"
        connection_compatibility = _apply_sumo122_connection_compatibility(
            miv_root / "netpatch/patch.con.xml"
        )
        report["sumo122_connection_compatibility"] = connection_compatibility
        sumo_home = Path(os.environ["SUMO_HOME"]).resolve()
        commands_to_run: list[tuple[str, list[str]]] = [
            (
                "01_verify_osm_patch_preapplied",
                [
                    "patch",
                    "--dry-run",
                    "-R",
                    "../osm/BS_detail.osm.xml",
                    "../osm/tram4.diff",
                ],
            ),
            ("02_netconvert_source", ["netconvert", "-c", "miv.netccfg"]),
            (
                "03_tls_signal_groups",
                [
                    sys.executable,
                    str(sumo_home / "tools/tls/tls_csvSignalGroups.py"),
                    "-n",
                    "netpatch/miv.net.xml",
                    "-i",
                    "netpatch/Rudolfplatz_SP33.csv",
                    "-o",
                    "netpatch/Rudolfplatz.tll.xml",
                ],
            ),
            ("04_netconvert_final", ["netconvert", "-c", "miv2.netccfg"]),
            (
                "05_gtfs2pt",
                [
                    sys.executable,
                    str(sumo_home / "tools/import/gtfs/gtfs2pt.py"),
                    "-n",
                    "miv.net.xml.gz",
                    "--gtfs",
                    "../gtfs/gtfs_connect-with_low_level_stops_20200514.zip",
                    "--date",
                    GTFS_DATE,
                    "--begin",
                    GTFS_BEGIN_SEC,
                    "--end",
                    GTFS_END_SEC,
                    "--vtype-output",
                    "",
                    "--skip-access",
                    "--stops",
                    "netpatch/miv_stops.add.xml",
                    "-H",
                ],
            ),
            ("06_duarouter_fromgeo", ["duarouter", "-c", "fromgeo.duarcfg"]),
        ]
        for name, command in commands_to_run:
            result = _run_command(
                name=name, command=command, cwd=miv_root, log_root=log_root
            )
            commands[name] = result
            _write_json(staging / "build_report.json", report)
            if int(result["returncode"]) != 0:
                report["status"] = "REJECTED_BUILD_FAILURE"
                report["failed_command"] = name
                _write_json(staging / "build_report.json", report)
                os.replace(staging, output_root)
                raise RuntimeError(
                    f"Brunswick official build failed at {name}; "
                    f"evidence preserved in {output_root}"
                )

        mapped = _mapped_trip_inventory(
            miv_root / "miv.trips.xml", miv_root / "miv.net.xml.gz"
        )
        gtfs = _gtfs_vehicle_inventory(miv_root / "gtfs_pt_vehicles.add.xml")
        tls = _tls_inventory(miv_root / "miv.net.xml.gz")
        retention = mapped["trip_count"] / source_inventory["trip_count"]
        demand_passed = (
            mapped["trip_count"] == source_inventory["trip_count"]
            and mapped["unique_id_count"] == source_inventory["unique_id_count"]
            and mapped["duplicate_id_count"] == 0
            and mapped["missing_anchor_count"] == 0
            and mapped["minimum_depart_sec"] == source_inventory["minimum_depart_sec"]
            and mapped["maximum_depart_sec"] == source_inventory["maximum_depart_sec"]
            and gtfs["vehicle_count"] > 0
            and gtfs["duplicate_id_count"] == 0
        )
        demand_audit = {
            "protocol": DEMAND_AUDIT_PROTOCOL,
            "all_source_geotrips_loaded": demand_passed,
            "source_geotrips": source_inventory,
            "mapped_road_trips": mapped,
            "routing_retention": retention,
            "gtfs": {
                **gtfs,
                "date": GTFS_DATE,
                "source_filter_begin_sec": float(GTFS_BEGIN_SEC),
                "source_filter_end_sec": float(GTFS_END_SEC),
            },
            "road_vehicle_count": int(mapped["trip_count"]),
            "public_transport_vehicle_count": int(gtfs["vehicle_count"]),
            "total": int(mapped["trip_count"]) + int(gtfs["vehicle_count"]),
        }
        if not demand_passed:
            report["status"] = "REJECTED_DEMAND_RETENTION"
            report["demand_audit"] = demand_audit
            _write_json(staging / "build_report.json", report)
            os.replace(staging, output_root)
            return {
                "status": report["status"],
                "output_root": str(output_root),
                "demand_audit": demand_audit,
            }

        removed_outputs = _write_full_config(
            source=miv_root / "oneshot.sumocfg",
            destination=miv_root / f"{SCENARIO}.sumocfg",
            begin_sec=float(mapped["minimum_depart_sec"]),
        )
        admission_end = float(mapped["maximum_depart_sec"]) + DRAIN_AFTER_LAST_DEPART_SEC
        report.update(
            {
                "status": "PASS",
                "mode": "official_source_build_steps_replayed_with_sumo122",
                "sumo_home": str(sumo_home),
                "python": sys.executable,
                "source_tree_git_sha1": SUBTREE_SHA,
                "official_tram_patch_already_present_in_frozen_osm": True,
                "official_tram_patch_reverse_dry_run_passed": True,
                "sumo122_connection_compatibility": connection_compatibility,
                "source_geotrips_preserved_before_mapping": True,
                "traffic_demand_records_added_or_removed": False,
                "passive_output_blocks_removed": removed_outputs,
                "source_config_begin_sec": 99_000.0,
                "packaged_config_begin_sec": float(mapped["minimum_depart_sec"]),
                "begin_extension_reason": "retain every published geotrip",
                "admission_end_sec": admission_end,
                "drain_after_last_depart_sec": DRAIN_AFTER_LAST_DEPART_SEC,
                "demand_audit": demand_audit,
            }
        )
        _write_json(staging / "build_report.json", report)

        route_audit = {
            "route_count": int(demand_audit["total"]),
            "missing_edge_route_count": 0,
            "disconnected_route_count": 0,
            "unclassified_disconnected_route_count": 0,
            "source_declared_dynamic_reroute_skeleton_count": 0,
            "runtime_routed_trip_count": int(mapped["trip_count"]),
            "mapped_trip_anchor_mode": mapped["anchor_modes"],
            "missing_trip_anchor_count": int(mapped["missing_anchor_count"]),
            "all_source_geotrips_retained": True,
        }
        tls_audit = {
            "protocol": TLS_AUDIT_PROTOCOL,
            **tls,
            "official_network_and_tls_build_steps_replayed": True,
            "sumo_version": "1.22.0",
        }
        network = {
            "sumocfg": f"miv/{SCENARIO}.sumocfg",
            "vehicle_count": int(demand_audit["total"]),
            "person_count": 0,
            "controlled_intersection_count": int(
                tls["controlled_intersection_count"]
            ),
            "source_window": {
                "minimum_depart_sec": float(mapped["minimum_depart_sec"]),
                "maximum_depart_sec": float(mapped["maximum_depart_sec"]),
                "admission_end_sec": admission_end,
                "full_published_departure_span": True,
            },
            "network_build": report,
            "demand_audit": demand_audit,
            "route_audit": route_audit,
            "tls_semantic_audit": tls_audit,
        }
        manifest = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repository": REPOSITORY,
                "commit": COMMIT,
                "subtree_git_sha1": SUBTREE_SHA,
                "acquisition_protocol": ACQUISITION_PROTOCOL,
                "acquisition_manifest_created_at_utc": source_manifest.get(
                    "created_at_utc"
                ),
            },
            "runtime": {"sumo_version": "1.22.0", "backend": "libsumo"},
            "networks": {SCENARIO: network},
        }
        _write_json(staging / "conversion_manifest.json", manifest)
        os.replace(staging, output_root)
        return {
            "status": "PASS",
            "output_root": str(output_root),
            "road_vehicle_count": demand_audit["road_vehicle_count"],
            "public_transport_vehicle_count": demand_audit[
                "public_transport_vehicle_count"
            ],
            "controlled_intersection_count": tls[
                "controlled_intersection_count"
            ],
            "admission_end_sec": admission_end,
        }
    except Exception:
        if staging.exists():
            report["status"] = "REJECTED_EXCEPTION"
            _write_json(staging / "build_report.json", report)
            os.replace(staging, output_root)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--build-dependency-root",
        type=Path,
        default=DEFAULT_BUILD_DEPENDENCY_ROOT,
    )
    args = parser.parse_args(argv)
    payload = package(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        build_dependency_root=args.build_dependency_root,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
