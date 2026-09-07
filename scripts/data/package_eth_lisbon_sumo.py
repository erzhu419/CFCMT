#!/usr/bin/env python3
"""Package and audit the complete ETH Lisbon demand for microscopic SUMO."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.sumo_static_inputs import (  # noqa: E402
    iterparse_xml,
    parse_sumo_time_seconds,
    parse_xml,
)
from scripts.data.acquire_eth_five_city_sumo import (  # noqa: E402
    ARCHIVE_NAME,
    DOI,
    EXPECTED_MD5,
    EXPECTED_SIZE_BYTES,
    PROTOCOL as ACQUISITION_PROTOCOL,
)


PROTOCOL = "cfcmt-eth-lisbon-microscopic-full-demand-package-v1"
INPUT_AUDIT_PROTOCOL = "eth-lisbon-full-demand-static-input-audit-v1"
MICRO_CONFIG_PROTOCOL = "eth-lisbon-derived-microscopic-sumo122-config-v1"
SCENARIO = "eth_lisbon_full_day_micro"
EXPECTED_LIS_ENTRIES = (
    "LIS/",
    "LIS/additional.add.xml",
    "LIS/LIS.net.xml",
    "LIS/meso.sumo.cfg",
    "LIS/taz.xml",
    "LIS/trips24h_smoothed.rou.xml",
)
MINIMUM_DEMAND_SPAN_SEC = 23 * 3600
MAXIMUM_DEMAND_SPAN_SEC = 25 * 3600
DRAIN_AFTER_LAST_DEPART_SEC = 6 * 3600


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
        "doi": DOI,
        "archive_name": ARCHIVE_NAME,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "md5": EXPECTED_MD5,
        "archive_entry_count": 30,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"ETH acquisition manifest is not frozen: {observed}")
    archive = acquisition_root / ARCHIVE_NAME
    if archive.stat().st_size != EXPECTED_SIZE_BYTES:
        raise ValueError("ETH archive size changed after acquisition")
    entries = tuple(
        (acquisition_root / "archive_entries.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    lis_entries = tuple(row for row in entries if row.startswith("LIS/"))
    if lis_entries != EXPECTED_LIS_ENTRIES:
        raise ValueError(f"ETH Lisbon archive entries changed: {lis_entries}")
    return payload


def _network_inventory(path: Path) -> tuple[dict[str, Any], set[str]]:
    edge_ids: set[str] = set()
    junction_ids: set[str] = set()
    tls_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        tag = element.tag
        counts[tag] += 1
        if tag == "edge":
            identity = str(element.attrib.get("id", ""))
            if not identity:
                raise ValueError("Lisbon network edge has no id")
            if not identity.startswith(":"):
                edge_ids.add(identity)
        elif tag == "junction":
            identity = str(element.attrib.get("id", ""))
            if identity:
                junction_ids.add(identity)
        elif tag == "tlLogic":
            identity = str(element.attrib.get("id", ""))
            if identity:
                tls_ids.add(identity)
        element.clear()
    return (
        {
            "edge_count": counts["edge"],
            "noninternal_edge_count": len(edge_ids),
            "lane_count": counts["lane"],
            "junction_count": len(junction_ids),
            "connection_count": counts["connection"],
            "traffic_light_count": len(tls_ids),
            "traffic_light_program_count": counts["tlLogic"],
            "traffic_light_phase_count": counts["phase"],
        },
        edge_ids,
    )


def _taz_inventory(path: Path) -> tuple[dict[str, Any], set[str]]:
    taz_ids: set[str] = set()
    duplicate_ids = 0
    taz_source_count = 0
    taz_sink_count = 0
    for _, element in iterparse_xml(path, events=("end",)):
        if element.tag == "taz":
            identity = str(element.attrib.get("id", ""))
            if not identity:
                raise ValueError("Lisbon TAZ has no id")
            duplicate_ids += int(identity in taz_ids)
            taz_ids.add(identity)
        elif element.tag == "tazSource":
            taz_source_count += 1
        elif element.tag == "tazSink":
            taz_sink_count += 1
        element.clear()
    return (
        {
            "taz_count": len(taz_ids),
            "duplicate_taz_id_count": duplicate_ids,
            "taz_source_count": taz_source_count,
            "taz_sink_count": taz_sink_count,
        },
        taz_ids,
    )


def _route_inventory(
    path: Path, *, edge_ids: set[str], taz_ids: set[str]
) -> dict[str, Any]:
    trip_ids: set[str] = set()
    duplicate_ids = 0
    missing_edge_anchors = 0
    missing_taz_anchors = 0
    missing_attributes = 0
    nonmonotonic_departures = 0
    minimum: float | None = None
    maximum: float | None = None
    previous: float | None = None
    tags: Counter[str] = Counter()
    for _, element in iterparse_xml(path, events=("end",)):
        tags[element.tag] += 1
        if element.tag != "trip":
            element.clear()
            continue
        identity = str(element.attrib.get("id", ""))
        source = str(element.attrib.get("from", ""))
        destination = str(element.attrib.get("to", ""))
        depart_text = element.attrib.get("depart")
        if not identity or not source or not destination or depart_text is None:
            missing_attributes += 1
            element.clear()
            continue
        duplicate_ids += int(identity in trip_ids)
        trip_ids.add(identity)
        missing_edge_anchors += int(
            source not in edge_ids or destination not in edge_ids
        )
        from_taz = element.attrib.get("fromTaz")
        to_taz = element.attrib.get("trip_toTaz")
        missing_taz_anchors += int(
            from_taz is None
            or to_taz is None
            or from_taz not in taz_ids
            or to_taz not in taz_ids
        )
        depart = parse_sumo_time_seconds(depart_text)
        minimum = depart if minimum is None else min(minimum, depart)
        maximum = depart if maximum is None else max(maximum, depart)
        if previous is not None and depart < previous:
            nonmonotonic_departures += 1
        previous = depart
        element.clear()
    span = None if minimum is None or maximum is None else maximum - minimum
    return {
        "trip_count": tags["trip"],
        "unique_trip_id_count": len(trip_ids),
        "duplicate_trip_id_count": duplicate_ids,
        "missing_required_attribute_count": missing_attributes,
        "missing_edge_anchor_count": missing_edge_anchors,
        "missing_taz_anchor_count": missing_taz_anchors,
        "minimum_depart_sec": minimum,
        "maximum_depart_sec": maximum,
        "depart_span_sec": span,
        "nonmonotonic_departure_count": nonmonotonic_departures,
        "other_route_element_counts": {
            key: value for key, value in sorted(tags.items()) if key != "trip"
        },
    }


def _native_config_inventory(path: Path) -> dict[str, Any]:
    root = parse_xml(path).getroot()

    def value(section: str, tag: str) -> str | None:
        element = root.find(f"./{section}/{tag}")
        return None if element is None else element.attrib.get("value")

    return {
        "net_file": value("input", "net-file"),
        "route_files": value("input", "route-files"),
        "additional_files": value("input", "additional-files"),
        "begin_sec": value("time", "begin"),
        "end_sec": value("time", "end"),
        "step_length_sec": value("time", "step-length"),
        "mesosim": value("mesoscopic", "mesosim"),
        "ignore_route_errors": value("processing", "ignore-route-errors"),
        "collision_action": value("processing", "collision.action"),
        "time_to_teleport_sec": value("processing", "time-to-teleport"),
        "max_depart_delay_sec": value("processing", "max-depart-delay"),
        "random_depart_offset_sec": value(
            "processing", "random-depart-offset"
        ),
        "rerouting_probability": value(
            "routing", "device.rerouting.probability"
        ),
        "rerouting_period_sec": value(
            "routing", "device.rerouting.period"
        ),
        "seed": value("random_number", "seed"),
    }


def _write_microscopic_additional(source: Path, destination: Path) -> dict[str, Any]:
    source_root = parse_xml(source).getroot()
    output_tags = {"edgeData", "laneData", "inductionLoop", "laneAreaDetector"}
    removed = []
    for child in list(source_root):
        if child.tag in output_tags:
            removed.append({"tag": child.tag, "id": child.attrib.get("id")})
            source_root.remove(child)
    ET.indent(source_root, space="    ")
    ET.ElementTree(source_root).write(
        destination, encoding="utf-8", xml_declaration=True
    )
    return {
        "protocol": "eth-lisbon-passive-output-removal-v1",
        "removed_passive_output_elements": removed,
        "remaining_element_count": len(source_root),
        "demand_or_network_elements_removed": 0,
    }


def _write_microscopic_config(
    *, destination: Path, begin_sec: float, end_sec: float, seed: int
) -> None:
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value="source/LIS/LIS.net.xml")
    ET.SubElement(
        inputs,
        "route-files",
        value="source/LIS/trips24h_smoothed.rou.xml",
    )
    ET.SubElement(
        inputs,
        "additional-files",
        value="microscopic_inputs.add.xml,source/LIS/taz.xml",
    )
    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value=str(int(math.floor(begin_sec))))
    ET.SubElement(time, "end", value=str(int(math.ceil(end_sec))))
    ET.SubElement(time, "step-length", value="1")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    ET.SubElement(processing, "collision.action", value="warn")
    ET.SubElement(processing, "collision.check-junctions", value="true")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    ET.SubElement(processing, "max-depart-delay", value="-1")
    ET.SubElement(processing, "random-depart-offset", value="1200")
    ET.SubElement(processing, "time-to-impatience", value="30")
    ET.SubElement(processing, "default.speeddev", value="0.28")
    routing = ET.SubElement(root, "routing")
    ET.SubElement(routing, "routing-algorithm", value="CH")
    ET.SubElement(routing, "weights.random-factor", value="1.7")
    ET.SubElement(routing, "device.rerouting.probability", value="0.5")
    ET.SubElement(routing, "device.rerouting.period", value="360")
    ET.SubElement(routing, "device.rerouting.pre-period", value="0")
    ET.SubElement(routing, "device.rerouting.adaptation-steps", value="3")
    ET.SubElement(routing, "device.rerouting.adaptation-interval", value="180")
    ET.SubElement(routing, "device.rerouting.with-taz", value="true")
    ET.SubElement(routing, "device.rerouting.threads", value="15")
    ET.SubElement(routing, "device.rerouting.synchronize", value="true")
    random_number = ET.SubElement(root, "random_number")
    ET.SubElement(random_number, "random", value="false")
    ET.SubElement(random_number, "seed", value=str(int(seed)))
    report = ET.SubElement(root, "report")
    ET.SubElement(report, "xml-validation", value="never")
    ET.SubElement(report, "xml-validation.routes", value="never")
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(
        destination, encoding="utf-8", xml_declaration=True
    )


def package_remote_source(
    *, acquisition_root: Path, output_root: Path, seed: int
) -> dict[str, Any]:
    source_manifest = _source_manifest(acquisition_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        extraction = subprocess.run(
            [
                "unzip",
                "-q",
                str(acquisition_root / ARCHIVE_NAME),
                "LIS/*",
                "-d",
                str(staging / "source"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if extraction.returncode != 0:
            raise RuntimeError(f"Lisbon extraction failed: {extraction.stderr}")
        lis = staging / "source/LIS"
        expected_files = {Path(value).name for value in EXPECTED_LIS_ENTRIES[1:]}
        observed_files = {path.name for path in lis.iterdir() if path.is_file()}
        if observed_files != expected_files:
            raise ValueError(f"extracted Lisbon files changed: {observed_files}")

        network, edge_ids = _network_inventory(lis / "LIS.net.xml")
        taz, taz_ids = _taz_inventory(lis / "taz.xml")
        demand = _route_inventory(
            lis / "trips24h_smoothed.rou.xml",
            edge_ids=edge_ids,
            taz_ids=taz_ids,
        )
        native = _native_config_inventory(lis / "meso.sumo.cfg")
        additional = _write_microscopic_additional(
            lis / "additional.add.xml",
            staging / "microscopic_inputs.add.xml",
        )
        minimum_depart = float(demand["minimum_depart_sec"])
        maximum_depart = float(demand["maximum_depart_sec"])
        simulation_end = maximum_depart + DRAIN_AFTER_LAST_DEPART_SEC
        _write_microscopic_config(
            destination=staging / "microscopic.sumo.cfg",
            begin_sec=minimum_depart,
            end_sec=simulation_end,
            seed=int(seed),
        )
        gates = {
            "source_archive_identity": True,
            "complete_lisbon_archive_entries": True,
            "nonempty_unique_demand": (
                demand["trip_count"] > 0
                and demand["trip_count"] == demand["unique_trip_id_count"]
                and demand["duplicate_trip_id_count"] == 0
            ),
            "complete_trip_attributes": (
                demand["missing_required_attribute_count"] == 0
            ),
            "all_trip_edges_exist": demand["missing_edge_anchor_count"] == 0,
            "all_trip_taz_exist": demand["missing_taz_anchor_count"] == 0,
            "monotonic_departures": (
                demand["nonmonotonic_departure_count"] == 0
            ),
            "published_24h_demand_span": (
                MINIMUM_DEMAND_SPAN_SEC
                <= float(demand["depart_span_sec"])
                <= MAXIMUM_DEMAND_SPAN_SEC
            ),
            "controllable_traffic_lights_present": (
                network["traffic_light_count"] > 0
                and network["traffic_light_phase_count"] > 0
            ),
            "native_config_boundary_verified": native
            == {
                "net_file": "LIS.net.xml",
                "route_files": "trips24h_smoothed.rou.xml",
                "additional_files": "additional.add.xml,taz.xml",
                "begin_sec": "14400",
                "end_sec": "49800",
                "step_length_sec": "2",
                "mesosim": "true",
                "ignore_route_errors": "true",
                "collision_action": "none",
                "time_to_teleport_sec": "360",
                "max_depart_delay_sec": "1200",
                "random_depart_offset_sec": "1200",
                "rerouting_probability": "0.5",
                "rerouting_period_sec": "360",
                "seed": "2136",
            },
            "only_passive_output_removed": (
                additional["demand_or_network_elements_removed"] == 0
            ),
        }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "pre-simulation-input-admission",
            "scenario": SCENARIO,
            "source": {
                "doi": source_manifest["doi"],
                "archive_name": source_manifest["archive_name"],
                "archive_size_bytes": source_manifest["size_bytes"],
                "archive_md5": source_manifest["md5"],
                "license": source_manifest["license"],
                "lisbon_entries": list(EXPECTED_LIS_ENTRIES),
            },
            "input_audit_protocol": INPUT_AUDIT_PROTOCOL,
            "network_inventory": network,
            "taz_inventory": taz,
            "demand_inventory": demand,
            "native_meso_config_inventory": native,
            "microscopic_instantiation": {
                "protocol": MICRO_CONFIG_PROTOCOL,
                "config": "microscopic.sumo.cfg",
                "seed": int(seed),
                "begin_sec": int(math.floor(minimum_depart)),
                "last_depart_sec": maximum_depart,
                "drain_after_last_depart_sec": DRAIN_AFTER_LAST_DEPART_SEC,
                "end_sec": int(math.ceil(simulation_end)),
                "mode": "microscopic",
                "step_length_sec": 1,
                "time_to_teleport_sec": -1,
                "collision_check_junctions": True,
                "collision_action": "warn",
                "ignore_route_errors": False,
                "max_depart_delay_sec": -1,
                "source_random_depart_offset_sec": 1200,
                "passive_output_change": additional,
            },
            "gates": gates,
            "passed": all(gates.values()),
        }
        _write_json(staging / "package_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5057)
    args = parser.parse_args(argv)
    payload = package_remote_source(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        seed=int(args.seed),
    )
    print(json.dumps({"status": "PASS" if payload["passed"] else "REJECT"}))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
