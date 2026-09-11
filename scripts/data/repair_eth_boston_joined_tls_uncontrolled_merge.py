#!/usr/bin/env python3
"""Yield one Boston joined-TLS movement across an uncontrolled merge."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from scripts.data.repair_eth_boston_merge_tls import (
    _network_exact_except_declared_tls_changes,
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_declared_tls_yield_repair,
    _write_json,
)
from scripts.data.repair_eth_boston_protected_uncontrolled_merge_tls import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_protected_uncontrolled_merge_manifest,
)


PROTOCOL = "cfcmt-eth-boston-joined-tls-uncontrolled-merge-yield-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v12"
TLS_ID = "joinedS_514"
JUNCTION_ID = "61400161"
MERGE_EDGE_ID = "8647416#3"
MERGE_LANE_ID = "8647416#3_0"
DECLARED_CHANGES: dict[tuple[int, int], tuple[str, str]] = {
    (0, 0): ("G", "g"),
    (2, 0): ("G", "g"),
    (6, 0): ("G", "g"),
}
EXPECTED_LINK_SERVICE_BEFORE = "GyGyrrGyrr"
EXPECTED_LINK_SERVICE_AFTER = "gygyrrgyrr"
CONTROLLED_STRAIGHT_CONNECTION = {
    "from": "128013681#0",
    "fromLane": "0",
    "to": MERGE_EDGE_ID,
    "toLane": "0",
    "tl": TLS_ID,
    "linkIndex": "0",
    "state": "O",
}
UNCONTROLLED_STRAIGHT_CONNECTION = {
    "from": "8647416#1",
    "fromLane": "0",
    "to": MERGE_EDGE_ID,
    "toLane": "0",
    "state": "M",
}
EXPECTED_REQUESTS = (
    {"index": "0", "response": "00", "foes": "10"},
    {"index": "1", "response": "01", "foes": "01"},
)


def _matches(attributes: Mapping[str, str], identity: Mapping[str, str]) -> bool:
    return all(attributes.get(key) == value for key, value in identity.items())


def _validate_declared_merge_topology(network: Path) -> dict[str, Any]:
    root = ET.parse(network).getroot()
    programs = [row for row in root.iter("tlLogic") if row.get("id") == TLS_ID]
    if len(programs) != 1:
        raise ValueError("declared Boston joined TLS changed")
    phases = programs[0].findall("phase")
    if len(phases) != len(EXPECTED_LINK_SERVICE_BEFORE):
        raise ValueError("declared Boston joined TLS phases changed")
    service_before = "".join(str(phase.get("state", ""))[0] for phase in phases)
    if service_before != EXPECTED_LINK_SERVICE_BEFORE:
        raise ValueError("declared Boston joined TLS service changed")

    edges = [row for row in root.iter("edge") if row.get("id") == MERGE_EDGE_ID]
    if len(edges) != 1 or edges[0].get("from") != JUNCTION_ID:
        raise ValueError("declared Boston joined-TLS merge edge changed")
    lanes = [row for row in edges[0].findall("lane") if row.get("id") == MERGE_LANE_ID]
    if len(lanes) != 1 or lanes[0].get("index") != "0":
        raise ValueError("declared Boston joined-TLS merge lane changed")

    connections = list(root.iter("connection"))
    for identity in (CONTROLLED_STRAIGHT_CONNECTION, UNCONTROLLED_STRAIGHT_CONNECTION):
        if sum(_matches(row.attrib, identity) for row in connections) != 1:
            raise ValueError("declared Boston joined-TLS connection changed")

    junctions = [row for row in root.iter("junction") if row.get("id") == JUNCTION_ID]
    if (
        len(junctions) != 1
        or junctions[0].get("type") != "traffic_light"
        or str(junctions[0].get("intLanes", "")) != ""
    ):
        raise ValueError("declared Boston joined-TLS junction changed")
    requests = [dict(row.attrib) for row in junctions[0].findall("request")]
    if requests != list(EXPECTED_REQUESTS):
        raise ValueError("declared Boston joined-TLS foe relation changed")

    return {
        "tls_id": TLS_ID,
        "junction_id": JUNCTION_ID,
        "merge_edge_id": MERGE_EDGE_ID,
        "merge_lane_id": MERGE_LANE_ID,
        "controlled_straight_connection": dict(CONTROLLED_STRAIGHT_CONNECTION),
        "uncontrolled_straight_connection": dict(UNCONTROLLED_STRAIGHT_CONNECTION),
        "junction_requests": requests,
        "link_service_before": service_before,
        "collision_evidence": {
            "time_sec": 13146,
            "phase_index": 6,
            "traffic_light_state": str(phases[6].get("state", "")),
            "uncontrolled_vehicle": "77188",
            "controlled_vehicle": "95307",
            "lane_id": MERGE_LANE_ID,
            "position_m": 2.628,
        },
    }


def validate_joined_tls_uncontrolled_merge_manifest(
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected Boston v12 package protocol: {manifest.get('protocol')}")
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_protected_uncontrolled_merge_manifest(predecessor)
    section = dict(manifest.get("joined_tls_uncontrolled_merge_repair", {}))
    repair = dict(section.get("repair", {}))
    compatibility = dict(section.get("compatibility_audit", {}))
    checks = dict(section.get("checks", {}))
    topology = dict(section.get("merge_topology", {}))
    expected_changes = [
        {
            "phase_index": phase_index,
            "link_index": link_index,
            "before": before,
            "after": after,
        }
        for (phase_index, link_index), (before, after) in DECLARED_CHANGES.items()
    ]
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or compatibility.get("passed") is not True
        or int(compatibility.get("observed_declared_changes", -1)) != len(DECLARED_CHANGES)
        or int(compatibility.get("expected_declared_changes", -1)) != len(DECLARED_CHANGES)
        or repair.get("tls_id") != TLS_ID
        or repair.get("changes") != expected_changes
        or topology.get("merge_lane_id") != MERGE_LANE_ID
        or topology.get("controlled_straight_connection")
        != CONTROLLED_STRAIGHT_CONNECTION
        or topology.get("uncontrolled_straight_connection")
        != UNCONTROLLED_STRAIGHT_CONNECTION
        or topology.get("junction_requests") != list(EXPECTED_REQUESTS)
        or dict(manifest.get("gates", {})).get(
            "boston_joined_tls_uncontrolled_merge_yield_repaired"
        )
        is not True
    ):
        raise ValueError("Boston joined-TLS uncontrolled merge repair manifest is incomplete")


def repair_package(
    *,
    base_root: Path,
    output_root: Path,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v11 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_protected_uncontrolled_merge_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v11 base package is not admitted for remediation")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")

    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    try:
        shutil.copytree(base_root, staging, copy_function=os.link)
        source_network = base_root / "microscopic_network.net.xml"
        topology = _validate_declared_merge_topology(source_network)
        derived_network = staging / "microscopic_network.net.xml"
        temporary_network = staging / ".microscopic_network.net.xml.tmp"
        repair = _write_declared_tls_yield_repair(
            source_network,
            temporary_network,
            tls_id=TLS_ID,
            declared_changes=DECLARED_CHANGES,
        )
        os.replace(temporary_network, derived_network)
        compatibility = _network_exact_except_declared_tls_changes(
            source_network,
            derived_network,
            tls_id=TLS_ID,
            declared_changes=DECLARED_CHANGES,
        )
        hardlinks_exact, unchanged_file_count = _unchanged_files_share_base_inodes(
            base_root,
            staging,
        )
        repaired_program = next(
            row
            for row in ET.parse(derived_network).getroot().iter("tlLogic")
            if row.get("id") == TLS_ID
        )
        service_after = "".join(
            str(phase.get("state", ""))[0]
            for phase in repaired_program.findall("phase")
        )
        checks = {
            "base_package_passed": base_manifest.get("passed") is True,
            "declared_tls_changes_exact": compatibility["passed"],
            "declared_merge_topology_exact": bool(topology),
            "all_active_service_is_yielding": service_after
            == EXPECTED_LINK_SERVICE_AFTER,
            "yellow_clearance_retained": all(
                service_after[index] == "y" for index in (1, 3, 7)
            ),
            "all_other_package_files_reused_exactly": hardlinks_exact,
        }
        payload = dict(base_manifest)
        payload.update(
            {
                "protocol": PACKAGE_PROTOCOL,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_status": "pre-simulation-input-admission",
                "derived_from": {
                    "package_root": str(base_root),
                    "package_manifest_sha256": observed_sha256,
                },
                "joined_tls_uncontrolled_merge_repair": {
                    "protocol": PROTOCOL,
                    "repair": repair,
                    "merge_topology": topology,
                    "link_service_after": service_after,
                    "compatibility_audit": compatibility,
                    "unchanged_file_count": unchanged_file_count,
                    "checks": checks,
                    "passed": all(checks.values()),
                },
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_joined_tls_uncontrolled_merge_yield_repaired"] = all(
            checks.values()
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_joined_tls_uncontrolled_merge_yield_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_joined_tls_uncontrolled_merge_manifest(payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    payload = repair_package(
        base_root=args.base_root,
        output_root=args.output_root,
        expected_base_manifest_sha256=args.expected_base_manifest_sha256,
    )
    section = payload["joined_tls_uncontrolled_merge_repair"]
    print(json.dumps({"status": "PASS" if payload["passed"] else "REJECT"}))
    print(json.dumps(section, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
