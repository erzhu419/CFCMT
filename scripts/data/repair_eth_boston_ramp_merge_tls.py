#!/usr/bin/env python3
"""Apply the declared Boston signalized ramp-to-mainline yield repair."""

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
from scripts.data.repair_eth_boston_uncontrolled_merge import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_uncontrolled_merge_manifest,
)


PROTOCOL = "cfcmt-eth-boston-signalized-ramp-merge-yield-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v9"
TLS_ID = "joinedS_777"
JUNCTION_ID = "cluster_73110325_73135282"
MERGE_EDGE_ID = "618556460#8"
DECLARED_CHANGES: dict[tuple[int, int], tuple[str, str]] = {
    (2, 8): ("G", "g"),
}
RAMP_CONNECTION = {
    "from": "9504239#0",
    "fromLane": "0",
    "to": MERGE_EDGE_ID,
    "toLane": "0",
    "tl": TLS_ID,
    "linkIndex": "8",
    "state": "O",
}
MAINLINE_CONNECTIONS = (
    {
        "from": "618556460#7",
        "fromLane": "0",
        "to": MERGE_EDGE_ID,
        "toLane": "0",
        "uncontrolled": "1",
        "state": "M",
    },
    {
        "from": "618556460#7",
        "fromLane": "1",
        "to": MERGE_EDGE_ID,
        "toLane": "1",
        "uncontrolled": "1",
        "state": "M",
    },
)


def _matches(attributes: Mapping[str, str], identity: Mapping[str, str]) -> bool:
    return all(attributes.get(key) == value for key, value in identity.items())


def _validate_declared_merge_topology(network: Path) -> dict[str, Any]:
    root = ET.parse(network).getroot()
    edges = [element for element in root.iter("edge") if element.get("id") == MERGE_EDGE_ID]
    if len(edges) != 1 or edges[0].get("from") != JUNCTION_ID:
        raise ValueError("declared Boston ramp merge edge topology changed")
    lane_indices = sorted(lane.get("index") for lane in edges[0].findall("lane"))
    if lane_indices != ["0", "1"]:
        raise ValueError("declared Boston ramp merge edge lanes changed")

    connections = list(root.iter("connection"))
    ramp_matches = [row for row in connections if _matches(row.attrib, RAMP_CONNECTION)]
    if len(ramp_matches) != 1:
        raise ValueError("declared Boston signalized ramp connection changed")
    mainline_counts = [
        sum(_matches(row.attrib, identity) for row in connections)
        for identity in MAINLINE_CONNECTIONS
    ]
    if mainline_counts != [1, 1]:
        raise ValueError("declared Boston uncontrolled mainline connections changed")

    junctions = [
        element for element in root.iter("junction") if element.get("id") == JUNCTION_ID
    ]
    if len(junctions) != 1 or junctions[0].get("type") != "traffic_light_right_on_red":
        raise ValueError("declared Boston ramp merge junction changed")
    return {
        "junction_id": JUNCTION_ID,
        "merge_edge_id": MERGE_EDGE_ID,
        "merge_edge_lane_indices": lane_indices,
        "ramp_connection": dict(RAMP_CONNECTION),
        "uncontrolled_mainline_connections": [
            dict(identity) for identity in MAINLINE_CONNECTIONS
        ],
    }


def validate_ramp_merge_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(
            f"unexpected Boston v9 package protocol: {manifest.get('protocol')}"
        )
    predecessor_view = dict(manifest)
    predecessor_view["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_uncontrolled_merge_manifest(predecessor_view)
    topology = dict(manifest.get("ramp_merge_tls_repair", {}))
    repair = dict(topology.get("repair", {}))
    compatibility = dict(topology.get("compatibility_audit", {}))
    checks = dict(topology.get("checks", {}))
    expected_changes = [
        {
            "phase_index": phase_index,
            "link_index": link_index,
            "before": before,
            "after": after,
        }
        for (phase_index, link_index), (before, after) in DECLARED_CHANGES.items()
    ]
    evidence = dict(topology.get("merge_topology", {}))
    if (
        topology.get("protocol") != PROTOCOL
        or topology.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or compatibility.get("passed") is not True
        or int(compatibility.get("observed_declared_changes", -1)) != 1
        or int(compatibility.get("expected_declared_changes", -1)) != 1
        or repair.get("tls_id") != TLS_ID
        or repair.get("changes") != expected_changes
        or evidence.get("junction_id") != JUNCTION_ID
        or evidence.get("merge_edge_id") != MERGE_EDGE_ID
        or evidence.get("ramp_connection") != RAMP_CONNECTION
        or dict(manifest.get("gates", {})).get(
            "boston_signalized_ramp_merge_yield_repaired"
        )
        is not True
    ):
        raise ValueError("Boston signalized ramp merge repair manifest is incomplete")


def repair_package(
    *,
    base_root: Path,
    output_root: Path,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v8 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_uncontrolled_merge_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v8 base package is not admitted for remediation")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")

    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    try:
        shutil.copytree(base_root, staging, copy_function=os.link)
        source_network = base_root / "microscopic_network.net.xml"
        merge_topology = _validate_declared_merge_topology(source_network)
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
        checks = {
            "base_package_passed": base_manifest.get("passed") is True,
            "declared_tls_change_exact": compatibility["passed"],
            "declared_merge_topology_exact": bool(merge_topology),
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
                "ramp_merge_tls_repair": {
                    "protocol": PROTOCOL,
                    "repair": repair,
                    "merge_topology": merge_topology,
                    "compatibility_audit": compatibility,
                    "unchanged_file_count": unchanged_file_count,
                    "checks": checks,
                    "passed": all(checks.values()),
                },
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_signalized_ramp_merge_yield_repaired"] = all(checks.values())
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_declared_signalized_ramp_merge_yield_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_ramp_merge_manifest(payload)
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
    print(json.dumps({"status": "PASS" if payload["passed"] else "REJECT"}))
    print(json.dumps(payload["ramp_merge_tls_repair"], indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
