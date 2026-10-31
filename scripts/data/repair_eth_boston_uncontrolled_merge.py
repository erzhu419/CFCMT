#!/usr/bin/env python3
"""Apply the declared Boston uncontrolled two-to-one merge yield repair."""

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
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_json,
    validate_remediated_package_manifest,
)


PROTOCOL = "cfcmt-eth-boston-uncontrolled-two-to-one-merge-yield-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v8"
JUNCTION_ID = "71937638"
CONNECTION_IDENTITY = {
    "from": "9429279#2",
    "fromLane": "0",
    "to": "185856197#0",
    "toLane": "0",
    "uncontrolled": "1",
}
BEFORE_STATE = "M"
AFTER_STATE = "m"


def _connection_matches(attributes: Mapping[str, str]) -> bool:
    return all(attributes.get(key) == value for key, value in CONNECTION_IDENTITY.items())


def validate_uncontrolled_merge_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(
            f"unexpected Boston v8 package protocol: {manifest.get('protocol')}"
        )
    predecessor_view = dict(manifest)
    predecessor_view["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_remediated_package_manifest(predecessor_view)
    topology = dict(manifest.get("uncontrolled_merge_repair", {}))
    repair = dict(topology.get("repair", {}))
    compatibility = dict(topology.get("compatibility_audit", {}))
    checks = dict(topology.get("checks", {}))
    if (
        topology.get("protocol") != PROTOCOL
        or topology.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or compatibility.get("passed") is not True
        or int(compatibility.get("observed_declared_changes", -1)) != 1
        or int(compatibility.get("expected_declared_changes", -1)) != 1
        or repair.get("junction_id") != JUNCTION_ID
        or repair.get("connection") != CONNECTION_IDENTITY
        or repair.get("before") != BEFORE_STATE
        or repair.get("after") != AFTER_STATE
        or dict(manifest.get("gates", {})).get(
            "boston_uncontrolled_two_to_one_merge_yield_repaired"
        )
        is not True
    ):
        raise ValueError("Boston uncontrolled merge repair manifest is incomplete")


def _write_declared_connection_repair(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    tree = ET.parse(source)
    matches = [
        element
        for element in tree.getroot().iter("connection")
        if _connection_matches(element.attrib)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one declared uncontrolled merge connection, "
            f"found {len(matches)}"
        )
    connection = matches[0]
    observed = connection.attrib.get("state")
    if observed != BEFORE_STATE:
        raise ValueError(
            "declared uncontrolled merge state changed before repair: "
            f"expected={BEFORE_STATE!r} actual={observed!r}"
        )
    connection.set("state", AFTER_STATE)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return {
        "junction_id": JUNCTION_ID,
        "connection": dict(CONNECTION_IDENTITY),
        "before": BEFORE_STATE,
        "after": AFTER_STATE,
        "method": "declared_uncontrolled_side_branch_major_to_minor_yield",
    }


def _network_exact_except_declared_connection(
    source: Path,
    repaired: Path,
) -> dict[str, Any]:
    before = ET.iterparse(source, events=("start", "end"))
    after = ET.iterparse(repaired, events=("start", "end"))
    sentinel = object()
    observed = 0
    while True:
        left = next(before, sentinel)
        right = next(after, sentinel)
        if left is sentinel or right is sentinel:
            exact = left is sentinel and right is sentinel and observed == 1
            return {
                "network_xml_exact_except_declared_connection": exact,
                "observed_declared_changes": observed,
                "expected_declared_changes": 1,
                "passed": exact,
            }
        left_event, left_element = left
        right_event, right_element = right
        if left_event != right_event or left_element.tag != right_element.tag:
            return {
                "network_xml_exact_except_declared_connection": False,
                "observed_declared_changes": observed,
                "expected_declared_changes": 1,
                "passed": False,
            }
        if left_event == "start":
            left_attributes = dict(left_element.attrib)
            right_attributes = dict(right_element.attrib)
            if left_element.tag == "connection" and _connection_matches(
                left_attributes
            ):
                if (
                    left_attributes.get("state") != BEFORE_STATE
                    or right_attributes.get("state") != AFTER_STATE
                ):
                    return {
                        "network_xml_exact_except_declared_connection": False,
                        "observed_declared_changes": observed,
                        "expected_declared_changes": 1,
                        "passed": False,
                    }
                left_attributes["state"] = AFTER_STATE
                observed += 1
            if left_attributes != right_attributes:
                return {
                    "network_xml_exact_except_declared_connection": False,
                    "observed_declared_changes": observed,
                    "expected_declared_changes": 1,
                    "passed": False,
                }
        else:
            left_element.clear()
            right_element.clear()


def repair_package(
    *,
    base_root: Path,
    output_root: Path,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v7 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_remediated_package_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v7 base package is not admitted for remediation")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite package: {output_root}")

    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path exists: {staging}")
    try:
        shutil.copytree(base_root, staging, copy_function=os.link)
        source_network = base_root / "microscopic_network.net.xml"
        derived_network = staging / "microscopic_network.net.xml"
        temporary_network = staging / ".microscopic_network.net.xml.tmp"
        repair = _write_declared_connection_repair(
            source_network,
            temporary_network,
        )
        os.replace(temporary_network, derived_network)
        compatibility = _network_exact_except_declared_connection(
            source_network,
            derived_network,
        )
        hardlinks_exact, unchanged_file_count = _unchanged_files_share_base_inodes(
            base_root,
            staging,
        )
        checks = {
            "base_package_passed": base_manifest.get("passed") is True,
            "declared_connection_change_exact": compatibility["passed"],
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
                "uncontrolled_merge_repair": {
                    "protocol": PROTOCOL,
                    "repair": repair,
                    "compatibility_audit": compatibility,
                    "unchanged_file_count": unchanged_file_count,
                    "checks": checks,
                    "passed": all(checks.values()),
                },
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_uncontrolled_two_to_one_merge_yield_repaired"] = all(
            checks.values()
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_declared_uncontrolled_two_to_one_merge_yield_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_uncontrolled_merge_manifest(payload)
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
    print(json.dumps(payload["uncontrolled_merge_repair"], indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
