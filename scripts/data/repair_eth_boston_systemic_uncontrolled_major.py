#!/usr/bin/env python3
"""Repair duplicate uncontrolled major states across the Boston network."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    _bit,
    _junction_indices,
    _lane_id,
)
from scripts.data.audit_sumo_uncontrolled_major_right_of_way import (
    PROTOCOL as AUDIT_PROTOCOL,
    _external_uncontrolled,
    audit_uncontrolled_major_right_of_way,
)
from scripts.data.repair_eth_boston_merge_tls import (
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_json,
)
from scripts.data.repair_eth_boston_systemic_right_of_way import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_systemic_right_of_way_manifest,
)


PROTOCOL = "cfcmt-eth-boston-systemic-uncontrolled-major-state-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v14"


def _connection_key(connection: ET.Element) -> tuple[str, ...]:
    return tuple(
        str(connection.get(name, ""))
        for name in ("from", "fromLane", "to", "toLane", "via", "tl", "linkIndex")
    )


def _repair_plan(root: ET.Element) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, Any]]:
    edges_from = {
        str(edge.get("id")): str(edge.get("from"))
        for edge in root.iter("edge")
        if edge.get("id") and edge.get("from")
    }
    connections = list(root.iter("connection"))
    indices, diagnostics = _junction_indices(
        root,
        edges_from=edges_from,
        connections=connections,
    )
    receiving: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        if _external_uncontrolled(connection):
            receiving[_lane_id(connection.attrib, "to")].append(connection)

    changes: dict[tuple[str, ...], dict[str, Any]] = {}
    candidate_groups = 0
    for receiving_lane, rows in sorted(receiving.items()):
        majors = [row for row in rows if row.get("state") == "M"]
        if len(majors) < 2:
            continue
        candidate_groups += 1
        if len(majors) != 2:
            raise ValueError(
                f"unsupported duplicate-major multiplicity at {receiving_lane}: "
                f"{len(majors)}"
            )
        junction_id = str(edges_from.get(str(rows[0].get("to", "")), ""))
        diagnostic = diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        left, right = majors
        left_index = indices.get(left)
        right_index = indices.get(right)
        if (
            not diagnostic.get("index_mapping_exact", False)
            or left_index is None
            or right_index is None
            or left_index >= len(requests)
            or right_index >= len(requests)
        ):
            raise ValueError(f"unresolved junction-index mapping at {junction_id}")
        left_request = requests[left_index]
        right_request = requests[right_index]
        symmetric_foes = bool(
            _bit(str(left_request.get("foes", "")), right_index) == "1"
            and _bit(str(right_request.get("foes", "")), left_index) == "1"
        )
        left_yields = bool(
            _bit(str(left_request.get("response", "")), right_index) == "1"
        )
        right_yields = bool(
            _bit(str(right_request.get("response", "")), left_index) == "1"
        )
        if not symmetric_foes or left_yields == right_yields:
            raise ValueError(
                f"duplicate majors lack an actionable priority relation at {junction_id}"
            )
        yielding = left if left_yields else right
        priority = right if left_yields else left
        key = _connection_key(yielding)
        if key in changes:
            raise ValueError(f"duplicate state repair key at {receiving_lane}")
        changes[key] = {
            "receiving_lane": receiving_lane,
            "junction_id": junction_id,
            "yielding_junction_index": int(indices[yielding]),
            "priority_junction_index": int(indices[priority]),
            "yielding_connection": dict(yielding.attrib),
            "priority_connection": dict(priority.attrib),
            "before": "M",
            "after": "m",
        }
    if not changes or len(changes) != candidate_groups:
        raise ValueError("Boston duplicate-major repair plan is empty or inconsistent")
    return changes, {
        "candidate_receiving_lane_count": candidate_groups,
        "connection_state_change_count": len(changes),
        "method": "existing_one_way_response_major_to_minor_state_alignment",
    }


def _network_exact_except_declared_states(
    source: Path,
    repaired: Path,
    *,
    changes: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    before = ET.iterparse(source, events=("start", "end"))
    after = ET.iterparse(repaired, events=("start", "end"))
    sentinel = object()
    observed: set[tuple[str, ...]] = set()
    element_count = 0
    while True:
        left = next(before, sentinel)
        right = next(after, sentinel)
        if left is sentinel or right is sentinel:
            exact = bool(
                left is sentinel and right is sentinel and observed == set(changes)
            )
            return {
                "network_xml_exact_except_declared_state_changes": exact,
                "element_count": element_count,
                "observed_declared_changes": len(observed),
                "expected_declared_changes": len(changes),
                "passed": exact,
            }
        left_event, left_element = left
        right_event, right_element = right
        element_count += int(left_event == "start")
        if left_event != right_event or left_element.tag != right_element.tag:
            return {"passed": False, "reason": "element_sequence_changed"}
        if left_event == "start":
            left_attributes = dict(left_element.attrib)
            right_attributes = dict(right_element.attrib)
            if left_element.tag == "connection":
                key = _connection_key(left_element)
                if key in changes:
                    if (
                        left_attributes.get("state") != "M"
                        or right_attributes.get("state") != "m"
                    ):
                        return {"passed": False, "reason": "state_change_mismatch"}
                    left_attributes["state"] = "m"
                    observed.add(key)
            if left_attributes != right_attributes:
                return {"passed": False, "reason": "undeclared_attribute_change"}
        else:
            left_element.clear()
            right_element.clear()


def repair_network(source: Path, destination: Path) -> dict[str, Any]:
    before = audit_uncontrolled_major_right_of_way(source, maximum_examples=0)
    counts = dict(before["counts"])
    if (
        before.get("protocol") != AUDIT_PROTOCOL
        or int(counts.get("duplicate_major_pair_count", -1)) <= 0
        or int(counts.get("duplicate_major_pair_count", -1))
        != int(counts.get("actionable_pair_count", -2))
        or int(counts.get("duplicate_major_pair_count", -1))
        != int(counts.get("duplicate_major_receiving_lane_count", -2))
        or int(counts.get("unresolved_junction_index_pair_count", -1)) != 0
        or int(counts.get("missing_symmetric_foe_pair_count", -1)) != 0
        or int(counts.get("ambiguous_response_pair_count", -1)) != 0
    ):
        raise ValueError("Boston duplicate-major audit is not safely actionable")

    tree = ET.parse(source)
    changes, repair = _repair_plan(tree.getroot())
    for connection in tree.getroot().iter("connection"):
        key = _connection_key(connection)
        if key in changes:
            connection.set("state", "m")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    compatibility = _network_exact_except_declared_states(
        source, destination, changes=changes
    )
    after = audit_uncontrolled_major_right_of_way(destination, maximum_examples=0)
    checks = {
        "all_pre_repair_pairs_actionable": int(counts["duplicate_major_pair_count"])
        == int(counts["actionable_pair_count"]),
        "all_duplicate_major_groups_removed": int(
            after["counts"]["duplicate_major_receiving_lane_count"]
        )
        == 0,
        "all_duplicate_major_pairs_removed": int(
            after["counts"]["duplicate_major_pair_count"]
        )
        == 0,
        "declared_state_changes_complete": int(repair["connection_state_change_count"])
        == int(counts["duplicate_major_pair_count"]),
        "network_exact_except_declared_changes": bool(compatibility["passed"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Boston duplicate-major repair failed validation: {checks}")
    return {
        "protocol": PROTOCOL,
        "pre_repair_counts": before["counts"],
        "post_repair_counts": after["counts"],
        "repair": repair,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": True,
    }


def validate_systemic_uncontrolled_major_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected Boston v14 package protocol: {manifest.get('protocol')}")
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_systemic_right_of_way_manifest(predecessor)
    section = dict(manifest.get("systemic_uncontrolled_major_repair", {}))
    before = dict(section.get("pre_repair_counts", {}))
    after = dict(section.get("post_repair_counts", {}))
    repair = dict(section.get("repair", {}))
    checks = dict(section.get("checks", {}))
    pair_count = int(before.get("duplicate_major_pair_count", -1))
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or pair_count <= 0
        or int(before.get("actionable_pair_count", -2)) != pair_count
        or int(after.get("duplicate_major_pair_count", -1)) != 0
        or int(after.get("duplicate_major_receiving_lane_count", -1)) != 0
        or int(repair.get("connection_state_change_count", -1)) != pair_count
        or dict(manifest.get("gates", {})).get(
            "boston_systemic_uncontrolled_major_states_repaired"
        )
        is not True
    ):
        raise ValueError("Boston v14 systemic uncontrolled-major manifest is incomplete")


def repair_package(
    *, base_root: Path, output_root: Path, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v13 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_systemic_right_of_way_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v13 base package is not admitted for remediation")
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
        section = repair_network(source_network, temporary_network)
        os.replace(temporary_network, derived_network)
        hardlinks_exact, unchanged_file_count = _unchanged_files_share_base_inodes(
            base_root, staging
        )
        checks = dict(section["checks"])
        checks["base_package_passed"] = base_manifest.get("passed") is True
        checks["all_other_package_files_reused_exactly"] = hardlinks_exact
        section.update(
            {
                "unchanged_file_count": unchanged_file_count,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
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
                "systemic_uncontrolled_major_repair": section,
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_systemic_uncontrolled_major_states_repaired"] = bool(
            section["passed"]
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_systemic_uncontrolled_major_state_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_systemic_uncontrolled_major_manifest(payload)
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
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    payload = repair_package(
        base_root=args.base_root,
        output_root=args.output_root,
        expected_base_manifest_sha256=args.expected_base_manifest_sha256,
    )
    if args.out is not None:
        _write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS" if payload["passed"] else "FAIL",
                "protocol": payload["protocol"],
                "package_root": str(args.output_root),
                "counts": payload["systemic_uncontrolled_major_repair"][
                    "post_repair_counts"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
