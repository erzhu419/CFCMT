#!/usr/bin/env python3
"""Repair implicit no-TLS duplicate-major states in the Boston v15 package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    IMPLICIT_NO_TLS_PROTOCOL as AUDIT_PROTOCOL,
    _external_no_tls,
    audit_uncontrolled_major_right_of_way,
)
from scripts.data.repair_eth_boston_merge_tls import (
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_json,
)
from scripts.data.repair_eth_boston_systemic_uncontrolled_major import (
    _connection_key,
    _network_exact_except_declared_states,
)
from scripts.data.repair_eth_boston_tls_yellow_clearance import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_tls_yellow_clearance_manifest,
)


PROTOCOL = "cfcmt-eth-boston-implicit-no-tls-major-state-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v16"
KNOWN_COLLISION_JUNCTION = "3185278424"
KNOWN_COLLISION_YIELDING_CONNECTIONS = {
    ("312661407", "0", "-8922897#4", "0"),
    ("312661407", "1", "-8922897#4", "1"),
}


def _short_connection_key(connection: ET.Element) -> tuple[str, str, str, str]:
    return tuple(
        str(connection.get(name, ""))
        for name in ("from", "fromLane", "to", "toLane")
    )


def _repair_plan(
    root: ET.Element,
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, Any]]:
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
        if _external_no_tls(connection):
            receiving[_lane_id(connection.attrib, "to")].append(connection)

    changes: dict[tuple[str, ...], dict[str, Any]] = {}
    multiplicity: Counter[int] = Counter()
    candidate_groups = 0
    pair_count = 0
    for receiving_lane, rows in sorted(receiving.items()):
        majors = [row for row in rows if row.get("state") == "M"]
        if len(majors) < 2:
            continue
        candidate_groups += 1
        multiplicity[len(majors)] += 1
        junction_id = str(edges_from.get(str(rows[0].get("to", "")), ""))
        diagnostic = diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        if not diagnostic.get("index_mapping_exact", False):
            raise ValueError(f"unresolved junction-index mapping at {junction_id}")

        yields_to_peer = {row: 0 for row in majors}
        for left_offset, left in enumerate(majors):
            for right in majors[left_offset + 1 :]:
                pair_count += 1
                left_index = indices.get(left)
                right_index = indices.get(right)
                if (
                    left_index is None
                    or right_index is None
                    or left_index >= len(requests)
                    or right_index >= len(requests)
                ):
                    raise ValueError(
                        f"unresolved major-pair index at {junction_id}"
                    )
                left_request = requests[left_index]
                right_request = requests[right_index]
                symmetric_foes = bool(
                    _bit(str(left_request.get("foes", "")), right_index) == "1"
                    and _bit(str(right_request.get("foes", "")), left_index)
                    == "1"
                )
                left_yields = bool(
                    _bit(str(left_request.get("response", "")), right_index)
                    == "1"
                )
                right_yields = bool(
                    _bit(str(right_request.get("response", "")), left_index)
                    == "1"
                )
                if not symmetric_foes or left_yields == right_yields:
                    raise ValueError(
                        "implicit duplicate majors lack an actionable priority "
                        f"relation at {junction_id}"
                    )
                yields_to_peer[left if left_yields else right] += 1

        priority = [row for row, count in yields_to_peer.items() if count == 0]
        if len(priority) != 1:
            raise ValueError(
                f"implicit duplicate-major group lacks one priority movement at "
                f"{receiving_lane}"
            )
        priority_row = priority[0]
        for yielding in majors:
            if yielding is priority_row:
                continue
            key = _connection_key(yielding)
            if key in changes:
                raise ValueError(f"duplicate state repair key at {receiving_lane}")
            changes[key] = {
                "receiving_lane": receiving_lane,
                "junction_id": junction_id,
                "yielding_junction_index": int(indices[yielding]),
                "priority_junction_index": int(indices[priority_row]),
                "yielding_connection": dict(yielding.attrib),
                "priority_connection": dict(priority_row.attrib),
                "before": "M",
                "after": "m",
            }

    known_changes = {
        _short_connection_key(row)
        for row in root.iter("connection")
        if _connection_key(row) in changes
        and changes[_connection_key(row)]["junction_id"]
        == KNOWN_COLLISION_JUNCTION
    }
    if known_changes != KNOWN_COLLISION_YIELDING_CONNECTIONS:
        raise ValueError(
            "Boston v15 collision movements are not exactly covered by the repair"
        )
    if not changes:
        raise ValueError("Boston implicit no-TLS repair plan is empty")
    return changes, {
        "candidate_receiving_lane_count": candidate_groups,
        "candidate_pair_count": pair_count,
        "major_multiplicity_histogram": {
            str(key): int(value) for key, value in sorted(multiplicity.items())
        },
        "connection_state_change_count": len(changes),
        "known_collision_junction": KNOWN_COLLISION_JUNCTION,
        "known_collision_yielding_connections": [
            list(value) for value in sorted(known_changes)
        ],
        "method": (
            "existing_groupwise_request_response_priority_major_to_minor_"
            "state_alignment"
        ),
        "change_examples": list(changes.values())[:20],
    }


def repair_network(source: Path, destination: Path) -> dict[str, Any]:
    before = audit_uncontrolled_major_right_of_way(
        source,
        maximum_examples=0,
        include_implicit_no_tls=True,
    )
    counts = dict(before["counts"])
    pair_count = int(counts.get("duplicate_major_pair_count", -1))
    group_count = int(counts.get("duplicate_major_receiving_lane_count", -1))
    planned_changes = int(counts.get("planned_connection_state_change_count", -1))
    if (
        before.get("protocol") != AUDIT_PROTOCOL
        or pair_count <= 0
        or group_count <= 0
        or pair_count != int(counts.get("actionable_pair_count", -2))
        or pair_count
        != int(counts.get("pair_with_implicit_uncontrolled_attribute_count", -2))
        or group_count != int(counts.get("unique_priority_group_count", -2))
        or int(counts.get("ambiguous_priority_group_count", -1)) != 0
        or int(counts.get("unresolved_junction_index_pair_count", -1)) != 0
        or int(counts.get("missing_symmetric_foe_pair_count", -1)) != 0
        or int(counts.get("ambiguous_response_pair_count", -1)) != 0
        or planned_changes <= 0
    ):
        raise ValueError("Boston implicit no-TLS major audit is not safely actionable")

    tree = ET.parse(source)
    changes, repair = _repair_plan(tree.getroot())
    if len(changes) != planned_changes:
        raise ValueError("repair plan differs from the read-only audit")
    for connection in tree.getroot().iter("connection"):
        key = _connection_key(connection)
        if key in changes:
            connection.set("state", "m")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    compatibility = _network_exact_except_declared_states(
        source,
        destination,
        changes=changes,
    )
    after = audit_uncontrolled_major_right_of_way(
        destination,
        maximum_examples=0,
        include_implicit_no_tls=True,
    )
    checks = {
        "all_pre_repair_pairs_actionable": pair_count
        == int(counts["actionable_pair_count"]),
        "all_pre_repair_groups_have_unique_priority": group_count
        == int(counts["unique_priority_group_count"]),
        "all_duplicate_major_groups_removed": int(
            after["counts"]["duplicate_major_receiving_lane_count"]
        )
        == 0,
        "all_duplicate_major_pairs_removed": int(
            after["counts"]["duplicate_major_pair_count"]
        )
        == 0,
        "declared_state_changes_complete": len(changes) == planned_changes,
        "known_collision_movements_repaired": len(
            repair["known_collision_yielding_connections"]
        )
        == len(KNOWN_COLLISION_YIELDING_CONNECTIONS),
        "network_exact_except_declared_changes": bool(compatibility["passed"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Boston implicit no-TLS repair failed validation: {checks}")
    return {
        "protocol": PROTOCOL,
        "pre_repair_counts": before["counts"],
        "post_repair_counts": after["counts"],
        "repair": repair,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": True,
    }


def validate_implicit_no_tls_major_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(
            f"unexpected Boston v16 package protocol: {manifest.get('protocol')}"
        )
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_tls_yellow_clearance_manifest(predecessor)
    section = dict(manifest.get("implicit_no_tls_major_repair", {}))
    pre = dict(section.get("pre_repair_counts", {}))
    post = dict(section.get("post_repair_counts", {}))
    repair = dict(section.get("repair", {}))
    checks = dict(section.get("checks", {}))
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or int(pre.get("duplicate_major_pair_count", 0)) <= 0
        or int(pre.get("ambiguous_priority_group_count", -1)) != 0
        or int(post.get("duplicate_major_pair_count", -1)) != 0
        or int(post.get("duplicate_major_receiving_lane_count", -1)) != 0
        or int(repair.get("connection_state_change_count", -1))
        != int(pre.get("planned_connection_state_change_count", -2))
        or dict(manifest.get("gates", {})).get(
            "boston_implicit_no_tls_major_states_repaired"
        )
        is not True
    ):
        raise ValueError("Boston v16 implicit no-TLS manifest is incomplete")


def repair_package(
    *, base_root: Path, output_root: Path, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v15 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_tls_yellow_clearance_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v15 base package is not admitted for remediation")
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
                "implicit_no_tls_major_repair": section,
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_implicit_no_tls_major_states_repaired"] = bool(
            section["passed"]
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_implicit_no_tls_groupwise_major_state_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_implicit_no_tls_major_manifest(payload)
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
    section = dict(payload["implicit_no_tls_major_repair"])
    print(
        json.dumps(
            {
                "status": "PASS" if payload["passed"] else "FAIL",
                "protocol": payload["protocol"],
                "package_root": str(args.output_root),
                "pre_repair_counts": section["pre_repair_counts"],
                "post_repair_counts": section["post_repair_counts"],
                "repair": section["repair"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
