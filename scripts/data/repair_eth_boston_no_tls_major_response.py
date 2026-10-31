#!/usr/bin/env python3
"""Align Boston no-TLS major states with declared no-TLS yield relations."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_no_tls_major_response import (
    PROTOCOL as AUDIT_PROTOCOL,
    _set_bits,
    audit_no_tls_major_response_consistency,
)
from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    _bit,
    _junction_indices,
    _lane_id,
)
from scripts.data.audit_sumo_uncontrolled_major_right_of_way import (
    _external_no_tls,
)
from scripts.data.repair_eth_boston_implicit_no_tls_major import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_implicit_no_tls_major_manifest,
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


PROTOCOL = "cfcmt-eth-boston-no-tls-major-response-state-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v17"
KNOWN_COLLISION_JUNCTION = (
    "cluster_3039940654_4045163181_4045163186_4045164696_"
    "4871137848_4871137850_61359520_7044708085_7816490861"
)
KNOWN_COLLISION_CONNECTION = (
    "-656820076#1",
    "0",
    "-8638955#0",
    "1",
)


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
    by_junction_index: dict[tuple[str, int], ET.Element] = {}
    for connection, index in indices.items():
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        by_junction_index[(junction_id, int(index))] = connection

    changes: dict[tuple[str, ...], dict[str, Any]] = {}
    relation_count = 0
    same_receiving_lane_relation_count = 0
    target_states: Counter[str] = Counter()
    junction_types: Counter[str] = Counter()
    for connection in connections:
        if not _external_no_tls(connection) or connection.get("state") != "M":
            continue
        junction_id = str(edges_from.get(str(connection.get("to", "")), ""))
        diagnostic = diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        index = indices.get(connection)
        if (
            not diagnostic.get("index_mapping_exact", False)
            or index is None
            or int(index) >= len(requests)
        ):
            raise ValueError(f"unresolved junction-index mapping at {junction_id}")
        request = requests[int(index)]
        relations: list[dict[str, Any]] = []
        receiving_lane = _lane_id(connection.attrib, "to")
        for response_index in _set_bits(str(request.get("response", ""))):
            target = by_junction_index.get((junction_id, response_index))
            if target is None or not _external_no_tls(target):
                continue
            if response_index >= len(requests):
                raise ValueError(f"unresolved response target at {junction_id}")
            target_request = requests[response_index]
            symmetric_foes = bool(
                _bit(str(request.get("foes", "")), response_index) == "1"
                and _bit(str(target_request.get("foes", "")), int(index)) == "1"
            )
            target_responds_back = bool(
                _bit(str(target_request.get("response", "")), int(index)) == "1"
            )
            if not symmetric_foes or target_responds_back:
                raise ValueError(
                    "no-TLS major response lacks a unique declared priority "
                    f"relation at {junction_id}"
                )
            same_receiving_lane = (
                _lane_id(target.attrib, "to") == receiving_lane
            )
            relation_count += 1
            same_receiving_lane_relation_count += int(same_receiving_lane)
            target_states[str(target.get("state", ""))] += 1
            relations.append(
                {
                    "priority_junction_index": int(response_index),
                    "priority_connection": dict(target.attrib),
                    "same_receiving_lane": same_receiving_lane,
                }
            )
        if not relations:
            continue
        key = _connection_key(connection)
        if key in changes:
            raise ValueError(f"duplicate response-state repair at {junction_id}")
        junction_types[str(diagnostic.get("junction_type", ""))] += 1
        changes[key] = {
            "junction_id": junction_id,
            "yielding_junction_index": int(index),
            "yielding_connection": dict(connection.attrib),
            "declared_no_tls_priority_relations": relations,
            "before": "M",
            "after": "m",
        }

    known = [
        value
        for value in changes.values()
        if value["junction_id"] == KNOWN_COLLISION_JUNCTION
        and _short_connection_key(
            ET.Element("connection", value["yielding_connection"])
        )
        == KNOWN_COLLISION_CONNECTION
    ]
    if len(known) != 1:
        raise ValueError("Boston v16 collision movement is not exactly covered")
    if not changes:
        raise ValueError("Boston no-TLS major-response repair plan is empty")
    examples = list(changes.values())[:20]
    if known[0] not in examples:
        examples.append(known[0])
    return changes, {
        "connection_state_change_count": len(changes),
        "declared_no_tls_priority_relation_count": relation_count,
        "same_receiving_lane_relation_count": (
            same_receiving_lane_relation_count
        ),
        "priority_target_state_histogram": {
            key: int(value) for key, value in sorted(target_states.items())
        },
        "junction_type_histogram": {
            key: int(value) for key, value in sorted(junction_types.items())
        },
        "known_collision_junction": KNOWN_COLLISION_JUNCTION,
        "known_collision_connection": list(KNOWN_COLLISION_CONNECTION),
        "known_collision_change": known[0],
        "method": (
            "existing_request_response_asymmetric_no_tls_priority_"
            "major_to_minor_state_alignment"
        ),
        "change_examples": examples,
    }


def repair_network(source: Path, destination: Path) -> dict[str, Any]:
    before = audit_no_tls_major_response_consistency(source, maximum_examples=0)
    counts = dict(before["counts"])
    planned_changes = int(
        counts.get("candidate_with_external_no_tls_response_count", -1)
    )
    if (
        before.get("protocol") != AUDIT_PROTOCOL
        or planned_changes <= 0
        or planned_changes
        != int(
            counts.get(
                "candidate_with_asymmetric_external_no_tls_response_count", -2
            )
        )
        or int(
            counts.get("candidate_with_mutual_external_no_tls_response_count", -1)
        )
        != 0
        or int(
            counts.get(
                "external_no_tls_response_without_symmetric_foe_bit_count", -1
            )
        )
        != 0
        or int(counts.get("response_without_foe_bit_count", -1)) != 0
        or int(counts.get("unresolved_major_connection_count", -1)) != 0
        or any(
            key not in {"external_no_tls", "tls_controlled"}
            for key in dict(counts.get("response_target_kind_histogram", {}))
        )
    ):
        raise ValueError(
            "Boston no-TLS major-response audit is not safely actionable"
        )

    tree = ET.parse(source)
    changes, repair = _repair_plan(tree.getroot())
    if len(changes) != planned_changes:
        raise ValueError("repair plan differs from the read-only audit")
    for connection in tree.getroot().iter("connection"):
        if _connection_key(connection) in changes:
            connection.set("state", "m")
    tree.write(destination, encoding="utf-8", xml_declaration=True)

    compatibility = _network_exact_except_declared_states(
        source,
        destination,
        changes=changes,
    )
    after = audit_no_tls_major_response_consistency(
        destination,
        maximum_examples=0,
    )
    after_counts = dict(after["counts"])
    checks = {
        "all_declared_no_tls_priorities_are_asymmetric": planned_changes
        == int(
            counts[
                "candidate_with_asymmetric_external_no_tls_response_count"
            ]
        ),
        "all_declared_no_tls_foes_are_symmetric": int(
            counts[
                "external_no_tls_response_without_symmetric_foe_bit_count"
            ]
        )
        == 0,
        "all_candidate_major_states_removed": int(
            after_counts["candidate_with_external_no_tls_response_count"]
        )
        == 0,
        "remaining_major_responses_are_tls_only": int(
            after_counts["major_with_declared_response_connection_count"]
        )
        == int(after_counts["candidate_with_only_tls_controlled_response_count"]),
        "declared_state_changes_complete": len(changes) == planned_changes,
        "known_collision_movement_repaired": repair[
            "known_collision_connection"
        ]
        == list(KNOWN_COLLISION_CONNECTION),
        "network_exact_except_declared_changes": bool(compatibility["passed"]),
    }
    if not all(checks.values()):
        raise ValueError(
            f"Boston no-TLS major-response repair failed validation: {checks}"
        )
    return {
        "protocol": PROTOCOL,
        "pre_repair_counts": before["counts"],
        "post_repair_counts": after["counts"],
        "repair": repair,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": True,
    }


def validate_no_tls_major_response_manifest(
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(
            f"unexpected Boston v17 package protocol: {manifest.get('protocol')}"
        )
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_implicit_no_tls_major_manifest(predecessor)
    section = dict(manifest.get("no_tls_major_response_repair", {}))
    before = dict(section.get("pre_repair_counts", {}))
    after = dict(section.get("post_repair_counts", {}))
    repair = dict(section.get("repair", {}))
    checks = dict(section.get("checks", {}))
    planned_changes = int(
        before.get("candidate_with_external_no_tls_response_count", -1)
    )
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or planned_changes <= 0
        or int(
            before.get(
                "candidate_with_asymmetric_external_no_tls_response_count", -2
            )
        )
        != planned_changes
        or int(
            before.get("candidate_with_mutual_external_no_tls_response_count", -1)
        )
        != 0
        or int(after.get("candidate_with_external_no_tls_response_count", -1))
        != 0
        or int(repair.get("connection_state_change_count", -1))
        != planned_changes
        or dict(manifest.get("gates", {})).get(
            "boston_no_tls_major_response_states_repaired"
        )
        is not True
    ):
        raise ValueError("Boston v17 no-TLS major-response manifest is incomplete")


def repair_package(
    *, base_root: Path, output_root: Path, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v16 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_implicit_no_tls_major_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v16 base package is not admitted for remediation")
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
            base_root,
            staging,
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
                "no_tls_major_response_repair": section,
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_no_tls_major_response_states_repaired"] = bool(
            section["passed"]
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_declared_no_tls_response_major_state_alignment"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_no_tls_major_response_manifest(payload)
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
    section = dict(payload["no_tls_major_response_repair"])
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
