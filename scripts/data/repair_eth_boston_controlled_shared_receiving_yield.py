#!/usr/bin/env python3
"""Repair missing TLS yield responses for Boston shared receiving lanes."""

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

from scripts.data.audit_sumo_controlled_shared_receiving_yield import (
    PROTOCOL as AUDIT_PROTOCOL,
    _coactive_requirements,
    _programs,
    audit_controlled_shared_receiving_yield,
)
from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    _bit,
    _connection_identity,
    _junction_indices,
    _lane_id,
)
from scripts.data.repair_eth_boston_merge_tls import (
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_json,
)
from scripts.data.repair_eth_boston_no_tls_major_response import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_no_tls_major_response_manifest,
)
from scripts.data.repair_eth_boston_systemic_right_of_way import (
    _network_exact_except_plan,
    _set_bit,
)


PROTOCOL = "cfcmt-eth-boston-controlled-shared-receiving-yield-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v18"
KNOWN_COLLISION_JUNCTION = "cluster_61369297_61369347"
KNOWN_COLLISION_RECEIVING_LANE = "426499591#0_0"
KNOWN_YIELDING_INDEX = 4
KNOWN_PROTECTED_INDEX = 3


def _repair_plan(root: ET.Element) -> dict[str, Any]:
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
    programs = _programs(root)
    receiving: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        if (
            connection.get("tl")
            and connection.get("linkIndex") not in {None, ""}
            and not str(connection.get("from", "")).startswith(":")
        ):
            receiving[_lane_id(connection.attrib, "to")].append(connection)

    requirements: dict[tuple[str, int, int], dict[str, Any]] = {}
    observed_requirements: set[tuple[str, int, int]] = set()
    for receiving_lane, rows in sorted(receiving.items()):
        if len(rows) < 2:
            continue
        for left_offset, left in enumerate(rows):
            for right in rows[left_offset + 1 :]:
                coactive, simultaneous = _coactive_requirements(
                    left,
                    right,
                    programs,
                )
                if simultaneous:
                    raise ValueError(
                        "simultaneous protected movements share a receiving lane"
                    )
                if not coactive:
                    continue
                junction_id = str(edges_from.get(str(left.get("to", "")), ""))
                diagnostic = diagnostics.get(junction_id, {})
                junction_requests = list(diagnostic.get("requests", ()))
                left_index = indices.get(left)
                right_index = indices.get(right)
                if (
                    not diagnostic.get("index_mapping_exact", False)
                    or left_index is None
                    or right_index is None
                    or max(left_index, right_index) >= len(junction_requests)
                ):
                    raise ValueError(
                        f"unresolved junction-index mapping at {junction_id}"
                    )
                for phase in coactive:
                    if phase["yielding_side"] == "left":
                        yielding_index, protected_index = left_index, right_index
                        yielding_connection = left
                        protected_connection = right
                    else:
                        yielding_index, protected_index = right_index, left_index
                        yielding_connection = right
                        protected_connection = left
                    key = (
                        junction_id,
                        int(yielding_index),
                        int(protected_index),
                    )
                    if key in observed_requirements:
                        continue
                    observed_requirements.add(key)
                    yielding_request = junction_requests[int(yielding_index)]
                    protected_request = junction_requests[int(protected_index)]
                    symmetric_foe = bool(
                        _bit(
                            str(yielding_request.get("foes", "")),
                            int(protected_index),
                        )
                        == "1"
                        and _bit(
                            str(protected_request.get("foes", "")),
                            int(yielding_index),
                        )
                        == "1"
                    )
                    if not symmetric_foe:
                        raise ValueError(
                            f"co-active shared-receiver movements are not foes at {junction_id}"
                        )
                    if (
                        _bit(
                            str(yielding_request.get("response", "")),
                            int(protected_index),
                        )
                        == "1"
                    ):
                        continue
                    requirements[key] = {
                        "junction_id": junction_id,
                        "receiving_lane": receiving_lane,
                        "yielding_junction_index": int(yielding_index),
                        "protected_junction_index": int(protected_index),
                        "yielding_connection": _connection_identity(
                            yielding_connection
                        ),
                        "protected_connection": _connection_identity(
                            protected_connection
                        ),
                        "phase": dict(phase),
                    }

    known_key = (
        KNOWN_COLLISION_JUNCTION,
        KNOWN_YIELDING_INDEX,
        KNOWN_PROTECTED_INDEX,
    )
    known = requirements.get(known_key)
    if known is None or known["receiving_lane"] != KNOWN_COLLISION_RECEIVING_LANE:
        raise ValueError("Boston v17 collision relation is not exactly covered")
    if not requirements:
        raise ValueError("Boston controlled shared-receiver repair plan is empty")
    return {
        "observed_directed_requirement_count": len(observed_requirements),
        "requirements": requirements,
        "known_collision_relation": known,
    }


def _apply_plan(root: ET.Element, plan: Mapping[str, Any]) -> dict[str, Any]:
    junctions = {
        str(junction.get("id", "")): junction for junction in root.iter("junction")
    }
    request_before: dict[tuple[str, int], str] = {}
    response_bits_set = 0
    for junction_id, yielding_index, protected_index in plan["requirements"]:
        requests = list(junctions[junction_id].findall("request"))
        request = requests[int(yielding_index)]
        request_before.setdefault(
            (junction_id, int(yielding_index)),
            str(request.get("response", "")),
        )
        before = str(request.get("response", ""))
        after = _set_bit(before, int(protected_index), "1")
        if after == before:
            raise ValueError("planned missing response bit was already set")
        request.set("response", after)
        response_bits_set += 1

    request_after = {
        key: str(list(junctions[key[0]].findall("request"))[key[1]].get("response", ""))
        for key in request_before
    }
    return {
        "response_bits_set": response_bits_set,
        "changed_request_count": len(request_before),
        "request_before": request_before,
        "request_after": request_after,
        "phase_before": {},
        "phase_after": {},
    }


def repair_network(source: Path, destination: Path) -> dict[str, Any]:
    before = audit_controlled_shared_receiving_yield(source, maximum_examples=0)
    before_counts = dict(before["counts"])
    missing_count = int(before_counts.get("missing_yield_response_count", -1))
    if (
        before.get("protocol") != AUDIT_PROTOCOL
        or missing_count <= 0
        or int(before_counts.get("missing_symmetric_foe_count", -1)) != 0
        or int(before_counts.get("simultaneous_protected_phase_count", -1)) != 0
        or int(before_counts.get("unresolved_junction_index_pair_count", -1)) != 0
        or int(before_counts.get("cross_tls_shared_receiving_pair_count", -1)) != 0
    ):
        raise ValueError(
            "Boston controlled shared-receiver audit is not safely actionable"
        )

    tree = ET.parse(source)
    plan = _repair_plan(tree.getroot())
    if len(plan["requirements"]) != missing_count:
        raise ValueError("repair plan differs from the read-only audit")
    changes = _apply_plan(tree.getroot(), plan)
    if int(changes["response_bits_set"]) != missing_count:
        raise ValueError("not every missing response bit was repaired")
    tree.write(destination, encoding="utf-8", xml_declaration=True)

    compatibility = _network_exact_except_plan(source, destination, changes=changes)
    after = audit_controlled_shared_receiving_yield(destination, maximum_examples=0)
    after_counts = dict(after["counts"])
    repaired_junctions = {
        str(junction.get("id", "")): junction
        for junction in ET.parse(destination).getroot().iter("junction")
    }
    known_response = str(
        list(
            repaired_junctions[KNOWN_COLLISION_JUNCTION].findall("request")
        )[KNOWN_YIELDING_INDEX].get("response", "")
    )
    invariant_count_keys = (
        "controlled_shared_receiving_pair_count",
        "cross_tls_shared_receiving_pair_count",
        "coactive_same_tls_pair_count",
        "directed_yield_requirement_count",
        "missing_symmetric_foe_count",
        "simultaneous_protected_phase_count",
        "unresolved_junction_index_pair_count",
    )
    checks = {
        "all_audit_population_counts_preserved": all(
            int(after_counts[key]) == int(before_counts[key])
            for key in invariant_count_keys
        ),
        "all_missing_yield_responses_repaired": int(
            after_counts["missing_yield_response_count"]
        )
        == 0,
        "declared_response_changes_complete": int(changes["response_bits_set"])
        == missing_count,
        "known_collision_relation_repaired": _bit(
            known_response,
            KNOWN_PROTECTED_INDEX,
        )
        == "1",
        "network_exact_except_declared_changes": bool(compatibility["passed"]),
    }
    if not all(checks.values()):
        raise ValueError(
            f"Boston controlled shared-receiver repair failed validation: {checks}"
        )
    public_changes = {
        "observed_directed_requirement_count": int(
            plan["observed_directed_requirement_count"]
        ),
        "response_bits_set": int(changes["response_bits_set"]),
        "changed_request_count": int(changes["changed_request_count"]),
        "known_collision_relation": plan["known_collision_relation"],
        "change_examples": list(plan["requirements"].values())[:20],
    }
    return {
        "protocol": PROTOCOL,
        "pre_repair_counts": before_counts,
        "post_repair_counts": after_counts,
        "repair": public_changes,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": True,
    }


def validate_controlled_shared_receiving_yield_manifest(
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(
            f"unexpected Boston v18 package protocol: {manifest.get('protocol')}"
        )
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_no_tls_major_response_manifest(predecessor)
    section = dict(manifest.get("controlled_shared_receiving_yield_repair", {}))
    before = dict(section.get("pre_repair_counts", {}))
    after = dict(section.get("post_repair_counts", {}))
    repair = dict(section.get("repair", {}))
    checks = dict(section.get("checks", {}))
    missing_count = int(before.get("missing_yield_response_count", -1))
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or missing_count <= 0
        or int(after.get("missing_yield_response_count", -1)) != 0
        or int(repair.get("response_bits_set", -1)) != missing_count
        or dict(manifest.get("gates", {})).get(
            "boston_controlled_shared_receiving_yield_repaired"
        )
        is not True
    ):
        raise ValueError("Boston v18 controlled-yield manifest is incomplete")


def repair_package(
    *, base_root: Path, output_root: Path, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v17 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_no_tls_major_response_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v17 base package is not admitted for remediation")
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
                "controlled_shared_receiving_yield_repair": section,
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_controlled_shared_receiving_yield_repaired"] = bool(
            section["passed"]
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_controlled_shared_receiving_yield_response_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_controlled_shared_receiving_yield_manifest(payload)
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
    section = dict(payload["controlled_shared_receiving_yield_repair"])
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
