#!/usr/bin/env python3
"""Repair Boston controlled/uncontrolled merge priority as one network class."""

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
    PROTOCOL as AUDIT_PROTOCOL,
    _bit,
    _junction_indices,
    _lane_id,
    audit_shared_receiving_right_of_way,
)
from scripts.data.repair_eth_boston_joined_tls_uncontrolled_merge import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_joined_tls_uncontrolled_merge_manifest,
)
from scripts.data.repair_eth_boston_merge_tls import (
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_json,
)


PROTOCOL = "cfcmt-eth-boston-systemic-controlled-major-right-of-way-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v13"


def _set_bit(value: str, junction_index: int, bit: str) -> str:
    text = str(value)
    offset = len(text) - 1 - int(junction_index)
    if bit not in {"0", "1"} or not 0 <= offset < len(text):
        raise ValueError("invalid junction response-bit update")
    return text[:offset] + bit + text[offset + 1 :]


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
    receiving: dict[str, list[ET.Element]] = defaultdict(list)
    for connection in connections:
        receiving[_lane_id(connection.attrib, "to")].append(connection)

    relation_pairs: set[tuple[str, int, int]] = set()
    controlled_links: set[tuple[str, int]] = set()
    for rows in receiving.values():
        controlled = [row for row in rows if row.get("tl") and row.get("linkIndex")]
        majors = [
            row
            for row in rows
            if not row.get("tl")
            and not str(row.get("from", "")).startswith(":")
            and (row.get("state") == "M" or row.get("uncontrolled") == "1")
        ]
        if not controlled or not majors:
            continue
        junction_id = str(edges_from.get(str(rows[0].get("to", "")), ""))
        diagnostic = diagnostics.get(junction_id, {})
        requests = list(diagnostic.get("requests", ()))
        if not diagnostic.get("index_mapping_exact", False):
            raise ValueError(f"unresolved junction-index mapping at {junction_id}")
        for controlled_connection in controlled:
            controlled_index = indices[controlled_connection]
            controlled_links.add(
                (
                    str(controlled_connection.get("tl")),
                    int(str(controlled_connection.get("linkIndex"))),
                )
            )
            for major_connection in majors:
                major_index = indices[major_connection]
                controlled_request = requests[controlled_index]
                major_request = requests[major_index]
                if (
                    _bit(str(controlled_request.get("foes", "")), major_index)
                    != "1"
                    or _bit(str(major_request.get("foes", "")), controlled_index)
                    != "1"
                ):
                    raise ValueError(
                        f"shared receiving movements are not symmetric foes at {junction_id}"
                    )
                relation_pairs.add((junction_id, controlled_index, major_index))
    if not relation_pairs or not controlled_links:
        raise ValueError("Boston systemic right-of-way repair found no candidates")
    return {
        "relation_pairs": tuple(sorted(relation_pairs)),
        "controlled_links": tuple(sorted(controlled_links)),
    }


def _apply_plan(root: ET.Element, plan: Mapping[str, Any]) -> dict[str, Any]:
    junctions = {
        str(junction.get("id", "")): junction for junction in root.iter("junction")
    }
    request_before: dict[tuple[str, int], str] = {}
    response_bits_set = 0
    response_bits_cleared = 0
    for junction_id, controlled_index, major_index in plan["relation_pairs"]:
        requests = list(junctions[junction_id].findall("request"))
        if max(controlled_index, major_index) >= len(requests):
            raise ValueError(f"request relation disappeared at {junction_id}")
        for index in (controlled_index, major_index):
            request_before.setdefault(
                (junction_id, index), str(requests[index].get("response", ""))
            )
        controlled = requests[controlled_index]
        major = requests[major_index]
        before = str(controlled.get("response", ""))
        after = _set_bit(before, major_index, "1")
        response_bits_set += int(after != before)
        controlled.set("response", after)
        before = str(major.get("response", ""))
        after = _set_bit(before, controlled_index, "0")
        response_bits_cleared += int(after != before)
        major.set("response", after)

    controlled_links = set(plan["controlled_links"])
    phase_before: dict[tuple[str, str, int], str] = {}
    phase_character_changes = 0
    for program in root.iter("tlLogic"):
        tls_id = str(program.get("id", ""))
        program_id = str(program.get("programID", ""))
        relevant = sorted(
            link_index
            for candidate_tls, link_index in controlled_links
            if candidate_tls == tls_id
        )
        if not relevant:
            continue
        for phase_index, phase in enumerate(program.findall("phase")):
            state = str(phase.get("state", ""))
            repaired = list(state)
            for link_index in relevant:
                if link_index >= len(repaired):
                    raise ValueError(
                        f"TLS state is too short: {tls_id}/{program_id}/{phase_index}"
                    )
                if repaired[link_index] == "G":
                    repaired[link_index] = "g"
                    phase_character_changes += 1
            updated = "".join(repaired)
            if updated != state:
                phase_before[(tls_id, program_id, phase_index)] = state
                phase.set("state", updated)

    request_after = {
        key: str(list(junctions[key[0]].findall("request"))[key[1]].get("response", ""))
        for key in request_before
    }
    phase_after = {}
    programs = {
        (str(program.get("id", "")), str(program.get("programID", ""))): program
        for program in root.iter("tlLogic")
    }
    for tls_id, program_id, phase_index in phase_before:
        phase_after[(tls_id, program_id, phase_index)] = str(
            programs[(tls_id, program_id)].findall("phase")[phase_index].get("state", "")
        )
    return {
        "relation_pair_count": len(plan["relation_pairs"]),
        "controlled_tls_link_count": len(controlled_links),
        "response_bits_set": response_bits_set,
        "response_bits_cleared": response_bits_cleared,
        "changed_request_count": len(request_before),
        "phase_character_changes": phase_character_changes,
        "changed_phase_count": len(phase_before),
        "request_before": request_before,
        "request_after": request_after,
        "phase_before": phase_before,
        "phase_after": phase_after,
    }


def _network_exact_except_plan(
    source: Path,
    repaired: Path,
    *,
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    left = ET.iterparse(source, events=("start", "end"))
    right = ET.iterparse(repaired, events=("start", "end"))
    sentinel = object()
    active_tls: tuple[str, str] | None = None
    active_junction: str | None = None
    phase_index = -1
    observed_requests: set[tuple[str, int]] = set()
    observed_phases: set[tuple[str, str, int]] = set()
    element_count = 0
    while True:
        lrow = next(left, sentinel)
        rrow = next(right, sentinel)
        if lrow is sentinel or rrow is sentinel:
            exact = bool(
                lrow is sentinel
                and rrow is sentinel
                and observed_requests == set(changes["request_before"])
                and observed_phases == set(changes["phase_before"])
            )
            return {
                "network_xml_exact_except_declared_right_of_way_changes": exact,
                "element_count": element_count,
                "observed_changed_request_count": len(observed_requests),
                "observed_changed_phase_count": len(observed_phases),
                "passed": exact,
            }
        levent, lelement = lrow
        revent, relement = rrow
        element_count += int(levent == "start")
        if levent != revent or lelement.tag != relement.tag:
            return {"passed": False, "reason": "element_sequence_changed"}
        if levent == "start":
            lattrs = dict(lelement.attrib)
            rattrs = dict(relement.attrib)
            if lelement.tag == "tlLogic":
                active_tls = (
                    str(lelement.get("id", "")),
                    str(lelement.get("programID", "")),
                )
                phase_index = -1
            elif lelement.tag == "junction":
                active_junction = str(lelement.get("id", ""))
            elif lelement.tag == "phase" and active_tls is not None:
                phase_index += 1
                key = (*active_tls, phase_index)
                if key in changes["phase_before"]:
                    if (
                        lattrs.get("state") != changes["phase_before"][key]
                        or rattrs.get("state") != changes["phase_after"][key]
                    ):
                        return {"passed": False, "reason": "phase_change_mismatch"}
                    lattrs["state"] = rattrs["state"]
                    observed_phases.add(key)
            elif lelement.tag == "request" and active_junction is not None:
                index = int(str(lelement.get("index")))
                key = (active_junction, index)
                if key in changes["request_before"]:
                    if (
                        lattrs.get("response") != changes["request_before"][key]
                        or rattrs.get("response") != changes["request_after"][key]
                    ):
                        return {"passed": False, "reason": "request_change_mismatch"}
                    lattrs["response"] = rattrs["response"]
                    observed_requests.add(key)
            if lattrs != rattrs:
                return {"passed": False, "reason": "undeclared_attribute_change"}
        else:
            if lelement.tag == "tlLogic":
                active_tls = None
                phase_index = -1
            elif lelement.tag == "junction":
                active_junction = None
            lelement.clear()
            relement.clear()


def repair_network(source: Path, destination: Path) -> dict[str, Any]:
    before = audit_shared_receiving_right_of_way(source, maximum_examples=0)
    if (
        before["protocol"] != AUDIT_PROTOCOL
        or before["counts"]["unresolved_junction_index_pair_count"] != 0
        or before["counts"]["missing_symmetric_foe_pair_count"] != 0
    ):
        raise ValueError("Boston pre-repair right-of-way audit is not actionable")
    tree = ET.parse(source)
    plan = _repair_plan(tree.getroot())
    changes = _apply_plan(tree.getroot(), plan)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    compatibility = _network_exact_except_plan(source, destination, changes=changes)
    after = audit_shared_receiving_right_of_way(destination, maximum_examples=0)
    pair_count = int(before["counts"]["controlled_uncontrolled_major_pair_count"])
    checks = {
        "candidate_pair_count_preserved": int(
            after["counts"]["controlled_uncontrolled_major_pair_count"]
        )
        == pair_count,
        "all_junction_indices_resolved": int(
            after["counts"]["unresolved_junction_index_pair_count"]
        )
        == 0,
        "all_controlled_movements_yield_to_major": int(
            after["counts"]["controlled_missing_yield_pair_count"]
        )
        == 0,
        "no_major_movement_yields_to_controlled": int(
            after["counts"]["major_wrongly_yields_pair_count"]
        )
        == 0,
        "all_pairs_are_symmetric_foes": int(
            after["counts"]["missing_symmetric_foe_pair_count"]
        )
        == 0,
        "all_candidate_signal_service_is_permissive": int(
            after["counts"]["protected_service_pair_count"]
        )
        == 0,
        "every_candidate_pair_is_exact": int(
            after["counts"]["already_exact_pair_count"]
        )
        == pair_count,
        "network_exact_except_declared_changes": bool(compatibility["passed"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Boston systemic repair failed validation: {checks}")
    public_changes = {
        key: value
        for key, value in changes.items()
        if key
        not in {"request_before", "request_after", "phase_before", "phase_after"}
    }
    return {
        "protocol": PROTOCOL,
        "pre_repair_counts": before["counts"],
        "post_repair_counts": after["counts"],
        "repair": public_changes,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_systemic_right_of_way_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected Boston v13 package protocol: {manifest.get('protocol')}")
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_joined_tls_uncontrolled_merge_manifest(predecessor)
    section = dict(manifest.get("systemic_right_of_way_repair", {}))
    before = dict(section.get("pre_repair_counts", {}))
    after = dict(section.get("post_repair_counts", {}))
    repair = dict(section.get("repair", {}))
    checks = dict(section.get("checks", {}))
    pair_count = int(before.get("controlled_uncontrolled_major_pair_count", -1))
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or pair_count <= 0
        or int(after.get("controlled_uncontrolled_major_pair_count", -2)) != pair_count
        or int(after.get("already_exact_pair_count", -1)) != pair_count
        or int(after.get("controlled_missing_yield_pair_count", -1)) != 0
        or int(after.get("major_wrongly_yields_pair_count", -1)) != 0
        or int(after.get("protected_service_pair_count", -1)) != 0
        or int(repair.get("relation_pair_count", -1)) != pair_count
        or dict(manifest.get("gates", {})).get(
            "boston_systemic_controlled_major_right_of_way_repaired"
        )
        is not True
    ):
        raise ValueError("Boston v13 systemic right-of-way manifest is incomplete")


def repair_package(
    *, base_root: Path, output_root: Path, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v12 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_joined_tls_uncontrolled_merge_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v12 base package is not admitted for remediation")
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
                "systemic_right_of_way_repair": section,
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_systemic_controlled_major_right_of_way_repaired"] = bool(
            section["passed"]
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_systemic_controlled_major_right_of_way_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_systemic_right_of_way_manifest(payload)
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
                "counts": payload["systemic_right_of_way_repair"]["post_repair_counts"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
