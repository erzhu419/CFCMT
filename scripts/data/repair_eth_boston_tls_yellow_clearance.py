#!/usr/bin/env python3
"""Normalize undersized Boston road-yellow phases to SUMO braking time."""

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

from scripts.data.audit_sumo_tls_yellow_clearance import (
    PROTOCOL as AUDIT_PROTOCOL,
    analyze_yellow_clearance,
    audit_tls_yellow_clearance,
)
from scripts.data.repair_eth_boston_merge_tls import (
    _sha256,
    _unchanged_files_share_base_inodes,
    _write_json,
)
from scripts.data.repair_eth_boston_systemic_uncontrolled_major import (
    PACKAGE_PROTOCOL as BASE_PACKAGE_PROTOCOL,
    validate_systemic_uncontrolled_major_manifest,
)


PROTOCOL = "cfcmt-eth-boston-tls-yellow-clearance-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v15"
KNOWN_FAILURE_TLS_ID = "7746087657"
KNOWN_FAILURE_PHASE_INDEX = 1

PhaseKey = tuple[str, str, int]


def _tag(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _duration_text(seconds: int) -> str:
    return str(int(seconds))


def _repair_plan(root: ET.Element) -> tuple[dict[PhaseKey, dict[str, Any]], dict[str, Any]]:
    analysis = analyze_yellow_clearance(root)
    counts = dict(analysis["counts"])
    if int(counts.get("anomaly_count", -1)) != 0:
        raise ValueError("Boston yellow-clearance audit has unresolved mappings")
    undersized = list(analysis["undersized_yellow_phases"])
    if not undersized:
        raise ValueError("Boston yellow-clearance repair plan is empty")

    plan: dict[PhaseKey, dict[str, Any]] = {}
    required_histogram: Counter[str] = Counter()
    shortfall_histogram: Counter[str] = Counter()
    for row in undersized:
        key = (
            str(row["tls_id"]),
            str(row["program_id"]),
            int(row["phase_index"]),
        )
        if key in plan:
            raise ValueError(f"duplicate yellow-clearance phase key: {key}")
        required = int(row["required_duration_sec"])
        changed_attributes: dict[str, dict[str, str | None]] = {
            "duration": {
                "before": str(int(row["duration_sec"]))
                if float(row["duration_sec"]).is_integer()
                else str(row["duration_sec"]),
                "after": _duration_text(required),
            }
        }
        for attribute, field in (("minDur", "min_duration_sec"), ("maxDur", "max_duration_sec")):
            value = row[field]
            if value is not None and float(value) < float(required):
                changed_attributes[attribute] = {
                    "before": str(int(value)) if float(value).is_integer() else str(value),
                    "after": _duration_text(required),
                }
        plan[key] = {
            **row,
            "changed_attributes": changed_attributes,
        }
        required_histogram[str(required)] += 1
        shortfall_histogram[str(int(row["shortfall_sec"]))] += 1

    failure_key = (KNOWN_FAILURE_TLS_ID, "0", KNOWN_FAILURE_PHASE_INDEX)
    if failure_key not in plan:
        raise ValueError(
            "known Boston v14 collision phase is absent from yellow repair plan"
        )
    failure = plan[failure_key]
    summary = {
        "method": "SUMO_computeBrakingTime_per_tls_maximum_incoming_speed",
        "changed_phase_count": len(plan),
        "changed_attribute_count": sum(
            len(row["changed_attributes"]) for row in plan.values()
        ),
        "required_duration_histogram_sec": dict(sorted(required_histogram.items())),
        "shortfall_histogram_sec": dict(sorted(shortfall_histogram.items())),
        "known_v14_failure_phase": {
            "tls_id": failure["tls_id"],
            "program_id": failure["program_id"],
            "phase_index": failure["phase_index"],
            "state": failure["state"],
            "duration_before_sec": failure["duration_sec"],
            "duration_after_sec": failure["required_duration_sec"],
            "maximum_incoming_speed_mps": failure["maximum_incoming_speed_mps"],
        },
        "change_examples": [
            {
                "tls_id": row["tls_id"],
                "program_id": row["program_id"],
                "phase_index": row["phase_index"],
                "duration_before_sec": row["duration_sec"],
                "duration_after_sec": row["required_duration_sec"],
                "maximum_incoming_speed_mps": row["maximum_incoming_speed_mps"],
            }
            for row in list(plan.values())[:20]
        ],
    }
    return plan, summary


def _apply_plan(root: ET.Element, plan: Mapping[PhaseKey, Mapping[str, Any]]) -> None:
    observed: set[PhaseKey] = set()
    for logic in root.iter():
        if _tag(logic) != "tlLogic":
            continue
        tls_id = str(logic.get("id", ""))
        program_id = str(logic.get("programID", ""))
        phase_index = 0
        for phase in logic:
            if _tag(phase) != "phase":
                continue
            key = (tls_id, program_id, phase_index)
            phase_index += 1
            if key not in plan:
                continue
            for attribute, change in dict(plan[key]["changed_attributes"]).items():
                expected = change["before"]
                observed_before = phase.get(attribute)
                if observed_before != expected:
                    raise ValueError(
                        f"yellow phase changed before repair: {key} {attribute} "
                        f"{observed_before!r} != {expected!r}"
                    )
                phase.set(attribute, str(change["after"]))
            observed.add(key)
    if observed != set(plan):
        raise ValueError("not every declared yellow phase was repaired")


def _network_exact_except_declared_durations(
    source: Path,
    repaired: Path,
    *,
    plan: Mapping[PhaseKey, Mapping[str, Any]],
) -> dict[str, Any]:
    before = ET.iterparse(source, events=("start", "end"))
    after = ET.iterparse(repaired, events=("start", "end"))
    sentinel = object()
    active_tls: tuple[str, str] | None = None
    phase_index = -1
    observed: set[PhaseKey] = set()
    element_count = 0
    while True:
        left = next(before, sentinel)
        right = next(after, sentinel)
        if left is sentinel or right is sentinel:
            exact = bool(
                left is sentinel and right is sentinel and observed == set(plan)
            )
            return {
                "network_xml_exact_except_declared_phase_durations": exact,
                "element_count": element_count,
                "observed_changed_phase_count": len(observed),
                "expected_changed_phase_count": len(plan),
                "passed": exact,
            }
        left_event, left_element = left
        right_event, right_element = right
        element_count += int(left_event == "start")
        if left_event != right_event or _tag(left_element) != _tag(right_element):
            return {"passed": False, "reason": "element_sequence_changed"}
        if left_event == "start":
            left_attributes = dict(left_element.attrib)
            right_attributes = dict(right_element.attrib)
            if _tag(left_element) == "tlLogic":
                active_tls = (
                    str(left_element.get("id", "")),
                    str(left_element.get("programID", "")),
                )
                phase_index = -1
            elif _tag(left_element) == "phase" and active_tls is not None:
                phase_index += 1
                key = (*active_tls, phase_index)
                if key in plan:
                    for attribute, change in dict(
                        plan[key]["changed_attributes"]
                    ).items():
                        if (
                            left_attributes.get(attribute) != change["before"]
                            or right_attributes.get(attribute) != change["after"]
                        ):
                            return {
                                "passed": False,
                                "reason": "declared_duration_change_mismatch",
                                "phase_key": list(key),
                                "attribute": attribute,
                            }
                        left_attributes[attribute] = str(change["after"])
                    observed.add(key)
            if left_attributes != right_attributes:
                return {
                    "passed": False,
                    "reason": "undeclared_attribute_change",
                }
        else:
            if _tag(left_element) == "tlLogic":
                active_tls = None
                phase_index = -1
            left_element.clear()
            right_element.clear()


def repair_network(source: Path, destination: Path) -> dict[str, Any]:
    tree = ET.parse(source)
    plan, repair = _repair_plan(tree.getroot())
    pre_audit = audit_tls_yellow_clearance(source, maximum_examples=0)
    _apply_plan(tree.getroot(), plan)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    compatibility = _network_exact_except_declared_durations(
        source, destination, plan=plan
    )
    post_audit = audit_tls_yellow_clearance(destination, maximum_examples=0)
    pre_counts = dict(pre_audit["counts"])
    post_counts = dict(post_audit["counts"])
    failure = dict(repair["known_v14_failure_phase"])
    checks = {
        "pre_repair_audit_protocol_exact": pre_audit.get("protocol") == AUDIT_PROTOCOL,
        "pre_repair_mapping_actionable": pre_audit.get("actionable") is True,
        "pre_repair_undersized_phases_present": int(
            pre_counts.get("undersized_yellow_phase_count", 0)
        )
        > 0,
        "known_v14_failure_phase_repaired": (
            failure.get("tls_id") == KNOWN_FAILURE_TLS_ID
            and int(failure.get("phase_index", -1)) == KNOWN_FAILURE_PHASE_INDEX
            and float(failure.get("duration_after_sec", 0.0))
            > float(failure.get("duration_before_sec", 0.0))
        ),
        "declared_phase_changes_complete": int(repair["changed_phase_count"])
        == int(pre_counts["undersized_yellow_phase_count"]),
        "yellow_phase_inventory_preserved": int(post_counts["yellow_phase_count"])
        == int(pre_counts["yellow_phase_count"]),
        "all_undersized_yellow_phases_removed": int(
            post_counts["undersized_yellow_phase_count"]
        )
        == 0,
        "post_repair_mapping_exact": int(post_counts["anomaly_count"]) == 0,
        "network_exact_except_declared_changes": bool(compatibility["passed"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Boston yellow-clearance repair failed validation: {checks}")
    return {
        "protocol": PROTOCOL,
        "pre_repair_counts": pre_counts,
        "post_repair_counts": post_counts,
        "repair": repair,
        "compatibility_audit": compatibility,
        "checks": checks,
        "passed": True,
    }


def validate_tls_yellow_clearance_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected Boston v15 package protocol: {manifest.get('protocol')}")
    predecessor = dict(manifest)
    predecessor["protocol"] = BASE_PACKAGE_PROTOCOL
    validate_systemic_uncontrolled_major_manifest(predecessor)
    section = dict(manifest.get("tls_yellow_clearance_repair", {}))
    pre = dict(section.get("pre_repair_counts", {}))
    post = dict(section.get("post_repair_counts", {}))
    repair = dict(section.get("repair", {}))
    failure = dict(repair.get("known_v14_failure_phase", {}))
    checks = dict(section.get("checks", {}))
    if (
        section.get("protocol") != PROTOCOL
        or section.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or int(pre.get("undersized_yellow_phase_count", 0)) <= 0
        or int(post.get("undersized_yellow_phase_count", -1)) != 0
        or int(pre.get("anomaly_count", -1)) != 0
        or int(post.get("anomaly_count", -1)) != 0
        or int(repair.get("changed_phase_count", -1))
        != int(pre.get("undersized_yellow_phase_count", -2))
        or failure.get("tls_id") != KNOWN_FAILURE_TLS_ID
        or int(failure.get("phase_index", -1)) != KNOWN_FAILURE_PHASE_INDEX
        or dict(manifest.get("gates", {})).get(
            "boston_tls_yellow_clearance_normalized"
        )
        is not True
    ):
        raise ValueError("Boston v15 yellow-clearance manifest is incomplete")


def repair_package(
    *, base_root: Path, output_root: Path, expected_base_manifest_sha256: str
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston v14 base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    validate_systemic_uncontrolled_major_manifest(base_manifest)
    if base_manifest.get("passed") is not True:
        raise ValueError("Boston v14 base package is not admitted for remediation")
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
                "tls_yellow_clearance_repair": section,
            }
        )
        gates = dict(payload.get("gates", {}))
        gates["boston_tls_yellow_clearance_normalized"] = bool(section["passed"])
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_sumo_braking_time_yellow_clearance_normalization"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
        validate_tls_yellow_clearance_manifest(payload)
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
    section = dict(payload["tls_yellow_clearance_repair"])
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
