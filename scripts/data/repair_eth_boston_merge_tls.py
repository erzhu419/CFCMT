#!/usr/bin/env python3
"""Apply the declared Boston three-to-one merge TLS yield repair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "cfcmt-eth-boston-three-to-one-merge-tls-yield-repair-v1"
PACKAGE_PROTOCOL = "cfcmt-eth-boston-complete-published-package-v7"
TLS_ID = "joinedS_726"
DECLARED_CHANGES: dict[tuple[int, int], tuple[str, str]] = {
    (10, 12): ("G", "g"),
    (12, 14): ("G", "g"),
}


def validate_remediated_package_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PACKAGE_PROTOCOL:
        raise ValueError(
            f"unexpected remediated package protocol: {manifest.get('protocol')}"
        )
    if manifest.get("city_code") != "BOS":
        raise ValueError("the declared merge TLS repair is Boston-specific")
    topology = dict(manifest.get("collision_topology_repair", {}))
    repair = dict(topology.get("repair", {}))
    compatibility = dict(topology.get("compatibility_audit", {}))
    expected_changes = [
        {
            "phase_index": phase_index,
            "link_index": link_index,
            "before": before,
            "after": after,
        }
        for (phase_index, link_index), (before, after) in DECLARED_CHANGES.items()
    ]
    checks = dict(topology.get("checks", {}))
    if (
        topology.get("protocol") != PROTOCOL
        or topology.get("passed") is not True
        or not checks
        or not all(bool(value) for value in checks.values())
        or compatibility.get("passed") is not True
        or int(compatibility.get("observed_declared_changes", -1))
        != len(DECLARED_CHANGES)
        or int(compatibility.get("expected_declared_changes", -1))
        != len(DECLARED_CHANGES)
        or repair.get("tls_id") != TLS_ID
        or repair.get("method")
        != "declared_phase_link_priority_substitution"
        or int(repair.get("changed_phase_character_count", -1))
        != len(DECLARED_CHANGES)
        or repair.get("changes") != expected_changes
        or dict(manifest.get("gates", {})).get(
            "boston_three_to_one_merge_tls_yield_repaired"
        )
        is not True
    ):
        raise ValueError("Boston merge TLS repair manifest is incomplete")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_declared_tls_yield_repair(
    source: Path,
    destination: Path,
    *,
    tls_id: str = TLS_ID,
    declared_changes: Mapping[tuple[int, int], tuple[str, str]] = DECLARED_CHANGES,
) -> dict[str, Any]:
    tree = ET.parse(source)
    programs = [
        element
        for element in tree.getroot().iter("tlLogic")
        if element.attrib.get("id") == tls_id
    ]
    if len(programs) != 1:
        raise ValueError(
            f"expected exactly one TLS program {tls_id!r}, found {len(programs)}"
        )
    phases = programs[0].findall("phase")
    observed: list[dict[str, Any]] = []
    for (phase_index, link_index), (before, after) in declared_changes.items():
        if phase_index >= len(phases):
            raise ValueError(f"declared phase is absent: {phase_index}")
        state = str(phases[phase_index].attrib.get("state", ""))
        if link_index >= len(state) or state[link_index] != before:
            actual = None if link_index >= len(state) else state[link_index]
            raise ValueError(
                "declared TLS state changed before repair: "
                f"phase={phase_index} link={link_index} "
                f"expected={before!r} actual={actual!r}"
            )
        repaired = state[:link_index] + after + state[link_index + 1 :]
        phases[phase_index].set("state", repaired)
        observed.append(
            {
                "phase_index": phase_index,
                "link_index": link_index,
                "before": before,
                "after": after,
            }
        )
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return {
        "tls_id": tls_id,
        "method": "declared_phase_link_priority_substitution",
        "changed_phase_character_count": len(observed),
        "changes": observed,
    }


def _network_exact_except_declared_tls_changes(
    source: Path,
    repaired: Path,
    *,
    tls_id: str = TLS_ID,
    declared_changes: Mapping[tuple[int, int], tuple[str, str]] = DECLARED_CHANGES,
) -> dict[str, Any]:
    before = ET.iterparse(source, events=("start", "end"))
    after = ET.iterparse(repaired, events=("start", "end"))
    sentinel = object()
    active_tls: str | None = None
    phase_index = -1
    observed: set[tuple[int, int]] = set()
    while True:
        left = next(before, sentinel)
        right = next(after, sentinel)
        if left is sentinel or right is sentinel:
            exact = (
                left is sentinel
                and right is sentinel
                and observed == set(declared_changes)
            )
            return {
                "network_xml_exact_except_declared_tls_changes": exact,
                "observed_declared_changes": len(observed),
                "expected_declared_changes": len(declared_changes),
                "passed": exact,
            }

        left_event, left_element = left
        right_event, right_element = right
        if left_event != right_event or left_element.tag != right_element.tag:
            return {
                "network_xml_exact_except_declared_tls_changes": False,
                "observed_declared_changes": len(observed),
                "expected_declared_changes": len(declared_changes),
                "passed": False,
            }

        if left_event == "start":
            left_attributes = dict(left_element.attrib)
            right_attributes = dict(right_element.attrib)
            if left_element.tag == "tlLogic":
                if left_attributes.get("id") != right_attributes.get("id"):
                    return {
                        "network_xml_exact_except_declared_tls_changes": False,
                        "observed_declared_changes": len(observed),
                        "expected_declared_changes": len(declared_changes),
                        "passed": False,
                    }
                active_tls = str(left_attributes.get("id", ""))
                phase_index = -1
            elif left_element.tag == "phase" and active_tls:
                phase_index += 1
                if active_tls == tls_id:
                    for link_index in (
                        index
                        for candidate_phase, index in declared_changes
                        if candidate_phase == phase_index
                    ):
                        before_state, after_state = declared_changes[
                            (phase_index, link_index)
                        ]
                        left_state = str(left_attributes.get("state", ""))
                        right_state = str(right_attributes.get("state", ""))
                        if (
                            link_index >= len(left_state)
                            or link_index >= len(right_state)
                            or left_state[link_index] != before_state
                            or right_state[link_index] != after_state
                            or right_state
                            != left_state[:link_index]
                            + after_state
                            + left_state[link_index + 1 :]
                        ):
                            return {
                                "network_xml_exact_except_declared_tls_changes": False,
                                "observed_declared_changes": len(observed),
                                "expected_declared_changes": len(declared_changes),
                                "passed": False,
                            }
                        left_attributes["state"] = right_state
                        observed.add((phase_index, link_index))
            if left_attributes != right_attributes:
                return {
                    "network_xml_exact_except_declared_tls_changes": False,
                    "observed_declared_changes": len(observed),
                    "expected_declared_changes": len(declared_changes),
                    "passed": False,
                }
        else:
            if left_element.tag == "tlLogic":
                active_tls = None
                phase_index = -1
            left_element.clear()
            right_element.clear()


def _unchanged_files_share_base_inodes(
    base_root: Path,
    derived_root: Path,
) -> tuple[bool, int]:
    excluded = {"microscopic_network.net.xml", "package_manifest.json"}
    files = [path for path in base_root.rglob("*") if path.is_file()]
    unchanged = [path for path in files if path.relative_to(base_root).as_posix() not in excluded]
    return (
        all(
            os.path.samefile(path, derived_root / path.relative_to(base_root))
            for path in unchanged
        ),
        len(unchanged),
    )


def repair_package(
    *,
    base_root: Path,
    output_root: Path,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    base_manifest_path = base_root / "package_manifest.json"
    observed_sha256 = _sha256(base_manifest_path)
    if observed_sha256 != expected_base_manifest_sha256:
        raise ValueError("Boston base package manifest identity changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("city_code") != "BOS" or base_manifest.get("passed") is not True:
        raise ValueError("Boston base package is not admitted for remediation")
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
        repair = _write_declared_tls_yield_repair(
            source_network,
            temporary_network,
        )
        os.replace(temporary_network, derived_network)
        compatibility = _network_exact_except_declared_tls_changes(
            source_network,
            derived_network,
        )
        hardlinks_exact, unchanged_file_count = _unchanged_files_share_base_inodes(
            base_root,
            staging,
        )
        checks = {
            "base_package_passed": base_manifest.get("passed") is True,
            "declared_tls_changes_exact": compatibility["passed"],
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
                "collision_topology_repair": {
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
        gates["boston_three_to_one_merge_tls_yield_repaired"] = all(
            checks.values()
        )
        payload["gates"] = gates
        payload["passed"] = all(gates.values())
        instantiation = dict(payload.get("microscopic_instantiation", {}))
        instantiation["transformation"] = (
            str(instantiation.get("transformation", ""))
            + "_plus_declared_three_to_one_merge_tls_yield_repair"
        )
        payload["microscopic_instantiation"] = instantiation
        _write_json(staging / "package_manifest.json", payload)
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
    print(json.dumps(payload["collision_topology_repair"], indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
