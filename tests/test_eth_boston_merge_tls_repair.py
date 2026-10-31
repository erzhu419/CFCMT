import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.repair_eth_boston_merge_tls import (
    DECLARED_CHANGES,
    PACKAGE_PROTOCOL,
    TLS_ID,
    _network_exact_except_declared_tls_changes,
    repair_package,
    validate_remediated_package_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _network(path: Path) -> None:
    root = ET.Element("net")
    tls = ET.SubElement(
        root,
        "tlLogic",
        {"id": TLS_ID, "type": "actuated", "programID": "0", "offset": "0"},
    )
    for phase_index in range(20):
        state = list("r" * 28)
        for (declared_phase, link_index), (before, _) in DECLARED_CHANGES.items():
            if phase_index == declared_phase:
                state[link_index] = before
        ET.SubElement(tls, "phase", {"duration": "8", "state": "".join(state)})
    ET.SubElement(root, "edge", {"id": "edge", "from": "a", "to": "b"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _base_package(root: Path) -> str:
    root.mkdir()
    _network(root / "microscopic_network.net.xml")
    (root / "microscopic_trips.rou.xml").write_text(
        "<routes><trip id=\"1\" depart=\"0\" from=\"a\" to=\"b\"/></routes>\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": "base",
        "city_code": "BOS",
        "passed": True,
        "gates": {"base": True},
        "microscopic_instantiation": {"transformation": "base"},
    }
    manifest_path = root / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha256(manifest_path)


def _states(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    tls = next(row for row in root.iter("tlLogic") if row.attrib["id"] == TLS_ID)
    return [str(phase.attrib["state"]) for phase in tls.findall("phase")]


def test_package_repair_changes_only_declared_tls_characters(tmp_path: Path) -> None:
    base = tmp_path / "base"
    expected_sha256 = _base_package(base)
    derived = tmp_path / "derived"

    payload = repair_package(
        base_root=base,
        output_root=derived,
        expected_base_manifest_sha256=expected_sha256,
    )

    base_states = _states(base / "microscopic_network.net.xml")
    derived_states = _states(derived / "microscopic_network.net.xml")
    for (phase_index, link_index), (before, after) in DECLARED_CHANGES.items():
        assert base_states[phase_index][link_index] == before
        assert derived_states[phase_index][link_index] == after
    assert payload["protocol"] == PACKAGE_PROTOCOL
    assert payload["passed"] is True
    assert payload["collision_topology_repair"]["passed"] is True
    assert payload["collision_topology_repair"]["unchanged_file_count"] == 1
    assert (base / "microscopic_trips.rou.xml").samefile(
        derived / "microscopic_trips.rou.xml"
    )
    assert _network_exact_except_declared_tls_changes(
        base / "microscopic_network.net.xml",
        derived / "microscopic_network.net.xml",
    )["passed"] is True
    validate_remediated_package_manifest(payload)

    incomplete = dict(payload)
    incomplete["collision_topology_repair"] = dict(
        payload["collision_topology_repair"]
    )
    incomplete["collision_topology_repair"]["repair"] = dict(
        payload["collision_topology_repair"]["repair"]
    )
    incomplete["collision_topology_repair"]["repair"][
        "changed_phase_character_count"
    ] = 1
    with pytest.raises(ValueError, match="manifest is incomplete"):
        validate_remediated_package_manifest(incomplete)


def test_package_repair_rejects_changed_declared_state(tmp_path: Path) -> None:
    base = tmp_path / "base"
    expected_sha256 = _base_package(base)
    tree = ET.parse(base / "microscopic_network.net.xml")
    phases = next(tree.getroot().iter("tlLogic")).findall("phase")
    state = phases[10].attrib["state"]
    phases[10].set("state", state[:12] + "r" + state[13:])
    tree.write(
        base / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )

    with pytest.raises(ValueError, match="declared TLS state changed"):
        repair_package(
            base_root=base,
            output_root=tmp_path / "derived",
            expected_base_manifest_sha256=expected_sha256,
        )
    assert not (tmp_path / "derived").exists()


def test_exact_audit_rejects_unrelated_network_change(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _base_package(base)
    repaired = tmp_path / "repaired.net.xml"
    tree = ET.parse(base / "microscopic_network.net.xml")
    tls = next(tree.getroot().iter("tlLogic"))
    for (phase_index, link_index), (_, after) in DECLARED_CHANGES.items():
        phase = tls.findall("phase")[phase_index]
        state = phase.attrib["state"]
        phase.set("state", state[:link_index] + after + state[link_index + 1 :])
    next(tree.getroot().iter("edge")).set("to", "changed")
    tree.write(repaired, encoding="utf-8", xml_declaration=True)

    audit = _network_exact_except_declared_tls_changes(
        base / "microscopic_network.net.xml",
        repaired,
    )
    assert audit["passed"] is False
