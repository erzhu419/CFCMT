import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.repair_eth_boston_merge_tls import (
    DECLARED_CHANGES,
    TLS_ID,
    repair_package as repair_v7_package,
)
from scripts.data.repair_eth_boston_uncontrolled_merge import (
    AFTER_STATE,
    BEFORE_STATE,
    CONNECTION_IDENTITY,
    PACKAGE_PROTOCOL,
    _network_exact_except_declared_connection,
    repair_package,
    validate_uncontrolled_merge_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _original_package(root: Path) -> str:
    root.mkdir()
    network = ET.Element("net")
    tls = ET.SubElement(
        network,
        "tlLogic",
        {"id": TLS_ID, "type": "actuated", "programID": "0", "offset": "0"},
    )
    for phase_index in range(20):
        state = list("r" * 28)
        for (declared_phase, link_index), (before, _) in DECLARED_CHANGES.items():
            if phase_index == declared_phase:
                state[link_index] = before
        ET.SubElement(tls, "phase", {"duration": "8", "state": "".join(state)})
    ET.SubElement(
        network,
        "connection",
        {**CONNECTION_IDENTITY, "state": BEFORE_STATE, "dir": "s"},
    )
    ET.SubElement(network, "edge", {"id": "unchanged", "from": "a", "to": "b"})
    ET.ElementTree(network).write(
        root / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
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
    path = root / "package_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _v7_package(tmp_path: Path) -> Path:
    original = tmp_path / "original"
    original_sha = _original_package(original)
    v7 = tmp_path / "v7"
    repair_v7_package(
        base_root=original,
        output_root=v7,
        expected_base_manifest_sha256=original_sha,
    )
    return v7


def _declared_state(path: Path) -> str:
    matches = [
        row
        for row in ET.parse(path).getroot().iter("connection")
        if all(row.attrib.get(key) == value for key, value in CONNECTION_IDENTITY.items())
    ]
    assert len(matches) == 1
    return matches[0].attrib["state"]


def test_v8_changes_only_declared_uncontrolled_merge_state(tmp_path: Path) -> None:
    v7 = _v7_package(tmp_path)
    v8 = tmp_path / "v8"
    payload = repair_package(
        base_root=v7,
        output_root=v8,
        expected_base_manifest_sha256=_sha256(v7 / "package_manifest.json"),
    )
    assert _declared_state(v7 / "microscopic_network.net.xml") == BEFORE_STATE
    assert _declared_state(v8 / "microscopic_network.net.xml") == AFTER_STATE
    assert payload["protocol"] == PACKAGE_PROTOCOL
    assert payload["passed"] is True
    assert (v7 / "microscopic_trips.rou.xml").samefile(
        v8 / "microscopic_trips.rou.xml"
    )
    assert _network_exact_except_declared_connection(
        v7 / "microscopic_network.net.xml",
        v8 / "microscopic_network.net.xml",
    )["passed"] is True
    validate_uncontrolled_merge_manifest(payload)


def test_v8_rejects_changed_declared_connection(tmp_path: Path) -> None:
    v7 = _v7_package(tmp_path)
    tree = ET.parse(v7 / "microscopic_network.net.xml")
    connection = next(tree.getroot().iter("connection"))
    connection.set("state", "m")
    tree.write(
        v7 / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    with pytest.raises(ValueError, match="state changed before repair"):
        repair_package(
            base_root=v7,
            output_root=tmp_path / "v8",
            expected_base_manifest_sha256=_sha256(v7 / "package_manifest.json"),
        )
