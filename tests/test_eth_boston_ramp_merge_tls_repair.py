import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.repair_eth_boston_merge_tls import (
    DECLARED_CHANGES as V7_CHANGES,
    TLS_ID as V7_TLS_ID,
    _sha256,
    repair_package as repair_v7_package,
)
from scripts.data.repair_eth_boston_ramp_merge_tls import (
    DECLARED_CHANGES,
    JUNCTION_ID,
    MAINLINE_CONNECTIONS,
    MERGE_EDGE_ID,
    PACKAGE_PROTOCOL,
    RAMP_CONNECTION,
    TLS_ID,
    _network_exact_except_declared_tls_changes,
    repair_package,
    validate_ramp_merge_manifest,
)
from scripts.data.repair_eth_boston_permissive_merge_tls import (
    DECLARED_CHANGES as V10_CHANGES,
    MERGE_EDGE_ID as V10_MERGE_EDGE_ID,
    MERGE_LANE_ID as V10_MERGE_LANE_ID,
    PACKAGE_PROTOCOL as V10_PACKAGE_PROTOCOL,
    RIGHT_TURN_CONNECTION as V10_RIGHT_TURN_CONNECTION,
    STRAIGHT_CONNECTION as V10_STRAIGHT_CONNECTION,
    TLS_ID as V10_TLS_ID,
    repair_package as repair_v10_package,
    validate_permissive_merge_manifest,
)
from scripts.data.repair_eth_boston_protected_uncontrolled_merge_tls import (
    CONTROLLED_LEFT_CONNECTION as V11_CONTROLLED_LEFT_CONNECTION,
    DECLARED_CHANGES as V11_CHANGES,
    JUNCTION_ID as V11_JUNCTION_ID,
    MERGE_EDGE_ID as V11_MERGE_EDGE_ID,
    MERGE_LANE_ID as V11_MERGE_LANE_ID,
    PACKAGE_PROTOCOL as V11_PACKAGE_PROTOCOL,
    TLS_ID as V11_TLS_ID,
    UNCONTROLLED_STRAIGHT_CONNECTION as V11_UNCONTROLLED_STRAIGHT_CONNECTION,
    repair_package as repair_v11_package,
    validate_protected_uncontrolled_merge_manifest,
)
from scripts.data.repair_eth_boston_uncontrolled_merge import (
    BEFORE_STATE,
    CONNECTION_IDENTITY,
    repair_package as repair_v8_package,
)


def _original_package(root: Path) -> str:
    root.mkdir()
    network = ET.Element("net")
    v7_tls = ET.SubElement(
        network,
        "tlLogic",
        {"id": V7_TLS_ID, "type": "actuated", "programID": "0", "offset": "0"},
    )
    for phase_index in range(20):
        state = list("r" * 28)
        for (declared_phase, link_index), (before, _) in V7_CHANGES.items():
            if phase_index == declared_phase:
                state[link_index] = before
        ET.SubElement(v7_tls, "phase", {"duration": "8", "state": "".join(state)})

    ramp_tls = ET.SubElement(
        network,
        "tlLogic",
        {"id": TLS_ID, "type": "actuated", "programID": "0", "offset": "0"},
    )
    for state in ("GGGGGgsrr", "yyyyyysrr", "GrrGGrsrG", "yrryyrsry", "GrrrrrGGr", "yrrrrryyr"):
        ET.SubElement(ramp_tls, "phase", {"duration": "8", "state": state})

    permissive_tls = ET.SubElement(
        network,
        "tlLogic",
        {"id": V10_TLS_ID, "type": "actuated", "programID": "0", "offset": "0"},
    )
    for state in (
        "srrGGgsrrGGg",
        "srryyysrryyy",
        "GGgsrrGGgsrr",
        "yyysrryyysrr",
    ):
        ET.SubElement(permissive_tls, "phase", {"duration": "8", "state": state})

    protected_uncontrolled_tls = ET.SubElement(
        network,
        "tlLogic",
        {"id": V11_TLS_ID, "type": "actuated", "programID": "0", "offset": "0"},
    )
    for state in (
        "Grrrrrrrsrr",
        "yrrrrrrrsrr",
        "rGrrrrrrsrr",
        "ryrrrrrrsrr",
        "rrGGrrrrsrr",
        "rryyrrrrsrr",
        "rrrrGGrrsrr",
        "rrrryyrrsrr",
        "rrrrrrGrsrr",
        "rrrrrryrsrr",
        "rrrrrrrGsrr",
        "rrrrrrrysrr",
        "GGrrGGrrGGG",
        "yyrryyrryyy",
    ):
        ET.SubElement(
            protected_uncontrolled_tls,
            "phase",
            {"duration": "8", "state": state},
        )

    ET.SubElement(
        network,
        "connection",
        {**CONNECTION_IDENTITY, "state": BEFORE_STATE, "dir": "s"},
    )
    merge_edge = ET.SubElement(
        network,
        "edge",
        {"id": MERGE_EDGE_ID, "from": JUNCTION_ID, "to": "downstream"},
    )
    ET.SubElement(merge_edge, "lane", {"id": f"{MERGE_EDGE_ID}_0", "index": "0"})
    ET.SubElement(merge_edge, "lane", {"id": f"{MERGE_EDGE_ID}_1", "index": "1"})
    ET.SubElement(
        network,
        "junction",
        {"id": JUNCTION_ID, "type": "traffic_light_right_on_red"},
    )
    ET.SubElement(network, "connection", {**RAMP_CONNECTION, "dir": "s"})
    for identity in MAINLINE_CONNECTIONS:
        ET.SubElement(network, "connection", {**identity, "dir": "s"})
    permissive_edge = ET.SubElement(
        network,
        "edge",
        {"id": V10_MERGE_EDGE_ID, "from": V10_TLS_ID, "to": "permissive_downstream"},
    )
    ET.SubElement(
        permissive_edge,
        "lane",
        {"id": V10_MERGE_LANE_ID, "index": "0"},
    )
    ET.SubElement(network, "junction", {"id": V10_TLS_ID, "type": "traffic_light"})
    ET.SubElement(network, "connection", {**V10_RIGHT_TURN_CONNECTION, "dir": "r"})
    ET.SubElement(network, "connection", {**V10_STRAIGHT_CONNECTION, "dir": "s"})
    protected_uncontrolled_edge = ET.SubElement(
        network,
        "edge",
        {
            "id": V11_MERGE_EDGE_ID,
            "from": V11_JUNCTION_ID,
            "to": "protected_uncontrolled_downstream",
        },
    )
    ET.SubElement(
        protected_uncontrolled_edge,
        "lane",
        {"id": V11_MERGE_LANE_ID, "index": "0"},
    )
    ET.SubElement(
        network,
        "junction",
        {
            "id": V11_JUNCTION_ID,
            "type": "traffic_light_right_on_red",
            "intLanes": "",
        },
    )
    ET.SubElement(
        network,
        "connection",
        {**V11_CONTROLLED_LEFT_CONNECTION, "dir": "L"},
    )
    ET.SubElement(
        network,
        "connection",
        {**V11_UNCONTROLLED_STRAIGHT_CONNECTION, "dir": "s"},
    )
    ET.SubElement(network, "edge", {"id": "unchanged", "from": "a", "to": "b"})
    ET.ElementTree(network).write(
        root / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    (root / "microscopic_trips.rou.xml").write_text(
        '<routes><trip id="1" depart="0" from="a" to="b"/></routes>\n',
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
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _sha256(manifest_path)


def _v8_package(tmp_path: Path) -> Path:
    original = tmp_path / "original"
    original_sha = _original_package(original)
    v7 = tmp_path / "v7"
    repair_v7_package(
        base_root=original,
        output_root=v7,
        expected_base_manifest_sha256=original_sha,
    )
    v8 = tmp_path / "v8"
    repair_v8_package(
        base_root=v7,
        output_root=v8,
        expected_base_manifest_sha256=_sha256(v7 / "package_manifest.json"),
    )
    return v8


def _ramp_phase_state(path: Path) -> str:
    program = next(
        row
        for row in ET.parse(path).getroot().iter("tlLogic")
        if row.get("id") == TLS_ID
    )
    return program.findall("phase")[2].get("state", "")


def _v9_package(tmp_path: Path) -> Path:
    v8 = _v8_package(tmp_path)
    v9 = tmp_path / "v9"
    repair_package(
        base_root=v8,
        output_root=v9,
        expected_base_manifest_sha256=_sha256(v8 / "package_manifest.json"),
    )
    return v9


def _permissive_phase_state(path: Path) -> str:
    program = next(
        row
        for row in ET.parse(path).getroot().iter("tlLogic")
        if row.get("id") == V10_TLS_ID
    )
    return program.findall("phase")[2].get("state", "")


def _protected_uncontrolled_phase_state(path: Path) -> str:
    program = next(
        row
        for row in ET.parse(path).getroot().iter("tlLogic")
        if row.get("id") == V11_TLS_ID
    )
    return program.findall("phase")[12].get("state", "")


def test_v9_changes_only_declared_ramp_tls_character(tmp_path: Path) -> None:
    v8 = _v8_package(tmp_path)
    v9 = tmp_path / "v9"
    payload = repair_package(
        base_root=v8,
        output_root=v9,
        expected_base_manifest_sha256=_sha256(v8 / "package_manifest.json"),
    )
    assert _ramp_phase_state(v8 / "microscopic_network.net.xml") == "GrrGGrsrG"
    assert _ramp_phase_state(v9 / "microscopic_network.net.xml") == "GrrGGrsrg"
    assert payload["protocol"] == PACKAGE_PROTOCOL
    assert payload["passed"] is True
    assert (v8 / "microscopic_trips.rou.xml").samefile(
        v9 / "microscopic_trips.rou.xml"
    )
    assert _network_exact_except_declared_tls_changes(
        v8 / "microscopic_network.net.xml",
        v9 / "microscopic_network.net.xml",
        tls_id=TLS_ID,
        declared_changes=DECLARED_CHANGES,
    )["passed"] is True
    validate_ramp_merge_manifest(payload)


def test_v9_rejects_changed_ramp_phase(tmp_path: Path) -> None:
    v8 = _v8_package(tmp_path)
    tree = ET.parse(v8 / "microscopic_network.net.xml")
    program = next(row for row in tree.getroot().iter("tlLogic") if row.get("id") == TLS_ID)
    program.findall("phase")[2].set("state", "GrrGGrsrg")
    tree.write(
        v8 / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    with pytest.raises(ValueError, match="state changed before repair"):
        repair_package(
            base_root=v8,
            output_root=tmp_path / "v9",
            expected_base_manifest_sha256=_sha256(v8 / "package_manifest.json"),
        )


def test_v9_rejects_changed_merge_topology(tmp_path: Path) -> None:
    v8 = _v8_package(tmp_path)
    tree = ET.parse(v8 / "microscopic_network.net.xml")
    ramp = next(
        row
        for row in tree.getroot().iter("connection")
        if row.get("from") == RAMP_CONNECTION["from"]
    )
    ramp.set("toLane", "1")
    tree.write(
        v8 / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    with pytest.raises(ValueError, match="ramp connection changed"):
        repair_package(
            base_root=v8,
            output_root=tmp_path / "v9",
            expected_base_manifest_sha256=_sha256(v8 / "package_manifest.json"),
        )


def test_v10_changes_only_collision_observed_permissive_character(
    tmp_path: Path,
) -> None:
    v9 = _v9_package(tmp_path)
    v10 = tmp_path / "v10"
    payload = repair_v10_package(
        base_root=v9,
        output_root=v10,
        expected_base_manifest_sha256=_sha256(v9 / "package_manifest.json"),
    )
    assert _permissive_phase_state(v9 / "microscopic_network.net.xml") == (
        "GGgsrrGGgsrr"
    )
    assert _permissive_phase_state(v10 / "microscopic_network.net.xml") == (
        "GGgrrrGGgsrr"
    )
    assert payload["protocol"] == V10_PACKAGE_PROTOCOL
    assert payload["passed"] is True
    assert (v9 / "microscopic_trips.rou.xml").samefile(
        v10 / "microscopic_trips.rou.xml"
    )
    assert _network_exact_except_declared_tls_changes(
        v9 / "microscopic_network.net.xml",
        v10 / "microscopic_network.net.xml",
        tls_id=V10_TLS_ID,
        declared_changes=V10_CHANGES,
    )["passed"] is True
    validate_permissive_merge_manifest(payload)


def test_v10_rejects_changed_collision_observed_movement(tmp_path: Path) -> None:
    v9 = _v9_package(tmp_path)
    tree = ET.parse(v9 / "microscopic_network.net.xml")
    right_turn = next(
        row
        for row in tree.getroot().iter("connection")
        if row.get("from") == V10_RIGHT_TURN_CONNECTION["from"]
    )
    right_turn.set("toLane", "1")
    tree.write(
        v9 / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    with pytest.raises(ValueError, match="connection changed"):
        repair_v10_package(
            base_root=v9,
            output_root=tmp_path / "v10",
            expected_base_manifest_sha256=_sha256(v9 / "package_manifest.json"),
        )


def test_v11_changes_only_protected_uncontrolled_merge_character(
    tmp_path: Path,
) -> None:
    v9 = _v9_package(tmp_path)
    v10 = tmp_path / "v10"
    repair_v10_package(
        base_root=v9,
        output_root=v10,
        expected_base_manifest_sha256=_sha256(v9 / "package_manifest.json"),
    )
    v11 = tmp_path / "v11"
    payload = repair_v11_package(
        base_root=v10,
        output_root=v11,
        expected_base_manifest_sha256=_sha256(v10 / "package_manifest.json"),
    )
    assert _protected_uncontrolled_phase_state(
        v10 / "microscopic_network.net.xml"
    ) == "GGrrGGrrGGG"
    assert _protected_uncontrolled_phase_state(
        v11 / "microscopic_network.net.xml"
    ) == "GGrrGGrrGGg"
    assert payload["protocol"] == V11_PACKAGE_PROTOCOL
    assert payload["passed"] is True
    assert (v10 / "microscopic_trips.rou.xml").samefile(
        v11 / "microscopic_trips.rou.xml"
    )
    assert _network_exact_except_declared_tls_changes(
        v10 / "microscopic_network.net.xml",
        v11 / "microscopic_network.net.xml",
        tls_id=V11_TLS_ID,
        declared_changes=V11_CHANGES,
    )["passed"] is True
    validate_protected_uncontrolled_merge_manifest(payload)


def test_v11_rejects_changed_uncontrolled_straight_connection(
    tmp_path: Path,
) -> None:
    v9 = _v9_package(tmp_path)
    v10 = tmp_path / "v10"
    repair_v10_package(
        base_root=v9,
        output_root=v10,
        expected_base_manifest_sha256=_sha256(v9 / "package_manifest.json"),
    )
    tree = ET.parse(v10 / "microscopic_network.net.xml")
    uncontrolled = next(
        row
        for row in tree.getroot().iter("connection")
        if row.get("from") == V11_UNCONTROLLED_STRAIGHT_CONNECTION["from"]
        and row.get("to") == V11_MERGE_EDGE_ID
    )
    uncontrolled.set("state", "m")
    tree.write(
        v10 / "microscopic_network.net.xml",
        encoding="utf-8",
        xml_declaration=True,
    )
    with pytest.raises(ValueError, match="connection changed"):
        repair_v11_package(
            base_root=v10,
            output_root=tmp_path / "v11",
            expected_base_manifest_sha256=_sha256(v10 / "package_manifest.json"),
        )
