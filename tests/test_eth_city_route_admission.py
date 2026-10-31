from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import cf_h2o.eval.traffic_signal_eth_city_route_admission as route_admission
from cf_h2o.eval.traffic_signal_eth_city_route_admission import (
    build_duarouter_command,
    validate_package,
)
from scripts.data.package_eth_city_sumo import (
    PROTOCOL as PACKAGE_PROTOCOL,
    ROUTE_CANONICALIZATION_PROTOCOL,
)


def _write_package(root: Path, *, trip_count: int = 2) -> tuple[Path, str]:
    (root / "source/BOS").mkdir(parents=True)
    for relative in (
        "source/BOS/BOS.net.xml",
        "source/BOS/taz.xml",
        "microscopic_network.net.xml",
        "microscopic_inputs.add.xml",
        "microscopic_trips.rou.xml",
    ):
        (root / relative).write_text("<x/>\n", encoding="utf-8")
    config = ET.Element("configuration")
    inputs = ET.SubElement(config, "input")
    ET.SubElement(inputs, "net-file", value="microscopic_network.net.xml")
    ET.SubElement(inputs, "route-files", value="microscopic_trips.rou.xml")
    ET.SubElement(
        inputs,
        "additional-files",
        value="microscopic_inputs.add.xml,source/BOS/taz.xml",
    )
    processing = ET.SubElement(config, "processing")
    ET.SubElement(processing, "ignore-route-errors", value="false")
    ET.ElementTree(config).write(root / "microscopic.sumo.cfg", encoding="utf-8")
    manifest = {
        "protocol": PACKAGE_PROTOCOL,
        "city_code": "BOS",
        "passed": True,
        "gates": {"static": True},
        "demand_inventory": {
            "trip_count": trip_count,
            "unique_trip_id_count": trip_count,
        },
        "microscopic_demand_inventory": {
            "trip_count": trip_count,
            "unique_trip_id_count": trip_count,
            "custom_trip_to_taz_attribute_count": 0,
            "canonical_to_taz_attribute_count": trip_count,
        },
        "destination_taz_schema_canonicalization": {
            "protocol": ROUTE_CANONICALIZATION_PROTOCOL,
            "renamed_trip_to_taz_count": trip_count,
            "trip_anchor_edges_changed": 0,
            "trips_removed": 0,
            "route_repair_applied": False,
        },
        "microscopic_instantiation": {
            "config": "microscopic.sumo.cfg",
            "network_file": "microscopic_network.net.xml",
            "route_file": "microscopic_trips.rou.xml",
        },
    }
    manifest_path = root / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, sha


def test_validate_package_requires_complete_uniform_canonicalization(
    tmp_path: Path,
) -> None:
    _, sha = _write_package(tmp_path)
    result = validate_package(
        tmp_path,
        expected_manifest_sha256=sha,
        expected_city_code="BOS",
    )
    assert result["trip_count"] == 2
    assert result["network"] == tmp_path / "microscopic_network.net.xml"
    assert result["routes"] == tmp_path / "microscopic_trips.rou.xml"


def test_validate_package_dispatches_boston_v9_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V9_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        route_admission,
        "validate_ramp_merge_manifest",
        lambda payload: observed.append(payload["protocol"]),
    )
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    validate_package(
        tmp_path,
        expected_manifest_sha256=sha,
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V9_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v10_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V10_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        route_admission,
        "validate_permissive_merge_manifest",
        lambda payload: observed.append(payload["protocol"]),
    )
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    validate_package(
        tmp_path,
        expected_manifest_sha256=sha,
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V10_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v11_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V11_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        route_admission,
        "validate_protected_uncontrolled_merge_manifest",
        lambda payload: observed.append(payload["protocol"]),
    )
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    validate_package(
        tmp_path,
        expected_manifest_sha256=sha,
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V11_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v12_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V12_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed: list[str] = []
    monkeypatch.setattr(
        route_admission,
        "validate_joined_tls_uncontrolled_merge_manifest",
        lambda value: observed.append(str(value["protocol"])),
    )
    validate_package(
        tmp_path,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V12_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v13_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V13_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed: list[str] = []
    monkeypatch.setattr(
        route_admission,
        "validate_systemic_right_of_way_manifest",
        lambda value: observed.append(str(value["protocol"])),
    )
    validate_package(
        tmp_path,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V13_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v14_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V14_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed: list[str] = []
    monkeypatch.setattr(
        route_admission,
        "validate_systemic_uncontrolled_major_manifest",
        lambda value: observed.append(str(value["protocol"])),
    )
    validate_package(
        tmp_path,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V14_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v15_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V15_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed: list[str] = []
    monkeypatch.setattr(
        route_admission,
        "validate_tls_yellow_clearance_manifest",
        lambda value: observed.append(str(value["protocol"])),
    )
    validate_package(
        tmp_path,
        expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_city_code="BOS",
    )
    assert observed == [route_admission.BOSTON_V15_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v16_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V16_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        route_admission,
        "validate_implicit_no_tls_major_manifest",
        lambda value: observed.append(value["protocol"]),
    )

    route_admission.validate_package(
        tmp_path,
        expected_manifest_sha256=route_admission._sha256(manifest_path),
        expected_city_code="BOS",
    )

    assert observed == [route_admission.BOSTON_V16_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v17_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V17_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        route_admission,
        "validate_no_tls_major_response_manifest",
        lambda value: observed.append(value["protocol"]),
    )

    route_admission.validate_package(
        tmp_path,
        expected_manifest_sha256=route_admission._sha256(manifest_path),
        expected_city_code="BOS",
    )

    assert observed == [route_admission.BOSTON_V17_PACKAGE_PROTOCOL]


def test_validate_package_dispatches_boston_v18_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = route_admission.BOSTON_V18_PACKAGE_PROTOCOL
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        route_admission,
        "validate_controlled_shared_receiving_yield_manifest",
        lambda value: observed.append(value["protocol"]),
    )

    route_admission.validate_package(
        tmp_path,
        expected_manifest_sha256=route_admission._sha256(manifest_path),
        expected_city_code="BOS",
    )

    assert observed == [route_admission.BOSTON_V18_PACKAGE_PROTOCOL]


def test_validate_package_rejects_config_network_mismatch(tmp_path: Path) -> None:
    manifest_path, _ = _write_package(tmp_path)
    config_path = tmp_path / "microscopic.sumo.cfg"
    config = ET.parse(config_path)
    config.getroot().find("./input/net-file").set(
        "value", "source/BOS/BOS.net.xml"
    )
    config.write(config_path, encoding="utf-8")
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="input boundary changed"):
        validate_package(
            tmp_path,
            expected_manifest_sha256=sha,
            expected_city_code="BOS",
        )


def test_validate_package_rejects_trip_loss(tmp_path: Path) -> None:
    manifest_path, _ = _write_package(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["microscopic_demand_inventory"]["trip_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonicalization is incomplete"):
        validate_package(
            tmp_path,
            expected_manifest_sha256=sha,
            expected_city_code="BOS",
        )


def test_duarouter_command_is_full_demand_fail_closed(tmp_path: Path) -> None:
    command = build_duarouter_command(
        binary=Path("/sumo/bin/duarouter"),
        network=tmp_path / "net.xml",
        routes=tmp_path / "routes.xml",
        taz=tmp_path / "taz.xml",
        error_log=tmp_path / "errors.log",
        routing_threads=20,
    )
    assert command[command.index("--route-files") + 1].endswith("routes.xml")
    assert command[command.index("--with-taz") + 1] == "true"
    assert command[command.index("--ignore-errors") + 1] == "false"
    assert command[command.index("--routing-algorithm") + 1] == "CH"
    assert command[command.index("--routing-threads") + 1] == "20"
    assert "--bulk-routing" not in command
    assert command[command.index("--output-file") + 1] == "/dev/null"
