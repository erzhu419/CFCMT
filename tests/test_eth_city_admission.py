from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cf_h2o.eval.traffic_signal_eth_city_admission as city_admission
from cf_h2o.eval.traffic_signal_eth_city_admission import (
    _full_admission_checks,
    _operational_preflight_checks,
    _sha256,
    _validate_package,
)
from scripts.data.package_eth_city_sumo import (
    MICRO_CONFIG_PROTOCOL,
    PROTOCOL as PACKAGE_PROTOCOL,
)


def _package(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "package"
    root.mkdir()
    (root / "microscopic.sumo.cfg").write_text(
        "<configuration/>", encoding="utf-8"
    )
    manifest = {
        "protocol": PACKAGE_PROTOCOL,
        "city_code": "BOS",
        "passed": True,
        "gates": {"one": True, "two": True},
        "demand_inventory": {"trip_count": 10},
        "network_inventory": {"traffic_light_count": 2},
        "microscopic_instantiation": {
            "protocol": MICRO_CONFIG_PROTOCOL,
            "config": "microscopic.sumo.cfg",
            "mode": "microscopic",
            "backend_required_by_protocol": "libsumo",
            "step_length_sec": 1,
            "time_to_teleport_sec": -1,
            "collision_check_junctions": True,
            "collision_action": "warn",
            "ignore_route_errors": False,
            "max_depart_delay_sec": -1,
        },
    }
    manifest_path = root / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, _sha256(manifest_path)


def test_validate_eth_city_package_holds_manifest_and_strict_contract(
    tmp_path: Path,
) -> None:
    root, digest = _package(tmp_path)
    config, manifest, observed_digest = _validate_package(
        root,
        expected_manifest_sha256=digest,
        expected_city_code="BOS",
    )
    assert config == (root / "microscopic.sumo.cfg").resolve()
    assert manifest["city_code"] == "BOS"
    assert observed_digest == digest
    with pytest.raises(ValueError, match="package city changed"):
        _validate_package(
            root,
            expected_manifest_sha256=digest,
            expected_city_code="LIS",
        )


def test_validate_eth_city_package_dispatches_boston_v9_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    root, _ = _package(tmp_path)
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = city_admission.BOSTON_V9_PACKAGE_PROTOCOL
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    observed = []
    monkeypatch.setattr(
        city_admission,
        "validate_ramp_merge_manifest",
        lambda payload: observed.append(payload["protocol"]),
    )
    digest = _sha256(manifest_path)
    _validate_package(
        root,
        expected_manifest_sha256=digest,
        expected_city_code="BOS",
    )
    assert observed == [city_admission.BOSTON_V9_PACKAGE_PROTOCOL]


def test_validate_eth_city_package_dispatches_boston_v10_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    root, _ = _package(tmp_path)
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = city_admission.BOSTON_V10_PACKAGE_PROTOCOL
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    observed = []
    monkeypatch.setattr(
        city_admission,
        "validate_permissive_merge_manifest",
        lambda payload: observed.append(payload["protocol"]),
    )
    digest = _sha256(manifest_path)
    _validate_package(
        root,
        expected_manifest_sha256=digest,
        expected_city_code="BOS",
    )
    assert observed == [city_admission.BOSTON_V10_PACKAGE_PROTOCOL]


def test_validate_eth_city_package_dispatches_boston_v11_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    root, _ = _package(tmp_path)
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = city_admission.BOSTON_V11_PACKAGE_PROTOCOL
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    observed = []
    monkeypatch.setattr(
        city_admission,
        "validate_protected_uncontrolled_merge_manifest",
        lambda payload: observed.append(payload["protocol"]),
    )
    digest = _sha256(manifest_path)
    _validate_package(
        root,
        expected_manifest_sha256=digest,
        expected_city_code="BOS",
    )
    assert observed == [city_admission.BOSTON_V11_PACKAGE_PROTOCOL]


def test_full_admission_checks_require_exact_conservation_and_safety() -> None:
    observed = {
        "termination_reason": "all_vehicles_completed",
        "departed_vehicles": 10,
        "arrived_vehicles": 10,
        "final_active_vehicles": 0,
        "final_pending_vehicles": 0,
        "final_min_expected": 0,
        "unique_collision_incidents": 0,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "controllable_tls_count": 2,
    }
    summary = {
        "loaded": 10,
        "arrived": 10,
        "running": 0,
        "waiting": 0,
        "collisions": 0,
        "teleports": 0,
    }
    checks = _full_admission_checks(
        expected_vehicle_count=10,
        expected_traffic_lights=2,
        observed=observed,
        summary=summary,
        error=None,
    )
    assert all(checks.values())
    observed["arrived_vehicles"] = 9
    assert not _full_admission_checks(
        expected_vehicle_count=10,
        expected_traffic_lights=2,
        observed=observed,
        summary=summary,
        error=None,
    )["exact_vehicle_demand_arrived"]


def test_operational_preflight_cannot_stand_in_for_full_conservation() -> None:
    checks = _operational_preflight_checks(
        expected_traffic_lights=2,
        observed={
            "termination_reason": "operational_horizon_reached",
            "unique_collision_incidents": 0,
            "starting_teleports": 0,
            "ending_teleports": 0,
            "controllable_tls_count": 2,
        },
        error=None,
    )
    assert all(checks.values())
    assert "exact_vehicle_demand_arrived" not in checks


def test_collision_sample_records_route_and_live_signal_state(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="out" from="junction" to="downstream"><lane id="out_0" index="0"/></edge>
  <connection from="approach" fromLane="0" to="out" toLane="0" tl="junction" linkIndex="3" state="O"/>
</net>""",
        encoding="utf-8",
    )

    class Vehicle:
        @staticmethod
        def getIDList():
            return ("collider", "victim")

        @staticmethod
        def getRoute(_vehicle_id):
            return ("approach", "out", "next")

        @staticmethod
        def getRouteIndex(_vehicle_id):
            return 1

        @staticmethod
        def getRoadID(_vehicle_id):
            return "out"

        @staticmethod
        def getLaneID(_vehicle_id):
            return "out_0"

        @staticmethod
        def getLanePosition(_vehicle_id):
            return 0.7

        @staticmethod
        def getSpeed(_vehicle_id):
            return 4.0

    class TrafficLight:
        @staticmethod
        def getProgram(_tls_id):
            return "0"

        @staticmethod
        def getPhase(_tls_id):
            return 2

        @staticmethod
        def getRedYellowGreenState(_tls_id):
            return "rrrG"

        @staticmethod
        def getNextSwitch(_tls_id):
            return 98.0

    sample = city_admission._collision_sample(
        SimpleNamespace(
            collider="collider",
            victim="victim",
            type="collision",
            lane="out_0",
            pos=0.7,
        ),
        time_sec=97.0,
        sumo_api=SimpleNamespace(vehicle=Vehicle(), trafficlight=TrafficLight()),
        network_path=network,
        topology_cache={},
    )
    assert sample["participants"][0]["previous_route_edge"] == "approach"
    assert sample["participants"][0]["current_route_edge"] == "out"
    assert sample["lane_topology"]["incoming_connections"][0]["linkIndex"] == "3"
    assert sample["traffic_light_state"] == [
        {
            "traffic_light_id": "junction",
            "program_id": "0",
            "phase_index": 2,
            "state": "rrrG",
            "next_switch_sec": 98.0,
        }
    ]
