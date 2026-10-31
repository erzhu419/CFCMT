from copy import deepcopy
from pathlib import Path

import pytest

from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    EXPECTED_TRIP_COUNT,
    TRIGGER_BEGIN_TIME_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v9 import (
    FORMER_FAILURE_TIMES_SEC,
    PACKAGE_PROTOCOL,
    SIGNATURE,
    build_spec,
    validate_trigger_package,
)
from scripts.data.repair_eth_boston_protected_uncontrolled_merge_tls import (
    CONTROLLED_LEFT_CONNECTION,
    DECLARED_CHANGES,
    MERGE_LANE_ID,
    PROTOCOL as REPAIR_PROTOCOL,
    UNCONTROLLED_STRAIGHT_CONNECTION,
)


PACKAGE_SHA256 = "a" * 64


def _manifest() -> dict:
    expected_changes = [
        {
            "phase_index": phase,
            "link_index": link,
            "before": before,
            "after": after,
        }
        for (phase, link), (before, after) in DECLARED_CHANGES.items()
    ]
    return {
        "protocol": PACKAGE_PROTOCOL,
        "passed": True,
        "city_code": "BOS",
        "gates": {
            "base": True,
            "boston_phase2_permissive_merge_removed": True,
            "boston_protected_uncontrolled_merge_yield_repaired": True,
        },
        "permissive_merge_tls_repair": {
            "protocol": "cfcmt-eth-boston-permissive-single-lane-merge-repair-v1",
            "passed": True,
            "repair": {
                "tls_id": "65200956",
                "changes": [{"phase_index": 2, "link_index": 3, "before": "s", "after": "r"}],
            },
            "compatibility_audit": {
                "passed": True,
                "observed_declared_changes": 1,
                "expected_declared_changes": 1,
            },
            "merge_topology": {
                "merge_lane_id": "846300601#2_0",
                "right_turn_connection": {
                    "from": "-592148008#0", "fromLane": "0", "to": "846300601#2", "toLane": "0", "tl": "65200956", "linkIndex": "3", "state": "O"
                },
                "straight_connection": {
                    "from": "846300601#1", "fromLane": "0", "to": "846300601#2", "toLane": "0", "tl": "65200956", "linkIndex": "7", "state": "o"
                },
            },
            "checks": {"base": True},
        },
        "protected_uncontrolled_merge_tls_repair": {
            "protocol": REPAIR_PROTOCOL,
            "passed": True,
            "repair": {"tls_id": "joinedS_1100", "changes": expected_changes},
            "compatibility_audit": {
                "passed": True,
                "observed_declared_changes": 1,
                "expected_declared_changes": 1,
            },
            "merge_topology": {
                "merge_lane_id": MERGE_LANE_ID,
                "controlled_left_connection": CONTROLLED_LEFT_CONNECTION,
                "uncontrolled_straight_connection": UNCONTROLLED_STRAIGHT_CONNECTION,
            },
            "checks": {"base": True},
        },
        "microscopic_instantiation": {
            "backend_required_by_protocol": "libsumo",
            "begin_sec": TRIGGER_BEGIN_TIME_SEC,
            "end_sec": 71_996,
        },
        "microscopic_demand_inventory": {"trip_count": EXPECTED_TRIP_COUNT},
    }


def test_v11_trigger_package_covers_every_observed_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.cluster.launch_eth_boston_trigger_admission_v9."
        "validate_protected_uncontrolled_merge_manifest",
        lambda manifest: None,
    )
    validate_trigger_package(_manifest())
    assert 11_871.0 in FORMER_FAILURE_TIMES_SEC
    assert min(FORMER_FAILURE_TIMES_SEC) >= TRIGGER_BEGIN_TIME_SEC
    assert max(FORMER_FAILURE_TIMES_SEC) <= TRIGGER_STOP_TIME_SEC
    assert TRIGGER_STOP_TIME_SEC == TRIGGER_BEGIN_TIME_SEC + TRIGGER_HORIZON_SEC


def test_v11_trigger_package_requires_new_gate_and_libsumo(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.cluster.launch_eth_boston_trigger_admission_v9."
        "validate_protected_uncontrolled_merge_manifest",
        lambda manifest: None,
    )
    manifest = _manifest()
    manifest["gates"]["boston_protected_uncontrolled_merge_yield_repaired"] = False
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(manifest)
    manifest = _manifest()
    manifest["microscopic_instantiation"]["backend_required_by_protocol"] = "traci"
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(manifest)


def test_v11_trigger_spec_is_full_failure_window_and_excludes_node004() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v11"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/result"),
        local_output_root=Path("/local/result"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == CPU_CORES == 16
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in spec["allowed_nodes"]
    assert f"--operational-horizon-sec {TRIGGER_HORIZON_SEC}" in spec["cmd"]
    assert "--expected-sumo-version 1.22.0" in spec["cmd"]
    assert "traci" not in spec["cmd"].lower()
    assert "TASK_DONE" in spec["cmd"]
