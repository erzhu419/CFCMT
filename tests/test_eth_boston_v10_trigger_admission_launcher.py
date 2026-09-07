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
from scripts.cluster.launch_eth_boston_trigger_admission_v8 import (
    FORMER_FAILURE_TIMES_SEC,
    PACKAGE_PROTOCOL,
    SIGNATURE,
    build_spec,
    validate_trigger_package,
)


PACKAGE_SHA256 = "a" * 64


def _manifest() -> dict:
    return {
        "protocol": PACKAGE_PROTOCOL,
        "passed": True,
        "city_code": "BOS",
        "gates": {"boston_phase2_permissive_merge_removed": True},
        "microscopic_instantiation": {
            "backend_required_by_protocol": "libsumo",
            "begin_sec": TRIGGER_BEGIN_TIME_SEC,
            "end_sec": 71_996,
        },
        "microscopic_demand_inventory": {"trip_count": EXPECTED_TRIP_COUNT},
    }


def test_v10_trigger_package_covers_every_observed_failure() -> None:
    validate_trigger_package(_manifest())
    assert 9_735.0 in FORMER_FAILURE_TIMES_SEC
    assert min(FORMER_FAILURE_TIMES_SEC) >= TRIGGER_BEGIN_TIME_SEC
    assert max(FORMER_FAILURE_TIMES_SEC) <= TRIGGER_STOP_TIME_SEC
    assert TRIGGER_STOP_TIME_SEC == TRIGGER_BEGIN_TIME_SEC + TRIGGER_HORIZON_SEC


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol", "cfcmt-eth-boston-complete-published-package-v9"),
        ("passed", False),
        ("city_code", "BER"),
    ],
)
def test_v10_trigger_package_rejects_wrong_identity(field: str, value: object) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(manifest)


def test_v10_trigger_package_requires_repair_gate_and_libsumo() -> None:
    manifest = _manifest()
    manifest["gates"]["boston_phase2_permissive_merge_removed"] = False
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(manifest)
    manifest = _manifest()
    manifest["microscopic_instantiation"]["backend_required_by_protocol"] = "traci"
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(manifest)


def test_v10_trigger_spec_is_full_failure_window_and_excludes_node004() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v10"),
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
