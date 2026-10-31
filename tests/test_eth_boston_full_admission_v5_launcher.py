from pathlib import Path

import pytest

from scripts.cluster.launch_eth_boston_full_admission_v4 import RAM_MB
from scripts.cluster.launch_eth_boston_full_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    SIGNATURE,
    TRIGGER_BEGIN_TIME_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
    build_spec,
    validate_trigger_result,
)


PACKAGE_SHA256 = "a" * 64


def _trigger() -> dict:
    return {
        "mode": "operational_preflight",
        "operational_preflight_passed": True,
        "scientific_admission_passed": False,
        "error": None,
        "checks": {
            "runtime_exception_free": True,
            "operational_horizon_reached": True,
            "zero_collisions_during_preflight": True,
            "zero_starting_and_ending_teleports_during_preflight": True,
        },
        "setting": {
            "package_manifest_sha256": PACKAGE_SHA256,
            "city_code": "BOS",
            "backend": "libsumo",
            "expected_sumo_version": "1.22.0",
            "begin_time_sec": TRIGGER_BEGIN_TIME_SEC,
            "operational_horizon_sec": TRIGGER_HORIZON_SEC,
            "stop_time_sec": TRIGGER_STOP_TIME_SEC,
        },
        "observed": {"final_time_sec": TRIGGER_STOP_TIME_SEC},
    }


def test_v7_trigger_gate_crosses_collision_time() -> None:
    validate_trigger_result(
        _trigger(),
        expected_package_manifest_sha256=PACKAGE_SHA256,
    )
    failed = _trigger()
    failed["observed"]["final_time_sec"] = 12_000.0
    with pytest.raises(ValueError, match="execution contract"):
        validate_trigger_result(
            failed,
            expected_package_manifest_sha256=PACKAGE_SHA256,
        )


def test_v7_full_spec_keeps_complete_admission_contract() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v7"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/result"),
        local_output_root=Path("/local/result"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == CPU_CORES == 16
    assert spec["ram_mb"] == RAM_MB == 24_576
    assert spec["ram_resource_family"].endswith("v7-complete")
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in spec["allowed_nodes"]
    assert "--operational-horizon-sec" not in spec["cmd"]
