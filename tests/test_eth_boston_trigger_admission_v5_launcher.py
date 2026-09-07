from pathlib import Path

import pytest

from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    EXPECTED_TRIP_COUNT,
    SIGNATURE,
    TRIGGER_HORIZON_SEC,
    build_spec,
    validate_route_result,
)


PACKAGE_SHA256 = "a" * 64


def _route_result() -> dict:
    return {
        "passed": True,
        "decision": "authorize_microscopic_operational_preflight",
        "city_code": "BOS",
        "package_manifest_sha256": PACKAGE_SHA256,
        "trip_count": EXPECTED_TRIP_COUNT,
        "duarouter": {"returncode": 0, "ignore_errors": False},
    }


def test_route_gate_requires_complete_v7_demand() -> None:
    validate_route_result(
        _route_result(),
        expected_package_manifest_sha256=PACKAGE_SHA256,
    )
    incomplete = _route_result()
    incomplete["trip_count"] -= 1
    with pytest.raises(ValueError, match="did not pass"):
        validate_route_result(
            incomplete,
            expected_package_manifest_sha256=PACKAGE_SHA256,
        )


def test_trigger_spec_crosses_both_prior_failures() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v7"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/result"),
        local_output_root=Path("/local/result"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == CPU_CORES == 16
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in spec["allowed_nodes"]
    assert f"--operational-horizon-sec {TRIGGER_HORIZON_SEC}" in spec["cmd"]
    assert "--expected-package-manifest-sha256 " + PACKAGE_SHA256 in spec["cmd"]
