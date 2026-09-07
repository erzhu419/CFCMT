from pathlib import Path

import pytest

from scripts.cluster.launch_eth_boston_full_admission_v4 import (
    ALLOWED_NODES,
    CPU_CORES,
    SIGNATURE,
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
        },
        "setting": {
            "package_manifest_sha256": PACKAGE_SHA256,
            "city_code": "BOS",
            "backend": "libsumo",
            "expected_sumo_version": "1.22.0",
            "begin_time_sec": 7_201.0,
            "operational_horizon_sec": 3_000,
            "stop_time_sec": 10_201.0,
        },
        "observed": {"final_time_sec": 10_201.0},
    }


def test_trigger_gate_requires_exact_execution_contract() -> None:
    validate_trigger_result(
        _trigger(), expected_package_manifest_sha256=PACKAGE_SHA256
    )
    failed = _trigger()
    failed["setting"]["operational_horizon_sec"] = 2_400
    with pytest.raises(ValueError, match="execution contract"):
        validate_trigger_result(
            failed, expected_package_manifest_sha256=PACKAGE_SHA256
        )


def test_full_admission_spec_uses_rerouting_cpu_budget_and_node_allowlist() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/result"),
        local_output_root=Path("/local/result"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == CPU_CORES == 16
    assert spec["ram_mb"] == 24_576
    assert spec["ram_resource_family"].endswith("v6-complete")
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in spec["allowed_nodes"]
    assert "--operational-horizon-sec" not in spec["cmd"]
    assert "--expected-package-manifest-sha256 " + PACKAGE_SHA256 in spec["cmd"]
