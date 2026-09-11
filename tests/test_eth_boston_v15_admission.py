from pathlib import Path
import subprocess
import sys

import pytest

from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    EXPECTED_TRIP_COUNT,
    TRIGGER_BEGIN_TIME_SEC,
)
from scripts.cluster.launch_eth_boston_v15_admission import (
    FORMER_FAILURE_TIMES_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
    V15_PACKAGE_PROTOCOL,
    build_full_spec,
    build_package_spec,
    build_route_spec,
    build_trigger_spec,
    validate_trigger_package,
    validate_trigger_result,
)


PACKAGE_SHA256 = "b" * 64


def _manifest() -> dict:
    return {
        "protocol": V15_PACKAGE_PROTOCOL,
        "passed": True,
        "city_code": "BOS",
        "gates": {
            "base": True,
            "boston_tls_yellow_clearance_normalized": True,
        },
        "microscopic_instantiation": {
            "backend_required_by_protocol": "libsumo",
            "begin_sec": TRIGGER_BEGIN_TIME_SEC,
            "end_sec": 71_996,
        },
        "microscopic_demand_inventory": {"trip_count": EXPECTED_TRIP_COUNT},
    }


def _trigger_result() -> dict:
    return {
        "mode": "operational_preflight",
        "operational_preflight_passed": True,
        "scientific_admission_passed": False,
        "error": None,
        "checks": {"runtime": True, "collision_free": True},
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


def test_v15_package_and_extended_trigger_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.cluster.launch_eth_boston_v15_admission."
        "validate_tls_yellow_clearance_manifest",
        lambda manifest: None,
    )
    validate_trigger_package(_manifest())
    assert TRIGGER_HORIZON_SEC == 10_800
    assert max(FORMER_FAILURE_TIMES_SEC) == 16_431.0
    assert max(FORMER_FAILURE_TIMES_SEC) < TRIGGER_STOP_TIME_SEC
    invalid = _manifest()
    invalid["gates"]["boston_tls_yellow_clearance_normalized"] = False
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(invalid)


def test_v15_specs_are_version_bound_and_small_result_scoped() -> None:
    common = {
        "snapshot_root": Path("/snapshot"),
        "package_root": Path("/package-v15"),
        "expected_package_manifest_sha256": PACKAGE_SHA256,
        "remote_output_root": Path("/remote/result"),
        "local_output_root": Path("/local/result"),
    }
    package = build_package_spec(
        snapshot_root=Path("/snapshot"),
        base_root=Path("/package-v14"),
        output_root=Path("/package-v15"),
        expected_base_manifest_sha256=PACKAGE_SHA256,
        remote_result_root=Path("/remote/package-evidence"),
        local_result_root=Path("/local/package-evidence"),
    )
    route = build_route_spec(**common)
    trigger = build_trigger_spec(**common)
    full = build_full_spec(**common)
    assert "repair_eth_boston_tls_yellow_clearance" in package["cmd"]
    assert package["result_dir"] == "/remote/package-evidence"
    assert route["ram_resource_family"].endswith("v15")
    assert "--routing-threads 20" in route["cmd"]
    assert trigger["allowed_nodes"] == list(ALLOWED_NODES)
    assert "--operational-horizon-sec 10800" in trigger["cmd"]
    assert full["ram_resource_family"].endswith("v15")
    assert "--operational-horizon-sec" not in full["cmd"]


def test_v15_full_requires_exact_extended_trigger() -> None:
    validate_trigger_result(
        _trigger_result(), expected_package_manifest_sha256=PACKAGE_SHA256
    )
    shortened = _trigger_result()
    shortened["observed"]["final_time_sec"] = 16_431.0
    with pytest.raises(ValueError, match="did not pass"):
        validate_trigger_result(
            shortened, expected_package_manifest_sha256=PACKAGE_SHA256
        )


def test_v15_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/cluster/launch_eth_boston_v15_admission.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
