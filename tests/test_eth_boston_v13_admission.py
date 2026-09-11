from pathlib import Path
import subprocess
import sys

import pytest

from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    EXPECTED_TRIP_COUNT,
    TRIGGER_BEGIN_TIME_SEC,
)
from scripts.cluster.launch_eth_boston_v13_admission import (
    FORMER_FAILURE_TIMES_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
    V13_PACKAGE_PROTOCOL,
    build_full_spec,
    build_package_spec,
    build_route_spec,
    build_trigger_spec,
    validate_trigger_package,
    validate_trigger_result,
)


PACKAGE_SHA256 = "a" * 64


def _manifest() -> dict:
    return {
        "protocol": V13_PACKAGE_PROTOCOL,
        "passed": True,
        "city_code": "BOS",
        "gates": {
            "base": True,
            "boston_systemic_controlled_major_right_of_way_repaired": True,
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


def test_v13_package_and_trigger_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.cluster.launch_eth_boston_v13_admission."
        "validate_systemic_right_of_way_manifest",
        lambda manifest: None,
    )
    validate_trigger_package(_manifest())
    assert 10_019.0 in FORMER_FAILURE_TIMES_SEC
    assert max(FORMER_FAILURE_TIMES_SEC) < TRIGGER_STOP_TIME_SEC
    invalid = _manifest()
    invalid["gates"][
        "boston_systemic_controlled_major_right_of_way_repaired"
    ] = False
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(invalid)


def test_v13_stage_specs_are_version_bound_and_small_result_scoped() -> None:
    common = {
        "snapshot_root": Path("/snapshot"),
        "package_root": Path("/package-v13"),
        "expected_package_manifest_sha256": PACKAGE_SHA256,
        "remote_output_root": Path("/remote/result"),
        "local_output_root": Path("/local/result"),
    }
    package = build_package_spec(
        snapshot_root=Path("/snapshot"),
        base_root=Path("/package-v12"),
        output_root=Path("/package-v13"),
        expected_base_manifest_sha256=PACKAGE_SHA256,
        remote_result_root=Path("/remote/package-evidence"),
        local_result_root=Path("/local/package-evidence"),
    )
    route = build_route_spec(**common)
    trigger = build_trigger_spec(**common)
    full = build_full_spec(**common)
    assert "repair_eth_boston_systemic_right_of_way" in package["cmd"]
    assert package["result_dir"] == "/remote/package-evidence"
    assert package["local_result_dir"] == "/local/package-evidence"
    assert route["ram_resource_family"].endswith("v13")
    assert "--routing-threads 20" in route["cmd"]
    assert trigger["allowed_nodes"] == list(ALLOWED_NODES)
    assert "--operational-horizon-sec 7200" in trigger["cmd"]
    assert full["ram_resource_family"].endswith("v13")
    assert "--operational-horizon-sec" not in full["cmd"]


def test_v13_full_requires_exact_extended_trigger() -> None:
    validate_trigger_result(
        _trigger_result(), expected_package_manifest_sha256=PACKAGE_SHA256
    )
    shortened = _trigger_result()
    shortened["observed"]["final_time_sec"] = 10_000.0
    with pytest.raises(ValueError, match="did not pass"):
        validate_trigger_result(
            shortened, expected_package_manifest_sha256=PACKAGE_SHA256
        )


def test_v13_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/cluster/launch_eth_boston_v13_admission.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
