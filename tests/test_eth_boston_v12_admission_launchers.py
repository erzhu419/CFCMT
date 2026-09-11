from pathlib import Path
import subprocess
import sys

import pytest

from scripts.cluster.launch_eth_boston_full_admission_v9 import (
    SIGNATURE as FULL_SIGNATURE,
    build_spec as build_full_spec,
    validate_trigger_result,
)
from scripts.cluster.launch_eth_boston_route_admission_v10 import (
    SIGNATURE as ROUTE_SIGNATURE,
    build_spec as build_route_spec,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    ALLOWED_NODES,
    EXPECTED_TRIP_COUNT,
    TRIGGER_BEGIN_TIME_SEC,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v10 import (
    FORMER_FAILURE_TIMES_SEC,
    PACKAGE_PROTOCOL,
    SIGNATURE as TRIGGER_SIGNATURE,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
    build_spec as build_trigger_spec,
    validate_trigger_package,
)


PACKAGE_SHA256 = "a" * 64


def _manifest() -> dict:
    return {
        "protocol": PACKAGE_PROTOCOL,
        "passed": True,
        "city_code": "BOS",
        "gates": {
            "base": True,
            "boston_joined_tls_uncontrolled_merge_yield_repaired": True,
        },
        "microscopic_instantiation": {
            "backend_required_by_protocol": "libsumo",
            "begin_sec": TRIGGER_BEGIN_TIME_SEC,
            "end_sec": 71_996,
        },
        "microscopic_demand_inventory": {"trip_count": EXPECTED_TRIP_COUNT},
    }


def _passing_trigger() -> dict:
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
            "exact_controllable_tls": True,
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


def test_v12_trigger_covers_new_failure_and_requires_new_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.cluster.launch_eth_boston_trigger_admission_v10."
        "validate_joined_tls_uncontrolled_merge_manifest",
        lambda manifest: None,
    )
    validate_trigger_package(_manifest())
    assert TRIGGER_HORIZON_SEC == 7_200
    assert TRIGGER_STOP_TIME_SEC == 14_401.0
    assert max(FORMER_FAILURE_TIMES_SEC) == 13_146.0
    assert max(FORMER_FAILURE_TIMES_SEC) < TRIGGER_STOP_TIME_SEC
    manifest = _manifest()
    manifest["gates"][
        "boston_joined_tls_uncontrolled_merge_yield_repaired"
    ] = False
    with pytest.raises(ValueError, match="does not cover"):
        validate_trigger_package(manifest)


def test_v12_route_trigger_and_full_specs_are_version_bound() -> None:
    route = build_route_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v12"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/route"),
        local_output_root=Path("/local/route"),
    )
    trigger = build_trigger_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v12"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/trigger"),
        local_output_root=Path("/local/trigger"),
    )
    full = build_full_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v12"),
        expected_package_manifest_sha256=PACKAGE_SHA256,
        remote_output_root=Path("/remote/full"),
        local_output_root=Path("/local/full"),
    )
    assert route["signature"] == ROUTE_SIGNATURE
    assert route["ram_resource_family"].endswith("v12")
    assert trigger["signature"] == TRIGGER_SIGNATURE
    assert trigger["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in trigger["allowed_nodes"]
    assert "--operational-horizon-sec 7200" in trigger["cmd"]
    assert "--expected-sumo-version 1.22.0" in trigger["cmd"]
    assert full["signature"] == FULL_SIGNATURE
    assert full["ram_resource_family"].endswith("v12-complete")
    assert "--operational-horizon-sec" not in full["cmd"]


def test_v12_full_gate_accepts_only_extended_trigger() -> None:
    validate_trigger_result(
        _passing_trigger(),
        expected_package_manifest_sha256=PACKAGE_SHA256,
    )
    shortened = _passing_trigger()
    shortened["setting"]["operational_horizon_sec"] = 5_400
    with pytest.raises(ValueError, match="execution contract changed"):
        validate_trigger_result(
            shortened,
            expected_package_manifest_sha256=PACKAGE_SHA256,
        )


@pytest.mark.parametrize(
    "script",
    (
        "scripts/cluster/launch_eth_boston_route_admission_v10.py",
        "scripts/cluster/launch_eth_boston_trigger_admission_v10.py",
        "scripts/cluster/launch_eth_boston_full_admission_v9.py",
    ),
)
def test_v12_launchers_are_directly_executable(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
