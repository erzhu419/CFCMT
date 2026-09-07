from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_eth_boston_route_admission_v6 import (
    SIGNATURE as ROUTE_SIGNATURE,
    build_spec as build_route_spec,
)
from scripts.cluster.launch_eth_boston_full_admission_v6 import (
    SIGNATURE as FULL_SIGNATURE,
    build_spec as build_full_spec,
    validate_trigger_result,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v6 import (
    FORMER_FAILURE_TIMES_SEC,
    SIGNATURE as TRIGGER_SIGNATURE,
    TRIGGER_HORIZON_SEC,
    build_spec as build_trigger_spec,
)


def test_v8_route_spec_keeps_complete_demand_contract() -> None:
    digest = "a" * 64
    spec = build_route_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v8"),
        expected_package_manifest_sha256=digest,
        remote_output_root=Path("/remote/route"),
        local_output_root=Path("/local/route"),
    )
    assert spec["signature"] == ROUTE_SIGNATURE
    assert spec["ram_resource_family"].endswith("v8")
    assert "--routing-threads 20" in spec["cmd"]
    assert "node004" not in spec["allowed_nodes"]


def test_v8_trigger_spec_crosses_all_observed_failure_times() -> None:
    digest = "b" * 64
    spec = build_trigger_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v8"),
        expected_package_manifest_sha256=digest,
        remote_output_root=Path("/remote/trigger"),
        local_output_root=Path("/local/trigger"),
    )
    assert spec["signature"] == TRIGGER_SIGNATURE
    assert spec["ram_resource_family"].endswith("v8")
    assert f"--operational-horizon-sec {TRIGGER_HORIZON_SEC}" in spec["cmd"]
    assert max(FORMER_FAILURE_TIMES_SEC) < 7_201 + TRIGGER_HORIZON_SEC
    assert "node004" not in spec["allowed_nodes"]


def test_v8_full_gate_and_spec_require_passing_trigger() -> None:
    digest = "c" * 64
    trigger = {
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
            "package_manifest_sha256": digest,
            "city_code": "BOS",
            "backend": "libsumo",
            "expected_sumo_version": "1.22.0",
            "begin_time_sec": 7_201.0,
            "operational_horizon_sec": TRIGGER_HORIZON_SEC,
            "stop_time_sec": 7_201.0 + TRIGGER_HORIZON_SEC,
        },
        "observed": {"final_time_sec": 7_201.0 + TRIGGER_HORIZON_SEC},
    }
    validate_trigger_result(trigger, expected_package_manifest_sha256=digest)
    spec = build_full_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v8"),
        expected_package_manifest_sha256=digest,
        remote_output_root=Path("/remote/full"),
        local_output_root=Path("/local/full"),
    )
    assert spec["signature"] == FULL_SIGNATURE
    assert spec["ram_resource_family"].endswith("v8-complete")
    assert "--operational-horizon-sec" not in spec["cmd"]
    assert "node004" not in spec["allowed_nodes"]


def test_v8_launchers_are_directly_executable() -> None:
    for relative in (
        "scripts/cluster/launch_eth_boston_route_admission_v6.py",
        "scripts/cluster/launch_eth_boston_trigger_admission_v6.py",
        "scripts/cluster/launch_eth_boston_full_admission_v6.py",
    ):
        completed = subprocess.run(
            [sys.executable, relative, "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
