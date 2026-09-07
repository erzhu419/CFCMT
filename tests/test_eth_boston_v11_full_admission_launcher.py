from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_eth_boston_full_admission_v8 import (
    SIGNATURE,
    build_spec,
    validate_trigger_result,
)
from scripts.cluster.launch_eth_boston_trigger_admission_v9 import (
    FORMER_FAILURE_TIMES_SEC,
    TRIGGER_BEGIN_TIME_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
)


def _passing_trigger(digest: str) -> dict:
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
            "package_manifest_sha256": digest,
            "city_code": "BOS",
            "backend": "libsumo",
            "expected_sumo_version": "1.22.0",
            "begin_time_sec": TRIGGER_BEGIN_TIME_SEC,
            "operational_horizon_sec": TRIGGER_HORIZON_SEC,
            "stop_time_sec": TRIGGER_STOP_TIME_SEC,
        },
        "observed": {"final_time_sec": TRIGGER_STOP_TIME_SEC},
    }


def test_v11_full_gate_and_spec_cover_all_failures_and_complete_day() -> None:
    digest = "a" * 64
    validate_trigger_result(
        _passing_trigger(digest), expected_package_manifest_sha256=digest
    )
    assert set(FORMER_FAILURE_TIMES_SEC) == {
        9_518.0,
        9_735.0,
        11_304.0,
        11_504.0,
        11_871.0,
        12_279.0,
    }
    assert max(FORMER_FAILURE_TIMES_SEC) < TRIGGER_STOP_TIME_SEC

    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v11"),
        expected_package_manifest_sha256=digest,
        remote_output_root=Path("/remote/full"),
        local_output_root=Path("/local/full"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["ram_resource_family"].endswith("v11-complete")
    assert "--operational-horizon-sec" not in spec["cmd"]
    assert "--expected-city-code BOS" in spec["cmd"]
    assert "--expected-sumo-version 1.22.0" in spec["cmd"]
    assert "node004" not in spec["allowed_nodes"]


def test_v11_full_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_eth_boston_full_admission_v8.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
