from pathlib import Path
import subprocess
import sys

import pytest

from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    EXPECTED_TRIP_COUNT,
    TRIGGER_BEGIN_TIME_SEC,
)
from scripts.cluster.launch_eth_boston_v16_admission import (
    FORMER_FAILURE_TIMES_SEC,
    TRIGGER_HORIZON_SEC,
    TRIGGER_STOP_TIME_SEC,
    V16_PACKAGE_PROTOCOL,
    build_full_spec,
    build_package_spec,
    build_route_spec,
    build_trigger_spec,
    validate_trigger_package,
)


PACKAGE_SHA256 = "c" * 64


def _manifest() -> dict:
    return {
        "protocol": V16_PACKAGE_PROTOCOL,
        "passed": True,
        "city_code": "BOS",
        "gates": {"boston_implicit_no_tls_major_states_repaired": True},
        "microscopic_instantiation": {
            "backend_required_by_protocol": "libsumo",
            "begin_sec": TRIGGER_BEGIN_TIME_SEC,
            "end_sec": 71_996,
        },
        "microscopic_demand_inventory": {"trip_count": EXPECTED_TRIP_COUNT},
    }


def test_v16_package_and_trigger_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.cluster.launch_eth_boston_v16_admission."
        "validate_implicit_no_tls_major_manifest",
        lambda manifest: None,
    )
    validate_trigger_package(_manifest())
    assert TRIGGER_HORIZON_SEC == 10_800
    assert 16_316.0 in FORMER_FAILURE_TIMES_SEC
    assert max(FORMER_FAILURE_TIMES_SEC) < TRIGGER_STOP_TIME_SEC


def test_v16_specs_are_version_bound() -> None:
    common = {
        "snapshot_root": Path("/snapshot"),
        "package_root": Path("/package-v16"),
        "expected_package_manifest_sha256": PACKAGE_SHA256,
        "remote_output_root": Path("/remote/result"),
        "local_output_root": Path("/local/result"),
    }
    package = build_package_spec(
        snapshot_root=Path("/snapshot"),
        base_root=Path("/package-v15"),
        output_root=Path("/package-v16"),
        expected_base_manifest_sha256=PACKAGE_SHA256,
        remote_result_root=Path("/remote/package-evidence"),
        local_result_root=Path("/local/package-evidence"),
    )
    route = build_route_spec(**common)
    trigger = build_trigger_spec(**common)
    full = build_full_spec(**common)
    assert "repair_eth_boston_implicit_no_tls_major" in package["cmd"]
    assert route["ram_resource_family"].endswith("v16")
    assert trigger["signature"].startswith(
        "CFCMT/v150/eth-boston-v16-implicit-no-tls-major"
    )
    assert full["ram_resource_family"].endswith("v16")


def test_v16_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/cluster/launch_eth_boston_v16_admission.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
