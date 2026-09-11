import subprocess
import sys
from pathlib import Path

from scripts.cluster.launch_eth_boston_v18_admission import (
    FORMER_FAILURE_TIMES_SEC,
    SIGNATURE_ROOT,
    build_full_spec,
    build_package_spec,
    build_route_spec,
    build_trigger_spec,
)


def _common() -> dict[str, Path | str]:
    return {
        "snapshot_root": Path("/snapshot"),
        "package_root": Path("/data/boston_complete_published_package_v18"),
        "expected_package_manifest_sha256": "a" * 64,
        "remote_output_root": Path("/remote/result"),
        "local_output_root": Path("/local/result"),
    }


def test_v18_specs_use_controlled_yield_repair_and_identity() -> None:
    package = build_package_spec(
        snapshot_root=Path("/snapshot"),
        base_root=Path("/data/boston_complete_published_package_v17"),
        output_root=Path("/data/boston_complete_published_package_v18"),
        expected_base_manifest_sha256="b" * 64,
        remote_result_root=Path("/remote/package"),
        local_result_root=Path("/local/package"),
    )
    assert "repair_eth_boston_controlled_shared_receiving_yield" in package["cmd"]
    assert package["require_node"] == "node002"

    route = build_route_spec(**_common())
    trigger = build_trigger_spec(**_common())
    full = build_full_spec(**_common())
    assert route["signature"].startswith(SIGNATURE_ROOT)
    assert trigger["ram_resource_family"].endswith("v18")
    assert full["ram_resource_family"].endswith("v18")
    assert 11_885.0 in FORMER_FAILURE_TIMES_SEC


def test_v18_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_eth_boston_v18_admission.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
