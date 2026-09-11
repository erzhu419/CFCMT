import subprocess
import sys
from pathlib import Path

from scripts.cluster.launch_eth_boston_v17_admission import (
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
        "package_root": Path("/data/boston_complete_published_package_v17"),
        "expected_package_manifest_sha256": "a" * 64,
        "remote_output_root": Path("/remote/result"),
        "local_output_root": Path("/local/result"),
    }


def test_v17_specs_use_new_repair_and_admission_identity() -> None:
    package = build_package_spec(
        snapshot_root=Path("/snapshot"),
        base_root=Path("/data/boston_complete_published_package_v16"),
        output_root=Path("/data/boston_complete_published_package_v17"),
        expected_base_manifest_sha256="b" * 64,
        remote_result_root=Path("/remote/package"),
        local_result_root=Path("/local/package"),
    )
    assert "repair_eth_boston_no_tls_major_response" in package["cmd"]
    assert package["require_node"] == "node002"

    route = build_route_spec(**_common())
    trigger = build_trigger_spec(**_common())
    full = build_full_spec(**_common())
    assert route["signature"].startswith(SIGNATURE_ROOT)
    assert trigger["ram_resource_family"].endswith("v17")
    assert full["ram_resource_family"].endswith("v17")
    assert 12_201.0 in FORMER_FAILURE_TIMES_SEC


def test_v17_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_eth_boston_v17_admission.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
