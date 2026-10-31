from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_eth_boston_route_admission_v5 import ALLOWED_NODES
from scripts.cluster.launch_eth_boston_route_admission_v9 import (
    SIGNATURE,
    build_spec,
)


def test_v11_route_spec_keeps_complete_demand_contract() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v11"),
        expected_package_manifest_sha256="a" * 64,
        remote_output_root=Path("/remote/route-v9"),
        local_output_root=Path("/local/route-v9"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "-m cf_h2o.eval.traffic_signal_eth_city_route_admission" in spec["cmd"]
    assert "--expected-sumo-version 1.22.0" in spec["cmd"]
    assert "--routing-threads 20" in spec["cmd"]
    assert "subset" not in spec["cmd"]
    assert spec["ram_resource_family"].endswith("v11")


def test_v11_route_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_eth_boston_route_admission_v9.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
