from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_eth_boston_collision_evidence_replay_v8 import (
    SIGNATURE,
    build_spec,
)


def test_boston_collision_replay_has_new_diagnostic_identity() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v9"),
        expected_package_manifest_sha256="a" * 64,
        remote_output_root=Path("/remote/replay-v8"),
        local_output_root=Path("/local/replay-v8"),
    )
    assert spec["signature"] == SIGNATURE
    assert "collision-evidence" in spec["resource_family"]
    assert "--operational-horizon-sec 5400" in spec["cmd"]
    assert "cf_h2o.eval.traffic_signal_eth_city_admission" in spec["cmd"]


def test_boston_collision_replay_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_eth_boston_collision_evidence_replay_v8.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
