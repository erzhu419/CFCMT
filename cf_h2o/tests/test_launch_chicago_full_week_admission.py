from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_chicago_full_week_admission import admission_command


def test_admission_command_uses_immutable_libsumo_runner_and_json_only() -> None:
    command = admission_command(
        snapshot_root=Path("/snap/abc"),
        package_root=Path("/data/routes"),
        expected_manifest_sha256="a" * 64,
        remote_day_root=Path("/results/day0"),
        day_index=0,
    )

    assert "CFCMT_SOURCE_ROOT=/snap/abc" in command
    assert "/snap/abc/scripts/cluster/run_cfcmt_sumo122.sh" in command
    assert "traffic_signal_chicago_full_week_admission.py" in command
    assert "--day-index 0" in command
    assert "--expected-package-manifest-sha256 " + "a" * 64 in command
    assert "traci" not in command.lower()
    assert "csv" not in command.lower()


def test_chicago_launcher_is_directly_executable_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_chicago_full_week_admission.py",
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run the seven Chicago safety days" in completed.stdout
