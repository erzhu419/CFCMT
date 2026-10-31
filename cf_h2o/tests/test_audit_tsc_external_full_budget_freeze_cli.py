from pathlib import Path
import subprocess
import sys


def test_full_budget_audit_cli_imports_project_from_outside_repo(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/cluster/audit_tsc_external_full_budget_freeze.py"),
            "--help",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--generation" in completed.stdout
