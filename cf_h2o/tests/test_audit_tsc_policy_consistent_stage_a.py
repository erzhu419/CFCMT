from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_policy_consistent_stage_a import (
    _sha256,
    _validate_source_rule_baseline,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _baseline(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    _write(
        path,
        {
            "protocol": "tsc-audited-source-rule-baseline-v1",
            "source_rule_policy_costs": {
                f"scenario_{scenario}": {
                    f"policy_{policy}": float(policy)
                    for policy in range(31)
                }
                for scenario in range(16)
            },
            "source_rule_cache": {
                "aggregate_sha256": "b" * 64,
                "source_tree_sha256": "a" * 64,
                "cache_file_count": 992,
                "scenario_count": 16,
                "policy_count": 31,
                "row_count": 992,
            },
        },
    )
    return path


def test_source_rule_baseline_provenance_passes_exact_contract(tmp_path: Path) -> None:
    path = _baseline(tmp_path)
    result = _validate_source_rule_baseline(
        path,
        expected_sha256=_sha256(path),
        expected_cache_aggregate_sha256="b" * 64,
        expected_source_tree_sha256="a" * 64,
    )
    assert result["cache_file_count"] == 992
    assert result["scenario_count"] == 16


def test_source_rule_baseline_provenance_rejects_drift(tmp_path: Path) -> None:
    path = _baseline(tmp_path)
    with pytest.raises(ValueError, match="aggregate_sha256 mismatch"):
        _validate_source_rule_baseline(
            path,
            expected_sha256=_sha256(path),
            expected_cache_aggregate_sha256="c" * 64,
            expected_source_tree_sha256="a" * 64,
        )


def test_audit_script_is_directly_executable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/cluster/audit_tsc_policy_consistent_stage_a.py"),
            "--help",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-rule-baseline-sha256" in completed.stdout
