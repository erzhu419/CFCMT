from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_rigid_residual_stage import (
    _validate_residual_diagnostics,
)


FAMILY = "cfcmt_one_step_rigid_residual_mobility_queue"


def _row() -> dict:
    return {
        "model_diagnostics": {
            "fit": {
                FAMILY: {
                    "config": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        )
                    },
                    "source": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "mechanism_dataset_protocol": (
                            "group_invariant_local_physical_simulator_residual_v1"
                        ),
                        "source_domain_count": 5,
                        "pair_excluded_core_fit_count": 10,
                        "pair_excluded_mechanism_fit_count": 10,
                        "stack_enabled": True,
                        "stack_weight": 0.5,
                        "selected_candidate": {"alpha": 1.0},
                    },
                    "target": {
                        "enabled": False,
                        "reason": "no_target_counterfactual_groups",
                    },
                }
            }
        }
    }


def test_residual_audit_requires_pair_excluded_rigid_fits() -> None:
    result = _validate_residual_diagnostics(
        _row(),
        family=FAMILY,
        source=Path("result.json"),
    )
    assert result["pair_excluded_core_fit_count"] == 10
    assert result["stack_weight"] == 0.5

    row = _row()
    row["model_diagnostics"]["fit"][FAMILY]["source"][
        "pair_excluded_core_fit_count"
    ] = 0
    with pytest.raises(ValueError, match="pair-excluded rigid"):
        _validate_residual_diagnostics(
            row,
            family=FAMILY,
            source=Path("result.json"),
        )


def test_rigid_residual_audit_script_is_directly_executable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/cluster/audit_tsc_rigid_residual_stage.py"),
            "--help",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-rule-generator-source-sha256" in completed.stdout
