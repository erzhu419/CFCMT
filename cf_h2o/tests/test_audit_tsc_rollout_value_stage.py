from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_rollout_value_stage import (
    ROLLOUT_VALUE_FAMILY,
    _validate_rollout_value_diagnostics,
)


def _row() -> dict:
    return {
        "model_diagnostics": {
            "fit": {
                ROLLOUT_VALUE_FAMILY: {
                    "config": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "source_expert_gate_protocol": "none",
                    },
                    "source": {
                        "mechanism_dataset_protocol": (
                            "group_invariant_local_physical_simulator_residual_v1"
                        ),
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "source_domain_count": 5,
                        "pair_excluded_core_fit_count": 10,
                        "pair_excluded_mechanism_fit_count": 10,
                        "mechanism_fit": {
                            "terminal_clearance": {
                                "target_name": "terminal_system_load",
                                "variant_name": "physical",
                                "parent_names": ["total_q"],
                                "enabled": True,
                            }
                        },
                        "stack_enabled": True,
                        "stack_weight": 1.0,
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


def test_rollout_value_audit_requires_terminal_value_only() -> None:
    result = _validate_rollout_value_diagnostics(
        _row(), source=Path("result.json")
    )
    assert result["mechanism_name"] == "terminal_clearance"
    assert result["target_name"] == "terminal_system_load"
    assert result["pair_excluded_core_fit_count"] == 10

    row = _row()
    row["model_diagnostics"]["fit"][ROLLOUT_VALUE_FAMILY]["source"][
        "mechanism_fit"
    ]["queue"] = {}
    with pytest.raises(ValueError, match="terminal-clearance only"):
        _validate_rollout_value_diagnostics(row, source=Path("result.json"))


def test_rollout_value_audit_script_is_directly_executable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/cluster/audit_tsc_rollout_value_stage.py"),
            "--help",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--reference-audit-sha256" in completed.stdout
