from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_selective_residual_stage import (
    _validate_gate_diagnostics,
)


FAMILY = "cfcmt_one_step_selective_residual_mobility_queue_q75"


def _row() -> dict:
    return {
        "model_diagnostics": {
            "fit": {
                FAMILY: {
                    "config": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "source_expert_gate_protocol": (
                            "source_city_oof_ridge_lcb_v1"
                        ),
                        "source_expert_gate_error_quantile": 0.75,
                        "source_expert_gate_ridge_alpha": 10.0,
                        "source_expert_gate_min_disagreement_groups": 24,
                    },
                    "source": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "source_expert_gate_protocol": (
                            "source_city_oof_ridge_lcb_v1"
                        ),
                        "mechanism_dataset_protocol": (
                            "group_invariant_local_physical_simulator_residual_v1"
                        ),
                        "source_domain_count": 5,
                        "pair_excluded_core_fit_count": 10,
                        "pair_excluded_mechanism_fit_count": 10,
                        "stack_enabled": True,
                        "stack_weight": 1.0,
                        "source_expert_gate": {
                            "enabled": True,
                            "protocol": "source_city_oof_ridge_lcb_v1",
                            "source_domain_count": 5,
                            "source_city_oof_fold_count": 5,
                            "disagreement_group_count": 200,
                            "feature_names": [
                                "core_penalty_norm",
                                "residual_preference_norm",
                                "core_margin_norm",
                                "residual_margin_norm",
                                "core_choice_correction_norm",
                                "residual_choice_correction_norm",
                                "correction_range_norm",
                                "log_action_count",
                            ],
                            "target_labels_used": False,
                            "error_quantile": 0.75,
                            "ridge_alpha": 10.0,
                            "selected_residual_fraction_of_all_groups": 0.1,
                            "one_sided_error_margin": 0.3,
                            "source_city_oof_one_sided_coverage": 0.75,
                        },
                    },
                    "target": {
                        "enabled": False,
                        "reason": "no_target_counterfactual_groups",
                    },
                }
            }
        }
    }


def test_selective_residual_audit_requires_source_only_gate_contract() -> None:
    result = _validate_gate_diagnostics(
        _row(),
        family=FAMILY,
        source=Path("result.json"),
    )
    assert result["source_domain_count"] == 5
    assert result["error_quantile"] == 0.75
    assert result["disagreement_group_count"] == 200

    row = _row()
    row["model_diagnostics"]["fit"][FAMILY]["source"][
        "source_expert_gate"
    ]["target_labels_used"] = True
    with pytest.raises(ValueError, match="target labels"):
        _validate_gate_diagnostics(
            row,
            family=FAMILY,
            source=Path("result.json"),
        )


def test_selective_residual_audit_script_is_directly_executable() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "scripts/cluster/audit_tsc_selective_residual_stage.py"
            ),
            "--help",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--reference-audit-sha256" in completed.stdout
