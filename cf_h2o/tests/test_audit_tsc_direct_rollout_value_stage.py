from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_direct_rollout_value_stage import (
    DIRECT_ROLLOUT_VALUE_FAMILY,
    _validate_direct_rollout_value_diagnostics,
)


def _row() -> dict:
    return {
        "model_diagnostics": {
            "fit": {
                DIRECT_ROLLOUT_VALUE_FAMILY: {
                    "config": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "source_expert_gate_protocol": "none",
                        "mechanism_dataset_protocol": (
                            "domain_normalized_outcome_zero_prior_v1"
                        ),
                        "mechanism_action_ranking_selection": True,
                        "mechanism_estimator_protocol": "invariant_ridge_v1",
                    },
                    "source": {
                        "mechanism_dataset_protocol": (
                            "domain_normalized_outcome_zero_prior_v1"
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
                                "variant_name": "minimal",
                                "parent_names": ["total_q"],
                                "enabled": True,
                            }
                        },
                        "stack_enabled": False,
                        "stack_weight": 0.0,
                        "selected_candidate": {"alpha": 0.1},
                    },
                    "target": {
                        "enabled": False,
                        "reason": "no_target_counterfactual_groups",
                    },
                }
            }
        }
    }


def test_direct_rollout_value_audit_requires_zero_prior_terminal_only():
    result = _validate_direct_rollout_value_diagnostics(
        _row(), source=Path("result.json")
    )
    assert result["mechanism_enabled"] is True
    assert result["target_name"] == "terminal_system_load"

    row = _row()
    row["model_diagnostics"]["fit"][DIRECT_ROLLOUT_VALUE_FAMILY]["config"][
        "mechanism_dataset_protocol"
    ] = "wrong"
    with pytest.raises(ValueError, match="zero-prior"):
        _validate_direct_rollout_value_diagnostics(
            row, source=Path("result.json")
        )
