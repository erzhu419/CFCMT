from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_boosted_rollout_value_stage import (
    BOOSTED_ROLLOUT_VALUE_FAMILY,
    _validate_boosted_rollout_value_diagnostics,
)


def _row() -> dict:
    return {
        "model_diagnostics": {
            "fit": {
                BOOSTED_ROLLOUT_VALUE_FAMILY: {
                    "config": {
                        "source_stack_protocol": (
                            "rigid_anchored_mechanism_residual_v1"
                        ),
                        "source_expert_gate_protocol": "none",
                        "mechanism_dataset_protocol": (
                            "domain_normalized_outcome_zero_prior_v1"
                        ),
                        "mechanism_action_ranking_selection": True,
                        "mechanism_estimator_protocol": "causal_boosted_v1",
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
                                "variant_name": "physical",
                                "parent_names": ["total_q"],
                                "enabled": True,
                                "estimator": "HistGradientBoostingRegressor",
                            }
                        },
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


def test_boosted_rollout_value_audit_requires_parent_restricted_estimator():
    result = _validate_boosted_rollout_value_diagnostics(
        _row(), source=Path("result.json")
    )
    assert result["estimator"] == "HistGradientBoostingRegressor"
    assert result["mechanism_estimator_protocol"] == "causal_boosted_v1"

    row = _row()
    row["model_diagnostics"]["fit"][BOOSTED_ROLLOUT_VALUE_FAMILY]["source"][
        "mechanism_fit"
    ]["terminal_clearance"]["variant_name"] = "dense_all_local_plus_context"
    with pytest.raises(ValueError, match="dense mode"):
        _validate_boosted_rollout_value_diagnostics(
            row, source=Path("result.json")
        )
