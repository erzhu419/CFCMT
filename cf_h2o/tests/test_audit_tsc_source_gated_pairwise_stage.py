from pathlib import Path

import pytest

from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
)
from scripts.cluster.audit_tsc_source_gated_pairwise_stage import (
    SOURCE_GATED_PAIRWISE_FAMILY,
    _validate_source_gated_pairwise_diagnostics,
)


def _row() -> dict:
    return {
        "model_diagnostics": {
            "source_domains": ["a", "b", "c", "d", "e"],
            "target_adaptation_groups": 0,
            "fit": {
                SOURCE_GATED_PAIRWISE_FAMILY: {
                    "estimator": "source_gated_antisymmetric_pairwise_advantage",
                    "config": {
                        "gate_ridge_alpha": 10.0,
                        "gate_error_quantile": 0.75,
                        "minimum_disagreement_groups": 24,
                        "minimum_robust_gain": 0.002,
                        "worst_regret_tolerance": 0.01,
                    },
                    "source": {
                        "protocol": "source_city_oof_pairwise_lcb_gate_v1",
                        "source_domain_count": 5,
                        "source_city_oof_fold_count": 5,
                        "target_labels_used": False,
                        "rigid_fit": {
                            "feature_names": list(CAUSAL_RIGID_RANKING_PARENTS)
                        },
                        "pairwise_fit": {
                            "preference_protocol": (
                                "all_action_antisymmetric_pairwise_v1"
                            ),
                            "state_feature_names": list(
                                PAIRWISE_CAUSAL_STATE_PARENTS
                            ),
                            "action_feature_names": list(
                                PAIRWISE_CAUSAL_ACTION_PARENTS
                            ),
                        },
                        "disagreement_group_count": 100,
                        "gate_enabled": True,
                        "reason": "selected_by_nested_source_city_oof_regret",
                        "selected_pairwise_group_count": 12,
                        "one_sided_error_margin": 0.1,
                        "robust_gain_vs_rigid": 0.01,
                    },
                }
            },
        }
    }


def test_source_gated_pairwise_audit_requires_nested_source_only_gate():
    result = _validate_source_gated_pairwise_diagnostics(
        _row(), source=Path("result.json")
    )
    assert result["gate_enabled"] is True

    row = _row()
    row["model_diagnostics"]["fit"][SOURCE_GATED_PAIRWISE_FAMILY]["source"][
        "target_labels_used"
    ] = True
    with pytest.raises(ValueError, match="target-label"):
        _validate_source_gated_pairwise_diagnostics(
            row, source=Path("result.json")
        )
