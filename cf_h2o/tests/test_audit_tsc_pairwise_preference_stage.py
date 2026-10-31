from pathlib import Path

import pytest

from cf_h2o.traffic_signal.action_ranker import (
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
)
from scripts.cluster.audit_tsc_pairwise_preference_stage import (
    PAIRWISE_PREFERENCE_FAMILY,
    _validate_pairwise_preference_diagnostics,
)


def _row() -> dict:
    return {
        "model_diagnostics": {
            "source_domains": ["a", "b", "c", "d", "e"],
            "target_adaptation_groups": 0,
            "fit": {
                PAIRWISE_PREFERENCE_FAMILY: {
                    "target_name": "interval_cost",
                    "estimator": "HistGradientBoostingRegressor",
                    "causal": True,
                    "preference_protocol": "all_action_antisymmetric_pairwise_v1",
                    "target_normalization_protocol": "group_action_range_v1",
                    "target_scale_protocol": (
                        "within-group-action-range-relative-floor-v1"
                    ),
                    "state_feature_names": list(PAIRWISE_CAUSAL_STATE_PARENTS),
                    "action_feature_names": list(PAIRWISE_CAUSAL_ACTION_PARENTS),
                    "state_feature_count": len(PAIRWISE_CAUSAL_STATE_PARENTS),
                    "action_feature_count": len(PAIRWISE_CAUSAL_ACTION_PARENTS),
                    "uses_context_features": False,
                    "source_domain_count": 5,
                    "action_group_count": 100,
                    "unordered_pair_count": 300,
                    "oriented_pair_count": 600,
                    "antisymmetric_augmentation": True,
                    "pair_target_minimum": -1.0,
                    "pair_target_maximum": 1.0,
                    "group_target_scale_summary": {"group_count": 100},
                }
            },
        }
    }


def test_pairwise_audit_requires_antisymmetry_and_exact_parents():
    result = _validate_pairwise_preference_diagnostics(
        _row(), source=Path("result.json")
    )
    assert result["oriented_pair_count"] == 600

    row = _row()
    row["model_diagnostics"]["fit"][PAIRWISE_PREFERENCE_FAMILY][
        "antisymmetric_augmentation"
    ] = False
    with pytest.raises(ValueError, match="antisymmetric augmentation"):
        _validate_pairwise_preference_diagnostics(
            row, source=Path("result.json")
        )
