from pathlib import Path

import pytest

from scripts.cluster.audit_tsc_group_normalized_stage import (
    GROUP_NORMALIZED_RIGID_FAMILY,
    _validate_group_normalized_diagnostics,
)
from cf_h2o.traffic_signal.action_ranker import CAUSAL_RIGID_RANKING_PARENTS


def _row() -> dict:
    return {
        "model_diagnostics": {
            "source_domains": ["a", "b", "c", "d", "e"],
            "target_adaptation_groups": 0,
            "fit": {
                GROUP_NORMALIZED_RIGID_FAMILY: {
                    "target_name": "interval_cost",
                    "estimator": "HistGradientBoostingRegressor",
                    "causal": True,
                    "feature_names": list(CAUSAL_RIGID_RANKING_PARENTS),
                    "feature_count": len(CAUSAL_RIGID_RANKING_PARENTS),
                    "rows": 100,
                    "training_rows": 80,
                    "candidate_only": True,
                    "sign_balance": {"enabled": True},
                    "target_normalization_protocol": "group_action_range_v1",
                    "domain_target_scale_protocol": (
                        "within-group-action-range-relative-floor-v1"
                    ),
                    "domain_target_scales": {},
                    "group_target_scale_summary": {
                        "group_count": 20,
                        "minimum": 0.1,
                        "median": 1.0,
                        "maximum": 10.0,
                    },
                }
            },
        }
    }


def test_group_normalized_audit_requires_exact_estimand_and_parents():
    result = _validate_group_normalized_diagnostics(
        _row(), source=Path("result.json")
    )
    assert result["source_domain_count"] == 5
    assert result["target_normalization_protocol"] == "group_action_range_v1"

    row = _row()
    row["model_diagnostics"]["fit"][GROUP_NORMALIZED_RIGID_FAMILY][
        "target_normalization_protocol"
    ] = "domain_action_scale_v1"
    with pytest.raises(ValueError, match="normalization protocol"):
        _validate_group_normalized_diagnostics(
            row, source=Path("result.json")
        )
