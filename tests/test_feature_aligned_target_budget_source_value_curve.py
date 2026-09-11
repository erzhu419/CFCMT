from __future__ import annotations

from copy import deepcopy
import json

import pytest

from cf_h2o.eval import (
    traffic_signal_feature_aligned_target_budget_source_value_curve as subject,
)


def _config() -> dict:
    return {
        "protocol": subject.CONFIG_PROTOCOL,
        "target_budgets": [0, 25, 50, 100, 250, 500, 1000],
        "frozen_identities": {
            "parent_snapshot_sha256": "parent",
            "derived_snapshot_sha256": "derived",
            "derived_source_tree_sha256": "tree",
            "parent_action_ranker_sha256": "old",
            "patched_action_ranker_sha256": "new",
        },
    }


def _launch() -> dict:
    return {
        "protocol": subject.LAUNCH_PROTOCOL,
        "submitted": True,
        "signature": subject.SIGNATURE,
        "snapshot_sha256": "derived",
        "source_tree_sha256": "tree",
        "snapshot_patch": {"parent_sha256": "old", "patched_sha256": "new"},
        "scheduler_stdout": json.dumps({"submitted": [{"id": "task"}]}),
        "execution_node": "node",
    }


def _budget(mean: float, upper: float, source_vs_pp: float = 0.02) -> dict:
    return {
        "nested_source_selection": {
            "mean_source_policy_delta": source_vs_pp,
            "mean_target_only_policy_delta": 0.03,
            "selected_candidate_counts": {"uniform": 22},
            "paired_source_minus_target_bootstrap": {
                "mean": mean,
                "lower_95": mean - 0.001,
                "upper_95": upper,
                "improving_seed_fraction": 0.9,
            },
        }
    }


def _result(*, b100_mean: float = -0.01, b100_upper: float = -0.005) -> dict:
    common = {
        "protocol": subject.V123_RESULT_PROTOCOL,
        "artifact": {"protocol": subject.V123_ARTIFACT_PROTOCOL, "path": "/p"},
        "city": "jinan",
        "estimand": "estimand",
        "target_name": "target",
        "target_budgets": [25, 50, 100, 250, 500, 1000],
        "selector_scenario": "jinan",
        "selector_seed_count": 22,
        "selector_group_count": 100,
        "selector_row_count": 800,
        "selection_audits": {"fixed": True},
        "input_audits": {"fixed": True},
        "inputs": {"fixed": True},
        "fit_diagnostics": [{"arm": "target", "elapsed_seconds": 1.0}],
        "selection": {"decision": "legacy"},
    }
    common["budget_results"] = {
        "0": {"nested_source_selection": {}},
        **{
            str(budget): _budget(
                b100_mean if budget == 100 else -0.001,
                b100_upper if budget == 100 else -0.0001,
            )
            for budget in (25, 50, 100, 250, 500, 1000)
        },
    }
    return common


def test_b100_relative_signal_authorizes_branch_even_when_source_trails_pp() -> None:
    legacy = _result()
    corrected = deepcopy(legacy)
    corrected["fit_diagnostics"][0]["elapsed_seconds"] = 2.0

    result = subject.aggregate_results(corrected, legacy, _launch(), _config())

    assert result["b100_branch_authorization"]["passed"] is True
    assert result["budget_comparison"]["100"]["corrected"][
        "v123_budget_level_absolute_conditions_passed"
    ] is False
    assert result["scientific_status"] == (
        "feature_aligned_b100_relative_source_contribution_retained"
    )
    assert result["b100_branch_authorization"]["decision"] == (
        "authorize_freezing_separate_same_state_450_second_branch_protocol"
    )


def test_b100_nonnegative_upper_bound_rejects_branch() -> None:
    legacy = _result()
    corrected = _result(b100_mean=-0.01, b100_upper=0.0001)

    result = subject.aggregate_results(corrected, legacy, _launch(), _config())

    assert result["b100_branch_authorization"]["passed"] is False


def test_changed_input_contract_is_rejected() -> None:
    legacy = _result()
    corrected = deepcopy(legacy)
    corrected["input_audits"] = {"fixed": False}

    with pytest.raises(ValueError, match="changed more than feature binding"):
        subject.aggregate_results(corrected, legacy, _launch(), _config())
