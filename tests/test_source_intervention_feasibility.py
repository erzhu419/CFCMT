from __future__ import annotations

import json

import numpy as np

from cf_h2o.eval.traffic_signal_source_intervention_feasibility import (
    _oracle_rows,
    _seed_policy_summary,
)


def test_oracle_respects_action_eligibility() -> None:
    actual = np.asarray([0.0, -0.4, -0.2, 0.0, 0.2, -0.1])
    rows = np.asarray([[0, 1, 2], [3, 4, 5]])
    unrestricted = _oracle_rows(actual, rows)
    eligible = np.asarray([True, False, True, True, True, False])
    constrained = _oracle_rows(actual, rows, eligible=eligible)
    np.testing.assert_array_equal(unrestricted, [1, 5])
    np.testing.assert_array_equal(constrained, [2, 3])


def test_seed_policy_summary_is_json_native() -> None:
    actual = np.asarray([0.0, -0.2, 0.0, 0.1])
    summary = _seed_policy_summary(
        actual,
        selected_rows=np.asarray([1, 3]),
        reference_rows=np.asarray([0, 2]),
        group_seeds=np.asarray([10, 20]),
    )
    assert summary["intervention_count"] == 2
    assert summary["beneficial_intervention_fraction"] == 0.5
    assert summary["harmful_intervention_fraction"] == 0.5
    json.dumps(summary, allow_nan=False)
