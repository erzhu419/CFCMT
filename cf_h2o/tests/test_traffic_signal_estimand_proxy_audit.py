from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_estimand_proxy_audit import (
    _group_ranking_metrics,
    _median_positive_group_range,
)


def test_median_positive_group_range_ignores_tied_groups() -> None:
    values = np.asarray([1.0, 1.0, 2.0, 5.0, 4.0, 8.0])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])

    assert _median_positive_group_range(values, groups) == 3.5


def test_group_ranking_metrics_use_one_reference_and_group_range() -> None:
    metrics = _group_ranking_metrics(
        prediction=np.asarray([0.0, -1.0, 0.0, 1.0]),
        global_cost_delta=np.asarray([0.0, -2.0, 0.0, -1.0]),
        groups=np.asarray(["a", "a", "b", "b"]),
        is_reference=np.asarray([True, False, True, False]),
    )

    assert metrics["group_count"] == 2
    assert metrics["normalized_regret"] == 0.5
    assert metrics["normalized_delta_vs_reference"] == -0.5
    assert metrics["oracle_action_rate"] == 0.5
    assert metrics["oracle_cost_tie_rate"] == 0.5
