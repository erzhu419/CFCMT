from __future__ import annotations

from pathlib import Path
import shlex

import numpy as np

from cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic import (
    _cross_fitted_weighted_score,
    _policy_metrics,
    _predictive_metrics,
    _row_weights,
    _weights,
)
from scripts.cluster.launch_tsc_b100_source_reliability_diagnostic import (
    SIGNATURE,
    build_spec,
)


def test_reliability_weights_are_normalized_and_select_low_error_sources() -> None:
    error = np.asarray([0.3, 0.1, 0.2], dtype=float)
    assert np.allclose(_weights(error, "top1"), [0.0, 1.0, 0.0])
    assert np.allclose(_weights(error, "top2"), [0.0, 0.5, 0.5])
    for method in ("inverse_mse", "softmax_0.05"):
        weight = _weights(error, method)
        assert np.isclose(np.sum(weight), 1.0)
        assert int(np.argmax(weight)) == 1


def test_group_equal_row_weights_are_vectorized_and_masked() -> None:
    groups = np.asarray(["a", "a", "a", "b", "b", "c"])
    mask = np.asarray([True, False, True, True, True, False])
    observed = _row_weights(groups, mask)
    assert np.allclose(observed, [0.5, 0.0, 0.5, 0.5, 0.5, 0.0])


def test_cross_fitted_reliability_does_not_use_heldout_fold_labels() -> None:
    groups = np.asarray(["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"])
    references = np.asarray([True, False] * 4)
    row_folds = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    actual = np.asarray([0.0, -1.0, 0.0, -0.8, 0.0, 0.7, 0.0, 0.9])
    source_scores = np.asarray(
        [
            [0.0, -0.9, 0.0, -0.7, 0.0, -0.8, 0.0, -0.6],
            [0.0, 0.8, 0.0, 0.7, 0.0, 0.6, 0.0, 0.8],
        ]
    )
    original, _ = _cross_fitted_weighted_score(
        actual=actual,
        source_scores=source_scores,
        groups=groups,
        references=references,
        row_folds=row_folds,
        method="top1",
    )
    changed = actual.copy()
    changed[row_folds == 0] += 100.0
    rerun, _ = _cross_fitted_weighted_score(
        actual=changed,
        source_scores=source_scores,
        groups=groups,
        references=references,
        row_folds=row_folds,
        method="top1",
    )
    assert np.array_equal(original[row_folds == 0], rerun[row_folds == 0])


def test_metrics_are_group_equal_and_policy_uses_one_action_per_group() -> None:
    groups = np.asarray(["g0", "g0", "g0", "g1", "g1"])
    references = np.asarray([True, False, False, True, False])
    actual = np.asarray([0.0, -0.4, 0.2, 0.0, 0.3])
    predicted = np.asarray([0.0, -0.3, 0.1, 0.0, -0.2])
    group_rows = (np.asarray([0, 1, 2]), np.asarray([3, 4]))
    predictive = _predictive_metrics(actual, predicted, groups, references)
    policy = _policy_metrics(actual, predicted, group_rows, references)
    expected_mse = 0.5 * ((0.1**2 + 0.1**2) / 2.0) + 0.5 * 0.5**2
    assert np.isclose(predictive["group_equal_mse"], expected_mse)
    assert np.isclose(policy["mean_normalized_delta_vs_phase_pressure"], -0.05)
    assert np.isclose(policy["intervention_fraction"], 1.0)
    assert np.isclose(policy["harmful_intervention_fraction"], 0.5)


def test_v122_launcher_consumes_only_frozen_b100_training_artifacts() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit/result.json"),
        fit_result_sha256="a" * 64,
        crossfit_artifact_remote=Path("/crossfit/predictions.pkl"),
        crossfit_artifact_sha256="b" * 64,
        target_cache_root=Path("/cache/target"),
        target_cache_audit_remote=Path("/audit/target.json"),
        remote_output_root=Path("/results/v122"),
        node="node003",
        cache_workers=20,
    )
    tokens = shlex.split(spec["cmd"].split(" && ", 1)[0])
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == 20
    assert spec["ram_mb"] == 32768
    assert "cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic" in tokens
    assert "--selector-cache-root" not in tokens
    assert tokens[tokens.index("--crossfit-artifact-sha256") + 1] == "b" * 64
    assert spec["cmd"].endswith("printf 'TASK_DONE\\n'")
