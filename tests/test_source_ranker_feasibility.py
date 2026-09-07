from __future__ import annotations

from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_conservative_source_intervention_gate import (
    Prediction,
)
from cf_h2o.eval.traffic_signal_source_ranker_feasibility import (
    _fit_ridge,
    _permute_source_groups,
    _policy_seed_values,
    _relative_prediction_features,
    _sufficient_statistics,
    _standardized_arm_matrices,
)
from scripts.cluster.launch_tsc_source_ranker_feasibility import build_spec


def test_relative_prediction_features_anchor_each_group() -> None:
    prediction = Prediction(
        score=np.asarray([2.0, 1.0, 4.0, 5.0]),
        uncertainty=np.asarray([0.1, 0.2, 0.3, 0.4]),
        trust=np.asarray([0.9, 0.8, 0.7, 0.6]),
    )
    rows = np.asarray([[0, 1], [2, 3]])
    references = np.asarray([True, False, True, False])
    features = _relative_prediction_features(prediction, rows, references)
    np.testing.assert_allclose(features[:, 0], [0.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(features[:, 1], [0.2, 0.3, 0.6, 0.7])
    np.testing.assert_allclose(features[:, 2], [0.9, 0.8, 0.7, 0.6])


def test_group_permutation_preserves_action_slots_and_breaks_groups() -> None:
    source = np.arange(24, dtype=float).reshape(8, 3)
    rows = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]])
    seeds = np.asarray([10, 10, 20, 20])
    permuted = _permute_source_groups(source, rows, seeds)
    assert not np.array_equal(permuted, source)
    for seed in (10, 20):
        groups = np.flatnonzero(seeds == seed)
        np.testing.assert_array_equal(
            np.sort(permuted[rows[groups], 0], axis=0),
            np.sort(source[rows[groups], 0], axis=0),
        )


def test_architecture_matched_matrices_share_dimension() -> None:
    common = np.asarray([[0.0], [1.0], [2.0]])
    source = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    matrices = _standardized_arm_matrices(common, source, source[::-1])
    assert {matrix.shape for matrix in matrices.values()} == {(3, 3)}
    np.testing.assert_array_equal(matrices["target_only"][:, 1:], 0.0)


def test_ridge_statistics_exclude_heldout_seed() -> None:
    design = np.asarray([[0.0], [1.0], [10.0], [11.0]])
    target = np.asarray([0.0, 1.0, -100.0, -100.0])
    row_seeds = np.asarray([1, 1, 2, 2])
    statistics = _sufficient_statistics(design, target, row_seeds)
    coefficients = _fit_ridge(statistics, [1], alpha=1.0)
    assert coefficients[1] > 0.0


def test_policy_evaluation_uses_only_eligible_actions() -> None:
    predicted = np.asarray([0.0, -2.0, -1.0, 0.0, -3.0, -2.0])
    actual = np.asarray([0.0, -1.0, -0.2, 0.0, -2.0, -0.1])
    rows = np.asarray([[0, 1, 2], [3, 4, 5]])
    eligible = np.asarray([True, False, True, True, False, True])
    references = np.asarray([True, False, False, True, False, False])
    values, interventions = _policy_seed_values(
        predicted,
        actual,
        rows,
        eligible,
        references,
        np.asarray([10, 20]),
        (10, 20),
    )
    assert values == {10: -0.2, 20: -0.1}
    assert interventions == {10: 1, 20: 1}


def test_v126_launcher_freezes_b500_and_worker_contract() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        prediction_artifact_remote=Path("/evidence/predictions.pkl"),
        prediction_artifact_sha256="a" * 64,
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/result"),
        node="node005",
        cache_workers=20,
        fold_workers=20,
    )
    command = spec["cmd"]
    assert "traffic_signal_source_ranker_feasibility" in command
    assert "--target-budget 500" in command
    assert "--cache-workers 20" in command
    assert "--fold-workers 20" in command
    assert command.endswith("printf 'TASK_DONE\\n'")
    assert spec["require_node"] == "node005"
