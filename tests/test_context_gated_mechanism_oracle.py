from __future__ import annotations

from pathlib import Path

import numpy as np

import cf_h2o.eval.traffic_signal_context_gated_mechanism_oracle as v135
from scripts.cluster.launch_tsc_context_gated_mechanism_oracle import (
    SIGNATURE,
    build_spec,
)


def test_outer_partition_is_disjoint_and_rotates() -> None:
    seeds = tuple(range(22))
    training, calibration = v135._outer_partition(seeds, 7)
    assert len(training) == 16
    assert len(calibration) == 5
    assert set(training).isdisjoint(calibration)
    assert 7 not in training
    assert 7 not in calibration
    assert set(training) | set(calibration) | {7} == set(seeds)


def test_seed_local_retention_never_mixes_seed_thresholds() -> None:
    proposals = {
        "proposed": np.ones(8, dtype=bool),
        "predicted_advantage": np.asarray(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        ),
    }
    seeds = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    accepted, thresholds = v135._seed_local_retention_mask(
        proposals, seeds, 0.25
    )
    assert np.array_equal(
        accepted, np.asarray([False, False, False, True] * 2)
    )
    assert thresholds == {1: 4.0, 2: 40.0}


def test_context_cross_fit_covers_only_outer_training_proposals(monkeypatch) -> None:
    monkeypatch.setattr(
        v135,
        "CONTEXT_GATE_CONFIG",
        v135.ContextGateConfig(max_iter=5, min_samples_leaf=2),
    )
    seeds = np.repeat(np.arange(6), 4)
    features = np.column_stack(
        [np.linspace(-1.0, 1.0, seeds.size), seeds.astype(float)]
    )
    target = -0.1 * features[:, 0] + 0.001 * seeds
    proposed = np.ones(seeds.size, dtype=bool)
    prediction, error, diagnostics = v135._cross_fitted_context_prediction(
        features=features,
        target=target,
        proposal_mask=proposed,
        group_seeds=seeds,
        training_seeds=tuple(range(5)),
    )
    assert prediction.shape == target.shape
    assert np.all(np.isfinite(prediction))
    assert error > 0.0
    assert diagnostics["oof_row_count"] == 20
    assert diagnostics["outer_training_seed_count"] == 5


def test_calibration_candidate_authorizes_consistent_negative_context() -> None:
    group_seeds = np.repeat(np.arange(5), 8)
    proposals = {
        "actual_selected_delta": np.full(group_seeds.size, -0.01),
    }
    candidate, accepted = v135._calibration_candidate(
        proposals=proposals,
        base_mask=np.ones(group_seeds.size, dtype=bool),
        context_prediction=np.full(group_seeds.size, -0.02),
        error_quantile=0.005,
        group_seeds=group_seeds,
        calibration_seeds=tuple(range(5)),
        retention_fraction=0.1,
        risk_multiplier=1.0,
    )
    assert np.all(accepted)
    assert candidate["authorized"] is True
    assert candidate["intervention_count"] == 40
    assert candidate["improving_seed_fraction"] == 1.0


def test_v135_launcher_keeps_oracle_screen_source_free() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        selector_cache_root=Path("/selector"),
        selector_cache_audit_remote=Path("/evidence/audit.json"),
        selector_cache_audit_sha256="a" * 64,
        remote_output_root=Path("/result"),
        nodes=("node001", "node002", "node003", "node005", "node006"),
        cache_workers=20,
        fold_workers=20,
    )
    assert "context_gated_mechanism_oracle" in spec["cmd"]
    assert "prediction-artifact" not in spec["cmd"]
    assert spec["signature"] == SIGNATURE
    assert "node004" not in spec["allowed_nodes"]
