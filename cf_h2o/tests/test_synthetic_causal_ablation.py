from __future__ import annotations

import math

from cf_h2o.eval.synthetic_bus_ablation import (
    run_imperfect_sim_h2o_online_ablation,
    run_synthetic_causal_ablation,
    run_synthetic_data_efficiency_ablation,
)


def test_synthetic_causal_ablation_shows_causal_factorization_helps():
    result = run_synthetic_causal_ablation(
        {
            "seed": 31,
            "n_train": 192,
            "n_eval": 128,
            "hidden_dim": 24,
            "batch_size": 64,
            "train_epochs_residual": 80,
            "residual_lr": 3e-3,
        }
    )

    assert result.residual_final_loss < result.residual_initial_loss * 0.5
    assert result.causal_factored_mse < result.uncalibrated_mse * 0.35
    assert result.causal_factored_mse < result.monolithic_linear_mse * 0.85
    assert result.causal_factored_mse < result.wrong_mask_mse * 0.75


def test_synthetic_causal_factorization_is_data_efficient():
    result = run_synthetic_data_efficiency_ablation(
        {
            "seed": 41,
            "train_sizes": [8, 16, 32, 64, 96],
            "n_eval": 512,
            "ridge": 1.0,
            "target_mse": 0.02,
        }
    )

    assert result.causal_n_at_target == 16
    assert result.dense_n_at_target == 96
    assert result.sample_efficiency_gain >= 6.0
    assert result.points[1].causal_sparse_mse < result.points[-1].dense_unfactored_mse
    assert result.points[0].causal_to_dense_ratio < 0.25
    assert all(point.causal_to_dense_ratio < 0.02 for point in result.points[1:4])


def test_imperfect_sim_h2o_online_bridge_is_weighted_not_rejected():
    result = run_imperfect_sim_h2o_online_ablation(
        {
            "seed": 53,
            "n_train": 160,
            "n_eval": 128,
            "hidden_dim": 24,
            "batch_size": 64,
            "train_epochs_residual": 70,
            "residual_lr": 3e-3,
            "residual_scale": 1.4,
        }
    )

    assert result.uncalibrated_sim_mse > 0.25
    assert result.corrected_sim_mse < result.uncalibrated_sim_mse * 0.15
    assert result.good_sim_weight_mean > result.imperfect_sim_weight_mean
    assert result.imperfect_sim_weight_mean > result.bad_sim_weight_mean
    assert result.imperfect_sim_weight_mean > 0.05
    assert math.isfinite(result.weighted_sim_loss)
    assert result.weight_ess_ratio > 0.8
