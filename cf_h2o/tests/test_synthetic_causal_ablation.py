from __future__ import annotations

from cf_h2o.eval.synthetic_bus_ablation import run_synthetic_causal_ablation


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
