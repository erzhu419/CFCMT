from __future__ import annotations

import inspect

import torch

from cf_h2o.latent.factor_encoder import TimeVaryingFactorEncoder, build_history_windows
from cf_h2o.latent.factor_prior import StandardNormalFactorPrior, theta_norm_metrics
from cf_h2o.latent.factor_regularizers import (
    domain_contrast_loss,
    mechanism_independence_loss,
    temporal_smoothness_loss,
)


def test_factor_encoder_outputs_named_theta_shapes():
    torch.manual_seed(3)
    mechanism_names = ["demand", "speed", "dwell", "headway", "reward"]
    latent_dims = {"demand": 2, "speed": 2, "dwell": 2, "headway": 2, "reward": 1}
    encoder = TimeVaryingFactorEncoder(6, mechanism_names, latent_dims, hidden_dim=16)
    history = torch.randn(4, 8, 6)

    theta = encoder(history)

    assert list(theta.keys()) == mechanism_names
    for name in mechanism_names:
        assert theta[name].shape == (4, latent_dims[name])
        assert torch.isfinite(theta[name]).all()


def test_history_padding_mask_blocks_padded_values():
    torch.manual_seed(4)
    encoder = TimeVaryingFactorEncoder(
        3,
        mechanism_names=["demand", "reward"],
        latent_dims={"demand": 2, "reward": 1},
        hidden_dim=12,
    )
    valid = torch.tensor([[[0.2, 0.1, -0.3], [0.5, -0.2, 0.7]]])
    padded_a = torch.cat([torch.full((1, 3, 3), 123456.0), valid], dim=1)
    padded_b = torch.cat([torch.full((1, 3, 3), -98765.0), valid], dim=1)
    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])

    theta_a = encoder(padded_a, mask)
    theta_b = encoder(padded_b, mask)

    for name in theta_a:
        assert torch.allclose(theta_a[name], theta_b[name], atol=1e-6)


def test_build_history_windows_left_pads_short_history():
    features = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    windows, masks = build_history_windows(features, history_len=4)

    assert windows.shape == (5, 4, 3)
    assert masks.shape == (5, 4)
    assert masks[0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert torch.allclose(windows[0, -1], features[0])
    assert masks[3].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_factor_encoder_forward_has_no_domain_shortcut_argument():
    params = inspect.signature(TimeVaryingFactorEncoder.forward).parameters
    forbidden = {"domain_id", "source", "real_sim_label", "city_id"}

    assert forbidden.isdisjoint(params.keys())


def test_factor_regularizers_and_metrics_are_finite():
    torch.manual_seed(8)
    encoder = TimeVaryingFactorEncoder(
        4,
        mechanism_names=["demand", "speed", "reward"],
        latent_dims={"demand": 2, "speed": 2, "reward": 1},
        hidden_dim=16,
    )
    history = torch.randn(6, 5, 4)
    masks = torch.ones(6, 5)
    theta = encoder(history, masks)
    theta_seq = encoder.forward_sequence(history, masks)
    domain_id = torch.tensor([0, 0, 1, 1, 2, 2])

    losses = [
        temporal_smoothness_loss(theta_seq),
        mechanism_independence_loss(theta),
        domain_contrast_loss(theta, domain_id),
        StandardNormalFactorPrior()(theta),
    ]
    for loss in losses:
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    metrics = theta_norm_metrics(theta)
    assert set(metrics) == {"theta_norm/demand", "theta_norm/speed", "theta_norm/reward"}
    assert all(value >= 0.0 for value in metrics.values())
