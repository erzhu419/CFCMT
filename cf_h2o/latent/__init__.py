"""Latent mechanism factor inference for CF-H2O."""

from cf_h2o.latent.factor_encoder import TimeVaryingFactorEncoder, build_history_windows
from cf_h2o.latent.factor_prior import StandardNormalFactorPrior, theta_norm_metrics
from cf_h2o.latent.factor_regularizers import (
    domain_contrast_loss,
    mechanism_independence_loss,
    temporal_smoothness_loss,
)

__all__ = [
    "StandardNormalFactorPrior",
    "TimeVaryingFactorEncoder",
    "build_history_windows",
    "domain_contrast_loss",
    "mechanism_independence_loss",
    "temporal_smoothness_loss",
    "theta_norm_metrics",
]
