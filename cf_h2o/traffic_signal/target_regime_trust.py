"""Low-dimensional trust model for target-history traffic regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAME = "historical_phase_mean_queue_per_lane"
MODEL_PROTOCOL = "cfcmt-target-regime-trust-ridge-v1"


@dataclass(frozen=True)
class TargetRegimeTrustModel:
    """Predict whether the local CFCMT controller should replace PhasePressure."""

    feature_mean: float
    feature_scale: float
    intercept: float
    coefficient: float
    l2: float

    def predict_relative_delta(self, historical_mean_queue_per_lane: float) -> float:
        value = float(historical_mean_queue_per_lane)
        if not np.isfinite(value):
            raise ValueError("target regime feature must be finite")
        standardized = (value - self.feature_mean) / self.feature_scale
        return float(self.intercept + self.coefficient * standardized)

    def trusts_local_controller(self, historical_mean_queue_per_lane: float) -> bool:
        return self.predict_relative_delta(historical_mean_queue_per_lane) < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": MODEL_PROTOCOL,
            "feature_names": [FEATURE_NAME],
            "feature_mean": float(self.feature_mean),
            "feature_scale": float(self.feature_scale),
            "intercept": float(self.intercept),
            "coefficient": float(self.coefficient),
            "l2": float(self.l2),
            "trust_rule": "predicted_relative_delta_strictly_less_than_zero",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetRegimeTrustModel":
        if (
            payload.get("protocol") != MODEL_PROTOCOL
            or payload.get("feature_names") != [FEATURE_NAME]
            or payload.get("trust_rule")
            != "predicted_relative_delta_strictly_less_than_zero"
        ):
            raise ValueError("target regime trust artifact protocol changed")
        model = cls(
            feature_mean=float(payload["feature_mean"]),
            feature_scale=float(payload["feature_scale"]),
            intercept=float(payload["intercept"]),
            coefficient=float(payload["coefficient"]),
            l2=float(payload["l2"]),
        )
        values = (
            model.feature_mean,
            model.feature_scale,
            model.intercept,
            model.coefficient,
            model.l2,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("target regime trust artifact contains non-finite values")
        if model.feature_scale <= 0.0 or model.l2 < 0.0:
            raise ValueError("target regime trust artifact has invalid scale or penalty")
        return model


def fit_target_regime_trust(
    features: Sequence[float],
    relative_deltas: Sequence[float],
    *,
    l2: float,
) -> TargetRegimeTrustModel:
    """Fit an intercept-unpenalized ridge model with one declared feature."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(relative_deltas, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("target regime trust fit requires aligned one-dimensional data")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("target regime trust fit data must be finite")
    penalty = float(l2)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("target regime trust ridge penalty must be finite and nonnegative")
    feature_mean = float(np.mean(x))
    feature_scale = float(np.std(x))
    if feature_scale <= 1e-12:
        raise ValueError("target regime trust feature has no usable variation")
    standardized = (x - feature_mean) / feature_scale
    design = np.column_stack((np.ones(x.size, dtype=float), standardized))
    regularizer = np.diag(np.asarray([0.0, penalty], dtype=float))
    beta = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return TargetRegimeTrustModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        intercept=float(beta[0]),
        coefficient=float(beta[1]),
        l2=penalty,
    )


def phase_history_feature(metrics: Mapping[str, Any]) -> float:
    """Extract the sole trust feature from a passive PhasePressure history."""

    if metrics.get("policy") != "phase_pressure":
        raise ValueError("target regime history must come from PhasePressure")
    value = float(metrics["mean_queue_per_lane"])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("PhasePressure history has invalid mean queue per lane")
    return value
