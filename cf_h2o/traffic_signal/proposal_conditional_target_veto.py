"""Proposal-conditional target-simulator veto for frozen CFCMT actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


VETO_PROTOCOL = "target-simulator-300s-proposal-conditional-causal-veto-v1"
MODEL_PROTOCOL = "proposal-conditional-causal-regressor-v1"


@dataclass(frozen=True)
class ProposalConditionalRegressorConfig:
    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 7
    min_samples_leaf: int = 12
    l2_regularization: float = 16.0
    random_state: int = 20260803
    uncertainty_quantile: float = 0.90


@dataclass(frozen=True)
class ProposalConditionalVetoConfig:
    enabled: bool
    acceptance_quantile: float
    score_threshold: float
    rollout_value_horizon_sec: int = 300

    def __post_init__(self) -> None:
        if (
            not 0.0 < float(self.acceptance_quantile) < 1.0
            or not np.isfinite(float(self.score_threshold))
            or int(self.rollout_value_horizon_sec) <= 0
        ):
            raise ValueError("invalid proposal-conditional veto configuration")


class ProposalConditionalRegressor:
    """Regress long-horizon effects only on states where CFCMT proposes."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        config: ProposalConditionalRegressorConfig | None = None,
    ) -> None:
        self.feature_names = tuple(str(value) for value in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("proposal-conditional features must be unique and nonempty")
        self.config = config or ProposalConditionalRegressorConfig()
        self.estimator: HistGradientBoostingRegressor | None = None
        self.constant_score = 0.0
        self.calibration_error = 1.0

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        if (
            x.ndim != 2
            or x.shape != (y.size, len(self.feature_names))
            or y.size < 2
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("invalid proposal-conditional training matrix")
        weights = (
            np.ones(y.size, dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        if (
            weights.shape != (y.size,)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
        ):
            raise ValueError("invalid proposal-conditional sample weights")
        weights *= weights.size / max(float(weights.sum()), 1e-12)
        self.constant_score = float(np.average(y, weights=weights))
        if float(np.std(y)) < 1e-8:
            self.estimator = None
            fitted = np.full(y.shape, self.constant_score, dtype=float)
        else:
            self.estimator = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(self.config.learning_rate),
                max_iter=int(self.config.max_iter),
                max_leaf_nodes=int(self.config.max_leaf_nodes),
                min_samples_leaf=int(self.config.min_samples_leaf),
                l2_regularization=float(self.config.l2_regularization),
                early_stopping=False,
                random_state=int(self.config.random_state),
            )
            self.estimator.fit(x, y, sample_weight=weights)
            fitted = np.asarray(self.estimator.predict(x), dtype=float)
        self.calibration_error = max(
            float(
                np.quantile(
                    np.abs(y - fitted), float(self.config.uncertainty_quantile)
                )
            ),
            1e-6,
        )
        return {
            "protocol": MODEL_PROTOCOL,
            "rows": int(y.size),
            "feature_names": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "weighted_mae": float(np.average(np.abs(y - fitted), weights=weights)),
            "nonzero_sign_accuracy": float(
                np.mean((fitted[np.abs(y) > 1e-8] < 0.0) == (y[np.abs(y) > 1e-8] < 0.0))
            )
            if np.any(np.abs(y) > 1e-8)
            else 1.0,
            "calibration_error": float(self.calibration_error),
            "constant": self.estimator is None,
        }

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("proposal-conditional prediction matrix changed")
        if not np.isfinite(x).all():
            raise ValueError("proposal-conditional prediction matrix is nonfinite")
        if self.estimator is None:
            return np.full(x.shape[0], self.constant_score, dtype=float)
        return np.asarray(self.estimator.predict(x), dtype=float)

    def predict_dataset(self, dataset: MechanismDataset) -> np.ndarray:
        index = {name: position for position, name in enumerate(dataset.feature_names)}
        missing = [name for name in self.feature_names if name not in index]
        if missing:
            raise KeyError(f"proposal-conditional dataset missing features: {missing}")
        columns = np.asarray([index[name] for name in self.feature_names], dtype=int)
        return self.predict_matrix(dataset.features[:, columns])


class ProposalConditionalTargetVeto:
    """Retain or veto one pre-existing proposal without originating an action."""

    def __init__(
        self,
        *,
        model: ProposalConditionalRegressor,
        config: ProposalConditionalVetoConfig,
    ) -> None:
        self.model = model
        self.config = config

    def evaluate(
        self,
        dataset: MechanismDataset,
        *,
        candidate_index: int,
        reference_index: int,
    ) -> dict[str, float | bool | str]:
        candidate = int(candidate_index)
        reference = int(reference_index)
        if not 0 <= candidate < dataset.size or not 0 <= reference < dataset.size:
            raise IndexError("proposal-conditional action index is outside the group")
        is_reference = np.asarray(
            dataset.metadata.get("is_reference", ()), dtype=bool
        )
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
        if (
            is_reference.shape != (dataset.size,)
            or groups.shape != (dataset.size,)
            or not bool(is_reference[reference])
            or bool(is_reference[candidate])
            or groups[candidate] != groups[reference]
        ):
            raise ValueError("proposal-conditional candidate/reference is invalid")
        score = float(self.model.predict_dataset(dataset)[candidate])
        if not self.config.enabled:
            rejection = "disabled"
        elif score > float(self.config.score_threshold):
            rejection = "proposal_conditional_score"
        else:
            rejection = "none"
        return {
            "protocol": VETO_PROTOCOL,
            "eligible": rejection == "none",
            "rejection": rejection,
            "predicted_delta": score,
            "uncertainty": float(self.model.calibration_error),
            "context_trust": 1.0,
            "upper": score,
            "score_threshold": float(self.config.score_threshold),
            "acceptance_quantile": float(self.config.acceptance_quantile),
            "rollout_value_horizon_sec": int(
                self.config.rollout_value_horizon_sec
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": VETO_PROTOCOL,
            "role": "proposal_conditional_execution_veto_only",
            "action_originator": False,
            "target_labels": "target_simulator_counterfactual_recovery_value",
            "zero_shot": False,
            "feature_names": list(self.model.feature_names),
            "config": asdict(self.config),
        }
