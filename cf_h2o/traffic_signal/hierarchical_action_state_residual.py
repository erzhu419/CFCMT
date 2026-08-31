"""Hierarchical fixed-effect plus pooled causal action-state residual model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    group_normalized_action_target,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
    TargetSpatiotemporalActionLatent,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_ACTION_FEATURES,
)


HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL = (
    "hierarchical-action-state-residual-v1"
)
HIERARCHICAL_FEATURE_SETS = {
    "compact_state_action": tuple(STATE_ACTION_FEATURES),
    "causal_rigid": tuple(CAUSAL_RIGID_RANKING_PARENTS),
}


@dataclass(frozen=True)
class HierarchicalActionStateResidualConfig:
    base: SpatiotemporalActionLatentConfig
    feature_set: str = "compact_state_action"
    residual_scale: float = 0.5
    residual_clip: float = 1.0
    learning_rate: float = 0.05
    max_iter: int = 120
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 100
    l2_regularization: float = 20.0
    uncertainty_quantile: float = 0.90
    random_state: int = 20260830

    def __post_init__(self) -> None:
        if self.feature_set not in HIERARCHICAL_FEATURE_SETS:
            raise ValueError(f"unknown hierarchical feature set: {self.feature_set!r}")
        if not 0.0 < float(self.residual_scale) <= 1.0:
            raise ValueError("hierarchical residual scale must be in (0, 1]")
        if float(self.residual_clip) <= 0.0:
            raise ValueError("hierarchical residual clip must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("hierarchical learning rate must be positive")
        if int(self.max_iter) < 1 or int(self.max_leaf_nodes) < 2:
            raise ValueError("invalid hierarchical boosting capacity")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("hierarchical min_samples_leaf must be positive")
        if float(self.l2_regularization) < 0.0:
            raise ValueError("hierarchical l2 regularization must be nonnegative")
        if not 0.5 <= float(self.uncertainty_quantile) < 1.0:
            raise ValueError("invalid hierarchical uncertainty quantile")


class HierarchicalActionStateResidual:
    """Pool state response while preserving target TLS-time-phase effects."""

    def __init__(self, config: HierarchicalActionStateResidualConfig) -> None:
        self.config = config
        self.base = TargetSpatiotemporalActionLatent(config.base)
        self.feature_names = tuple(HIERARCHICAL_FEATURE_SETS[config.feature_set])
        self.feature_indices = np.asarray([], dtype=int)
        self.cell_centers: dict[tuple[str, int, str, str], np.ndarray] = {}
        self.feature_scale = np.asarray([], dtype=float)
        self.model: HistGradientBoostingRegressor | None = None
        self.constant_residual = 0.0
        self.calibration_error = 1.0
        self.fitted = False

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        base_diagnostics = self.base.fit(dataset)
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        feature_index = {name: index for index, name in enumerate(dataset.feature_names)}
        missing = [name for name in self.feature_names if name not in feature_index]
        if missing:
            raise KeyError(f"hierarchical action-state residual lacks features: {missing}")
        self.feature_indices = np.asarray(
            [feature_index[name] for name in self.feature_names], dtype=int
        )
        raw_features = np.asarray(
            dataset.features[:, self.feature_indices], dtype=float
        )
        bins = self._bins(metadata)
        cell_rows: dict[tuple[str, int, str, str], list[int]] = {}
        for row in range(dataset.size):
            if bool(metadata["is_reference"][row]):
                continue
            cell_rows.setdefault(self._key(metadata, bins, row), []).append(row)
        self.cell_centers = {
            key: np.median(raw_features[np.asarray(rows, dtype=int)], axis=0)
            for key, rows in cell_rows.items()
        }
        centered, known = self._centered_features(
            raw_features,
            metadata=metadata,
            bins=bins,
        )
        fit_mask = (~metadata["is_reference"]) & known
        if not np.any(fit_mask):
            raise ValueError("hierarchical action-state residual has no training rows")
        self.feature_scale = np.maximum(
            np.quantile(np.abs(centered[fit_mask]), 0.75, axis=0),
            0.05,
        )
        normalized = centered / self.feature_scale[None, :]
        base = self.base.predict(dataset)["control_cost"]
        base_mean = np.asarray(base["mean"], dtype=float)
        base_trust = np.asarray(base["context_trust"], dtype=float)
        design = self._design(normalized, base_mean, base_trust)
        target, target_scale_summary = group_normalized_action_target(
            dataset, "interval_cost"
        )
        residual_target = target - base_mean
        fit_target = residual_target[fit_mask]
        if float(np.std(fit_target)) < 1e-8:
            self.model = None
            self.constant_residual = float(np.mean(fit_target))
            fitted_residual = np.full(dataset.size, self.constant_residual, dtype=float)
        else:
            self.model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(self.config.learning_rate),
                max_iter=int(self.config.max_iter),
                max_leaf_nodes=int(self.config.max_leaf_nodes),
                min_samples_leaf=int(self.config.min_samples_leaf),
                l2_regularization=float(self.config.l2_regularization),
                early_stopping=False,
                random_state=int(self.config.random_state),
            )
            self.model.fit(design[fit_mask], fit_target)
            self.constant_residual = float(np.mean(fit_target))
            fitted_residual = np.asarray(self.model.predict(design), dtype=float)
        fitted_residual = np.clip(
            fitted_residual,
            -float(self.config.residual_clip),
            float(self.config.residual_clip),
        )
        fitted_mean = base_mean + float(self.config.residual_scale) * fitted_residual
        fitted_mean[metadata["is_reference"]] = 0.0
        calibration_rows = fit_mask
        calibration_residual = np.abs(
            target[calibration_rows] - fitted_mean[calibration_rows]
        )
        self.calibration_error = max(
            float(
                np.quantile(
                    calibration_residual,
                    float(self.config.uncertainty_quantile),
                )
            ),
            1e-6,
        )
        self.fitted = True
        supports = [len(rows) for rows in cell_rows.values()]
        return {
            "protocol": HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL,
            "config": {**asdict(self.config), "base": asdict(self.config.base)},
            "feature_names": list(self.feature_names),
            "row_count": int(dataset.size),
            "training_row_count": int(np.sum(fit_mask)),
            "cell_count": len(self.cell_centers),
            "cell_support": {
                "minimum": int(min(supports)),
                "median": float(np.median(supports)),
                "maximum": int(max(supports)),
            },
            "target_scale_summary": target_scale_summary,
            "training_residual_mae": float(
                np.mean(np.abs(fit_target - fitted_residual[fit_mask]))
            ),
            "training_action_value_mae": float(np.mean(calibration_residual)),
            "calibration_error": float(self.calibration_error),
            "uncertainty_calibration": "regularized_training_residual_q90",
            "base": base_diagnostics,
            "uses_target_labels_at_fit": True,
            "uses_future_outcomes_at_prediction": False,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.fitted:
            raise RuntimeError("hierarchical action-state residual has not been fitted")
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        raw_features = np.asarray(
            dataset.features[:, self.feature_indices], dtype=float
        )
        bins = self._bins(metadata)
        centered, known = self._centered_features(
            raw_features,
            metadata=metadata,
            bins=bins,
        )
        normalized = centered / self.feature_scale[None, :]
        base = self.base.predict(dataset)["control_cost"]
        base_mean = np.asarray(base["mean"], dtype=float)
        base_trust = np.asarray(base["context_trust"], dtype=float)
        design = self._design(normalized, base_mean, base_trust)
        residual = (
            np.full(dataset.size, self.constant_residual, dtype=float)
            if self.model is None
            else np.asarray(self.model.predict(design), dtype=float)
        )
        residual = np.clip(
            residual,
            -float(self.config.residual_clip),
            float(self.config.residual_clip),
        )
        residual[~known] = 0.0
        scaled_residual = float(self.config.residual_scale) * residual
        mean = base_mean + scaled_residual
        distance = np.sqrt(np.mean(normalized**2, axis=1))
        trust = base_trust * np.exp(-0.25 * distance)
        trust[~known] = base_trust[~known]
        support = np.asarray(base["latent_support_count"], dtype=float)
        level = np.asarray(base["latent_level"], dtype=float).copy()
        level[known] = 5.0
        uncertainty = self.calibration_error * (1.0 + 0.25 * distance)
        uncertainty[~known] = np.asarray(base["uncertainty"], dtype=float)[~known]
        reference = metadata["is_reference"]
        mean[reference] = 0.0
        scaled_residual[reference] = 0.0
        trust[reference] = 1.0
        distance[reference] = 0.0
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": base_mean,
                "latent_residual": scaled_residual,
                "mean": mean,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": np.clip(trust, 0.0, 1.0),
                "latent_support_count": support,
                "latent_level": level,
                "state_residual_known": known.astype(float),
            }
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": HIERARCHICAL_ACTION_STATE_RESIDUAL_PROTOCOL,
            "config": {**asdict(self.config), "base": asdict(self.config.base)},
            "feature_names": list(self.feature_names),
            "cell_count": len(self.cell_centers),
            "calibration_error": float(self.calibration_error),
            "base": self.base.diagnostics(),
            "fitted": bool(self.fitted),
        }

    def _bins(self, metadata: dict[str, np.ndarray]) -> np.ndarray:
        return np.floor(
            metadata["row_times"] / float(self.config.base.time_bin_sec)
        ).astype(int)

    def _centered_features(
        self,
        raw_features: np.ndarray,
        *,
        metadata: dict[str, np.ndarray],
        bins: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        centered = np.zeros_like(raw_features, dtype=float)
        known = np.zeros(raw_features.shape[0], dtype=bool)
        for row in range(raw_features.shape[0]):
            if bool(metadata["is_reference"][row]):
                continue
            center = self.cell_centers.get(self._key(metadata, bins, row))
            if center is None:
                continue
            centered[row] = raw_features[row] - center
            known[row] = True
        return centered, known

    @staticmethod
    def _design(
        normalized: np.ndarray,
        base_mean: np.ndarray,
        base_trust: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack((normalized, base_mean, base_trust))

    @staticmethod
    def _key(
        metadata: dict[str, np.ndarray],
        bins: np.ndarray,
        row: int,
    ) -> tuple[str, int, str, str]:
        return (
            str(metadata["row_tls"][row]),
            int(bins[row]),
            str(metadata["reference_states"][row]),
            str(metadata["candidate_states"][row]),
        )
