"""Distributional safety calibration for local action-state value latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from cf_h2o.traffic_signal.action_ranker import group_normalized_action_target
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    TargetSpatiotemporalActionLatent,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    StateConditionedLatentConfig,
    StateConditionedSpatiotemporalActionLatent,
)


DISTRIBUTIONAL_ACTION_STATE_PROTOCOL = "distributional-action-state-latent-v1"


@dataclass(frozen=True)
class DistributionalActionStateConfig:
    state_model: StateConditionedLatentConfig
    benefit_prior_alpha: float = 1.0
    benefit_prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if float(self.benefit_prior_alpha) <= 0.0:
            raise ValueError("benefit prior alpha must be positive")
        if float(self.benefit_prior_beta) <= 0.0:
            raise ValueError("benefit prior beta must be positive")


class DistributionalActionStateLatent:
    """Estimate action value and calibrated probability of beating the rule."""

    def __init__(self, config: DistributionalActionStateConfig) -> None:
        self.config = config
        self.state_model = StateConditionedSpatiotemporalActionLatent(
            config.state_model
        )
        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        self.calibration_error = 1.0
        self.fitted = False

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        state_diagnostics = self.state_model.fit(dataset)
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        normalized_target, _ = group_normalized_action_target(
            dataset, "interval_cost"
        )
        raw = self._raw_probability(dataset, exclude_source_rows=True)
        calibration_rows = (~metadata["is_reference"]) & (raw["support"] > 0.0)
        if not np.any(calibration_rows):
            raise ValueError("distributional action-state calibration has no supported rows")
        labels = (normalized_target[calibration_rows] < 0.0).astype(float)
        raw_probability = raw["probability"][calibration_rows]
        self.calibrator.fit(raw_probability, labels)
        calibrated = np.asarray(
            self.calibrator.predict(raw_probability), dtype=float
        )
        calibration_residual = np.abs(calibrated - labels)
        self.calibration_error = max(
            float(np.quantile(calibration_residual, 0.90)),
            1e-6,
        )
        self.fitted = True
        return {
            "protocol": DISTRIBUTIONAL_ACTION_STATE_PROTOCOL,
            "config": {
                **asdict(self.config),
                "state_model": {
                    **asdict(self.config.state_model),
                    "base": asdict(self.config.state_model.base),
                },
            },
            "row_count": int(dataset.size),
            "calibration_row_count": int(np.sum(calibration_rows)),
            "beneficial_row_fraction": float(np.mean(labels)),
            "raw_brier_score": float(np.mean((raw_probability - labels) ** 2)),
            "calibrated_brier_score": float(np.mean((calibrated - labels) ** 2)),
            "calibration_error_q90": float(self.calibration_error),
            "calibration_mode": "leave_self_out_local_probability_isotonic",
            "isotonic_threshold_count": int(len(self.calibrator.X_thresholds_)),
            "state_model": state_diagnostics,
            "uses_target_labels_at_fit": True,
            "uses_future_outcomes_at_prediction": False,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.fitted:
            raise RuntimeError("distributional action-state latent has not been fitted")
        state = self.state_model.predict(dataset)["control_cost"]
        raw = self._raw_probability(dataset, exclude_source_rows=False)
        calibrated = np.asarray(
            self.calibrator.predict(raw["probability"]), dtype=float
        )
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        calibrated[metadata["is_reference"]] = 1.0
        context_trust = np.asarray(state["context_trust"], dtype=float).copy()
        unsupported = raw["effective_support"] <= 0.0
        context_trust[unsupported] = 0.0
        context_trust[metadata["is_reference"]] = 1.0
        probability_uncertainty = np.sqrt(
            np.maximum(calibrated * (1.0 - calibrated), 0.0)
            / np.maximum(raw["effective_support"] + 3.0, 1.0)
        )
        return {
            "control_cost": {
                **state,
                "context_trust": context_trust,
                "context_support": context_trust,
                "benefit_probability_raw": np.clip(
                    raw["probability"], 0.0, 1.0
                ),
                "benefit_probability_calibrated": np.clip(
                    calibrated, 0.0, 1.0
                ),
                "benefit_probability_uncertainty": probability_uncertainty,
                "benefit_effective_support": raw["effective_support"],
                "benefit_nearest_distance": raw["nearest_distance"],
            }
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": DISTRIBUTIONAL_ACTION_STATE_PROTOCOL,
            "config": {
                **asdict(self.config),
                "state_model": {
                    **asdict(self.config.state_model),
                    "base": asdict(self.config.state_model.base),
                },
            },
            "calibration_error_q90": float(self.calibration_error),
            "isotonic_threshold_count": (
                int(len(self.calibrator.X_thresholds_)) if self.fitted else 0
            ),
            "state_model": self.state_model.diagnostics(),
            "fitted": bool(self.fitted),
        }

    def _raw_probability(
        self,
        dataset: MechanismDataset,
        *,
        exclude_source_rows: bool,
    ) -> dict[str, np.ndarray]:
        state_model = self.state_model
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        features = np.asarray(
            dataset.features[:, state_model.feature_indices], dtype=float
        )
        normalized = (
            features - state_model.feature_center[None, :]
        ) / state_model.feature_scale[None, :]
        bins = np.floor(
            metadata["row_times"]
            / float(state_model.config.base.time_bin_sec)
        ).astype(int)
        alpha = float(self.config.benefit_prior_alpha)
        beta = float(self.config.benefit_prior_beta)
        probability = np.full(dataset.size, alpha / (alpha + beta), dtype=float)
        support = np.zeros(dataset.size, dtype=float)
        effective_support = np.zeros(dataset.size, dtype=float)
        nearest = np.full(dataset.size, 10.0, dtype=float)
        bandwidth = float(state_model.config.distance_bandwidth)
        neighbor_count = int(state_model.config.neighbor_count)
        for row in range(dataset.size):
            if bool(metadata["is_reference"][row]):
                continue
            cell = state_model.cells.get(state_model._key(metadata, bins, row))
            if cell is None:
                continue
            eligible = np.flatnonzero(
                cell.source_rows != row
                if exclude_source_rows
                else np.ones(cell.source_rows.shape, dtype=bool)
            )
            if eligible.size == 0:
                continue
            distance = np.sqrt(
                np.mean(
                    (
                        cell.features[eligible]
                        - normalized[row][None, :]
                    )
                    ** 2,
                    axis=1,
                )
            )
            local_order = np.argsort(distance, kind="stable")[:neighbor_count]
            selected = eligible[local_order]
            selected_distance = distance[local_order]
            weights = np.exp(-0.5 * (selected_distance / bandwidth) ** 2)
            weight_sum = float(np.sum(weights))
            beneficial = (cell.targets[selected] < 0.0).astype(float)
            probability[row] = (
                float(np.sum(weights * beneficial)) + alpha
            ) / max(weight_sum + alpha + beta, 1e-12)
            support[row] = float(len(selected))
            effective_support[row] = weight_sum**2 / max(
                float(np.sum(weights**2)), 1e-12
            )
            nearest[row] = float(selected_distance[0])
        is_reference = metadata["is_reference"]
        probability[is_reference] = 1.0
        support[is_reference] = 1.0
        effective_support[is_reference] = 1.0
        nearest[is_reference] = 0.0
        return {
            "probability": probability,
            "support": support,
            "effective_support": effective_support,
            "nearest_distance": nearest,
        }
