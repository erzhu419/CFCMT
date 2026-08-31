"""Local-state-conditioned refinement of target spatiotemporal action latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.action_ranker import group_normalized_action_target
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
    TargetSpatiotemporalActionLatent,
)


STATE_CONDITIONED_PROTOCOL = "target-state-conditioned-spatiotemporal-latent-v1"
COMPACT_STATE_FEATURES = (
    "total_q",
    "total_veh",
    "mean_speed",
    "mean_occ",
    "current_green_elapsed_norm",
    "queue_concentration",
    "graph_upstream_q_mean",
    "graph_downstream_q_mean",
    "graph_neighbor_q_max",
    "graph_neighbor_occ_mean",
    "graph_global_q_per_tls",
    "graph_focal_q_percentile",
    "network_active_per_tls",
)
STATE_ACTION_FEATURES = (
    *COMPACT_STATE_FEATURES,
    "delta_green_q",
    "delta_red_q",
    "delta_green_down_q",
    "delta_green_down_occ",
    "delta_service_pressure",
    "delta_red_pressure",
    "delta_green_corridor_pressure",
)
FEATURE_SETS = {
    "compact_state": COMPACT_STATE_FEATURES,
    "state_action": STATE_ACTION_FEATURES,
}


@dataclass(frozen=True)
class StateConditionedLatentConfig:
    base: SpatiotemporalActionLatentConfig
    feature_set: str = "compact_state"
    neighbor_count: int = 10
    local_shrinkage: float = 5.0
    distance_bandwidth: float = 1.0
    uncertainty_quantile: float = 0.90

    def __post_init__(self) -> None:
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown state-conditioned feature set: {self.feature_set!r}")
        if int(self.neighbor_count) < 1:
            raise ValueError("state-conditioned neighbor count must be positive")
        if float(self.local_shrinkage) < 0.0:
            raise ValueError("state-conditioned shrinkage must be nonnegative")
        if float(self.distance_bandwidth) <= 0.0:
            raise ValueError("state-conditioned bandwidth must be positive")
        if not 0.5 <= float(self.uncertainty_quantile) < 1.0:
            raise ValueError("invalid state-conditioned uncertainty quantile")


@dataclass
class _CellRows:
    features: np.ndarray
    targets: np.ndarray
    source_rows: np.ndarray


class StateConditionedSpatiotemporalActionLatent:
    """Refine a spatiotemporal cell mean with deterministic local neighbors."""

    def __init__(
        self,
        config: StateConditionedLatentConfig,
        *,
        target_name: str = "interval_cost",
    ) -> None:
        self.config = config
        self.target_name = str(target_name)
        if not self.target_name:
            raise ValueError("state-conditioned latent target name cannot be empty")
        self.base = TargetSpatiotemporalActionLatent(
            config.base,
            target_name=self.target_name,
        )
        self.feature_names = tuple(FEATURE_SETS[config.feature_set])
        self.feature_indices = np.asarray([], dtype=int)
        self.feature_center = np.asarray([], dtype=float)
        self.feature_scale = np.asarray([], dtype=float)
        self.cells: dict[tuple[str, int, str, str], _CellRows] = {}
        self.calibration_error = 1.0
        self.fitted = False

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        base_diagnostics = self.base.fit(dataset)
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        feature_index = {name: index for index, name in enumerate(dataset.feature_names)}
        missing = [name for name in self.feature_names if name not in feature_index]
        if missing:
            raise KeyError(f"state-conditioned latent lacks features: {missing}")
        self.feature_indices = np.asarray(
            [feature_index[name] for name in self.feature_names], dtype=int
        )
        features = np.asarray(dataset.features[:, self.feature_indices], dtype=float)
        self.feature_center = np.median(features, axis=0)
        scale = np.quantile(np.abs(features - self.feature_center[None, :]), 0.75, axis=0)
        self.feature_scale = np.maximum(scale, 0.05)
        normalized = (features - self.feature_center[None, :]) / self.feature_scale[None, :]
        target, target_scale_summary = group_normalized_action_target(
            dataset, self.target_name
        )
        bins = np.floor(
            metadata["row_times"] / float(self.config.base.time_bin_sec)
        ).astype(int)
        cell_rows: dict[tuple[str, int, str, str], list[int]] = {}
        for row in range(dataset.size):
            if bool(metadata["is_reference"][row]):
                continue
            key = self._key(metadata, bins, row)
            cell_rows.setdefault(key, []).append(row)
        self.cells = {
            key: _CellRows(
                features=normalized[np.asarray(rows, dtype=int)],
                targets=target[np.asarray(rows, dtype=int)],
                source_rows=np.asarray(rows, dtype=int),
            )
            for key, rows in cell_rows.items()
        }
        self.fitted = True
        fitted = self._predict_arrays(dataset, exclude_source_rows=True)
        calibration_rows = ~metadata["is_reference"]
        residual = np.abs(target[calibration_rows] - fitted["mean"][calibration_rows])
        self.calibration_error = max(
            float(np.quantile(residual, self.config.uncertainty_quantile))
            if residual.size
            else 0.0,
            1e-6,
        )
        return {
            "protocol": STATE_CONDITIONED_PROTOCOL,
            "target_name": self.target_name,
            "config": {
                **asdict(self.config),
                "base": asdict(self.config.base),
            },
            "feature_names": list(self.feature_names),
            "row_count": int(dataset.size),
            "action_group_count": int(
                np.unique(metadata["action_group_ids"]).size
            ),
            "cell_count": len(self.cells),
            "cell_support": {
                "minimum": int(min(len(cell.targets) for cell in self.cells.values())),
                "median": float(np.median([len(cell.targets) for cell in self.cells.values()])),
                "maximum": int(max(len(cell.targets) for cell in self.cells.values())),
            },
            "target_scale_summary": target_scale_summary,
            "training_nonreference_mae": float(np.mean(residual)),
            "calibration_error": float(self.calibration_error),
            "calibration_mode": "leave_self_out_within_cell",
            "base": base_diagnostics,
            "uses_future_outcomes_at_prediction": False,
            "uses_target_labels_at_fit": True,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.fitted:
            raise RuntimeError("state-conditioned latent has not been fitted")
        values = self._predict_arrays(dataset)
        uncertainty = self.calibration_error * (
            1.0
            + values["nearest_distance"]
            + 1.0 / np.sqrt(np.maximum(values["support_count"], 1.0))
        )
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        uncertainty[metadata["is_reference"]] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": values["base_mean"],
                "latent_residual": values["mean"],
                "mean": values["mean"],
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(values["trust"], 0.0, 1.0),
                "context_distance": values["nearest_distance"],
                "context_support": np.clip(values["trust"], 0.0, 1.0),
                "latent_support_count": values["support_count"],
                "latent_level": values["level"],
            }
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": STATE_CONDITIONED_PROTOCOL,
            "config": {
                **asdict(self.config),
                "base": asdict(self.config.base),
            },
            "target_name": self.target_name,
            "feature_names": list(self.feature_names),
            "cell_count": len(self.cells),
            "calibration_error": float(self.calibration_error),
            "base": self.base.diagnostics(),
            "fitted": bool(self.fitted),
        }

    def _predict_arrays(
        self,
        dataset: MechanismDataset,
        *,
        exclude_source_rows: bool = False,
    ) -> dict[str, np.ndarray]:
        metadata = TargetSpatiotemporalActionLatent._metadata(dataset)
        base = self.base.predict(dataset)["control_cost"]
        base_mean = np.asarray(base["mean"], dtype=float)
        base_trust = np.asarray(base["context_trust"], dtype=float)
        features = np.asarray(dataset.features[:, self.feature_indices], dtype=float)
        normalized = (
            features - self.feature_center[None, :]
        ) / self.feature_scale[None, :]
        bins = np.floor(
            metadata["row_times"] / float(self.config.base.time_bin_sec)
        ).astype(int)
        means = base_mean.copy()
        trusts = np.zeros(dataset.size, dtype=float)
        support = np.zeros(dataset.size, dtype=float)
        nearest = np.full(dataset.size, 10.0, dtype=float)
        levels = np.asarray(base["latent_level"], dtype=float).copy()
        bandwidth = float(self.config.distance_bandwidth)
        shrinkage = float(self.config.local_shrinkage)
        for row in range(dataset.size):
            if bool(metadata["is_reference"][row]):
                continue
            cell = self.cells.get(self._key(metadata, bins, row))
            if cell is None:
                trusts[row] = float(base_trust[row])
                continue
            eligible = np.flatnonzero(
                cell.source_rows != row
                if exclude_source_rows
                else np.ones(cell.source_rows.shape, dtype=bool)
            )
            if eligible.size == 0:
                trusts[row] = float(base_trust[row])
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
            order = eligible[
                np.argsort(distance, kind="stable")[
                    : min(int(self.config.neighbor_count), distance.size)
                ]
            ]
            selected_distance = np.sqrt(
                np.mean(
                    (cell.features[order] - normalized[row][None, :]) ** 2,
                    axis=1,
                )
            )
            weights = np.exp(-0.5 * (selected_distance / bandwidth) ** 2)
            weight_sum = float(np.sum(weights))
            local_mean = float(
                np.sum(weights * cell.targets[order]) / max(weight_sum, 1e-12)
            )
            alpha = weight_sum / max(weight_sum + shrinkage, 1e-12)
            means[row] = float(base_mean[row]) + alpha * (
                local_mean - float(base_mean[row])
            )
            nearest[row] = float(selected_distance[0])
            proximity = float(np.exp(-nearest[row] / bandwidth))
            trusts[row] = float(base_trust[row]) * alpha * proximity
            support[row] = float(len(order))
            levels[row] = 4.0
        is_reference = metadata["is_reference"]
        means[is_reference] = 0.0
        trusts[is_reference] = 1.0
        support[is_reference] = 1.0
        nearest[is_reference] = 0.0
        return {
            "mean": means,
            "base_mean": base_mean,
            "trust": trusts,
            "support_count": support,
            "nearest_distance": nearest,
            "level": levels,
        }

    @staticmethod
    def _key(
        metadata: Mapping[str, np.ndarray],
        bins: np.ndarray,
        row: int,
    ) -> tuple[str, int, str, str]:
        return (
            str(metadata["row_tls"][row]),
            int(bins[row]),
            str(metadata["reference_states"][row]),
            str(metadata["candidate_states"][row]),
        )
