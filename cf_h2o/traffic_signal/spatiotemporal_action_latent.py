"""Hierarchically shrunk target action latents for long-horizon control cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from cf_h2o.traffic_signal.action_ranker import group_normalized_action_target
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PHASE_PAIR_SCOPE = "phase_pair"
TLS_PHASE_PAIR_SCOPE = "tls_phase_pair"
TLS_TIME_PHASE_PAIR_SCOPE = "tls_time_phase_pair"
LATENT_SCOPES = {
    PHASE_PAIR_SCOPE,
    TLS_PHASE_PAIR_SCOPE,
    TLS_TIME_PHASE_PAIR_SCOPE,
}
LATENT_PROTOCOL = "target-spatiotemporal-action-latent-v1"


@dataclass(frozen=True)
class SpatiotemporalActionLatentConfig:
    scope: str
    shrinkage: float = 20.0
    time_bin_sec: int | None = None
    uncertainty_quantile: float = 0.90

    def __post_init__(self) -> None:
        if self.scope not in LATENT_SCOPES:
            raise ValueError(f"unknown spatiotemporal latent scope: {self.scope!r}")
        if float(self.shrinkage) < 0.0:
            raise ValueError("spatiotemporal latent shrinkage must be nonnegative")
        if not 0.5 <= float(self.uncertainty_quantile) < 1.0:
            raise ValueError("invalid spatiotemporal uncertainty quantile")
        if self.scope == TLS_TIME_PHASE_PAIR_SCOPE:
            if self.time_bin_sec is None or int(self.time_bin_sec) <= 0:
                raise ValueError("TLS-time latent requires a positive time bin")
        elif self.time_bin_sec is not None:
            raise ValueError("time bins are only valid for the TLS-time latent")


@dataclass(frozen=True)
class _LatentValue:
    mean: float
    count: int
    trust: float


class TargetSpatiotemporalActionLatent:
    """Estimate target action value through phase, TLS, and time residuals."""

    def __init__(
        self,
        config: SpatiotemporalActionLatentConfig,
        *,
        target_name: str = "interval_cost",
    ) -> None:
        self.config = config
        self.target_name = str(target_name)
        if not self.target_name:
            raise ValueError("spatiotemporal latent target name cannot be empty")
        self.phase_pair: dict[tuple[str, str], _LatentValue] = {}
        self.tls_phase_pair: dict[tuple[str, str, str], _LatentValue] = {}
        self.tls_time_phase_pair: dict[
            tuple[str, int, str, str], _LatentValue
        ] = {}
        self.calibration_error = 1.0
        self.fitted = False

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        metadata = self._metadata(dataset)
        target, target_scale_summary = group_normalized_action_target(
            dataset, self.target_name
        )
        phase_raw = _aggregate(
            zip(
                metadata["reference_states"],
                metadata["candidate_states"],
                strict=True,
            ),
            target,
        )
        self.phase_pair = {
            key: _LatentValue(
                mean=float(values["sum"] / values["count"]),
                count=int(values["count"]),
                trust=1.0,
            )
            for key, values in phase_raw.items()
        }
        if self.config.scope in {
            TLS_PHASE_PAIR_SCOPE,
            TLS_TIME_PHASE_PAIR_SCOPE,
        }:
            tls_keys = zip(
                metadata["row_tls"],
                metadata["reference_states"],
                metadata["candidate_states"],
                strict=True,
            )
            self.tls_phase_pair = self._shrink_level(
                _aggregate(tls_keys, target),
                parent=lambda key: self.phase_pair.get((key[1], key[2])),
            )
        if self.config.scope == TLS_TIME_PHASE_PAIR_SCOPE:
            bins = np.floor(
                metadata["row_times"] / float(self.config.time_bin_sec)
            ).astype(int)
            time_keys = zip(
                metadata["row_tls"],
                bins,
                metadata["reference_states"],
                metadata["candidate_states"],
                strict=True,
            )
            self.tls_time_phase_pair = self._shrink_level(
                _aggregate(time_keys, target),
                parent=lambda key: self.tls_phase_pair.get(
                    (key[0], key[2], key[3])
                )
                or self.phase_pair.get((key[2], key[3])),
            )
        self.fitted = True
        fitted = self._predict_arrays(dataset)
        is_reference = metadata["is_reference"]
        calibration_rows = ~is_reference
        residual = np.abs(target[calibration_rows] - fitted["mean"][calibration_rows])
        self.calibration_error = max(
            float(np.quantile(residual, self.config.uncertainty_quantile))
            if residual.size
            else 0.0,
            1e-6,
        )
        return {
            "protocol": LATENT_PROTOCOL,
            "target_name": self.target_name,
            "config": asdict(self.config),
            "row_count": int(dataset.size),
            "action_group_count": int(
                np.unique(metadata["action_group_ids"]).size
            ),
            "phase_pair_count": len(self.phase_pair),
            "tls_phase_pair_count": len(self.tls_phase_pair),
            "tls_time_phase_pair_count": len(self.tls_time_phase_pair),
            "target_scale_summary": target_scale_summary,
            "training_nonreference_mae": float(np.mean(residual))
            if residual.size
            else 0.0,
            "calibration_error": float(self.calibration_error),
            "uses_future_outcomes_at_prediction": False,
            "uses_target_labels_at_fit": True,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.fitted:
            raise RuntimeError("spatiotemporal action latent has not been fitted")
        values = self._predict_arrays(dataset)
        uncertainty = self.calibration_error * (
            1.0 + 1.0 / np.sqrt(np.maximum(values["support_count"], 1.0))
        )
        is_reference = self._metadata(dataset)["is_reference"]
        uncertainty[is_reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": values["mean"],
                "latent_residual": values["mean"],
                "mean": values["mean"],
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(values["trust"], 0.0, 1.0),
                "context_distance": 1.0 - np.clip(values["trust"], 0.0, 1.0),
                "context_support": np.clip(values["trust"], 0.0, 1.0),
                "latent_support_count": values["support_count"],
                "latent_level": values["level"],
            }
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": LATENT_PROTOCOL,
            "config": asdict(self.config),
            "target_name": self.target_name,
            "phase_pair_count": len(self.phase_pair),
            "tls_phase_pair_count": len(self.tls_phase_pair),
            "tls_time_phase_pair_count": len(self.tls_time_phase_pair),
            "calibration_error": float(self.calibration_error),
            "fitted": bool(self.fitted),
        }

    def _shrink_level(
        self,
        raw: Mapping[tuple[Any, ...], Mapping[str, float]],
        *,
        parent: Any,
    ) -> dict[tuple[Any, ...], _LatentValue]:
        result = {}
        shrinkage = float(self.config.shrinkage)
        for key, values in raw.items():
            parent_value = parent(key)
            parent_mean = float(parent_value.mean) if parent_value is not None else 0.0
            count = int(values["count"])
            denominator = float(count) + shrinkage
            mean = (
                float(values["sum"]) + shrinkage * parent_mean
            ) / max(denominator, 1.0)
            result[key] = _LatentValue(
                mean=float(mean),
                count=count,
                trust=float(count / max(denominator, 1.0)),
            )
        return result

    def _predict_arrays(self, dataset: MechanismDataset) -> dict[str, np.ndarray]:
        metadata = self._metadata(dataset)
        means = np.zeros(dataset.size, dtype=float)
        trusts = np.zeros(dataset.size, dtype=float)
        counts = np.zeros(dataset.size, dtype=float)
        levels = np.zeros(dataset.size, dtype=float)
        bins = (
            np.floor(metadata["row_times"] / float(self.config.time_bin_sec)).astype(int)
            if self.config.scope == TLS_TIME_PHASE_PAIR_SCOPE
            else np.zeros(dataset.size, dtype=int)
        )
        for row in range(dataset.size):
            phase_key = (
                str(metadata["reference_states"][row]),
                str(metadata["candidate_states"][row]),
            )
            value = self.phase_pair.get(phase_key)
            level = 1.0 if value is not None else 0.0
            if self.config.scope in {
                TLS_PHASE_PAIR_SCOPE,
                TLS_TIME_PHASE_PAIR_SCOPE,
            }:
                tls_key = (str(metadata["row_tls"][row]), *phase_key)
                tls_value = self.tls_phase_pair.get(tls_key)
                if tls_value is not None:
                    value = tls_value
                    level = 2.0
            if self.config.scope == TLS_TIME_PHASE_PAIR_SCOPE:
                time_key = (
                    str(metadata["row_tls"][row]),
                    int(bins[row]),
                    *phase_key,
                )
                time_value = self.tls_time_phase_pair.get(time_key)
                if time_value is not None:
                    value = time_value
                    level = 3.0
            if value is not None:
                means[row] = float(value.mean)
                trusts[row] = float(value.trust)
                counts[row] = float(value.count)
                levels[row] = level
        is_reference = metadata["is_reference"]
        means[is_reference] = 0.0
        trusts[is_reference] = 1.0
        counts[is_reference] = np.maximum(counts[is_reference], 1.0)
        return {
            "mean": means,
            "trust": trusts,
            "support_count": counts,
            "level": levels,
        }

    @staticmethod
    def _metadata(dataset: MechanismDataset) -> dict[str, np.ndarray]:
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
        is_reference = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
        candidate_states = np.asarray(
            dataset.metadata.get("candidate_states", ()), dtype=str
        )
        row_tls = np.asarray(dataset.metadata.get("row_tls", ()), dtype=str)
        row_times = np.asarray(dataset.metadata.get("row_times", ()), dtype=float)
        if not all(
            values.shape == (dataset.size,)
            for values in (groups, is_reference, candidate_states, row_tls, row_times)
        ):
            raise ValueError("spatiotemporal latent requires row-aligned action metadata")
        reference_by_group: dict[str, str] = {}
        reference_counts: dict[str, int] = {}
        for group, reference, candidate_state in zip(
            groups, is_reference, candidate_states, strict=True
        ):
            if not bool(reference):
                continue
            key = str(group)
            reference_by_group[key] = str(candidate_state)
            reference_counts[key] = reference_counts.get(key, 0) + 1
        unique_groups = {str(value) for value in groups}
        invalid = [
            group
            for group in unique_groups
            if reference_counts.get(group, 0) != 1
        ]
        if invalid:
            raise ValueError(
                f"spatiotemporal action groups require one reference: {invalid[:3]}"
            )
        reference_states = np.asarray(
            [reference_by_group[str(group)] for group in groups], dtype=str
        )
        return {
            "action_group_ids": groups,
            "is_reference": is_reference,
            "candidate_states": candidate_states,
            "reference_states": reference_states,
            "row_tls": row_tls,
            "row_times": row_times,
        }


def _aggregate(
    keys: Any,
    values: np.ndarray,
) -> dict[tuple[Any, ...], dict[str, float]]:
    result: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, value in zip(keys, np.asarray(values, dtype=float), strict=True):
        normalized_key = tuple(key)
        row = result.setdefault(normalized_key, {"count": 0.0, "sum": 0.0})
        row["count"] += 1.0
        row["sum"] += float(value)
    return result
