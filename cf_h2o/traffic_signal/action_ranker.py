"""Source-balanced pairwise action ranking for safe pressure overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cf_h2o.traffic_signal.action_scaling import (
    ACTION_TARGET_SCALE_ABSOLUTE_FLOOR,
    ACTION_TARGET_SCALE_PROTOCOL,
    ACTION_TARGET_SCALE_RELATIVE_FLOOR,
    GROUP_ACTION_REGRET_SCALE_PROTOCOL,
    action_group_range,
    domain_action_scale,
)
from cf_h2o.traffic_signal.dynamic_graph_context import (
    SIGNAL_GRAPH_ACTION_FEATURES,
    SIGNAL_GRAPH_STATE_FEATURES,
)
from cf_h2o.traffic_signal.action_contrast import DEFAULT_CONTRAST_FEATURES
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.context_reference import fit_context_reference_geometry


CAUSAL_MOVEMENT_FEATURES = (
    "green_q_mean",
    "green_q_max",
    "green_q_cv",
    "red_q_max",
    "red_q_cv",
    "green_speed_mean",
    "red_speed_mean",
    "green_occ_max",
    "red_occ_max",
    "green_down_q_max",
    "green_down_occ_max",
    "green_down_occ_std",
    "green_movement_ratio",
    "green_outgoing_ratio",
    "green_movement_down_q_mean",
    "green_movement_down_occ_mean",
    "green_movement_down_occ_max",
    "movement_service_pressure",
    "queue_concentration",
)


CAUSAL_RIGID_RANKING_PARENTS = (
    "total_q",
    "total_veh",
    "mean_speed",
    "mean_occ",
    "green_q",
    "reference_green_q",
    "delta_green_q",
    "red_q",
    "reference_red_q",
    "delta_red_q",
    "green_down_q",
    "reference_green_down_q",
    "delta_green_down_q",
    "green_down_occ",
    "reference_green_down_occ",
    "delta_green_down_occ",
    "service_pressure",
    "reference_service_pressure",
    "delta_service_pressure",
    "red_pressure",
    "delta_red_pressure",
    "switch_indicator",
    "reference_switch_indicator",
    "delta_switch_indicator",
    "clearance_fraction",
    "reference_clearance_fraction",
    "delta_clearance_fraction",
    "current_phase_overlap",
    "delta_current_phase_overlap",
)


CAUSAL_CORE_RANKING_PARENTS = (
    *CAUSAL_RIGID_RANKING_PARENTS,
    *SIGNAL_GRAPH_STATE_FEATURES,
    *(
        name
        for feature in SIGNAL_GRAPH_ACTION_FEATURES
        for name in (feature, f"reference_{feature}", f"delta_{feature}")
    ),
)


CAUSAL_RANKING_PARENTS = (
    *CAUSAL_CORE_RANKING_PARENTS,
    *(
        name
        for feature in CAUSAL_MOVEMENT_FEATURES
        for name in (feature, f"reference_{feature}", f"delta_{feature}")
    ),
)


DOMAIN_ACTION_TARGET_NORMALIZATION_PROTOCOL = "domain_action_scale_v1"
GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL = "group_action_range_v1"
ACTION_TARGET_NORMALIZATION_PROTOCOLS = {
    DOMAIN_ACTION_TARGET_NORMALIZATION_PROTOCOL,
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
}


PAIRWISE_CAUSAL_STATE_PARENTS = (
    "total_q",
    "total_veh",
    "mean_speed",
    "mean_occ",
    "lane_count_norm",
    "candidate_count_norm",
    "time_sin",
    "time_cos",
    "current_green_elapsed_norm",
    "queue_concentration",
    *SIGNAL_GRAPH_STATE_FEATURES,
)


PAIRWISE_CAUSAL_ACTION_PARENTS = (
    "phase_duration_norm",
    *(name for name in DEFAULT_CONTRAST_FEATURES if name != "queue_concentration"),
    *SIGNAL_GRAPH_ACTION_FEATURES,
)


@dataclass(frozen=True)
class ActionRankerConfig:
    learning_rate: float = 0.05
    max_iter: int = 60
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 15
    l2_regularization: float = 8.0
    tie_scale_fraction: float = 0.05
    uncertainty_quantile: float = 0.90
    random_state: int = 20260803


@dataclass(frozen=True)
class ActionAdvantageConfig:
    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 12
    l2_regularization: float = 10.0
    target_clip: float = 5.0
    uncertainty_quantile: float = 0.90
    random_state: int = 20260803
    candidate_only: bool = False
    balance_candidate_signs: bool = False
    max_sign_weight_multiplier: float = 4.0
    target_normalization_protocol: str = (
        DOMAIN_ACTION_TARGET_NORMALIZATION_PROTOCOL
    )


@dataclass(frozen=True)
class PairwisePreferenceConfig:
    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 10.0
    uncertainty_quantile: float = 0.90
    random_state: int = 20260803


class PairwiseActionRanker:
    """Predict whether an action improves on its same-state rule reference."""

    def __init__(self, *, causal: bool, config: ActionRankerConfig | None = None) -> None:
        self.causal = bool(causal)
        self.config = config or ActionRankerConfig()
        self.feature_names: tuple[str, ...] = ()
        self.feature_indices = np.zeros(0, dtype=int)
        self.include_context = not self.causal
        self.model: HistGradientBoostingClassifier | None = None
        self.constant_probability = 0.5
        self.advantage_scale = 1.0
        self.calibration_error = 1.0
        self.source_contexts = np.zeros((0, 0), dtype=float)
        self.context_mean = np.zeros(0, dtype=float)
        self.context_scale = np.ones(0, dtype=float)

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        is_reference = _reference_mask(dataset)
        candidates = ~is_reference
        if not np.any(candidates):
            raise ValueError("pairwise action ranker requires non-reference candidates")
        if self.causal:
            self.feature_names = tuple(name for name in CAUSAL_RANKING_PARENTS if name in dataset.feature_names)
        else:
            self.feature_names = tuple(dataset.feature_names)
        self.feature_indices = _feature_indices(dataset.feature_names, self.feature_names)
        features = self._features(dataset)[candidates]
        target = np.asarray(dataset.targets["interval_cost"], dtype=float)[candidates]
        domains = np.asarray(dataset.domains)[candidates]
        groups = np.asarray(dataset.metadata["action_group_ids"])[candidates]
        group_scale = _group_scales(dataset)[candidates]
        normalized = target / group_scale
        tie_margin = max(
            float(np.quantile(np.abs(normalized), 0.25)) * self.config.tie_scale_fraction,
            1e-6,
        )
        labels = normalized < -tie_margin
        weights = _domain_group_balanced_weights(domains, groups)
        weights *= np.clip(np.abs(normalized) / max(tie_margin, 1e-6), 0.25, 4.0)
        weights *= weights.size / max(float(np.sum(weights)), 1e-12)
        self.advantage_scale = max(float(np.median(np.abs(target))), 0.25)
        if np.unique(labels).size < 2:
            self.model = None
            self.constant_probability = float(np.mean(labels))
        else:
            self.model = HistGradientBoostingClassifier(
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                early_stopping=False,
                random_state=self.config.random_state,
            )
            self.model.fit(features, labels.astype(int), sample_weight=weights)
            self.constant_probability = float(np.average(labels, weights=weights))
        probability = self._probability(dataset)[candidates]
        score = (0.5 - probability) * 2.0 * self.advantage_scale
        self.calibration_error = max(
            float(np.quantile(np.abs(target - score), self.config.uncertainty_quantile)),
            1e-6,
        )
        context_geometry = fit_context_reference_geometry(
            dataset.context, dataset.domains
        )
        self.source_contexts = context_geometry.support_anchors
        self.context_mean = context_geometry.mean
        self.context_scale = context_geometry.scale
        predicted_labels = probability >= 0.5
        return {
            "target_name": "interval_cost",
            "estimator": "HistGradientBoostingClassifier" if self.model is not None else "constant",
            "causal": self.causal,
            "feature_names": list(self.feature_names),
            "feature_count": int(features.shape[1]),
            "candidate_rows": int(features.shape[0]),
            "positive_rate": float(np.mean(labels)),
            "weighted_positive_rate": float(np.average(labels, weights=weights)),
            "training_balanced_accuracy": _balanced_accuracy(labels, predicted_labels),
            "tie_margin_normalized": float(tie_margin),
            "advantage_scale": float(self.advantage_scale),
            "calibration_error": float(self.calibration_error),
            "context_reference": context_geometry.diagnostics(),
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        probability = self._probability(dataset)
        score = (0.5 - probability) * 2.0 * self.advantage_scale
        reference = _reference_mask(dataset)
        score[reference] = 0.0
        ambiguity = 1.0 - 2.0 * np.abs(probability - 0.5)
        distance = _context_distance(
            dataset.context,
            self.source_contexts,
            self.context_mean,
            self.context_scale,
        )
        trust = np.exp(-distance)
        support = _context_support(dataset.context, self.source_contexts, self.context_scale)
        trust = np.minimum(trust, support)
        if self.include_context:
            distance = np.zeros(dataset.size, dtype=float)
            trust = np.ones(dataset.size, dtype=float)
            support = np.ones(dataset.size, dtype=float)
        uncertainty = self.calibration_error * (0.25 + 0.75 * ambiguity) * (1.0 + 0.25 * distance)
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": score,
                "latent_residual": np.zeros(dataset.size, dtype=float),
                "mean": score,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": support,
                "improvement_probability": probability,
            }
        }

    def _features(self, dataset: MechanismDataset) -> np.ndarray:
        features = np.asarray(dataset.features[:, self.feature_indices], dtype=float)
        if self.include_context:
            features = np.concatenate([features, np.asarray(dataset.context, dtype=float)], axis=1)
        return features

    def _probability(self, dataset: MechanismDataset) -> np.ndarray:
        if self.model is None:
            return np.full(dataset.size, self.constant_probability, dtype=float)
        return np.asarray(self.model.predict_proba(self._features(dataset))[:, 1], dtype=float)


class PairwiseActionAdvantageRegressor:
    """Regress signed action-minus-prior cost with equal-domain weighting."""

    def __init__(
        self,
        *,
        causal: bool,
        config: ActionAdvantageConfig | None = None,
        causal_feature_names: Sequence[str] | None = None,
    ) -> None:
        self.causal = bool(causal)
        self.config = config or ActionAdvantageConfig()
        self.causal_feature_names = tuple(causal_feature_names or CAUSAL_RANKING_PARENTS)
        self.feature_names: tuple[str, ...] = ()
        self.feature_indices = np.zeros(0, dtype=int)
        self.include_context = not self.causal
        self.model: HistGradientBoostingRegressor | None = None
        self.constant_score = 0.0
        self.calibration_error = 1.0
        self.source_contexts = np.zeros((0, 0), dtype=float)
        self.context_mean = np.zeros(0, dtype=float)
        self.context_scale = np.ones(0, dtype=float)

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        if (
            self.config.target_normalization_protocol
            not in ACTION_TARGET_NORMALIZATION_PROTOCOLS
        ):
            raise ValueError(
                "unknown action-target normalization protocol: "
                f"{self.config.target_normalization_protocol!r}"
            )
        if self.causal:
            self.feature_names = tuple(
                name for name in self.causal_feature_names if name in dataset.feature_names
            )
        else:
            self.feature_names = tuple(dataset.feature_names)
        self.feature_indices = _feature_indices(dataset.feature_names, self.feature_names)
        features_all = self._features(dataset)
        target = np.asarray(dataset.targets["interval_cost"], dtype=float)
        domains = np.asarray(dataset.domains)
        groups = np.asarray(dataset.metadata["action_group_ids"])
        if (
            self.config.target_normalization_protocol
            == GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
        ):
            normalized_all, group_scale_summary = group_normalized_action_target(
                dataset,
                "interval_cost",
                target_clip=float(self.config.target_clip),
            )
            domain_scales: dict[str, float] = {}
            target_scale_protocol = GROUP_ACTION_REGRET_SCALE_PROTOCOL
        else:
            normalized_all, domain_scales = domain_normalized_action_target(
                dataset,
                "interval_cost",
                target_clip=float(self.config.target_clip),
            )
            group_scale_summary = None
            target_scale_protocol = ACTION_TARGET_SCALE_PROTOCOL
        reference = _reference_mask(dataset)
        fit_mask = ~reference if self.config.candidate_only else np.ones(dataset.size, dtype=bool)
        features = features_all[fit_mask]
        normalized = normalized_all[fit_mask]
        fit_domains = domains[fit_mask]
        fit_groups = groups[fit_mask]
        weights = _domain_group_balanced_weights(fit_domains, fit_groups)
        weights *= np.clip(np.abs(normalized), 0.25, 4.0)
        sign_balance = {
            "enabled": False,
            "negative_weight_multiplier": 1.0,
            "nonnegative_weight_multiplier": 1.0,
        }
        if self.config.balance_candidate_signs:
            negative = normalized < -1e-8
            if np.any(negative) and np.any(~negative):
                total = float(np.sum(weights))
                cap = max(float(self.config.max_sign_weight_multiplier), 1.0)
                negative_multiplier = float(
                    np.clip(0.5 * total / float(np.sum(weights[negative])), 1.0 / cap, cap)
                )
                nonnegative_multiplier = float(
                    np.clip(0.5 * total / float(np.sum(weights[~negative])), 1.0 / cap, cap)
                )
                weights *= np.where(negative, negative_multiplier, nonnegative_multiplier)
                sign_balance = {
                    "enabled": True,
                    "negative_weight_multiplier": negative_multiplier,
                    "nonnegative_weight_multiplier": nonnegative_multiplier,
                }
        weights *= weights.size / max(float(np.sum(weights)), 1e-12)
        if float(np.std(normalized)) < 1e-8:
            self.model = None
            self.constant_score = float(np.average(normalized, weights=weights))
            fitted = np.full(normalized.shape, self.constant_score, dtype=float)
        else:
            self.model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                early_stopping=False,
                random_state=self.config.random_state,
            )
            self.model.fit(features, normalized, sample_weight=weights)
            self.constant_score = float(np.average(normalized, weights=weights))
            fitted = np.asarray(self.model.predict(features), dtype=float)
        self.calibration_error = max(
            float(
                np.quantile(
                    np.abs(normalized - fitted),
                    self.config.uncertainty_quantile,
                )
            ),
            1e-6,
        )
        context_geometry = fit_context_reference_geometry(
            dataset.context, dataset.domains
        )
        self.source_contexts = context_geometry.support_anchors
        self.context_mean = context_geometry.mean
        self.context_scale = context_geometry.scale
        fitted_all = (
            np.full(dataset.size, self.constant_score, dtype=float)
            if self.model is None
            else np.asarray(self.model.predict(features_all), dtype=float)
        )
        sign_mask = ~reference & (np.abs(normalized_all) > 1e-6)
        sign_accuracy = (
            float(
                np.mean(
                    (fitted_all[sign_mask] < 0.0)
                    == (normalized_all[sign_mask] < 0.0)
                )
            )
            if np.any(sign_mask)
            else 1.0
        )
        return {
            "target_name": "interval_cost",
            "estimator": "HistGradientBoostingRegressor" if self.model is not None else "constant",
            "causal": self.causal,
            "feature_names": list(self.feature_names),
            "feature_count": int(features.shape[1]),
            "rows": int(dataset.size),
            "training_rows": int(features.shape[0]),
            "candidate_only": bool(self.config.candidate_only),
            "candidate_negative_rate": float(
                np.mean(normalized_all[~reference] < -1e-8)
            )
            if np.any(~reference)
            else 0.0,
            "sign_balance": sign_balance,
            "domain_target_scales": domain_scales,
            "domain_target_scale_protocol": target_scale_protocol,
            "group_target_scale_summary": group_scale_summary,
            "target_normalization_protocol": (
                self.config.target_normalization_protocol
            ),
            "training_weighted_mae": float(np.average(np.abs(normalized - fitted), weights=weights)),
            "training_nonzero_sign_accuracy": sign_accuracy,
            "calibration_error": float(self.calibration_error),
            "context_reference": context_geometry.diagnostics(),
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if self.model is None:
            score = np.full(dataset.size, self.constant_score, dtype=float)
        else:
            score = np.asarray(self.model.predict(self._features(dataset)), dtype=float)
        reference = _reference_mask(dataset)
        score[reference] = 0.0
        distance = _context_distance(
            dataset.context,
            self.source_contexts,
            self.context_mean,
            self.context_scale,
        )
        trust = np.exp(-distance)
        support = _context_support(dataset.context, self.source_contexts, self.context_scale)
        trust = np.minimum(trust, support)
        if self.include_context:
            distance = np.zeros(dataset.size, dtype=float)
            trust = np.ones(dataset.size, dtype=float)
            support = np.ones(dataset.size, dtype=float)
        uncertainty = self.calibration_error * (1.0 + 0.25 * distance)
        uncertainty = np.full(dataset.size, uncertainty, dtype=float) if np.ndim(uncertainty) == 0 else uncertainty
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": score,
                "latent_residual": np.zeros(dataset.size, dtype=float),
                "mean": score,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": support,
            }
        }

    def _features(self, dataset: MechanismDataset) -> np.ndarray:
        indices = _feature_indices(dataset.feature_names, self.feature_names)
        features = np.asarray(dataset.features[:, indices], dtype=float)
        if self.include_context:
            features = np.concatenate([features, np.asarray(dataset.context, dtype=float)], axis=1)
        return features


class AntisymmetricPairwiseActionRegressor:
    """Rank all actions using group-normalized antisymmetric cost differences."""

    def __init__(
        self,
        *,
        config: PairwisePreferenceConfig | None = None,
        state_feature_names: Sequence[str] = PAIRWISE_CAUSAL_STATE_PARENTS,
        action_feature_names: Sequence[str] = PAIRWISE_CAUSAL_ACTION_PARENTS,
    ) -> None:
        self.config = config or PairwisePreferenceConfig()
        self.requested_state_feature_names = tuple(state_feature_names)
        self.requested_action_feature_names = tuple(action_feature_names)
        self.state_feature_names: tuple[str, ...] = ()
        self.action_feature_names: tuple[str, ...] = ()
        self.state_feature_indices = np.zeros(0, dtype=int)
        self.action_feature_indices = np.zeros(0, dtype=int)
        self.model: HistGradientBoostingRegressor | None = None
        self.constant_score = 0.0
        self.calibration_error = 1.0
        self.source_contexts = np.zeros((0, 0), dtype=float)
        self.context_mean = np.zeros(0, dtype=float)
        self.context_scale = np.ones(0, dtype=float)

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        index = {name: position for position, name in enumerate(dataset.feature_names)}
        self.state_feature_names = tuple(
            name for name in self.requested_state_feature_names if name in index
        )
        self.action_feature_names = tuple(
            name for name in self.requested_action_feature_names if name in index
        )
        if not self.state_feature_names or not self.action_feature_names:
            raise ValueError("pairwise action model requires state and action parents")
        self.state_feature_indices = np.asarray(
            [index[name] for name in self.state_feature_names], dtype=int
        )
        self.action_feature_indices = np.asarray(
            [index[name] for name in self.action_feature_names], dtype=int
        )
        design, left_rows, right_rows, pair_groups = self._pair_design(dataset)
        target = np.asarray(dataset.targets["interval_cost"], dtype=float)
        groups = np.asarray(dataset.metadata["action_group_ids"])
        domains = np.asarray(dataset.domains, dtype=str)
        pair_target = np.zeros(left_rows.size, dtype=float)
        pair_weights = np.zeros(left_rows.size, dtype=float)
        unique_domains = np.unique(domains)
        domain_group_counts = {
            domain: int(np.unique(groups[domains == domain]).size)
            for domain in unique_domains
        }
        scales = []
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            group_domains = np.unique(domains[rows])
            if group_domains.size != 1:
                raise ValueError(f"action group {group!r} crosses source domains")
            pair_mask = pair_groups == group
            pair_count = int(np.sum(pair_mask))
            if pair_count <= 0:
                continue
            scale = action_group_range(target[rows])
            scales.append(float(scale))
            pair_target[pair_mask] = (
                target[left_rows[pair_mask]] - target[right_rows[pair_mask]]
            ) / scale
            domain = str(group_domains[0])
            pair_weights[pair_mask] = 1.0 / max(
                len(unique_domains)
                * domain_group_counts[domain]
                * pair_count,
                1,
            )
        oriented_design = np.vstack(
            [
                design,
                np.column_stack(
                    [
                        design[:, : len(self.state_feature_names)],
                        -design[:, len(self.state_feature_names) :],
                    ]
                ),
            ]
        )
        oriented_target = np.concatenate([pair_target, -pair_target])
        oriented_weights = np.concatenate([pair_weights, pair_weights]) * 0.5
        oriented_weights *= oriented_weights.size / max(
            float(np.sum(oriented_weights)), 1e-12
        )
        if float(np.std(oriented_target)) < 1e-8:
            self.model = None
            self.constant_score = 0.0
            fitted = np.zeros(oriented_target.shape, dtype=float)
        else:
            self.model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                early_stopping=False,
                random_state=self.config.random_state,
            )
            self.model.fit(
                oriented_design,
                oriented_target,
                sample_weight=oriented_weights,
            )
            fitted = np.asarray(self.model.predict(oriented_design), dtype=float)
        self.calibration_error = max(
            float(
                np.quantile(
                    np.abs(oriented_target - fitted),
                    self.config.uncertainty_quantile,
                )
            ),
            1e-6,
        )
        context_geometry = fit_context_reference_geometry(
            dataset.context, dataset.domains
        )
        self.source_contexts = context_geometry.support_anchors
        self.context_mean = context_geometry.mean
        self.context_scale = context_geometry.scale
        scale_values = np.asarray(scales, dtype=float)
        return {
            "target_name": "interval_cost",
            "estimator": (
                "HistGradientBoostingRegressor"
                if self.model is not None
                else "constant"
            ),
            "causal": True,
            "preference_protocol": "all_action_antisymmetric_pairwise_v1",
            "target_normalization_protocol": (
                GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
            ),
            "target_scale_protocol": GROUP_ACTION_REGRET_SCALE_PROTOCOL,
            "state_feature_names": list(self.state_feature_names),
            "action_feature_names": list(self.action_feature_names),
            "state_feature_count": len(self.state_feature_names),
            "action_feature_count": len(self.action_feature_names),
            "uses_context_features": False,
            "source_domain_count": int(unique_domains.size),
            "action_group_count": int(np.unique(groups).size),
            "unordered_pair_count": int(left_rows.size),
            "oriented_pair_count": int(oriented_target.size),
            "antisymmetric_augmentation": True,
            "pair_target_minimum": float(np.min(oriented_target)),
            "pair_target_maximum": float(np.max(oriented_target)),
            "training_weighted_mae": float(
                np.average(np.abs(oriented_target - fitted), weights=oriented_weights)
            ),
            "calibration_error": float(self.calibration_error),
            "group_target_scale_summary": {
                "group_count": int(scale_values.size),
                "minimum": float(np.min(scale_values)),
                "median": float(np.median(scale_values)),
                "maximum": float(np.max(scale_values)),
            },
            "context_reference": context_geometry.diagnostics(),
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.state_feature_names or not self.action_feature_names:
            raise RuntimeError("pairwise action model has not been fitted")
        index = {name: position for position, name in enumerate(dataset.feature_names)}
        missing = [
            name
            for name in (*self.state_feature_names, *self.action_feature_names)
            if name not in index
        ]
        if missing:
            raise KeyError(f"pairwise action prediction missing features: {missing}")
        self.state_feature_indices = np.asarray(
            [index[name] for name in self.state_feature_names], dtype=int
        )
        self.action_feature_indices = np.asarray(
            [index[name] for name in self.action_feature_names], dtype=int
        )
        design, left_rows, right_rows, _ = self._pair_design(dataset)
        reverse_design = np.column_stack(
            [
                design[:, : len(self.state_feature_names)],
                -design[:, len(self.state_feature_names) :],
            ]
        )
        if self.model is None:
            pair_difference = np.zeros(left_rows.size, dtype=float)
        else:
            forward = np.asarray(self.model.predict(design), dtype=float)
            reverse = np.asarray(self.model.predict(reverse_design), dtype=float)
            pair_difference = 0.5 * (forward - reverse)
        score = np.zeros(dataset.size, dtype=float)
        pair_counts = np.zeros(dataset.size, dtype=float)
        np.add.at(score, left_rows, pair_difference)
        np.add.at(score, right_rows, -pair_difference)
        np.add.at(pair_counts, left_rows, 1.0)
        np.add.at(pair_counts, right_rows, 1.0)
        score /= np.maximum(pair_counts, 1.0)

        implied_difference = score[left_rows] - score[right_rows]
        pair_inconsistency = np.abs(pair_difference - implied_difference)
        row_inconsistency = np.zeros(dataset.size, dtype=float)
        np.add.at(row_inconsistency, left_rows, pair_inconsistency)
        np.add.at(row_inconsistency, right_rows, pair_inconsistency)
        row_inconsistency /= np.maximum(pair_counts, 1.0)

        groups = np.asarray(dataset.metadata["action_group_ids"])
        reference = _reference_mask(dataset)
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            group_reference = rows[reference[rows]]
            if group_reference.size != 1:
                raise ValueError(
                    f"pairwise action group {group!r} requires one reference row"
                )
            score[rows] -= score[int(group_reference[0])]
        distance = _context_distance(
            dataset.context,
            self.source_contexts,
            self.context_mean,
            self.context_scale,
        )
        support = _context_support(
            dataset.context,
            self.source_contexts,
            self.context_scale,
        )
        trust = np.minimum(np.exp(-distance), support)
        uncertainty = (
            self.calibration_error + row_inconsistency
        ) * (1.0 + 0.25 * distance)
        score[reference] = 0.0
        uncertainty[reference] = 0.0
        row_inconsistency[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": score,
                "latent_residual": np.zeros(dataset.size, dtype=float),
                "mean": score,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": support,
                "pairwise_cycle_inconsistency": row_inconsistency,
                "pairwise_comparison_count": pair_counts,
            }
        }

    def _pair_design(
        self,
        dataset: MechanismDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
        if groups.shape != (dataset.size,):
            raise ValueError("pairwise action model requires row-aligned groups")
        features = np.asarray(dataset.features, dtype=float)
        left_rows = []
        right_rows = []
        pair_groups = []
        designs = []
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            if rows.size < 2:
                raise ValueError(f"action group {group!r} has fewer than two actions")
            state = features[rows][:, self.state_feature_indices]
            if not np.allclose(state, state[0], rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"causal state parents vary across actions in group {group!r}"
                )
            actions = features[rows][:, self.action_feature_indices]
            for left_offset in range(rows.size - 1):
                for right_offset in range(left_offset + 1, rows.size):
                    left = int(rows[left_offset])
                    right = int(rows[right_offset])
                    designs.append(
                        np.concatenate(
                            [state[0], actions[left_offset] - actions[right_offset]]
                        )
                    )
                    left_rows.append(left)
                    right_rows.append(right)
                    pair_groups.append(group)
        if not designs:
            raise ValueError("pairwise action model found no action pairs")
        return (
            np.asarray(designs, dtype=float),
            np.asarray(left_rows, dtype=int),
            np.asarray(right_rows, dtype=int),
            np.asarray(pair_groups, dtype=groups.dtype),
        )


class CausalReferenceResidualRegressor:
    """Fit a causal residual directly against each group's rule reference."""

    def __init__(
        self,
        *,
        config: PairwisePreferenceConfig | None = None,
        state_feature_names: Sequence[str] = PAIRWISE_CAUSAL_STATE_PARENTS,
        action_feature_names: Sequence[str] = PAIRWISE_CAUSAL_ACTION_PARENTS,
    ) -> None:
        self.config = config or PairwisePreferenceConfig()
        self.requested_state_feature_names = tuple(state_feature_names)
        self.requested_action_feature_names = tuple(action_feature_names)
        self.state_feature_names: tuple[str, ...] = ()
        self.action_feature_names: tuple[str, ...] = ()
        self.state_feature_indices = np.zeros(0, dtype=int)
        self.action_feature_indices = np.zeros(0, dtype=int)
        self.model: HistGradientBoostingRegressor | None = None
        self.constant_score = 0.0
        self.calibration_error = 1.0
        self.source_contexts = np.zeros((0, 0), dtype=float)
        self.context_mean = np.zeros(0, dtype=float)
        self.context_scale = np.ones(0, dtype=float)

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        self._resolve_features(dataset)
        design, reference = self._design(dataset)
        target = np.asarray(dataset.targets["interval_cost"], dtype=float)
        candidate = ~reference
        if not np.any(candidate):
            raise ValueError("reference residual model found no candidate actions")
        weights = self._candidate_weights(dataset, candidate)
        candidate_target = target[candidate]
        if float(np.std(candidate_target)) < 1e-8:
            self.model = None
            self.constant_score = float(
                np.average(candidate_target, weights=weights)
            )
            fitted = np.full(candidate_target.shape, self.constant_score)
        else:
            self.model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=self.config.learning_rate,
                max_iter=self.config.max_iter,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                early_stopping=False,
                random_state=self.config.random_state,
            )
            self.model.fit(
                design[candidate],
                candidate_target,
                sample_weight=weights,
            )
            fitted = np.asarray(
                self.model.predict(design[candidate]), dtype=float
            )
        self.calibration_error = max(
            float(
                np.quantile(
                    np.abs(candidate_target - fitted),
                    self.config.uncertainty_quantile,
                )
            ),
            1e-6,
        )
        context_geometry = fit_context_reference_geometry(
            dataset.context, dataset.domains
        )
        self.source_contexts = context_geometry.support_anchors
        self.context_mean = context_geometry.mean
        self.context_scale = context_geometry.scale
        groups = np.asarray(dataset.metadata["action_group_ids"])
        return {
            "target_name": "interval_cost",
            "estimator": (
                "HistGradientBoostingRegressor"
                if self.model is not None
                else "constant"
            ),
            "causal": True,
            "residual_protocol": "causal_reference_contrast_residual_v1",
            "state_feature_names": list(self.state_feature_names),
            "action_feature_names": list(self.action_feature_names),
            "state_feature_count": len(self.state_feature_names),
            "action_feature_count": len(self.action_feature_names),
            "uses_context_features": False,
            "source_domain_count": int(
                np.unique(np.asarray(dataset.domains, dtype=str)).size
            ),
            "action_group_count": int(np.unique(groups).size),
            "candidate_row_count": int(np.count_nonzero(candidate)),
            "training_weighted_mae": float(
                np.average(
                    np.abs(candidate_target - fitted), weights=weights
                )
            ),
            "calibration_error": float(self.calibration_error),
            "context_reference": context_geometry.diagnostics(),
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.state_feature_names or not self.action_feature_names:
            raise RuntimeError("reference residual model has not been fitted")
        self._resolve_features(dataset, require_fitted_names=True)
        design, reference = self._design(dataset)
        if self.model is None:
            score = np.full(dataset.size, self.constant_score, dtype=float)
        else:
            score = np.asarray(self.model.predict(design), dtype=float)
        distance = _context_distance(
            dataset.context,
            self.source_contexts,
            self.context_mean,
            self.context_scale,
        )
        support = _context_support(
            dataset.context,
            self.source_contexts,
            self.context_scale,
        )
        trust = np.minimum(np.exp(-distance), support)
        uncertainty = self.calibration_error * (1.0 + 0.25 * distance)
        score[reference] = 0.0
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": score,
                "latent_residual": np.zeros(dataset.size, dtype=float),
                "mean": score,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": support,
            }
        }

    def _resolve_features(
        self,
        dataset: MechanismDataset,
        *,
        require_fitted_names: bool = False,
    ) -> None:
        index = {
            name: position for position, name in enumerate(dataset.feature_names)
        }
        if require_fitted_names:
            missing = [
                name
                for name in (*self.state_feature_names, *self.action_feature_names)
                if name not in index
            ]
            if missing:
                raise KeyError(
                    f"reference residual prediction missing features: {missing}"
                )
        else:
            self.state_feature_names = tuple(
                name
                for name in self.requested_state_feature_names
                if name in index
            )
            self.action_feature_names = tuple(
                name
                for name in self.requested_action_feature_names
                if name in index
            )
            if not self.state_feature_names or not self.action_feature_names:
                raise ValueError(
                    "reference residual model requires state and action parents"
                )
        self.state_feature_indices = np.asarray(
            [index[name] for name in self.state_feature_names], dtype=int
        )
        self.action_feature_indices = np.asarray(
            [index[name] for name in self.action_feature_names], dtype=int
        )

    def _design(
        self, dataset: MechanismDataset
    ) -> tuple[np.ndarray, np.ndarray]:
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
        if groups.shape != (dataset.size,):
            raise ValueError("reference residual model requires row-aligned groups")
        reference = _reference_mask(dataset)
        features = np.asarray(dataset.features, dtype=float)
        design = np.empty(
            (
                dataset.size,
                len(self.state_feature_names) + len(self.action_feature_names),
            ),
            dtype=float,
        )
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            reference_rows = rows[reference[rows]]
            if reference_rows.size != 1:
                raise ValueError(
                    f"reference residual group {group!r} requires one reference row"
                )
            state = features[rows][:, self.state_feature_indices]
            if not np.allclose(state, state[0], rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"causal state parents vary across actions in group {group!r}"
                )
            action = features[rows][:, self.action_feature_indices]
            reference_action = features[
                int(reference_rows[0]), self.action_feature_indices
            ]
            design[rows] = np.column_stack(
                [
                    state,
                    action - reference_action,
                ]
            )
        return design, reference

    @staticmethod
    def _candidate_weights(
        dataset: MechanismDataset, candidate: np.ndarray
    ) -> np.ndarray:
        groups = np.asarray(dataset.metadata["action_group_ids"])
        domains = np.asarray(dataset.domains, dtype=str)
        candidate_rows = np.flatnonzero(candidate)
        unique_domains = np.unique(domains[candidate_rows])
        domain_group_counts = {
            domain: int(np.unique(groups[candidate & (domains == domain)]).size)
            for domain in unique_domains
        }
        group_candidate_counts = {
            group: int(np.count_nonzero(candidate & (groups == group)))
            for group in np.unique(groups[candidate])
        }
        weights = np.asarray(
            [
                1.0
                / max(
                    len(unique_domains)
                    * domain_group_counts[str(domains[row])]
                    * group_candidate_counts[groups[row]],
                    1,
                )
                for row in candidate_rows
            ],
            dtype=float,
        )
        weights *= weights.size / max(float(np.sum(weights)), 1e-12)
        return weights


class TargetAdaptedActionAdvantageRegressor:
    """Shrink a source causal advantage prior toward a low-capacity target head."""

    def __init__(
        self,
        *,
        target_domain: str | None,
        prior_group_strength: float = 20.0,
        max_target_weight: float = 0.80,
    ) -> None:
        self.target_domain = str(target_domain) if target_domain is not None else None
        self.prior_group_strength = float(prior_group_strength)
        self.max_target_weight = float(max_target_weight)
        self.source_model = PairwiseActionAdvantageRegressor(causal=True)
        self.target_model: PairwiseActionAdvantageRegressor | None = None
        self.target_weight = 0.0
        self.target_group_count = 0

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        domains = np.asarray(dataset.domains, dtype=str)
        target_mask = (
            domains == self.target_domain
            if self.target_domain is not None
            else np.zeros(dataset.size, dtype=bool)
        )
        source = _row_subset(dataset, ~target_mask) if np.any(target_mask) else dataset
        source_diagnostics = self.source_model.fit(source)
        target_diagnostics = None
        if np.any(target_mask):
            target = _row_subset(dataset, target_mask)
            self.target_group_count = int(
                np.unique(np.asarray(target.metadata["action_group_ids"])).size
            )
            if self.target_group_count >= 4:
                self.target_model = PairwiseActionAdvantageRegressor(
                    causal=True,
                    config=ActionAdvantageConfig(
                        max_iter=80,
                        max_leaf_nodes=7,
                        min_samples_leaf=4,
                        l2_regularization=16.0,
                    ),
                )
                target_diagnostics = self.target_model.fit(target)
                empirical_weight = self.target_group_count / (
                    self.target_group_count + max(self.prior_group_strength, 1e-12)
                )
                self.target_weight = min(float(empirical_weight), self.max_target_weight)
        return {
            "estimator": "source_prior_plus_shrunk_target_causal_advantage",
            "target_domain": self.target_domain,
            "target_group_count": self.target_group_count,
            "prior_group_strength": self.prior_group_strength,
            "max_target_weight": self.max_target_weight,
            "selected_target_weight": self.target_weight,
            "source": source_diagnostics,
            "target": target_diagnostics,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        source = self.source_model.predict(dataset)["control_cost"]
        if self.target_model is None or self.target_weight <= 0.0:
            return {"control_cost": source}
        target = self.target_model.predict(dataset)["control_cost"]
        weight = float(self.target_weight)
        mean = (1.0 - weight) * source["mean"] + weight * target["mean"]
        disagreement = np.abs(source["mean"] - target["mean"])
        uncertainty = (
            (1.0 - weight) * source["uncertainty"]
            + weight * target["uncertainty"]
            + weight * (1.0 - weight) * disagreement
        )
        trust = (
            (1.0 - weight) * source["context_trust"]
            + weight * target["context_trust"]
        )
        distance = (
            (1.0 - weight) * source["context_distance"]
            + weight * target["context_distance"]
        )
        reference = _reference_mask(dataset)
        mean[reference] = 0.0
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": mean,
                "latent_residual": weight * target["mean"],
                "mean": mean,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "source_target_disagreement": disagreement,
                "target_weight": np.full(dataset.size, weight, dtype=float),
            }
        }


class TargetOnlyActionAdvantageRegressor:
    """Fit an action-advantage head on target rows and no source rows."""

    def __init__(self, *, target_domain: str | None) -> None:
        self.target_domain = (
            str(target_domain) if target_domain is not None else None
        )
        self.target_group_count = 0
        self.model = PairwiseActionAdvantageRegressor(
            causal=True,
            config=ActionAdvantageConfig(
                max_iter=80,
                max_leaf_nodes=7,
                min_samples_leaf=4,
                l2_regularization=16.0,
            ),
        )

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        if self.target_domain is None:
            raise ValueError("target-only action model requires a target domain")
        domains = np.asarray(dataset.domains, dtype=str)
        target_mask = domains == self.target_domain
        if not np.any(target_mask):
            raise ValueError("target-only action model received no target rows")
        target = _row_subset(dataset, target_mask)
        self.target_group_count = int(
            np.unique(np.asarray(target.metadata["action_group_ids"])).size
        )
        if self.target_group_count < 4:
            raise ValueError("target-only action model requires at least four groups")
        diagnostics = self.model.fit(target)
        return {
            "estimator": "strict_target_only_causal_advantage_v2",
            "target_domain": self.target_domain,
            "target_group_count": self.target_group_count,
            "target_row_count": int(target.size),
            "source_row_count_consumed": 0,
            "target": diagnostics,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        return self.model.predict(dataset)


def _reference_mask(dataset: MechanismDataset) -> np.ndarray:
    values = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if values.shape != (dataset.size,):
        raise ValueError("pairwise action ranker requires row-aligned is_reference metadata")
    return values


def _group_scales(dataset: MechanismDataset) -> np.ndarray:
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
    target = np.asarray(dataset.targets["interval_cost"], dtype=float)
    result = np.ones(dataset.size, dtype=float)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        values = target[rows]
        result[rows] = action_group_range(values)
    return result


def _domain_group_balanced_weights(domains: np.ndarray, groups: np.ndarray) -> np.ndarray:
    weights = np.zeros(domains.shape[0], dtype=float)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        domain_rows = np.flatnonzero(domains == domain)
        domain_groups = np.unique(groups[domain_rows])
        for group in domain_groups:
            rows = domain_rows[groups[domain_rows] == group]
            weights[rows] = 1.0 / max(len(unique_domains) * len(domain_groups) * len(rows), 1)
    return weights


def domain_normalized_action_target(
    dataset: MechanismDataset,
    target_name: str = "interval_cost",
    *,
    target_clip: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize paired action deltas by a robust within-domain action scale."""

    target = np.asarray(dataset.targets[target_name], dtype=float)
    domains = np.asarray(dataset.domains, dtype=str)
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
    if target.shape != (dataset.size,) or groups.shape != (dataset.size,):
        raise ValueError("action normalization requires row-aligned targets and groups")
    normalized = np.zeros(dataset.size, dtype=float)
    scales: dict[str, float] = {}
    for domain in np.unique(domains):
        mask = domains == domain
        scale = domain_action_scale(target[mask], groups[mask])
        normalized[mask] = target[mask] / scale
        scales[str(domain)] = float(scale)
    if target_clip is not None:
        normalized = np.clip(normalized, -float(target_clip), float(target_clip))
    return normalized, scales


def group_normalized_action_target(
    dataset: MechanismDataset,
    target_name: str = "interval_cost",
    *,
    target_clip: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize each matched action set by its own evaluation-regret scale."""

    target = np.asarray(dataset.targets[target_name], dtype=float)
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
    if target.shape != (dataset.size,) or groups.shape != (dataset.size,):
        raise ValueError("group action normalization requires row-aligned targets and groups")
    normalized = np.zeros(dataset.size, dtype=float)
    scales = []
    if dataset.size == 0:
        return normalized, {
            "group_count": 0,
            "minimum": 0.0,
            "median": 0.0,
            "maximum": 0.0,
        }
    order = np.argsort(groups, kind="stable")
    ordered_groups = groups[order]
    boundaries = np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1
    for rows in np.split(order, boundaries):
        scale = action_group_range(target[rows])
        normalized[rows] = target[rows] / scale
        scales.append(float(scale))
    if target_clip is not None:
        normalized = np.clip(normalized, -float(target_clip), float(target_clip))
    scale_values = np.asarray(scales, dtype=float)
    return normalized, {
        "group_count": int(scale_values.size),
        "minimum": float(np.min(scale_values)) if scale_values.size else 0.0,
        "median": float(np.median(scale_values)) if scale_values.size else 0.0,
        "maximum": float(np.max(scale_values)) if scale_values.size else 0.0,
    }

def _row_subset(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    mask = np.asarray(mask, dtype=bool)
    subset = dataset.subset(mask)
    metadata = dict(dataset.metadata)
    for key, value in tuple(metadata.items()):
        array = np.asarray(value)
        if array.shape == (dataset.size,):
            metadata[key] = array[mask].tolist()
    return MechanismDataset(
        feature_names=subset.feature_names,
        features=subset.features,
        context_names=subset.context_names,
        context=subset.context,
        priors=subset.priors,
        targets=subset.targets,
        domains=subset.domains,
        metadata=metadata,
    )


def _feature_indices(all_names: Sequence[str], selected_names: Sequence[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(all_names)}
    return np.asarray([index[name] for name in selected_names], dtype=int)


def _context_distance(
    context: np.ndarray,
    source_contexts: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = (np.asarray(context, dtype=float) - mean[None, :]) / scale[None, :]
    sources = (np.asarray(source_contexts, dtype=float) - mean[None, :]) / scale[None, :]
    return np.min(np.sqrt(np.sum((values[:, None, :] - sources[None, :, :]) ** 2, axis=2)), axis=1)


def _context_support(
    context: np.ndarray,
    source_contexts: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(context, dtype=float)
    sources = np.asarray(source_contexts, dtype=float)
    lower = np.min(sources, axis=0)
    upper = np.max(sources, axis=0)
    outside = np.maximum(
        np.maximum((lower[None, :] - values) / scale[None, :], 0.0),
        np.maximum((values - upper[None, :]) / scale[None, :], 0.0),
    )
    return np.exp(-np.max(outside, axis=1))


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = []
    for label in (False, True):
        mask = actual == label
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0
