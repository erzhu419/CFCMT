"""Cross-city stacked mechanism advantages with conservative target adaptation.

The deployment score is always an action contrast against a pressure-rule
reference. Mechanism heads use either the legacy normalized-outcome ablation
or a group-invariant physical simulator-residual protocol. The stacking head
only sees out-of-domain predictions, so a flexible world model is never
selected on the same city rows that trained it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from cf_h2o.traffic_signal.boosted_mechanism_world_model import (
    BoostedFitConfig,
    BoostedMechanismWorldModel,
)

from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    ActionAdvantageConfig,
    PairwiseActionAdvantageRegressor,
    domain_normalized_action_target,
)
from cf_h2o.traffic_signal.mechanism_world_model import (
    InvariantMechanismWorldModel,
    MechanismDataset,
    MechanismDefinition,
    MechanismFitConfig,
)


@dataclass(frozen=True)
class CausalMechanismAdvantageConfig:
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0)
    blend_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    target_blend_weights: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75)
    min_source_domains_for_stack: int = 4
    min_source_robust_gain: float = 0.002
    source_worst_regret_tolerance: float = 0.01
    min_target_groups: int = 4
    target_prior_group_strength: float = 12.0
    max_target_weight: float = 0.75
    uncertainty_quantile: float = 0.90
    target_specialist_blend_weights: tuple[float, ...] = (
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        1.0,
    )
    target_specialist_prior_group_strength: float = 8.0
    target_specialist_max_weight: float = 1.0
    target_specialist_min_mean_gain: float = 0.002
    target_specialist_max_harm_fraction: float = 0.35
    target_specialist_worst_regret_tolerance: float = 0.05
    target_mechanism_gate_weights: tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    )
    target_mechanism_min_mean_gain: float = 0.001
    target_mechanism_max_harm_fraction: float = 0.35
    target_mechanism_worst_regret_tolerance: float = 0.05
    mechanism_dataset_protocol: str = "domain_normalized_outcome_zero_prior_v1"
    mechanism_latent_ranks: tuple[int, ...] = (0,)
    mechanism_adaptation_shrinkages: tuple[float, ...] = (0.0,)
    mechanism_action_ranking_selection: bool = False
    mechanism_estimator_protocol: str = "invariant_ridge_v1"
    source_stack_protocol: str = "mechanism_outcome_replacement_v1"
    source_expert_gate_protocol: str = "none"
    source_expert_gate_ridge_alpha: float = 10.0
    source_expert_gate_error_quantile: float = 0.75
    source_expert_gate_min_disagreement_groups: int = 24


LEGACY_MECHANISM_DATASET_PROTOCOL = "domain_normalized_outcome_zero_prior_v1"
LOCAL_PHYSICAL_RESIDUAL_PROTOCOL = (
    "group_invariant_local_physical_simulator_residual_v1"
)
LOCAL_PHYSICAL_SPEED_SCALE_MPS = 13.89
MECHANISM_OUTCOME_REPLACEMENT_STACK_PROTOCOL = (
    "mechanism_outcome_replacement_v1"
)
RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL = (
    "rigid_anchored_mechanism_residual_v1"
)
SOURCE_STACK_PROTOCOLS = {
    MECHANISM_OUTCOME_REPLACEMENT_STACK_PROTOCOL,
    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL,
}
NO_SOURCE_EXPERT_GATE_PROTOCOL = "none"
SOURCE_OOF_RIDGE_LCB_EXPERT_GATE_PROTOCOL = "source_city_oof_ridge_lcb_v1"
SOURCE_EXPERT_GATE_PROTOCOLS = {
    NO_SOURCE_EXPERT_GATE_PROTOCOL,
    SOURCE_OOF_RIDGE_LCB_EXPERT_GATE_PROTOCOL,
}
SOURCE_EXPERT_GATE_FEATURE_NAMES = (
    "core_penalty_norm",
    "residual_preference_norm",
    "core_margin_norm",
    "residual_margin_norm",
    "core_choice_correction_norm",
    "residual_choice_correction_norm",
    "correction_range_norm",
    "log_action_count",
)
INVARIANT_RIDGE_MECHANISM_ESTIMATOR_PROTOCOL = "invariant_ridge_v1"
CAUSAL_BOOSTED_MECHANISM_ESTIMATOR_PROTOCOL = "causal_boosted_v1"
MECHANISM_ESTIMATOR_PROTOCOLS = {
    INVARIANT_RIDGE_MECHANISM_ESTIMATOR_PROTOCOL,
    CAUSAL_BOOSTED_MECHANISM_ESTIMATOR_PROTOCOL,
}


class CausalMechanismAdvantageModel:
    """Stack a direct causal advantage with cross-fitted mechanism heads.

    Source-city labels fit the invariant mechanism stack.  When ``target_domain``
    is present, target labels fit only a strongly regularized residual head on
    top of frozen source predictions.  The target city never enters source
    stack selection.
    """

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
    ) -> None:
        self.definitions = tuple(
            definition for definition in definitions if definition.target_name != "interval_cost"
        )
        if not self.definitions:
            raise ValueError("causal mechanism advantage requires auxiliary mechanisms")
        self.target_domain = str(target_domain) if target_domain is not None else None
        self.config = config or CausalMechanismAdvantageConfig()
        if self.config.source_stack_protocol not in SOURCE_STACK_PROTOCOLS:
            raise ValueError(
                "unknown causal mechanism source-stack protocol: "
                f"{self.config.source_stack_protocol!r}"
            )
        if self.config.source_expert_gate_protocol not in SOURCE_EXPERT_GATE_PROTOCOLS:
            raise ValueError(
                "unknown causal mechanism source expert-gate protocol: "
                f"{self.config.source_expert_gate_protocol!r}"
            )
        if self.config.mechanism_estimator_protocol not in MECHANISM_ESTIMATOR_PROTOCOLS:
            raise ValueError(
                "unknown causal mechanism estimator protocol: "
                f"{self.config.mechanism_estimator_protocol!r}"
            )
        if not 0.0 < float(self.config.source_expert_gate_error_quantile) < 1.0:
            raise ValueError("source expert-gate error quantile must be in (0, 1)")
        if int(self.config.source_expert_gate_min_disagreement_groups) < 1:
            raise ValueError("source expert gate requires at least one disagreement group")
        self.core_model: PairwiseActionAdvantageRegressor | None = None
        self.mechanism_model: Any | None = None
        self.stack_model: Ridge | None = None
        self.stack_weight = 0.0
        self.stack_calibration_error = 0.0
        self.source_expert_gate_model: Ridge | None = None
        self.source_expert_gate_feature_mean = np.zeros(
            len(SOURCE_EXPERT_GATE_FEATURE_NAMES), dtype=float
        )
        self.source_expert_gate_feature_scale = np.ones(
            len(SOURCE_EXPERT_GATE_FEATURE_NAMES), dtype=float
        )
        self.source_expert_gate_error_margin = 0.0
        self.target_core_model: PairwiseActionAdvantageRegressor | None = None
        self.target_core_weight = 0.0
        self.target_core_calibration_error = 0.0
        self.target_model: Ridge | None = None
        self.target_weight = 0.0
        self.target_calibration_error = 0.0
        self.source_diagnostics: dict[str, Any] = {}
        self.target_diagnostics: dict[str, Any] = {}

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        domains = np.asarray(dataset.domains, dtype=str)
        target_mask = (
            domains == self.target_domain
            if self.target_domain is not None
            else np.zeros(dataset.size, dtype=bool)
        )
        source = _row_subset(dataset, ~target_mask) if np.any(target_mask) else dataset
        if np.unique(source.domains).size < 2:
            raise ValueError("causal mechanism advantage requires at least two source domains")

        self.source_diagnostics = self._fit_source(source)
        if np.any(target_mask):
            target = _row_subset(dataset, target_mask)
            self.target_diagnostics = self._fit_target(target, source)
        else:
            self.target_diagnostics = {
                "enabled": False,
                "reason": "no_target_counterfactual_groups",
                "target_domain": self.target_domain,
                "group_count": 0,
            }
        return {
            "estimator": "cross_fitted_causal_mechanism_advantage",
            "target_domain": self.target_domain,
            "source": self.source_diagnostics,
            "target": self.target_diagnostics,
            "config": asdict(self.config),
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if self.core_model is None:
            raise RuntimeError("causal mechanism advantage model has not been fitted")
        source_score, source_uncertainty, trust, distance, support, mechanism = (
            self._source_prediction(dataset)
        )
        score, stack_component, expert_gate, expert_gate_delta = (
            self._source_stack_details(dataset, source_score, mechanism)
        )
        latent = np.zeros(dataset.size, dtype=float)
        disagreement = np.zeros(dataset.size, dtype=float)
        uncertainty = np.asarray(source_uncertainty, dtype=float).copy()
        effective_stack_weight = self._effective_stack_weight()

        if self.stack_model is not None and effective_stack_weight > 0.0:
            disagreement = self._stack_disagreement(
                source_score,
                stack_component,
            )
            uncertainty += expert_gate * effective_stack_weight * (
                self.stack_calibration_error + disagreement
            )

        if self.target_core_model is not None and self.target_core_weight > 0.0:
            target_core = self.target_core_model.predict(dataset)["control_cost"]
            target_core_delta = np.asarray(target_core["mean"], dtype=float) - source_score
            target_core_adjustment = self.target_core_weight * target_core_delta
            latent += target_core_adjustment
            score += target_core_adjustment
            uncertainty += self.target_core_weight * (
                np.asarray(target_core["uncertainty"], dtype=float)
                + self.target_core_calibration_error
                + np.abs(target_core_delta)
            )
            trust = np.maximum(
                trust,
                self.target_core_weight
                * np.asarray(target_core["context_trust"], dtype=float),
            )

        if self.target_model is not None and self.target_weight > 0.0:
            target_features = np.column_stack([score, mechanism])
            target_residual = np.asarray(self.target_model.predict(target_features), dtype=float)
            target_adjustment = self.target_weight * target_residual
            latent += target_adjustment
            score = score + target_adjustment
            uncertainty += self.target_weight * (
                self.target_calibration_error + np.abs(target_residual)
            )
            trust = (1.0 - self.target_weight) * trust + self.target_weight

        reference = _reference_mask(dataset)
        score[reference] = 0.0
        latent[reference] = 0.0
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": score - latent,
                "latent_residual": latent,
                "mean": score,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": support,
                "mechanism_stack_weight": np.full(
                    dataset.size,
                    effective_stack_weight,
                )
                * expert_gate,
                "target_core_weight": np.full(dataset.size, self.target_core_weight),
                "target_adaptation_weight": np.full(dataset.size, self.target_weight),
                "core_stack_disagreement": disagreement,
                "source_expert_gate": expert_gate,
                "source_expert_gate_predicted_cost_delta": expert_gate_delta,
            }
        }

    def _fit_source(self, dataset: MechanismDataset) -> dict[str, Any]:
        source_domains = tuple(str(value) for value in np.unique(dataset.domains))
        self.source_expert_gate_model = None
        self.source_expert_gate_feature_mean = np.zeros(
            len(SOURCE_EXPERT_GATE_FEATURE_NAMES), dtype=float
        )
        self.source_expert_gate_feature_scale = np.ones(
            len(SOURCE_EXPERT_GATE_FEATURE_NAMES), dtype=float
        )
        self.source_expert_gate_error_margin = 0.0
        self.core_model = _new_core_model()
        core_fit = self.core_model.fit(dataset)
        self.mechanism_model, mechanism_fit = _fit_mechanism_model(
            dataset,
            self.definitions,
            self.config,
        )
        diagnostics: dict[str, Any] = {
            "source_domains": list(source_domains),
            "source_domain_count": len(source_domains),
            "core_fit": core_fit,
            "mechanism_fit": mechanism_fit,
            "mechanism_dataset_protocol": self.config.mechanism_dataset_protocol,
            "mechanism_latent_ranks": list(self.config.mechanism_latent_ranks),
            "mechanism_adaptation_shrinkages": list(
                self.config.mechanism_adaptation_shrinkages
            ),
            "mechanism_action_ranking_selection": bool(
                self.config.mechanism_action_ranking_selection
            ),
            "mechanism_estimator_protocol": (
                self.config.mechanism_estimator_protocol
            ),
            "source_stack_protocol": self.config.source_stack_protocol,
            "source_expert_gate_protocol": self.config.source_expert_gate_protocol,
            "stack_enabled": False,
            "stack_weight": 0.0,
            "source_expert_gate": {
                "enabled": False,
                "reason": "mechanism_stack_not_selected",
            },
            "source_selection_protocol": (
                "nested_outer_city_inner_city_cross_fit_v2"
            ),
            "reason": "insufficient_source_domains_for_nested_cross_fit",
        }
        if len(source_domains) < int(self.config.min_source_domains_for_stack):
            return diagnostics

        core_oof = np.zeros(dataset.size, dtype=float)
        mechanism_oof = np.zeros((dataset.size, len(self.definitions)), dtype=float)
        covered = np.zeros(dataset.size, dtype=bool)
        domain_values = np.asarray(dataset.domains, dtype=str)
        for heldout in source_domains:
            valid_mask = domain_values == heldout
            train = _row_subset(dataset, ~valid_mask)
            valid = _row_subset(dataset, valid_mask)
            if np.unique(train.domains).size < 2:
                continue
            fold_core = _new_core_model()
            fold_core.fit(train)
            fold_mechanism, _ = _fit_mechanism_model(
                train,
                self.definitions,
                self.config,
            )
            core_oof[valid_mask] = fold_core.predict(valid)["control_cost"]["mean"]
            mechanism_oof[valid_mask] = _mechanism_features(
                fold_mechanism,
                valid,
                self.definitions,
                self.config,
            )
            covered[valid_mask] = True
        if not np.all(covered):
            diagnostics["reason"] = "incomplete_source_domain_cross_fit"
            return diagnostics

        pair_excluded_models: dict[tuple[str, str], Any] = {}
        pair_excluded_core_models: dict[
            tuple[str, str], PairwiseActionAdvantageRegressor
        ] = {}
        for excluded in combinations(source_domains, 2):
            train_mask = ~np.isin(domain_values, excluded)
            pair_train = _row_subset(dataset, train_mask)
            if np.unique(pair_train.domains).size < 2:
                diagnostics["reason"] = "insufficient_inner_source_domains"
                return diagnostics
            pair_excluded_models[tuple(sorted(excluded))], _ = _fit_mechanism_model(
                pair_train,
                self.definitions,
                self.config,
            )
            if self._uses_rigid_anchored_residual_stack():
                pair_core = _new_core_model()
                pair_core.fit(pair_train)
                pair_excluded_core_models[tuple(sorted(excluded))] = pair_core

        target = _domain_normalized_target(dataset, "interval_cost")
        groups = _groups(dataset)
        weights = _domain_group_balanced_weights(domain_values, groups)
        reference = _reference_mask(dataset)
        core_oof[reference] = 0.0
        mechanism_oof[reference] = 0.0

        nested_predictions = {
            float(alpha): np.zeros(dataset.size, dtype=float)
            for alpha in self.config.ridge_alphas
        }
        for heldout in source_domains:
            valid_mask = domain_values == heldout
            train_mask = ~valid_mask
            inner_features = np.zeros_like(mechanism_oof)
            inner_core = np.zeros(dataset.size, dtype=float)
            inner_covered = np.zeros(dataset.size, dtype=bool)
            for inner_heldout in source_domains:
                if inner_heldout == heldout:
                    continue
                inner_mask = domain_values == inner_heldout
                excluded = tuple(sorted((heldout, inner_heldout)))
                inner_valid = _row_subset(dataset, inner_mask)
                inner_features[inner_mask] = _mechanism_features(
                    pair_excluded_models[excluded],
                    inner_valid,
                    self.definitions,
                    self.config,
                )
                if self._uses_rigid_anchored_residual_stack():
                    inner_core[inner_mask] = pair_excluded_core_models[
                        excluded
                    ].predict(inner_valid)["control_cost"]["mean"]
                inner_covered[inner_mask] = True
            if not np.all(inner_covered[train_mask]):
                diagnostics["reason"] = "incomplete_inner_source_domain_cross_fit"
                return diagnostics
            inner_weights = _domain_group_balanced_weights(
                domain_values[train_mask],
                groups[train_mask],
            )
            for alpha in self.config.ridge_alphas:
                meta_model = Ridge(alpha=float(alpha), fit_intercept=False)
                meta_target = target[train_mask]
                if self._uses_rigid_anchored_residual_stack():
                    meta_target = meta_target - inner_core[train_mask]
                meta_model.fit(
                    inner_features[train_mask],
                    meta_target,
                    sample_weight=inner_weights,
                )
                nested_predictions[float(alpha)][valid_mask] = meta_model.predict(
                    mechanism_oof[valid_mask]
                )
        for prediction in nested_predictions.values():
            prediction[reference] = 0.0

        core_domains = _domain_action_regret(
            target,
            core_oof,
            groups,
            domain_values,
        )
        core_robust = _robust_regret(core_domains)
        core_worst = max(core_domains.values(), default=0.0)
        grid = []
        for alpha in self.config.ridge_alphas:
            prediction = nested_predictions[float(alpha)]
            for blend in self.config.blend_weights:
                blended = self._combine_source_stack(
                    core_oof,
                    prediction,
                    float(blend),
                )
                domain_regret = _domain_action_regret(
                    target,
                    blended,
                    groups,
                    domain_values,
                )
                improving = sum(
                    domain_regret[name] + 1e-12 < core_domains[name]
                    for name in core_domains
                )
                grid.append(
                    {
                        "alpha": float(alpha),
                        "blend_weight": float(blend),
                        "robust_regret": _robust_regret(domain_regret),
                        "mean_regret": float(np.mean(list(domain_regret.values()))),
                        "worst_regret": max(domain_regret.values(), default=0.0),
                        "improving_domains": int(improving),
                        "domain_regret": domain_regret,
                    }
                )
        candidate = min(
            grid,
            key=lambda row: (
                row["robust_regret"] + 0.0005 * row["blend_weight"],
                row["worst_regret"],
                row["blend_weight"],
                row["alpha"],
            ),
        )
        robust_gain = float(core_robust - candidate["robust_regret"])
        required_improving = max(1, int(np.ceil(len(source_domains) / 3.0)))
        selected = bool(
            candidate["blend_weight"] > 0.0
            and robust_gain >= float(self.config.min_source_robust_gain)
            and candidate["worst_regret"]
            <= core_worst + float(self.config.source_worst_regret_tolerance)
            and candidate["improving_domains"] >= required_improving
        )
        expert_gate_diagnostics: dict[str, Any] = {
            "enabled": False,
            "reason": "mechanism_stack_not_selected",
            "protocol": self.config.source_expert_gate_protocol,
        }
        if selected:
            self.stack_weight = float(candidate["blend_weight"])
            self.stack_model = Ridge(alpha=float(candidate["alpha"]), fit_intercept=False)
            stack_target = target
            if self._uses_rigid_anchored_residual_stack():
                stack_target = target - core_oof
            self.stack_model.fit(mechanism_oof, stack_target, sample_weight=weights)
            stacked_oof = self._combine_source_stack(
                core_oof,
                nested_predictions[float(candidate["alpha"])],
                self.stack_weight,
            )
            expert_gate_diagnostics, calibrated_oof = self._fit_source_expert_gate(
                dataset,
                target=target,
                core_score=core_oof,
                residual_score=stacked_oof,
                domains=domain_values,
                groups=groups,
            )
            self.stack_calibration_error = max(
                float(
                    np.quantile(
                        np.abs(target[~reference] - calibrated_oof[~reference]),
                        self.config.uncertainty_quantile,
                    )
                ),
                1e-6,
            )
            reason = "selected_by_nested_leave_one_city_out_action_regret"
        else:
            self.stack_weight = 0.0
            self.stack_model = None
            self.stack_calibration_error = 0.0
            reason = "mechanism_stack_failed_source_city_gate"
        diagnostics.update(
            {
                "stack_enabled": selected,
                "stack_weight": self.stack_weight,
                "reason": reason,
                "source_expert_gate": expert_gate_diagnostics,
                "core_domain_regret": core_domains,
                "core_robust_regret": core_robust,
                "core_worst_regret": core_worst,
                "selected_candidate": candidate,
                "robust_gain_vs_core": robust_gain,
                "required_improving_domains": required_improving,
                "stack_calibration_error": self.stack_calibration_error,
                "single_excluded_mechanism_fit_count": len(source_domains),
                "pair_excluded_mechanism_fit_count": len(pair_excluded_models),
                "pair_excluded_core_fit_count": len(pair_excluded_core_models),
                "grid": grid,
            }
        )
        return diagnostics

    def _fit_source_expert_gate(
        self,
        dataset: MechanismDataset,
        *,
        target: np.ndarray,
        core_score: np.ndarray,
        residual_score: np.ndarray,
        domains: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
        protocol = self.config.source_expert_gate_protocol
        if protocol == NO_SOURCE_EXPERT_GATE_PROTOCOL:
            return (
                {
                    "enabled": False,
                    "reason": "source_expert_gate_not_requested",
                    "protocol": protocol,
                },
                np.asarray(residual_score, dtype=float).copy(),
            )

        records = _source_expert_gate_records(
            core_score,
            residual_score,
            groups,
            domains=domains,
            target=target,
        )
        record_count = int(records["features"].shape[0])
        minimum = int(self.config.source_expert_gate_min_disagreement_groups)
        record_domains = np.asarray(records["domains"], dtype=str)
        unique_domains = tuple(str(value) for value in np.unique(record_domains))
        if record_count < minimum or len(unique_domains) < 3:
            return (
                {
                    "enabled": False,
                    "reason": "insufficient_cross_city_expert_disagreements",
                    "protocol": protocol,
                    "disagreement_group_count": record_count,
                    "minimum_disagreement_group_count": minimum,
                    "source_domain_count": len(unique_domains),
                },
                np.asarray(core_score, dtype=float).copy(),
            )

        features = np.asarray(records["features"], dtype=float)
        cost_delta = np.asarray(records["cost_delta"], dtype=float)
        weights = _equal_domain_record_weights(record_domains)
        gate_oof = np.zeros(record_count, dtype=float)
        covered = np.zeros(record_count, dtype=bool)
        for heldout in unique_domains:
            valid_mask = record_domains == heldout
            train_mask = ~valid_mask
            mean, scale = _weighted_standardizer(
                features[train_mask],
                weights[train_mask],
            )
            fold_model = Ridge(
                alpha=float(self.config.source_expert_gate_ridge_alpha),
                fit_intercept=True,
            )
            fold_model.fit(
                (features[train_mask] - mean) / scale,
                cost_delta[train_mask],
                sample_weight=weights[train_mask],
            )
            gate_oof[valid_mask] = fold_model.predict(
                (features[valid_mask] - mean) / scale
            )
            covered[valid_mask] = True
        if not np.all(covered):
            raise RuntimeError("source expert gate produced incomplete city OOF predictions")

        one_sided_error = cost_delta - gate_oof
        error_margin = max(
            float(
                np.quantile(
                    one_sided_error,
                    float(self.config.source_expert_gate_error_quantile),
                )
            ),
            0.0,
        )
        gate_choice = gate_oof + error_margin < 0.0
        gated_oof = _apply_source_expert_gate(
            core_score,
            residual_score,
            groups,
            records["group_ids"],
            gate_choice,
        )

        self.source_expert_gate_feature_mean, self.source_expert_gate_feature_scale = (
            _weighted_standardizer(features, weights)
        )
        self.source_expert_gate_model = Ridge(
            alpha=float(self.config.source_expert_gate_ridge_alpha),
            fit_intercept=True,
        )
        self.source_expert_gate_model.fit(
            (features - self.source_expert_gate_feature_mean)
            / self.source_expert_gate_feature_scale,
            cost_delta,
            sample_weight=weights,
        )
        self.source_expert_gate_error_margin = float(error_margin)

        core_domain_regret = _domain_action_regret(
            target, core_score, groups, domains
        )
        residual_domain_regret = _domain_action_regret(
            target, residual_score, groups, domains
        )
        gated_domain_regret = _domain_action_regret(
            target, gated_oof, groups, domains
        )
        oracle_choice = cost_delta < 0.0
        oracle_score = _apply_source_expert_gate(
            core_score,
            residual_score,
            groups,
            records["group_ids"],
            oracle_choice,
        )
        oracle_domain_regret = _domain_action_regret(
            target, oracle_score, groups, domains
        )
        total_groups = int(np.unique(groups).size)
        return (
            {
                "enabled": True,
                "reason": "fit_on_source_city_oof_expert_disagreements",
                "protocol": protocol,
                "feature_names": list(SOURCE_EXPERT_GATE_FEATURE_NAMES),
                "ridge_alpha": float(self.config.source_expert_gate_ridge_alpha),
                "error_quantile": float(
                    self.config.source_expert_gate_error_quantile
                ),
                "one_sided_error_margin": float(error_margin),
                "source_domain_count": len(unique_domains),
                "source_city_oof_fold_count": len(unique_domains),
                "total_group_count": total_groups,
                "disagreement_group_count": record_count,
                "disagreement_fraction": float(record_count / max(total_groups, 1)),
                "selected_residual_group_count": int(np.sum(gate_choice)),
                "selected_residual_fraction_of_disagreements": float(
                    np.mean(gate_choice)
                ),
                "selected_residual_fraction_of_all_groups": float(
                    np.sum(gate_choice) / max(total_groups, 1)
                ),
                "source_city_oof_mae": float(np.mean(np.abs(cost_delta - gate_oof))),
                "source_city_oof_one_sided_coverage": float(
                    np.mean(cost_delta <= gate_oof + error_margin)
                ),
                "core_domain_regret": core_domain_regret,
                "residual_domain_regret": residual_domain_regret,
                "gated_domain_regret": gated_domain_regret,
                "oracle_domain_regret": oracle_domain_regret,
                "target_labels_used": False,
            },
            gated_oof,
        )

    def _fit_target(
        self,
        dataset: MechanismDataset,
        source_dataset: MechanismDataset,
    ) -> dict[str, Any]:
        groups = _groups(dataset)
        unique_groups = np.unique(groups)
        diagnostics: dict[str, Any] = {
            "target_domain": self.target_domain,
            "group_count": int(unique_groups.size),
            "enabled": False,
            "target_core_weight": 0.0,
            "target_weight": 0.0,
            "reason": "insufficient_target_groups",
        }
        if unique_groups.size < int(self.config.min_target_groups):
            return diagnostics

        source_core, _, _, _, _, mechanism = self._source_prediction(dataset)
        source_score = self._stacked_source_score(dataset, source_core, mechanism)
        target = _domain_normalized_target(dataset, "interval_cost")
        weights = _domain_group_balanced_weights(np.asarray(dataset.domains), groups)
        reference = _reference_mask(dataset)

        split_count = min(5, int(unique_groups.size))
        splitter = GroupKFold(n_splits=split_count)
        target_core_oof = np.zeros(dataset.size, dtype=float)
        for train_rows, valid_rows in splitter.split(
            dataset.features,
            target,
            groups=groups,
        ):
            fold_target = _row_subset(
                dataset,
                np.isin(np.arange(dataset.size), train_rows),
            )
            fold_valid = _row_subset(
                dataset,
                np.isin(np.arange(dataset.size), valid_rows),
            )
            fold_core = _new_core_model()
            fold_core.fit(_concat_contrast_datasets(source_dataset, fold_target))
            target_core_oof[valid_rows] = fold_core.predict(fold_valid)["control_cost"]["mean"]
        target_core_oof[reference] = 0.0
        empirical_cap = min(
            float(self.config.max_target_weight),
            unique_groups.size
            / (unique_groups.size + float(self.config.target_prior_group_strength)),
        )
        source_group_regret = _group_action_regrets(target, source_score, groups)
        target_core_grid = []
        for requested_weight in self.config.target_blend_weights:
            blend = min(float(requested_weight), float(empirical_cap))
            prediction = source_score + blend * (target_core_oof - source_core)
            group_regret = _group_action_regrets(target, prediction, groups)
            target_core_grid.append(
                {
                    "requested_weight": float(requested_weight),
                    "blend_weight": float(blend),
                    "mean_regret": float(np.mean(group_regret)),
                    "worst_regret": float(np.max(group_regret)),
                    "harm_fraction_vs_source": float(
                        np.mean(group_regret > source_group_regret + 1e-12)
                    ),
                    "mean_gain_vs_source": float(
                        np.mean(source_group_regret) - np.mean(group_regret)
                    ),
                }
            )
        target_core_baseline = next(
            row for row in target_core_grid if row["blend_weight"] == 0.0
        )
        target_core_candidate = min(
            target_core_grid,
            key=lambda row: (
                row["mean_regret"] + 0.001 * row["blend_weight"],
                row["worst_regret"],
                row["blend_weight"],
            ),
        )
        target_core_selected = bool(
            target_core_candidate["blend_weight"] > 0.0
            and target_core_candidate["mean_gain_vs_source"] >= 0.002
            and target_core_candidate["harm_fraction_vs_source"] <= 0.35
            and target_core_candidate["worst_regret"]
            <= target_core_baseline["worst_regret"] + 0.05
        )
        if target_core_selected:
            self.target_core_weight = float(target_core_candidate["blend_weight"])
            self.target_core_model = _new_core_model()
            self.target_core_model.fit(
                _concat_contrast_datasets(source_dataset, dataset)
            )
            core_error = target - target_core_oof
            self.target_core_calibration_error = max(
                float(
                    np.quantile(
                        np.abs(core_error[~reference]),
                        self.config.uncertainty_quantile,
                    )
                ),
                1e-6,
            )
        else:
            self.target_core_weight = 0.0
            self.target_core_model = None
            self.target_core_calibration_error = 0.0

        adapted_core_oof = source_score + self.target_core_weight * (
            target_core_oof - source_core
        )
        features = np.column_stack([adapted_core_oof, mechanism])
        residual = target - adapted_core_oof
        features[reference] = 0.0
        residual[reference] = 0.0

        grid = []
        oof_by_alpha: dict[float, np.ndarray] = {}
        for alpha in self.config.ridge_alphas:
            oof = np.zeros(dataset.size, dtype=float)
            for train_rows, valid_rows in splitter.split(features, residual, groups=groups):
                model = Ridge(alpha=float(alpha), fit_intercept=False)
                model.fit(
                    features[train_rows],
                    residual[train_rows],
                    sample_weight=weights[train_rows],
                )
                oof[valid_rows] = model.predict(features[valid_rows])
            oof[reference] = 0.0
            oof_by_alpha[float(alpha)] = oof
            for requested_weight in self.config.target_blend_weights:
                blend = min(float(requested_weight), float(empirical_cap))
                prediction = adapted_core_oof + blend * oof
                group_regret = _group_action_regrets(target, prediction, groups)
                source_regret = _group_action_regrets(target, adapted_core_oof, groups)
                harm_fraction = float(
                    np.mean(group_regret > source_regret + 1e-12)
                )
                grid.append(
                    {
                        "alpha": float(alpha),
                        "requested_weight": float(requested_weight),
                        "blend_weight": float(blend),
                        "mean_regret": float(np.mean(group_regret)),
                        "worst_regret": float(np.max(group_regret)),
                        "harm_fraction_vs_source": harm_fraction,
                        "mean_gain_vs_source": float(
                            np.mean(source_regret) - np.mean(group_regret)
                        ),
                    }
                )
        baseline = min(
            (row for row in grid if row["blend_weight"] == 0.0),
            key=lambda row: (row["mean_regret"], row["alpha"]),
        )
        candidate = min(
            grid,
            key=lambda row: (
                row["mean_regret"] + 0.001 * row["blend_weight"],
                row["worst_regret"],
                row["blend_weight"],
                row["alpha"],
            ),
        )
        selected = bool(
            candidate["blend_weight"] > 0.0
            and candidate["mean_gain_vs_source"] >= 0.002
            and candidate["harm_fraction_vs_source"] <= 0.35
            and candidate["worst_regret"] <= baseline["worst_regret"] + 0.05
        )
        if selected:
            self.target_weight = float(candidate["blend_weight"])
            self.target_model = Ridge(alpha=float(candidate["alpha"]), fit_intercept=False)
            self.target_model.fit(features, residual, sample_weight=weights)
            oof_residual = oof_by_alpha[float(candidate["alpha"])]
            self.target_calibration_error = max(
                float(
                    np.quantile(
                        np.abs(residual[~reference] - oof_residual[~reference]),
                        self.config.uncertainty_quantile,
                    )
                ),
                1e-6,
            )
            reason = "selected_by_group_cross_fitted_target_mechanism_advantage"
        else:
            self.target_weight = 0.0
            self.target_model = None
            self.target_calibration_error = 0.0
            reason = (
                "selected_target_core_only"
                if target_core_selected
                else "target_adapters_failed_group_gate"
            )
        diagnostics.update(
            {
                "enabled": bool(target_core_selected or selected),
                "target_core_weight": self.target_core_weight,
                "target_weight": self.target_weight,
                "reason": reason,
                "target_core_adapter": {
                    "enabled": target_core_selected,
                    "selected_candidate": target_core_candidate,
                    "source_baseline": target_core_baseline,
                    "calibration_error": self.target_core_calibration_error,
                    "grid": target_core_grid,
                },
                "selected_candidate": candidate,
                "source_baseline": baseline,
                "target_calibration_error": self.target_calibration_error,
                "grid": grid,
            }
        )
        return diagnostics

    def _stacked_source_score(
        self,
        dataset: MechanismDataset,
        source_score: np.ndarray,
        mechanism: np.ndarray,
    ) -> np.ndarray:
        return self._source_stack_details(dataset, source_score, mechanism)[0]

    def _source_stack_details(
        self,
        dataset: MechanismDataset,
        source_score: np.ndarray,
        mechanism: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        score = np.asarray(source_score, dtype=float).copy()
        stack_component = np.zeros(dataset.size, dtype=float)
        gate_rows = np.zeros(dataset.size, dtype=float)
        predicted_delta_rows = np.zeros(dataset.size, dtype=float)
        effective_stack_weight = self._effective_stack_weight()
        if self.stack_model is not None and effective_stack_weight > 0.0:
            stack_component = np.asarray(
                self.stack_model.predict(mechanism),
                dtype=float,
            )
            residual_score = self._combine_source_stack(
                score,
                stack_component,
                effective_stack_weight,
            )
            if (
                self.config.source_expert_gate_protocol
                == NO_SOURCE_EXPERT_GATE_PROTOCOL
            ):
                score = residual_score
                gate_rows.fill(1.0)
            elif self.source_expert_gate_model is not None:
                groups = _groups(dataset)
                records = _source_expert_gate_records(
                    score,
                    residual_score,
                    groups,
                )
                if records["features"].shape[0]:
                    standardized = (
                        np.asarray(records["features"], dtype=float)
                        - self.source_expert_gate_feature_mean
                    ) / self.source_expert_gate_feature_scale
                    predicted_delta = np.asarray(
                        self.source_expert_gate_model.predict(standardized),
                        dtype=float,
                    )
                    gate_choice = (
                        predicted_delta + self.source_expert_gate_error_margin
                        < 0.0
                    )
                    score = _apply_source_expert_gate(
                        score,
                        residual_score,
                        groups,
                        records["group_ids"],
                        gate_choice,
                    )
                    for group, choice, delta in zip(
                        records["group_ids"],
                        gate_choice,
                        predicted_delta,
                    ):
                        rows = np.flatnonzero(groups == group)
                        gate_rows[rows] = float(choice)
                        predicted_delta_rows[rows] = float(delta)
                else:
                    score = np.asarray(source_score, dtype=float).copy()
            else:
                score = np.asarray(source_score, dtype=float).copy()
        return score, stack_component, gate_rows, predicted_delta_rows

    def _uses_rigid_anchored_residual_stack(self) -> bool:
        return (
            self.config.source_stack_protocol
            == RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
        )

    def _combine_source_stack(
        self,
        source_score: np.ndarray,
        stack_component: np.ndarray,
        weight: float,
    ) -> np.ndarray:
        source = np.asarray(source_score, dtype=float)
        component = np.asarray(stack_component, dtype=float)
        if self._uses_rigid_anchored_residual_stack():
            return source + float(weight) * component
        return source + float(weight) * (component - source)

    def _stack_disagreement(
        self,
        source_score: np.ndarray,
        stack_component: np.ndarray,
    ) -> np.ndarray:
        if self._uses_rigid_anchored_residual_stack():
            return np.abs(np.asarray(stack_component, dtype=float))
        return np.abs(
            np.asarray(stack_component, dtype=float)
            - np.asarray(source_score, dtype=float)
        )

    def _effective_stack_weight(self) -> float:
        return float(self.stack_weight)

    def _source_prediction(
        self,
        dataset: MechanismDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.core_model is None or self.mechanism_model is None:
            raise RuntimeError("source mechanism stack has not been fitted")
        core = self.core_model.predict(dataset)["control_cost"]
        mechanism = _mechanism_features(
            self.mechanism_model,
            dataset,
            self.definitions,
            self.config,
        )
        return (
            np.asarray(core["mean"], dtype=float),
            np.asarray(core["uncertainty"], dtype=float),
            np.asarray(core["context_trust"], dtype=float),
            np.asarray(core["context_distance"], dtype=float),
            np.asarray(core.get("context_support", core["context_trust"]), dtype=float),
            mechanism,
        )


class FusedCausalMechanismAdvantageModel(CausalMechanismAdvantageModel):
    """Fuse a frozen source mechanism score with an OOF target specialist.

    The specialist weight is selected only from target adaptation groups using
    group-level out-of-fold predictions.  Target calibration groups remain
    untouched for the subsequent deployment guard.
    """

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
    ) -> None:
        super().__init__(
            definitions,
            target_domain=target_domain,
            config=config,
        )
        self.fused_target_model: PairwiseActionAdvantageRegressor | None = None
        self.fused_target_weight = 0.0
        self.fused_target_calibration_error = 0.0
        self.target_mechanism_gate = 1.0

    def fit_from_fitted_source(
        self,
        dataset: MechanismDataset,
        source_model: CausalMechanismAdvantageModel,
    ) -> dict[str, Any]:
        """Reuse an exactly matched frozen source stack and fit only target fusion.

        ``CausalMechanismAdvantageModel`` and this class have identical source
        fitting semantics.  The expensive nested source-city cross-fitting can
        therefore be shared, provided both models use the same target exclusion
        and the same source domains.  Target heads and target gates are never
        copied.
        """

        if tuple(self.definitions) != tuple(source_model.definitions):
            raise ValueError("source reuse requires identical mechanism definitions")
        if self.target_domain != source_model.target_domain:
            raise ValueError("source reuse requires identical target-domain exclusion")
        if self.config != source_model.config:
            raise ValueError("source reuse requires identical estimator configuration")
        if source_model.core_model is None or source_model.mechanism_model is None:
            raise ValueError("source reuse requires a fitted mechanism source model")

        domains = np.asarray(dataset.domains, dtype=str)
        target_mask = (
            domains == self.target_domain
            if self.target_domain is not None
            else np.zeros(dataset.size, dtype=bool)
        )
        observed_source_domains = sorted(str(value) for value in np.unique(domains[~target_mask]))
        fitted_source_domains = sorted(
            str(value) for value in source_model.source_diagnostics.get("source_domains", ())
        )
        if observed_source_domains != fitted_source_domains:
            raise ValueError(
                "source reuse domain mismatch: "
                f"observed={observed_source_domains}, fitted={fitted_source_domains}"
            )

        self.core_model = source_model.core_model
        self.mechanism_model = source_model.mechanism_model
        self.stack_model = source_model.stack_model
        self.stack_weight = float(source_model.stack_weight)
        self.stack_calibration_error = float(source_model.stack_calibration_error)
        self.source_expert_gate_model = source_model.source_expert_gate_model
        self.source_expert_gate_feature_mean = np.asarray(
            source_model.source_expert_gate_feature_mean,
            dtype=float,
        ).copy()
        self.source_expert_gate_feature_scale = np.asarray(
            source_model.source_expert_gate_feature_scale,
            dtype=float,
        ).copy()
        self.source_expert_gate_error_margin = float(
            source_model.source_expert_gate_error_margin
        )
        self.source_diagnostics = dict(source_model.source_diagnostics)
        if np.any(target_mask):
            target = _row_subset(dataset, target_mask)
            self.target_diagnostics = self._fit_target(
                target,
                _row_subset(dataset, ~target_mask),
            )
        else:
            self.target_diagnostics = {
                "enabled": False,
                "reason": "no_target_counterfactual_groups",
                "target_domain": self.target_domain,
                "group_count": 0,
            }
        return {
            "estimator": "cross_fitted_causal_mechanism_advantage",
            "target_domain": self.target_domain,
            "source": self.source_diagnostics,
            "target": self.target_diagnostics,
            "config": asdict(self.config),
            "source_reuse": {
                "protocol": "exact_matched_frozen_source_stack_v1",
                "source_domain_count": len(observed_source_domains),
                "target_heads_copied": False,
            },
        }

    def _effective_stack_weight(self) -> float:
        return float(self.stack_weight) * float(self.target_mechanism_gate)

    def _fit_target(
        self,
        dataset: MechanismDataset,
        source_dataset: MechanismDataset,
    ) -> dict[str, Any]:
        del source_dataset
        groups = _groups(dataset)
        unique_groups = np.unique(groups)
        diagnostics: dict[str, Any] = {
            "target_domain": self.target_domain,
            "group_count": int(unique_groups.size),
            "enabled": False,
            "target_specialist_weight": 0.0,
            "selection_protocol": "target-action-group-oof-source-mechanism-specialist-fusion-v1",
            "reason": "insufficient_target_groups",
        }
        if unique_groups.size < int(self.config.min_target_groups):
            return diagnostics

        source_core, _, _, _, _, mechanism = self._source_prediction(dataset)
        source_full_score = self._stacked_source_score(
            dataset, source_core, mechanism
        )
        target = _domain_normalized_target(dataset, "interval_cost")
        weights = _domain_group_balanced_weights(np.asarray(dataset.domains), groups)
        reference = _reference_mask(dataset)
        split_count = min(5, int(unique_groups.size))
        splitter = GroupKFold(n_splits=split_count)
        target_oof = np.zeros(dataset.size, dtype=float)
        row_indices = np.arange(dataset.size)
        for train_rows, valid_rows in splitter.split(
            dataset.features,
            target,
            groups=groups,
        ):
            fold_target = _row_subset(dataset, np.isin(row_indices, train_rows))
            fold_valid = _row_subset(dataset, np.isin(row_indices, valid_rows))
            fold_model = _new_target_specialist_model()
            fold_model.fit(fold_target)
            target_oof[valid_rows] = fold_model.predict(fold_valid)["control_cost"]["mean"]
        target_oof[reference] = 0.0

        empirical_cap = min(
            float(self.config.target_specialist_max_weight),
            unique_groups.size
            / (
                unique_groups.size
                + float(self.config.target_specialist_prior_group_strength)
            ),
        )
        gate_values = (
            tuple(float(value) for value in self.config.target_mechanism_gate_weights)
            if self.stack_model is not None and self.stack_weight > 0.0
            else (0.0,)
        )
        grid: list[dict[str, Any]] = []
        gate_candidates: list[dict[str, Any]] = []
        for gate in gate_values:
            source_score = source_core + gate * (source_full_score - source_core)
            source_group_regret = _group_action_regrets(target, source_score, groups)
            gate_rows = []
            for requested_weight in self.config.target_specialist_blend_weights:
                blend = min(float(requested_weight), float(empirical_cap))
                prediction = source_score + blend * (target_oof - source_score)
                group_regret = _group_action_regrets(target, prediction, groups)
                mean_regret = float(np.mean(group_regret))
                worst_regret = float(np.max(group_regret))
                robust_regret = float(
                    mean_regret
                    + 0.25 * np.std(group_regret)
                    + 0.25 * worst_regret
                )
                row = {
                    "mechanism_gate": float(gate),
                    "effective_stack_weight": float(gate * self.stack_weight),
                    "requested_weight": float(requested_weight),
                    "blend_weight": float(blend),
                    "mean_regret": mean_regret,
                    "worst_regret": worst_regret,
                    "robust_regret": robust_regret,
                    "harm_fraction_vs_source": float(
                        np.mean(group_regret > source_group_regret + 1e-12)
                    ),
                    "mean_gain_vs_source": float(
                        np.mean(source_group_regret) - mean_regret
                    ),
                    "_group_regret": group_regret,
                }
                grid.append(row)
                gate_rows.append(row)
            baseline = min(
                (row for row in gate_rows if row["blend_weight"] == 0.0),
                key=lambda row: (row["robust_regret"], row["requested_weight"]),
            )
            candidate = min(
                gate_rows,
                key=lambda row: (
                    row["robust_regret"] + 0.0005 * row["blend_weight"],
                    row["worst_regret"],
                    row["blend_weight"],
                ),
            )
            specialist_valid = bool(
                candidate["blend_weight"] <= 0.0
                or (
                    candidate["mean_gain_vs_source"]
                    >= float(self.config.target_specialist_min_mean_gain)
                    and candidate["harm_fraction_vs_source"]
                    <= float(self.config.target_specialist_max_harm_fraction)
                    and candidate["worst_regret"]
                    <= baseline["worst_regret"]
                    + float(self.config.target_specialist_worst_regret_tolerance)
                )
            )
            gate_candidates.append(candidate if specialist_valid else baseline)

        exact_rigid_candidate = min(
            (row for row in gate_candidates if row["mechanism_gate"] == 0.0),
            key=lambda row: (
                row["robust_regret"],
                row["worst_regret"],
                row["blend_weight"],
            ),
        )
        candidate = min(
            gate_candidates,
            key=lambda row: (
                row["robust_regret"]
                + 0.0005 * row["blend_weight"]
                + 0.0005 * row["effective_stack_weight"],
                row["worst_regret"],
                row["effective_stack_weight"],
                row["blend_weight"],
            ),
        )
        candidate_regret = np.asarray(candidate["_group_regret"], dtype=float)
        rigid_regret = np.asarray(
            exact_rigid_candidate["_group_regret"],
            dtype=float,
        )
        mechanism_mean_gain = float(
            exact_rigid_candidate["mean_regret"] - candidate["mean_regret"]
        )
        mechanism_harm_fraction = float(
            np.mean(candidate_regret > rigid_regret + 1e-12)
        )
        mechanism_selected = bool(
            candidate["mechanism_gate"] > 0.0
            and mechanism_mean_gain
            >= float(self.config.target_mechanism_min_mean_gain)
            and mechanism_harm_fraction
            <= float(self.config.target_mechanism_max_harm_fraction)
            and candidate["worst_regret"]
            <= exact_rigid_candidate["worst_regret"]
            + float(self.config.target_mechanism_worst_regret_tolerance)
        )
        if candidate["mechanism_gate"] > 0.0 and not mechanism_selected:
            candidate = exact_rigid_candidate
        self.target_mechanism_gate = float(candidate["mechanism_gate"])
        selected = bool(candidate["blend_weight"] > 0.0)
        if selected:
            self.fused_target_weight = float(candidate["blend_weight"])
            self.fused_target_model = _new_target_specialist_model()
            self.fused_target_model.fit(dataset)
            residual = target - target_oof
            self.fused_target_calibration_error = max(
                float(
                    np.quantile(
                        np.abs(residual[~reference]),
                        self.config.uncertainty_quantile,
                    )
                ),
                1e-6,
            )
            reason = "selected_group_oof_target_specialist_fusion"
        else:
            self.fused_target_weight = 0.0
            self.fused_target_model = None
            self.fused_target_calibration_error = 0.0
            reason = "target_specialist_failed_group_oof_gate"

        def public(row: Mapping[str, Any]) -> dict[str, Any]:
            return {
                str(key): value
                for key, value in row.items()
                if not str(key).startswith("_")
            }

        reason = (
            "selected_joint_target_mechanism_specialist_fusion"
            if mechanism_selected and selected
            else "selected_target_mechanism_gate"
            if mechanism_selected
            else "selected_group_oof_target_specialist_fusion"
            if selected
            else "target_mechanism_and_specialist_failed_group_oof_gate"
        )
        diagnostics.update(
            {
                "enabled": bool(mechanism_selected or selected),
                "selection_protocol": (
                    "target-action-group-oof-joint-mechanism-gate-specialist-fusion-v2"
                ),
                "target_mechanism_gate": self.target_mechanism_gate,
                "effective_mechanism_stack_weight": self._effective_stack_weight(),
                "mechanism_gate_selected": mechanism_selected,
                "mechanism_mean_gain_vs_exact_rigid": mechanism_mean_gain,
                "mechanism_harm_fraction_vs_exact_rigid": mechanism_harm_fraction,
                "target_specialist_weight": self.fused_target_weight,
                "empirical_weight_cap": float(empirical_cap),
                "reason": reason,
                "selected_candidate": public(candidate),
                "exact_rigid_candidate": public(exact_rigid_candidate),
                "gate_candidates": [public(row) for row in gate_candidates],
                "target_specialist_calibration_error": self.fused_target_calibration_error,
                "grid": [public(row) for row in grid],
            }
        )
        return diagnostics

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        source = super().predict(dataset)["control_cost"]
        source["target_mechanism_gate"] = np.full(
            dataset.size,
            self.target_mechanism_gate,
            dtype=float,
        )
        if self.fused_target_model is None or self.fused_target_weight <= 0.0:
            return {"control_cost": source}
        target = self.fused_target_model.predict(dataset)["control_cost"]
        weight = float(self.fused_target_weight)
        source_mean = np.asarray(source["mean"], dtype=float)
        target_mean = np.asarray(target["mean"], dtype=float)
        disagreement = np.abs(source_mean - target_mean)
        mean = source_mean + weight * (target_mean - source_mean)
        uncertainty = (
            (1.0 - weight) * np.asarray(source["uncertainty"], dtype=float)
            + weight * np.asarray(target["uncertainty"], dtype=float)
            + weight * (1.0 - weight) * disagreement
            + weight * float(self.fused_target_calibration_error)
        )
        trust = (
            (1.0 - weight) * np.asarray(source["context_trust"], dtype=float)
            + weight * np.asarray(target["context_trust"], dtype=float)
        )
        distance = (
            (1.0 - weight) * np.asarray(source["context_distance"], dtype=float)
            + weight * np.asarray(target["context_distance"], dtype=float)
        )
        latent = weight * (target_mean - source_mean)
        reference = _reference_mask(dataset)
        mean[reference] = 0.0
        uncertainty[reference] = 0.0
        latent[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": mean - latent,
                "latent_residual": latent,
                "mean": mean,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": np.asarray(
                    source.get("context_support", source["context_trust"]),
                    dtype=float,
                ),
                "mechanism_stack_weight": np.asarray(
                    source["mechanism_stack_weight"],
                    dtype=float,
                ),
                "target_mechanism_gate": np.asarray(
                    source["target_mechanism_gate"],
                    dtype=float,
                ),
                "core_stack_disagreement": np.asarray(
                    source.get(
                        "core_stack_disagreement",
                        np.zeros(dataset.size, dtype=float),
                    ),
                    dtype=float,
                ),
                "source_target_disagreement": disagreement,
                "target_specialist_weight": np.full(dataset.size, weight, dtype=float),
                "target_specialist_calibration_error": np.full(
                    dataset.size,
                    self.fused_target_calibration_error,
                    dtype=float,
                ),
            }
        }


class FusedRigidActionAdvantageModel(FusedCausalMechanismAdvantageModel):
    """Exact fused-controller ablation with the source mechanism stack disabled.

    The rigid source core, target specialist, target OOF weight selection,
    uncertainty propagation, and deployment interface are unchanged. This
    isolates the contribution of the source mechanism stack in closed loop.
    """

    def _fit_source(self, dataset: MechanismDataset) -> dict[str, Any]:
        source_domains = tuple(str(value) for value in np.unique(dataset.domains))
        self.core_model = _new_core_model()
        core_fit = self.core_model.fit(dataset)
        self.stack_model = None
        self.stack_weight = 0.0
        self.stack_calibration_error = 0.0
        return {
            "source_domains": list(source_domains),
            "source_domain_count": len(source_domains),
            "core_fit": core_fit,
            "mechanism_fit": None,
            "stack_enabled": False,
            "stack_weight": 0.0,
            "stack_calibration_error": 0.0,
            "reason": "source_mechanism_stack_forced_off_exact_ablation",
            "ablation": "source_mechanism_stack_forced_off",
            "source_selection_protocol": "exact_rigid_core_fast_path_v1",
            "full_stack_fit_performed": False,
            "prediction_equivalence_contract": (
                "full_fused_model_with_source_mechanism_stack_forced_off"
            ),
        }

    def _source_prediction(
        self,
        dataset: MechanismDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.core_model is None:
            raise RuntimeError("source rigid action model has not been fitted")
        core = self.core_model.predict(dataset)["control_cost"]
        return (
            np.asarray(core["mean"], dtype=float),
            np.asarray(core["uncertainty"], dtype=float),
            np.asarray(core["context_trust"], dtype=float),
            np.asarray(core["context_distance"], dtype=float),
            np.asarray(core.get("context_support", core["context_trust"]), dtype=float),
            np.zeros((dataset.size, len(self.definitions)), dtype=float),
        )


class PhysicalResidualCausalMechanismAdvantageModel(CausalMechanismAdvantageModel):
    """Rank-0 causal mechanism stack with a preserved analytic prior."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
        latent: bool = False,
    ) -> None:
        physical_definitions = tuple(
            definition
            for definition in definitions
            if definition.target_name
            not in {"interval_cost", "terminal_system_load"}
        )
        super().__init__(
            physical_definitions,
            target_domain=target_domain,
            config=_physical_residual_config(config, latent=latent),
        )


class RolloutValueResidualCausalMechanismAdvantageModel(
    CausalMechanismAdvantageModel
):
    """Rigid-anchored residual from the fixed-policy terminal system load."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
    ) -> None:
        rollout_definitions = tuple(
            definition
            for definition in definitions
            if definition.target_name == "terminal_system_load"
        )
        if len(rollout_definitions) != 1:
            raise ValueError(
                "rollout-value residual requires exactly one terminal-system-load "
                "mechanism definition"
            )
        super().__init__(
            rollout_definitions,
            target_domain=target_domain,
            config=_physical_residual_config(config, latent=False),
        )


class DirectRolloutValueCausalMechanismAdvantageModel(
    CausalMechanismAdvantageModel
):
    """Rigid-anchored residual from a zero-prior invariant terminal value."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
    ) -> None:
        rollout_definitions = tuple(
            definition
            for definition in definitions
            if definition.target_name == "terminal_system_load"
        )
        if len(rollout_definitions) != 1:
            raise ValueError(
                "direct rollout-value model requires exactly one "
                "terminal-system-load mechanism definition"
            )
        base = config or CausalMechanismAdvantageConfig()
        super().__init__(
            rollout_definitions,
            target_domain=target_domain,
            config=replace(
                base,
                mechanism_dataset_protocol=LEGACY_MECHANISM_DATASET_PROTOCOL,
                mechanism_latent_ranks=(0,),
                mechanism_adaptation_shrinkages=(0.0,),
                mechanism_action_ranking_selection=True,
            ),
        )


class BoostedDirectRolloutValueCausalMechanismAdvantageModel(
    DirectRolloutValueCausalMechanismAdvantageModel
):
    """Parent-restricted nonlinear terminal value with rigid anchoring."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
    ) -> None:
        base = config or CausalMechanismAdvantageConfig()
        super().__init__(
            definitions,
            target_domain=target_domain,
            config=replace(
                base,
                mechanism_estimator_protocol=(
                    CAUSAL_BOOSTED_MECHANISM_ESTIMATOR_PROTOCOL
                ),
            ),
        )


class FusedPhysicalResidualCausalMechanismAdvantageModel(
    FusedCausalMechanismAdvantageModel
):
    """Target-gated physical-residual mechanism stack."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        target_domain: str | None = None,
        config: CausalMechanismAdvantageConfig | None = None,
        latent: bool = False,
    ) -> None:
        physical_definitions = tuple(
            definition
            for definition in definitions
            if definition.target_name
            not in {"interval_cost", "terminal_system_load"}
        )
        super().__init__(
            physical_definitions,
            target_domain=target_domain,
            config=_physical_residual_config(config, latent=latent),
        )


def _physical_residual_config(
    config: CausalMechanismAdvantageConfig | None,
    *,
    latent: bool,
) -> CausalMechanismAdvantageConfig:
    base = config or CausalMechanismAdvantageConfig()
    return replace(
        base,
        mechanism_dataset_protocol=LOCAL_PHYSICAL_RESIDUAL_PROTOCOL,
        mechanism_latent_ranks=(0, 1, 2) if latent else (0,),
        mechanism_adaptation_shrinkages=(0.0, 0.5, 1.0) if latent else (0.0,),
        mechanism_action_ranking_selection=True,
    )


def _new_core_model() -> PairwiseActionAdvantageRegressor:
    return PairwiseActionAdvantageRegressor(
        causal=True,
        config=ActionAdvantageConfig(
            candidate_only=True,
            balance_candidate_signs=True,
        ),
        causal_feature_names=CAUSAL_RIGID_RANKING_PARENTS,
    )


def _new_target_specialist_model() -> PairwiseActionAdvantageRegressor:
    return PairwiseActionAdvantageRegressor(
        causal=True,
        config=ActionAdvantageConfig(
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=4,
            l2_regularization=16.0,
        ),
    )


def _fit_mechanism_model(
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
    config: CausalMechanismAdvantageConfig,
) -> tuple[Any, dict[str, Any]]:
    normalized = _mechanism_dataset(dataset, definitions, config)
    action_ranking_targets = (
        tuple(definition.target_name for definition in definitions)
        if config.mechanism_action_ranking_selection
        else ()
    )
    if (
        config.mechanism_estimator_protocol
        == CAUSAL_BOOSTED_MECHANISM_ESTIMATOR_PROTOCOL
    ):
        model = BoostedMechanismWorldModel(
            definitions,
            causal=True,
            config=replace(
                BoostedFitConfig(),
                action_ranking_targets=action_ranking_targets,
                select_parent_variants=True,
            ),
        )
        return model, model.fit(normalized)
    model = InvariantMechanismWorldModel(
        definitions,
        MechanismFitConfig(
            ridge_l2=6.0,
            domain_offset_l2=24.0,
            context_l2=16.0,
            latent_ranks=tuple(int(value) for value in config.mechanism_latent_ranks),
            adaptation_shrinkages=tuple(
                float(value) for value in config.mechanism_adaptation_shrinkages
            ),
            min_domain_rows=6,
            complexity_penalty=0.003,
            action_ranking_targets=action_ranking_targets,
        ),
    )
    return model, model.fit(normalized)


def _mechanism_features(
    model: Any,
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
    config: CausalMechanismAdvantageConfig,
) -> np.ndarray:
    prediction = model.predict(
        _mechanism_prediction_dataset(dataset, definitions, config)
    )
    values = np.column_stack(
        [np.asarray(prediction[definition.name]["mean"], dtype=float) for definition in definitions]
    )
    values[_reference_mask(dataset)] = 0.0
    return values


def _mechanism_dataset(
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
    config: CausalMechanismAdvantageConfig,
) -> MechanismDataset:
    protocol = str(config.mechanism_dataset_protocol)
    if protocol == LEGACY_MECHANISM_DATASET_PROTOCOL:
        return _normalized_mechanism_dataset(dataset, definitions)
    if protocol == LOCAL_PHYSICAL_RESIDUAL_PROTOCOL:
        return _local_physical_mechanism_dataset(dataset, definitions)
    raise ValueError(f"unknown mechanism dataset protocol: {protocol!r}")


def _mechanism_prediction_dataset(
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
    config: CausalMechanismAdvantageConfig,
) -> MechanismDataset:
    protocol = str(config.mechanism_dataset_protocol)
    if protocol == LEGACY_MECHANISM_DATASET_PROTOCOL:
        return _zero_prior_dataset(dataset, definitions)
    if protocol == LOCAL_PHYSICAL_RESIDUAL_PROTOCOL:
        return _local_physical_mechanism_dataset(dataset, definitions)
    raise ValueError(f"unknown mechanism dataset protocol: {protocol!r}")


def _normalized_mechanism_dataset(
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
) -> MechanismDataset:
    targets = dict(dataset.targets)
    priors = dict(dataset.priors)
    for definition in definitions:
        targets[definition.target_name] = _domain_normalized_target(
            dataset,
            definition.target_name,
        )
        priors[definition.target_name] = np.zeros(dataset.size, dtype=float)
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=priors,
        targets=targets,
        domains=dataset.domains,
        metadata=dict(dataset.metadata),
    )


def _zero_prior_dataset(
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
) -> MechanismDataset:
    priors = dict(dataset.priors)
    for definition in definitions:
        priors[definition.target_name] = np.zeros(dataset.size, dtype=float)
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata=dict(dataset.metadata),
    )


def _local_physical_mechanism_dataset(
    dataset: MechanismDataset,
    definitions: Sequence[MechanismDefinition],
) -> MechanismDataset:
    """Put local states, simulator priors, and labels in portable units.

    Every output scale is fixed by same-state, target-label-free quantities.
    The analytic prior and observed outcome are divided by the same scale, so
    the world model fits a genuine simulator residual instead of replacing the
    prior with a direct outcome proxy.
    """

    local_lane_scale, graph_lane_scale = _physical_lane_scales(dataset)
    features = np.asarray(dataset.features, dtype=float).copy()
    for index, name in enumerate(dataset.feature_names):
        scale = _physical_feature_scale(
            str(name),
            local_lane_scale=local_lane_scale,
            graph_lane_scale=graph_lane_scale,
        )
        features[:, index] /= scale

    targets = dict(dataset.targets)
    priors = dict(dataset.priors)
    scale_names: dict[str, str] = {}
    for definition in definitions:
        name = definition.target_name
        scale, scale_name = _physical_output_scale(
            name,
            local_lane_scale=local_lane_scale,
        )
        _require_group_invariant_scale(dataset, scale, name=name)
        targets[name] = np.asarray(dataset.targets[name], dtype=float) / scale
        priors[name] = np.asarray(dataset.priors[name], dtype=float) / scale
        scale_names[name] = scale_name

    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=priors,
        targets=targets,
        domains=dataset.domains,
        metadata={
            **dict(dataset.metadata),
            "mechanism_dataset_protocol": LOCAL_PHYSICAL_RESIDUAL_PROTOCOL,
            "mechanism_output_scale_names": scale_names,
            "mechanism_speed_scale_mps": LOCAL_PHYSICAL_SPEED_SCALE_MPS,
        },
    )


def _physical_lane_scales(dataset: MechanismDataset) -> tuple[np.ndarray, np.ndarray]:
    feature_index = {name: index for index, name in enumerate(dataset.feature_names)}
    context_index = {name: index for index, name in enumerate(dataset.context_names)}
    if "lane_count_norm" in feature_index:
        local = np.asarray(
            dataset.features[:, feature_index["lane_count_norm"]],
            dtype=float,
        ) * 12.0
    else:
        local = np.ones(dataset.size, dtype=float)
    local = np.maximum(local, 1.0)

    if {"network_lane_count_norm", "tls_count_norm"} <= set(context_index):
        network_lanes = np.asarray(
            dataset.context[:, context_index["network_lane_count_norm"]],
            dtype=float,
        ) * 160.0
        tls_count = np.asarray(
            dataset.context[:, context_index["tls_count_norm"]],
            dtype=float,
        ) * 20.0
        graph = network_lanes / np.maximum(tls_count, 1.0)
    else:
        graph = local.copy()
    graph = np.maximum(graph, 1.0)
    _require_group_invariant_scale(dataset, local, name="local_lane_count")
    _require_group_invariant_scale(dataset, graph, name="network_lanes_per_tls")
    return local, graph


def _physical_feature_scale(
    name: str,
    *,
    local_lane_scale: np.ndarray,
    graph_lane_scale: np.ndarray,
) -> np.ndarray:
    base = str(name).lower()
    for prefix in ("reference_", "delta_"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    if "occ" in base:
        # SUMO lane occupancy is collected as a [0, 1] fraction throughout the
        # counterfactual cache.  Keeping that unit avoids shrinking spillback
        # residuals by another factor of 100 before ridge regularization.
        return np.ones(local_lane_scale.shape, dtype=float)
    if "speed" in base:
        return np.full(
            local_lane_scale.shape,
            LOCAL_PHYSICAL_SPEED_SCALE_MPS,
            dtype=float,
        )
    dimensionless_suffixes = ("_ratio", "_percentile", "_norm", "_cv")
    if base.endswith(dimensionless_suffixes):
        return np.ones(local_lane_scale.shape, dtype=float)
    count_like = bool(
        base in {"total_q", "total_veh", "green_q", "red_q", "green_veh", "red_veh"}
        or "_q" in base
        or base.endswith("_veh")
        or "pressure" in base
        or base == "network_active_per_tls"
    )
    if not count_like:
        return np.ones(local_lane_scale.shape, dtype=float)
    graph_like = bool(
        base.startswith("graph_")
        or base.startswith("green_signal_down_")
        or base.startswith("network_active_")
        or "corridor_pressure" in base
    )
    return graph_lane_scale if graph_like else local_lane_scale


def _physical_output_scale(
    target_name: str,
    *,
    local_lane_scale: np.ndarray,
) -> tuple[np.ndarray, str]:
    semantic_name = {
        "one_step_total_queue": "next_total_queue",
        "one_step_green_queue": "next_green_queue",
        "one_step_red_queue": "next_red_queue",
        "one_step_downstream_occupancy": "next_downstream_occupancy",
        "one_step_mean_speed": "next_mean_speed",
    }.get(target_name, target_name)
    if semantic_name in {
        "next_total_queue",
        "next_green_queue",
        "next_red_queue",
    }:
        return local_lane_scale, "vehicles_per_local_controlled_lane"
    if semantic_name == "next_downstream_occupancy":
        return (
            np.ones(local_lane_scale.shape, dtype=float),
            "fraction_of_lane_occupancy",
        )
    if semantic_name == "next_mean_speed":
        return (
            np.full(
                local_lane_scale.shape,
                LOCAL_PHYSICAL_SPEED_SCALE_MPS,
                dtype=float,
            ),
            "fraction_of_13p89_mps",
        )
    if semantic_name in {"terminal_system_load", "interval_cost"}:
        return np.ones(local_lane_scale.shape, dtype=float), "vehicles_per_network_lane"
    raise KeyError(f"no physical mechanism scale for target {target_name!r}")


def _require_group_invariant_scale(
    dataset: MechanismDataset,
    scale: np.ndarray,
    *,
    name: str,
) -> None:
    groups = _groups(dataset)
    values = np.asarray(scale, dtype=float)
    if values.shape != (dataset.size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: invalid row-aligned physical scale")
    if np.any(values <= 0.0):
        raise ValueError(f"{name}: physical scale must be positive")
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if not np.allclose(values[rows], values[rows[0]], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{name}: physical scale depends on candidate action in group {group!r}"
            )


def _domain_normalized_target(dataset: MechanismDataset, target_name: str) -> np.ndarray:
    return domain_normalized_action_target(
        dataset,
        target_name,
        target_clip=8.0,
    )[0]


def _source_expert_gate_records(
    core_score: np.ndarray,
    residual_score: np.ndarray,
    groups: np.ndarray,
    *,
    domains: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    core = np.asarray(core_score, dtype=float)
    residual = np.asarray(residual_score, dtype=float)
    groups = np.asarray(groups)
    if core.shape != residual.shape or core.shape != groups.shape:
        raise ValueError("source expert gate requires aligned score and group rows")
    if domains is not None and np.asarray(domains).shape != core.shape:
        raise ValueError("source expert gate requires aligned domain rows")
    if target is not None and np.asarray(target).shape != core.shape:
        raise ValueError("source expert gate requires aligned target rows")

    features: list[list[float]] = []
    group_ids: list[Any] = []
    record_domains: list[str] = []
    cost_delta: list[float] = []
    core_rows: list[int] = []
    residual_rows: list[int] = []
    correction = residual - core
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if rows.size < 2:
            continue
        core_order = rows[np.argsort(core[rows], kind="stable")]
        residual_order = rows[np.argsort(residual[rows], kind="stable")]
        core_choice = int(core_order[0])
        residual_choice = int(residual_order[0])
        if core_choice == residual_choice:
            continue
        scale = max(
            float(np.ptp(core[rows])),
            float(np.ptp(residual[rows])),
            0.05,
        )
        core_margin = float(core[core_order[1]] - core[core_order[0]])
        residual_margin = float(
            residual[residual_order[1]] - residual[residual_order[0]]
        )
        features.append(
            [
                float(core[residual_choice] - core[core_choice]) / scale,
                float(residual[core_choice] - residual[residual_choice]) / scale,
                core_margin / scale,
                residual_margin / scale,
                float(correction[core_choice]) / scale,
                float(correction[residual_choice]) / scale,
                float(np.ptp(correction[rows])) / scale,
                float(np.log(float(rows.size))),
            ]
        )
        group_ids.append(group)
        core_rows.append(core_choice)
        residual_rows.append(residual_choice)
        if domains is not None:
            group_domains = np.unique(np.asarray(domains, dtype=str)[rows])
            if group_domains.size != 1:
                raise ValueError("source expert-gate action group crosses domains")
            record_domains.append(str(group_domains[0]))
        if target is not None:
            values = np.asarray(target, dtype=float)
            cost_delta.append(
                float(values[residual_choice] - values[core_choice])
            )

    feature_array = np.asarray(features, dtype=float)
    if not features:
        feature_array = np.zeros(
            (0, len(SOURCE_EXPERT_GATE_FEATURE_NAMES)), dtype=float
        )
    return {
        "features": feature_array,
        "group_ids": np.asarray(group_ids, dtype=groups.dtype),
        "domains": np.asarray(record_domains, dtype=str),
        "cost_delta": np.asarray(cost_delta, dtype=float),
        "core_rows": np.asarray(core_rows, dtype=int),
        "residual_rows": np.asarray(residual_rows, dtype=int),
    }


def _apply_source_expert_gate(
    core_score: np.ndarray,
    residual_score: np.ndarray,
    groups: np.ndarray,
    disagreement_groups: np.ndarray,
    choose_residual: np.ndarray,
) -> np.ndarray:
    result = np.asarray(core_score, dtype=float).copy()
    residual = np.asarray(residual_score, dtype=float)
    groups = np.asarray(groups)
    disagreement_groups = np.asarray(disagreement_groups)
    choose_residual = np.asarray(choose_residual, dtype=bool)
    if disagreement_groups.shape != choose_residual.shape:
        raise ValueError("source expert gate choices must align with disagreement groups")
    for group, selected in zip(disagreement_groups, choose_residual):
        if selected:
            result[groups == group] = residual[groups == group]
    return result


def _equal_domain_record_weights(domains: np.ndarray) -> np.ndarray:
    domains = np.asarray(domains, dtype=str)
    weights = np.zeros(domains.shape[0], dtype=float)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        rows = np.flatnonzero(domains == domain)
        weights[rows] = 1.0 / max(len(unique_domains) * rows.size, 1)
    return weights * (weights.size / max(float(np.sum(weights)), 1e-12))


def _weighted_standardizer(
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if features.ndim != 2 or weights.shape != (features.shape[0],):
        raise ValueError("source expert-gate standardizer requires aligned rows")
    normalized_weights = weights / max(float(np.sum(weights)), 1e-12)
    mean = np.sum(features * normalized_weights[:, None], axis=0)
    variance = np.sum(
        np.square(features - mean) * normalized_weights[:, None],
        axis=0,
    )
    scale = np.sqrt(np.maximum(variance, 1e-8))
    return mean, scale


def _domain_action_regret(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    domains: np.ndarray,
) -> dict[str, float]:
    result = {}
    for domain in np.unique(domains):
        mask = domains == domain
        result[str(domain)] = float(
            np.mean(_group_action_regrets(target[mask], prediction[mask], groups[mask]))
        )
    return result


def _group_action_regrets(
    target: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    regrets = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        selected = int(rows[int(np.argmin(prediction[rows]))])
        regrets.append(float(target[selected]) - float(np.min(target[rows])))
    return np.asarray(regrets, dtype=float)


def _robust_regret(values: Mapping[str, float]) -> float:
    rows = np.asarray(list(values.values()), dtype=float)
    if not rows.size:
        return float("inf")
    return float(np.mean(rows) + 0.25 * np.std(rows) + 0.25 * np.max(rows))


def _groups(dataset: MechanismDataset) -> np.ndarray:
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
    if groups.shape != (dataset.size,):
        raise ValueError("mechanism advantage requires row-aligned action groups")
    return groups


def _reference_mask(dataset: MechanismDataset) -> np.ndarray:
    values = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if values.shape != (dataset.size,):
        raise ValueError("mechanism advantage requires row-aligned reference flags")
    return values


def _domain_group_balanced_weights(domains: np.ndarray, groups: np.ndarray) -> np.ndarray:
    domains = np.asarray(domains)
    groups = np.asarray(groups)
    weights = np.zeros(domains.shape[0], dtype=float)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        domain_rows = np.flatnonzero(domains == domain)
        domain_groups = np.unique(groups[domain_rows])
        for group in domain_groups:
            rows = domain_rows[groups[domain_rows] == group]
            weights[rows] = 1.0 / max(len(unique_domains) * len(domain_groups) * len(rows), 1)
    return weights * (weights.size / max(float(np.sum(weights)), 1e-12))


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


def _concat_contrast_datasets(
    first: MechanismDataset,
    second: MechanismDataset,
) -> MechanismDataset:
    if first.feature_names != second.feature_names or first.context_names != second.context_names:
        raise ValueError("cannot combine incompatible contrast datasets")
    if set(first.targets) != set(second.targets) or set(first.priors) != set(second.priors):
        raise ValueError("cannot combine contrast datasets with different outputs")
    return MechanismDataset(
        feature_names=first.feature_names,
        features=np.vstack([first.features, second.features]),
        context_names=first.context_names,
        context=np.vstack([first.context, second.context]),
        priors={
            name: np.concatenate([first.priors[name], second.priors[name]])
            for name in first.priors
        },
        targets={
            name: np.concatenate([first.targets[name], second.targets[name]])
            for name in first.targets
        },
        domains=np.concatenate([first.domains, second.domains]),
        metadata={
            "action_group_ids": [
                *np.asarray(first.metadata["action_group_ids"]).tolist(),
                *np.asarray(second.metadata["action_group_ids"]).tolist(),
            ],
            "is_reference": [
                *np.asarray(first.metadata["is_reference"], dtype=bool).tolist(),
                *np.asarray(second.metadata["is_reference"], dtype=bool).tolist(),
            ],
        },
    )
