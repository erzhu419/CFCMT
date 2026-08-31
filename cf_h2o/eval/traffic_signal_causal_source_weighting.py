"""Causal source shrinkage and source-benefit selection."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    _group_adjusted_scores,
    _relative_contrast_rule_gap,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range


SOURCE_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTION_PROTOCOL = "source-city-loo-closed-loop-robust-weight-selector-v1"
SOURCE_CITY_SELECTION_PROTOCOL = (
    "source-city-component-closed-loop-robust-selector-v1"
)
OFFLINE_SOURCE_EVALUATION_PROTOCOL = (
    "target-offline-source-weight-action-regret-evaluation-v1"
)
OFFLINE_SOURCE_SELECTION_PROTOCOL = (
    "target-offline-seed-blocked-causal-source-selector-v1"
)
FAMILYWISE_OFFLINE_SOURCE_SELECTION_PROTOCOL = (
    "target-offline-familywise-negative-transfer-selector-v1"
)
OFFLINE_GUARD_EVALUATION_PROTOCOL = (
    "target-offline-cross-fitted-deployment-guard-evaluation-v1"
)
OFFLINE_GUARD_SELECTION_PROTOCOL = (
    "target-offline-familywise-deployment-guard-selector-v1"
)


@dataclass
class SourceWeightAudit:
    prediction_calls: int = 0
    prediction_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prediction_calls": int(self.prediction_calls),
            "prediction_rows": int(self.prediction_rows),
        }


class FrozenCausalSourceWeightModel:
    """Blend a source-plus-target causal score toward its target-only twin."""

    def __init__(
        self,
        *,
        source_model: Any,
        target_only_model: Any,
        source_objective_mode: str,
        target_only_objective_mode: str,
        source_weight: float,
    ) -> None:
        weight = float(source_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("source_weight must be in [0, 1]")
        self.source_model = source_model
        self.target_only_model = target_only_model
        self.source_objective_mode = str(source_objective_mode)
        self.target_only_objective_mode = str(target_only_objective_mode)
        self.source_weight = weight
        self.audit = SourceWeightAudit()

    def predict(self, dataset):
        self.audit.prediction_calls += 1
        self.audit.prediction_rows += int(dataset.size)
        source_score, source_uncertainty, source_trust, _ = (
            _group_adjusted_scores(
                dataset,
                self.source_model.predict(dataset),
                objective_mode=self.source_objective_mode,
            )
        )
        target_score, target_uncertainty, target_trust, _ = (
            _group_adjusted_scores(
                dataset,
                self.target_only_model.predict(dataset),
                objective_mode=self.target_only_objective_mode,
            )
        )
        weight = float(self.source_weight)
        return {
            "control_cost": {
                "mean": target_score + weight * (source_score - target_score),
                "uncertainty": (
                    (1.0 - weight) * target_uncertainty
                    + weight * source_uncertainty
                ),
                "context_trust": np.minimum(source_trust, target_trust),
            }
        }


class FrozenCausalSourceMixtureModel:
    """Convexly mix source-city components around one target-only anchor."""

    def __init__(
        self,
        *,
        source_models: Mapping[str, Any],
        target_only_model: Any,
        source_objective_modes: Mapping[str, str],
        target_only_objective_mode: str,
        source_city_weights: Mapping[str, float],
        uncertainty_scale: float = 1.0,
    ) -> None:
        names = set(str(value) for value in source_models)
        if set(str(value) for value in source_objective_modes) != names:
            raise ValueError("source model/objective-mode keys differ")
        weights = {
            str(name): float(value)
            for name, value in source_city_weights.items()
        }
        if set(weights) - names:
            raise ValueError("source-city weights reference unknown models")
        if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("source-city weights must be finite and nonnegative")
        mass = float(sum(weights.values()))
        if mass > 1.0 + 1e-12:
            raise ValueError("source-city weights must sum to at most one")
        scale = float(uncertainty_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("uncertainty_scale must be finite and positive")
        self.source_models = {
            name: source_models[name] for name in sorted(names)
        }
        self.target_only_model = target_only_model
        self.source_objective_modes = {
            name: str(source_objective_modes[name]) for name in sorted(names)
        }
        self.target_only_objective_mode = str(target_only_objective_mode)
        self.source_city_weights = {
            name: weights[name] for name in sorted(weights) if weights[name] > 0.0
        }
        self.source_mass = mass
        self.uncertainty_scale = scale
        self.audit = SourceWeightAudit()

    def predict(self, dataset):
        self.audit.prediction_calls += 1
        self.audit.prediction_rows += int(dataset.size)
        target_score, target_uncertainty, target_trust, _ = (
            _group_adjusted_scores(
                dataset,
                self.target_only_model.predict(dataset),
                objective_mode=self.target_only_objective_mode,
            )
        )
        score = (1.0 - self.source_mass) * target_score
        uncertainty = (1.0 - self.source_mass) * target_uncertainty
        trust = np.asarray(target_trust, dtype=float).copy()
        for name, weight in self.source_city_weights.items():
            source_score, source_uncertainty, source_trust, _ = (
                _group_adjusted_scores(
                    dataset,
                    self.source_models[name].predict(dataset),
                    objective_mode=self.source_objective_modes[name],
                )
            )
            score = score + weight * source_score
            uncertainty = uncertainty + weight * source_uncertainty
            trust = np.minimum(trust, source_trust)
        return {
            "control_cost": {
                "mean": score,
                "uncertainty": self.uncertainty_scale * uncertainty,
                "context_trust": trust,
            }
        }


def offline_guard_profile_key(
    risk_multiplier: float,
    max_relative_rule_gap: float,
) -> str:
    def token(value: float) -> str:
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    return (
        f"risk_{token(risk_multiplier)}"
        f"__gap_{token(max_relative_rule_gap)}"
    )


def evaluate_target_offline_guard_grid(
    dataset,
    *,
    reference_policy: Any,
    source_model: Any,
    target_only_model: Any,
    source_objective_mode: str,
    target_only_objective_mode: str,
    source_weight: float,
    scenarios: Sequence[str],
    risk_multipliers: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    max_relative_rule_gaps: Sequence[float] = (0.0, 0.10, 0.25, 0.50, 1.0),
    min_context_trust: float = 0.1,
    margin: float = 0.0,
    contrast_features: Sequence[str] = CONTRAST_FEATURES_V3,
) -> dict[str, Any]:
    """Score deployment guards on held-out target offline action groups."""

    weight = float(source_weight)
    risks = tuple(float(value) for value in risk_multipliers)
    gaps = tuple(float(value) for value in max_relative_rule_gaps)
    scenario_names = tuple(str(value) for value in scenarios)
    if (
        not 0.0 <= weight <= 1.0
        or not risks
        or not gaps
        or len(risks) != len(set(risks))
        or len(gaps) != len(set(gaps))
        or any(value < 0.0 for value in risks)
        or any(value < 0.0 for value in gaps)
        or not 0.0 <= float(min_context_trust) <= 1.0
        or float(margin) < 0.0
        or not scenario_names
    ):
        raise ValueError("invalid target-offline guard grid")
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=reference_policy,
        contrast_features=tuple(str(value) for value in contrast_features),
    )
    actual = np.asarray(contrast.targets["interval_cost"], dtype=float)
    groups = np.asarray(action_group_ids(contrast), dtype=str)
    is_reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    source_score, source_uncertainty, source_trust, _ = _group_adjusted_scores(
        contrast,
        source_model.predict(contrast),
        objective_mode=str(source_objective_mode),
    )
    target_score, target_uncertainty, target_trust, _ = _group_adjusted_scores(
        contrast,
        target_only_model.predict(contrast),
        objective_mode=str(target_only_objective_mode),
    )
    score = target_score + weight * (source_score - target_score)
    uncertainty = (
        (1.0 - weight) * target_uncertainty + weight * source_uncertainty
    )
    trust = (
        target_trust
        if weight == 0.0
        else source_trust
        if weight == 1.0
        else np.minimum(source_trust, target_trust)
    )
    rule_gap = _relative_contrast_rule_gap(contrast)
    arrays = (actual, score, uncertainty, trust, rule_gap)
    if any(
        np.asarray(values).shape != (contrast.size,)
        or not np.all(np.isfinite(values))
        for values in arrays
    ):
        raise ValueError("target-offline guard arrays must be finite and aligned")

    group_rows = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        reference = rows[is_reference[rows]]
        matches = [
            scenario
            for scenario in scenario_names
            if str(group).startswith(f"{scenario}:")
        ]
        if reference.size != 1 or len(matches) != 1:
            raise ValueError(f"invalid target-offline guard group: {group}")
        group_rows.append((str(group), matches[0], rows, int(reference[0])))
    if {row[1] for row in group_rows} != set(scenario_names):
        raise ValueError("target-offline guard fold misses a scenario")

    grid = {}
    for risk in risks:
        for gap_limit in gaps:
            scenario_deltas = {scenario: [] for scenario in scenario_names}
            scenario_regrets = {scenario: [] for scenario in scenario_names}
            accepted = 0
            learned_differences = 0
            for _, scenario, rows, reference in group_rows:
                learned = int(rows[int(np.argmin(score[rows]))])
                learned_differences += int(learned != reference)
                use_learned = bool(
                    learned != reference
                    and float(trust[learned]) >= float(min_context_trust)
                    and float(rule_gap[learned]) <= gap_limit + 1e-12
                    and float(score[learned])
                    + risk * float(uncertainty[learned])
                    + float(margin)
                    < 0.0
                )
                selected = learned if use_learned else reference
                accepted += int(use_learned)
                values = actual[rows]
                scale = action_group_range(values)
                scenario_deltas[scenario].append(
                    (float(actual[selected]) - float(actual[reference])) / scale
                )
                scenario_regrets[scenario].append(
                    (float(actual[selected]) - float(np.min(values))) / scale
                )
            mean_delta_by_scenario = {
                scenario: float(np.mean(values))
                for scenario, values in scenario_deltas.items()
            }
            mean_regret_by_scenario = {
                scenario: float(np.mean(values))
                for scenario, values in scenario_regrets.items()
            }
            key = offline_guard_profile_key(risk, gap_limit)
            grid[key] = {
                "profile_key": key,
                "risk_multiplier": risk,
                "max_relative_rule_gap": gap_limit,
                "min_context_trust": float(min_context_trust),
                "margin": float(margin),
                "group_count": len(group_rows),
                "learned_difference_count": learned_differences,
                "accepted_override_count": accepted,
                "scenario_mean_normalized_delta_vs_rule": mean_delta_by_scenario,
                "scenario_mean_normalized_action_regret": mean_regret_by_scenario,
                "equal_scenario_mean_normalized_delta_vs_rule": float(
                    np.mean(
                        [mean_delta_by_scenario[name] for name in scenario_names]
                    )
                ),
                "equal_scenario_mean_normalized_action_regret": float(
                    np.mean(
                        [mean_regret_by_scenario[name] for name in scenario_names]
                    )
                ),
            }
    return {
        "protocol": OFFLINE_GUARD_EVALUATION_PROTOCOL,
        "source_weight": weight,
        "scenarios": list(scenario_names),
        "profile_count": len(grid),
        "grid": grid,
    }


def select_target_offline_guard_with_familywise_control(
    rows: Sequence[Mapping[str, Any]],
    *,
    familywise_alpha: float = 0.025,
    minimum_mean_improvement: float = 0.002,
    maximum_fold_regression: float = 0.01,
) -> dict[str, Any]:
    """Freeze a deployment guard from target-offline OOF folds only."""

    alpha = float(familywise_alpha)
    if (
        not 0.0 < alpha < 1.0
        or float(minimum_mean_improvement) < 0.0
        or float(maximum_fold_regression) < 0.0
    ):
        raise ValueError("invalid family-wise target-offline guard thresholds")
    normalized = []
    for row in rows:
        normalized.append(
            {
                "fold_index": int(row["fold_index"]),
                "profile_key": str(row["profile_key"]),
                "risk_multiplier": float(row["risk_multiplier"]),
                "max_relative_rule_gap": float(row["max_relative_rule_gap"]),
                "min_context_trust": float(row["min_context_trust"]),
                "margin": float(row["margin"]),
                "delta": float(
                    row["equal_scenario_mean_normalized_delta_vs_rule"]
                ),
                "accepted_override_count": int(row["accepted_override_count"]),
            }
        )
    folds = tuple(sorted({row["fold_index"] for row in normalized}))
    profiles = tuple(sorted({row["profile_key"] for row in normalized}))
    identities = {
        (row["fold_index"], row["profile_key"]): row for row in normalized
    }
    if (
        len(folds) < 2
        or not profiles
        or len(identities) != len(normalized)
        or set(identities)
        != {(fold, profile) for fold in folds for profile in profiles}
    ):
        raise ValueError("target-offline guard fold matrix is incomplete")
    critical = float(NormalDist().inv_cdf(1.0 - alpha / len(profiles)))
    grid = {}
    eligible = []
    for profile in profiles:
        profile_rows = [identities[(fold, profile)] for fold in folds]
        specs = {
            (
                row["risk_multiplier"],
                row["max_relative_rule_gap"],
                row["min_context_trust"],
                row["margin"],
            )
            for row in profile_rows
        }
        if len(specs) != 1:
            raise ValueError("target-offline guard profile changed across folds")
        risk, gap, trust, margin = specs.pop()
        values = np.asarray([row["delta"] for row in profile_rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("target-offline guard deltas must be finite")
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        mean_delta = float(np.mean(values))
        upper = mean_delta + critical * standard_error
        worst = float(np.max(values))
        accepted = int(
            sum(row["accepted_override_count"] for row in profile_rows)
        )
        is_eligible = bool(
            accepted > 0
            and upper <= -float(minimum_mean_improvement)
            and worst <= float(maximum_fold_regression)
        )
        summary = {
            "profile_key": profile,
            "risk_multiplier": risk,
            "max_relative_rule_gap": gap,
            "min_context_trust": trust,
            "margin": margin,
            "fold_normalized_deltas_vs_rule": values.tolist(),
            "mean_normalized_delta_vs_rule": mean_delta,
            "standard_error": standard_error,
            "simultaneous_upper_confidence_delta": upper,
            "worst_fold_delta": worst,
            "accepted_override_count": accepted,
            "eligible": is_eligible,
        }
        grid[profile] = summary
        if is_eligible:
            eligible.append(summary)
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["mean_normalized_delta_vs_rule"]),
                float(row["simultaneous_upper_confidence_delta"]),
                float(row["max_relative_rule_gap"]),
                -float(row["risk_multiplier"]),
            ),
        )
        guard_config = {
            "enabled": True,
            "risk_multiplier": selected["risk_multiplier"],
            "min_context_trust": selected["min_context_trust"],
            "margin": selected["margin"],
            "max_relative_rule_gap": selected["max_relative_rule_gap"],
        }
        reason = "familywise_target_offline_improvement"
    else:
        selected = None
        guard_config = {
            "enabled": False,
            "risk_multiplier": 1.0,
            "min_context_trust": 0.1,
            "margin": 0.0,
            "max_relative_rule_gap": 1.0,
        }
        reason = "pressure_prior_fallback"
    return {
        "protocol": OFFLINE_GUARD_SELECTION_PROTOCOL,
        "folds": list(folds),
        "profile_count": len(profiles),
        "familywise_alpha": alpha,
        "multiple_comparison_correction": "bonferroni_one_sided_normal",
        "simultaneous_confidence_multiplier": critical,
        "minimum_mean_improvement": float(minimum_mean_improvement),
        "maximum_fold_regression": float(maximum_fold_regression),
        "selection_reason": reason,
        "selected_profile_key": (
            selected["profile_key"] if selected is not None else None
        ),
        "guard_config": guard_config,
        "grid": grid,
        "target_closed_loop_rollouts_used_for_selection": False,
    }


def evaluate_source_weight_grid_from_offline_groups(
    dataset,
    *,
    reference_policy: Any,
    source_model: Any,
    target_only_model: Any,
    source_objective_mode: str,
    target_only_objective_mode: str,
    scenarios: Sequence[str],
    weights: Sequence[float] = SOURCE_WEIGHT_GRID,
    contrast_features: Sequence[str] = CONTRAST_FEATURES_V3,
) -> dict[str, Any]:
    """Evaluate source shrinkage on held-out target action groups.

    The model is never scored on target groups used to fit it.  Each scenario
    receives equal weight so a large scenario cannot decide source admission
    merely by contributing more intersections or snapshots.
    """

    allowed = tuple(float(value) for value in weights)
    scenario_names = tuple(str(value) for value in scenarios)
    if (
        not allowed
        or len(allowed) != len(set(allowed))
        or 0.0 not in allowed
        or any(not 0.0 <= value <= 1.0 for value in allowed)
        or not scenario_names
        or len(scenario_names) != len(set(scenario_names))
    ):
        raise ValueError("invalid offline source evaluation grid")
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=reference_policy,
        contrast_features=tuple(str(value) for value in contrast_features),
    )
    actual = np.asarray(contrast.targets["interval_cost"], dtype=float)
    if actual.shape != (contrast.size,) or not np.all(np.isfinite(actual)):
        raise ValueError("offline action targets must be finite")
    groups = np.asarray(action_group_ids(contrast), dtype=str)
    source_score, source_uncertainty, source_trust, _ = _group_adjusted_scores(
        contrast,
        source_model.predict(contrast),
        objective_mode=str(source_objective_mode),
    )
    target_score, target_uncertainty, target_trust, _ = _group_adjusted_scores(
        contrast,
        target_only_model.predict(contrast),
        objective_mode=str(target_only_objective_mode),
    )
    for name, values in (
        ("source_score", source_score),
        ("target_score", target_score),
        ("source_uncertainty", source_uncertainty),
        ("target_uncertainty", target_uncertainty),
        ("source_trust", source_trust),
        ("target_trust", target_trust),
    ):
        array = np.asarray(values, dtype=float)
        if array.shape != (contrast.size,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain one finite value per row")

    group_rows = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        matches = [
            scenario
            for scenario in scenario_names
            if str(group).startswith(f"{scenario}:")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot identify exactly one target scenario for action group {group}"
            )
        group_rows.append((str(group), matches[0], rows))
    observed_scenarios = {scenario for _, scenario, _ in group_rows}
    if observed_scenarios != set(scenario_names):
        raise ValueError("offline source fold does not cover every target scenario")

    grid: dict[str, Any] = {}
    score_delta = np.asarray(source_score) - np.asarray(target_score)
    for weight in allowed:
        score = np.asarray(target_score) + float(weight) * score_delta
        scenario_regrets: dict[str, list[float]] = {
            scenario: [] for scenario in scenario_names
        }
        optimal = 0
        for _, scenario, rows in group_rows:
            selected = int(rows[int(np.argmin(score[rows]))])
            values = actual[rows]
            best = float(np.min(values))
            regret = (float(actual[selected]) - best) / action_group_range(values)
            scenario_regrets[scenario].append(float(regret))
            optimal += int(np.isclose(regret, 0.0, rtol=0.0, atol=1e-12))
        scenario_means = {
            scenario: float(np.mean(values))
            for scenario, values in scenario_regrets.items()
        }
        grid[str(weight)] = {
            "source_weight": float(weight),
            "group_count": len(group_rows),
            "scenario_group_counts": {
                scenario: len(scenario_regrets[scenario])
                for scenario in scenario_names
            },
            "scenario_mean_normalized_action_regret": scenario_means,
            "equal_scenario_mean_normalized_action_regret": float(
                np.mean([scenario_means[scenario] for scenario in scenario_names])
            ),
            "optimal_action_rate": float(optimal / len(group_rows)),
        }
    return {
        "protocol": OFFLINE_SOURCE_EVALUATION_PROTOCOL,
        "group_count": len(group_rows),
        "scenarios": list(scenario_names),
        "weights": list(allowed),
        "grid": grid,
    }


def select_source_city_weight_from_offline_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str = "target_only",
    minimum_mean_improvement: float = 0.005,
    maximum_fold_regression: float = 0.01,
    confidence_multiplier: float = 1.645,
) -> dict[str, Any]:
    """Select a source city and mass using only held-out target offline folds."""

    if (
        float(minimum_mean_improvement) < 0.0
        or float(maximum_fold_regression) < 0.0
        or float(confidence_multiplier) < 0.0
    ):
        raise ValueError("offline source selection thresholds must be nonnegative")
    normalized = []
    for row in rows:
        regret = float(row["equal_scenario_mean_normalized_action_regret"])
        weight = float(row["source_weight"])
        candidate = str(row["candidate_key"])
        source_group = row.get("source_city_group")
        if (
            not np.isfinite(regret)
            or regret < 0.0
            or not 0.0 <= weight <= 1.0
            or (candidate == baseline_key) != (weight == 0.0)
            or (candidate == baseline_key) != (source_group is None)
        ):
            raise ValueError("invalid offline source-selection row")
        normalized.append(
            {
                "fold_index": int(row["fold_index"]),
                "candidate_key": candidate,
                "source_city_group": (
                    None if source_group is None else str(source_group)
                ),
                "source_weight": weight,
                "regret": regret,
            }
        )
    folds = tuple(sorted({row["fold_index"] for row in normalized}))
    candidates = tuple(sorted({row["candidate_key"] for row in normalized}))
    if not folds or baseline_key not in candidates:
        raise ValueError("offline source-selection matrix has no baseline")
    identities = {
        (row["fold_index"], row["candidate_key"]): row for row in normalized
    }
    expected = {
        (fold, candidate) for fold in folds for candidate in candidates
    }
    if len(identities) != len(normalized) or set(identities) != expected:
        raise ValueError("offline source-selection matrix is incomplete")
    candidate_specs = {}
    for candidate in candidates:
        specs = {
            (
                identities[(fold, candidate)]["source_city_group"],
                identities[(fold, candidate)]["source_weight"],
            )
            for fold in folds
        }
        if len(specs) != 1:
            raise ValueError("offline source candidate changes identity across folds")
        candidate_specs[candidate] = specs.pop()

    grid: dict[str, Any] = {}
    eligible = []
    for candidate in candidates:
        deltas = np.asarray(
            [
                identities[(fold, candidate)]["regret"]
                - identities[(fold, baseline_key)]["regret"]
                for fold in folds
            ],
            dtype=float,
        )
        mean_delta = float(np.mean(deltas))
        standard_error = float(
            np.std(deltas, ddof=1) / np.sqrt(len(deltas))
            if len(deltas) > 1
            else 0.0
        )
        upper_bound = float(
            mean_delta + float(confidence_multiplier) * standard_error
        )
        worst_delta = float(np.max(deltas))
        source_group, weight = candidate_specs[candidate]
        is_eligible = bool(
            candidate != baseline_key
            and upper_bound <= -float(minimum_mean_improvement)
            and worst_delta <= float(maximum_fold_regression)
        )
        summary = {
            "candidate_key": candidate,
            "source_city_group": source_group,
            "source_weight": float(weight),
            "fold_regret_deltas": deltas.tolist(),
            "mean_regret_delta": mean_delta,
            "standard_error": standard_error,
            "upper_confidence_regret_delta": upper_bound,
            "worst_fold_regret_delta": worst_delta,
            "eligible": is_eligible,
        }
        grid[candidate] = summary
        if is_eligible:
            eligible.append(summary)
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["upper_confidence_regret_delta"]),
                float(row["mean_regret_delta"]),
                float(row["source_weight"]),
                str(row["candidate_key"]),
            ),
        )
        reason = "offline_cross_fitted_source_benefit"
    else:
        selected = grid[baseline_key]
        reason = "target_only_fallback"
    return {
        "protocol": OFFLINE_SOURCE_SELECTION_PROTOCOL,
        "folds": list(folds),
        "baseline_key": str(baseline_key),
        "selected_candidate_key": str(selected["candidate_key"]),
        "selected_source_city_group": selected["source_city_group"],
        "selected_source_weight": float(selected["source_weight"]),
        "selection_reason": reason,
        "minimum_mean_improvement": float(minimum_mean_improvement),
        "maximum_fold_regression": float(maximum_fold_regression),
        "confidence_multiplier": float(confidence_multiplier),
        "grid": grid,
    }


def select_source_city_weight_with_familywise_control(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str = "target_only",
    minimum_mean_improvement: float = 0.005,
    maximum_fold_regression: float = 0.01,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Suppress negative transfer after simultaneous source-weight search.

    The selector applies a one-sided Bonferroni critical value across every
    nonbaseline source-weight candidate. A rejected search returns the exact
    target-only candidate, so source data cannot alter deployment ties.
    """

    alpha = float(familywise_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0, 1)")
    candidates = {
        str(row["candidate_key"])
        for row in rows
        if str(row["candidate_key"]) != str(baseline_key)
    }
    if not candidates:
        raise ValueError("familywise source selection has no transfer candidates")
    critical = float(NormalDist().inv_cdf(1.0 - alpha / len(candidates)))
    selected = select_source_city_weight_from_offline_folds(
        rows,
        baseline_key=baseline_key,
        minimum_mean_improvement=minimum_mean_improvement,
        maximum_fold_regression=maximum_fold_regression,
        confidence_multiplier=critical,
    )
    eligible = [
        row
        for key, row in selected["grid"].items()
        if key != baseline_key and bool(row["eligible"])
    ]
    if eligible:
        winner = min(
            eligible,
            key=lambda row: (
                float(row["mean_regret_delta"]),
                float(row["upper_confidence_regret_delta"]),
                float(row["source_weight"]),
                str(row["candidate_key"]),
            ),
        )
        selected = {
            **selected,
            "selected_candidate_key": str(winner["candidate_key"]),
            "selected_source_city_group": winner["source_city_group"],
            "selected_source_weight": float(winner["source_weight"]),
            "selection_reason": "best_mean_among_familywise_certified_sources",
        }
    return {
        **selected,
        "protocol": FAMILYWISE_OFFLINE_SOURCE_SELECTION_PROTOCOL,
        "familywise_alpha": alpha,
        "multiple_comparison_correction": "bonferroni_one_sided_normal",
        "nonbaseline_candidate_count": len(candidates),
        "simultaneous_confidence_multiplier": critical,
        "certified_candidate_selection_rule": (
            "minimum_mean_regret_delta_after_familywise_admission"
        ),
        "negative_transfer_fallback": "exact_target_only",
    }


def select_source_weight_from_closed_loop(
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[float] = SOURCE_WEIGHT_GRID,
    baseline_weight: float = 0.0,
    minimum_mean_improvement: float = 0.005,
    maximum_seed_regression: float = 0.01,
) -> dict[str, Any]:
    """Select source weight on paired target-simulator calibration seeds.

    Waiting is minimized.  A nonzero source weight must improve the mean,
    respect the per-seed regression tolerance, and not add collision or teleport
    incidents.  Otherwise deployment falls back to target-only weight zero.
    """

    allowed = tuple(float(value) for value in weights)
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or any(not 0.0 <= value <= 1.0 for value in allowed)
        or float(baseline_weight) not in allowed
    ):
        raise ValueError("invalid source-weight grid")
    normalized = []
    for row in rows:
        weight = float(row["source_weight"])
        if weight not in allowed:
            raise ValueError(f"unexpected source weight: {weight}")
        waiting = float(row["waiting"])
        if not np.isfinite(waiting) or waiting < 0.0:
            raise ValueError("waiting must be finite and nonnegative")
        normalized.append(
            {
                "seed": int(row["seed"]),
                "source_weight": weight,
                "waiting": waiting,
                "collision_incidents": int(row.get("collision_incidents", 0)),
                "teleports": int(row.get("teleports", 0)),
            }
        )
    seeds = sorted({row["seed"] for row in normalized})
    by_identity = {
        (row["seed"], row["source_weight"]): row for row in normalized
    }
    if len(by_identity) != len(normalized) or any(
        (seed, weight) not in by_identity for seed in seeds for weight in allowed
    ):
        raise ValueError("source-weight calibration matrix is incomplete")

    grid: dict[str, Any] = {}
    feasible = []
    for weight in allowed:
        deltas = []
        collision_excess = 0
        teleport_excess = 0
        for seed in seeds:
            candidate = by_identity[(seed, weight)]
            baseline = by_identity[(seed, float(baseline_weight))]
            deltas.append(
                (candidate["waiting"] - baseline["waiting"])
                / max(baseline["waiting"], 1e-9)
            )
            collision_excess += max(
                candidate["collision_incidents"]
                - baseline["collision_incidents"],
                0,
            )
            teleport_excess += max(
                candidate["teleports"] - baseline["teleports"],
                0,
            )
        values = np.asarray(deltas, dtype=float)
        mean_delta = float(np.mean(values))
        worst_delta = float(np.max(values))
        robust_score = float(
            mean_delta
            + 0.5 * np.std(values)
            + max(worst_delta, 0.0)
        )
        eligible = bool(
            weight > 0.0
            and mean_delta <= -float(minimum_mean_improvement)
            and worst_delta <= float(maximum_seed_regression)
            and collision_excess == 0
            and teleport_excess == 0
        )
        row = {
            "source_weight": weight,
            "seed_relative_deltas": values.tolist(),
            "mean_relative_delta": mean_delta,
            "worst_seed_relative_delta": worst_delta,
            "robust_score": robust_score,
            "collision_excess": int(collision_excess),
            "teleport_excess": int(teleport_excess),
            "eligible": eligible,
        }
        grid[str(weight)] = row
        if eligible:
            feasible.append(row)
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                float(row["robust_score"]),
                float(row["mean_relative_delta"]),
                float(row["source_weight"]),
            ),
        )
        reason = "robust_nonzero_source_weight"
    else:
        selected = grid[str(float(baseline_weight))]
        reason = "target_only_fallback"
    return {
        "protocol": SELECTION_PROTOCOL,
        "selected_source_weight": float(selected["source_weight"]),
        "selection_reason": reason,
        "calibration_seeds": seeds,
        "minimum_mean_improvement": float(minimum_mean_improvement),
        "maximum_seed_regression": float(maximum_seed_regression),
        "grid": grid,
    }


def select_source_city_candidate_from_closed_loop(
    rows: Sequence[Mapping[str, Any]],
    *,
    city: str,
    scenarios: Sequence[str],
    candidate_keys: Sequence[str],
    baseline_key: str = "target_only",
    minimum_mean_improvement: float = 0.005,
    maximum_seed_regression: float = 0.01,
) -> dict[str, Any]:
    """Select one source-city candidate on paired, equal-scenario rollouts."""

    expected_scenarios = tuple(str(value) for value in scenarios)
    candidates = tuple(str(value) for value in candidate_keys)
    if (
        not expected_scenarios
        or len(expected_scenarios) != len(set(expected_scenarios))
        or not candidates
        or len(candidates) != len(set(candidates))
        or baseline_key not in candidates
    ):
        raise ValueError("invalid source-city selection grid")
    normalized = []
    for row in rows:
        if str(row["city"]) != str(city):
            continue
        scenario = str(row["scenario"])
        candidate = str(row["candidate_key"])
        if scenario not in expected_scenarios or candidate not in candidates:
            raise ValueError("source-city selection row is outside its grid")
        metrics = row.get("metrics", row)
        waiting = float(
            metrics.get("mean_tripinfo_waiting_time", metrics.get("waiting"))
        )
        if not np.isfinite(waiting) or waiting < 0.0:
            raise ValueError("waiting must be finite and nonnegative")
        normalized.append(
            {
                "scenario": scenario,
                "seed": int(row["seed"]),
                "candidate_key": candidate,
                "waiting": waiting,
                "collision_incidents": int(metrics.get("collision_incidents", 0)),
                "teleports": int(metrics.get("starting_teleports", 0))
                + int(metrics.get("ending_teleports", 0))
                + int(metrics.get("teleports", 0)),
            }
        )
    seeds = tuple(sorted({row["seed"] for row in normalized}))
    by_identity = {
        (row["scenario"], row["seed"], row["candidate_key"]): row
        for row in normalized
    }
    expected = {
        (scenario, seed, candidate)
        for scenario in expected_scenarios
        for seed in seeds
        for candidate in candidates
    }
    if not seeds or len(by_identity) != len(normalized) or set(by_identity) != expected:
        raise ValueError("source-city closed-loop matrix is incomplete")

    aggregate = {}
    for seed in seeds:
        for candidate in candidates:
            scenario_rows = [
                by_identity[(scenario, seed, candidate)]
                for scenario in expected_scenarios
            ]
            aggregate[(seed, candidate)] = {
                "waiting": float(np.mean([row["waiting"] for row in scenario_rows])),
                "collision_incidents": int(
                    sum(row["collision_incidents"] for row in scenario_rows)
                ),
                "teleports": int(sum(row["teleports"] for row in scenario_rows)),
            }

    grid: dict[str, Any] = {}
    eligible = []
    for candidate in candidates:
        relative = []
        collision_excess = 0
        teleport_excess = 0
        for seed in seeds:
            row = aggregate[(seed, candidate)]
            baseline = aggregate[(seed, baseline_key)]
            relative.append(
                (row["waiting"] - baseline["waiting"])
                / max(baseline["waiting"], 1e-9)
            )
            collision_excess += max(
                row["collision_incidents"] - baseline["collision_incidents"], 0
            )
            teleport_excess += max(row["teleports"] - baseline["teleports"], 0)
        values = np.asarray(relative, dtype=float)
        mean_delta = float(np.mean(values))
        worst_delta = float(np.max(values))
        robust_score = float(
            mean_delta + 0.5 * np.std(values) + max(worst_delta, 0.0)
        )
        is_eligible = bool(
            candidate != baseline_key
            and mean_delta <= -float(minimum_mean_improvement)
            and worst_delta <= float(maximum_seed_regression)
            and collision_excess == 0
            and teleport_excess == 0
        )
        summary = {
            "candidate_key": candidate,
            "seed_relative_deltas": values.tolist(),
            "mean_relative_delta": mean_delta,
            "worst_seed_relative_delta": worst_delta,
            "robust_score": robust_score,
            "collision_excess": int(collision_excess),
            "teleport_excess": int(teleport_excess),
            "eligible": is_eligible,
        }
        grid[candidate] = summary
        if is_eligible:
            eligible.append(summary)
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["robust_score"]),
                float(row["mean_relative_delta"]),
                str(row["candidate_key"]),
            ),
        )
        reason = "robust_source_city_component"
    else:
        selected = grid[baseline_key]
        reason = "target_only_fallback"
    return {
        "protocol": SOURCE_CITY_SELECTION_PROTOCOL,
        "city": str(city),
        "scenarios": list(expected_scenarios),
        "seeds": list(seeds),
        "baseline_key": str(baseline_key),
        "selected_candidate_key": str(selected["candidate_key"]),
        "selection_reason": reason,
        "minimum_mean_improvement": float(minimum_mean_improvement),
        "maximum_seed_regression": float(maximum_seed_regression),
        "grid": grid,
    }
