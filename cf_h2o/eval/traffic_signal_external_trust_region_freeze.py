"""Freeze an OOF-calibrated deployment trust region for external TSC cities.

This stage does not refit or inspect a held-out closed-loop rollout.  It
recreates the two seed-blocked adaptation folds, obtains out-of-fold action
predictions for the already frozen anchored candidate, and selects a
pressure-anchored residual guard.  The deployment model remains the v43 full
refit; this artifact contains only the guard and its audit trail.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    COLLECTION_SHARDS,
    CORRECTION_FAMILY,
    SOURCE_SEEDS,
    _local_group_alpha,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    CANDIDATE_KEYS,
    CONFIDENCE_MULTIPLIERS,
    CONSTANT_ALPHAS,
    DISAGREEMENT_CAPS,
    constant_candidate_key,
    local_candidate_key,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    SELECTION_SEED,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    _load_city_models,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    FOLD_COUNT,
    PROTOCOL,
    _validate_fit_isolation,
    _validate_full_protocol,
    assign_leave_one_seed_out_folds,
    select_all_city_adaptation_groups,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    ContrastGuardConfig,
    PriorRegularizationConfig,
    _group_adjusted_scores,
    _relative_contrast_rule_gap,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_ranker import domain_normalized_action_target
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig
from cf_h2o.traffic_signal.target_action_support import TARGET_SUPPORT_FEATURES


RESULT_PROTOCOL = "tsc-v44r40-external-oof-trust-region-freeze-v1"
SELECTION_PROTOCOL = "seed-blocked-oof-conformal-pressure-trust-region-v1"
COORDINATION_MODE = "sparse"
CONFORMAL_COVERAGE = 0.90
RISK_MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
MIN_CONTEXT_TRUSTS = (0.10, 0.25, 0.50)
MAX_RELATIVE_RULE_GAPS = (0.0, 0.05, 0.10, 0.25)
MAX_STRATUM_MEAN_DELTA = 0.01
MAX_FOLD_MEAN_DELTA = 0.005
MIN_MEAN_IMPROVEMENT = 0.001
MIN_OVERRIDE_FRACTION = 0.005
MAX_OVERRIDE_FRACTION = 0.35
MIN_OPERATIONAL_TOTAL_VEHICLES = 1.0
PRESSURE_REGULARIZER_SELECTION_PROTOCOL = (
    "seed-blocked-oof-pressure-regularized-advantage-v1"
)
PRESSURE_REGULARIZER_BLEND_WEIGHTS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
PRESSURE_REGULARIZER_RISK_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)
_FIT_STATE: dict[str, Any] | None = None


def _finite_sample_upper_quantile(values: Sequence[float], coverage: float) -> float:
    rows = np.sort(np.asarray(values, dtype=float))
    if rows.size == 0 or not np.all(np.isfinite(rows)):
        raise ValueError("conformal calibration values must be finite and nonempty")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("conformal coverage must be in (0, 1)")
    rank = min(int(math.ceil((rows.size + 1) * float(coverage))), rows.size) - 1
    return float(rows[rank])


def _candidate_scores(
    contrast,
    *,
    models,
    selected_candidate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected_candidate not in CANDIDATE_KEYS:
        raise ValueError(f"unknown anchored candidate: {selected_candidate}")
    anchor_score, anchor_uncertainty, anchor_trust, _ = _group_adjusted_scores(
        contrast,
        models.family_models[ANCHOR_FAMILY].predict(contrast),
        objective_mode=models.objective_modes[ANCHOR_FAMILY],
    )
    correction_score, correction_uncertainty, correction_trust, _ = (
        _group_adjusted_scores(
            contrast,
            models.family_models[CORRECTION_FAMILY].predict(contrast),
            objective_mode=models.objective_modes[CORRECTION_FAMILY],
        )
    )
    groups = action_group_ids(contrast)
    alpha = np.zeros(contrast.size, dtype=float)
    constants = {
        constant_candidate_key(value): float(value) for value in CONSTANT_ALPHAS
    }
    locals_ = {
        local_candidate_key(cap, confidence): (float(cap), float(confidence))
        for cap in DISAGREEMENT_CAPS
        for confidence in CONFIDENCE_MULTIPLIERS
    }
    if selected_candidate in constants:
        alpha.fill(constants[selected_candidate])
    else:
        cap, confidence = locals_[selected_candidate]
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            value, _ = _local_group_alpha(
                anchor_score=anchor_score,
                correction_score=correction_score,
                anchor_uncertainty=anchor_uncertainty,
                correction_uncertainty=correction_uncertainty,
                group_rows=rows,
                disagreement_cap=cap,
                confidence_multiplier=confidence,
            )
            alpha[rows] = value
    score = anchor_score + alpha * (correction_score - anchor_score)
    uncertainty = (
        (1.0 - alpha) * anchor_uncertainty + alpha * correction_uncertainty
    )
    trust = np.minimum(anchor_trust, correction_trust)
    return score, uncertainty, trust


def oof_action_records(
    validation,
    *,
    models,
    selected_candidate: str,
    fold_index: int,
    validation_seed: int,
    scenarios: Sequence[str],
    target_support: Any | None = None,
) -> list[dict[str, Any]]:
    all_groups = set(str(value) for value in validation.metadata["action_group_ids"])
    rows_out: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_groups = {
            group for group in all_groups if group.startswith(f"{scenario}:")
        }
        if not scenario_groups:
            raise ValueError(f"trust-region OOF fold lacks scenario {scenario}")
        scenario_data = _group_subset_v3(
            validation,
            selected_groups=scenario_groups,
            metadata_updates={"target_data_role": "trust_region_oof_validation"},
        )
        contrast = build_action_contrast_dataset(
            scenario_data,
            reference_policy=models.prior_spec,
            contrast_features=CONTRAST_FEATURES_V3,
        )
        score, uncertainty, trust = _candidate_scores(
            contrast,
            models=models,
            selected_candidate=selected_candidate,
        )
        local_support = (
            target_support.support(contrast)
            if target_support is not None
            else np.ones(contrast.size, dtype=float)
        )
        trust = np.minimum(trust, local_support)
        actual, _ = domain_normalized_action_target(contrast, "interval_cost")
        groups = action_group_ids(contrast)
        is_reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
        rule_gap = _relative_contrast_rule_gap(contrast)
        feature_index = {name: index for index, name in enumerate(contrast.feature_names)}
        missing_support_features = [
            name for name in TARGET_SUPPORT_FEATURES if name not in feature_index
        ]
        if missing_support_features:
            raise KeyError(
                f"OOF support features missing: {missing_support_features}"
            )
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            references = rows[is_reference[rows]]
            if references.size != 1:
                raise ValueError(f"OOF group {group!r} lacks one pressure reference")
            reference = int(references[0])
            learned = int(rows[int(np.argmin(score[rows]))])
            if not np.isclose(float(score[reference]), 0.0, rtol=0.0, atol=1e-10):
                raise ValueError("anchored score reference is not zero")
            rows_out.append(
                {
                    "fold_index": int(fold_index),
                    "validation_seed": int(validation_seed),
                    "scenario": str(scenario),
                    "group_id": str(group),
                    "learned_differs": bool(learned != reference),
                    "predicted_delta": float(score[learned]),
                    "uncertainty": float(uncertainty[learned]),
                    "context_trust": float(trust[learned]),
                    "local_action_support": float(local_support[learned]),
                    "relative_rule_gap": float(rule_gap[learned]),
                    "actual_delta": float(actual[learned] - actual[reference]),
                    "candidate_features": {
                        name: float(contrast.features[learned, feature_index[name]])
                        for name in TARGET_SUPPORT_FEATURES
                    },
                    "reference_features": {
                        name: float(contrast.features[reference, feature_index[name]])
                        for name in TARGET_SUPPORT_FEATURES
                    },
                }
            )
    return rows_out


def _summarize_guard_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    risk_multiplier: float,
    min_context_trust: float,
    max_relative_rule_gap: float,
    coverage: float,
) -> dict[str, Any]:
    learned = [row for row in records if bool(row["learned_differs"])]
    precondition = [
        row
        for row in learned
        if float(row["context_trust"]) >= float(min_context_trust)
        and float(row["relative_rule_gap"])
        <= float(max_relative_rule_gap) + 1e-12
        and float(row["candidate_features"]["total_veh"])
        >= MIN_OPERATIONAL_TOTAL_VEHICLES
    ]
    if not precondition:
        return {
            "risk_multiplier": float(risk_multiplier),
            "min_context_trust": float(min_context_trust),
            "max_relative_rule_gap": float(max_relative_rule_gap),
            "margin": None,
            "feasible": False,
            "reason": "no_precondition_eligible_oof_proposals",
            "decisions": len(records),
            "learned_proposals": len(learned),
            "precondition_eligible": 0,
            "accepted_overrides": 0,
        }
    fold_errors: dict[str, list[float]] = {}
    for row in precondition:
        fold_errors.setdefault(str(int(row["fold_index"])), []).append(
            float(row["actual_delta"])
            - float(row["predicted_delta"])
            - float(risk_multiplier) * float(row["uncertainty"])
        )
    fold_margins = {
        fold: _finite_sample_upper_quantile(values, coverage)
        for fold, values in fold_errors.items()
    }
    margin = max(0.0, max(fold_margins.values()))
    selected_delta: dict[tuple[int, str], list[float]] = {}
    accepted = 0
    conformal_violations = 0
    for row in records:
        upper = (
            float(row["predicted_delta"])
            + float(risk_multiplier) * float(row["uncertainty"])
            + margin
        )
        use_learned = bool(
            row["learned_differs"]
            and float(row["context_trust"]) >= float(min_context_trust)
            and float(row["relative_rule_gap"])
            <= float(max_relative_rule_gap) + 1e-12
            and float(row["candidate_features"]["total_veh"])
            >= MIN_OPERATIONAL_TOTAL_VEHICLES
            and upper < 0.0
        )
        accepted += int(use_learned)
        conformal_violations += int(
            use_learned and float(row["actual_delta"]) > upper + 1e-12
        )
        key = (int(row["fold_index"]), str(row["scenario"]))
        selected_delta.setdefault(key, []).append(
            float(row["actual_delta"]) if use_learned else 0.0
        )
    stratum_means = {
        f"fold{fold}:{scenario}": float(np.mean(values))
        for (fold, scenario), values in selected_delta.items()
    }
    folds = sorted({fold for fold, _ in selected_delta})
    scenarios = sorted({scenario for _, scenario in selected_delta})
    fold_means = {
        str(fold): float(
            np.mean(
                [
                    np.mean(selected_delta[(fold, scenario)])
                    for scenario in scenarios
                    if (fold, scenario) in selected_delta
                ]
            )
        )
        for fold in folds
    }
    scenario_means = {
        scenario: float(
            np.mean(
                [
                    np.mean(selected_delta[(fold, scenario)])
                    for fold in folds
                    if (fold, scenario) in selected_delta
                ]
            )
        )
        for scenario in scenarios
    }
    stratum_values = np.asarray(list(stratum_means.values()), dtype=float)
    mean_delta = float(np.mean(list(fold_means.values())))
    std_delta = float(np.std(stratum_values))
    override_fraction = accepted / max(len(records), 1)
    minimum_overrides = max(
        2, int(math.ceil(float(MIN_OVERRIDE_FRACTION) * len(records)))
    )
    feasible = bool(
        accepted >= minimum_overrides
        and override_fraction <= MAX_OVERRIDE_FRACTION
        and mean_delta <= -MIN_MEAN_IMPROVEMENT
        and max(fold_means.values(), default=0.0) <= MAX_FOLD_MEAN_DELTA
        and max(stratum_means.values(), default=0.0) <= MAX_STRATUM_MEAN_DELTA
    )
    return {
        "risk_multiplier": float(risk_multiplier),
        "min_context_trust": float(min_context_trust),
        "max_relative_rule_gap": float(max_relative_rule_gap),
        "margin": float(margin),
        "coverage": float(coverage),
        "minimum_operational_total_vehicles": float(
            MIN_OPERATIONAL_TOTAL_VEHICLES
        ),
        "fold_conformal_margins": fold_margins,
        "decisions": len(records),
        "learned_proposals": len(learned),
        "precondition_eligible": len(precondition),
        "accepted_overrides": int(accepted),
        "minimum_required_overrides": int(minimum_overrides),
        "override_fraction": float(override_fraction),
        "mean_oof_delta_vs_pressure": mean_delta,
        "std_stratum_delta_vs_pressure": std_delta,
        "robust_selection_score": float(
            mean_delta
            + 0.25 * std_delta
            + 0.25 * max(max(stratum_means.values(), default=0.0), 0.0)
        ),
        "worst_fold_mean_delta": float(max(fold_means.values(), default=0.0)),
        "worst_stratum_mean_delta": float(
            max(stratum_means.values(), default=0.0)
        ),
        "fold_mean_delta": fold_means,
        "scenario_mean_delta": scenario_means,
        "stratum_mean_delta": stratum_means,
        "accepted_conformal_violations": int(conformal_violations),
        "accepted_conformal_violation_fraction": float(
            conformal_violations / max(accepted, 1)
        ),
        "feasible": feasible,
        "reason": "passed_oof_safe_improvement_gate" if feasible else "failed_oof_safe_improvement_gate",
    }


def select_oof_trust_region(
    records: Sequence[Mapping[str, Any]],
) -> tuple[ContrastGuardConfig, dict[str, Any]]:
    if not records:
        raise ValueError("trust-region selection requires OOF action records")
    identities = {
        (int(row["fold_index"]), str(row["scenario"]), str(row["group_id"]))
        for row in records
    }
    if len(identities) != len(records):
        raise ValueError("OOF trust-region action records are duplicated")
    for row in records:
        features = row.get("candidate_features")
        if not isinstance(features, Mapping) or "total_veh" not in features:
            raise ValueError("OOF trust-region record lacks operational features")
    grid = [
        _summarize_guard_candidate(
            records,
            risk_multiplier=risk,
            min_context_trust=trust,
            max_relative_rule_gap=gap,
            coverage=CONFORMAL_COVERAGE,
        )
        for risk in RISK_MULTIPLIERS
        for trust in MIN_CONTEXT_TRUSTS
        for gap in MAX_RELATIVE_RULE_GAPS
    ]
    feasible = [row for row in grid if bool(row["feasible"])]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                float(row["robust_selection_score"]),
                float(row["mean_oof_delta_vs_pressure"]),
                float(row["override_fraction"]),
                float(row["max_relative_rule_gap"]),
                -float(row["min_context_trust"]),
                -float(row["risk_multiplier"]),
            ),
        )
        config = ContrastGuardConfig(
            enabled=True,
            risk_multiplier=float(selected["risk_multiplier"]),
            min_context_trust=float(selected["min_context_trust"]),
            margin=float(selected["margin"]),
            max_relative_rule_gap=float(selected["max_relative_rule_gap"]),
        )
        decision = "deploy_oof_conformal_trust_region"
    else:
        selected = None
        config = ContrastGuardConfig(enabled=False)
        decision = "deploy_pressure_prior_only"
    return config, {
        "protocol": SELECTION_PROTOCOL,
        "decision": decision,
        "coordination_mode": COORDINATION_MODE,
        "conformal_coverage": CONFORMAL_COVERAGE,
        "grid_contract": {
            "risk_multipliers": list(RISK_MULTIPLIERS),
            "min_context_trusts": list(MIN_CONTEXT_TRUSTS),
            "max_relative_rule_gaps": list(MAX_RELATIVE_RULE_GAPS),
            "minimum_operational_total_vehicles": float(
                MIN_OPERATIONAL_TOTAL_VEHICLES
            ),
        },
        "gate_contract": {
            "max_stratum_mean_delta": MAX_STRATUM_MEAN_DELTA,
            "max_fold_mean_delta": MAX_FOLD_MEAN_DELTA,
            "min_mean_improvement": MIN_MEAN_IMPROVEMENT,
            "min_override_fraction": MIN_OVERRIDE_FRACTION,
            "max_override_fraction": MAX_OVERRIDE_FRACTION,
        },
        "selected": selected,
        "grid": grid,
    }


def _summarize_pressure_regularizer_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    blend_weight: float,
    risk_multiplier: float,
    min_context_trust: float,
) -> dict[str, Any]:
    selected_delta: dict[tuple[int, str], list[float]] = {}
    accepted_by_fold: dict[int, int] = {}
    accepted = 0
    learned_proposals = 0
    for row in records:
        learned_proposals += int(bool(row["learned_differs"]))
        objective = (
            float(row["relative_rule_gap"])
            + float(blend_weight) * float(row["predicted_delta"])
            + float(risk_multiplier) * float(row["uncertainty"])
        )
        use_learned = bool(
            row["learned_differs"]
            and float(row["context_trust"]) >= float(min_context_trust)
            and float(row["candidate_features"]["total_veh"])
            >= MIN_OPERATIONAL_TOTAL_VEHICLES
            and objective < 0.0
        )
        accepted += int(use_learned)
        fold = int(row["fold_index"])
        accepted_by_fold[fold] = accepted_by_fold.get(fold, 0) + int(use_learned)
        selected_delta.setdefault((fold, str(row["scenario"])), []).append(
            float(row["actual_delta"]) if use_learned else 0.0
        )

    folds = sorted({fold for fold, _ in selected_delta})
    scenarios = sorted({scenario for _, scenario in selected_delta})
    stratum_means = {
        f"fold{fold}:{scenario}": float(np.mean(values))
        for (fold, scenario), values in selected_delta.items()
    }
    fold_means = {
        str(fold): float(
            np.mean(
                [
                    np.mean(selected_delta[(fold, scenario)])
                    for scenario in scenarios
                    if (fold, scenario) in selected_delta
                ]
            )
        )
        for fold in folds
    }
    mean_delta = float(np.mean(list(fold_means.values())))
    std_delta = float(np.std(list(stratum_means.values())))
    worst_fold = float(max(fold_means.values(), default=0.0))
    worst_stratum = float(max(stratum_means.values(), default=0.0))
    override_fraction = accepted / max(len(records), 1)
    minimum_overrides = max(
        2, int(math.ceil(float(MIN_OVERRIDE_FRACTION) * len(records)))
    )
    every_fold_covered = bool(folds) and all(
        accepted_by_fold.get(fold, 0) > 0 for fold in folds
    )
    feasible = bool(
        accepted >= minimum_overrides
        and every_fold_covered
        and override_fraction <= MAX_OVERRIDE_FRACTION
        and mean_delta <= -MIN_MEAN_IMPROVEMENT
        and worst_fold <= MAX_FOLD_MEAN_DELTA
        and worst_stratum <= MAX_STRATUM_MEAN_DELTA
    )
    return {
        "blend_weight": float(blend_weight),
        "risk_multiplier": float(risk_multiplier),
        "min_context_trust": float(min_context_trust),
        "minimum_operational_total_vehicles": MIN_OPERATIONAL_TOTAL_VEHICLES,
        "decisions": len(records),
        "learned_proposals": int(learned_proposals),
        "accepted_overrides": int(accepted),
        "accepted_by_fold": {
            str(fold): int(accepted_by_fold.get(fold, 0)) for fold in folds
        },
        "every_fold_covered": every_fold_covered,
        "minimum_required_overrides": int(minimum_overrides),
        "override_fraction": float(override_fraction),
        "mean_oof_delta_vs_pressure": mean_delta,
        "std_stratum_delta_vs_pressure": std_delta,
        "robust_selection_score": float(
            mean_delta
            + 0.25 * std_delta
            + 0.25 * max(worst_stratum, 0.0)
        ),
        "worst_fold_mean_delta": worst_fold,
        "worst_stratum_mean_delta": worst_stratum,
        "fold_mean_delta": fold_means,
        "stratum_mean_delta": stratum_means,
        "feasible": feasible,
        "reason": (
            "passed_oof_pressure_regularizer_gate"
            if feasible
            else "failed_oof_pressure_regularizer_gate"
        ),
    }


def select_oof_pressure_regularizer(
    records: Sequence[Mapping[str, Any]],
) -> tuple[PriorRegularizationConfig, dict[str, Any]]:
    """Select a direct learned-action gate using adaptation OOF outcomes only."""

    if not records:
        raise ValueError("pressure-regularizer selection requires OOF action records")
    identities = {
        (int(row["fold_index"]), str(row["scenario"]), str(row["group_id"]))
        for row in records
    }
    if len(identities) != len(records):
        raise ValueError("OOF pressure-regularizer action records are duplicated")
    for row in records:
        features = row.get("candidate_features")
        if not isinstance(features, Mapping) or "total_veh" not in features:
            raise ValueError("OOF pressure-regularizer record lacks operational features")
    grid = [
        _summarize_pressure_regularizer_candidate(
            records,
            blend_weight=blend,
            risk_multiplier=risk,
            min_context_trust=trust,
        )
        for blend in PRESSURE_REGULARIZER_BLEND_WEIGHTS
        for risk in PRESSURE_REGULARIZER_RISK_MULTIPLIERS
        for trust in MIN_CONTEXT_TRUSTS
    ]
    feasible = [row for row in grid if bool(row["feasible"])]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                float(row["robust_selection_score"]),
                float(row["mean_oof_delta_vs_pressure"]),
                float(row["override_fraction"]),
                float(row["blend_weight"]),
                -float(row["min_context_trust"]),
                -float(row["risk_multiplier"]),
            ),
        )
        config = PriorRegularizationConfig(
            enabled=True,
            blend_weight=float(selected["blend_weight"]),
            risk_multiplier=float(selected["risk_multiplier"]),
            min_context_trust=float(selected["min_context_trust"]),
        )
        decision = "deploy_oof_pressure_regularized_direct_mpc"
    else:
        selected = None
        config = PriorRegularizationConfig()
        decision = "deploy_pressure_prior_only"
    return config, {
        "protocol": PRESSURE_REGULARIZER_SELECTION_PROTOCOL,
        "decision": decision,
        "coordination_mode": "direct",
        "objective": (
            "relative_pressure_gap + blend_weight * predicted_advantage + "
            "risk_multiplier * uncertainty"
        ),
        "grid_contract": {
            "blend_weights": list(PRESSURE_REGULARIZER_BLEND_WEIGHTS),
            "risk_multipliers": list(PRESSURE_REGULARIZER_RISK_MULTIPLIERS),
            "min_context_trusts": list(MIN_CONTEXT_TRUSTS),
            "minimum_operational_total_vehicles": MIN_OPERATIONAL_TOTAL_VEHICLES,
        },
        "gate_contract": {
            "max_stratum_mean_delta": MAX_STRATUM_MEAN_DELTA,
            "max_fold_mean_delta": MAX_FOLD_MEAN_DELTA,
            "min_mean_improvement": MIN_MEAN_IMPROVEMENT,
            "min_override_fraction": MIN_OVERRIDE_FRACTION,
            "max_override_fraction": MAX_OVERRIDE_FRACTION,
            "every_oof_fold_must_have_an_override": True,
        },
        "selected": selected,
        "grid": grid,
    }


def accepted_oof_records(
    records: Sequence[Mapping[str, Any]],
    guard: ContrastGuardConfig,
) -> list[dict[str, Any]]:
    """Return the exact OOF actions authorized by a frozen guard."""

    if not guard.enabled:
        return []
    accepted = []
    for raw in records:
        row = dict(raw)
        features = row.get("candidate_features")
        if not isinstance(features, Mapping) or "total_veh" not in features:
            raise ValueError("OOF trust-region record lacks operational features")
        upper = (
            float(row["predicted_delta"])
            + float(guard.risk_multiplier) * float(row["uncertainty"])
            + float(guard.margin)
        )
        if (
            bool(row["learned_differs"])
            and float(row["context_trust"]) >= float(guard.min_context_trust)
            and float(row["relative_rule_gap"])
            <= float(guard.max_relative_rule_gap) + 1e-12
            and float(features["total_veh"])
            >= MIN_OPERATIONAL_TOTAL_VEHICLES
            and upper < 0.0
        ):
            accepted.append(row)
    return accepted


def _fit_fold(role: str, *, state: Mapping[str, Any]) -> dict[str, Any]:
    fold_index = int(role.removeprefix("fold_"))
    validation_ids = tuple(
        sorted(
            group
            for group, fold in state["fold_assignment"]["assignments"].items()
            if int(fold) == fold_index
        )
    )
    validation_set = set(validation_ids)
    training_ids = tuple(
        group for group in state["selected_group_ids"] if group not in validation_set
    )
    validation_seed = int(
        state["fold_assignment"]["validation_seed_by_fold"][str(fold_index)]
    )
    started = time.monotonic()
    models = fit_target_screening_models(
        state["fit_bank"],
        state["target_key"],
        fit_config=MechanismFitConfig(),
        source_rule_costs=state["source_rule_costs"],
        source_rule_specs=state["source_rule_specs"],
        model_families=BASE_FAMILIES,
        target_group_budget=len(training_ids),
        target_adaptation_selection_seed=SELECTION_SEED,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=5,
        scenario_city_groups=state["city_groups"],
        target_adaptation_group_ids=training_ids,
        target_static_context=state["city_static_context"],
    )
    _validate_fit_isolation(
        models.diagnostics,
        target_key=state["target_key"],
        source_scenarios=state["source_scenarios"],
        expected_domains=set(state["source_city_groups"]) | {state["city"]},
        external_cities=state["external_cities"],
        city=state["city"],
        city_static_context=state["city_static_context"],
    )
    validation = _group_subset_v3(
        state["selected_dataset"],
        selected_groups=validation_set,
        metadata_updates={
            "target_data_role": "external_trust_region_seed_blocked_oof",
            "cross_fit_fold_index": fold_index,
        },
    )
    records = oof_action_records(
        validation,
        models=models,
        selected_candidate=state["selected_candidate"],
        fold_index=fold_index,
        validation_seed=validation_seed,
        scenarios=state["scenarios"],
    )
    elapsed = float(time.monotonic() - started)
    print(
        "CFCMT_EXTERNAL_TRUST_REGION_PROGRESS "
        f"city={state['city']} fold={fold_index + 1}/{FOLD_COUNT} "
        f"records={len(records)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "fold_index": fold_index,
        "validation_seed": validation_seed,
        "training_group_ids": list(training_ids),
        "validation_group_ids": list(validation_ids),
        "model_diagnostics": models.diagnostics,
        "records": records,
        "elapsed_sec": elapsed,
    }


def _fit_worker(role: str) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("trust-region fit process state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_fold(role, state=_FIT_STATE)


def run_external_trust_region_freeze(
    *,
    source_cache_root: Path,
    source_baseline_result: Path,
    source_manifest_path: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_spec_path: Path,
    development_audit_path: Path,
    diagnostic_failure_result_path: Path,
    joint_freeze_audit_path: Path,
    freeze_root: Path,
    city: str,
    expected_source_cache_sha256: str,
    expected_source_baseline_sha256: str,
    expected_external_cache_sha256: str,
    expected_joint_freeze_sha256: str,
    expected_source_tree_sha256: str | None,
    workers: int,
    fit_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol, parent, _ = _validate_full_protocol(
        protocol_spec_path,
        development_audit_path,
        diagnostic_failure_result_path,
    )
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("trust-region parent protocol changed")
    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("trust-region source-tree SHA-256 mismatch")
    joint = json.loads(Path(joint_freeze_audit_path).read_text(encoding="utf-8"))
    if (
        _sha256(joint_freeze_audit_path) != str(expected_joint_freeze_sha256)
        or joint.get("status") != "PASS"
    ):
        raise ValueError("trust-region joint model freeze audit changed")

    source_sha, source_files = aggregate_cache_sha256(source_cache_root)
    external_sha, external_files = aggregate_cache_sha256(external_cache_root)
    source_evidence = dict(parent["source_evidence"])
    parent_cache = dict(parent["counterfactual_cache"])
    if (
        source_sha != str(expected_source_cache_sha256)
        or str(source_evidence["counterfactual_cache_sha256"])
        != str(expected_source_cache_sha256)
        or source_files != 18 * len(SOURCE_SEEDS) * COLLECTION_SHARDS
        or external_sha != str(expected_external_cache_sha256)
        or _sha256(source_manifest_path) != str(source_evidence["manifest_sha256"])
        or _sha256(external_manifest_path) != str(parent_cache["manifest_sha256"])
    ):
        raise ValueError("trust-region counterfactual cache identity changed")
    if (
        _sha256(source_baseline_result) != str(expected_source_baseline_sha256)
        or str(source_evidence["source_rule_baseline_sha256"])
        != str(expected_source_baseline_sha256)
    ):
        raise ValueError("trust-region source baseline identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    city_scenarios = {
        str(name): tuple(str(value) for value in values)
        for name, values in protocol["target_protocol"]["external_city_scenarios"].items()
    }
    if city not in city_scenarios:
        raise ValueError(f"unknown trust-region city: {city}")
    scenarios = city_scenarios[city]
    expected_external_files = (
        sum(len(values) for values in city_scenarios.values())
        * len(ADAPTATION_SEEDS)
        * EXTERNAL_COLLECTION_SHARDS
    )
    if external_files != expected_external_files:
        raise ValueError("trust-region external cache file count changed")

    method_result, method_payload, _ = _load_city_models(
        city=city,
        freeze_root=freeze_root,
        joint_audit=joint,
    )
    selected_candidate = str(method_result["selected_candidate"])
    source_bank, source_cache_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        source_manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    external_bank, external_cache_audit = load_frozen_counterfactual_bank(
        external_cache_root,
        external_manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    selected_group_ids, selection_audit = select_all_city_adaptation_groups(
        external_bank, city=city, scenarios=scenarios
    )
    if (
        tuple(method_payload["selected_group_ids"]) != selected_group_ids
        or tuple(method_payload["deployment_refit_group_ids"]) != selected_group_ids
    ):
        raise ValueError("trust-region groups differ from the deployment refit")
    full_city_dataset = _merge_city_datasets(external_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={"target_data_role": "trust_region_adaptation_pool"},
    )
    records = target_group_records(
        selected_dataset,
        selected_group_ids=selected_group_ids,
        expected_group_count=len(selected_group_ids),
    )
    fold_assignment = assign_leave_one_seed_out_folds(records, scenarios=scenarios)
    if fold_assignment["assignments"] != method_payload["fold_assignment"]:
        raise ValueError("trust-region fold assignment differs from frozen model")

    scenario_static_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_static_context = np.mean(
        np.vstack([scenario_static_contexts[scenario] for scenario in scenarios]),
        axis=0,
    )
    target_key = f"external_city_{city}"
    fit_bank = dict(source_bank)
    fit_bank[target_key] = selected_dataset
    city_groups = dict(source_manifest.city_groups)
    city_groups[target_key] = city
    source_baseline = json.loads(Path(source_baseline_result).read_text(encoding="utf-8"))
    source_rule_costs = source_baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    state = {
        "fit_bank": fit_bank,
        "target_key": target_key,
        "city_groups": city_groups,
        "source_rule_costs": source_rule_costs,
        "source_rule_specs": source_rule_specs,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment,
        "selected_dataset": selected_dataset,
        "selected_candidate": selected_candidate,
        "scenarios": scenarios,
        "source_scenarios": tuple(source_manifest.sumocfgs),
        "source_city_groups": tuple(set(source_manifest.city_groups.values())),
        "external_cities": tuple(city_scenarios),
        "city": city,
        "city_static_context": city_static_context,
    }
    roles = tuple(f"fold_{index}" for index in range(FOLD_COUNT))
    actual_workers = min(max(int(fit_workers), 1), len(roles))
    if actual_workers == 1:
        fold_results = [_fit_fold(role, state=state) for role in roles]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("trust-region parallel fit requires Linux fork")
        global _FIT_STATE
        _FIT_STATE = state
        try:
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {role: pool.submit(_fit_worker, role) for role in roles}
                fold_results = [futures[role].result() for role in roles]
        finally:
            _FIT_STATE = None
    fold_results.sort(key=lambda row: int(row["fold_index"]))
    oof_records = [record for fold in fold_results for record in fold.pop("records")]
    guard, selection = select_oof_trust_region(oof_records)
    guard_payload = {
        "enabled": guard.enabled,
        "risk_multiplier": guard.risk_multiplier,
        "min_context_trust": guard.min_context_trust,
        "margin": guard.margin,
        "max_relative_rule_gap": guard.max_relative_rule_gap,
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "claim_boundary": (
            "target-offline adaptation guard selection only; no held-out seed, "
            "closed-loop outcome, or other external city enters selection"
        ),
        "parent_protocol_sha256": _sha256(protocol_spec_path),
        "joint_freeze_audit_sha256": _sha256(joint_freeze_audit_path),
        "frozen_method_model_sha256": joint["cities"][city]["method_model"]["sha256"],
        "frozen_method_result_sha256": joint["cities"][city]["method_result"]["sha256"],
        "source_cache_aggregate_sha256": source_sha,
        "source_cache_file_count": source_files,
        "source_cache_audit": source_cache_audit,
        "external_cache_aggregate_sha256": external_sha,
        "external_cache_file_count": external_files,
        "external_cache_audit": external_cache_audit,
        "source_baseline_sha256": _sha256(source_baseline_result),
        "city": city,
        "scenarios": list(scenarios),
        "adaptation_seeds": list(ADAPTATION_SEEDS),
        "excluded_seed_roles": {
            "diagnostic_only": protocol["target_protocol"]["diagnostic_only_seeds"],
            "offline_evaluation": protocol["target_protocol"]["evaluation_seeds"],
            "closed_loop_development": protocol["target_protocol"]["closed_loop_seeds"],
        },
        "selected_candidate": selected_candidate,
        "selected_group_ids": list(selected_group_ids),
        "selection_audit": selection_audit,
        "fold_assignment": fold_assignment,
        "fit_workers": actual_workers,
        "fold_results": fold_results,
        "oof_action_record_count": len(oof_records),
        "oof_action_records": oof_records,
        "deployment": {
            "policy": (
                "cfcmt_selected_conformal_guard"
                if guard.enabled
                else "selected_source_prior"
            ),
            "guard": guard_payload,
            "coordination_mode": COORDINATION_MODE,
            "cooldown_intervals": "prediction_horizon_minus_one_interval",
        },
        "selection": selection,
        "freeze_gate": {
            "all_adaptation_groups_used": True,
            "seed_blocked_oof_predictions": True,
            "heldout_seed8171_absent": True,
            "closed_loop_seeds_8081_9091_absent": True,
            "other_external_city_absent": True,
            "deployment_model_unchanged": True,
            "passed": True,
        },
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-baseline-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--diagnostic-failure-result", type=Path, required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-baseline-sha256", required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fit-workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite trust-region freeze: {args.out}")
    payload = run_external_trust_region_freeze(
        source_cache_root=args.source_cache_root,
        source_baseline_result=args.source_baseline_result,
        source_manifest_path=args.source_manifest,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_spec_path=args.protocol_spec,
        development_audit_path=args.development_audit,
        diagnostic_failure_result_path=args.diagnostic_failure_result,
        joint_freeze_audit_path=args.joint_freeze_audit,
        freeze_root=args.freeze_root,
        city=args.city,
        expected_source_cache_sha256=args.expected_source_cache_sha256,
        expected_source_baseline_sha256=args.expected_source_baseline_sha256,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        expected_joint_freeze_sha256=args.expected_joint_freeze_sha256,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        workers=args.workers,
        fit_workers=args.fit_workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "city": args.city,
                "decision": payload["selection"]["decision"],
                "guard": payload["deployment"]["guard"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
