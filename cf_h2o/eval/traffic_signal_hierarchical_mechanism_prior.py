"""Leave-target-city-out robust pooled mechanism priors for rigid CFCMT."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_cross_city_meta_source_utility import (
    AGGREGATE_PROTOCOL as V151A_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    MINIMUM_OOF_GAIN,
    RIDGE_L2,
    _arm_summary,
    _group_rows,
    _paired_effects,
    _read_json,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    _city_bootstrap,
    _rigid_score,
    _validate_authorization,
    residual_corrected_rigid_score,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    _paired_bootstrap,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _policy_arrays,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    CONTRAST_FEATURES,
    TARGET_BUDGET,
    _PreparedProblem,
    _SourceStatistics,
    _feature_scale,
    _prepare_inventory,
    _prepare_problem,
    _prepare_source_statistics_inventory,
    _ridge_from_unscaled_statistics,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import _target_model
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _normalized_pressure_targets,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    MECHANISM_ACTION_FEATURES,
    group_balanced_weights,
    mechanism_parameter_design,
)


RESULT_PROTOCOL = "tsc-v152a-hierarchical-mechanism-prior-target-v1"
AGGREGATE_PROTOCOL = "tsc-v152a-hierarchical-mechanism-prior-aggregate-v1"
METHOD_PROTOCOL = "leave-target-city-out-robust-pooled-mechanism-prior-v1"
PRIOR_PROTOCOL = "equal-city-median-coefficient-adaptive-precision-v1"
V151A_REJECT_DECISION = (
    "retain_rigid_target_adaptation_and_reject_v151a_source_claim"
)
V152A_REJECT_DECISION = (
    "retain_rigid_target_adaptation_and_reject_v152a_source_claim"
)
PRIOR_STRENGTHS = (0.005, 0.01, 0.025, 0.05, 0.1)
META_NONDEGRADING_CITY_COUNT = 5
META_MINIMUM_INTERVENTION_GROUPS = 6
PRECISION_MINIMUM = 0.25
PRECISION_MAXIMUM = 2.0


@dataclass(frozen=True)
class _RobustPrior:
    center: np.ndarray
    precision: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _PriorPrediction:
    contrast: Any
    actual: np.ndarray
    groups: np.ndarray
    policy_rows: np.ndarray | tuple[np.ndarray, ...]
    references: np.ndarray
    rigid_score: np.ndarray
    source_scores: Mapping[str, np.ndarray]
    placebo_scores: Mapping[str, np.ndarray]
    zero_scores: Mapping[str, np.ndarray]
    source_interventions: Mapping[str, int]
    source_prior: _RobustPrior
    placebo_prior: _RobustPrior


@dataclass(frozen=True)
class _MetaFold:
    city: str
    rigid_values: np.ndarray
    source_values: Mapping[str, np.ndarray]
    placebo_values: Mapping[str, np.ndarray]
    zero_values: Mapping[str, np.ndarray]
    source_interventions: Mapping[str, int]
    source_cities: tuple[str, ...]
    prior_diagnostics: Mapping[str, Any]


def _candidate_key(block: str, strength: float) -> str:
    return f"{block}|lambda={float(strength):g}"


def _split_candidate_key(key: str) -> tuple[str, float]:
    block, value = str(key).split("|lambda=", 1)
    strength = float(value)
    if block not in MECHANISM_ACTION_FEATURES or strength not in PRIOR_STRENGTHS:
        raise ValueError(f"invalid hierarchical-prior candidate: {key}")
    return block, strength


def _robust_prior(
    statistics: Mapping[str, _SourceStatistics],
    scale: np.ndarray,
    block_indices: Mapping[str, np.ndarray],
    *,
    placebo: bool,
) -> _RobustPrior:
    if len(statistics) < 2:
        raise ValueError("robust prior requires at least two source cities")
    coefficients = []
    for city in sorted(statistics):
        row = statistics[city]
        right = row.placebo_right if placebo else row.right
        coefficients.append(
            _ridge_from_unscaled_statistics(row.normal, right, scale)
        )
    matrix = np.vstack(coefficients)
    center = np.median(matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    precision = np.ones(center.shape, dtype=float)
    block_diagnostics: dict[str, Any] = {}
    for block, raw_indices in block_indices.items():
        indices = np.asarray(raw_indices, dtype=int)
        block_mad = mad[indices]
        positive = block_mad[block_mad > 1e-8]
        reference = float(np.median(positive)) if positive.size else 1.0
        reliability = 1.0 / (1.0 + np.square(block_mad / reference))
        reliability /= max(float(np.mean(reliability)), 1e-12)
        reliability = np.clip(
            reliability,
            PRECISION_MINIMUM,
            PRECISION_MAXIMUM,
        )
        precision[indices] = reliability
        block_diagnostics[block] = {
            "coefficient_count": int(indices.size),
            "median_abs_center": float(np.median(np.abs(center[indices]))),
            "median_intercity_mad": float(np.median(block_mad)),
            "precision_min": float(np.min(reliability)),
            "precision_max": float(np.max(reliability)),
        }
    if (
        not np.all(np.isfinite(center))
        or not np.all(np.isfinite(precision))
        or np.any(precision <= 0.0)
    ):
        raise ValueError("robust mechanism prior is non-finite")
    return _RobustPrior(
        center=center,
        precision=precision,
        diagnostics={
            "protocol": PRIOR_PROTOCOL,
            "source_city_count": len(statistics),
            "source_cities": sorted(statistics),
            "placebo": bool(placebo),
            "coefficient_width": int(center.size),
            "block_diagnostics": block_diagnostics,
        },
    )


def _target_statistics(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    weights = group_balanced_weights(groups)
    return (
        features.T @ (weights[:, None] * features),
        features.T @ (weights * target),
    )


def _fit_from_statistics(
    normal: np.ndarray,
    right: np.ndarray,
    *,
    prior: _RobustPrior,
    block_indices: Sequence[int],
    strength: float,
) -> np.ndarray:
    matrix = np.asarray(normal, dtype=float).copy()
    vector = np.asarray(right, dtype=float).copy()
    width = int(vector.size)
    if matrix.shape != (width, width):
        raise ValueError("target prior statistics have incompatible shapes")
    matrix.flat[:: width + 1] += RIDGE_L2
    indices = np.asarray(tuple(int(value) for value in block_indices), dtype=int)
    value = float(strength)
    if value < 0.0 or indices.size < 1:
        raise ValueError("hierarchical prior strength or block is invalid")
    if value > 0.0:
        diagonal = value * prior.precision[indices]
        matrix[indices, indices] += diagonal
        vector[indices] += diagonal * prior.center[indices]
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def _choice_indices(
    score: np.ndarray,
    policy_rows: np.ndarray | tuple[np.ndarray, ...],
) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    if isinstance(policy_rows, np.ndarray):
        rows = np.asarray(policy_rows, dtype=int)
        return rows[np.arange(rows.shape[0]), np.argmin(values[rows], axis=1)]
    return np.asarray(
        [int(rows[int(np.argmin(values[rows]))]) for rows in policy_rows],
        dtype=int,
    )


def _fit_prior_split(
    problem: _PreparedProblem,
    *,
    training_groups: Sequence[str],
    evaluation_groups: Sequence[str],
) -> _PriorPrediction:
    training_ids = tuple(sorted(str(value) for value in training_groups))
    evaluation_ids = tuple(sorted(str(value) for value in evaluation_groups))
    if not training_ids or not evaluation_ids or set(training_ids) & set(
        evaluation_ids
    ):
        raise ValueError("hierarchical-prior split is empty or overlapping")
    training = _group_subset_v3(
        problem.target_full,
        selected_groups=set(training_ids),
        metadata_updates={"target_data_role": "v152a_mechanism_adaptation"},
    )
    evaluation = _group_subset_v3(
        problem.target_full,
        selected_groups=set(evaluation_ids),
        metadata_updates={"target_data_role": "v152a_evaluation_only"},
    )
    training_contrast = build_action_contrast_dataset(
        training,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    training_actual, _, _, _, _ = _normalized_pressure_targets(
        training,
        training_contrast,
    )
    training_design = mechanism_parameter_design(training_contrast)
    evaluation_contrast = build_action_contrast_dataset(
        evaluation,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    evaluation_actual, _, _, evaluation_references, _ = (
        _normalized_pressure_targets(evaluation, evaluation_contrast)
    )
    evaluation_design = mechanism_parameter_design(evaluation_contrast)
    if (
        training_design.feature_names != problem.mechanism_feature_names
        or evaluation_design.feature_names != problem.mechanism_feature_names
    ):
        raise ValueError("hierarchical-prior mechanism schema changed")

    scale = _feature_scale(problem.source_statistics, training_design.values)
    x_training = np.asarray(training_design.values, dtype=float) / scale
    x_evaluation = np.asarray(evaluation_design.values, dtype=float) / scale
    training_action_groups = np.asarray(
        action_group_ids(training_contrast),
        dtype=str,
    )
    normal, right = _target_statistics(
        x_training,
        training_actual,
        training_action_groups,
    )
    null_prior = _RobustPrior(
        center=np.zeros(x_training.shape[1], dtype=float),
        precision=np.ones(x_training.shape[1], dtype=float),
        diagnostics={"protocol": "source-null"},
    )
    beta_target = _fit_from_statistics(
        normal,
        right,
        prior=null_prior,
        block_indices=training_design.block_indices["queue_service"],
        strength=0.0,
    )
    source_prior = _robust_prior(
        problem.source_statistics,
        scale,
        training_design.block_indices,
        placebo=False,
    )
    placebo_prior = _robust_prior(
        problem.source_statistics,
        scale,
        training_design.block_indices,
        placebo=True,
    )
    zero_prior = _RobustPrior(
        center=np.zeros_like(source_prior.center),
        precision=source_prior.precision.copy(),
        diagnostics={
            "protocol": "source-derived-precision-zero-centred-control-v1",
            "source_city_count": len(problem.source_statistics),
        },
    )
    rigid_model = _target_model(
        problem.target_full,
        group_ids=training_ids,
        candidate=problem.frozen_candidate,
        target_domain=problem.city,
    )
    rigid_score = _rigid_score(rigid_model, evaluation_contrast)
    target_mechanism_score = x_evaluation @ beta_target
    groups = np.asarray(action_group_ids(evaluation_contrast), dtype=str)
    _, raw_policy_rows = _group_rows(groups)
    row_widths = {int(rows.size) for rows in raw_policy_rows}
    policy_rows: np.ndarray | tuple[np.ndarray, ...]
    if len(row_widths) == 1:
        policy_rows = np.vstack(raw_policy_rows).astype(int, copy=False)
    else:
        policy_rows = raw_policy_rows
    rigid_choices = _choice_indices(rigid_score, policy_rows)

    source_scores: dict[str, np.ndarray] = {}
    placebo_scores: dict[str, np.ndarray] = {}
    zero_scores: dict[str, np.ndarray] = {}
    interventions: dict[str, int] = {}
    for block, indices in training_design.block_indices.items():
        for strength in PRIOR_STRENGTHS:
            key = _candidate_key(block, strength)
            source_beta = _fit_from_statistics(
                normal,
                right,
                prior=source_prior,
                block_indices=indices,
                strength=strength,
            )
            placebo_beta = _fit_from_statistics(
                normal,
                right,
                prior=placebo_prior,
                block_indices=indices,
                strength=strength,
            )
            zero_beta = _fit_from_statistics(
                normal,
                right,
                prior=zero_prior,
                block_indices=indices,
                strength=strength,
            )
            source_scores[key] = residual_corrected_rigid_score(
                rigid_score,
                x_evaluation @ source_beta,
                target_mechanism_score,
                evaluation_references,
            )
            placebo_scores[key] = residual_corrected_rigid_score(
                rigid_score,
                x_evaluation @ placebo_beta,
                target_mechanism_score,
                evaluation_references,
            )
            zero_scores[key] = residual_corrected_rigid_score(
                rigid_score,
                x_evaluation @ zero_beta,
                target_mechanism_score,
                evaluation_references,
            )
            source_choices = _choice_indices(source_scores[key], policy_rows)
            interventions[key] = int(
                np.count_nonzero(source_choices != rigid_choices)
            )

    return _PriorPrediction(
        contrast=evaluation_contrast,
        actual=np.asarray(evaluation_actual, dtype=float),
        groups=groups,
        policy_rows=policy_rows,
        references=np.asarray(evaluation_references, dtype=bool),
        rigid_score=np.asarray(rigid_score, dtype=float),
        source_scores=source_scores,
        placebo_scores=placebo_scores,
        zero_scores=zero_scores,
        source_interventions=interventions,
        source_prior=source_prior,
        placebo_prior=placebo_prior,
    )


def _policy_values_for(prediction: _PriorPrediction, score: np.ndarray) -> np.ndarray:
    values, _ = _policy_arrays(
        prediction.actual,
        np.asarray(score, dtype=float),
        prediction.policy_rows,
        prediction.references,
    )
    return values


def _meta_fold(problem: _PreparedProblem) -> _MetaFold:
    prediction = _fit_prior_split(
        problem,
        training_groups=problem.selected_groups,
        evaluation_groups=problem.evaluation_groups,
    )
    return _MetaFold(
        city=problem.city,
        rigid_values=_policy_values_for(prediction, prediction.rigid_score),
        source_values={
            key: _policy_values_for(prediction, score)
            for key, score in prediction.source_scores.items()
        },
        placebo_values={
            key: _policy_values_for(prediction, score)
            for key, score in prediction.placebo_scores.items()
        },
        zero_values={
            key: _policy_values_for(prediction, score)
            for key, score in prediction.zero_scores.items()
        },
        source_interventions=prediction.source_interventions,
        source_cities=problem.source_order,
        prior_diagnostics=prediction.source_prior.diagnostics,
    )


def _select_meta_candidate(folds: Sequence[_MetaFold]) -> dict[str, Any]:
    rows = tuple(folds)
    if len(rows) != len(EXPECTED_CITY_GROUPS) - 1:
        raise ValueError("V152A requires six pseudo-target folds")
    keys = set(rows[0].source_values)
    if not keys or any(
        set(row.source_values) != keys
        or set(row.placebo_values) != keys
        or set(row.zero_values) != keys
        for row in rows
    ):
        raise ValueError("V152A meta candidates differ across cities")

    diagnostics: dict[str, Any] = {}
    eligible: list[str] = []
    for key in sorted(keys):
        gains_rigid = np.asarray(
            [
                np.mean(row.rigid_values) - np.mean(row.source_values[key])
                for row in rows
            ],
            dtype=float,
        )
        gains_placebo = np.asarray(
            [
                np.mean(row.placebo_values[key])
                - np.mean(row.source_values[key])
                for row in rows
            ],
            dtype=float,
        )
        gains_zero = np.asarray(
            [
                np.mean(row.zero_values[key]) - np.mean(row.source_values[key])
                for row in rows
            ],
            dtype=float,
        )
        interventions = int(
            sum(row.source_interventions[key] for row in rows)
        )
        checks = {
            "mean_gain_vs_rigid": float(np.mean(gains_rigid))
            >= MINIMUM_OOF_GAIN,
            "mean_gain_vs_placebo": float(np.mean(gains_placebo))
            >= MINIMUM_OOF_GAIN,
            "mean_gain_vs_zero_prior": float(np.mean(gains_zero))
            >= MINIMUM_OOF_GAIN,
            "nondegrading_vs_rigid": int(np.count_nonzero(gains_rigid >= 0.0))
            >= META_NONDEGRADING_CITY_COUNT,
            "nondegrading_vs_placebo": int(
                np.count_nonzero(gains_placebo >= 0.0)
            )
            >= META_NONDEGRADING_CITY_COUNT,
            "nondegrading_vs_zero_prior": int(
                np.count_nonzero(gains_zero >= 0.0)
            )
            >= META_NONDEGRADING_CITY_COUNT,
            "minimum_interventions": interventions
            >= META_MINIMUM_INTERVENTION_GROUPS,
        }
        means = {
            "gain_vs_rigid": float(np.mean(gains_rigid)),
            "gain_vs_matched_placebo": float(np.mean(gains_placebo)),
            "gain_vs_zero_centred_prior": float(np.mean(gains_zero)),
        }
        diagnostics[key] = {
            "means": means,
            "minimum_comparator_gain": min(means.values()),
            "nondegrading_city_count_vs_rigid": int(
                np.count_nonzero(gains_rigid >= 0.0)
            ),
            "nondegrading_city_count_vs_matched_placebo": int(
                np.count_nonzero(gains_placebo >= 0.0)
            ),
            "nondegrading_city_count_vs_zero_centred_prior": int(
                np.count_nonzero(gains_zero >= 0.0)
            ),
            "intervention_group_count": interventions,
            "city_gains_vs_rigid": {
                row.city: float(value)
                for row, value in zip(rows, gains_rigid, strict=True)
            },
            "checks": checks,
            "eligible": all(checks.values()),
        }
        if all(checks.values()):
            eligible.append(key)
    pool = eligible or list(diagnostics)
    selected = max(
        pool,
        key=lambda key: (
            diagnostics[key]["minimum_comparator_gain"],
            diagnostics[key]["means"]["gain_vs_rigid"],
            key,
        ),
    )
    block, strength = _split_candidate_key(selected)
    return {
        "admitted": bool(selected in eligible),
        "candidate": selected,
        "mechanism_block": block,
        "prior_strength": strength,
        "eligible_candidate_count": len(eligible),
        "selection_rule": "maximize_worst_comparator_mean_gain",
        "pseudo_target_cities": [row.city for row in rows],
        "source_cities_by_pseudo_target": {
            row.city: list(row.source_cities) for row in rows
        },
        "selected_diagnostics": diagnostics[selected],
        "candidate_diagnostics": diagnostics,
    }


def _effect_summary(
    by_city: Mapping[str, Mapping[str, Any]],
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    effects, city_means = _paired_effects(by_city, candidate, reference)
    return {
        "seed_unit": _paired_bootstrap(effects),
        "city_unit": {**_city_bootstrap(city_means), "city_means": city_means},
    }


def run_target(
    *,
    target_city: str,
    authorization_result_path: Path,
    expected_authorization_sha256: str,
    v151a_rejection_path: Path,
    expected_v151a_rejection_sha256: str,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    expected_source_cache_audit_sha256: str,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int = 6,
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V152A target city: {city}")
    authorization = _validate_authorization(
        authorization_result_path,
        expected_authorization_sha256,
    )
    if _sha256(v151a_rejection_path) != expected_v151a_rejection_sha256:
        raise ValueError("V151A rejection identity changed")
    rejection = _read_json(v151a_rejection_path)
    if (
        rejection.get("protocol") != V151A_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != V151A_REJECT_DECISION
    ):
        raise ValueError("V152A requires the frozen V151A rejection")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V152A source cache audit identity changed")
    fit_result = _fit_result_contract(
        fit_result_path,
        expected_sha256=expected_fit_result_sha256,
    )
    inventory = _prepare_inventory(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
    )
    inventory_seconds = time.monotonic() - started
    meta_cities = tuple(
        value for value in EXPECTED_CITY_GROUPS if value != city
    )
    stage_started = time.monotonic()
    source_statistics_inventory = _prepare_source_statistics_inventory(
        inventory,
        fit_workers=fit_workers,
        stable_placebo_city_identity=True,
        cities=meta_cities,
    )
    source_statistics_seconds = time.monotonic() - stage_started

    def build_meta(pseudo_target: str) -> _MetaFold:
        problem = _prepare_problem(
            city=pseudo_target,
            fit_result=fit_result,
            fit_protocol_path=fit_protocol_path,
            source_cache_root=source_cache_root,
            source_manifest_path=source_manifest_path,
            source_cache_audit_path=source_cache_audit_path,
            conversion_root=conversion_root,
            cache_workers=cache_workers,
            fit_workers=1,
            excluded_source_cities=(city,),
            inventory=inventory,
            stable_placebo_city_identity=True,
            source_statistics_inventory=source_statistics_inventory,
        )
        fold = _meta_fold(problem)
        print(
            "V152A_PROGRESS "
            f"target={city} pseudo_target={pseudo_target} "
            f"source_count={len(fold.source_cities)}",
            flush=True,
        )
        return fold

    stage_started = time.monotonic()
    maximum_workers = min(max(int(fit_workers), 1), len(meta_cities))
    if maximum_workers == 1:
        meta_folds = tuple(build_meta(value) for value in meta_cities)
    else:
        with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
            meta_folds = tuple(pool.map(build_meta, meta_cities))
    meta_seconds = time.monotonic() - stage_started
    selector = _select_meta_candidate(meta_folds)

    stage_started = time.monotonic()
    target_problem = _prepare_problem(
        city=city,
        fit_result=fit_result,
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
        fit_workers=fit_workers,
        inventory=inventory,
        stable_placebo_city_identity=True,
        source_statistics_inventory=source_statistics_inventory,
    )
    prediction = _fit_prior_split(
        target_problem,
        training_groups=target_problem.selected_groups,
        evaluation_groups=target_problem.evaluation_groups,
    )
    target_seconds = time.monotonic() - stage_started
    key = str(selector["candidate"])
    admitted = bool(selector["admitted"])
    fallback = prediction.rigid_score.copy()
    forced_source = prediction.source_scores[key]
    forced_placebo = prediction.placebo_scores[key]
    forced_zero = prediction.zero_scores[key]
    arms = {
        "rigid_target_only": prediction.rigid_score,
        "source_null_exact_rigid": fallback,
        "forced_hierarchical_source_prior": forced_source,
        "selected_hierarchical_source_prior": (
            forced_source if admitted else fallback.copy()
        ),
        "forced_matched_placebo_prior": forced_placebo,
        "selected_matched_placebo_prior": (
            forced_placebo if admitted else fallback.copy()
        ),
        "forced_zero_centred_prior": forced_zero,
        "selected_zero_centred_prior": (
            forced_zero if admitted else fallback.copy()
        ),
    }
    evaluation = _group_subset_v3(
        target_problem.target_full,
        selected_groups=set(target_problem.evaluation_groups),
        metadata_updates={"target_data_role": "v152a_evaluation_only"},
    )
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=prediction.contrast,
            actual=prediction.actual,
            score=score,
            scenarios=target_problem.scenarios,
        )
        for name, score in arms.items()
    }
    if arm_summaries["rigid_target_only"] != arm_summaries[
        "source_null_exact_rigid"
    ]:
        raise ValueError("V152A source-null differs from rigid CFCMT")
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "hierarchical-prior-development-target",
        "target_city": city,
        "target_budget": TARGET_BUDGET,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "prior_protocol": PRIOR_PROTOCOL,
            "prior_strength_grid": list(PRIOR_STRENGTHS),
            "minimum_meta_gain": MINIMUM_OOF_GAIN,
            "minimum_nondegrading_meta_cities": META_NONDEGRADING_CITY_COUNT,
            "minimum_meta_intervention_groups": (
                META_MINIMUM_INTERVENTION_GROUPS
            ),
            "selector": selector,
            "target_source_prior": prediction.source_prior.diagnostics,
            "target_placebo_prior": prediction.placebo_prior.diagnostics,
            "target_forced_intervention_group_count": int(
                prediction.source_interventions[key]
            ),
        },
        "information_budget": {
            "target_mechanism_adaptation_group_count": len(
                target_problem.selected_groups
            ),
            "target_meta_selection_label_count": 0,
            "target_evaluation_group_count": len(
                target_problem.evaluation_groups
            ),
            "target_excluded_from_meta_targets": True,
            "target_excluded_from_meta_source_pools": True,
            "stable_placebo_seed_by_source_city_identity": True,
        },
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "source_disabled_returns_rigid_score_bitwise": True,
            "summaries_equal": True,
        },
        "inputs": {
            "authorization_protocol": authorization["protocol"],
            "authorization_result_sha256": expected_authorization_sha256,
            "v151a_rejection_sha256": expected_v151a_rejection_sha256,
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "runtime_breakdown_seconds": {
            "shared_inventory": float(inventory_seconds),
            "shared_source_statistics": float(source_statistics_seconds),
            "parallel_meta_folds": float(meta_seconds),
            "target_fit": float(target_seconds),
        },
        "runtime_seconds": float(time.monotonic() - started),
        "claim_boundary": (
            "V152A is a leave-target-city-out B100 development test of a robust "
            "pooled source mechanism prior. It is not closed-loop or untouched-"
            "city confirmation."
        ),
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
        or any(int(row.get("target_budget", -1)) != TARGET_BUDGET for row in rows)
        or any(
            row.get("method", {}).get("prior_protocol") != PRIOR_PROTOCOL
            for row in rows
        )
        or any(
            row.get("information_budget", {}).get(
                "target_meta_selection_label_count"
            )
            != 0
            for row in rows
        )
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V152A aggregate requires seven valid target results")
    comparisons = {
        "selected_effect_vs_rigid_target_only": _effect_summary(
            by_city,
            "selected_hierarchical_source_prior",
            "rigid_target_only",
        ),
        "selected_effect_vs_matched_placebo": _effect_summary(
            by_city,
            "selected_hierarchical_source_prior",
            "selected_matched_placebo_prior",
        ),
        "selected_effect_vs_zero_centred_prior": _effect_summary(
            by_city,
            "selected_hierarchical_source_prior",
            "selected_zero_centred_prior",
        ),
        "forced_effect_vs_rigid_target_only": _effect_summary(
            by_city,
            "forced_hierarchical_source_prior",
            "rigid_target_only",
        ),
    }
    admitted = {
        city: bool(by_city[city]["method"]["selector"]["admitted"])
        for city in EXPECTED_CITY_GROUPS
    }
    admitted_cities = [city for city, value in admitted.items() if value]
    rigid_means = comparisons["selected_effect_vs_rigid_target_only"][
        "city_unit"
    ]["city_means"]
    placebo_means = comparisons["selected_effect_vs_matched_placebo"][
        "city_unit"
    ]["city_means"]
    zero_means = comparisons["selected_effect_vs_zero_centred_prior"][
        "city_unit"
    ]["city_means"]
    tolerance = 1e-12
    checks = {
        "mean_selected_effect_improves_rigid": comparisons[
            "selected_effect_vs_rigid_target_only"
        ]["city_unit"]["mean"]
        < 0.0,
        "mean_selected_effect_beats_matched_placebo": comparisons[
            "selected_effect_vs_matched_placebo"
        ]["city_unit"]["mean"]
        < 0.0,
        "mean_selected_effect_beats_zero_centred_prior": comparisons[
            "selected_effect_vs_zero_centred_prior"
        ]["city_unit"]["mean"]
        < 0.0,
        "no_city_regresses_rigid": max(rigid_means.values()) <= tolerance,
        "at_least_two_targets_admit_hierarchical_prior": len(admitted_cities)
        >= 2,
        "every_admitted_target_improves_rigid": all(
            rigid_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_target_beats_matched_placebo": all(
            placebo_means[city] < 0.0 for city in admitted_cities
        ),
        "every_admitted_target_beats_zero_centred_prior": all(
            zero_means[city] < 0.0 for city in admitted_cities
        ),
        "every_rejected_target_exactly_falls_back": all(
            abs(rigid_means[city]) <= tolerance
            for city in EXPECTED_CITY_GROUPS
            if not admitted[city]
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(rows),
        "target_budget": TARGET_BUDGET,
        "prior_protocol": PRIOR_PROTOCOL,
        "prior_strength_grid": list(PRIOR_STRENGTHS),
        "source_admission_count": len(admitted_cities),
        "admitted_cities": admitted,
        **comparisons,
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "development_gate": {
            "inference_unit": "city",
            "checks": checks,
            "passed": passed,
            "decision": (
                "authorize_v152a_hierarchical_prior_closed_loop_development"
                if passed
                else V152A_REJECT_DECISION
            ),
        },
        "claim_boundary": (
            "V152A tests a pooled mechanism prior under leave-target-city-out "
            "development. Passing cannot establish closed-loop or untouched-"
            "city efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--v151a-rejection", type=Path, required=True)
    target.add_argument("--v151a-rejection-sha256", required=True)
    target.add_argument("--fit-result", type=Path, required=True)
    target.add_argument("--fit-result-sha256", required=True)
    target.add_argument("--fit-protocol", type=Path, required=True)
    target.add_argument("--source-cache-root", type=Path, required=True)
    target.add_argument("--source-manifest", type=Path, required=True)
    target.add_argument("--source-cache-audit", type=Path, required=True)
    target.add_argument("--source-cache-audit-sha256", required=True)
    target.add_argument("--conversion-root", type=Path, required=True)
    target.add_argument("--cache-workers", type=int, default=8)
    target.add_argument("--fit-workers", type=int, default=6)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            v151a_rejection_path=args.v151a_rejection,
            expected_v151a_rejection_sha256=args.v151a_rejection_sha256,
            fit_result_path=args.fit_result,
            expected_fit_result_sha256=args.fit_result_sha256,
            fit_protocol_path=args.fit_protocol,
            source_cache_root=args.source_cache_root,
            source_manifest_path=args.source_manifest,
            source_cache_audit_path=args.source_cache_audit,
            expected_source_cache_audit_sha256=(
                args.source_cache_audit_sha256
            ),
            conversion_root=args.conversion_root,
            cache_workers=args.cache_workers,
            fit_workers=args.fit_workers,
        )
        status = "DONE"
    else:
        result = aggregate_results([_read_json(path) for path in args.results])
        status = "PASS" if result["development_gate"]["passed"] else "REJECT"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": status,
                "protocol": result["protocol"],
                "target_city": result.get("target_city"),
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
