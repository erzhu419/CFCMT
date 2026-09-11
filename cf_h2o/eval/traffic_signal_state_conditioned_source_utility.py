"""Nested state-conditioned utility selection for mechanism source priors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_bytes,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    CONTRAST_FEATURES,
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MINIMUM_OOF_GAIN,
    PLACEBO_SEED,
    RIDGE_L2,
    _arm_summary,
    _candidate_key,
    _fit_with_prior,
    _fold_groups,
    _policy_values,
    _read_json,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    _reference_policy_score,
    select_city_budget_groups,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_right_of_way_signature import _read_manifest
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
    _rigid_score,
    _validate_authorization,
    residual_corrected_rigid_score,
)
from cf_h2o.eval.traffic_signal_sample_coherent_source_prior import (
    AGGREGATE_PROTOCOL as V150I_AGGREGATE_PROTOCOL,
    prior_strength_for_budget,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    COMMON_EVALUATION_RESERVE_BUDGET,
    PRIMARY_BUDGET,
    _budget_summary,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _source_inputs,
    _target_model,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _normalized_pressure_targets,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    MECHANISM_ACTION_FEATURES,
    group_balanced_weights,
    mechanism_parameter_design,
    permute_group_targets,
    ridge_coefficients,
)
from cf_h2o.traffic_signal.right_of_way_context import (
    augment_dataset_with_right_of_way_context,
    net_file_from_sumocfg,
    read_network_right_of_way_context,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
    UtilityGateConfig,
    UtilityRecords,
    apply_utility_gate,
    build_utility_records,
    concatenate_utility_records,
    fit_utility_gate,
    pressure_align_candidate_scores,
    pressure_pairwise_candidate_scores,
)


RESULT_PROTOCOL = "tsc-v150k-state-conditioned-source-utility-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150k-state-conditioned-source-utility-aggregate-v1"
METHOD_PROTOCOL = "nested-state-conditioned-source-mechanism-utility-v1"
UTILITY_GATE_PROTOCOL = "ridge-upper-cost-bound-over-source-mechanism-proposals-v1"
RUNTIME_MODEL_PROTOCOL = "tsc-v150k-state-conditioned-source-runtime-model-v2"
TARGET_BUDGET = PRIMARY_BUDGET
SOURCE_PRIOR_STRENGTH = prior_strength_for_budget(TARGET_BUDGET)
MINIMUM_NONDEGRADING_FOLDS = 4
MINIMUM_INTERVENTION_GROUPS = 5
GATE_CONFIG = UtilityGateConfig()
V150J_AGGREGATE_PROTOCOL = "tsc-v150j-conditional-mechanism-prior-aggregate-v1"


@dataclass(frozen=True)
class _SourceStatistics:
    second_moment: np.ndarray
    normal: np.ndarray
    right: np.ndarray
    placebo_right: np.ndarray


@dataclass(frozen=True)
class _PreparedInventory:
    city_datasets: Mapping[str, Any]
    scenarios_by_group: Mapping[str, Sequence[str]]
    target_name: str
    source_bank_audit: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedSourceStatisticsInventory:
    statistics: Mapping[str, _SourceStatistics]
    mechanism_feature_names: tuple[str, ...]
    mechanism_block_indices: Mapping[str, np.ndarray]
    stable_placebo_city_identity: bool


@dataclass(frozen=True)
class _PreparedProblem:
    city: str
    target_full: Any
    selected_groups: tuple[str, ...]
    reserve_groups: tuple[str, ...]
    evaluation_groups: tuple[str, ...]
    selection_audit: Mapping[str, Any]
    reserve_audit: Mapping[str, Any]
    scenarios: tuple[str, ...]
    source_order: tuple[str, ...]
    source_statistics: Mapping[str, _SourceStatistics]
    mechanism_feature_names: tuple[str, ...]
    mechanism_block_indices: Mapping[str, np.ndarray]
    frozen_candidate: str
    target_name: str
    source_bank_audit: Mapping[str, Any]


@dataclass(frozen=True)
class _SplitPrediction:
    contrast: Any
    actual: np.ndarray
    groups: np.ndarray
    references: np.ndarray
    rigid_score: np.ndarray
    candidate_scores: Mapping[str, np.ndarray]
    placebo_scores: Mapping[str, np.ndarray]
    rigid_model: Any
    feature_scale: np.ndarray
    target_beta: np.ndarray
    candidate_betas: Mapping[str, np.ndarray]
    placebo_betas: Mapping[str, np.ndarray]
    training_group_count: int
    evaluation_group_count: int


def _ridge_statistics(
    x: np.ndarray, y: np.ndarray, groups: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    weights = group_balanced_weights(groups)
    return (
        values.T @ (weights[:, None] * values),
        values.T @ (weights * target),
    )


def _ridge_from_unscaled_statistics(
    normal: np.ndarray,
    right: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    feature_scale = np.asarray(scale, dtype=float)
    matrix = np.asarray(normal, dtype=float) / (
        feature_scale[:, None] * feature_scale[None, :]
    )
    vector = np.asarray(right, dtype=float) / feature_scale
    matrix = matrix.copy()
    matrix.flat[:: matrix.shape[0] + 1] += RIDGE_L2
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def _feature_scale(
    source_statistics: Mapping[str, _SourceStatistics],
    target_design: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_design, dtype=float)
    moments = [
        np.asarray(source_statistics[city].second_moment, dtype=float)
        for city in sorted(source_statistics)
    ]
    moments.append(np.mean(target * target, axis=0))
    scale = np.sqrt(np.maximum(np.mean(np.vstack(moments), axis=0), 1e-12))
    return np.where(scale > 1e-6, scale, 1.0)


def _source_statistics_for_city(
    source: str,
    source_index: int,
    part: Mapping[str, Any],
) -> tuple[str, _SourceStatistics]:
    design = np.asarray(part["design"].values, dtype=float)
    actual = np.asarray(part["actual"], dtype=float)
    groups = np.asarray(action_group_ids(part["contrast"]), dtype=str)
    references = np.asarray(part["references"], dtype=bool)
    placebo = permute_group_targets(
        actual,
        groups,
        seed=PLACEBO_SEED + int(source_index),
        references=references,
    )
    normal, right = _ridge_statistics(design, actual, groups)
    _, placebo_right = _ridge_statistics(design, placebo, groups)
    return source, _SourceStatistics(
        second_moment=np.mean(design * design, axis=0),
        normal=normal,
        right=right,
        placebo_right=placebo_right,
    )


def _prepare_inventory(
    *,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    conversion_root: Path,
    cache_workers: int,
) -> _PreparedInventory:
    """Load and augment the immutable seven-city bank once per process."""

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    protocol, bank, scenarios_by_group, bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    if tuple(sorted(scenarios_by_group)) != EXPECTED_CITY_GROUPS:
        raise ValueError("V150K seven-city inventory changed")
    _, sumocfg_by_scenario = _read_manifest(source_manifest_path)
    contexts = {
        scenario: read_network_right_of_way_context(
            net_file_from_sumocfg(sumocfg_by_scenario[scenario])
        )
        for scenario in bank
    }
    augmented_bank = {
        scenario: augment_dataset_with_right_of_way_context(
            dataset, {scenario: contexts[scenario]}
        )
        for scenario, dataset in bank.items()
    }
    city_datasets = {
        group: _merge_city_datasets(
            augmented_bank, scenarios_by_group[group], city=group
        )
        for group in EXPECTED_CITY_GROUPS
    }
    return _PreparedInventory(
        city_datasets=city_datasets,
        scenarios_by_group={
            str(group): tuple(str(value) for value in scenarios)
            for group, scenarios in scenarios_by_group.items()
        },
        target_name=str(protocol["target_name"]),
        source_bank_audit=bank_audit,
    )


def _prepare_source_statistics_inventory(
    inventory: _PreparedInventory,
    *,
    fit_workers: int,
    stable_placebo_city_identity: bool,
    cities: Sequence[str] = EXPECTED_CITY_GROUPS,
) -> _PreparedSourceStatisticsInventory:
    """Build each city's reusable unscaled mechanism statistics once."""

    def build(item: tuple[int, str]) -> tuple[
        str,
        _SourceStatistics,
        tuple[str, ...],
        Mapping[str, np.ndarray],
    ]:
        source_index, source = item
        absolute = inventory.city_datasets[source]
        contrast = build_action_contrast_dataset(
            absolute,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES,
        )
        actual, _, _, references, _ = _normalized_pressure_targets(
            absolute,
            contrast,
        )
        design = mechanism_parameter_design(contrast)
        _, statistics = _source_statistics_for_city(
            source,
            source_index,
            {
                "contrast": contrast,
                "actual": actual,
                "references": references,
                "design": design,
            },
        )
        return source, statistics, design.feature_names, design.block_indices

    selected_cities = tuple(str(value) for value in cities)
    if (
        not selected_cities
        or len(set(selected_cities)) != len(selected_cities)
        or any(value not in EXPECTED_CITY_GROUPS for value in selected_cities)
    ):
        raise ValueError("shared source city inventory is invalid")
    items = tuple(
        (
            EXPECTED_CITY_GROUPS.index(source)
            if stable_placebo_city_identity
            else source_index,
            source,
        )
        for source_index, source in enumerate(selected_cities)
    )
    maximum_workers = min(max(int(fit_workers), 1), len(items))
    if maximum_workers == 1:
        rows = tuple(build(item) for item in items)
    else:
        with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
            rows = tuple(pool.map(build, items))
    feature_names = rows[0][2]
    block_indices = rows[0][3]
    if any(row[2] != feature_names for row in rows):
        raise ValueError("shared source mechanism schemas differ")
    for _, _, _, candidate_blocks in rows[1:]:
        if set(candidate_blocks) != set(block_indices) or any(
            not np.array_equal(candidate_blocks[key], block_indices[key])
            for key in block_indices
        ):
            raise ValueError("shared source mechanism block indices differ")
    return _PreparedSourceStatisticsInventory(
        statistics={source: statistics for source, statistics, _, _ in rows},
        mechanism_feature_names=feature_names,
        mechanism_block_indices=block_indices,
        stable_placebo_city_identity=bool(stable_placebo_city_identity),
    )


def _prepare_problem(
    *,
    city: str,
    fit_result: Mapping[str, Any],
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int,
    excluded_source_cities: Sequence[str] = (),
    inventory: _PreparedInventory | None = None,
    stable_placebo_city_identity: bool = False,
    source_statistics_inventory: _PreparedSourceStatisticsInventory | None = None,
) -> _PreparedProblem:
    prepared = inventory or _prepare_inventory(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
    )
    scenarios_by_group = prepared.scenarios_by_group
    city_datasets = prepared.city_datasets
    target_full = city_datasets[city]
    selected_groups, selection_audit = select_city_budget_groups(
        target_full,
        city=city,
        scenarios=scenarios_by_group[city],
        budget=TARGET_BUDGET,
    )
    reserve_groups, reserve_audit = select_city_budget_groups(
        target_full,
        city=city,
        scenarios=scenarios_by_group[city],
        budget=COMMON_EVALUATION_RESERVE_BUDGET,
    )
    if tuple(selected_groups) != tuple(reserve_groups):
        raise ValueError("V150K B100 selection and reserve differ")
    all_groups = set(str(value) for value in action_group_ids(target_full))
    evaluation_groups = tuple(sorted(all_groups - set(reserve_groups)))
    if not evaluation_groups or set(evaluation_groups) & set(reserve_groups):
        raise ValueError("V150K adaptation/evaluation split failed")

    excluded = tuple(sorted(set(str(value) for value in excluded_source_cities)))
    if city in excluded or any(value not in EXPECTED_CITY_GROUPS for value in excluded):
        raise ValueError("invalid excluded source city set")
    source_order = tuple(
        group
        for group in EXPECTED_CITY_GROUPS
        if group != city and group not in excluded
    )
    if len(source_order) < 2:
        raise ValueError("source mechanism fitting requires at least two cities")
    if source_statistics_inventory is not None:
        if (
            source_statistics_inventory.stable_placebo_city_identity
            != bool(stable_placebo_city_identity)
        ):
            raise ValueError("shared source placebo identity mode changed")
        if any(
            source not in source_statistics_inventory.statistics
            for source in source_order
        ):
            raise ValueError("shared source statistics inventory is incomplete")
        statistics = {
            source: source_statistics_inventory.statistics[source]
            for source in source_order
        }
        feature_names = source_statistics_inventory.mechanism_feature_names
        block_indices = source_statistics_inventory.mechanism_block_indices
    else:
        city_parts: dict[str, dict[str, Any]] = {}
        for source in source_order:
            absolute = city_datasets[source]
            contrast = build_action_contrast_dataset(
                absolute,
                reference_policy="phase_pressure",
                contrast_features=CONTRAST_FEATURES,
            )
            actual, _, _, references, _ = _normalized_pressure_targets(
                absolute, contrast
            )
            city_parts[source] = {
                "contrast": contrast,
                "actual": actual,
                "references": references,
                "design": mechanism_parameter_design(contrast),
            }
        feature_names = city_parts[source_order[0]]["design"].feature_names
        block_indices = city_parts[source_order[0]]["design"].block_indices
        if any(
            part["design"].feature_names != feature_names
            for part in city_parts.values()
        ):
            raise ValueError("V150K source mechanism schemas differ")
        with ThreadPoolExecutor(max_workers=min(max(int(fit_workers), 1), 6)) as pool:
            statistics = dict(
                pool.map(
                    lambda item: _source_statistics_for_city(
                        item[1],
                        (
                            EXPECTED_CITY_GROUPS.index(item[1])
                            if stable_placebo_city_identity
                            else item[0]
                        ),
                        city_parts[item[1]],
                    ),
                    enumerate(source_order),
                )
            )
    return _PreparedProblem(
        city=city,
        target_full=target_full,
        selected_groups=tuple(str(value) for value in selected_groups),
        reserve_groups=tuple(str(value) for value in reserve_groups),
        evaluation_groups=evaluation_groups,
        selection_audit=selection_audit,
        reserve_audit=reserve_audit,
        scenarios=tuple(str(value) for value in scenarios_by_group[city]),
        source_order=source_order,
        source_statistics=statistics,
        mechanism_feature_names=feature_names,
        mechanism_block_indices=block_indices,
        frozen_candidate=str(fit_result["frozen_blend_candidate"]),
        target_name=prepared.target_name,
        source_bank_audit=prepared.source_bank_audit,
    )


def _fit_split(
    problem: _PreparedProblem,
    *,
    training_groups: Sequence[str],
    evaluation_groups: Sequence[str],
    fit_workers: int,
) -> _SplitPrediction:
    training_ids = tuple(sorted(str(value) for value in training_groups))
    evaluation_ids = tuple(sorted(str(value) for value in evaluation_groups))
    if not training_ids or not evaluation_ids or set(training_ids) & set(evaluation_ids):
        raise ValueError("V150K split groups are empty or overlapping")
    training = _group_subset_v3(
        problem.target_full,
        selected_groups=set(training_ids),
        metadata_updates={"target_data_role": "v150k_nested_utility_training"},
    )
    evaluation = _group_subset_v3(
        problem.target_full,
        selected_groups=set(evaluation_ids),
        metadata_updates={"target_data_role": "v150k_nested_utility_evaluation"},
    )
    training_contrast = build_action_contrast_dataset(
        training,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    training_actual, _, _, _, _ = _normalized_pressure_targets(
        training, training_contrast
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
        raise ValueError("V150K target mechanism schema changed")
    scale = _feature_scale(problem.source_statistics, training_design.values)
    x_training = np.asarray(training_design.values, dtype=float) / scale
    x_evaluation = np.asarray(evaluation_design.values, dtype=float) / scale
    training_action_groups = np.asarray(
        action_group_ids(training_contrast), dtype=str
    )
    beta_target = ridge_coefficients(
        x_training,
        training_actual,
        training_action_groups,
        ridge_l2=RIDGE_L2,
    )
    target_mechanism_score = x_evaluation @ beta_target
    rigid_model = _target_model(
        problem.target_full,
        group_ids=training_ids,
        candidate=problem.frozen_candidate,
        target_domain=problem.city,
    )
    rigid_score = _rigid_score(rigid_model, evaluation_contrast)

    def fit_source(
        source: str,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        statistics = problem.source_statistics[source]
        source_beta = _ridge_from_unscaled_statistics(
            statistics.normal, statistics.right, scale
        )
        placebo_beta = _ridge_from_unscaled_statistics(
            statistics.normal, statistics.placebo_right, scale
        )
        source_scores: dict[str, np.ndarray] = {}
        placebo_scores: dict[str, np.ndarray] = {}
        source_betas: dict[str, np.ndarray] = {}
        placebo_betas: dict[str, np.ndarray] = {}
        for block, indices in training_design.block_indices.items():
            key = _candidate_key(source, block)
            beta = _fit_with_prior(
                x_training,
                training_actual,
                training_action_groups,
                source_coefficients=source_beta,
                block_indices=indices,
                prior_strength=SOURCE_PRIOR_STRENGTH,
            )
            source_scores[key] = residual_corrected_rigid_score(
                rigid_score,
                x_evaluation @ beta,
                target_mechanism_score,
                evaluation_references,
            )
            source_betas[key] = np.asarray(beta, dtype=float)
            beta_placebo = _fit_with_prior(
                x_training,
                training_actual,
                training_action_groups,
                source_coefficients=placebo_beta,
                block_indices=indices,
                prior_strength=SOURCE_PRIOR_STRENGTH,
            )
            placebo_scores[key] = residual_corrected_rigid_score(
                rigid_score,
                x_evaluation @ beta_placebo,
                target_mechanism_score,
                evaluation_references,
            )
            placebo_betas[key] = np.asarray(beta_placebo, dtype=float)
        return source_scores, placebo_scores, source_betas, placebo_betas

    candidate_scores: dict[str, np.ndarray] = {}
    placebo_scores: dict[str, np.ndarray] = {}
    candidate_betas: dict[str, np.ndarray] = {}
    placebo_betas: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=min(max(int(fit_workers), 1), 6)) as pool:
        fitted = list(pool.map(fit_source, problem.source_order))
    for source_scores, source_placebos, source_betas, source_placebo_betas in fitted:
        candidate_scores.update(source_scores)
        placebo_scores.update(source_placebos)
        candidate_betas.update(source_betas)
        placebo_betas.update(source_placebo_betas)
    expected_count = len(problem.source_order) * len(MECHANISM_ACTION_FEATURES)
    if len(candidate_scores) != expected_count or set(candidate_scores) != set(
        placebo_scores
    ):
        raise ValueError("V150K candidate pool is incomplete")
    return _SplitPrediction(
        contrast=evaluation_contrast,
        actual=np.asarray(evaluation_actual, dtype=float),
        groups=np.asarray(action_group_ids(evaluation_contrast), dtype=str),
        references=np.asarray(evaluation_references, dtype=bool),
        rigid_score=np.asarray(rigid_score, dtype=float),
        candidate_scores=candidate_scores,
        placebo_scores=placebo_scores,
        rigid_model=rigid_model,
        feature_scale=np.asarray(scale, dtype=float),
        target_beta=np.asarray(beta_target, dtype=float),
        candidate_betas=candidate_betas,
        placebo_betas=placebo_betas,
        training_group_count=len(training_ids),
        evaluation_group_count=len(evaluation_ids),
    )


def _policy_values_for(prediction: _SplitPrediction, score: np.ndarray) -> np.ndarray:
    return _policy_values(
        prediction.actual,
        score,
        prediction.groups,
        prediction.references,
    )


def _candidate_scores_for(
    prediction: _SplitPrediction,
    *,
    placebo: bool,
    candidate_score_constraint: str | None,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any] | None]:
    scores = prediction.placebo_scores if placebo else prediction.candidate_scores
    if candidate_score_constraint is None:
        return scores, None
    if candidate_score_constraint == PRESSURE_ALIGNED_CANDIDATE_PROTOCOL:
        return pressure_align_candidate_scores(
            prediction.contrast,
            prediction.rigid_score,
            scores,
        )
    if candidate_score_constraint == PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL:
        return pressure_pairwise_candidate_scores(
            prediction.contrast,
            prediction.rigid_score,
            scores,
        )
    raise ValueError(
        f"unknown source candidate score constraint: {candidate_score_constraint}"
    )


def _records_for(
    prediction: _SplitPrediction,
    *,
    placebo: bool,
    fold_id: int,
    candidate_score_constraint: str | None = None,
) -> UtilityRecords:
    scores, _ = _candidate_scores_for(
        prediction,
        placebo=placebo,
        candidate_score_constraint=candidate_score_constraint,
    )
    return build_utility_records(
        prediction.contrast,
        prediction.rigid_score,
        scores,
        actual=prediction.actual,
        fold_id=fold_id,
    )


def _nested_utility_audit(
    problem: _PreparedProblem,
    *,
    fit_workers: int,
    candidate_score_constraint: str | None = None,
) -> tuple[dict[str, Any], UtilityRecords, UtilityRecords]:
    folds = _fold_groups(problem.selected_groups)
    selected_set = set(problem.selected_groups)
    outer_rows: list[dict[str, Any]] = []
    rigid_values: list[float] = []
    source_values: list[float] = []
    placebo_values: list[float] = []
    final_source_records: list[UtilityRecords] = []
    final_placebo_records: list[UtilityRecords] = []
    for outer in folds:
        outer_index = int(outer["fold_index"])
        outer_heldout = tuple(str(value) for value in outer["heldout_groups"])
        outer_training = selected_set - set(outer_heldout)
        inner_specs = []
        for inner in folds:
            inner_index = int(inner["fold_index"])
            if inner_index == outer_index:
                continue
            inner_heldout = tuple(str(value) for value in inner["heldout_groups"])
            inner_training = outer_training - set(inner_heldout)
            inner_specs.append((inner_index, inner_training, inner_heldout))

        def fit_inner(
            spec: tuple[int, set[str], tuple[str, ...]],
        ) -> tuple[UtilityRecords, UtilityRecords]:
            inner_index, inner_training, inner_heldout = spec
            prediction = _fit_split(
                problem,
                training_groups=inner_training,
                evaluation_groups=inner_heldout,
                fit_workers=1,
            )
            return (
                _records_for(
                    prediction,
                    placebo=False,
                    fold_id=inner_index,
                    candidate_score_constraint=candidate_score_constraint,
                ),
                _records_for(
                    prediction,
                    placebo=True,
                    fold_id=inner_index,
                    candidate_score_constraint=candidate_score_constraint,
                ),
            )

        with ThreadPoolExecutor(max_workers=len(inner_specs)) as pool:
            inner_fitted = list(pool.map(fit_inner, inner_specs))
        inner_source_records = [row[0] for row in inner_fitted]
        inner_placebo_records = [row[1] for row in inner_fitted]
        print(
            "V150K_PROGRESS "
            f"city={problem.city} outer={outer_index + 1}/{len(folds)} "
            "phase=inner_complete",
            flush=True,
        )
        source_gate = fit_utility_gate(
            concatenate_utility_records(inner_source_records),
            config=GATE_CONFIG,
        )
        placebo_gate = fit_utility_gate(
            concatenate_utility_records(inner_placebo_records),
            config=GATE_CONFIG,
        )
        outer_prediction = _fit_split(
            problem,
            training_groups=outer_training,
            evaluation_groups=outer_heldout,
            fit_workers=fit_workers,
        )
        final_source_records.append(
            _records_for(
                outer_prediction,
                placebo=False,
                fold_id=outer_index,
                candidate_score_constraint=candidate_score_constraint,
            )
        )
        final_placebo_records.append(
            _records_for(
                outer_prediction,
                placebo=True,
                fold_id=outer_index,
                candidate_score_constraint=candidate_score_constraint,
            )
        )
        source_candidates, source_constraint_audit = _candidate_scores_for(
            outer_prediction,
            placebo=False,
            candidate_score_constraint=candidate_score_constraint,
        )
        placebo_candidates, placebo_constraint_audit = _candidate_scores_for(
            outer_prediction,
            placebo=True,
            candidate_score_constraint=candidate_score_constraint,
        )
        source_score, source_apply = apply_utility_gate(
            outer_prediction.contrast,
            outer_prediction.rigid_score,
            source_candidates,
            source_gate,
            fold_id=outer_index,
        )
        placebo_score, placebo_apply = apply_utility_gate(
            outer_prediction.contrast,
            outer_prediction.rigid_score,
            placebo_candidates,
            placebo_gate,
            fold_id=outer_index,
        )
        fold_rigid = _policy_values_for(
            outer_prediction, outer_prediction.rigid_score
        )
        fold_source = _policy_values_for(outer_prediction, source_score)
        fold_placebo = _policy_values_for(outer_prediction, placebo_score)
        rigid_values.extend(fold_rigid.tolist())
        source_values.extend(fold_source.tolist())
        placebo_values.extend(fold_placebo.tolist())
        outer_rows.append(
            {
                "outer_fold_index": outer_index,
                "outer_training_group_count": len(outer_training),
                "outer_heldout_group_count": len(outer_heldout),
                "inner_fold_count": len(folds) - 1,
                "gain_vs_rigid": float(
                    np.mean(fold_rigid) - np.mean(fold_source)
                ),
                "gain_vs_matched_placebo": float(
                    np.mean(fold_placebo) - np.mean(fold_source)
                ),
                "source_gate": dict(source_gate.diagnostics),
                "matched_placebo_gate": dict(placebo_gate.diagnostics),
                "source_application": source_apply,
                "matched_placebo_application": placebo_apply,
                "source_candidate_constraint": source_constraint_audit,
                "matched_placebo_candidate_constraint": placebo_constraint_audit,
            }
        )
        print(
            "V150K_PROGRESS "
            f"city={problem.city} outer={outer_index + 1}/{len(folds)} "
            f"phase=outer_complete interventions="
            f"{source_apply['selected_group_count']}",
            flush=True,
        )
    target_gains = np.asarray(
        [row["gain_vs_rigid"] for row in outer_rows], dtype=float
    )
    identity_gains = np.asarray(
        [row["gain_vs_matched_placebo"] for row in outer_rows], dtype=float
    )
    source_interventions = sum(
        int(row["source_application"]["selected_group_count"])
        for row in outer_rows
    )
    checks = {
        "mean_gain_vs_rigid_at_least_0_0005": (
            float(np.mean(target_gains)) >= MINIMUM_OOF_GAIN
        ),
        "mean_gain_vs_matched_placebo_at_least_0_0005": (
            float(np.mean(identity_gains)) >= MINIMUM_OOF_GAIN
        ),
        "at_least_four_nondegrading_folds_vs_rigid": (
            int(np.count_nonzero(target_gains >= 0.0))
            >= MINIMUM_NONDEGRADING_FOLDS
        ),
        "at_least_four_nondegrading_folds_vs_matched_placebo": (
            int(np.count_nonzero(identity_gains >= 0.0))
            >= MINIMUM_NONDEGRADING_FOLDS
        ),
        "at_least_five_crossfitted_intervention_groups": (
            source_interventions >= MINIMUM_INTERVENTION_GROUPS
        ),
    }
    return (
        {
            "fold_count": len(folds),
            "mean_gain_vs_rigid": float(np.mean(target_gains)),
            "mean_gain_vs_matched_placebo": float(np.mean(identity_gains)),
            "nondegrading_fold_count_vs_rigid": int(
                np.count_nonzero(target_gains >= 0.0)
            ),
            "nondegrading_fold_count_vs_matched_placebo": int(
                np.count_nonzero(identity_gains >= 0.0)
            ),
            "source_intervention_group_count": source_interventions,
            "candidate_score_constraint": candidate_score_constraint,
            "checks": checks,
            "passed": all(checks.values()),
            "outer_folds": outer_rows,
            "rigid_policy_mean": float(np.mean(rigid_values)),
            "source_policy_mean": float(np.mean(source_values)),
            "matched_placebo_policy_mean": float(np.mean(placebo_values)),
        },
        concatenate_utility_records(final_source_records),
        concatenate_utility_records(final_placebo_records),
    )


def run_target(
    *,
    target_city: str,
    authorization_result_path: Path,
    expected_authorization_sha256: str,
    v150j_rejection_path: Path,
    expected_v150j_rejection_sha256: str,
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
    runtime_model_path: Path | None = None,
    candidate_score_constraint: str | None = None,
    result_protocol: str = RESULT_PROTOCOL,
    method_protocol: str = METHOD_PROTOCOL,
    utility_gate_protocol: str = UTILITY_GATE_PROTOCOL,
    runtime_model_protocol: str = RUNTIME_MODEL_PROTOCOL,
    scientific_status: str = "seven-city-state-utility-development-target",
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V150K target city: {city}")
    authorization = _validate_authorization(
        authorization_result_path, expected_authorization_sha256
    )
    if _sha256(v150j_rejection_path) != str(expected_v150j_rejection_sha256):
        raise ValueError("V150J rejection identity changed")
    rejection = _read_json(v150j_rejection_path)
    if (
        rejection.get("protocol") != V150J_AGGREGATE_PROTOCOL
        or rejection.get("development_gate", {}).get("passed") is not False
        or rejection.get("development_gate", {}).get("decision")
        != "retain_sample_coherent_rigid_cfcmt_and_reject_source_claim"
    ):
        raise ValueError("V150K requires the frozen V150J rejection")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V150K source cache audit identity changed")
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    problem = _prepare_problem(
        city=city,
        fit_result=fit_result,
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        conversion_root=conversion_root,
        cache_workers=cache_workers,
        fit_workers=fit_workers,
    )
    nested_audit, source_oof_records, placebo_oof_records = _nested_utility_audit(
        problem,
        fit_workers=fit_workers,
        candidate_score_constraint=candidate_score_constraint,
    )
    source_gate = fit_utility_gate(source_oof_records, config=GATE_CONFIG)
    placebo_gate = fit_utility_gate(placebo_oof_records, config=GATE_CONFIG)
    final_prediction = _fit_split(
        problem,
        training_groups=problem.selected_groups,
        evaluation_groups=problem.evaluation_groups,
        fit_workers=fit_workers,
    )
    final_source_candidates, final_source_constraint_audit = _candidate_scores_for(
        final_prediction,
        placebo=False,
        candidate_score_constraint=candidate_score_constraint,
    )
    final_placebo_candidates, final_placebo_constraint_audit = (
        _candidate_scores_for(
            final_prediction,
            placebo=True,
            candidate_score_constraint=candidate_score_constraint,
        )
    )
    forced_source_score, source_application = apply_utility_gate(
        final_prediction.contrast,
        final_prediction.rigid_score,
        final_source_candidates,
        source_gate,
    )
    forced_placebo_score, placebo_application = apply_utility_gate(
        final_prediction.contrast,
        final_prediction.rigid_score,
        final_placebo_candidates,
        placebo_gate,
    )
    admitted = bool(nested_audit["passed"])
    selected_source_score = (
        forced_source_score
        if admitted
        else final_prediction.rigid_score.copy()
    )
    matched_placebo_score = (
        forced_placebo_score
        if admitted
        else final_prediction.rigid_score.copy()
    )
    source_null_score = final_prediction.rigid_score.copy()
    phase_score = _reference_policy_score(final_prediction.references)
    arms = {
        "phase_pressure": phase_score,
        "rigid_target_only": final_prediction.rigid_score,
        "source_null_exact_rigid": source_null_score,
        "forced_source_corrected_rigid": forced_source_score,
        "selected_source_corrected_rigid": selected_source_score,
        "same_candidate_placebo_corrected_rigid": matched_placebo_score,
        "forced_matched_placebo_corrected_rigid": forced_placebo_score,
    }
    evaluation = _group_subset_v3(
        problem.target_full,
        selected_groups=set(problem.evaluation_groups),
        metadata_updates={"target_data_role": "v150k_outside_b100_evaluation_only"},
    )
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=final_prediction.contrast,
            actual=final_prediction.actual,
            score=score,
            scenarios=problem.scenarios,
        )
        for name, score in arms.items()
    }
    if arm_summaries["rigid_target_only"] != arm_summaries[
        "source_null_exact_rigid"
    ]:
        raise ValueError("V150K source-null summary differs from rigid CFCMT")
    selector = {
        "candidate": "per_state_source_mechanism_pool",
        "source_city": "selected_per_state_without_source_identity_feature",
        "mechanism_block": "selected_per_state",
        "admitted": admitted,
        "crossfitted_selection_audit": nested_audit,
        "final_source_gate": dict(source_gate.diagnostics),
        "final_source_application": source_application,
        "final_matched_placebo_gate": dict(placebo_gate.diagnostics),
        "final_matched_placebo_application": placebo_application,
        "candidate_score_constraint": candidate_score_constraint,
        "final_source_candidate_constraint": final_source_constraint_audit,
        "final_matched_placebo_candidate_constraint": (
            final_placebo_constraint_audit
        ),
    }
    result = {
        "protocol": str(result_protocol),
        "occupancy_equation_protocol": problem.target_full.metadata.get("occupancy_equation_protocol"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": str(scientific_status),
        "target_city": city,
        "target_scenarios": list(problem.scenarios),
        "source_city_groups": list(problem.source_order),
        "target_budget": TARGET_BUDGET,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": problem.target_name,
        "method": {
            "protocol": str(method_protocol),
            "utility_gate_protocol": str(utility_gate_protocol),
            "ridge_l2": RIDGE_L2,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "candidate_count": len(problem.source_order)
            * len(MECHANISM_ACTION_FEATURES),
            "utility_gate_config": {
                "ridge_alpha": GATE_CONFIG.ridge_alpha,
                "error_quantile": GATE_CONFIG.error_quantile,
                "minimum_records": GATE_CONFIG.minimum_records,
                "minimum_predicted_gain": GATE_CONFIG.minimum_predicted_gain,
            },
            "selector": selector,
        },
        "information_budget": {
            "target_adaptation_group_count": len(problem.selected_groups),
            "common_evaluation_reserve_group_count": len(problem.reserve_groups),
            "target_evaluation_group_count": len(problem.evaluation_groups),
            "target_labels_used_for_nested_candidate_and_utility_gate_fit": True,
            "evaluation_features_or_labels_used_for_fit_or_selection": False,
            "source_labels_used_only_for_mechanism_coefficient_priors": True,
            "source_identity_used_as_utility_feature": False,
        },
        "selection_audit": problem.selection_audit,
        "evaluation_reserve_audit": problem.reserve_audit,
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "source_disabled_returns_rigid_score_bitwise": True,
            "rigid_scores_bitwise_equal": True,
            "summaries_equal": True,
        },
        "input_audits": {"source_bank": problem.source_bank_audit},
        "inputs": {
            "authorization_result_sha256": expected_authorization_sha256,
            "authorization_protocol": authorization["protocol"],
            "v150j_rejection_sha256": expected_v150j_rejection_sha256,
            "v150j_rejection_protocol": rejection["protocol"],
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "claim_boundary": claim_boundary
        or (
            "V150K is a seven-city B100 target-offline adaptation experiment. "
            "It nests source-mechanism proposal fitting and state-utility gating "
            "within the adaptation set and evaluates once outside the fixed B100 "
            "reserve. It is not closed-loop or fresh-city confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }
    if runtime_model_path is not None:
        if problem.target_full.metadata.get("occupancy_equation_protocol") != OCCUPANCY_EQUATION_PROTOCOL:
            raise ValueError("V150K fitted dataset occupancy equation contract mismatch")
        runtime_payload = {
            "protocol": str(runtime_model_protocol),
            "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "city": city,
            "target_budget": TARGET_BUDGET,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "admitted": admitted,
            "selected_groups": problem.selected_groups,
            "source_order": problem.source_order,
            "frozen_candidate": problem.frozen_candidate,
            "mechanism_feature_names": problem.mechanism_feature_names,
            "feature_scale": final_prediction.feature_scale,
            "target_beta": final_prediction.target_beta,
            "candidate_betas": dict(final_prediction.candidate_betas),
            "placebo_betas": dict(final_prediction.placebo_betas),
            "rigid_model": final_prediction.rigid_model,
            "source_utility_gate": source_gate,
            "placebo_utility_gate": placebo_gate,
            "nested_selection_audit": nested_audit,
            "candidate_score_constraint": candidate_score_constraint,
            "fit_contract": {
                "authorization_result_sha256": expected_authorization_sha256,
                "v150j_rejection_sha256": expected_v150j_rejection_sha256,
                "fit_result_sha256": expected_fit_result_sha256,
                "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            },
        }
        _atomic_bytes(
            Path(runtime_model_path),
            pickle.dumps(runtime_payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        result["runtime_model"] = {
            "protocol": str(runtime_model_protocol),
            "path": str(Path(runtime_model_path)),
            "sha256": _sha256(Path(runtime_model_path)),
            "size_bytes": int(Path(runtime_model_path).stat().st_size),
            "stored_remote_only": True,
        }
    return result


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    result_protocol: str = RESULT_PROTOCOL,
    aggregate_protocol: str = AGGREGATE_PROTOCOL,
    utility_gate_protocol: str = UTILITY_GATE_PROTOCOL,
    pass_decision: str = "authorize_state_conditioned_source_closed_loop_development",
    reject_decision: str = "retain_rigid_cfcmt_and_close_v150_source_gate_branch",
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != result_protocol for row in rows)
        or any(int(row.get("target_budget", -1)) != TARGET_BUDGET for row in rows)
        or any(
            row.get("method", {}).get("utility_gate_protocol")
            != utility_gate_protocol
            for row in rows
        )
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V150K aggregate requires seven valid B100 results")
    summary = _budget_summary(by_city)
    passed = bool(summary["passed"])
    return {
        "protocol": str(aggregate_protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(rows),
        "target_budget": TARGET_BUDGET,
        "source_prior_strength": SOURCE_PRIOR_STRENGTH,
        "utility_gate_protocol": str(utility_gate_protocol),
        **{
            key: value
            for key, value in summary.items()
            if key not in {"checks", "passed"}
        },
        "development_gate": {
            "inference_unit": "city",
            "checks": summary["checks"],
            "passed": passed,
            "decision": (
                pass_decision
                if passed
                else reject_decision
            ),
        },
        "claim_boundary": claim_boundary
        or (
            "V150K tests a nested state-conditioned utility gate on seven "
            "development cities. Passing cannot establish closed-loop or "
            "fresh-city efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--v150j-rejection", type=Path, required=True)
    target.add_argument("--v150j-rejection-sha256", required=True)
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
    target.add_argument("--runtime-model-out", type=Path)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150K result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            v150j_rejection_path=args.v150j_rejection,
            expected_v150j_rejection_sha256=args.v150j_rejection_sha256,
            fit_result_path=args.fit_result,
            expected_fit_result_sha256=args.fit_result_sha256,
            fit_protocol_path=args.fit_protocol,
            source_cache_root=args.source_cache_root,
            source_manifest_path=args.source_manifest,
            source_cache_audit_path=args.source_cache_audit,
            expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
            conversion_root=args.conversion_root,
            cache_workers=args.cache_workers,
            fit_workers=args.fit_workers,
            runtime_model_path=args.runtime_model_out,
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
