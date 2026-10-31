"""Integrate a selected mechanism source prior as a rigid-CFCMT score residual."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    CONTRAST_FEATURES,
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MECHANISM_ACTION_FEATURES,
    MINIMUM_OOF_GAIN,
    PLACEBO_SEED,
    RIDGE_L2,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
    _arm_summary,
    _candidate_key,
    _fit_with_prior,
    _fold_groups,
    _paired_effects,
    _policy_values,
    _read_json,
    _select_candidate,
    _split_candidate_key,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    SOURCE_SEEDS,
    _reference_policy_score,
    select_city_budget_groups,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_right_of_way_signature import _read_manifest
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
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
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    equal_city_rms_scale,
    mechanism_parameter_design,
    permute_group_targets,
    ridge_coefficients,
)
from cf_h2o.traffic_signal.right_of_way_context import (
    augment_dataset_with_right_of_way_context,
    net_file_from_sumocfg,
    read_network_right_of_way_context,
)


RESULT_PROTOCOL = "tsc-v150d-rigid-mechanism-prior-integration-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150d-rigid-mechanism-prior-integration-aggregate-v1"
AUTHORIZATION_PROTOCOL = "tsc-v150c-mechanism-parameter-prior-aggregate-v1"


def residual_corrected_rigid_score(
    rigid_score: np.ndarray,
    mechanism_source_score: np.ndarray,
    mechanism_target_score: np.ndarray,
    references: Sequence[bool],
) -> np.ndarray:
    """Add only the source-induced mechanism delta to a rigid score."""

    rigid = np.asarray(rigid_score, dtype=float)
    source = np.asarray(mechanism_source_score, dtype=float)
    target = np.asarray(mechanism_target_score, dtype=float)
    reference_mask = np.asarray(references, dtype=bool)
    if (
        rigid.ndim != 1
        or source.shape != rigid.shape
        or target.shape != rigid.shape
        or reference_mask.shape != rigid.shape
        or not np.all(np.isfinite(rigid))
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("rigid mechanism-prior scores are not row aligned")
    if np.array_equal(source, target):
        return rigid.copy()
    corrected = rigid + (source - target)
    corrected[reference_mask] = 0.0
    if not np.all(np.isfinite(corrected)):
        raise ValueError("rigid mechanism-prior correction is non-finite")
    return corrected


def select_identity_verified_candidate(
    values: Mapping[str, Sequence[float]],
    placebo_values: Mapping[str, Sequence[float]],
    *,
    target_values: Sequence[float],
    minimum_target_gain: float,
    minimum_identity_gain: float,
) -> dict[str, Any]:
    """Select only candidates whose source labels beat their matched placebo."""

    if not values or set(values) != set(placebo_values):
        raise ValueError("identity-verified selector candidates are not aligned")
    target = np.asarray(target_values, dtype=float)
    if target.ndim != 1 or target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("identity-verified selector target values are invalid")
    target_mean = float(np.mean(target))
    diagnostics: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for key in sorted(values):
        candidate = np.asarray(values[key], dtype=float)
        placebo = np.asarray(placebo_values[key], dtype=float)
        if (
            candidate.shape != target.shape
            or placebo.shape != target.shape
            or not np.all(np.isfinite(candidate))
            or not np.all(np.isfinite(placebo))
        ):
            raise ValueError(f"identity-verified candidate rows changed: {key}")
        candidate_mean = float(np.mean(candidate))
        placebo_mean = float(np.mean(placebo))
        target_gain = target_mean - candidate_mean
        identity_gain = placebo_mean - candidate_mean
        admitted = bool(
            target_gain >= float(minimum_target_gain)
            and identity_gain >= float(minimum_identity_gain)
        )
        eligible.extend([key] if admitted else [])
        diagnostics[key] = {
            "candidate_mean": candidate_mean,
            "matched_placebo_mean": placebo_mean,
            "gain_vs_target_only": float(target_gain),
            "gain_vs_matched_placebo": float(identity_gain),
            "passes_target_gain": bool(target_gain >= float(minimum_target_gain)),
            "passes_identity_gain": bool(
                identity_gain >= float(minimum_identity_gain)
            ),
            "eligible": admitted,
        }
    selected_pool = eligible or list(diagnostics)
    best = min(
        selected_pool,
        key=lambda key: (diagnostics[key]["candidate_mean"], key),
    )
    source, block = _split_candidate_key(best)
    selected = diagnostics[best]
    return {
        "candidate": best,
        "source_city": source,
        "mechanism_block": block,
        "target_only_mean": target_mean,
        **selected,
        "minimum_target_gain": float(minimum_target_gain),
        "minimum_identity_gain": float(minimum_identity_gain),
        "admitted": bool(best in eligible),
        "eligible_candidate_count": len(eligible),
        "candidate_diagnostics": diagnostics,
    }


def select_crossfitted_identity_verified_candidate(
    fold_values: Mapping[str, Sequence[Sequence[float]]],
    fold_placebo_values: Mapping[str, Sequence[Sequence[float]]],
    *,
    target_fold_values: Sequence[Sequence[float]],
    minimum_target_gain: float,
    minimum_identity_gain: float,
    minimum_nondegrading_folds: int = 4,
) -> dict[str, Any]:
    """Cross-fit candidate selection, not only candidate model prediction."""

    if not fold_values or set(fold_values) != set(fold_placebo_values):
        raise ValueError("crossfitted selector candidates are not aligned")
    fold_count = len(target_fold_values)
    if fold_count < 2 or not 1 <= int(minimum_nondegrading_folds) <= fold_count:
        raise ValueError("crossfitted selector fold requirements are invalid")
    for key in fold_values:
        if (
            len(fold_values[key]) != fold_count
            or len(fold_placebo_values[key]) != fold_count
        ):
            raise ValueError(f"crossfitted candidate fold count changed: {key}")

    heldout_rows: list[dict[str, Any]] = []
    target_gains: list[float] = []
    identity_gains: list[float] = []
    for heldout_index in range(fold_count):
        training_indices = tuple(
            index for index in range(fold_count) if index != heldout_index
        )
        training_target = [
            value
            for index in training_indices
            for value in target_fold_values[index]
        ]
        training_candidates = {
            key: [
                value
                for index in training_indices
                for value in fold_values[key][index]
            ]
            for key in fold_values
        }
        training_placebos = {
            key: [
                value
                for index in training_indices
                for value in fold_placebo_values[key][index]
            ]
            for key in fold_values
        }
        training_selector = select_identity_verified_candidate(
            training_candidates,
            training_placebos,
            target_values=training_target,
            minimum_target_gain=minimum_target_gain,
            minimum_identity_gain=minimum_identity_gain,
        )
        key = str(training_selector["candidate"])
        heldout_target = np.asarray(target_fold_values[heldout_index], dtype=float)
        heldout_candidate = np.asarray(fold_values[key][heldout_index], dtype=float)
        heldout_placebo = np.asarray(
            fold_placebo_values[key][heldout_index], dtype=float
        )
        if (
            heldout_target.ndim != 1
            or heldout_target.size == 0
            or heldout_candidate.shape != heldout_target.shape
            or heldout_placebo.shape != heldout_target.shape
            or not np.all(np.isfinite(heldout_target))
            or not np.all(np.isfinite(heldout_candidate))
            or not np.all(np.isfinite(heldout_placebo))
        ):
            raise ValueError("crossfitted selector held-out values changed")
        candidate_admitted = bool(training_selector["admitted"])
        target_gain = (
            float(np.mean(heldout_target) - np.mean(heldout_candidate))
            if candidate_admitted
            else 0.0
        )
        identity_gain = (
            float(np.mean(heldout_placebo) - np.mean(heldout_candidate))
            if candidate_admitted
            else 0.0
        )
        target_gains.append(target_gain)
        identity_gains.append(identity_gain)
        heldout_rows.append(
            {
                "heldout_fold_index": heldout_index,
                "training_fold_indices": list(training_indices),
                "selected_candidate": key,
                "training_candidate_admitted": candidate_admitted,
                "heldout_gain_vs_target_only": target_gain,
                "heldout_gain_vs_matched_placebo": identity_gain,
            }
        )

    target_mean = float(np.mean(target_gains))
    identity_mean = float(np.mean(identity_gains))
    target_nondegrading = int(np.count_nonzero(np.asarray(target_gains) >= 0.0))
    identity_nondegrading = int(
        np.count_nonzero(np.asarray(identity_gains) >= 0.0)
    )
    crossfit_passed = bool(
        target_mean >= float(minimum_target_gain)
        and identity_mean >= float(minimum_identity_gain)
        and target_nondegrading >= int(minimum_nondegrading_folds)
        and identity_nondegrading >= int(minimum_nondegrading_folds)
    )
    full_selector = select_identity_verified_candidate(
        {
            key: [value for fold in fold_values[key] for value in fold]
            for key in fold_values
        },
        {
            key: [value for fold in fold_placebo_values[key] for value in fold]
            for key in fold_values
        },
        target_values=[value for fold in target_fold_values for value in fold],
        minimum_target_gain=minimum_target_gain,
        minimum_identity_gain=minimum_identity_gain,
    )
    return {
        **full_selector,
        "admitted": bool(full_selector["admitted"] and crossfit_passed),
        "full_data_candidate_admitted": bool(full_selector["admitted"]),
        "crossfitted_selection_audit": {
            "fold_count": fold_count,
            "minimum_nondegrading_folds": int(minimum_nondegrading_folds),
            "heldout_mean_gain_vs_target_only": target_mean,
            "heldout_mean_gain_vs_matched_placebo": identity_mean,
            "nondegrading_fold_count_vs_target_only": target_nondegrading,
            "nondegrading_fold_count_vs_matched_placebo": identity_nondegrading,
            "passed": crossfit_passed,
            "folds": heldout_rows,
        },
    }


def _rigid_score(model: Any, contrast: Any) -> np.ndarray:
    score, _, _, _ = _group_adjusted_scores(
        contrast,
        model.predict(contrast),
        objective_mode="control_only",
    )
    return np.asarray(score, dtype=float)


def _validate_authorization(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("V150D authorization result identity changed")
    value = _read_json(path)
    if (
        value.get("protocol") != AUTHORIZATION_PROTOCOL
        or value.get("development_gate", {}).get("passed") is not True
        or value.get("development_gate", {}).get("decision")
        != "promote_mechanism_prior_to_cfcmt_integration"
    ):
        raise ValueError("V150D lacks V150C integration authorization")
    return value


def run_target(
    *,
    target_city: str,
    authorization_result_path: Path,
    expected_authorization_sha256: str,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    expected_source_cache_audit_sha256: str,
    conversion_root: Path,
    cache_workers: int,
    target_budget: int = TARGET_BUDGET,
    evaluation_reserve_budget: int | None = None,
    source_prior_strength: float = SOURCE_PRIOR_STRENGTH,
    mechanism_design_builder: Callable[[Any], Any] = mechanism_parameter_design,
    mechanism_design_protocol: str = "single-state-conditioner-v1",
    result_protocol: str = RESULT_PROTOCOL,
    scientific_status: str = "seven-city-rigid-mechanism-integration-development-target",
    method_protocol: str = "rigid-score-plus-source-induced-mechanism-delta-v1",
    selector_mode: str = "target_gain_only",
    minimum_identity_gain: float = MINIMUM_OOF_GAIN,
    matched_placebo_follows_source_admission: bool = False,
    version_label: str = "v150d",
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    budget = int(target_budget)
    reserve_budget = (
        budget
        if evaluation_reserve_budget is None
        else int(evaluation_reserve_budget)
    )
    prior_strength = float(source_prior_strength)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"unknown V150D target city: {city}")
    if (
        budget < FOLD_COUNT
        or budget % FOLD_COUNT
        or reserve_budget < budget
    ):
        raise ValueError(
            "mechanism-prior target budget must be divisible by the fold count "
            "and no larger than the evaluation reserve"
        )
    if not np.isfinite(prior_strength) or prior_strength < 0.0:
        raise ValueError("mechanism-prior source strength must be finite and nonnegative")
    authorization = _validate_authorization(
        authorization_result_path, expected_authorization_sha256
    )
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V150D source cache audit identity changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    protocol, bank, scenarios_by_group, bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    if tuple(sorted(scenarios_by_group)) != EXPECTED_CITY_GROUPS:
        raise ValueError("V150D seven-city inventory changed")

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
    target_full = city_datasets[city]
    selected_groups, selection_audit = select_city_budget_groups(
        target_full,
        city=city,
        scenarios=scenarios_by_group[city],
        budget=budget,
    )
    if reserve_budget == budget:
        reserve_groups = selected_groups
        reserve_audit = selection_audit
    else:
        reserve_groups, reserve_audit = select_city_budget_groups(
            target_full,
            city=city,
            scenarios=scenarios_by_group[city],
            budget=reserve_budget,
        )
        if not set(selected_groups).issubset(reserve_groups):
            raise ValueError("target-budget selections are not nested")
    all_target_groups = set(
        str(value) for value in target_full.metadata["action_group_ids"]
    )
    evaluation_groups = all_target_groups - set(reserve_groups)
    if not evaluation_groups or evaluation_groups & set(reserve_groups):
        raise ValueError("V150D adaptation/evaluation split failed")
    adaptation = _group_subset_v3(
        target_full,
        selected_groups=set(selected_groups),
        metadata_updates={
            "target_data_role": f"{version_label}_b{budget}_adaptation_only"
        },
    )
    evaluation = _group_subset_v3(
        target_full,
        selected_groups=evaluation_groups,
        metadata_updates={
            "target_data_role": (
                f"{version_label}_outside_b{reserve_budget}_evaluation_only"
            )
        },
    )

    city_parts: dict[str, dict[str, Any]] = {}
    for group, absolute in city_datasets.items():
        contrast = build_action_contrast_dataset(
            absolute,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES,
        )
        actual, _, _, references, _ = _normalized_pressure_targets(
            absolute, contrast
        )
        city_parts[group] = {
            "contrast": contrast,
            "actual": actual,
            "references": references,
            "design": mechanism_design_builder(contrast),
        }

    adaptation_contrast = build_action_contrast_dataset(
        adaptation,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    adaptation_actual, _, _, adaptation_references, _ = _normalized_pressure_targets(
        adaptation, adaptation_contrast
    )
    adaptation_design = mechanism_design_builder(adaptation_contrast)
    adaptation_groups = np.asarray(
        action_group_ids(adaptation_contrast), dtype=str
    )
    evaluation_contrast = build_action_contrast_dataset(
        evaluation,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    evaluation_actual, _, _, evaluation_references, _ = _normalized_pressure_targets(
        evaluation, evaluation_contrast
    )
    evaluation_design = mechanism_design_builder(evaluation_contrast)
    if (
        adaptation_design.feature_names != evaluation_design.feature_names
        or adaptation_design.feature_names
        != next(iter(city_parts.values()))["design"].feature_names
    ):
        raise ValueError("V150D mechanism design schema changed across cities")

    source_order = tuple(group for group in EXPECTED_CITY_GROUPS if group != city)
    scale = equal_city_rms_scale(
        [city_parts[group]["design"].values for group in source_order]
        + [adaptation_design.values]
    )
    source_coefficients: dict[str, np.ndarray] = {}
    placebo_coefficients: dict[str, np.ndarray] = {}
    for source_index, source in enumerate(source_order):
        part = city_parts[source]
        source_x = np.asarray(part["design"].values, dtype=float) / scale
        source_y = np.asarray(part["actual"], dtype=float)
        source_groups = np.asarray(
            action_group_ids(part["contrast"]), dtype=str
        )
        source_coefficients[source] = ridge_coefficients(
            source_x,
            source_y,
            source_groups,
            ridge_l2=RIDGE_L2,
        )
        placebo_y = permute_group_targets(
            source_y,
            source_groups,
            seed=PLACEBO_SEED + source_index,
            references=part["references"],
        )
        placebo_coefficients[source] = ridge_coefficients(
            source_x,
            placebo_y,
            source_groups,
            ridge_l2=RIDGE_L2,
        )

    x_adaptation = np.asarray(adaptation_design.values, dtype=float) / scale
    folds = _fold_groups(selected_groups)
    rigid_oof_values: list[float] = []
    candidate_oof_values = {
        _candidate_key(source, block): []
        for source in source_order
        for block in MECHANISM_ACTION_FEATURES
    }
    placebo_oof_values = {key: [] for key in candidate_oof_values}
    rigid_fold_values: list[list[float]] = []
    candidate_fold_values: dict[str, list[list[float]]] = {
        key: [] for key in candidate_oof_values
    }
    placebo_fold_values: dict[str, list[list[float]]] = {
        key: [] for key in candidate_oof_values
    }
    fold_audit = []
    frozen_candidate = str(fit_result["frozen_blend_candidate"])
    for fold in folds:
        train_mask = np.isin(adaptation_groups, fold["training_groups"])
        heldout_mask = np.isin(adaptation_groups, fold["heldout_groups"])
        if np.any(train_mask & heldout_mask) or not np.all(train_mask | heldout_mask):
            raise ValueError(
                f"V150D fold masks do not partition B{budget} rows"
            )
        heldout_absolute = _group_subset_v3(
            adaptation,
            selected_groups=set(fold["heldout_groups"]),
            metadata_updates={
                "target_data_role": f"v150d_oof_fold_{fold['fold_index']}"
            },
        )
        heldout_contrast = build_action_contrast_dataset(
            heldout_absolute,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES,
        )
        heldout_actual, _, _, heldout_references, _ = _normalized_pressure_targets(
            heldout_absolute, heldout_contrast
        )
        heldout_groups = np.asarray(action_group_ids(heldout_contrast), dtype=str)
        expected_groups = adaptation_groups[heldout_mask]
        if not np.array_equal(heldout_groups, expected_groups):
            raise ValueError("V150D held-out contrast row order changed")
        rigid_model = _target_model(
            adaptation,
            group_ids=fold["training_groups"],
            candidate=frozen_candidate,
            target_domain=city,
        )
        rigid_score = _rigid_score(rigid_model, heldout_contrast)
        beta_target = ridge_coefficients(
            x_adaptation[train_mask],
            adaptation_actual[train_mask],
            adaptation_groups[train_mask],
            ridge_l2=RIDGE_L2,
        )
        heldout_x = x_adaptation[heldout_mask]
        target_mechanism_score = heldout_x @ beta_target
        fold_rigid_values = _policy_values(
            heldout_actual,
            rigid_score,
            heldout_groups,
            heldout_references,
        )
        rigid_oof_values.extend(fold_rigid_values.tolist())
        rigid_fold_values.append(fold_rigid_values.tolist())
        for source in source_order:
            for block, indices in adaptation_design.block_indices.items():
                key = _candidate_key(source, block)
                beta = _fit_with_prior(
                    x_adaptation[train_mask],
                    adaptation_actual[train_mask],
                    adaptation_groups[train_mask],
                    source_coefficients=source_coefficients[source],
                    block_indices=indices,
                    prior_strength=prior_strength,
                )
                corrected = residual_corrected_rigid_score(
                    rigid_score,
                    heldout_x @ beta,
                    target_mechanism_score,
                    heldout_references,
                )
                fold_candidate_values = _policy_values(
                    heldout_actual,
                    corrected,
                    heldout_groups,
                    heldout_references,
                )
                candidate_oof_values[key].extend(fold_candidate_values.tolist())
                candidate_fold_values[key].append(
                    fold_candidate_values.tolist()
                )
                placebo_beta = _fit_with_prior(
                    x_adaptation[train_mask],
                    adaptation_actual[train_mask],
                    adaptation_groups[train_mask],
                    source_coefficients=placebo_coefficients[source],
                    block_indices=indices,
                    prior_strength=prior_strength,
                )
                placebo_corrected = residual_corrected_rigid_score(
                    rigid_score,
                    heldout_x @ placebo_beta,
                    target_mechanism_score,
                    heldout_references,
                )
                fold_placebo_candidate_values = _policy_values(
                    heldout_actual,
                    placebo_corrected,
                    heldout_groups,
                    heldout_references,
                )
                placebo_oof_values[key].extend(
                    fold_placebo_candidate_values.tolist()
                )
                placebo_fold_values[key].append(
                    fold_placebo_candidate_values.tolist()
                )
        fold_audit.append(
            {
                **fold,
                "training_row_count": int(np.count_nonzero(train_mask)),
                "heldout_row_count": int(np.count_nonzero(heldout_mask)),
            }
        )

    if selector_mode == "target_gain_only":
        selector = _select_candidate(
            candidate_oof_values,
            target_values=rigid_oof_values,
            minimum_gain=MINIMUM_OOF_GAIN,
        )
    elif selector_mode == "target_and_matched_placebo_gain":
        selector = select_identity_verified_candidate(
            candidate_oof_values,
            placebo_oof_values,
            target_values=rigid_oof_values,
            minimum_target_gain=MINIMUM_OOF_GAIN,
            minimum_identity_gain=minimum_identity_gain,
        )
    elif selector_mode == "crossfitted_target_and_matched_placebo_gain":
        selector = select_crossfitted_identity_verified_candidate(
            candidate_fold_values,
            placebo_fold_values,
            target_fold_values=rigid_fold_values,
            minimum_target_gain=MINIMUM_OOF_GAIN,
            minimum_identity_gain=minimum_identity_gain,
        )
    else:
        raise ValueError(f"unknown rigid mechanism selector mode: {selector_mode}")
    placebo_selector = _select_candidate(
        placebo_oof_values,
        target_values=rigid_oof_values,
        minimum_gain=MINIMUM_OOF_GAIN,
    )
    rigid_model = _target_model(
        adaptation,
        group_ids=selected_groups,
        candidate=frozen_candidate,
        target_domain=city,
    )
    rigid_score = _rigid_score(rigid_model, evaluation_contrast)
    beta_target = ridge_coefficients(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        ridge_l2=RIDGE_L2,
    )
    source_null_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=np.full(beta_target.shape, 1e6),
        block_indices=adaptation_design.block_indices["queue_service"],
        prior_strength=0.0,
    )
    if not np.array_equal(beta_target, source_null_beta):
        raise ValueError("V150D zero source prior changed target mechanism fit")
    x_evaluation = np.asarray(evaluation_design.values, dtype=float) / scale
    target_mechanism_score = x_evaluation @ beta_target
    source_null_score = residual_corrected_rigid_score(
        rigid_score,
        x_evaluation @ source_null_beta,
        target_mechanism_score,
        evaluation_references,
    )
    if not np.array_equal(rigid_score, source_null_score):
        raise ValueError("V150D source-null does not exactly recover rigid CFCMT")

    selected_source, selected_block = _split_candidate_key(selector["candidate"])
    selected_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=source_coefficients[selected_source],
        block_indices=adaptation_design.block_indices[selected_block],
        prior_strength=prior_strength,
    )
    forced_source_score = residual_corrected_rigid_score(
        rigid_score,
        x_evaluation @ selected_beta,
        target_mechanism_score,
        evaluation_references,
    )
    selected_source_score = (
        forced_source_score if selector["admitted"] else rigid_score.copy()
    )
    same_candidate_placebo_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=placebo_coefficients[selected_source],
        block_indices=adaptation_design.block_indices[selected_block],
        prior_strength=prior_strength,
    )
    forced_same_candidate_placebo_score = residual_corrected_rigid_score(
        rigid_score,
        x_evaluation @ same_candidate_placebo_beta,
        target_mechanism_score,
        evaluation_references,
    )
    same_candidate_placebo_score = (
        forced_same_candidate_placebo_score
        if selector["admitted"] or not matched_placebo_follows_source_admission
        else rigid_score.copy()
    )
    placebo_source, placebo_block = _split_candidate_key(
        placebo_selector["candidate"]
    )
    selected_placebo_beta = _fit_with_prior(
        x_adaptation,
        adaptation_actual,
        adaptation_groups,
        source_coefficients=placebo_coefficients[placebo_source],
        block_indices=adaptation_design.block_indices[placebo_block],
        prior_strength=prior_strength,
    )
    forced_placebo_score = residual_corrected_rigid_score(
        rigid_score,
        x_evaluation @ selected_placebo_beta,
        target_mechanism_score,
        evaluation_references,
    )
    selected_placebo_score = (
        forced_placebo_score
        if placebo_selector["admitted"]
        else rigid_score.copy()
    )
    phase_score = _reference_policy_score(evaluation_references)
    arms = {
        "phase_pressure": phase_score,
        "rigid_target_only": rigid_score,
        "source_null_exact_rigid": source_null_score,
        "forced_source_corrected_rigid": forced_source_score,
        "selected_source_corrected_rigid": selected_source_score,
        "same_candidate_placebo_corrected_rigid": same_candidate_placebo_score,
        "selected_placebo_corrected_rigid": selected_placebo_score,
    }
    arm_summaries = {
        name: _arm_summary(
            absolute=evaluation,
            contrast=evaluation_contrast,
            actual=evaluation_actual,
            score=score,
            scenarios=scenarios_by_group[city],
        )
        for name, score in arms.items()
    }
    if arm_summaries["rigid_target_only"] != arm_summaries["source_null_exact_rigid"]:
        raise ValueError("V150D source-null summary differs from rigid CFCMT")

    return {
        "protocol": str(result_protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": str(scientific_status),
        "target_city": city,
        "target_scenarios": list(scenarios_by_group[city]),
        "source_city_groups": list(source_order),
        "target_budget": budget,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": str(protocol["target_name"]),
        "method": {
            "protocol": str(method_protocol),
            "equation": "rigid_score + mechanism_source_prior_score - mechanism_target_score",
            "frozen_rigid_candidate": frozen_candidate,
            "ridge_l2": RIDGE_L2,
            "source_prior_strength": prior_strength,
            "mechanism_design_protocol": str(mechanism_design_protocol),
            "minimum_oof_gain": MINIMUM_OOF_GAIN,
            "minimum_identity_gain": float(minimum_identity_gain),
            "selector_mode": str(selector_mode),
            "matched_placebo_follows_source_admission": bool(
                matched_placebo_follows_source_admission
            ),
            "selector": selector,
            "placebo_selector": placebo_selector,
        },
        "information_budget": {
            "target_adaptation_group_count": len(selected_groups),
            "common_evaluation_reserve_group_count": len(reserve_groups),
            "unused_reserved_group_count": len(reserve_groups) - len(selected_groups),
            "target_evaluation_group_count": len(evaluation_groups),
            "adaptation_labels_used_for_five_fold_source_and_block_selection": True,
            "evaluation_features_or_labels_used_for_fit_or_selection": False,
            "source_labels_used_only_for_mechanism_coefficient_priors": True,
        },
        "selection_audit": selection_audit,
        "evaluation_reserve_audit": reserve_audit,
        "crossfit_folds": fold_audit,
        "arm_summaries": arm_summaries,
        "source_null_contract": {
            "prior_strength_zero": True,
            "mechanism_coefficients_bitwise_equal": True,
            "rigid_scores_bitwise_equal": True,
            "summaries_equal": True,
        },
        "input_audits": {"source_bank": bank_audit},
        "inputs": {
            "authorization_result_sha256": expected_authorization_sha256,
            "authorization_protocol": authorization["protocol"],
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
            "source_manifest": str(source_manifest_path),
            "conversion_root": str(conversion_root),
        },
        "claim_boundary": claim_boundary
        or (
            f"{version_label.upper()} is a seven-city development integration "
            f"test. It uses B{budget} target outcomes for source and mechanism "
            f"selection and evaluates outside a fixed B{reserve_budget} reserve; "
            "it is not closed-loop or fresh-city confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def _city_bootstrap(city_means: Mapping[str, float]) -> dict[str, float]:
    if set(city_means) != set(EXPECTED_CITY_GROUPS):
        raise ValueError("V150D city bootstrap inventory changed")
    return _paired_bootstrap(
        [float(city_means[city]) for city in EXPECTED_CITY_GROUPS]
    )


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_city = {str(row.get("target_city")): row for row in rows}
    if (
        len(rows) != len(EXPECTED_CITY_GROUPS)
        or set(by_city) != set(EXPECTED_CITY_GROUPS)
        or any(row.get("protocol") != RESULT_PROTOCOL for row in rows)
        or any(
            row.get("source_null_contract", {}).get("summaries_equal") is not True
            for row in rows
        )
    ):
        raise ValueError("V150D aggregate requires seven valid exact-fallback results")
    target_effects, target_city_means = _paired_effects(
        by_city,
        "selected_source_corrected_rigid",
        "rigid_target_only",
    )
    selected_placebo_effects, selected_placebo_city_means = _paired_effects(
        by_city,
        "selected_source_corrected_rigid",
        "selected_placebo_corrected_rigid",
    )
    matched_placebo_effects, matched_placebo_city_means = _paired_effects(
        by_city,
        "selected_source_corrected_rigid",
        "same_candidate_placebo_corrected_rigid",
    )
    forced_effects, forced_city_means = _paired_effects(
        by_city,
        "forced_source_corrected_rigid",
        "rigid_target_only",
    )
    target_seed_summary = _paired_bootstrap(target_effects)
    selected_placebo_seed_summary = _paired_bootstrap(selected_placebo_effects)
    matched_placebo_seed_summary = _paired_bootstrap(matched_placebo_effects)
    forced_seed_summary = _paired_bootstrap(forced_effects)
    target_city_summary = _city_bootstrap(target_city_means)
    selected_placebo_city_summary = _city_bootstrap(selected_placebo_city_means)
    matched_placebo_city_summary = _city_bootstrap(matched_placebo_city_means)
    admitted_cities = sum(
        bool(row["method"]["selector"]["admitted"]) for row in rows
    )
    gates = {
        "mean_selected_effect_improves_rigid": target_city_summary["mean"] < 0.0,
        "city_bootstrap_upper_vs_rigid_below_zero": target_city_summary["upper_95"]
        < 0.0,
        "at_least_five_cities_improve_rigid": sum(
            value < 0.0 for value in target_city_means.values()
        )
        >= 5,
        "mean_selected_effect_beats_selected_placebo": selected_placebo_city_summary[
            "mean"
        ]
        < 0.0,
        "mean_selected_effect_beats_same_candidate_placebo": matched_placebo_city_summary[
            "mean"
        ]
        < 0.0,
        "city_bootstrap_upper_vs_same_candidate_placebo_below_zero": matched_placebo_city_summary[
            "upper_95"
        ]
        < 0.0,
        "at_least_four_cities_beat_same_candidate_placebo": sum(
            value < 0.0 for value in matched_placebo_city_means.values()
        )
        >= 4,
        "worst_city_regression_at_most_0p01": max(target_city_means.values())
        <= 0.01,
        "at_least_five_cities_admit_a_source_prior": admitted_cities >= 5,
    }
    passed = all(gates.values())
    return {
        "protocol": AGGREGATE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_gate_pass" if passed else "development_gate_reject"
        ),
        "city_count": len(rows),
        "target_seed_unit_count": len(target_effects),
        "selected_effect_vs_rigid_target_only": {
            "seed_unit": target_seed_summary,
            "city_unit": {**target_city_summary, "city_means": target_city_means},
        },
        "selected_effect_vs_selected_placebo": {
            "seed_unit": selected_placebo_seed_summary,
            "city_unit": {
                **selected_placebo_city_summary,
                "city_means": selected_placebo_city_means,
            },
        },
        "selected_effect_vs_same_candidate_placebo": {
            "seed_unit": matched_placebo_seed_summary,
            "city_unit": {
                **matched_placebo_city_summary,
                "city_means": matched_placebo_city_means,
            },
        },
        "forced_effect_vs_rigid_target_only": {
            "seed_unit": forced_seed_summary,
            "city_means": forced_city_means,
        },
        "selected_candidates": {
            city: by_city[city]["method"]["selector"]
            for city in EXPECTED_CITY_GROUPS
        },
        "development_gate": {
            "inference_unit": "city",
            "checks": gates,
            "passed": passed,
            "decision": (
                "authorize_closed_loop_mechanism_prior_development"
                if passed
                else "retain_rigid_cfcmt_and_do_not_launch_closed_loop_source_claim"
            ),
        },
        "claim_boundary": (
            "V150D integrates source-induced mechanism deltas with rigid CFCMT "
            "on seven development cities. A pass only authorizes closed-loop "
            "development; untouched-city confirmation remains required."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    target = subparsers.add_parser("target")
    target.add_argument("--target-city", required=True)
    target.add_argument("--authorization-result", type=Path, required=True)
    target.add_argument("--authorization-result-sha256", required=True)
    target.add_argument("--fit-result", type=Path, required=True)
    target.add_argument("--fit-result-sha256", required=True)
    target.add_argument("--fit-protocol", type=Path, required=True)
    target.add_argument("--source-cache-root", type=Path, required=True)
    target.add_argument("--source-manifest", type=Path, required=True)
    target.add_argument("--source-cache-audit", type=Path, required=True)
    target.add_argument("--source-cache-audit-sha256", required=True)
    target.add_argument("--conversion-root", type=Path, required=True)
    target.add_argument("--cache-workers", type=int, default=8)
    target.add_argument("--out", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results", nargs=7, type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150D result: {args.out}")
    if args.mode == "target":
        result = run_target(
            target_city=args.target_city,
            authorization_result_path=args.authorization_result,
            expected_authorization_sha256=args.authorization_result_sha256,
            fit_result_path=args.fit_result,
            expected_fit_result_sha256=args.fit_result_sha256,
            fit_protocol_path=args.fit_protocol,
            source_cache_root=args.source_cache_root,
            source_manifest_path=args.source_manifest,
            source_cache_audit_path=args.source_cache_audit,
            expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
            conversion_root=args.conversion_root,
            cache_workers=args.cache_workers,
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
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
