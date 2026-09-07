"""B100 target calibration of symmetric causal source predictions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
from statistics import NormalDist
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    TARGET_GROUP_BUDGET,
    _atomic_bytes,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    RESULT_PROTOCOL as COMPONENT_FIT_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    BOOTSTRAP_REPLICATES,
    MAXIMUM_HARMFUL_OVERRIDE_FRACTION,
    MAXIMUM_INTERVENTION_FRACTION,
    MAXIMUM_SEED_REGRESSION,
    MINIMUM_INTERVENTION_FRACTION,
    MINIMUM_PRESSURE_IMPROVEMENT,
    MINIMUM_SOURCE_CONTRIBUTION,
    SOURCE_FAMILYWISE_ALPHA,
    TARGET_FAMILYWISE_ALPHA,
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
    _load_zero_shot_models,
    _predict_source_models,
    _stable_group_rows,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.target_calibrated_source_gate import (
    ARM_NAMES,
    GATE_BASE_FEATURES,
    SOURCE_AGGREGATE_FEATURES,
    aggregate_source_predictions,
    extract_gate_base_features,
    fit_architecture_matched_gate_arms,
    gate_feature_matrix,
    group_conformal_margins,
    permute_source_features_by_group,
)


RESULT_PROTOCOL = "tsc-v119-target-calibrated-causal-source-gate-v1"
MODEL_PROTOCOL = "cfcmt-target-calibrated-causal-source-gate-v1"
CONFIDENCE_QUANTILES = (0.5, 0.75, 0.9, 0.95)
MINIMUM_PREDICTED_GAINS = (0.0, 0.0015, 0.003)
MINIMUM_SOURCE_SUPPORT = 0.5
BOOTSTRAP_SEED = 119003


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fit_result_contract(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError("V115 fit result identity changed")
    result = _read_json(path)
    selected = tuple(
        str(value)
        for value in result.get("information_budget", {}).get(
            "selected_group_ids", ()
        )
    )
    artifact = result.get("model_artifacts", {}).get("source_components_b0", {})
    if (
        result.get("protocol") != COMPONENT_FIT_RESULT_PROTOCOL
        or result.get("city") != "jinan"
        or result.get("target_name") != "prefix_mean_cost_450s"
        or result.get("prior_policy") != "phase_pressure"
        or len(selected) != TARGET_GROUP_BUDGET
        or len(set(selected)) != TARGET_GROUP_BUDGET
        or int(result.get("information_budget", {}).get(
            "zero_shot_target_transition_labels", -1
        )) != 0
        or not result.get("information_budget", {}).get(
            "target_only_architecture_matched", False
        )
        or not artifact.get("path")
        or not artifact.get("sha256")
    ):
        raise ValueError("V115 result does not authorize target calibration")
    return result


def _target_cache_audit_contract(path: Path, fit_result: Mapping[str, Any]) -> dict[str, Any]:
    expected = fit_result.get("input_audits", {}).get("target", {})
    if str(Path(path)) != str(expected.get("path")):
        raise ValueError("target cache audit path differs from V115")
    if _sha256(path) != str(expected.get("sha256")):
        raise ValueError("target cache audit identity differs from V115")
    audit = _read_json(path)
    if (
        audit.get("status") != "PASS"
        or not audit.get("gate", {}).get("passed", False)
    ):
        raise ValueError("target cache audit does not authorize B100 calibration")
    return audit


def _selected_adaptation_dataset(
    *,
    cache_root: Path,
    manifest_path: Path,
    cache_workers: int,
    selected_group_ids: Sequence[str],
) -> tuple[Any, dict[str, Any], tuple[str, ...]]:
    manifest = load_traffic_signal_manifest(manifest_path)
    scenarios = tuple(
        name for name, group in manifest.city_groups.items() if group == "jinan"
    )
    if not scenarios:
        raise ValueError("Jinan target scenarios are absent from the manifest")
    bank, audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(cache_workers), 1),
    )
    full = _merge_city_datasets(bank, scenarios, city="jinan")
    selected = _group_subset_v3(
        full,
        selected_groups=set(str(value) for value in selected_group_ids),
        metadata_updates={
            "target_data_role": "v119_b100_target_calibration"
        },
    )
    _assert_pure_waiting_selector_dataset(
        selected, target_name="prefix_mean_cost_450s"
    )
    groups = np.asarray(action_group_ids(selected), dtype=str)
    if set(np.unique(groups).tolist()) != set(selected_group_ids):
        raise ValueError("B100 calibration group identity changed")
    return selected, audit, scenarios


def _selector_dataset(
    *,
    cache_root: Path,
    manifest_path: Path,
    scenario: str,
    seeds: Sequence[int],
    collection_shards: int,
    cache_workers: int,
) -> tuple[Any, dict[str, Any]]:
    manifest = load_traffic_signal_manifest(manifest_path)
    if scenario not in manifest.sumocfgs:
        raise ValueError("selector scenario is absent from the manifest")
    selected_manifest = replace(
        manifest,
        scenarios=tuple(
            row for row in manifest.scenarios if row.scenario == scenario
        ),
    )
    bank, audit = load_frozen_counterfactual_bank(
        cache_root,
        selected_manifest,
        seeds=tuple(int(value) for value in seeds),
        collection_shards=int(collection_shards),
        workers=max(int(cache_workers), 1),
    )
    dataset = _merge_city_datasets(bank, (scenario,), city="jinan")
    _assert_pure_waiting_selector_dataset(
        dataset, target_name="prefix_mean_cost_450s"
    )
    return dataset, audit


def _normalized_pressure_targets(
    absolute_dataset: Any,
    contrast_dataset: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    groups = np.asarray(action_group_ids(contrast_dataset), dtype=str)
    unique_groups, group_rows = _stable_group_rows(groups)
    references = np.asarray(contrast_dataset.metadata["is_reference"], dtype=bool)
    if any(np.count_nonzero(references[rows]) != 1 for rows in group_rows):
        raise ValueError("target calibration requires one PhasePressure row per group")
    absolute = np.asarray(
        absolute_dataset.targets["prefix_mean_cost_450s"], dtype=float
    )
    delta = np.asarray(
        contrast_dataset.targets["prefix_mean_cost_450s"], dtype=float
    )
    normalized = np.zeros(contrast_dataset.size, dtype=float)
    for rows in group_rows:
        normalized[rows] = delta[rows] / action_group_range(absolute[rows])
    if not np.allclose(normalized[references], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("PhasePressure normalized target is not zero")
    seeds = np.asarray(
        [parse_action_group_seed(str(group)) for group in unique_groups], dtype=int
    )
    return normalized, unique_groups, group_rows, references, seeds


def _profile_key(arm: str, quantile: float, minimum_gain: float) -> str:
    q = f"{float(quantile):g}".replace(".", "p")
    gain = f"{float(minimum_gain):g}".replace(".", "p")
    return f"{arm}__q{q}__gain{gain}"


def _seed_metrics(
    *,
    seeds: np.ndarray,
    accepted: np.ndarray,
    actual_delta: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for seed in sorted(int(value) for value in np.unique(seeds)):
        rows = seeds == seed
        accepted_rows = rows & accepted
        outcomes = np.where(accepted[rows], actual_delta[rows], 0.0)
        result[str(seed)] = {
            "group_count": int(np.count_nonzero(rows)),
            "accepted_override_count": int(np.count_nonzero(accepted_rows)),
            "harmful_override_count": int(
                np.count_nonzero(accepted_rows & (actual_delta > 0.0))
            ),
            "mean_normalized_delta_vs_phase_pressure": float(np.mean(outcomes)),
        }
    return result


def _build_profiles(
    *,
    arm: str,
    predicted: np.ndarray,
    margins: Mapping[str, float],
    group_rows: Sequence[np.ndarray],
    references: np.ndarray,
    group_seeds: np.ndarray,
    normalized_actual: np.ndarray,
    source_score_stack: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_rows = []
    reference_rows = []
    for rows in group_rows:
        candidate = rows[~references[rows]]
        if candidate.size == 0:
            raise ValueError("target-calibrated group has no candidate action")
        candidate_rows.append(int(candidate[int(np.argmin(predicted[candidate]))]))
        reference_rows.append(int(rows[references[rows]][0]))
    candidate_rows = np.asarray(candidate_rows, dtype=int)
    reference_rows = np.asarray(reference_rows, dtype=int)
    predicted_delta = np.asarray(predicted[candidate_rows], dtype=float)
    actual_delta = np.asarray(normalized_actual[candidate_rows], dtype=float)
    if source_score_stack is None:
        support = np.ones(candidate_rows.size, dtype=float)
    else:
        scores = np.asarray(source_score_stack, dtype=float)
        support = np.mean(
            scores[:, candidate_rows] < scores[:, reference_rows], axis=0
        )
    profiles = {}
    for quantile in CONFIDENCE_QUANTILES:
        margin = float(margins[str(float(quantile))])
        for minimum_gain in MINIMUM_PREDICTED_GAINS:
            accepted = predicted_delta + margin < -float(minimum_gain)
            if arm == "source_aligned":
                accepted &= support >= MINIMUM_SOURCE_SUPPORT
            key = _profile_key(arm, quantile, minimum_gain)
            profiles[key] = {
                "profile_key": key,
                "arm": arm,
                "confidence_quantile": float(quantile),
                "conformal_margin": margin,
                "minimum_predicted_gain": float(minimum_gain),
                "minimum_source_support": (
                    MINIMUM_SOURCE_SUPPORT if arm == "source_aligned" else 0.0
                ),
                "matched_target_profile_key": (
                    _profile_key("target_only", quantile, minimum_gain)
                    if arm == "source_aligned"
                    else None
                ),
                "matched_placebo_profile_key": (
                    _profile_key("source_placebo", quantile, minimum_gain)
                    if arm == "source_aligned"
                    else None
                ),
                "seed_metrics": _seed_metrics(
                    seeds=group_seeds,
                    accepted=accepted,
                    actual_delta=actual_delta,
                ),
            }
    return profiles, {
        "candidate_action_count": int(candidate_rows.size),
        "predicted_improvement_fraction": float(np.mean(predicted_delta < 0.0)),
        "mean_predicted_delta_vs_phase_pressure": float(np.mean(predicted_delta)),
        "mean_actual_candidate_delta_vs_phase_pressure": float(np.mean(actual_delta)),
        "mean_source_support_fraction": float(np.mean(support)),
    }


def _training_metrics(profile: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    rows = [profile["seed_metrics"][str(int(seed))] for seed in seeds]
    group_count = int(sum(int(row["group_count"]) for row in rows))
    accepted = int(sum(int(row["accepted_override_count"]) for row in rows))
    harmful = int(sum(int(row["harmful_override_count"]) for row in rows))
    return {
        "seed_deltas": [
            float(row["mean_normalized_delta_vs_phase_pressure"]) for row in rows
        ],
        "group_count": group_count,
        "accepted_override_count": accepted,
        "harmful_override_count": harmful,
        "intervention_fraction": accepted / max(group_count, 1),
        "harmful_override_fraction": harmful / max(accepted, 1),
    }


def _normal_upper(values: Sequence[float], *, critical: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("profile confidence requires at least two finite seeds")
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(array.size))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "upper_confidence_bound": float(mean + critical * standard_error),
        "worst": float(np.max(array)),
    }


def _absolute_eligibility(
    aggregate: Mapping[str, Any], confidence: Mapping[str, float]
) -> bool:
    return bool(
        int(aggregate["accepted_override_count"]) > 0
        and float(confidence["mean"]) <= -MINIMUM_PRESSURE_IMPROVEMENT
        and float(confidence["upper_confidence_bound"]) <= 0.0
        and float(confidence["worst"]) <= MAXIMUM_SEED_REGRESSION
        and float(aggregate["harmful_override_fraction"])
        <= MAXIMUM_HARMFUL_OVERRIDE_FRACTION
        and MINIMUM_INTERVENTION_FRACTION
        <= float(aggregate["intervention_fraction"])
        <= MAXIMUM_INTERVENTION_FRACTION
    )


def select_safe_arm_profile(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    training_seeds: Sequence[int],
    familywise_alpha: float = TARGET_FAMILYWISE_ALPHA,
) -> dict[str, Any]:
    seeds = tuple(sorted(int(value) for value in training_seeds))
    if len(seeds) < 2 or not profiles:
        raise ValueError("profile selection requires multiple seeds and profiles")
    critical = float(NormalDist().inv_cdf(1.0 - familywise_alpha / len(profiles)))
    grid = {}
    eligible = []
    for key, profile in sorted(profiles.items()):
        aggregate = _training_metrics(profile, seeds)
        pressure = _normal_upper(aggregate["seed_deltas"], critical=critical)
        admitted = _absolute_eligibility(aggregate, pressure)
        row = {
            "profile_key": key,
            **aggregate,
            "pressure_comparison": pressure,
            "eligible": admitted,
        }
        grid[key] = row
        if admitted:
            eligible.append(row)
    selected = min(
        eligible,
        key=lambda row: (
            float(row["pressure_comparison"]["mean"]),
            float(row["pressure_comparison"]["upper_confidence_bound"]),
            float(row["intervention_fraction"]),
            str(row["profile_key"]),
        ),
        default=None,
    )
    return {
        "selected_profile_key": None if selected is None else selected["profile_key"],
        "selection_reason": (
            "familywise_absolute_safety_gate_passed"
            if selected is not None
            else "phase_pressure_fallback"
        ),
        "training_seeds": list(seeds),
        "simultaneous_confidence_multiplier": critical,
        "grid": grid,
    }


def select_identifiable_source_profile(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    training_seeds: Sequence[int],
) -> dict[str, Any]:
    source_profiles = {
        key: row for key, row in profiles.items() if row["arm"] == "source_aligned"
    }
    seeds = tuple(sorted(int(value) for value in training_seeds))
    critical = float(
        NormalDist().inv_cdf(
            1.0 - SOURCE_FAMILYWISE_ALPHA / max(len(source_profiles), 1)
        )
    )
    grid = {}
    eligible = []
    for key, profile in sorted(source_profiles.items()):
        target = profiles[str(profile["matched_target_profile_key"])]
        placebo = profiles[str(profile["matched_placebo_profile_key"])]
        aggregate = _training_metrics(profile, seeds)
        target_aggregate = _training_metrics(target, seeds)
        placebo_aggregate = _training_metrics(placebo, seeds)
        pressure = _normal_upper(aggregate["seed_deltas"], critical=critical)
        target_contribution = _normal_upper(
            np.asarray(aggregate["seed_deltas"])
            - np.asarray(target_aggregate["seed_deltas"]),
            critical=critical,
        )
        placebo_contribution = _normal_upper(
            np.asarray(aggregate["seed_deltas"])
            - np.asarray(placebo_aggregate["seed_deltas"]),
            critical=critical,
        )
        admitted = bool(
            _absolute_eligibility(aggregate, pressure)
            and target_contribution["mean"] <= -MINIMUM_SOURCE_CONTRIBUTION
            and target_contribution["upper_confidence_bound"] <= 0.0
            and placebo_contribution["mean"] <= -MINIMUM_SOURCE_CONTRIBUTION
            and placebo_contribution["upper_confidence_bound"] <= 0.0
        )
        row = {
            "profile_key": key,
            **aggregate,
            "pressure_comparison": pressure,
            "target_only_contribution": target_contribution,
            "source_placebo_contribution": placebo_contribution,
            "eligible": admitted,
        }
        grid[key] = row
        if admitted:
            eligible.append(row)
    selected = min(
        eligible,
        key=lambda row: (
            float(row["pressure_comparison"]["mean"]),
            float(row["pressure_comparison"]["upper_confidence_bound"]),
            float(row["intervention_fraction"]),
            str(row["profile_key"]),
        ),
        default=None,
    )
    return {
        "selected_profile_key": None if selected is None else selected["profile_key"],
        "selection_reason": (
            "familywise_absolute_and_identifiable_source_gate_passed"
            if selected is not None
            else "source_rejected_phase_pressure_fallback"
        ),
        "training_seeds": list(seeds),
        "simultaneous_confidence_multiplier": critical,
        "grid": grid,
    }


def _heldout_metric(
    profiles: Mapping[str, Mapping[str, Any]],
    profile_key: str | None,
    seed: int,
) -> dict[str, Any]:
    if profile_key is None:
        group_count = next(iter(profiles.values()))["seed_metrics"][str(seed)][
            "group_count"
        ]
        return {
            "group_count": int(group_count),
            "accepted_override_count": 0,
            "harmful_override_count": 0,
            "mean_normalized_delta_vs_phase_pressure": 0.0,
        }
    return dict(profiles[profile_key]["seed_metrics"][str(seed)])


def _bootstrap(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(
        array, size=(BOOTSTRAP_REPLICATES, array.size), replace=True
    ).mean(axis=1)
    return {
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "worst": float(np.max(array)),
        "bootstrap_95pct_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_nonnegative": float(np.mean(draws >= 0.0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": int(seed),
    }


def nested_leave_one_seed_gate_evaluation(
    profiles: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    ordered = tuple(sorted(int(value) for value in seeds))
    arm_profiles = {
        arm: {key: row for key, row in profiles.items() if row["arm"] == arm}
        for arm in ARM_NAMES
    }
    folds = []
    for heldout in ordered:
        training = tuple(seed for seed in ordered if seed != heldout)
        source_selection = select_identifiable_source_profile(
            profiles, training_seeds=training
        )
        source_key = source_selection["selected_profile_key"]
        if source_key is None:
            matched_target_key = None
            matched_placebo_key = None
        else:
            source_profile = profiles[str(source_key)]
            matched_target_key = str(source_profile["matched_target_profile_key"])
            matched_placebo_key = str(source_profile["matched_placebo_profile_key"])
        independently_selected = {
            arm: select_safe_arm_profile(
                arm_profiles[arm], training_seeds=training
            )["selected_profile_key"]
            for arm in ("target_only", "source_placebo")
        }
        source_metric = _heldout_metric(profiles, source_key, heldout)
        matched_target = _heldout_metric(profiles, matched_target_key, heldout)
        matched_placebo = _heldout_metric(profiles, matched_placebo_key, heldout)
        folds.append(
            {
                "heldout_seed": heldout,
                "training_seed_count": len(training),
                "source_selected_profile_key": source_key,
                "source_selection_reason": source_selection["selection_reason"],
                "matched_target_profile_key": matched_target_key,
                "matched_placebo_profile_key": matched_placebo_key,
                "independent_target_profile_key": independently_selected["target_only"],
                "independent_placebo_profile_key": independently_selected["source_placebo"],
                "source": source_metric,
                "matched_target_only": matched_target,
                "matched_source_placebo": matched_placebo,
            }
        )
    source_values = np.asarray(
        [row["source"]["mean_normalized_delta_vs_phase_pressure"] for row in folds],
        dtype=float,
    )
    target_values = np.asarray(
        [
            row["matched_target_only"]["mean_normalized_delta_vs_phase_pressure"]
            for row in folds
        ],
        dtype=float,
    )
    placebo_values = np.asarray(
        [
            row["matched_source_placebo"]["mean_normalized_delta_vs_phase_pressure"]
            for row in folds
        ],
        dtype=float,
    )
    source_summary = _bootstrap(source_values, seed=BOOTSTRAP_SEED)
    target_contribution = _bootstrap(
        source_values - target_values, seed=BOOTSTRAP_SEED + 1
    )
    placebo_contribution = _bootstrap(
        source_values - placebo_values, seed=BOOTSTRAP_SEED + 2
    )
    selected_folds = sum(row["source_selected_profile_key"] is not None for row in folds)
    harmful_seed_fraction = float(np.mean(source_values > 0.0))
    passed = bool(
        selected_folds / len(folds) >= 0.5
        and source_summary["mean"] <= -MINIMUM_PRESSURE_IMPROVEMENT
        and source_summary["bootstrap_95pct_ci"][1] < 0.0
        and source_summary["worst"] <= MAXIMUM_SEED_REGRESSION
        and harmful_seed_fraction <= 0.25
        and target_contribution["mean"] <= -MINIMUM_SOURCE_CONTRIBUTION
        and target_contribution["bootstrap_95pct_ci"][1] < 0.0
        and placebo_contribution["mean"] <= -MINIMUM_SOURCE_CONTRIBUTION
        and placebo_contribution["bootstrap_95pct_ci"][1] < 0.0
    )
    return {
        "folds": folds,
        "summary": {
            "source": source_summary,
            "source_contribution_vs_matched_target_only": target_contribution,
            "source_contribution_vs_matched_placebo": placebo_contribution,
            "source_selected_fold_count": int(selected_folds),
            "source_selected_fold_fraction": float(selected_folds / len(folds)),
            "harmful_seed_fraction": harmful_seed_fraction,
            "source_transfer_gate_passed": passed,
        },
    }


def run_target_calibrated_source_gate(
    *,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    target_cache_root: Path,
    target_manifest_path: Path,
    target_cache_audit_path: Path,
    selector_cache_root: Path,
    selector_manifest_path: Path,
    selector_cache_audit_path: Path,
    expected_selector_cache_audit_sha256: str,
    selector_scenario: str,
    selector_seeds: Sequence[int],
    selector_collection_shards: int,
    conversion_root: Path,
    cache_workers: int,
    prediction_workers: int,
    gate_artifact_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if gate_artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite gate artifact: {gate_artifact_path}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    target_cache_audit = _target_cache_audit_contract(
        target_cache_audit_path, fit_result
    )
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
    )
    selected_group_ids = tuple(
        str(value)
        for value in fit_result["information_budget"]["selected_group_ids"]
    )
    adaptation, adaptation_bank_audit, adaptation_scenarios = (
        _selected_adaptation_dataset(
            cache_root=target_cache_root,
            manifest_path=target_manifest_path,
            cache_workers=cache_workers,
            selected_group_ids=selected_group_ids,
        )
    )
    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
        cache_workers=cache_workers,
    )
    source_artifact = fit_result["model_artifacts"]["source_components_b0"]
    bundle, source_models, prior_spec = _load_zero_shot_models(
        component_bundle_path=Path(source_artifact["path"]),
        expected_component_bundle_sha256=str(source_artifact["sha256"]),
        city="jinan",
    )
    if str(prior_spec.key) != "phase_pressure":
        raise ValueError("target calibration prior is not PhasePressure")
    adaptation_contrast = build_action_contrast_dataset(
        adaptation,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    selector_contrast = build_action_contrast_dataset(
        selector,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    (
        adaptation_actual,
        adaptation_groups,
        adaptation_group_rows,
        adaptation_references,
        _,
    ) = _normalized_pressure_targets(adaptation, adaptation_contrast)
    (
        selector_actual,
        selector_groups,
        selector_group_rows,
        selector_references,
        selector_group_seeds,
    ) = _normalized_pressure_targets(selector, selector_contrast)
    if set(selector_group_seeds.tolist()) != set(int(value) for value in selector_seeds):
        raise ValueError("selector seed coverage changed")
    adaptation_predictions = _predict_source_models(
        source_models=source_models,
        model_contrast=adaptation_contrast,
        workers=prediction_workers,
    )
    selector_predictions = _predict_source_models(
        source_models=source_models,
        model_contrast=selector_contrast,
        workers=prediction_workers,
    )
    adaptation_source, adaptation_score_stack, source_order = (
        aggregate_source_predictions(adaptation_predictions)
    )
    selector_source, selector_score_stack, selector_source_order = (
        aggregate_source_predictions(selector_predictions)
    )
    if source_order != selector_source_order:
        raise ValueError("source model order changed between calibration and selector")
    adaptation_base = extract_gate_base_features(adaptation_contrast)
    selector_base = extract_gate_base_features(selector_contrast)
    adaptation_row_groups = np.asarray(
        action_group_ids(adaptation_contrast), dtype=str
    )
    selector_row_groups = np.asarray(action_group_ids(selector_contrast), dtype=str)
    rank_index = GATE_BASE_FEATURES.index("delta_service_pressure")
    adaptation_placebo = permute_source_features_by_group(
        adaptation_source,
        adaptation_row_groups,
        adaptation_references,
        adaptation_base[:, rank_index],
    )
    selector_placebo = permute_source_features_by_group(
        selector_source,
        selector_row_groups,
        selector_references,
        selector_base[:, rank_index],
    )
    adaptation_matrices = {
        "target_only": gate_feature_matrix(
            adaptation_base, adaptation_source, arm="target_only"
        ),
        "source_aligned": gate_feature_matrix(
            adaptation_base, adaptation_source, arm="source_aligned"
        ),
        "source_placebo": gate_feature_matrix(
            adaptation_base, adaptation_placebo, arm="source_placebo"
        ),
    }
    fitted = fit_architecture_matched_gate_arms(
        feature_matrices=adaptation_matrices,
        targets=adaptation_actual,
        groups=adaptation_row_groups,
        is_reference=adaptation_references,
    )
    margins = group_conformal_margins(
        actual=adaptation_actual,
        oof_predictions=fitted["oof_predictions"],
        groups=adaptation_row_groups,
        is_reference=adaptation_references,
        quantiles=CONFIDENCE_QUANTILES,
    )
    selector_matrices = {
        "target_only": gate_feature_matrix(
            selector_base, selector_source, arm="target_only"
        ),
        "source_aligned": gate_feature_matrix(
            selector_base, selector_source, arm="source_aligned"
        ),
        "source_placebo": gate_feature_matrix(
            selector_base, selector_placebo, arm="source_placebo"
        ),
    }
    profiles = {}
    compatibility = {}
    for arm in ARM_NAMES:
        predicted = fitted["models"][arm].predict(selector_matrices[arm])
        arm_profiles, arm_compatibility = _build_profiles(
            arm=arm,
            predicted=predicted,
            margins=margins[arm],
            group_rows=selector_group_rows,
            references=selector_references,
            group_seeds=selector_group_seeds,
            normalized_actual=selector_actual,
            source_score_stack=(
                selector_score_stack if arm == "source_aligned" else None
            ),
        )
        profiles.update(arm_profiles)
        compatibility[arm] = arm_compatibility
    nested = nested_leave_one_seed_gate_evaluation(profiles, selector_seeds)
    source_selection = select_identifiable_source_profile(
        profiles, training_seeds=selector_seeds
    )
    target_selection = select_safe_arm_profile(
        {key: row for key, row in profiles.items() if row["arm"] == "target_only"},
        training_seeds=selector_seeds,
    )
    placebo_selection = select_safe_arm_profile(
        {key: row for key, row in profiles.items() if row["arm"] == "source_placebo"},
        training_seeds=selector_seeds,
    )
    deployment_authorized = bool(
        nested["summary"]["source_transfer_gate_passed"]
        and source_selection["selected_profile_key"] is not None
    )
    gate_payload = {
        "protocol": MODEL_PROTOCOL,
        "city": "jinan",
        "target_name": "prefix_mean_cost_450s",
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "prior_policy": "phase_pressure",
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": selected_group_ids,
        "source_model_artifact": dict(source_artifact),
        "source_group_order": source_order,
        "base_feature_names": GATE_BASE_FEATURES,
        "source_aggregate_feature_names": SOURCE_AGGREGATE_FEATURES,
        "gate_feature_names": fitted["feature_names"],
        "ridge_alpha": fitted["selected_alpha"],
        "models": fitted["models"],
        "conformal_margins": margins,
        "minimum_source_support": MINIMUM_SOURCE_SUPPORT,
        "selected_source_profile_key": source_selection["selected_profile_key"],
        "deployment_authorized": deployment_authorized,
        "fit_result_sha256": expected_fit_result_sha256,
        "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
    }
    gate_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        gate_artifact_path,
        pickle.dumps(gate_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    round_trip = pickle.loads(gate_artifact_path.read_bytes())
    if (
        round_trip.get("protocol") != MODEL_PROTOCOL
        or tuple(round_trip.get("source_group_order", ())) != source_order
        or set(round_trip.get("models", {})) != set(ARM_NAMES)
    ):
        raise ValueError("target-calibrated gate artifact round trip failed")
    result = {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "development_selector_not_fresh_city_confirmation",
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "city": "jinan",
        "selector_scenario": selector_scenario,
        "target_name": "prefix_mean_cost_450s",
        "information_budget": {
            "target_calibration_action_groups": TARGET_GROUP_BUDGET,
            "target_calibration_seeds": list(ADAPTATION_SEEDS),
            "selector_seed_count": len(tuple(selector_seeds)),
            "selector_seeds": [int(value) for value in selector_seeds],
            "selector_target_labels_used_for_model_fit": 0,
            "source_model_target_transition_labels": 0,
            "source_city_identity_features": 0,
            "symmetric_source_aggregate_feature_count": len(
                SOURCE_AGGREGATE_FEATURES
            ),
        },
        "architecture_match": {
            "arms": list(ARM_NAMES),
            "same_model_family": True,
            "same_ridge_alpha": True,
            "same_feature_dimension": True,
            "same_b100_groups": True,
            "same_group_cross_fit_folds": True,
            "only_aligned_source_information_differs": True,
            "source_placebo_preserves_group_block_marginals": True,
            "alpha_selection": fitted["alpha_selection_arm"],
            "selected_alpha": fitted["selected_alpha"],
        },
        "calibration_oof": {
            "group_count": len(adaptation_groups),
            "row_count": int(adaptation_contrast.size),
            "arm_diagnostics": fitted["arm_diagnostics"],
            "alpha_diagnostics": fitted["alpha_diagnostics"],
            "conformal_margins": margins,
        },
        "selector": {
            "group_count": len(selector_groups),
            "row_count": int(selector_contrast.size),
            "profile_count": len(profiles),
            "compatibility": compatibility,
            "nested_leave_one_seed": nested,
            "full_development_selection": {
                "source_aligned": source_selection,
                "target_only": target_selection,
                "source_placebo": placebo_selection,
            },
        },
        "gate": {
            "passed": deployment_authorized,
            "decision": (
                "freeze_for_fresh_city_confirmation"
                if deployment_authorized
                else "reject_source_gate_and_retain_phase_pressure_fallback"
            ),
            "requirements": {
                "minimum_pressure_improvement": MINIMUM_PRESSURE_IMPROVEMENT,
                "minimum_source_contribution": MINIMUM_SOURCE_CONTRIBUTION,
                "maximum_seed_regression": MAXIMUM_SEED_REGRESSION,
                "maximum_harmful_override_fraction": MAXIMUM_HARMFUL_OVERRIDE_FRACTION,
                "intervention_fraction_range": [
                    MINIMUM_INTERVENTION_FRACTION,
                    MAXIMUM_INTERVENTION_FRACTION,
                ],
                "minimum_source_support": MINIMUM_SOURCE_SUPPORT,
            },
        },
        "inputs": {
            "fit_result": {
                "path": str(Path(fit_result_path).resolve()),
                "sha256": expected_fit_result_sha256,
            },
            "target_cache_audit": {
                "path": str(Path(target_cache_audit_path).resolve()),
                "sha256": _sha256(target_cache_audit_path),
                "protocol": target_cache_audit.get("protocol"),
            },
            "selector_cache_audit": selector_audit,
            "source_model_artifact": dict(source_artifact),
            "source_bundle_adaptation_contract_sha256": bundle.get(
                "adaptation_contract_sha256"
            ),
        },
        "data_audits": {
            "adaptation_scenarios": list(adaptation_scenarios),
            "adaptation_bank": adaptation_bank_audit,
            "selector_bank": selector_bank_audit,
        },
        "model_artifact": {
            "path": str(gate_artifact_path.resolve()),
            "sha256": _sha256(gate_artifact_path),
            "size_bytes": gate_artifact_path.stat().st_size,
            "protocol": MODEL_PROTOCOL,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", default="jinan_3x4_real")
    parser.add_argument("--selector-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--selector-collection-shards", type=int, default=32)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--prediction-workers", type=int, default=7)
    parser.add_argument("--gate-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("cache workers must be in [1, 20]")
    if not 1 <= int(args.prediction_workers) <= 7:
        raise ValueError("prediction workers must be in [1, 7]")
    result = run_target_calibrated_source_gate(
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        target_cache_root=args.target_cache_root,
        target_manifest_path=args.target_manifest,
        target_cache_audit_path=args.target_cache_audit,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        prediction_workers=args.prediction_workers,
        gate_artifact_path=args.gate_artifact,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
