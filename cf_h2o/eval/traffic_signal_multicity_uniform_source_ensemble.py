"""Evaluate a fixed uniform causal source ensemble on one held-out city."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic import (
    _policy_metrics,
    _predictive_metrics,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    SELECTION_SEED,
    _balanced_quotas,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _group_adjusted_scores,
    merge_counterfactual_datasets_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _coverage_first_group_order_v3,
    _group_subset_v3,
    _relabel_dataset_domain,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _fit_component_budget,
    _paired_bootstrap,
    _policy_arrays,
    _source_inputs,
    _target_model,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting import (
    permute_source_residuals_by_seed_group,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    H2OPLUS_FAMILY,
    _fit_h2oplus_budget,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import (
    FixedParentResidualWorldModel,
    MechanismFitConfig,
)


RESULT_PROTOCOL = "tsc-v143-multicity-uniform-source-ensemble-target-v1"
TARGET_BUDGET = 25
SOURCE_SEEDS = (2027, 3037, 4047)
EXPECTED_CITY_GROUPS = (
    "atlanta",
    "cologne",
    "hangzhou",
    "ingolstadt",
    "new_york",
    "resco_synthetic",
    "salt_lake_city",
)
MECHANISM_COMPATIBILITY_NAMES = (
    "queue_propagation",
    "served_movement",
    "red_accumulation",
    "spillback",
    "mobility",
)
MECHANISM_COMPATIBILITY_DEFINITIONS = tuple(
    definition
    for definition in ONE_STEP_CONTRAST_MECHANISM_DEFINITIONS_V4
    if definition.name in set(MECHANISM_COMPATIBILITY_NAMES)
)
if tuple(definition.name for definition in MECHANISM_COMPATIBILITY_DEFINITIONS) != (
    MECHANISM_COMPATIBILITY_NAMES
):
    raise RuntimeError("V148 mechanism definition order changed")
_FIT_STATE: Mapping[str, Any] | None = None


def _scenario_from_group(group: str, scenarios: Sequence[str]) -> str:
    matches = [
        str(scenario)
        for scenario in scenarios
        if str(group).startswith(f"{scenario}:")
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot identify one V143 scenario for group {group}")
    return matches[0]


def _city_covariate_signature(
    bank: Mapping[str, Any], scenarios: Sequence[str]
) -> dict[str, Any]:
    names = tuple(str(value) for value in scenarios)
    if not names:
        raise ValueError("V144 city signature requires at least one scenario")
    datasets = [bank[name] for name in names]
    feature_names = tuple(str(value) for value in datasets[0].feature_names)
    context_names = tuple(str(value) for value in datasets[0].context_names)
    if any(tuple(dataset.feature_names) != feature_names for dataset in datasets):
        raise ValueError("V144 city signature feature schema changed")
    if any(tuple(dataset.context_names) != context_names for dataset in datasets):
        raise ValueError("V144 city signature context schema changed")
    feature_arrays = [np.asarray(dataset.features, dtype=float) for dataset in datasets]
    if any(array.ndim != 2 or array.shape[0] < 1 for array in feature_arrays):
        raise ValueError("V144 city signature has an empty feature matrix")
    context_rows = np.vstack(
        [_static_context_from_dataset(dataset) for dataset in datasets]
    )
    per_scenario_mean = np.vstack([np.mean(array, axis=0) for array in feature_arrays])
    per_scenario_std = np.vstack([np.std(array, axis=0) for array in feature_arrays])
    per_scenario_q25 = np.vstack(
        [np.quantile(array, 0.25, axis=0) for array in feature_arrays]
    )
    per_scenario_q75 = np.vstack(
        [np.quantile(array, 0.75, axis=0) for array in feature_arrays]
    )
    values = (
        context_rows,
        per_scenario_mean,
        per_scenario_std,
        per_scenario_q25,
        per_scenario_q75,
    )
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("V144 city signature contains non-finite covariates")
    return {
        "protocol": "scenario-equal-label-free-city-covariate-signature-v1",
        "information_boundary": "features_and_static_context_only_no_priors_or_targets",
        "scenario_count": len(names),
        "scenarios": list(names),
        "context_names": list(context_names),
        "context_mean": np.mean(context_rows, axis=0).tolist(),
        "context_std": np.std(context_rows, axis=0).tolist(),
        "feature_names": list(feature_names),
        "feature_mean": np.mean(per_scenario_mean, axis=0).tolist(),
        "feature_std": np.mean(per_scenario_std, axis=0).tolist(),
        "feature_q25": np.mean(per_scenario_q25, axis=0).tolist(),
        "feature_q75": np.mean(per_scenario_q75, axis=0).tolist(),
    }


def select_city_budget_groups(
    dataset: Any,
    *,
    city: str,
    scenarios: Sequence[str],
    budget: int = TARGET_BUDGET,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    scenario_order = tuple(str(value) for value in scenarios)
    groups = sorted(set(str(value) for value in dataset.metadata["action_group_ids"]))
    strata = tuple(
        (scenario, seed)
        for scenario in scenario_order
        for seed in SOURCE_SEEDS
        if any(
            group.startswith(f"{scenario}:")
            and parse_action_group_seed(group) == seed
            for group in groups
        )
    )
    if not strata:
        raise ValueError(f"{city}: V143 target has no scenario-seed strata")
    quota_keys = tuple(f"{scenario}|{seed}" for scenario, seed in strata)
    quotas = _balanced_quotas(int(budget), quota_keys)
    selected: list[str] = []
    audit_rows = {}
    for scenario, seed in strata:
        key = f"{scenario}|{seed}"
        candidates = [
            group
            for group in groups
            if group.startswith(f"{scenario}:")
            and parse_action_group_seed(group) == seed
        ]
        requested = int(quotas[key])
        if len(candidates) <= requested:
            raise ValueError(
                f"{city}: {key} cannot retain evaluation groups: "
                f"{len(candidates)} <= {requested}"
            )
        ordered = _coverage_first_group_order_v3(
            dataset,
            candidates,
            selection_seed=SELECTION_SEED,
            role=f"v143_city_b25:{city}:{scenario}:seed{seed}",
        )
        chosen = tuple(ordered[:requested])
        selected.extend(chosen)
        audit_rows[key] = {
            "available_group_count": len(candidates),
            "selected_group_count": len(chosen),
            "evaluation_group_count": len(candidates) - len(chosen),
        }
    if len(selected) != int(budget) or len(set(selected)) != int(budget):
        raise ValueError(f"{city}: V143 B{budget} group selection is incomplete")
    return tuple(sorted(selected)), {
        "protocol": "city-scenario-seed-balanced-coverage-first-b25-v1",
        "city": str(city),
        "budget": int(budget),
        "selection_seed": int(SELECTION_SEED),
        "scenario_seed_quotas": quotas,
        "strata": audit_rows,
        "selected_group_ids": sorted(selected),
    }


def _fit_role_model(
    state: Mapping[str, Any],
    *,
    role: str,
    source_group: str | None,
    target_group_ids: Sequence[str],
) -> tuple[Any, tuple[str, ...], str]:
    if role == "target_only":
        model = _target_model(
            state["target_dataset"],
            group_ids=target_group_ids,
            candidate=state["candidate"],
            target_domain=state["target_city"],
        )
        return model, (str(state["target_city"]),), "control_only"
    if role == "source":
        if source_group not in state["source_states"]:
            raise ValueError("V143 source task has an unknown city group")
        fitted = _fit_component_budget(
            state["source_states"][str(source_group)],
            target_group_ids=target_group_ids,
        )
        expected = {str(source_group), str(state["target_city"])}
        if set(fitted.diagnostics["source_domains"]) != expected:
            raise ValueError("V143 one-source information boundary changed")
        model = FrozenAnchoredBlendModel(
            anchor_model=fitted.family_models[state["anchor_family"]],
            correction_model=fitted.family_models[state["correction_family"]],
            candidate=state["candidate"],
            anchor_objective_mode=fitted.objective_modes[state["anchor_family"]],
            correction_objective_mode=fitted.objective_modes[
                state["correction_family"]
            ],
        )
        return model, tuple(sorted(expected)), "control_only"
    if role == "h2oplus":
        fitted = _fit_h2oplus_budget(
            fit_bank=state["h2oplus_fit_bank"],
            target_key=state["target_key"],
            city_groups=state["h2oplus_city_groups"],
            target_group_ids=target_group_ids,
            target_static_context=state["target_static_context"],
            prior_policy="phase_pressure",
            source_rule_specs=state["source_rule_specs"],
        )
        expected = set(state["source_order"]) | {str(state["target_city"])}
        if set(fitted.diagnostics["source_domains"]) != expected:
            raise ValueError("V143 pooled H2O+ information boundary changed")
        return (
            fitted.family_models[H2OPLUS_FAMILY],
            tuple(sorted(expected)),
            str(fitted.objective_modes[H2OPLUS_FAMILY]),
        )
    raise ValueError(f"unknown V143 fit role: {role!r}")


def _fit_worker(task: tuple[str, str | None]) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("V143 fit state is missing")
    from threadpoolctl import threadpool_limits

    role, source_group = task
    state = _FIT_STATE
    started = time.monotonic()
    with threadpool_limits(limits=1):
        model, domains, objective_mode = _fit_role_model(
            state,
            role=role,
            source_group=source_group,
            target_group_ids=state["selected_groups"],
        )
        score, _, _, _ = _group_adjusted_scores(
            state["evaluation_contrast"],
            model.predict(state["evaluation_contrast"]),
            objective_mode=objective_mode,
        )
        selector_diagnostic = None
        if state.get("adaptation_selector_contrast") is not None:
            selector_score, _, _, _ = _group_adjusted_scores(
                state["adaptation_selector_contrast"],
                model.predict(state["adaptation_selector_contrast"]),
                objective_mode=objective_mode,
            )
            selector_diagnostic = _label_free_intervention_diagnostic(
                selector_score,
                state["adaptation_selector_policy_rows"],
                state["adaptation_selector_references"],
            )
    return {
        "role": role,
        "source_group": source_group,
        "score": np.asarray(score, dtype=float),
        "training_domains": list(domains),
        "target_group_count": len(state["selected_groups"]),
        "adaptation_selector": selector_diagnostic,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _compatibility_fit_worker(
    task: tuple[str, str | None, int],
) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("V146 fit state is missing")
    from threadpoolctl import threadpool_limits

    role, source_group, fold_index = task
    state = _FIT_STATE
    fold = state["adaptation_compatibility_folds"][fold_index]
    started = time.monotonic()
    with threadpool_limits(limits=1):
        model, domains, objective_mode = _fit_role_model(
            state,
            role=role,
            source_group=source_group,
            target_group_ids=fold["training_groups"],
        )
        score, _, _, _ = _group_adjusted_scores(
            fold["contrast"],
            model.predict(fold["contrast"]),
            objective_mode=objective_mode,
        )
    return {
        "role": role,
        "source_group": source_group,
        "fold_index": int(fold_index),
        "score": np.asarray(score, dtype=float),
        "training_domains": list(domains),
        "training_group_count": len(fold["training_groups"]),
        "heldout_group_count": len(fold["heldout_groups"]),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _scenario_seed_policy_summary(
    *,
    actual: np.ndarray,
    score: np.ndarray,
    policy_rows: Sequence[np.ndarray],
    references: np.ndarray,
    unique_groups: Sequence[str],
    scenarios: Sequence[str],
) -> dict[str, Any]:
    values, interventions = _policy_arrays(actual, score, policy_rows, references)
    groups = tuple(str(value) for value in unique_groups)
    if len(groups) != values.size:
        raise ValueError("V143 policy values and groups are misaligned")
    units: dict[str, list[float]] = {}
    unit_interventions: dict[str, list[bool]] = {}
    for index, group in enumerate(groups):
        scenario = _scenario_from_group(group, scenarios)
        seed = parse_action_group_seed(group)
        key = f"{scenario}|{seed}"
        units.setdefault(key, []).append(float(values[index]))
        unit_interventions.setdefault(key, []).append(bool(interventions[index]))
    expected = {
        f"{scenario}|{seed}" for scenario in scenarios for seed in SOURCE_SEEDS
    }
    if set(units) != expected:
        raise ValueError("V143 evaluation lacks a scenario-seed unit")
    unit_means = {key: float(np.mean(rows)) for key, rows in units.items()}
    seed_values = {
        str(seed): float(
            np.mean([unit_means[f"{scenario}|{seed}"] for scenario in scenarios])
        )
        for seed in SOURCE_SEEDS
    }
    return {
        "mean": float(np.mean(list(seed_values.values()))),
        "seed_values": seed_values,
        "scenario_seed_values": unit_means,
        "scenario_seed_group_counts": {
            key: len(rows) for key, rows in units.items()
        },
        "scenario_seed_intervention_fraction": {
            key: float(np.mean(rows)) for key, rows in unit_interventions.items()
        },
    }


def _reference_policy_score(references: np.ndarray) -> np.ndarray:
    """Return scores that select the frozen reference row in every action group."""
    return np.where(np.asarray(references, dtype=bool), 0.0, 1.0)


def _label_free_intervention_diagnostic(
    score: np.ndarray,
    policy_rows: Sequence[np.ndarray],
    references: np.ndarray,
) -> dict[str, Any]:
    """Summarize departures from the reference without reading outcome labels."""

    values = np.asarray(score, dtype=float)
    reference_mask = np.asarray(references, dtype=bool)
    if values.shape != reference_mask.shape or not np.all(np.isfinite(values)):
        raise ValueError("selector score/reference rows are invalid")
    interventions = []
    for raw_rows in policy_rows:
        rows = np.asarray(raw_rows, dtype=int)
        if rows.ndim != 1 or rows.size < 1:
            raise ValueError("selector action group is empty")
        group_references = rows[reference_mask[rows]]
        if group_references.size != 1:
            raise ValueError("selector action group lacks one reference row")
        chosen = int(rows[int(np.argmin(values[rows]))])
        interventions.append(chosen != int(group_references[0]))
    if not interventions:
        raise ValueError("selector has no action groups")
    return {
        "protocol": "adaptation-action-reference-intervention-v1",
        "information_boundary": (
            "fitted_b25_model_predictions_and_phase_pressure_reference_only"
        ),
        "group_count": len(interventions),
        "intervention_fraction": float(np.mean(interventions)),
    }


def _label_free_policy_structure(
    contrast: Any,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    groups = np.asarray(action_group_ids(contrast), dtype=str)
    references = np.asarray(contrast.metadata.get("is_reference", ()), dtype=bool)
    if groups.shape != (contrast.size,) or references.shape != (contrast.size,):
        raise ValueError("selector action-group metadata is not row aligned")
    unique_groups = tuple(dict.fromkeys(groups.tolist()))
    rows = tuple(np.flatnonzero(groups == group) for group in unique_groups)
    if any(np.count_nonzero(references[group_rows]) != 1 for group_rows in rows):
        raise ValueError("selector action group lacks exactly one reference")
    return rows, references


def _crossfit_adaptation_folds(
    target_dataset: Any,
    *,
    selected_groups: Sequence[str],
    fold_count: int,
) -> tuple[dict[str, Any], ...]:
    ordered = tuple(sorted(str(group) for group in selected_groups))
    if fold_count < 2 or len(ordered) < fold_count or len(ordered) % fold_count:
        raise ValueError("V146 adaptation groups do not form equal cross-fit folds")
    folds = []
    for fold_index in range(fold_count):
        heldout_groups = tuple(ordered[fold_index::fold_count])
        heldout_set = set(heldout_groups)
        training_groups = tuple(group for group in ordered if group not in heldout_set)
        training_dataset = _group_subset_v3(
            target_dataset,
            selected_groups=set(training_groups),
            metadata_updates={
                "target_data_role": f"crossfit_adaptation_train_fold_{fold_index}"
            },
        )
        training_contrast = build_action_contrast_dataset(
            training_dataset,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES_V3,
        )
        heldout_dataset = _group_subset_v3(
            target_dataset,
            selected_groups=heldout_set,
            metadata_updates={
                "target_data_role": f"v146_adaptation_oof_fold_{fold_index}"
            },
        )
        contrast = build_action_contrast_dataset(
            heldout_dataset,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES_V3,
        )
        policy_rows, references = _label_free_policy_structure(contrast)
        if len(policy_rows) != len(heldout_groups):
            raise ValueError("V146 held-out fold action-group coverage changed")
        folds.append(
            {
                "fold_index": int(fold_index),
                "training_groups": training_groups,
                "heldout_groups": heldout_groups,
                "training_contrast": training_contrast,
                "contrast": contrast,
                "policy_rows": policy_rows,
                "references": references,
            }
        )
    observed = [
        group for fold in folds for group in fold["heldout_groups"]
    ]
    if sorted(observed) != list(ordered) or len(set(observed)) != len(ordered):
        raise ValueError("V146 cross-fit folds do not partition B25")
    return tuple(folds)


def _group_equal_mechanism_losses(
    contrast: Any,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    policy_rows: Sequence[np.ndarray],
) -> dict[str, float]:
    losses: dict[str, float] = {}
    for definition in MECHANISM_COMPATIBILITY_DEFINITIONS:
        values = np.asarray(
            predictions[definition.name]["mean"], dtype=float
        ).reshape(-1)
        actual = np.asarray(
            contrast.targets[definition.target_name], dtype=float
        ).reshape(-1)
        if (
            values.shape != (contrast.size,)
            or actual.shape != (contrast.size,)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(actual))
        ):
            raise ValueError(
                f"V148 invalid mechanism prediction: {definition.name}"
            )
        group_losses = []
        for raw_rows in policy_rows:
            rows = np.asarray(raw_rows, dtype=int)
            if rows.ndim != 1 or rows.size < 2:
                raise ValueError("V148 mechanism loss has an invalid action group")
            group_losses.append(float(np.mean((values[rows] - actual[rows]) ** 2)))
        if not group_losses:
            raise ValueError("V148 mechanism loss has no action groups")
        losses[definition.name] = float(np.mean(group_losses))
    if set(losses) != set(MECHANISM_COMPATIBILITY_NAMES):
        raise ValueError("V148 mechanism loss coverage changed")
    return losses


def _mechanism_compatibility_fit_worker(
    task: tuple[str, str | None, int],
) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("V148 fit state is missing")
    from threadpoolctl import threadpool_limits

    role, source_group, fold_index = task
    state = _FIT_STATE
    fold = state["adaptation_compatibility_folds"][fold_index]
    started = time.monotonic()
    with threadpool_limits(limits=1):
        target_training = _relabel_dataset_domain(
            fold["training_contrast"],
            str(state["target_city"]),
        )
        if role == "target_only":
            training = target_training
            training_domains = (str(state["target_city"]),)
        elif role == "source":
            if source_group not in state["source_mechanism_contrasts"]:
                raise ValueError("V148 mechanism task has an unknown source")
            training = merge_counterfactual_datasets_v3(
                (
                    state["source_mechanism_contrasts"][str(source_group)],
                    target_training,
                )
            )
            training_domains = tuple(
                sorted({str(source_group), str(state["target_city"])})
            )
        else:
            raise ValueError(f"unknown V148 mechanism fit role: {role!r}")
        observed_domains = tuple(
            sorted(str(value) for value in np.unique(training.domains))
        )
        if observed_domains != training_domains:
            raise ValueError("V148 mechanism fit domain boundary changed")
        model = FixedParentResidualWorldModel(
            MECHANISM_COMPATIBILITY_DEFINITIONS,
            parent_variant="physical",
            config=MechanismFitConfig(
                latent_ranks=(0,),
                adaptation_shrinkages=(0.0,),
            ),
        )
        fit_diagnostics = model.fit(training)
        predictions = model.predict(fold["contrast"])
        losses = _group_equal_mechanism_losses(
            fold["contrast"], predictions, fold["policy_rows"]
        )
    return {
        "role": role,
        "source_group": source_group,
        "fold_index": int(fold_index),
        "losses": losses,
        "training_domains": list(training_domains),
        "training_group_count": len(fold["training_groups"]),
        "heldout_group_count": len(fold["heldout_groups"]),
        "training_row_count": int(training.size),
        "fit": fit_diagnostics,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _mechanism_oof_compatibility_diagnostic(
    *,
    target_losses: Mapping[int, Mapping[str, float]],
    source_losses: Mapping[int, Mapping[str, float]],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_folds = set(range(len(folds)))
    if set(target_losses) != expected_folds or set(source_losses) != expected_folds:
        raise ValueError("V148 mechanism OOF loss coverage is incomplete")
    target_mean: dict[str, float] = {}
    source_mean: dict[str, float] = {}
    trusts: dict[str, float] = {}
    fold_rows = []
    for definition in MECHANISM_COMPATIBILITY_DEFINITIONS:
        name = definition.name
        target_values = np.asarray(
            [target_losses[index][name] for index in sorted(expected_folds)],
            dtype=float,
        )
        source_values = np.asarray(
            [source_losses[index][name] for index in sorted(expected_folds)],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(target_values))
            or not np.all(np.isfinite(source_values))
            or np.any(target_values < 0.0)
            or np.any(source_values < 0.0)
        ):
            raise ValueError(f"V148 invalid OOF losses for {name}")
        target_mean[name] = float(np.mean(target_values))
        source_mean[name] = float(np.mean(source_values))
        trusts[name] = float(
            np.clip(
                (target_mean[name] - source_mean[name])
                / max(target_mean[name], 1e-12),
                0.0,
                1.0,
            )
        )
    for fold_index in sorted(expected_folds):
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "target_only_losses": {
                    name: float(target_losses[fold_index][name])
                    for name in MECHANISM_COMPATIBILITY_NAMES
                },
                "source_plus_target_losses": {
                    name: float(source_losses[fold_index][name])
                    for name in MECHANISM_COMPATIBILITY_NAMES
                },
            }
        )
    return {
        "protocol": "five-fold-oof-labeled-mechanism-compatibility-v1",
        "information_boundary": (
            "b20_fixed_parent_mechanism_fits_scored_on_disjoint_b5_"
            "next_state_labels_no_evaluation_groups"
        ),
        "fold_count": len(folds),
        "group_count": sum(len(fold["heldout_groups"]) for fold in folds),
        "labels_from_scored_groups_used": True,
        "estimator": "fixed-parent-rank0-domain-balanced-residual-v1",
        "mechanism_names": list(MECHANISM_COMPATIBILITY_NAMES),
        "target_only_losses": target_mean,
        "source_plus_target_losses": source_mean,
        "mechanism_trust": trusts,
        "source_mass": float(np.mean(list(trusts.values()))),
        "fold_losses": fold_rows,
    }


def _label_free_oof_compatibility_diagnostic(
    *,
    target_scores: Mapping[int, np.ndarray],
    source_scores: Mapping[int, np.ndarray],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    agreements = []
    chance_agreements = []
    target_regrets = []
    target_margins = []
    target_interventions = []
    source_interventions = []
    pairwise_agreements = []
    for fold in folds:
        fold_index = int(fold["fold_index"])
        target = np.asarray(target_scores[fold_index], dtype=float)
        source = np.asarray(source_scores[fold_index], dtype=float)
        references = np.asarray(fold["references"], dtype=bool)
        if (
            target.shape != source.shape
            or target.shape != references.shape
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(source))
        ):
            raise ValueError("V146 OOF score arrays are not row aligned")
        for raw_rows in fold["policy_rows"]:
            rows = np.asarray(raw_rows, dtype=int)
            if rows.size < 2 or np.count_nonzero(references[rows]) != 1:
                raise ValueError("V146 OOF action group is malformed")
            target_values = target[rows]
            source_values = source[rows]
            target_local = int(np.argmin(target_values))
            source_local = int(np.argmin(source_values))
            reference_local = int(np.flatnonzero(references[rows])[0])
            agreements.append(target_local == source_local)
            chance_agreements.append(1.0 / float(rows.size))
            target_interventions.append(target_local != reference_local)
            source_interventions.append(source_local != reference_local)
            target_span = float(np.ptp(target_values))
            scale = max(1.0, float(np.max(np.abs(target_values))))
            if target_span <= np.finfo(float).eps * scale:
                target_regrets.append(0.0)
                target_margins.append(0.0)
            else:
                target_regrets.append(
                    float(
                        (target_values[source_local] - target_values[target_local])
                        / target_span
                    )
                )
                ordered_values = np.sort(target_values)
                target_margins.append(
                    float((ordered_values[1] - ordered_values[0]) / target_span)
                )
            for left in range(rows.size):
                for right in range(left + 1, rows.size):
                    target_sign = float(
                        np.sign(target_values[left] - target_values[right])
                    )
                    source_sign = float(
                        np.sign(source_values[left] - source_values[right])
                    )
                    if target_sign == source_sign:
                        pairwise_agreements.append(1.0)
                    elif target_sign == 0.0 or source_sign == 0.0:
                        pairwise_agreements.append(0.5)
                    else:
                        pairwise_agreements.append(0.0)
    if not agreements or not pairwise_agreements:
        raise ValueError("V146 OOF compatibility has no action groups")
    observed_agreement = float(np.mean(agreements))
    chance_agreement = float(np.mean(chance_agreements))
    trust_mass = float(
        np.clip(
            (observed_agreement - chance_agreement)
            / max(1.0 - chance_agreement, np.finfo(float).eps),
            0.0,
            1.0,
        )
    )
    return {
        "protocol": "five-fold-oof-policy-compatibility-v1",
        "information_boundary": (
            "b20_model_predictions_on_disjoint_b5_features_and_phase_pressure_"
            "reference_only"
        ),
        "fold_count": len(folds),
        "group_count": len(agreements),
        "labels_from_scored_groups_used": False,
        "action_agreement_fraction": observed_agreement,
        "chance_action_agreement": chance_agreement,
        "chance_corrected_trust_mass": trust_mass,
        "pairwise_rank_agreement_fraction": float(np.mean(pairwise_agreements)),
        "mean_normalized_target_regret": float(np.mean(target_regrets)),
        "q90_normalized_target_regret": float(np.quantile(target_regrets, 0.90)),
        "mean_normalized_target_margin": float(np.mean(target_margins)),
        "target_intervention_fraction": float(np.mean(target_interventions)),
        "source_intervention_fraction": float(np.mean(source_interventions)),
    }


def _paired_seed_summary(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, Any]:
    seeds = tuple(str(seed) for seed in SOURCE_SEEDS)
    if set(candidate) != set(seeds) or set(reference) != set(seeds):
        raise ValueError("V143 paired seed inventory changed")
    differences = [float(candidate[seed] - reference[seed]) for seed in seeds]
    return {**_paired_bootstrap(differences), "seed_values": differences}


def run_multicity_uniform_source_target(
    *,
    target_city: str,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    expected_source_cache_audit_sha256: str,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int,
    include_adaptation_selector_diagnostics: bool = False,
    adaptation_compatibility_fold_count: int = 0,
    adaptation_mechanism_fold_count: int = 0,
    result_protocol: str = RESULT_PROTOCOL,
    scientific_status: str = "seven-city-complete-holdout-development-target",
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    city = str(target_city)
    if city not in EXPECTED_CITY_GROUPS:
        raise ValueError(f"V143 unknown target city: {city}")
    if _sha256(source_cache_audit_path) != expected_source_cache_audit_sha256:
        raise ValueError("V143 source cache audit identity changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    protocol, source_bank, scenarios_by_group, source_bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    if tuple(sorted(scenarios_by_group)) != EXPECTED_CITY_GROUPS:
        raise ValueError("V143 seven-city source inventory changed")
    city_covariate_signatures = {
        group: _city_covariate_signature(source_bank, scenarios_by_group[group])
        for group in EXPECTED_CITY_GROUPS
    }
    target_scenarios = tuple(scenarios_by_group[city])
    target_dataset = _merge_city_datasets(source_bank, target_scenarios, city=city)
    selected_groups, selection_audit = select_city_budget_groups(
        target_dataset,
        city=city,
        scenarios=target_scenarios,
    )
    all_groups = set(str(value) for value in target_dataset.metadata["action_group_ids"])
    evaluation_groups = all_groups - set(selected_groups)
    if not evaluation_groups or evaluation_groups & set(selected_groups):
        raise ValueError("V143 target adaptation/evaluation split failed")
    evaluation_dataset = _group_subset_v3(
        target_dataset,
        selected_groups=evaluation_groups,
        metadata_updates={"target_data_role": "v143_multicity_evaluation_only"},
    )
    evaluation_contrast = build_action_contrast_dataset(
        evaluation_dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    adaptation_selector_contrast = None
    adaptation_selector_policy_rows = None
    adaptation_selector_references = None
    if include_adaptation_selector_diagnostics:
        adaptation_dataset = _group_subset_v3(
            target_dataset,
            selected_groups=set(selected_groups),
            metadata_updates={
                "target_data_role": "v145_adaptation_selector_only"
            },
        )
        adaptation_selector_contrast = build_action_contrast_dataset(
            adaptation_dataset,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES_V3,
        )
        (
            adaptation_selector_policy_rows,
            adaptation_selector_references,
        ) = _label_free_policy_structure(adaptation_selector_contrast)
    compatibility_folds: tuple[dict[str, Any], ...] = ()
    requested_fold_counts = {
        int(value)
        for value in (
            adaptation_compatibility_fold_count,
            adaptation_mechanism_fold_count,
        )
        if int(value) > 0
    }
    if len(requested_fold_counts) > 1:
        raise ValueError("adaptation compatibility folds use inconsistent counts")
    if requested_fold_counts:
        compatibility_folds = _crossfit_adaptation_folds(
            target_dataset,
            selected_groups=selected_groups,
            fold_count=next(iter(requested_fold_counts)),
        )
    actual, unique_groups, group_rows, references, _ = _normalized_pressure_targets(
        evaluation_dataset, evaluation_contrast
    )
    policy_rows = tuple(np.asarray(rows, dtype=int) for rows in group_rows)
    contrast_groups = np.asarray(action_group_ids(evaluation_contrast), dtype=str)
    target_context = np.mean(
        np.vstack(
            [_static_context_from_dataset(source_bank[name]) for name in target_scenarios]
        ),
        axis=0,
    )
    source_order = tuple(group for group in EXPECTED_CITY_GROUPS if group != city)
    source_mechanism_contrasts: dict[str, Any] = {}
    if adaptation_mechanism_fold_count:
        for source_group in source_order:
            source_city_dataset = _merge_city_datasets(
                source_bank,
                tuple(scenarios_by_group[source_group]),
                city=source_group,
            )
            source_city_dataset = _relabel_dataset_domain(
                source_city_dataset,
                source_group,
            )
            source_mechanism_contrasts[source_group] = (
                build_action_contrast_dataset(
                    source_city_dataset,
                    reference_policy="phase_pressure",
                    contrast_features=CONTRAST_FEATURES_V3,
                )
            )
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    target_key = f"v143_target_{city}_B{TARGET_BUDGET}"
    source_states = {}
    h2oplus_fit_bank = {target_key: target_dataset}
    h2oplus_city_groups = {target_key: city}
    for source_group in source_order:
        scenarios = tuple(scenarios_by_group[source_group])
        fit_bank = {name: source_bank[name] for name in scenarios}
        fit_bank[target_key] = target_dataset
        city_groups = {name: source_group for name in scenarios}
        city_groups[target_key] = city
        source_states[source_group] = {
            "fit_bank": fit_bank,
            "target_key": target_key,
            "city_groups": city_groups,
            "target_static_context": target_context,
            "prior_policy": "phase_pressure",
            "source_rule_specs": source_rule_specs,
            "city": city,
        }
        for name in scenarios:
            h2oplus_fit_bank[name] = source_bank[name]
            h2oplus_city_groups[name] = source_group
    if set(h2oplus_fit_bank) & set(target_scenarios):
        raise ValueError("V143 target-city scenario leaked into the source bank")

    from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
        ANCHOR_FAMILY,
        CORRECTION_FAMILY,
    )

    tasks = [("target_only", None), ("h2oplus", None)] + [
        ("source", source) for source in source_order
    ]
    global _FIT_STATE
    _FIT_STATE = {
        "target_city": city,
        "target_dataset": target_dataset,
        "target_key": target_key,
        "selected_groups": selected_groups,
        "target_static_context": target_context,
        "evaluation_contrast": evaluation_contrast,
        "candidate": str(fit_result["frozen_blend_candidate"]),
        "source_order": source_order,
        "source_states": source_states,
        "h2oplus_fit_bank": h2oplus_fit_bank,
        "h2oplus_city_groups": h2oplus_city_groups,
        "source_rule_specs": source_rule_specs,
        "anchor_family": ANCHOR_FAMILY,
        "correction_family": CORRECTION_FAMILY,
        "adaptation_selector_contrast": adaptation_selector_contrast,
        "adaptation_selector_policy_rows": adaptation_selector_policy_rows,
        "adaptation_selector_references": adaptation_selector_references,
        "adaptation_compatibility_folds": compatibility_folds,
        "source_mechanism_contrasts": source_mechanism_contrasts,
    }
    actual_workers = min(max(1, int(fit_workers)), len(tasks))
    compatibility_rows: list[dict[str, Any]] = []
    mechanism_compatibility_rows: list[dict[str, Any]] = []
    try:
        if actual_workers == 1:
            fitted_rows = [_fit_worker(task) for task in tasks]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("V143 parallel fit requires Linux fork")
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                fitted_rows = list(pool.map(_fit_worker, tasks))
        if compatibility_folds and adaptation_compatibility_fold_count:
            compatibility_tasks = [
                (role, source, fold_index)
                for fold_index in range(len(compatibility_folds))
                for role, source in (
                    [("target_only", None)]
                    + [("source", source) for source in source_order]
                )
            ]
            compatibility_workers = min(
                max(1, int(fit_workers)), len(compatibility_tasks)
            )
            if compatibility_workers == 1:
                compatibility_rows = [
                    _compatibility_fit_worker(task)
                    for task in compatibility_tasks
                ]
            else:
                with ProcessPoolExecutor(
                    max_workers=compatibility_workers,
                    mp_context=mp.get_context("fork"),
                ) as pool:
                    compatibility_rows = list(
                        pool.map(_compatibility_fit_worker, compatibility_tasks)
                    )
        if compatibility_folds and adaptation_mechanism_fold_count:
            mechanism_tasks = [
                (role, source, fold_index)
                for fold_index in range(len(compatibility_folds))
                for role, source in (
                    [("target_only", None)]
                    + [("source", source) for source in source_order]
                )
            ]
            mechanism_workers = min(
                max(1, int(fit_workers)), len(mechanism_tasks)
            )
            if mechanism_workers == 1:
                mechanism_compatibility_rows = [
                    _mechanism_compatibility_fit_worker(task)
                    for task in mechanism_tasks
                ]
            else:
                with ProcessPoolExecutor(
                    max_workers=mechanism_workers,
                    mp_context=mp.get_context("fork"),
                ) as pool:
                    mechanism_compatibility_rows = list(
                        pool.map(
                            _mechanism_compatibility_fit_worker,
                            mechanism_tasks,
                        )
                    )
    finally:
        _FIT_STATE = None

    target_score = None
    h2oplus_score = None
    source_scores = {}
    adaptation_selector_diagnostics = {}
    fit_diagnostics = []
    for row in fitted_rows:
        score = np.asarray(row.pop("score"), dtype=float)
        selector_diagnostic = row.pop("adaptation_selector")
        fit_diagnostics.append(row)
        selector_key = (
            str(row["source_group"])
            if row["role"] == "source"
            else str(row["role"])
        )
        if selector_diagnostic is not None:
            adaptation_selector_diagnostics[selector_key] = selector_diagnostic
        if row["role"] == "target_only":
            target_score = score
        elif row["role"] == "h2oplus":
            h2oplus_score = score
        else:
            source_scores[str(row["source_group"])] = score
    if target_score is None or h2oplus_score is None:
        raise ValueError("V143 target-only or H2O+ prediction is missing")
    if set(source_scores) != set(source_order):
        raise ValueError("V143 one-source prediction coverage is incomplete")
    compatibility_diagnostics: dict[str, dict[str, Any]] = {}
    compatibility_fit_diagnostics = []
    if compatibility_folds and adaptation_compatibility_fold_count:
        target_oof_scores: dict[int, np.ndarray] = {}
        source_oof_scores: dict[str, dict[int, np.ndarray]] = {
            source: {} for source in source_order
        }
        for row in compatibility_rows:
            score = np.asarray(row.pop("score"), dtype=float)
            compatibility_fit_diagnostics.append(row)
            fold_index = int(row["fold_index"])
            if row["role"] == "target_only":
                if fold_index in target_oof_scores:
                    raise ValueError("V146 duplicate target OOF fold")
                target_oof_scores[fold_index] = score
            elif row["role"] == "source":
                source = str(row["source_group"])
                if source not in source_oof_scores or fold_index in source_oof_scores[source]:
                    raise ValueError("V146 duplicate or unknown source OOF fold")
                source_oof_scores[source][fold_index] = score
            else:
                raise ValueError("V146 OOF fit role changed")
        expected_folds = set(range(len(compatibility_folds)))
        if set(target_oof_scores) != expected_folds or any(
            set(values) != expected_folds for values in source_oof_scores.values()
        ):
            raise ValueError("V146 OOF prediction coverage is incomplete")
        compatibility_diagnostics = {
            source: _label_free_oof_compatibility_diagnostic(
                target_scores=target_oof_scores,
                source_scores=source_oof_scores[source],
                folds=compatibility_folds,
            )
            for source in source_order
        }
    mechanism_compatibility_diagnostics: dict[str, dict[str, Any]] = {}
    mechanism_compatibility_fit_diagnostics = []
    if compatibility_folds and adaptation_mechanism_fold_count:
        target_oof_losses: dict[int, dict[str, float]] = {}
        source_oof_losses: dict[str, dict[int, dict[str, float]]] = {
            source: {} for source in source_order
        }
        for row in mechanism_compatibility_rows:
            losses = {
                str(name): float(value)
                for name, value in dict(row.pop("losses")).items()
            }
            mechanism_compatibility_fit_diagnostics.append(row)
            fold_index = int(row["fold_index"])
            if row["role"] == "target_only":
                if fold_index in target_oof_losses:
                    raise ValueError("V148 duplicate target mechanism OOF fold")
                target_oof_losses[fold_index] = losses
            elif row["role"] == "source":
                source = str(row["source_group"])
                if (
                    source not in source_oof_losses
                    or fold_index in source_oof_losses[source]
                ):
                    raise ValueError("V148 duplicate or unknown source OOF fold")
                source_oof_losses[source][fold_index] = losses
            else:
                raise ValueError("V148 mechanism OOF fit role changed")
        expected_folds = set(range(len(compatibility_folds)))
        if set(target_oof_losses) != expected_folds or any(
            set(values) != expected_folds
            for values in source_oof_losses.values()
        ):
            raise ValueError("V148 mechanism OOF prediction coverage is incomplete")
        mechanism_compatibility_diagnostics = {
            source: _mechanism_oof_compatibility_diagnostic(
                target_losses=target_oof_losses,
                source_losses=source_oof_losses[source],
                folds=compatibility_folds,
            )
            for source in source_order
        }
    source_matrix = np.column_stack([source_scores[name] for name in source_order])
    uniform_score = np.mean(source_matrix, axis=1)
    residuals = source_matrix - target_score[:, None]
    pressure_index = evaluation_contrast.feature_names.index("delta_service_pressure")
    rank_signal = np.asarray(
        evaluation_contrast.features[:, pressure_index], dtype=float
    )
    placebo_residuals = permute_source_residuals_by_seed_group(
        residuals,
        contrast_groups,
        references,
        rank_signal,
    )
    placebo_score = target_score + np.mean(placebo_residuals, axis=1)
    arm_scores = {
        "phase_pressure": _reference_policy_score(references),
        "target_only_causal": target_score,
        "pooled_h2oplus": h2oplus_score,
        "uniform_cfcmt": uniform_score,
        "matched_source_placebo": placebo_score,
    }
    arm_summaries = {}
    for arm, score in arm_scores.items():
        summary = _scenario_seed_policy_summary(
            actual=actual,
            score=score,
            policy_rows=policy_rows,
            references=references,
            unique_groups=unique_groups,
            scenarios=target_scenarios,
        )
        summary["policy"] = _policy_metrics(actual, score, policy_rows, references)
        summary["predictive"] = _predictive_metrics(
            actual, score, contrast_groups, references
        )
        arm_summaries[arm] = summary
    source_arm_summaries = {}
    source_placebo_arm_summaries = {}
    compatibility_shrunk_source_arm_summaries = {}
    compatibility_shrunk_placebo_arm_summaries = {}
    mechanism_shrunk_source_arm_summaries = {}
    mechanism_shrunk_placebo_arm_summaries = {}
    for source_index, source_group in enumerate(source_order):
        source_score = source_scores[source_group]
        source_placebo_score = target_score + placebo_residuals[:, source_index]
        output_scores = [
            (source_arm_summaries, source_score),
            (source_placebo_arm_summaries, source_placebo_score),
        ]
        if compatibility_folds and adaptation_compatibility_fold_count:
            trust_mass = float(
                compatibility_diagnostics[source_group][
                    "chance_corrected_trust_mass"
                ]
            )
            output_scores.extend(
                [
                    (
                        compatibility_shrunk_source_arm_summaries,
                        target_score + trust_mass * (source_score - target_score),
                    ),
                    (
                        compatibility_shrunk_placebo_arm_summaries,
                        target_score + trust_mass * placebo_residuals[:, source_index],
                    ),
                ]
            )
        if compatibility_folds and adaptation_mechanism_fold_count:
            mechanism_mass = float(
                mechanism_compatibility_diagnostics[source_group]["source_mass"]
            )
            output_scores.extend(
                [
                    (
                        mechanism_shrunk_source_arm_summaries,
                        target_score
                        + mechanism_mass * (source_score - target_score),
                    ),
                    (
                        mechanism_shrunk_placebo_arm_summaries,
                        target_score
                        + mechanism_mass * placebo_residuals[:, source_index],
                    ),
                ]
            )
        for output, score in output_scores:
            summary = _scenario_seed_policy_summary(
                actual=actual,
                score=score,
                policy_rows=policy_rows,
                references=references,
                unique_groups=unique_groups,
                scenarios=target_scenarios,
            )
            summary["policy"] = _policy_metrics(
                actual, score, policy_rows, references
            )
            summary["predictive"] = _predictive_metrics(
                actual, score, contrast_groups, references
            )
            output[source_group] = summary
    paired_effects = {
        "uniform_cfcmt_minus_target_only": _paired_seed_summary(
            arm_summaries["uniform_cfcmt"]["seed_values"],
            arm_summaries["target_only_causal"]["seed_values"],
        ),
        "uniform_cfcmt_minus_pooled_h2oplus": _paired_seed_summary(
            arm_summaries["uniform_cfcmt"]["seed_values"],
            arm_summaries["pooled_h2oplus"]["seed_values"],
        ),
        "uniform_cfcmt_minus_matched_placebo": _paired_seed_summary(
            arm_summaries["uniform_cfcmt"]["seed_values"],
            arm_summaries["matched_source_placebo"]["seed_values"],
        ),
    }
    result = {
        "protocol": str(result_protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": str(scientific_status),
        "target_city": city,
        "target_scenarios": list(target_scenarios),
        "heldout_target_city_scenarios": list(target_scenarios),
        "source_city_groups": list(source_order),
        "source_scenarios_by_group": {
            group: list(scenarios_by_group[group]) for group in source_order
        },
        "target_budget": TARGET_BUDGET,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": str(protocol["target_name"]),
        "information_budget": {
            "unique_target_action_groups": len(selected_groups),
            "target_only_groups": len(selected_groups),
            "pooled_h2oplus_target_groups": len(selected_groups),
            "uniform_cfcmt_target_groups": len(selected_groups),
            "evaluation_groups_used_for_fit_or_selection": False,
            "adaptation_groups_used_for_selector_diagnostics": bool(
                include_adaptation_selector_diagnostics or compatibility_folds
            ),
            "adaptation_compatibility_fold_count": (
                len(compatibility_folds)
                if adaptation_compatibility_fold_count
                else 0
            ),
            "adaptation_compatibility_scored_group_labels_used": False,
            "adaptation_mechanism_fold_count": (
                len(compatibility_folds)
                if adaptation_mechanism_fold_count
                else 0
            ),
            "adaptation_mechanism_scored_group_labels_used": bool(
                adaptation_mechanism_fold_count
            ),
            "adaptation_group_ids": list(selected_groups),
        },
        "selection_audit": selection_audit,
        "evaluation_group_count": len(unique_groups),
        "evaluation_row_count": int(evaluation_contrast.size),
        "uniform_source_weights": {
            group: 1.0 / len(source_order) for group in source_order
        },
        "arm_summaries": arm_summaries,
        "source_arm_summaries": source_arm_summaries,
        "source_placebo_arm_summaries": source_placebo_arm_summaries,
        "city_covariate_signatures": city_covariate_signatures,
        "paired_effects": paired_effects,
        "fit_diagnostics": fit_diagnostics,
        "input_audits": {"source_bank": source_bank_audit},
        "inputs": {
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": expected_source_cache_audit_sha256,
        },
        "claim_boundary": claim_boundary
        or (
            "V143 is seven-city development using complete-city source exclusion "
            "and B25 target-simulator adaptation. It is not untouched-city or "
            "real-world closed-loop confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }
    if include_adaptation_selector_diagnostics:
        expected_selector_keys = {"target_only", "h2oplus", *source_order}
        if set(adaptation_selector_diagnostics) != expected_selector_keys:
            raise ValueError("adaptation selector diagnostic coverage is incomplete")
        result["adaptation_selector_diagnostics"] = {
            "protocol": "b25-adaptation-only-label-free-policy-diagnostics-v1",
            "evaluation_features_or_labels_used": False,
            "adaptation_group_count": len(selected_groups),
            "arms": adaptation_selector_diagnostics,
        }
    if compatibility_folds and adaptation_compatibility_fold_count:
        if set(compatibility_diagnostics) != set(source_order):
            raise ValueError("V146 compatibility diagnostic coverage is incomplete")
        result["adaptation_compatibility_diagnostics"] = {
            "protocol": "b25-five-fold-oof-policy-compatibility-v1",
            "evaluation_features_or_labels_used": False,
            "scored_adaptation_labels_used": False,
            "adaptation_group_count": len(selected_groups),
            "fold_count": len(compatibility_folds),
            "folds": [
                {
                    "fold_index": int(fold["fold_index"]),
                    "training_group_count": len(fold["training_groups"]),
                    "heldout_group_count": len(fold["heldout_groups"]),
                    "heldout_group_ids": list(fold["heldout_groups"]),
                }
                for fold in compatibility_folds
            ],
            "source_arms": compatibility_diagnostics,
        }
        result["compatibility_shrunk_source_arm_summaries"] = (
            compatibility_shrunk_source_arm_summaries
        )
        result["compatibility_shrunk_placebo_arm_summaries"] = (
            compatibility_shrunk_placebo_arm_summaries
        )
        result["compatibility_fit_diagnostics"] = compatibility_fit_diagnostics
    if compatibility_folds and adaptation_mechanism_fold_count:
        if set(mechanism_compatibility_diagnostics) != set(source_order):
            raise ValueError("V148 mechanism diagnostic coverage is incomplete")
        result["adaptation_mechanism_diagnostics"] = {
            "protocol": "b25-five-fold-oof-labeled-mechanism-compatibility-v1",
            "evaluation_features_or_labels_used": False,
            "scored_adaptation_labels_used": True,
            "adaptation_group_count": len(selected_groups),
            "fold_count": len(compatibility_folds),
            "mechanism_names": list(MECHANISM_COMPATIBILITY_NAMES),
            "folds": [
                {
                    "fold_index": int(fold["fold_index"]),
                    "training_group_count": len(fold["training_groups"]),
                    "heldout_group_count": len(fold["heldout_groups"]),
                    "heldout_group_ids": list(fold["heldout_groups"]),
                }
                for fold in compatibility_folds
            ],
            "source_arms": mechanism_compatibility_diagnostics,
        }
        result["mechanism_shrunk_source_arm_summaries"] = (
            mechanism_shrunk_source_arm_summaries
        )
        result["mechanism_shrunk_placebo_arm_summaries"] = (
            mechanism_shrunk_placebo_arm_summaries
        )
        result["mechanism_compatibility_fit_diagnostics"] = (
            mechanism_compatibility_fit_diagnostics
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-city", required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--source-cache-audit-sha256", required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V143 result: {args.out}")
    result = run_multicity_uniform_source_target(
        target_city=args.target_city,
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        expected_source_cache_audit_sha256=args.source_cache_audit_sha256,
        conversion_root=args.conversion_root,
        cache_workers=max(1, int(args.cache_workers)),
        fit_workers=max(1, int(args.fit_workers)),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "DONE",
                "protocol": result["protocol"],
                "target_city": result["target_city"],
                "evaluation_group_count": result["evaluation_group_count"],
                "paired_effects": result["paired_effects"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
