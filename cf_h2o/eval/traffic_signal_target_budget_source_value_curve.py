"""Measure causal source value across target-data budgets on one fixed selector."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic import (
    _policy_metrics,
    _predictive_metrics,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    SELECTION_SEED,
    _atomic_bytes,
    _balanced_quotas,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _fit_family_v3,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _coverage_first_group_order_v3,
    _group_seed_v3,
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import (
    _selector_dataset,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (
    _source_inputs,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _normalized_pressure_targets,
    _target_cache_audit_contract,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    _assert_waiting_aligned_bank,
    _fit_component_budget,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


RESULT_PROTOCOL = "tsc-v123-target-budget-source-value-curve-v1"
ARTIFACT_PROTOCOL = "cfcmt-target-budget-source-predictions-v1"
TARGET_BUDGETS = (25, 50, 100, 250, 500, 1000)
SOURCE_BLEND_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260901
_CURVE_STATE: Mapping[str, Any] | None = None


def select_city_adaptation_groups_for_budget(
    bank: Mapping[str, Any],
    *,
    city: str,
    scenarios: Sequence[str],
    budget: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Generalize the frozen V115 coverage-first B100 order to nested budgets."""

    if int(budget) <= 0:
        raise ValueError("target adaptation budget must be positive")
    scenario_names = tuple(str(value) for value in scenarios)
    quotas = _balanced_quotas(int(budget), scenario_names)
    selected: list[str] = []
    scenario_rows = {}
    seed_totals = {int(seed): 0 for seed in ADAPTATION_SEEDS}
    for scenario_index, scenario in enumerate(scenario_names):
        dataset = bank[scenario]
        groups = sorted(
            set(str(value) for value in dataset.metadata["action_group_ids"])
        )
        if {_group_seed_v3(group) for group in groups} != set(ADAPTATION_SEEDS):
            raise ValueError(f"adaptation seed coverage changed for {scenario}")
        quota = int(quotas[scenario])
        seed_order = (
            ADAPTATION_SEEDS
            if scenario_index % 2 == 0
            else tuple(reversed(ADAPTATION_SEEDS))
        )
        per_seed = _balanced_quotas(
            quota, tuple(str(seed) for seed in seed_order)
        )
        chosen_for_scenario: list[str] = []
        for seed in ADAPTATION_SEEDS:
            candidates = [
                group for group in groups if _group_seed_v3(group) == int(seed)
            ]
            requested = int(per_seed[str(seed)])
            ordered = _coverage_first_group_order_v3(
                dataset,
                candidates,
                selection_seed=SELECTION_SEED,
                role=f"external_city_b100:{city}:{scenario}:seed{seed}",
            )
            if len(ordered) < requested:
                raise ValueError(
                    f"insufficient groups for {scenario}/seed={seed}: "
                    f"{len(ordered)} < {requested}"
                )
            chosen = ordered[:requested]
            chosen_for_scenario.extend(chosen)
            seed_totals[int(seed)] += len(chosen)
        selected.extend(chosen_for_scenario)
        scenario_rows[scenario] = {
            "available_group_count": len(groups),
            "selected_group_count": len(chosen_for_scenario),
            "selected_group_count_by_seed": {
                str(seed): sum(
                    _group_seed_v3(group) == int(seed)
                    for group in chosen_for_scenario
                )
                for seed in ADAPTATION_SEEDS
            },
        }
    if len(selected) != int(budget) or len(set(selected)) != int(budget):
        raise ValueError("target adaptation selection is incomplete or duplicated")
    if max(seed_totals.values()) - min(seed_totals.values()) > 1:
        raise ValueError("target adaptation selection is not seed balanced")
    return tuple(sorted(selected)), {
        "protocol": "city-scenario-and-seed-balanced-coverage-first-nested-v1",
        "city": str(city),
        "budget": int(budget),
        "selection_seed": SELECTION_SEED,
        "scenario_quotas": quotas,
        "selected_group_count_by_seed": {
            str(seed): seed_totals[int(seed)] for seed in ADAPTATION_SEEDS
        },
        "scenarios": scenario_rows,
    }


def validate_nested_budget_groups(
    selected: Mapping[int, Sequence[str]],
    *,
    expected_b100: Sequence[str],
) -> None:
    budgets = sorted(int(value) for value in selected)
    for lower, upper in zip(budgets, budgets[1:]):
        if not set(selected[lower]) < set(selected[upper]):
            raise ValueError(f"adaptation budgets are not strictly nested: {lower}, {upper}")
    if tuple(selected[100]) != tuple(expected_b100):
        raise ValueError("V123 B100 groups differ from the frozen V115 selection")


def _target_model(
    adaptation_dataset: Any,
    *,
    group_ids: Sequence[str],
    candidate: str,
    target_domain: str = "jinan",
) -> FrozenAnchoredBlendModel:
    training = _group_subset_v3(
        adaptation_dataset,
        selected_groups=set(str(value) for value in group_ids),
        metadata_updates={"target_data_role": "v123_budget_curve_training"},
    )
    contrast = build_action_contrast_dataset(
        training,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    models = {
        family: _fit_family_v3(
            contrast,
            family,
            MechanismFitConfig(),
            target_domain=str(target_domain),
        )[0]
        for family in BASE_FAMILIES
    }
    return FrozenAnchoredBlendModel(
        anchor_model=models[ANCHOR_FAMILY],
        correction_model=models[CORRECTION_FAMILY],
        candidate=str(candidate),
        anchor_objective_mode="control_only",
        correction_objective_mode="control_only",
    )


def _fit_curve_worker(task: tuple[str, int, str | None]) -> dict[str, Any]:
    if _CURVE_STATE is None:
        raise RuntimeError("V123 worker state is missing")
    from threadpoolctl import threadpool_limits

    arm, budget, source_group = task
    state = _CURVE_STATE
    target_groups = tuple(state["selected_groups"].get(int(budget), ()))
    started = time.monotonic()
    with threadpool_limits(limits=1):
        if arm == "target_only":
            if int(budget) <= 0 or source_group is not None:
                raise ValueError("invalid V123 target-only task")
            model = _target_model(
                state["adaptation_dataset"],
                group_ids=target_groups,
                candidate=str(state["candidate"]),
            )
            domains = ("jinan",)
        elif arm == "source_augmented":
            if source_group not in state["source_states"]:
                raise ValueError("invalid V123 source task")
            fitted = _fit_component_budget(
                state["source_states"][str(source_group)],
                target_group_ids=target_groups,
            )
            expected_domains = (
                {str(source_group)}
                if int(budget) == 0
                else {str(source_group), "jinan"}
            )
            if set(fitted.diagnostics["source_domains"]) != expected_domains:
                raise ValueError("V123 source information boundary changed")
            model = FrozenAnchoredBlendModel(
                anchor_model=fitted.family_models[ANCHOR_FAMILY],
                correction_model=fitted.family_models[CORRECTION_FAMILY],
                candidate=str(state["candidate"]),
                anchor_objective_mode=fitted.objective_modes[ANCHOR_FAMILY],
                correction_objective_mode=fitted.objective_modes[CORRECTION_FAMILY],
            )
            domains = tuple(sorted(expected_domains))
        else:
            raise ValueError(f"unknown V123 task arm: {arm}")
        score, uncertainty, trust, _ = _group_adjusted_scores(
            state["selector_contrast"],
            model.predict(state["selector_contrast"]),
            objective_mode="control_only",
        )
    return {
        "arm": str(arm),
        "budget": int(budget),
        "source_group": source_group,
        "score": np.asarray(score, dtype=float),
        "uncertainty": np.asarray(uncertainty, dtype=float),
        "trust": np.asarray(trust, dtype=float),
        "training_domains": list(domains),
        "target_group_count": len(target_groups),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _policy_arrays(
    actual: np.ndarray,
    score: np.ndarray,
    group_rows: Sequence[np.ndarray],
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(group_rows, np.ndarray) and group_rows.ndim == 2:
        rows = np.asarray(group_rows, dtype=int)
        chosen = rows[
            np.arange(rows.shape[0]), np.argmin(score[rows], axis=1)
        ]
        reference_mask = references[rows]
        if not np.all(np.sum(reference_mask, axis=1) == 1):
            raise ValueError("policy groups do not contain exactly one reference")
        reference = rows[
            np.arange(rows.shape[0]), np.argmax(reference_mask, axis=1)
        ]
    else:
        chosen = np.asarray(
            [int(rows[int(np.argmin(score[rows]))]) for rows in group_rows],
            dtype=int,
        )
        reference = np.asarray(
            [int(rows[references[rows]][0]) for rows in group_rows], dtype=int
        )
    return np.asarray(actual[chosen], dtype=float), chosen != reference


def _seed_policy_metrics(
    actual: np.ndarray,
    score: np.ndarray,
    group_rows: Sequence[np.ndarray],
    references: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[str, dict[str, float]]:
    delta, intervention = _policy_arrays(actual, score, group_rows, references)
    result = {}
    for seed in sorted(int(value) for value in np.unique(group_seeds)):
        mask = group_seeds == seed
        active = intervention[mask]
        values = delta[mask]
        result[str(seed)] = {
            "group_count": int(np.count_nonzero(mask)),
            "mean_normalized_delta_vs_phase_pressure": float(np.mean(values)),
            "intervention_fraction": float(np.mean(active)),
            "harmful_intervention_fraction": float(
                np.count_nonzero(active & (values > 0.0))
                / max(np.count_nonzero(active), 1)
            ),
        }
    return result


def nested_leave_one_seed_source_selection(
    candidate_seed_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    target_seed_metrics: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, Any]:
    if not candidate_seed_metrics:
        raise ValueError("nested source selection requires candidates")
    seeds = tuple(sorted(next(iter(candidate_seed_metrics.values())), key=int))
    if any(tuple(sorted(rows, key=int)) != seeds for rows in candidate_seed_metrics.values()):
        raise ValueError("nested source candidates do not share selector seeds")
    folds = []
    source_values = []
    target_values = []
    source_target_differences = []
    for heldout in seeds:
        training = tuple(seed for seed in seeds if seed != heldout)
        selected = min(
            candidate_seed_metrics,
            key=lambda name: (
                float(
                    np.mean(
                        [
                            candidate_seed_metrics[name][seed][
                                "mean_normalized_delta_vs_phase_pressure"
                            ]
                            for seed in training
                        ]
                    )
                ),
                name,
            ),
        )
        source_value = float(
            candidate_seed_metrics[selected][heldout][
                "mean_normalized_delta_vs_phase_pressure"
            ]
        )
        target_value = (
            None
            if target_seed_metrics is None
            else float(
                target_seed_metrics[heldout][
                    "mean_normalized_delta_vs_phase_pressure"
                ]
            )
        )
        folds.append(
            {
                "heldout_seed": int(heldout),
                "selected_candidate": selected,
                "source_policy_delta": source_value,
                "target_only_policy_delta": target_value,
                "source_minus_target": (
                    None if target_value is None else source_value - target_value
                ),
            }
        )
        source_values.append(source_value)
        if target_value is not None:
            target_values.append(target_value)
            source_target_differences.append(source_value - target_value)
    return {
        "folds": folds,
        "mean_source_policy_delta": float(np.mean(source_values)),
        "mean_target_only_policy_delta": (
            None if not target_values else float(np.mean(target_values))
        ),
        "source_minus_target_by_seed": source_target_differences,
        "mean_source_minus_target": (
            None
            if not source_target_differences
            else float(np.mean(source_target_differences))
        ),
        "selected_candidate_counts": {
            name: sum(row["selected_candidate"] == name for row in folds)
            for name in sorted(candidate_seed_metrics)
            if any(row["selected_candidate"] == name for row in folds)
        },
    }


def _paired_bootstrap(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("paired bootstrap values are invalid")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, array.size, size=(BOOTSTRAP_REPLICATES, array.size)
    )
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "improving_seed_fraction": float(np.mean(array < 0.0)),
    }


def run_target_budget_source_curve(
    *,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
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
    fit_workers: int,
    artifact_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite V123 artifact: {artifact_path}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    _target_cache_audit_contract(target_cache_audit_path, fit_result)
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
    )
    protocol, source_bank, source_scenarios, source_bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    target_manifest = load_traffic_signal_manifest(target_manifest_path)
    target_scenarios = tuple(
        name
        for name, group in target_manifest.city_groups.items()
        if group == "jinan"
    )
    target_bank, target_bank_audit = load_frozen_counterfactual_bank(
        target_cache_root,
        target_manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(cache_workers), 1),
    )
    _assert_waiting_aligned_bank(
        target_bank,
        target_name=str(protocol["target_name"]),
        tolerance=float(protocol["required_target_equivalence_tolerance"]),
    )
    adaptation = _merge_city_datasets(target_bank, target_scenarios, city="jinan")
    _assert_pure_waiting_selector_dataset(
        adaptation, target_name="prefix_mean_cost_450s"
    )
    selected_groups = {}
    selection_audits = {}
    for budget in TARGET_BUDGETS:
        groups, audit = select_city_adaptation_groups_for_budget(
            target_bank,
            city="jinan",
            scenarios=target_scenarios,
            budget=budget,
        )
        selected_groups[int(budget)] = groups
        selection_audits[str(budget)] = audit
    expected_b100 = tuple(
        str(value)
        for value in fit_result["information_budget"]["selected_group_ids"]
    )
    validate_nested_budget_groups(
        selected_groups, expected_b100=expected_b100
    )

    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
        cache_workers=cache_workers,
    )
    selector_contrast = build_action_contrast_dataset(
        selector,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual, unique_groups, group_rows, references, group_seeds = (
        _normalized_pressure_targets(selector, selector_contrast)
    )
    if set(group_seeds.tolist()) != set(int(value) for value in selector_seeds):
        raise ValueError("V123 selector seed coverage changed")
    row_groups = np.asarray(action_group_ids(selector_contrast), dtype=str)
    action_counts = {int(np.asarray(rows).size) for rows in group_rows}
    if len(action_counts) != 1:
        raise ValueError("V123 selector action count is not rectangular")
    policy_rows = np.vstack(group_rows).astype(int, copy=False)

    target_context = np.mean(
        np.vstack(
            [_static_context_from_dataset(target_bank[name]) for name in target_scenarios]
        ),
        axis=0,
    )
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    target_key = "waiting_aligned_target_jinan_v123"
    source_states = {}
    for source_group, scenarios in source_scenarios.items():
        fit_bank = {name: source_bank[name] for name in scenarios}
        fit_bank[target_key] = adaptation
        city_groups = {name: source_group for name in scenarios}
        city_groups[target_key] = "jinan"
        source_states[source_group] = {
            "fit_bank": fit_bank,
            "target_key": target_key,
            "city_groups": city_groups,
            "target_static_context": target_context,
            "prior_policy": "phase_pressure",
            "source_rule_specs": source_rule_specs,
            "city": "jinan",
        }
    candidate = str(fit_result["frozen_blend_candidate"])
    tasks = [
        ("target_only", budget, None) for budget in TARGET_BUDGETS
    ] + [
        ("source_augmented", budget, source_group)
        for budget in (0, *TARGET_BUDGETS)
        for source_group in sorted(source_states)
    ]
    global _CURVE_STATE
    _CURVE_STATE = {
        "adaptation_dataset": adaptation,
        "selected_groups": selected_groups,
        "source_states": source_states,
        "selector_contrast": selector_contrast,
        "candidate": candidate,
    }
    actual_workers = min(max(int(fit_workers), 1), len(tasks))
    try:
        if actual_workers == 1:
            fitted_rows = [_fit_curve_worker(task) for task in tasks]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("V123 parallel fit requires Linux fork")
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                fitted_rows = list(pool.map(_fit_curve_worker, tasks))
    finally:
        _CURVE_STATE = None

    target_predictions = {}
    source_predictions = {
        str(budget): {} for budget in (0, *TARGET_BUDGETS)
    }
    for row in fitted_rows:
        prediction = (
            row.pop("score"),
            row.pop("uncertainty"),
            row.pop("trust"),
        )
        budget_key = str(int(row["budget"]))
        if row["arm"] == "target_only":
            target_predictions[budget_key] = prediction
        else:
            source_predictions[budget_key][str(row["source_group"])] = prediction
    if set(target_predictions) != {str(value) for value in TARGET_BUDGETS}:
        raise ValueError("V123 target prediction coverage is incomplete")
    expected_sources = set(source_states)
    if any(set(rows) != expected_sources for rows in source_predictions.values()):
        raise ValueError("V123 source prediction coverage is incomplete")

    artifact = {
        "protocol": ARTIFACT_PROTOCOL,
        "city": "jinan",
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "target_budgets": TARGET_BUDGETS,
        "selected_group_ids": selected_groups,
        "selector_seeds": tuple(int(value) for value in selector_seeds),
        "source_group_order": tuple(sorted(source_states)),
        "target_predictions": target_predictions,
        "source_predictions": source_predictions,
        "fit_result_sha256": expected_fit_result_sha256,
        "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        artifact_path, pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    )
    round_trip = pickle.loads(artifact_path.read_bytes())
    if (
        round_trip.get("protocol") != ARTIFACT_PROTOCOL
        or tuple(round_trip.get("target_budgets", ())) != TARGET_BUDGETS
        or set(round_trip.get("source_predictions", {}))
        != {str(value) for value in (0, *TARGET_BUDGETS)}
    ):
        raise ValueError("V123 artifact round trip failed")

    budget_results = {}
    for budget in (0, *TARGET_BUDGETS):
        key = str(budget)
        target_score = (
            None if budget == 0 else np.asarray(target_predictions[key][0], dtype=float)
        )
        source_scores = {
            name: np.asarray(values[0], dtype=float)
            for name, values in source_predictions[key].items()
        }
        per_source = {
            name: {
                "predictive": _predictive_metrics(
                    actual, score, row_groups, references
                ),
                "policy": _policy_metrics(
                    actual, score, policy_rows, references
                ),
                "seed_policy": _seed_policy_metrics(
                    actual, score, policy_rows, references, group_seeds
                ),
            }
            for name, score in source_scores.items()
        }
        target_result = None
        target_seed_metrics = None
        if target_score is not None:
            target_seed_metrics = _seed_policy_metrics(
                actual, target_score, policy_rows, references, group_seeds
            )
            target_result = {
                "predictive": _predictive_metrics(
                    actual, target_score, row_groups, references
                ),
                "policy": _policy_metrics(
                    actual, target_score, policy_rows, references
                ),
                "seed_policy": target_seed_metrics,
            }
        candidate_scores = {}
        weights = (1.0,) if target_score is None else SOURCE_BLEND_WEIGHTS
        for source_group, source_score in source_scores.items():
            for weight in weights:
                candidate_name = f"{source_group}__source_weight_{weight:g}"
                candidate_scores[candidate_name] = (
                    source_score
                    if target_score is None
                    else (1.0 - weight) * target_score + weight * source_score
                )
        source_mean = np.mean(np.vstack(list(source_scores.values())), axis=0)
        for weight in weights:
            candidate_name = f"uniform_all_sources__source_weight_{weight:g}"
            candidate_scores[candidate_name] = (
                source_mean
                if target_score is None
                else (1.0 - weight) * target_score + weight * source_mean
            )
        candidate_seed_metrics = {
            name: _seed_policy_metrics(
                actual, score, policy_rows, references, group_seeds
            )
            for name, score in candidate_scores.items()
        }
        nested = nested_leave_one_seed_source_selection(
            candidate_seed_metrics, target_seed_metrics
        )
        if nested["source_minus_target_by_seed"]:
            nested["paired_source_minus_target_bootstrap"] = _paired_bootstrap(
                nested["source_minus_target_by_seed"]
            )
        budget_results[key] = {
            "target_group_budget": int(budget),
            "target_only": target_result,
            "per_source": per_source,
            "nested_source_selection": nested,
        }

    comparable = {
        budget: rows
        for budget, rows in budget_results.items()
        if int(budget) > 0
    }
    best_budget = min(
        comparable,
        key=lambda budget: (
            comparable[budget]["nested_source_selection"][
                "mean_source_minus_target"
            ],
            int(budget),
        ),
    )
    best_nested = comparable[best_budget]["nested_source_selection"]
    best_bootstrap = best_nested["paired_source_minus_target_bootstrap"]
    source_signal_present = bool(
        float(best_bootstrap["mean"]) <= -0.0005
        and float(best_bootstrap["upper_95"]) < 0.0
        and float(best_nested["mean_source_policy_delta"]) < 0.0
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "development_budget_curve_not_confirmation",
        "city": "jinan",
        "selector_scenario": selector_scenario,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "target_budgets": list(TARGET_BUDGETS),
        "selector_seed_count": len(selector_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(selector_contrast.size),
        "selection_audits": selection_audits,
        "budget_results": budget_results,
        "selection": {
            "best_budget": int(best_budget),
            "best_mean_source_minus_target": float(best_bootstrap["mean"]),
            "best_source_minus_target_upper_95": float(best_bootstrap["upper_95"]),
            "best_source_policy_delta_vs_phase_pressure": float(
                best_nested["mean_source_policy_delta"]
            ),
            "minimum_required_source_contribution": 0.0005,
            "source_signal_present": source_signal_present,
            "decision": (
                "authorize_guarded_budget_specific_source_successor"
                if source_signal_present
                else "reject_source_value_across_tested_target_budgets"
            ),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "protocol": ARTIFACT_PROTOCOL,
        },
        "fit_diagnostics": fitted_rows,
        "input_audits": {
            "source_bank": source_bank_audit,
            "target_bank": target_bank_audit,
            "selector_bank": selector_bank_audit,
            "selector_cache": selector_audit,
        },
        "inputs": {
            "fit_result_sha256": expected_fit_result_sha256,
            "source_cache_audit_sha256": _sha256(source_cache_audit_path),
            "target_cache_audit_sha256": _sha256(target_cache_audit_path),
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", required=True)
    parser.add_argument("--selector-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selector-collection-shards", type=int, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V123 cache workers must be in [1, 20]")
    if not 1 <= int(args.fit_workers) <= 20:
        raise ValueError("V123 fit workers must be in [1, 20]")
    result = run_target_budget_source_curve(
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
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
        fit_workers=args.fit_workers,
        artifact_path=args.artifact,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
