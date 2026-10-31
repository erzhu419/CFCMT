"""Evaluate causal source weighting under one total target-label budget."""

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
from scipy.optimize import minimize
from scipy.stats import t as student_t

from cf_h2o.eval.traffic_signal_b100_source_reliability_diagnostic import (
    _policy_metrics,
    _predictive_metrics,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    SELECTION_SEED,
    _balanced_quotas,
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
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _coverage_first_group_order_v3,
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import (
    _selector_dataset,
)
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    TARGET_BUDGETS,
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
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid


RESULT_PROTOCOL = "tsc-v140-total-budget-causal-source-weighting-v1"
ADAPTATION_SEED_COUNT = 5
EVALUATION_SEED_COUNT = 17
EXPECTED_SELECTOR_SEED_COUNT = ADAPTATION_SEED_COUNT + EVALUATION_SEED_COUNT
EXPECTED_SOURCE_COUNT = 7
RIDGE_L2 = 0.05
MINIMUM_SOURCE_EFFECT = 0.0005
MINIMUM_IMPROVING_SEED_FRACTION = 0.80
ONE_SIDED_T_CRITICAL_DF4 = 2.132
PRIMARY_BUDGET = 25
POLICY_ALIGNED_SOURCE_MASSES = (0.25, 0.5, 0.75, 1.0)
_WEIGHT_STATE: Mapping[str, Any] | None = None


def split_adaptation_evaluation_seeds(
    seeds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted(int(value) for value in seeds))
    if len(ordered) != EXPECTED_SELECTOR_SEED_COUNT or len(set(ordered)) != len(ordered):
        raise ValueError(
            f"V140 requires exactly {EXPECTED_SELECTOR_SEED_COUNT} unique selector seeds"
        )
    adaptation = tuple(ordered[:ADAPTATION_SEED_COUNT])
    evaluation = tuple(ordered[ADAPTATION_SEED_COUNT:])
    return adaptation, evaluation


def select_total_budget_groups(
    dataset: Any,
    *,
    adaptation_seeds: Sequence[int],
    budget: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if int(budget) <= 0:
        raise ValueError("V140 target budget must be positive")
    seed_order = tuple(int(value) for value in adaptation_seeds)
    quotas = _balanced_quotas(
        int(budget), tuple(str(seed) for seed in seed_order)
    )
    groups = sorted(
        set(str(value) for value in dataset.metadata["action_group_ids"])
    )
    selected: list[str] = []
    by_seed: dict[str, dict[str, int]] = {}
    for seed in seed_order:
        candidates = [
            group for group in groups if parse_action_group_seed(group) == seed
        ]
        requested = int(quotas[str(seed)])
        ordered = _coverage_first_group_order_v3(
            dataset,
            candidates,
            selection_seed=SELECTION_SEED,
            role=f"v140_total_budget:jinan:seed{seed}",
        )
        if len(ordered) < requested:
            raise ValueError(
                f"V140 seed {seed} has {len(ordered)} groups, needs {requested}"
            )
        chosen = ordered[:requested]
        selected.extend(chosen)
        by_seed[str(seed)] = {
            "available_group_count": len(candidates),
            "selected_group_count": len(chosen),
        }
    if len(selected) != int(budget) or len(set(selected)) != int(budget):
        raise ValueError("V140 target-budget groups are incomplete or duplicated")
    return tuple(sorted(selected)), {
        "protocol": "numeric-seed-partitioned-balanced-coverage-first-v1",
        "budget": int(budget),
        "adaptation_partition": "five_smallest_numeric_selector_seeds",
        "adaptation_seeds": list(seed_order),
        "seed_quotas": quotas,
        "seeds": by_seed,
        "selected_group_ids": sorted(selected),
    }


def validate_nested_total_budgets(
    selected: Mapping[int, Sequence[str]],
) -> None:
    budgets = sorted(int(value) for value in selected)
    for lower, upper in zip(budgets, budgets[1:]):
        if not set(selected[lower]) < set(selected[upper]):
            raise ValueError(
                f"V140 target budgets are not strictly nested: {lower}, {upper}"
            )


def _fit_worker(task: tuple[str, int | None, str | None]) -> dict[str, Any]:
    if _WEIGHT_STATE is None:
        raise RuntimeError("V140 worker state is missing")
    from threadpoolctl import threadpool_limits

    role, heldout_seed, source_group = task
    state = _WEIGHT_STATE
    if role.startswith("inner_"):
        if heldout_seed is None:
            raise ValueError("V140 inner fit lacks a held-out seed")
        target_groups = tuple(state["inner_training_groups"][int(heldout_seed)])
        prediction_dataset = state["adaptation_contrast"]
    elif role.startswith("final_"):
        if heldout_seed is not None:
            raise ValueError("V140 final fit unexpectedly has a held-out seed")
        target_groups = tuple(state["selected_groups"])
        prediction_dataset = state["evaluation_contrast"]
    else:
        raise ValueError(f"unknown V140 fit role: {role!r}")

    started = time.monotonic()
    with threadpool_limits(limits=1):
        if role.endswith("target"):
            if source_group is not None:
                raise ValueError("V140 target task has a source group")
            model = _target_model(
                state["adaptation_dataset"],
                group_ids=target_groups,
                candidate=str(state["candidate"]),
            )
            domains = ("jinan",)
        elif role.endswith("source"):
            if source_group not in state["source_states"]:
                raise ValueError("V140 source task has an unknown source group")
            fitted = _fit_component_budget(
                state["source_states"][str(source_group)],
                target_group_ids=target_groups,
            )
            expected_domains = {str(source_group), "jinan"}
            if set(fitted.diagnostics["source_domains"]) != expected_domains:
                raise ValueError("V140 source information boundary changed")
            model = FrozenAnchoredBlendModel(
                anchor_model=fitted.family_models[state["anchor_family"]],
                correction_model=fitted.family_models[state["correction_family"]],
                candidate=str(state["candidate"]),
                anchor_objective_mode=fitted.objective_modes[state["anchor_family"]],
                correction_objective_mode=fitted.objective_modes[
                    state["correction_family"]
                ],
            )
            domains = tuple(sorted(expected_domains))
        else:
            raise ValueError(f"V140 fit role has no model family: {role!r}")
        score, _, _, _ = _group_adjusted_scores(
            prediction_dataset,
            model.predict(prediction_dataset),
            objective_mode="control_only",
        )
    return {
        "role": role,
        "heldout_seed": heldout_seed,
        "source_group": source_group,
        "score": np.asarray(score, dtype=float),
        "training_group_count": len(target_groups),
        "training_domains": list(domains),
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _group_and_seed_equal_weights(
    groups: np.ndarray,
    references: np.ndarray,
    row_seeds: np.ndarray,
    fit_seeds: Sequence[int],
) -> np.ndarray:
    fit = set(int(value) for value in fit_seeds)
    weights = np.zeros(groups.size, dtype=float)
    for seed in sorted(fit):
        seed_groups = sorted(
            set(groups[(row_seeds == seed) & ~references].tolist())
        )
        if not seed_groups:
            raise ValueError(f"V140 weight fit has no groups for seed {seed}")
        seed_mass = 1.0 / len(fit)
        for group in seed_groups:
            rows = (groups == group) & ~references
            count = int(np.count_nonzero(rows))
            weights[rows] = seed_mass / len(seed_groups) / max(count, 1)
    return weights


def fit_nonnegative_source_weights(
    *,
    actual: np.ndarray,
    target_score: np.ndarray,
    source_residuals: np.ndarray,
    groups: np.ndarray,
    references: np.ndarray,
    row_seeds: np.ndarray,
    fit_seeds: Sequence[int],
    ridge_l2: float = RIDGE_L2,
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(actual, dtype=float) - np.asarray(target_score, dtype=float)
    x = np.asarray(source_residuals, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("V140 source residual matrix is misaligned")
    weights = _group_and_seed_equal_weights(
        groups, references, row_seeds, fit_seeds
    )
    active = weights > 0.0
    if np.count_nonzero(active) <= x.shape[1]:
        raise ValueError("V140 source weighting has insufficient active rows")
    w = weights[active]
    x_fit = x[active]
    y_fit = y[active]

    def objective(beta: np.ndarray) -> float:
        error = y_fit - x_fit @ beta
        return float(np.sum(w * np.square(error)) + float(ridge_l2) * beta @ beta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        error = y_fit - x_fit @ beta
        return -2.0 * (x_fit.T @ (w * error)) + 2.0 * float(ridge_l2) * beta

    result = minimize(
        objective,
        np.zeros(x.shape[1], dtype=float),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * x.shape[1],
        constraints=(
            {
                "type": "ineq",
                "fun": lambda beta: 1.0 - float(np.sum(beta)),
                "jac": lambda beta: -np.ones(beta.shape, dtype=float),
            },
        ),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"V140 source-weight optimization failed: {result.message}")
    beta = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    if float(np.sum(beta)) > 1.0 + 1e-8:
        raise ValueError("V140 source weights exceed the target/source simplex")
    return beta, {
        "fit_seeds": sorted(int(value) for value in fit_seeds),
        "active_row_count": int(np.count_nonzero(active)),
        "ridge_l2": float(ridge_l2),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "source_weight_sum": float(np.sum(beta)),
        "source_null_weight": float(1.0 - np.sum(beta)),
    }


def permute_source_residuals_by_seed_group(
    source_residuals: np.ndarray,
    groups: np.ndarray,
    references: np.ndarray,
    rank_signal: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_residuals, dtype=float)
    result = np.zeros_like(source)
    strata: dict[tuple[str, int, int], list[tuple[str, np.ndarray]]] = {}
    for group in np.unique(groups):
        rows = np.flatnonzero((groups == group) & ~references)
        if rows.size == 0:
            raise ValueError("V140 placebo group has no candidate action")
        scenario = str(group).partition(":seed")[0]
        seed = parse_action_group_seed(str(group))
        ordered = rows[np.lexsort((rows, rank_signal[rows]))]
        strata.setdefault((scenario, seed, int(rows.size)), []).append(
            (str(group), ordered)
        )
    changed = False
    for key in sorted(strata):
        blocks = sorted(strata[key], key=lambda value: value[0])
        if len(blocks) < 2:
            continue
        for index, (_, target_rows) in enumerate(blocks):
            donor_rows = blocks[(index + 1) % len(blocks)][1]
            result[target_rows] = source[donor_rows]
            changed |= not np.allclose(
                result[target_rows], source[target_rows], rtol=0.0, atol=0.0
            )
    result[references] = source[references]
    if not changed:
        raise ValueError("V140 placebo permutation did not change any source block")
    return result


def _seed_policy_values(
    *,
    actual: np.ndarray,
    score: np.ndarray,
    policy_rows: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[int, float]:
    delta, _ = _policy_arrays(actual, score, policy_rows, references)
    return {
        seed: float(np.mean(delta[group_seeds == seed]))
        for seed in sorted(int(value) for value in np.unique(group_seeds))
    }


def _one_sided_upper(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size != ADAPTATION_SEED_COUNT:
        raise ValueError("V140 adaptation gate seed count changed")
    return float(
        np.mean(array)
        + ONE_SIDED_T_CRITICAL_DF4
        * np.std(array, ddof=1)
        / np.sqrt(array.size)
    )


def _effect_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "upper_95_one_sided": _one_sided_upper(array),
        "improving_seed_fraction": float(np.mean(array < 0.0)),
        "seed_values": [float(value) for value in array],
    }


def _target_effect_authorized(summary: Mapping[str, Any]) -> bool:
    return bool(
        float(summary["mean"]) <= -MINIMUM_SOURCE_EFFECT
        and float(summary["upper_95_one_sided"]) < 0.0
        and float(summary["improving_seed_fraction"])
        >= MINIMUM_IMPROVING_SEED_FRACTION
    )


def nested_source_weight_gate(
    *,
    actual: np.ndarray,
    target_score: np.ndarray,
    source_residuals: np.ndarray,
    placebo_residuals: np.ndarray,
    groups: np.ndarray,
    references: np.ndarray,
    row_seeds: np.ndarray,
    policy_rows: np.ndarray,
    group_seeds: np.ndarray,
    adaptation_seeds: Sequence[int],
) -> dict[str, Any]:
    seeds = tuple(int(value) for value in adaptation_seeds)
    target_values = _seed_policy_values(
        actual=actual,
        score=target_score,
        policy_rows=policy_rows,
        references=references,
        group_seeds=group_seeds,
    )
    folds = []
    aligned_values: dict[int, float] = {}
    placebo_values: dict[int, float] = {}
    for heldout in seeds:
        fit_seeds = tuple(seed for seed in seeds if seed != heldout)
        aligned_weights, aligned_fit = fit_nonnegative_source_weights(
            actual=actual,
            target_score=target_score,
            source_residuals=source_residuals,
            groups=groups,
            references=references,
            row_seeds=row_seeds,
            fit_seeds=fit_seeds,
        )
        placebo_weights, placebo_fit = fit_nonnegative_source_weights(
            actual=actual,
            target_score=target_score,
            source_residuals=placebo_residuals,
            groups=groups,
            references=references,
            row_seeds=row_seeds,
            fit_seeds=fit_seeds,
        )
        aligned_score = target_score + source_residuals @ aligned_weights
        placebo_score = target_score + placebo_residuals @ placebo_weights
        aligned = _seed_policy_values(
            actual=actual,
            score=aligned_score,
            policy_rows=policy_rows,
            references=references,
            group_seeds=group_seeds,
        )[heldout]
        placebo = _seed_policy_values(
            actual=actual,
            score=placebo_score,
            policy_rows=policy_rows,
            references=references,
            group_seeds=group_seeds,
        )[heldout]
        aligned_values[heldout] = aligned
        placebo_values[heldout] = placebo
        folds.append(
            {
                "heldout_seed": heldout,
                "fit_seeds": list(fit_seeds),
                "target_only_value": target_values[heldout],
                "aligned_value": aligned,
                "placebo_value": placebo,
                "aligned_minus_target": aligned - target_values[heldout],
                "aligned_minus_placebo": aligned - placebo,
                "placebo_minus_target": placebo - target_values[heldout],
                "aligned_weights": aligned_weights.tolist(),
                "placebo_weights": placebo_weights.tolist(),
                "aligned_fit": aligned_fit,
                "placebo_fit": placebo_fit,
            }
        )
    aligned_target = _effect_summary(
        [aligned_values[seed] - target_values[seed] for seed in seeds]
    )
    aligned_placebo = _effect_summary(
        [aligned_values[seed] - placebo_values[seed] for seed in seeds]
    )
    placebo_target = _effect_summary(
        [placebo_values[seed] - target_values[seed] for seed in seeds]
    )
    aligned_authorized = bool(
        _target_effect_authorized(aligned_target)
        and float(aligned_placebo["mean"]) <= -MINIMUM_SOURCE_EFFECT
        and float(aligned_placebo["upper_95_one_sided"]) < 0.0
    )
    placebo_authorized = _target_effect_authorized(placebo_target)
    aligned_full, aligned_full_fit = fit_nonnegative_source_weights(
        actual=actual,
        target_score=target_score,
        source_residuals=source_residuals,
        groups=groups,
        references=references,
        row_seeds=row_seeds,
        fit_seeds=seeds,
    )
    placebo_full, placebo_full_fit = fit_nonnegative_source_weights(
        actual=actual,
        target_score=target_score,
        source_residuals=placebo_residuals,
        groups=groups,
        references=references,
        row_seeds=row_seeds,
        fit_seeds=seeds,
    )
    return {
        "folds": folds,
        "aligned_minus_target": aligned_target,
        "aligned_minus_placebo": aligned_placebo,
        "placebo_minus_target": placebo_target,
        "aligned_authorized": aligned_authorized,
        "placebo_authorized": placebo_authorized,
        "aligned_full_weights_raw": aligned_full.tolist(),
        "placebo_full_weights_raw": placebo_full.tolist(),
        "aligned_full_weights_effective": (
            aligned_full.tolist()
            if aligned_authorized
            else np.zeros_like(aligned_full).tolist()
        ),
        "placebo_full_weights_effective": (
            placebo_full.tolist()
            if placebo_authorized
            else np.zeros_like(placebo_full).tolist()
        ),
        "aligned_full_fit": aligned_full_fit,
        "placebo_full_fit": placebo_full_fit,
    }


def _selector_arm_summary(values: Mapping[int, float]) -> dict[str, Any]:
    ordered = [float(values[seed]) for seed in sorted(values)]
    return {
        **_paired_bootstrap(ordered),
        "seed_values": {
            str(seed): float(values[seed]) for seed in sorted(values)
        },
    }


def _paired_selector_summary(
    candidate: Mapping[int, float], reference: Mapping[int, float]
) -> dict[str, Any]:
    seeds = tuple(sorted(candidate))
    if tuple(sorted(reference)) != seeds:
        raise ValueError("V140 selector arm seeds are misaligned")
    differences = [float(candidate[seed] - reference[seed]) for seed in seeds]
    return {**_paired_bootstrap(differences), "seed_values": differences}


def policy_aligned_source_candidates(
    source_order: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return the frozen finite source/null candidate family used by V142."""
    sources = tuple(str(value) for value in source_order)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("policy-aligned source candidates require unique sources")
    candidates = {"source_null": np.zeros(len(sources), dtype=float)}
    for index, source in enumerate(sources):
        for mass in POLICY_ALIGNED_SOURCE_MASSES:
            weights = np.zeros(len(sources), dtype=float)
            weights[index] = float(mass)
            candidates[f"single:{source}:mass={mass:g}"] = weights
    for mass in POLICY_ALIGNED_SOURCE_MASSES:
        candidates[f"uniform:mass={mass:g}"] = np.full(
            len(sources), float(mass) / len(sources), dtype=float
        )
    return candidates


def _effect_summary_for_seed_count(
    values: Sequence[float], *, expected_seed_count: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or array.size != int(expected_seed_count)
        or array.size < 2
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("policy-aligned effect seed inventory changed")
    critical = float(student_t.ppf(0.95, df=array.size - 1))
    return {
        "mean": float(np.mean(array)),
        "upper_95_one_sided": float(
            np.mean(array)
            + critical * np.std(array, ddof=1) / np.sqrt(array.size)
        ),
        "improving_seed_fraction": float(np.mean(array < 0.0)),
        "seed_values": [float(value) for value in array],
    }


def nested_policy_aligned_source_gate(
    *,
    actual: np.ndarray,
    target_score: np.ndarray,
    source_residuals: np.ndarray,
    placebo_residuals: np.ndarray,
    policy_rows: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
    adaptation_seeds: Sequence[int],
    source_order: Sequence[str],
    minimum_improving_seed_fraction: float = MINIMUM_IMPROVING_SEED_FRACTION,
) -> dict[str, Any]:
    """Select a finite source blend by downstream policy value, not row MSE."""
    seeds = tuple(int(value) for value in adaptation_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("policy-aligned gate requires unique adaptation seeds")
    candidates = policy_aligned_source_candidates(source_order)
    target_values = _seed_policy_values(
        actual=actual,
        score=target_score,
        policy_rows=policy_rows,
        references=references,
        group_seeds=group_seeds,
    )
    aligned_values = {
        name: _seed_policy_values(
            actual=actual,
            score=target_score + source_residuals @ weights,
            policy_rows=policy_rows,
            references=references,
            group_seeds=group_seeds,
        )
        for name, weights in candidates.items()
    }
    placebo_values = {
        name: _seed_policy_values(
            actual=actual,
            score=target_score + placebo_residuals @ weights,
            policy_rows=policy_rows,
            references=references,
            group_seeds=group_seeds,
        )
        for name, weights in candidates.items()
    }

    def select_candidate(fit_seeds: Sequence[int]) -> str:
        return min(
            candidates,
            key=lambda name: (
                float(np.mean([aligned_values[name][seed] for seed in fit_seeds])),
                0 if name == "source_null" else 1,
                name,
            ),
        )

    folds = []
    aligned_target_effects = []
    aligned_placebo_effects = []
    selected_non_null = 0
    for heldout in seeds:
        fit_seeds = tuple(seed for seed in seeds if seed != heldout)
        selected = select_candidate(fit_seeds)
        aligned = float(aligned_values[selected][heldout])
        placebo = float(placebo_values[selected][heldout])
        target = float(target_values[heldout])
        aligned_target_effects.append(aligned - target)
        aligned_placebo_effects.append(aligned - placebo)
        selected_non_null += int(selected != "source_null")
        folds.append(
            {
                "heldout_seed": heldout,
                "fit_seeds": list(fit_seeds),
                "selected_candidate": selected,
                "selected_weights": candidates[selected].tolist(),
                "target_only_value": target,
                "aligned_value": aligned,
                "placebo_value": placebo,
                "aligned_minus_target": aligned - target,
                "aligned_minus_placebo": aligned - placebo,
            }
        )
    aligned_target = _effect_summary_for_seed_count(
        aligned_target_effects, expected_seed_count=len(seeds)
    )
    aligned_placebo = _effect_summary_for_seed_count(
        aligned_placebo_effects, expected_seed_count=len(seeds)
    )
    selected_full = select_candidate(seeds)
    full_weights = candidates[selected_full]
    aligned_authorized = bool(
        selected_full != "source_null"
        and float(aligned_target["mean"]) <= -MINIMUM_SOURCE_EFFECT
        and float(aligned_target["upper_95_one_sided"]) < 0.0
        and float(aligned_target["improving_seed_fraction"])
        >= float(minimum_improving_seed_fraction)
        and float(aligned_placebo["mean"]) <= -MINIMUM_SOURCE_EFFECT
        and float(aligned_placebo["upper_95_one_sided"]) < 0.0
    )
    effective = full_weights if aligned_authorized else np.zeros_like(full_weights)
    return {
        "strategy": "nested-policy-value-discrete-source-selection-v1",
        "candidate_count": len(candidates),
        "candidate_names": list(candidates),
        "folds": folds,
        "aligned_minus_target": aligned_target,
        "aligned_minus_placebo": aligned_placebo,
        "aligned_authorized": aligned_authorized,
        "selected_non_null_fold_count": selected_non_null,
        "selected_full_candidate_raw": selected_full,
        "aligned_full_weights_raw": full_weights.tolist(),
        "aligned_full_weights_effective": effective.tolist(),
        "placebo_full_weights_effective": effective.tolist(),
        "minimum_improving_seed_fraction": float(
            minimum_improving_seed_fraction
        ),
    }


def run_total_budget_source_weighting(
    *,
    budget: int,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
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
    seed_partition: tuple[Sequence[int], Sequence[int]] | None = None,
    selection_strategy: str = "nonnegative_ridge",
    result_protocol: str = RESULT_PROTOCOL,
    run_label: str = "v140",
    adaptation_partition_label: str = "five_smallest_numeric_selector_seeds",
) -> dict[str, Any]:
    started = time.monotonic()
    if int(budget) not in TARGET_BUDGETS:
        raise ValueError(f"V140 unsupported target budget: {budget}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
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
    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
        cache_workers=cache_workers,
    )
    _assert_pure_waiting_selector_dataset(
        selector, target_name="prefix_mean_cost_450s"
    )
    if seed_partition is None:
        adaptation_seeds, evaluation_seeds = split_adaptation_evaluation_seeds(
            selector_seeds
        )
    else:
        adaptation_seeds = tuple(int(value) for value in seed_partition[0])
        evaluation_seeds = tuple(int(value) for value in seed_partition[1])
        observed = tuple(int(value) for value in selector_seeds)
        if (
            not adaptation_seeds
            or not evaluation_seeds
            or set(adaptation_seeds) & set(evaluation_seeds)
            or set(adaptation_seeds) | set(evaluation_seeds) != set(observed)
            or len(adaptation_seeds) + len(evaluation_seeds) != len(observed)
        ):
            raise ValueError("custom adaptation/evaluation seed partition is invalid")
    all_groups = sorted(
        set(str(value) for value in selector.metadata["action_group_ids"])
    )
    adaptation_group_pool = {
        group
        for group in all_groups
        if parse_action_group_seed(group) in set(adaptation_seeds)
    }
    evaluation_groups = {
        group
        for group in all_groups
        if parse_action_group_seed(group) in set(evaluation_seeds)
    }
    if adaptation_group_pool & evaluation_groups:
        raise ValueError("V140 adaptation/evaluation groups overlap")
    observed_selector_seeds = {
        parse_action_group_seed(group) for group in all_groups
    }
    if observed_selector_seeds != set(adaptation_seeds) | set(evaluation_seeds):
        raise ValueError("V140 selector seed inventory changed")
    adaptation_dataset = _group_subset_v3(
        selector,
        selected_groups=adaptation_group_pool,
        metadata_updates={"target_data_role": f"{run_label}_adaptation_pool"},
    )
    evaluation_dataset = _group_subset_v3(
        selector,
        selected_groups=evaluation_groups,
        metadata_updates={"target_data_role": f"{run_label}_evaluation_only"},
    )
    selected_by_budget = {}
    selection_audits = {}
    for candidate_budget in TARGET_BUDGETS:
        selected, audit = select_total_budget_groups(
            adaptation_dataset,
            adaptation_seeds=adaptation_seeds,
            budget=candidate_budget,
        )
        selected_by_budget[int(candidate_budget)] = selected
        audit["adaptation_partition"] = str(adaptation_partition_label)
        selection_audits[str(candidate_budget)] = audit
    validate_nested_total_budgets(selected_by_budget)
    selected_groups = tuple(selected_by_budget[int(budget)])
    selected_dataset = _group_subset_v3(
        adaptation_dataset,
        selected_groups=set(selected_groups),
        metadata_updates={"target_data_role": f"{run_label}_B{int(budget)}"},
    )
    adaptation_contrast = build_action_contrast_dataset(
        selected_dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    evaluation_contrast = build_action_contrast_dataset(
        evaluation_dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    adaptation_actual, _, adaptation_group_rows, adaptation_references, adaptation_group_seeds = (
        _normalized_pressure_targets(selected_dataset, adaptation_contrast)
    )
    evaluation_actual, evaluation_unique_groups, evaluation_group_rows, evaluation_references, evaluation_group_seeds = (
        _normalized_pressure_targets(evaluation_dataset, evaluation_contrast)
    )
    adaptation_groups = np.asarray(
        action_group_ids(adaptation_contrast), dtype=str
    )
    adaptation_row_seeds = np.asarray(
        [parse_action_group_seed(group) for group in adaptation_groups], dtype=int
    )
    evaluation_groups_rows = np.asarray(
        action_group_ids(evaluation_contrast), dtype=str
    )
    adaptation_policy_rows = np.vstack(adaptation_group_rows).astype(int, copy=False)
    evaluation_policy_rows = np.vstack(evaluation_group_rows).astype(int, copy=False)
    if set(adaptation_group_seeds.tolist()) != set(adaptation_seeds):
        raise ValueError("V140 selected budget lacks an adaptation seed")
    if set(evaluation_group_seeds.tolist()) != set(evaluation_seeds):
        raise ValueError("V140 evaluation seed coverage changed")

    target_context = _static_context_from_dataset(adaptation_dataset)
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    target_key = f"{run_label}_target_jinan_B{int(budget)}"
    source_states = {}
    for source_group, scenarios in source_scenarios.items():
        fit_bank = {name: source_bank[name] for name in scenarios}
        fit_bank[target_key] = adaptation_dataset
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
    source_order = tuple(sorted(source_states))
    if len(source_order) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"V140 requires exactly {EXPECTED_SOURCE_COUNT} causal source domains"
        )
    inner_training_groups = {
        seed: tuple(
            group
            for group in selected_groups
            if parse_action_group_seed(group) != int(seed)
        )
        for seed in adaptation_seeds
    }
    from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
        ANCHOR_FAMILY,
        CORRECTION_FAMILY,
    )

    tasks = []
    for seed in adaptation_seeds:
        tasks.append(("inner_target", seed, None))
        tasks.extend(("inner_source", seed, source) for source in source_order)
    tasks.append(("final_target", None, None))
    tasks.extend(("final_source", None, source) for source in source_order)
    global _WEIGHT_STATE
    _WEIGHT_STATE = {
        "adaptation_dataset": adaptation_dataset,
        "adaptation_contrast": adaptation_contrast,
        "evaluation_contrast": evaluation_contrast,
        "selected_groups": selected_groups,
        "inner_training_groups": inner_training_groups,
        "source_states": source_states,
        "candidate": str(fit_result["frozen_blend_candidate"]),
        "anchor_family": ANCHOR_FAMILY,
        "correction_family": CORRECTION_FAMILY,
    }
    actual_workers = min(max(1, int(fit_workers)), len(tasks))
    try:
        if actual_workers == 1:
            fitted_rows = [_fit_worker(task) for task in tasks]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("V140 parallel fit requires Linux fork")
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                fitted_rows = list(pool.map(_fit_worker, tasks))
    finally:
        _WEIGHT_STATE = None

    adaptation_target_score = np.full(adaptation_contrast.size, np.nan, dtype=float)
    adaptation_source_scores = {
        source: np.full(adaptation_contrast.size, np.nan, dtype=float)
        for source in source_order
    }
    evaluation_target_score = None
    evaluation_source_scores: dict[str, np.ndarray] = {}
    fit_diagnostics = []
    for row in fitted_rows:
        score = np.asarray(row.pop("score"), dtype=float)
        fit_diagnostics.append(row)
        role = str(row["role"])
        if role == "inner_target":
            seed = int(row["heldout_seed"])
            mask = adaptation_row_seeds == seed
            adaptation_target_score[mask] = score[mask]
        elif role == "inner_source":
            seed = int(row["heldout_seed"])
            source = str(row["source_group"])
            mask = adaptation_row_seeds == seed
            adaptation_source_scores[source][mask] = score[mask]
        elif role == "final_target":
            evaluation_target_score = score
        elif role == "final_source":
            evaluation_source_scores[str(row["source_group"])] = score
    if evaluation_target_score is None:
        raise ValueError("V140 final target prediction is missing")
    if not np.all(np.isfinite(adaptation_target_score)) or any(
        not np.all(np.isfinite(values))
        for values in adaptation_source_scores.values()
    ):
        raise ValueError("V140 adaptation OOF prediction coverage is incomplete")
    if set(evaluation_source_scores) != set(source_order):
        raise ValueError("V140 final source prediction coverage is incomplete")

    adaptation_source_matrix = np.column_stack(
        [adaptation_source_scores[source] for source in source_order]
    )
    adaptation_residuals = (
        adaptation_source_matrix - adaptation_target_score[:, None]
    )
    adaptation_pressure_index = adaptation_contrast.feature_names.index(
        "delta_service_pressure"
    )
    adaptation_rank_signal = np.asarray(
        adaptation_contrast.features[:, adaptation_pressure_index], dtype=float
    )
    adaptation_placebo = permute_source_residuals_by_seed_group(
        adaptation_residuals,
        adaptation_groups,
        adaptation_references,
        adaptation_rank_signal,
    )
    if selection_strategy == "nonnegative_ridge":
        gate = nested_source_weight_gate(
            actual=adaptation_actual,
            target_score=adaptation_target_score,
            source_residuals=adaptation_residuals,
            placebo_residuals=adaptation_placebo,
            groups=adaptation_groups,
            references=adaptation_references,
            row_seeds=adaptation_row_seeds,
            policy_rows=adaptation_policy_rows,
            group_seeds=adaptation_group_seeds,
            adaptation_seeds=adaptation_seeds,
        )
    elif selection_strategy == "policy_aligned_discrete":
        gate = nested_policy_aligned_source_gate(
            actual=adaptation_actual,
            target_score=adaptation_target_score,
            source_residuals=adaptation_residuals,
            placebo_residuals=adaptation_placebo,
            policy_rows=adaptation_policy_rows,
            references=adaptation_references,
            group_seeds=adaptation_group_seeds,
            adaptation_seeds=adaptation_seeds,
            source_order=source_order,
        )
    else:
        raise ValueError(f"unknown source selection strategy: {selection_strategy!r}")
    aligned_weights = np.asarray(
        gate["aligned_full_weights_effective"], dtype=float
    )
    placebo_weights = np.asarray(
        gate["placebo_full_weights_effective"], dtype=float
    )
    evaluation_source_matrix = np.column_stack(
        [evaluation_source_scores[source] for source in source_order]
    )
    evaluation_residuals = (
        evaluation_source_matrix - evaluation_target_score[:, None]
    )
    evaluation_pressure_index = evaluation_contrast.feature_names.index(
        "delta_service_pressure"
    )
    evaluation_rank_signal = np.asarray(
        evaluation_contrast.features[:, evaluation_pressure_index], dtype=float
    )
    evaluation_placebo = permute_source_residuals_by_seed_group(
        evaluation_residuals,
        evaluation_groups_rows,
        evaluation_references,
        evaluation_rank_signal,
    )
    aligned_score = evaluation_target_score + evaluation_residuals @ aligned_weights
    placebo_score = evaluation_target_score + evaluation_placebo @ placebo_weights
    uniform_score = np.mean(evaluation_source_matrix, axis=1)

    adaptation_source_seed_values = {}
    for index, source in enumerate(source_order):
        adaptation_source_seed_values[source] = _seed_policy_values(
            actual=adaptation_actual,
            score=adaptation_source_matrix[:, index],
            policy_rows=adaptation_policy_rows,
            references=adaptation_references,
            group_seeds=adaptation_group_seeds,
        )
    best_single_source = min(
        source_order,
        key=lambda source: (
            float(np.mean(list(adaptation_source_seed_values[source].values()))),
            source,
        ),
    )
    best_single_score = evaluation_source_scores[best_single_source]

    arm_scores = {
        "target_only": np.asarray(evaluation_target_score, dtype=float),
        "causal_source_weighted": aligned_score,
        "matched_source_placebo": placebo_score,
        "uniform_all_sources": uniform_score,
        "adaptation_selected_single_source": best_single_score,
    }
    seed_values = {
        arm: _seed_policy_values(
            actual=evaluation_actual,
            score=score,
            policy_rows=evaluation_policy_rows,
            references=evaluation_references,
            group_seeds=evaluation_group_seeds,
        )
        for arm, score in arm_scores.items()
    }
    arm_summaries = {
        arm: {
            **_selector_arm_summary(values),
            "policy": _policy_metrics(
                evaluation_actual,
                arm_scores[arm],
                evaluation_policy_rows,
                evaluation_references,
            ),
            "predictive": _predictive_metrics(
                evaluation_actual,
                arm_scores[arm],
                evaluation_groups_rows,
                evaluation_references,
            ),
        }
        for arm, values in seed_values.items()
    }
    source_minus_target = _paired_selector_summary(
        seed_values["causal_source_weighted"], seed_values["target_only"]
    )
    source_minus_placebo = _paired_selector_summary(
        seed_values["causal_source_weighted"],
        seed_values["matched_source_placebo"],
    )
    selector_relative_passed = bool(
        gate["aligned_authorized"]
        and source_minus_target["mean"] <= -MINIMUM_SOURCE_EFFECT
        and source_minus_target["upper_95"] < 0.0
        and source_minus_target["improving_seed_fraction"]
        >= MINIMUM_IMPROVING_SEED_FRACTION
        and source_minus_placebo["mean"] <= -MINIMUM_SOURCE_EFFECT
        and source_minus_placebo["upper_95"] < 0.0
    )
    return {
        "protocol": str(result_protocol),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "total-target-label-budget-relative-transfer-development",
        "city": "jinan",
        "budget": int(budget),
        "primary_budget": PRIMARY_BUDGET,
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": str(protocol["target_name"]),
        "source_group_order": list(source_order),
        "selection_strategy": str(selection_strategy),
        "adaptation_seeds": list(adaptation_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "information_budget": {
            "unique_target_action_groups": len(selected_groups),
            "world_model_target_groups": len(selected_groups),
            "source_weight_target_groups": len(selected_groups),
            "union_target_group_count": len(selected_groups),
            "evaluation_seed_labels_used_for_fit_weighting_or_selection": False,
            "adaptation_group_ids": list(selected_groups),
        },
        "selection_audit": selection_audits[str(int(budget))],
        "source_weight_gate": gate,
        "effective_source_weights": {
            source: float(aligned_weights[index])
            for index, source in enumerate(source_order)
        },
        "effective_source_null_weight": float(1.0 - np.sum(aligned_weights)),
        "placebo_source_weights": {
            source: float(placebo_weights[index])
            for index, source in enumerate(source_order)
        },
        "adaptation_selected_single_source": best_single_source,
        "evaluation_group_count": len(evaluation_unique_groups),
        "evaluation_row_count": int(evaluation_contrast.size),
        "arm_summaries": arm_summaries,
        "paired_effects": {
            "source_weighted_minus_target_only": source_minus_target,
            "source_weighted_minus_matched_placebo": source_minus_placebo,
        },
        "development_gate": {
            "adaptation_source_authorized": bool(gate["aligned_authorized"]),
            "selector_relative_passed": selector_relative_passed,
            "primary_budget_passed": bool(
                int(budget) == PRIMARY_BUDGET and selector_relative_passed
            ),
            "decision": (
                "authorize_untouched_city_confirmation"
                if int(budget) == PRIMARY_BUDGET and selector_relative_passed
                else "do_not_authorize_confirmation_from_this_budget"
            ),
        },
        "fit_diagnostics": fit_diagnostics,
        "input_audits": {
            "source_bank": source_bank_audit,
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "inputs": {
            "fit_result_sha256": expected_fit_result_sha256,
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
            "source_cache_audit_sha256": _sha256(source_cache_audit_path),
        },
        "claim_boundary": (
            "V140 tests relative source value under one total target-label "
            "budget. Absolute performance versus PhasePressure remains the "
            "separate arm-summary estimand."
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V140 result: {args.out}")
    result = run_total_budget_source_weighting(
        budget=args.budget,
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=max(1, int(args.cache_workers)),
        fit_workers=max(1, int(args.fit_workers)),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if result["development_gate"]["selector_relative_passed"]
                    else "REJECT"
                ),
                "protocol": result["protocol"],
                "budget": result["budget"],
                "development_gate": result["development_gate"],
                "effective_source_weights": result["effective_source_weights"],
                "effective_source_null_weight": result[
                    "effective_source_null_weight"
                ],
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
