"""Test whether pre-action context can gate a balanced-mechanism oracle."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_h2o.eval.traffic_signal_anchored_source_null_residual_feasibility import (
    MINIMUM_EFFECT,
    _arm_summary,
    _group_layout,
    _retention_threshold,
    _seed_values,
)
from cf_h2o.eval.traffic_signal_local_mechanism_surrogate_oracle import (
    _normalized_target_contrasts,
    _surrogate_proposals,
    _surrogate_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _selector_dataset
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.action_ranker import CAUSAL_CORE_RANKING_PARENTS
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v135-context-gated-mechanism-oracle-v1"
ARM = "context_gated_balanced_mechanism_oracle"
BASE_RETENTION_FRACTIONS = (0.05, 0.10)
RISK_MULTIPLIERS = (0.0, 0.5, 1.0)
CALIBRATION_SEED_COUNT = 5
MINIMUM_CALIBRATION_INTERVENTIONS = 20
MINIMUM_FINAL_INTERVENTIONS = 40
MINIMUM_IMPROVING_SEED_FRACTION = 0.80
ONE_SIDED_T_CRITICAL_DF4 = 2.132
CONTEXT_ERROR_QUANTILE = 0.90


@dataclass(frozen=True)
class ContextGateConfig:
    learning_rate: float = 0.05
    max_iter: int = 80
    max_leaf_nodes: int = 7
    min_samples_leaf: int = 12
    l2_regularization: float = 20.0
    random_state: int = 20260803


CONTEXT_GATE_CONFIG = ContextGateConfig()


def _outer_partition(
    all_seeds: Sequence[int], heldout_seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = tuple(sorted(int(value) for value in all_seeds))
    heldout = int(heldout_seed)
    if heldout not in ordered:
        raise ValueError("V135 held-out seed is absent")
    start = ordered.index(heldout)
    calibration = tuple(
        ordered[(start + offset) % len(ordered)]
        for offset in range(1, CALIBRATION_SEED_COUNT + 1)
    )
    training = tuple(
        seed for seed in ordered if seed != heldout and seed not in calibration
    )
    if (
        len(training) + len(calibration) + 1 != len(ordered)
        or set(training) & set(calibration)
        or heldout in training
        or heldout in calibration
    ):
        raise ValueError("V135 outer partition is not disjoint")
    return training, calibration


def _seed_local_retention_mask(
    proposals: Mapping[str, np.ndarray],
    group_seeds: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, dict[int, float | None]]:
    proposed = np.asarray(proposals["proposed"], dtype=bool)
    advantage = np.asarray(proposals["predicted_advantage"], dtype=float)
    accepted = np.zeros(proposed.shape, dtype=bool)
    thresholds: dict[int, float | None] = {}
    for seed in sorted(int(value) for value in np.unique(group_seeds)):
        seed_proposed = proposed & (group_seeds == seed)
        threshold = _retention_threshold(advantage[seed_proposed], fraction)
        thresholds[seed] = threshold
        if threshold is not None:
            accepted |= seed_proposed & (advantage >= float(threshold))
    return accepted, thresholds


def _seed_balanced_weights(seeds: np.ndarray) -> np.ndarray:
    values = np.asarray(seeds, dtype=int)
    weights = np.zeros(values.size, dtype=float)
    unique = np.unique(values)
    for seed in unique:
        mask = values == int(seed)
        weights[mask] = 1.0 / max(int(np.count_nonzero(mask)), 1)
    weights *= weights.size / max(float(np.sum(weights)), 1e-12)
    return weights


def _fit_regressor(
    features: np.ndarray,
    target: np.ndarray,
    seeds: np.ndarray,
) -> tuple[HistGradientBoostingRegressor | None, float, dict[str, Any]]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or seeds.shape != y.shape:
        raise ValueError("V135 context-gate fit arrays are misaligned")
    if x.shape[0] < 2:
        raise ValueError("V135 context gate requires at least two proposal rows")
    weights = _seed_balanced_weights(seeds)
    constant = float(np.average(y, weights=weights))
    if float(np.std(y)) < 1e-10:
        model = None
        fitted = np.full(y.shape, constant, dtype=float)
    else:
        config = CONTEXT_GATE_CONFIG
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            early_stopping=False,
            random_state=config.random_state,
        )
        model.fit(x, y, sample_weight=weights)
        fitted = np.asarray(model.predict(x), dtype=float)
    return model, constant, {
        "row_count": int(y.size),
        "seed_count": int(np.unique(seeds).size),
        "target_mean": float(np.mean(y)),
        "target_improving_fraction": float(np.mean(y < 0.0)),
        "training_mae": float(np.mean(np.abs(y - fitted))),
        "estimator": (
            "HistGradientBoostingRegressor" if model is not None else "constant"
        ),
    }


def _predict_regressor(
    model: HistGradientBoostingRegressor | None,
    constant: float,
    features: np.ndarray,
) -> np.ndarray:
    if model is None:
        return np.full(features.shape[0], float(constant), dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def _cross_fitted_context_prediction(
    *,
    features: np.ndarray,
    target: np.ndarray,
    proposal_mask: np.ndarray,
    group_seeds: np.ndarray,
    training_seeds: Sequence[int],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    training = tuple(int(value) for value in training_seeds)
    prediction = np.full(group_seeds.shape, np.nan, dtype=float)
    oof_rows = np.zeros(group_seeds.shape, dtype=bool)
    for heldout in training:
        fit_mask = proposal_mask & np.isin(
            group_seeds,
            np.asarray([seed for seed in training if seed != heldout], dtype=int),
        )
        valid_mask = proposal_mask & (group_seeds == heldout)
        if not np.any(valid_mask):
            continue
        model, constant, _ = _fit_regressor(
            features[fit_mask], target[fit_mask], group_seeds[fit_mask]
        )
        prediction[valid_mask] = _predict_regressor(
            model, constant, features[valid_mask]
        )
        oof_rows |= valid_mask
    expected_oof = proposal_mask & np.isin(
        group_seeds, np.asarray(training, dtype=int)
    )
    if not np.array_equal(oof_rows, expected_oof):
        raise ValueError("V135 context-gate OOF coverage is incomplete")
    absolute_error = np.abs(target[oof_rows] - prediction[oof_rows])
    error_quantile = max(
        float(np.quantile(absolute_error, CONTEXT_ERROR_QUANTILE)), 1e-8
    )
    full_mask = expected_oof
    model, constant, fit = _fit_regressor(
        features[full_mask], target[full_mask], group_seeds[full_mask]
    )
    prediction[:] = _predict_regressor(model, constant, features)
    return prediction, error_quantile, {
        **fit,
        "outer_training_seed_count": len(training),
        "oof_row_count": int(np.count_nonzero(oof_rows)),
        "oof_mae": float(np.mean(absolute_error)),
        "oof_error_quantile": error_quantile,
        "oof_error_quantile_level": CONTEXT_ERROR_QUANTILE,
    }


def _one_sided_upper(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size != CALIBRATION_SEED_COUNT:
        raise ValueError("V135 calibration seed count changed")
    return float(
        np.mean(array)
        + ONE_SIDED_T_CRITICAL_DF4
        * np.std(array, ddof=1)
        / np.sqrt(array.size)
    )


def _calibration_candidate(
    *,
    proposals: Mapping[str, np.ndarray],
    base_mask: np.ndarray,
    context_prediction: np.ndarray,
    error_quantile: float,
    group_seeds: np.ndarray,
    calibration_seeds: Sequence[int],
    retention_fraction: float,
    risk_multiplier: float,
) -> tuple[dict[str, Any], np.ndarray]:
    calibration_mask = np.isin(
        group_seeds, np.asarray(calibration_seeds, dtype=int)
    )
    predicted_upper = np.asarray(context_prediction, dtype=float) + (
        float(risk_multiplier) * float(error_quantile)
    )
    accepted = np.asarray(base_mask, dtype=bool) & (predicted_upper < 0.0)
    accepted_calibration = accepted & calibration_mask
    values = _seed_values(
        proposals, accepted_calibration, group_seeds, calibration_seeds
    )
    seed_array = np.asarray(
        [values[int(seed)] for seed in calibration_seeds], dtype=float
    )
    count = int(np.count_nonzero(accepted_calibration))
    mean = float(np.mean(seed_array))
    upper = _one_sided_upper(seed_array)
    improving = float(np.mean(seed_array < 0.0))
    return {
        "retention_fraction": float(retention_fraction),
        "risk_multiplier": float(risk_multiplier),
        "context_error_quantile": float(error_quantile),
        "intervention_count": count,
        "mean": mean,
        "upper_95_one_sided": upper,
        "improving_seed_fraction": improving,
        "authorized": bool(
            count >= MINIMUM_CALIBRATION_INTERVENTIONS
            and mean <= -MINIMUM_EFFECT
            and upper < 0.0
            and improving >= MINIMUM_IMPROVING_SEED_FRACTION
        ),
        "seed_values": {
            str(seed): float(values[int(seed)]) for seed in calibration_seeds
        },
    }, accepted


def _outer_fold(
    heldout_seed: int,
    *,
    all_seeds: Sequence[int],
    proposals: Mapping[str, np.ndarray],
    context_features: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[str, Any]:
    training_seeds, calibration_seeds = _outer_partition(
        all_seeds, heldout_seed
    )
    actual = np.asarray(proposals["actual_selected_delta"], dtype=float)
    candidates: list[dict[str, Any]] = []
    accepted_by_key: dict[tuple[float, float], np.ndarray] = {}
    fit_diagnostics: dict[str, Any] = {}
    base_thresholds: dict[str, dict[str, float | None]] = {}
    for retention in BASE_RETENTION_FRACTIONS:
        base_mask, thresholds = _seed_local_retention_mask(
            proposals, group_seeds, retention
        )
        context_prediction, error_quantile, fit = (
            _cross_fitted_context_prediction(
                features=context_features,
                target=actual,
                proposal_mask=base_mask,
                group_seeds=group_seeds,
                training_seeds=training_seeds,
            )
        )
        retention_key = f"{retention:g}"
        fit_diagnostics[retention_key] = fit
        base_thresholds[retention_key] = {
            str(seed): (None if value is None else float(value))
            for seed, value in thresholds.items()
        }
        for risk in RISK_MULTIPLIERS:
            candidate, accepted = _calibration_candidate(
                proposals=proposals,
                base_mask=base_mask,
                context_prediction=context_prediction,
                error_quantile=error_quantile,
                group_seeds=group_seeds,
                calibration_seeds=calibration_seeds,
                retention_fraction=retention,
                risk_multiplier=risk,
            )
            candidates.append(candidate)
            accepted_by_key[(retention, risk)] = accepted
    authorized = [row for row in candidates if row["authorized"]]
    selected = (
        min(
            authorized,
            key=lambda row: (
                float(row["upper_95_one_sided"]),
                float(row["mean"]),
                -float(row["risk_multiplier"]),
                float(row["retention_fraction"]),
            ),
        )
        if authorized
        else None
    )
    heldout_mask = group_seeds == int(heldout_seed)
    if selected is None:
        accepted = np.zeros(group_seeds.shape, dtype=bool)
    else:
        accepted = accepted_by_key[
            (
                float(selected["retention_fraction"]),
                float(selected["risk_multiplier"]),
            )
        ] & heldout_mask
    heldout_value = _seed_values(
        proposals, accepted, group_seeds, (int(heldout_seed),)
    )[int(heldout_seed)]
    active = actual[accepted]
    return {
        "heldout_seed": int(heldout_seed),
        "training_seeds": list(training_seeds),
        "calibration_seeds": list(calibration_seeds),
        "context_gate_fit": fit_diagnostics,
        "base_advantage_thresholds": base_thresholds,
        "calibration_candidates": candidates,
        "selected_calibration_profile": selected,
        "arms": {
            ARM: {
                "heldout_value": float(heldout_value),
                "heldout_intervention_count": int(np.count_nonzero(accepted)),
                "heldout_harmful_intervention_fraction": (
                    float(np.mean(active > 0.0)) if active.size else 0.0
                ),
                "deployment_enabled": selected is not None,
            }
        },
    }


def run_context_gated_mechanism_oracle(
    *,
    selector_cache_root: Path,
    selector_manifest_path: Path,
    selector_cache_audit_path: Path,
    expected_selector_cache_audit_sha256: str,
    selector_scenario: str,
    selector_seeds: Sequence[int],
    selector_collection_shards: int,
    conversion_root: Path,
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
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
    contrast = build_action_contrast_dataset(
        selector,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual, unique_groups, group_rows, _, group_seeds = (
        _normalized_pressure_targets(selector, contrast)
    )
    policy_rows, references, layout_seeds = _group_layout(contrast)
    if not np.array_equal(policy_rows, np.vstack(group_rows)) or not np.array_equal(
        layout_seeds, group_seeds
    ):
        raise ValueError("V135 action-group layout changed")
    normalized = _normalized_target_contrasts(selector, contrast, group_rows)
    balanced_score = _surrogate_scores(normalized)["balanced_physical"]
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = np.asarray(
        contrast.features[:, pressure_index], dtype=float
    ) >= -1e-12
    proposals = _surrogate_proposals(
        balanced_score,
        actual,
        policy_rows,
        eligible,
        references,
    )
    feature_names = tuple(
        name
        for name in CAUSAL_CORE_RANKING_PARENTS
        if name in contrast.feature_names
    )
    if not feature_names:
        raise ValueError("V135 has no admitted pre-action context features")
    feature_indices = np.asarray(
        [contrast.feature_names.index(name) for name in feature_names], dtype=int
    )
    selected_rows = np.asarray(proposals["selected_rows"], dtype=int)
    context_features = np.asarray(
        contrast.features[selected_rows][:, feature_indices], dtype=float
    )
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    if len(all_seeds) <= CALIBRATION_SEED_COUNT + 2:
        raise ValueError("V135 has too few selector seeds")
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    all_seeds=all_seeds,
                    proposals=proposals,
                    context_features=context_features,
                    group_seeds=group_seeds,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    summary = _arm_summary(folds, ARM)
    passed = bool(
        summary["intervention_count"] >= MINIMUM_FINAL_INTERVENTIONS
        and summary["mean"] <= -MINIMUM_EFFECT
        and summary["upper_95"] < 0.0
        and summary["improving_seed_fraction"]
        >= MINIMUM_IMPROVING_SEED_FRACTION
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "nested-context-gated-local-mechanism-oracle-screen",
        "city": "jinan",
        "estimand": "normalized_450s_waiting_delta_vs_phase_pressure",
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "source_predictions_used": False,
            "same_seed_one_step_counterfactual_labels_used_for_oracle_proposal": True,
            "heldout_seed_450s_labels_used_for_fit_or_calibration": False,
            "heldout_seed_one_step_labels_used_by_context_gate": False,
            "context_gate_training_seeds_per_fold": (
                len(all_seeds) - CALIBRATION_SEED_COUNT - 1
            ),
            "calibration_seeds_per_fold": CALIBRATION_SEED_COUNT,
            "heldout_seeds_per_fold": 1,
        },
        "oracle_proposal": {
            "surrogate": "balanced_physical",
            "retention_fractions": list(BASE_RETENTION_FRACTIONS),
            "action_constraint": "nondecreasing_instantaneous_service_pressure",
        },
        "context_gate": {
            "feature_names": list(feature_names),
            "feature_count": len(feature_names),
            "config": asdict(CONTEXT_GATE_CONFIG),
            "cross_fit_unit": "selector_seed",
            "oof_error_quantile": CONTEXT_ERROR_QUANTILE,
            "risk_multipliers": list(RISK_MULTIPLIERS),
        },
        "calibration_gate": {
            "minimum_interventions": MINIMUM_CALIBRATION_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": MINIMUM_IMPROVING_SEED_FRACTION,
            "one_sided_t_critical_df4": ONE_SIDED_T_CRITICAL_DF4,
        },
        "folds": folds,
        "arm_summary": summary,
        "development_gate": {
            "passed": passed,
            "minimum_final_interventions": MINIMUM_FINAL_INTERVENTIONS,
            "minimum_mean_effect": MINIMUM_EFFECT,
            "minimum_improving_seed_fraction": MINIMUM_IMPROVING_SEED_FRACTION,
            "decision": (
                "authorize_learned_balanced_mechanism_proposal_successor"
                if passed
                else "close_context_gated_one_step_mechanism_control_branch"
            ),
        },
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "inputs": {
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "claim_boundary": (
            "V135 still uses the held-out branch's realized one-step outcomes "
            "to propose an action. Only the context gate is target-label-held-out; "
            "this is not a learned, deployable, zero-shot, or transfer policy."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", required=True)
    parser.add_argument("--selector-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selector-collection-shards", type=int, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fold-workers", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V135 result: {args.out}")
    result = run_context_gated_mechanism_oracle(
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=max(1, int(args.cache_workers)),
        fold_workers=max(1, int(args.fold_workers)),
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "PASS" if result["development_gate"]["passed"] else "REJECT",
                "protocol": result["protocol"],
                "development_gate": result["development_gate"],
                "arm_summary": result["arm_summary"],
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
