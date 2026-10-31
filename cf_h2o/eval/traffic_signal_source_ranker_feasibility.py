"""Cross-fit an architecture-matched source-conditioned action ranker."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_conservative_source_intervention_gate import (
    Prediction,
    _load_prediction_artifact,
    _prediction,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _selector_dataset
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v126-source-ranker-feasibility-v1"
ARMS = ("target_only", "source_aligned", "source_placebo")
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
INNER_FOLDS = 3
MINIMUM_EFFECT = 0.0005


def _relative_prediction_features(
    prediction: Prediction,
    policy_rows: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(policy_rows, dtype=int)
    reference_mask = references[rows]
    if not np.all(np.sum(reference_mask, axis=1) == 1):
        raise ValueError("ranker groups require one reference action")
    reference_rows = rows[
        np.arange(rows.shape[0]), np.argmax(reference_mask, axis=1)
    ]
    output = np.zeros((prediction.score.size, 3), dtype=float)
    output[rows, 0] = (
        prediction.score[rows] - prediction.score[reference_rows, None]
    )
    output[rows, 1] = (
        prediction.uncertainty[rows]
        + prediction.uncertainty[reference_rows, None]
    )
    output[rows, 2] = np.minimum(
        prediction.trust[rows], prediction.trust[reference_rows, None]
    )
    return output


def _source_feature_matrix(
    sources: Mapping[str, Prediction],
    policy_rows: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    source_order = tuple(sorted(str(value) for value in sources))
    blocks = [
        _relative_prediction_features(
            sources[source], policy_rows, references
        )
        for source in source_order
    ]
    score_matrix = np.column_stack([block[:, 0] for block in blocks])
    uncertainty_matrix = np.column_stack([block[:, 1] for block in blocks])
    trust_matrix = np.column_stack([block[:, 2] for block in blocks])
    aggregate = np.column_stack(
        [
            np.mean(score_matrix, axis=1),
            np.median(score_matrix, axis=1),
            np.std(score_matrix, axis=1),
            np.min(score_matrix, axis=1),
            np.max(score_matrix, axis=1),
            np.mean(score_matrix < 0.0, axis=1),
            np.max(uncertainty_matrix, axis=1),
            np.min(trust_matrix, axis=1),
        ]
    )
    matrix = np.column_stack([*blocks, aggregate])
    names = tuple(
        f"source_{source}_{suffix}"
        for source in source_order
        for suffix in ("score_delta", "uncertainty", "trust")
    ) + (
        "source_score_mean",
        "source_score_median",
        "source_score_std",
        "source_score_min",
        "source_score_max",
        "source_improving_vote_fraction",
        "source_uncertainty_max",
        "source_trust_min",
    )
    return matrix, names


def _permute_source_groups(
    source_features: np.ndarray,
    policy_rows: np.ndarray,
    group_seeds: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(source_features)
    for seed in sorted(int(value) for value in np.unique(group_seeds)):
        group_indices = np.flatnonzero(group_seeds == seed)
        generator = np.random.default_rng(126000 + seed)
        permutation = generator.permutation(group_indices)
        if np.array_equal(permutation, group_indices):
            permutation = np.roll(permutation, 1)
        output[policy_rows[group_indices]] = source_features[
            policy_rows[permutation]
        ]
    return output


def _standardized_arm_matrices(
    common: np.ndarray,
    source_aligned: np.ndarray,
    source_placebo: np.ndarray,
) -> dict[str, np.ndarray]:
    raw = np.column_stack([common, source_aligned])
    mean = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    common_width = common.shape[1]
    common_scaled = (common - mean[:common_width]) / scale[:common_width]
    source_scaled = (
        source_aligned - mean[common_width:]
    ) / scale[common_width:]
    placebo_scaled = (
        source_placebo - mean[common_width:]
    ) / scale[common_width:]
    zeros = np.zeros_like(source_scaled)
    return {
        "target_only": np.column_stack([common_scaled, zeros]),
        "source_aligned": np.column_stack([common_scaled, source_scaled]),
        "source_placebo": np.column_stack([common_scaled, placebo_scaled]),
    }


def _sufficient_statistics(
    design: np.ndarray,
    target: np.ndarray,
    row_seeds: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with_intercept = np.column_stack([np.ones(design.shape[0]), design])
    return {
        seed: (
            with_intercept[row_seeds == seed].T
            @ with_intercept[row_seeds == seed],
            with_intercept[row_seeds == seed].T @ target[row_seeds == seed],
        )
        for seed in sorted(int(value) for value in np.unique(row_seeds))
    }


def _fit_ridge(
    statistics: Mapping[int, tuple[np.ndarray, np.ndarray]],
    training_seeds: Sequence[int],
    *,
    alpha: float,
) -> np.ndarray:
    xtx = sum(
        (statistics[int(seed)][0] for seed in training_seeds),
        start=np.zeros_like(next(iter(statistics.values()))[0]),
    )
    xty = sum(
        (statistics[int(seed)][1] for seed in training_seeds),
        start=np.zeros_like(next(iter(statistics.values()))[1]),
    )
    penalty = np.eye(xtx.shape[0]) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def _predict(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + design @ coefficients[1:]


def _policy_seed_values(
    predicted: np.ndarray,
    actual: np.ndarray,
    policy_rows: np.ndarray,
    eligible: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
    evaluation_seeds: Sequence[int],
) -> tuple[dict[int, float], dict[int, int]]:
    rows = np.asarray(policy_rows, dtype=int)
    allowed = np.asarray(eligible[rows], dtype=bool)
    if not np.all(np.any(allowed, axis=1)):
        raise ValueError("ranker eligibility removed every group action")
    scores = np.where(allowed, predicted[rows], np.inf)
    selected = rows[np.arange(rows.shape[0]), np.argmin(scores, axis=1)]
    reference_rows = rows[
        np.arange(rows.shape[0]), np.argmax(references[rows], axis=1)
    ]
    values = np.asarray(actual[selected], dtype=float)
    return (
        {
            int(seed): float(np.mean(values[group_seeds == seed]))
            for seed in evaluation_seeds
        },
        {
            int(seed): int(
                np.count_nonzero(
                    (selected != reference_rows) & (group_seeds == seed)
                )
            )
            for seed in evaluation_seeds
        },
    )


def _inner_seed_folds(training_seeds: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(sorted(int(value) for value in training_seeds))
    return tuple(
        tuple(ordered[index::INNER_FOLDS]) for index in range(INNER_FOLDS)
    )


def _select_alpha(
    design: np.ndarray,
    statistics: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    training_seeds: Sequence[int],
    actual: np.ndarray,
    policy_rows: np.ndarray,
    eligible: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
) -> tuple[float, dict[str, float]]:
    seed_set = set(int(value) for value in training_seeds)
    scores = {}
    for alpha in RIDGE_ALPHAS:
        validation_values = []
        for validation_seeds in _inner_seed_folds(training_seeds):
            fit_seeds = sorted(seed_set - set(validation_seeds))
            coefficients = _fit_ridge(
                statistics, fit_seeds, alpha=alpha
            )
            values, _ = _policy_seed_values(
                _predict(design, coefficients),
                actual,
                policy_rows,
                eligible,
                references,
                group_seeds,
                validation_seeds,
            )
            validation_values.extend(values.values())
        scores[str(alpha)] = float(np.mean(validation_values))
    selected = min(RIDGE_ALPHAS, key=lambda value: (scores[str(value)], value))
    return float(selected), scores


def _outer_fold(
    heldout_seed: int,
    *,
    matrices: Mapping[str, np.ndarray],
    statistics: Mapping[str, Mapping[int, tuple[np.ndarray, np.ndarray]]],
    all_seeds: Sequence[int],
    actual: np.ndarray,
    policy_rows: np.ndarray,
    eligible: np.ndarray,
    references: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[str, Any]:
    training_seeds = tuple(
        seed for seed in all_seeds if int(seed) != int(heldout_seed)
    )
    arms = {}
    for arm in ARMS:
        alpha, alpha_scores = _select_alpha(
            matrices[arm],
            statistics[arm],
            training_seeds=training_seeds,
            actual=actual,
            policy_rows=policy_rows,
            eligible=eligible,
            references=references,
            group_seeds=group_seeds,
        )
        coefficients = _fit_ridge(
            statistics[arm], training_seeds, alpha=alpha
        )
        values, interventions = _policy_seed_values(
            _predict(matrices[arm], coefficients),
            actual,
            policy_rows,
            eligible,
            references,
            group_seeds,
            (heldout_seed,),
        )
        arms[arm] = {
            "selected_alpha": alpha,
            "inner_policy_value_by_alpha": alpha_scores,
            "heldout_value": values[int(heldout_seed)],
            "heldout_intervention_count": interventions[int(heldout_seed)],
        }
    return {"heldout_seed": int(heldout_seed), "arms": arms}


def _arm_summary(folds: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    values = [float(row["arms"][arm]["heldout_value"]) for row in folds]
    return {
        **_paired_bootstrap(values),
        "intervention_count": int(
            sum(
                int(row["arms"][arm]["heldout_intervention_count"])
                for row in folds
            )
        ),
        "selected_alpha_counts": {
            str(alpha): int(
                sum(row["arms"][arm]["selected_alpha"] == alpha for row in folds)
            )
            for alpha in RIDGE_ALPHAS
        },
    }


def _paired_arm_summary(
    folds: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    return _paired_bootstrap(
        [
            float(row["arms"][candidate]["heldout_value"])
            - float(row["arms"][reference]["heldout_value"])
            for row in folds
        ]
    )


def run_source_ranker_feasibility(
    *,
    prediction_artifact_path: Path,
    expected_prediction_artifact_sha256: str,
    target_budget: int,
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
    artifact = _load_prediction_artifact(
        prediction_artifact_path,
        expected_sha256=expected_prediction_artifact_sha256,
    )
    budget_key = str(int(target_budget))
    if budget_key not in artifact["target_predictions"]:
        raise ValueError("V126 target budget is absent from V123")
    if (
        tuple(int(value) for value in artifact.get("selector_seeds", ()))
        != tuple(int(value) for value in selector_seeds)
        or artifact.get("selector_cache_audit_sha256")
        != expected_selector_cache_audit_sha256
    ):
        raise ValueError("V126 selector identity differs from V123")
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
    actual, unique_groups, group_rows, references, group_seeds = (
        _normalized_pressure_targets(selector, contrast)
    )
    policy_rows = np.vstack(group_rows).astype(int, copy=False)
    target_prediction = _prediction(artifact["target_predictions"][budget_key])
    sources = {
        str(source): _prediction(values)
        for source, values in artifact["source_predictions"][budget_key].items()
    }
    if target_prediction.score.size != contrast.size or any(
        value.score.size != contrast.size for value in sources.values()
    ):
        raise ValueError("V126 prediction rows differ from the selector")
    target_features = _relative_prediction_features(
        target_prediction, policy_rows, references
    )
    source_features, source_feature_names = _source_feature_matrix(
        sources, policy_rows, references
    )
    source_placebo = _permute_source_groups(
        source_features, policy_rows, group_seeds
    )
    common = np.column_stack([contrast.features, target_features])
    matrices = _standardized_arm_matrices(
        common, source_features, source_placebo
    )
    row_seeds = np.empty(contrast.size, dtype=int)
    row_seeds[policy_rows] = group_seeds[:, None]
    statistics = {
        arm: _sufficient_statistics(matrix, actual, row_seeds)
        for arm, matrix in matrices.items()
    }
    pressure_index = contrast.feature_names.index("delta_service_pressure")
    eligible = (
        np.asarray(contrast.features[:, pressure_index], dtype=float) >= -1e-12
    )
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    with ThreadPoolExecutor(max_workers=max(1, int(fold_workers))) as executor:
        folds = list(
            executor.map(
                lambda seed: _outer_fold(
                    seed,
                    matrices=matrices,
                    statistics=statistics,
                    all_seeds=all_seeds,
                    actual=actual,
                    policy_rows=policy_rows,
                    eligible=eligible,
                    references=references,
                    group_seeds=group_seeds,
                ),
                all_seeds,
            )
        )
    folds.sort(key=lambda row: row["heldout_seed"])
    arm_summaries = {arm: _arm_summary(folds, arm) for arm in ARMS}
    source_vs_target = _paired_arm_summary(
        folds, "source_aligned", "target_only"
    )
    source_vs_placebo = _paired_arm_summary(
        folds, "source_aligned", "source_placebo"
    )
    source = arm_summaries["source_aligned"]
    passed = bool(
        source["mean"] <= -MINIMUM_EFFECT
        and source["upper_95"] < 0.0
        and source_vs_target["mean"] <= -MINIMUM_EFFECT
        and source_vs_target["upper_95"] < 0.0
        and source_vs_placebo["mean"] <= -MINIMUM_EFFECT
        and source_vs_placebo["upper_95"] < 0.0
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "abundant-target-counterfactual-development-diagnostic",
        "city": artifact["city"],
        "target_budget": int(target_budget),
        "estimand": artifact["estimand"],
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "information_budget": {
            "outer_training_selector_seeds": len(all_seeds) - 1,
            "heldout_selector_seeds": 1,
            "target_counterfactual_labels_used_for_ranker_fit": True,
            "heldout_seed_labels_used_for_fit_or_alpha_selection": False,
            "unlabeled_full_selector_features_used_for_scaling": True,
        },
        "ranker": {
            "family": "group_policy_value_selected_ridge",
            "ridge_alphas": list(RIDGE_ALPHAS),
            "inner_seed_folds": INNER_FOLDS,
            "action_constraint": "nondecreasing_instantaneous_service_pressure",
            "common_feature_names": list(contrast.feature_names)
            + [
                "target_score_delta",
                "target_uncertainty",
                "target_trust",
            ],
            "source_feature_names": list(source_feature_names),
            "architecture_matched_dimension": int(
                next(iter(matrices.values())).shape[1]
            ),
        },
        "folds": folds,
        "arm_summaries": arm_summaries,
        "paired_source_contribution": {
            "source_aligned_minus_target_only": source_vs_target,
            "source_aligned_minus_source_placebo": source_vs_placebo,
        },
        "development_gate": {
            "minimum_effect": MINIMUM_EFFECT,
            "passed": passed,
            "decision": (
                "justify_budget_reduction_and_guard_development"
                if passed
                else "do_not_promote_source_ranker"
            ),
        },
        "inputs": {
            "prediction_artifact_sha256": expected_prediction_artifact_sha256,
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "claim_boundary": (
            "V126 uses abundant target counterfactual labels and is an upper "
            "feasibility test. It is neither few-shot nor confirmation."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--prediction-artifact-sha256", required=True)
    parser.add_argument("--target-budget", type=int, default=500)
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
        raise FileExistsError(f"refusing to overwrite V126 result: {args.out}")
    result = run_source_ranker_feasibility(
        prediction_artifact_path=args.prediction_artifact,
        expected_prediction_artifact_sha256=args.prediction_artifact_sha256,
        target_budget=args.target_budget,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if result["development_gate"]["passed"] else "REJECT"
                ),
                "protocol": result["protocol"],
                "arm_summaries": result["arm_summaries"],
                "paired_source_contribution": result[
                    "paired_source_contribution"
                ],
                "development_gate": result["development_gate"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
