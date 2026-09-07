"""Diagnose whether any B100 source component carries target-useful signal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit import (
    ARTIFACT_PROTOCOL as CROSSFIT_ARTIFACT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _normalized_pressure_targets,
    _selected_adaptation_dataset,
    _target_cache_audit_contract,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.target_calibrated_source_gate import (
    permute_source_features_by_group,
)


RESULT_PROTOCOL = "tsc-v122-b100-source-reliability-diagnostic-v1"
TEMPERATURES = (0.01, 0.05, 0.1, 0.25)
BLEND_WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def _load_crossfit(
    path: Path,
    *,
    expected_sha256: str,
    fit_result_sha256: str,
) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("V122 cross-fit artifact identity changed")
    artifact = pickle.loads(Path(path).read_bytes())
    if (
        artifact.get("protocol") != CROSSFIT_ARTIFACT_PROTOCOL
        or artifact.get("fit_result_sha256") != fit_result_sha256
        or int(artifact.get("target_group_budget", -1)) != 100
    ):
        raise ValueError("V122 cross-fit contract changed")
    return artifact


def _row_weights(groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(groups.size, dtype=float)
    active_rows = np.flatnonzero(mask)
    if active_rows.size == 0:
        raise ValueError("group-equal weights require at least one active row")
    _, inverse, counts = np.unique(
        groups[active_rows], return_inverse=True, return_counts=True
    )
    result[active_rows] = 1.0 / counts[inverse].astype(float)
    return result


def _predictive_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    references: np.ndarray,
) -> dict[str, float]:
    mask = ~references
    weights = _row_weights(groups, mask)
    normalized = weights / np.sum(weights)
    error = predicted - actual
    actual_values = actual[mask]
    predicted_values = predicted[mask]
    correlation = (
        float(np.corrcoef(actual_values, predicted_values)[0, 1])
        if np.std(actual_values) > 0.0 and np.std(predicted_values) > 0.0
        else 0.0
    )
    return {
        "group_equal_mse": float(np.sum(normalized * np.square(error))),
        "group_equal_mae": float(np.sum(normalized * np.abs(error))),
        "nonreference_pearson_correlation": correlation,
    }


def _policy_metrics(
    actual: np.ndarray,
    score: np.ndarray,
    group_rows: Sequence[np.ndarray],
    references: np.ndarray,
) -> dict[str, Any]:
    if isinstance(group_rows, np.ndarray) and group_rows.ndim == 2:
        rows = np.asarray(group_rows, dtype=int)
        positions = np.argmin(score[rows], axis=1)
        chosen_rows = rows[np.arange(rows.shape[0]), positions]
        reference_mask = references[rows]
        if not np.all(np.sum(reference_mask, axis=1) == 1):
            raise ValueError("policy groups do not contain exactly one reference")
        reference_rows = rows[
            np.arange(rows.shape[0]), np.argmax(reference_mask, axis=1)
        ]
    else:
        chosen_rows = np.asarray(
            [int(rows[int(np.argmin(score[rows]))]) for rows in group_rows],
            dtype=int,
        )
        reference_rows = np.asarray(
            [int(rows[references[rows]][0]) for rows in group_rows], dtype=int
        )
    delta = np.asarray(actual[chosen_rows], dtype=float)
    intervention = chosen_rows != reference_rows
    return {
        "mean_normalized_delta_vs_phase_pressure": float(np.mean(delta)),
        "worst_group_delta": float(np.max(delta)),
        "intervention_fraction": float(np.mean(intervention)),
        "harmful_intervention_fraction": float(
            np.count_nonzero(intervention & (delta > 0.0))
            / max(np.count_nonzero(intervention), 1)
        ),
        "improving_intervention_fraction": float(
            np.count_nonzero(intervention & (delta < 0.0))
            / max(np.count_nonzero(intervention), 1)
        ),
        "selected_action_residual_quantiles": {
            str(quantile): float(
                np.quantile(delta - score[chosen_rows], quantile, method="higher")
            )
            for quantile in (0.5, 0.75, 0.9, 0.95)
        },
    }


def _source_error(
    actual: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    references: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    trainable = mask & ~references
    weights = _row_weights(groups, trainable)
    weights /= np.sum(weights)
    return np.sum(
        weights[None, :] * np.square(scores - actual[None, :]), axis=1
    )


def _weights(error: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(error, dtype=float)
    if method.startswith("top"):
        count = int(method[3:])
        order = np.argsort(values, kind="stable")[:count]
        result = np.zeros(values.size, dtype=float)
        result[order] = 1.0 / float(count)
        return result
    if method == "inverse_mse":
        inverse = 1.0 / np.maximum(values, 1e-8)
        return inverse / np.sum(inverse)
    if method.startswith("softmax_"):
        temperature = float(method.split("_", 1)[1])
        logits = -(values - np.min(values)) / temperature
        result = np.exp(logits - np.max(logits))
        return result / np.sum(result)
    raise ValueError(f"unknown source reliability method: {method}")


def _cross_fitted_weighted_score(
    *,
    actual: np.ndarray,
    source_scores: np.ndarray,
    groups: np.ndarray,
    references: np.ndarray,
    row_folds: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full(actual.size, np.nan, dtype=float)
    fold_weights = {}
    for fold in sorted(int(value) for value in np.unique(row_folds)):
        training = row_folds != fold
        error = _source_error(
            actual, source_scores, groups, references, training
        )
        weight = _weights(error, method)
        heldout = row_folds == fold
        result[heldout] = weight @ source_scores[:, heldout]
        fold_weights[str(fold)] = weight.tolist()
    if not np.all(np.isfinite(result)):
        raise ValueError("V122 weighted source score coverage is incomplete")
    full_error = _source_error(
        actual,
        source_scores,
        groups,
        references,
        np.ones(actual.size, dtype=bool),
    )
    return result, {
        "fold_weights": fold_weights,
        "deployment_weights": _weights(full_error, method).tolist(),
        "deployment_source_error": full_error.tolist(),
    }


def run_diagnostic(
    *,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    crossfit_artifact_path: Path,
    expected_crossfit_artifact_sha256: str,
    target_cache_root: Path,
    target_manifest_path: Path,
    target_cache_audit_path: Path,
    conversion_root: Path,
    cache_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    _target_cache_audit_contract(target_cache_audit_path, fit_result)
    crossfit = _load_crossfit(
        crossfit_artifact_path,
        expected_sha256=expected_crossfit_artifact_sha256,
        fit_result_sha256=expected_fit_result_sha256,
    )
    selected_group_ids = tuple(
        str(value)
        for value in fit_result["information_budget"]["selected_group_ids"]
    )
    dataset, bank_audit, scenarios = _selected_adaptation_dataset(
        cache_root=target_cache_root,
        manifest_path=target_manifest_path,
        cache_workers=cache_workers,
        selected_group_ids=selected_group_ids,
    )
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual, unique_groups, group_rows, references, _ = _normalized_pressure_targets(
        dataset, contrast
    )
    groups = np.asarray(action_group_ids(contrast), dtype=str)
    if not np.array_equal(groups, np.asarray(crossfit["row_group_ids"], dtype=str)):
        raise ValueError("V122 rows differ from the V120 artifact")
    row_folds = np.asarray(crossfit["row_folds"], dtype=int)
    source_order = tuple(str(value) for value in crossfit["source_group_order"])
    source_scores = np.vstack(
        [
            np.asarray(crossfit["source_predictions"][group][0], dtype=float)
            for group in source_order
        ]
    )
    target_score = np.asarray(crossfit["target_predictions"][0], dtype=float)
    per_source = {}
    for index, group in enumerate(source_order):
        per_source[group] = {
            "predictive": _predictive_metrics(
                actual, source_scores[index], groups, references
            ),
            "policy": _policy_metrics(
                actual, source_scores[index], group_rows, references
            ),
        }
    target_metrics = {
        "predictive": _predictive_metrics(
            actual, target_score, groups, references
        ),
        "policy": _policy_metrics(actual, target_score, group_rows, references),
    }
    feature_index = {
        str(name): index for index, name in enumerate(contrast.feature_names)
    }
    rank_signal = np.asarray(
        contrast.features[:, feature_index["delta_service_pressure"]], dtype=float
    )
    placebo_scores = permute_source_features_by_group(
        source_scores.T,
        groups,
        references,
        rank_signal,
    ).T
    methods = (
        "top1",
        "top2",
        "top3",
        "top5",
        "top7",
        "inverse_mse",
        *(f"softmax_{value:g}" for value in TEMPERATURES),
    )
    strategies = {}
    placebo_strategies = {}
    for method in methods:
        weighted, audit = _cross_fitted_weighted_score(
            actual=actual,
            source_scores=source_scores,
            groups=groups,
            references=references,
            row_folds=row_folds,
            method=method,
        )
        placebo, placebo_audit = _cross_fitted_weighted_score(
            actual=actual,
            source_scores=placebo_scores,
            groups=groups,
            references=references,
            row_folds=row_folds,
            method=method,
        )
        for source_weight in BLEND_WEIGHTS:
            key = f"{method}__source_weight_{source_weight:g}"
            blended = (
                (1.0 - float(source_weight)) * target_score
                + float(source_weight) * weighted
            )
            placebo_blend = (
                (1.0 - float(source_weight)) * target_score
                + float(source_weight) * placebo
            )
            strategies[key] = {
                "method": method,
                "source_weight": float(source_weight),
                "predictive": _predictive_metrics(
                    actual, blended, groups, references
                ),
                "policy": _policy_metrics(
                    actual, blended, group_rows, references
                ),
                "source_weight_audit": audit,
            }
            placebo_strategies[key] = {
                "predictive": _predictive_metrics(
                    actual, placebo_blend, groups, references
                ),
                "policy": _policy_metrics(
                    actual, placebo_blend, group_rows, references
                ),
                "source_weight_audit": placebo_audit,
            }
    best_predictive = min(
        strategies,
        key=lambda key: (
            strategies[key]["predictive"]["group_equal_mse"], key
        ),
    )
    best_policy = min(
        strategies,
        key=lambda key: (
            strategies[key]["policy"][
                "mean_normalized_delta_vs_phase_pressure"
            ],
            key,
        ),
    )
    target_policy = float(
        target_metrics["policy"]["mean_normalized_delta_vs_phase_pressure"]
    )
    best_policy_value = float(
        strategies[best_policy]["policy"][
            "mean_normalized_delta_vs_phase_pressure"
        ]
    )
    matched_placebo_value = float(
        placebo_strategies[best_policy]["policy"][
            "mean_normalized_delta_vs_phase_pressure"
        ]
    )
    source_signal_present = bool(
        best_policy_value <= target_policy - 0.0005
        and best_policy_value <= matched_placebo_value - 0.0005
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "b100_training_diagnostic_not_selector_result",
        "city": "jinan",
        "group_count": len(unique_groups),
        "row_count": int(contrast.size),
        "source_group_order": list(source_order),
        "target_only": target_metrics,
        "per_source": per_source,
        "strategies": strategies,
        "placebo_strategies": placebo_strategies,
        "selection": {
            "best_predictive_strategy": best_predictive,
            "best_policy_strategy": best_policy,
            "target_only_policy_delta": target_policy,
            "best_policy_delta": best_policy_value,
            "matched_placebo_policy_delta": matched_placebo_value,
            "minimum_required_source_contribution": 0.0005,
            "source_signal_present": source_signal_present,
            "decision": (
                "authorize_frozen_reliability_weight_for_selector_test"
                if source_signal_present
                else "reject_b100_source_reliability_weighting"
            ),
        },
        "inputs": {
            "fit_result_sha256": expected_fit_result_sha256,
            "crossfit_artifact": {
                "path": str(Path(crossfit_artifact_path).resolve()),
                "sha256": expected_crossfit_artifact_sha256,
            },
            "target_cache_audit_sha256": _sha256(target_cache_audit_path),
            "adaptation_scenarios": list(scenarios),
        },
        "data_audit": bank_audit,
        "runtime_seconds": float(time.monotonic() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--crossfit-artifact", type=Path, required=True)
    parser.add_argument("--crossfit-artifact-sha256", required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V122 cache workers must be in [1, 20]")
    result = run_diagnostic(
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        crossfit_artifact_path=args.crossfit_artifact,
        expected_crossfit_artifact_sha256=args.crossfit_artifact_sha256,
        target_cache_root=args.target_cache_root,
        target_manifest_path=args.target_manifest,
        target_cache_audit_path=args.target_cache_audit,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
