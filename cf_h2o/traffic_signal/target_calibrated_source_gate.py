"""Architecture-matched target calibration for causal source predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge


GATE_BASE_FEATURES = (
    "delta_green_q",
    "delta_red_q",
    "delta_green_occ",
    "delta_red_occ",
    "delta_green_down_q",
    "delta_green_down_occ",
    "delta_service_pressure",
    "delta_red_pressure",
    "delta_switch_indicator",
    "delta_clearance_fraction",
    "delta_movement_service_pressure",
    "delta_queue_concentration",
    "delta_green_signal_down_q_mean",
    "delta_green_signal_down_occ_mean",
    "reference_service_pressure",
    "reference_green_down_occ_max",
    "reference_queue_concentration",
)

SOURCE_AGGREGATE_FEATURES = (
    "source_score_mean",
    "source_score_median",
    "source_score_min",
    "source_score_max",
    "source_score_std",
    "source_score_q25",
    "source_score_q75",
    "source_improvement_support",
    "source_ucb_mean",
    "source_ucb_max",
)

ARM_NAMES = ("target_only", "source_aligned", "source_placebo")
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class FrozenStandardizedRidge:
    """Small, version-stable representation of a standardized ridge model."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("gate feature matrix changed")
        if not np.all(np.isfinite(values)):
            raise ValueError("gate feature matrix contains non-finite values")
        return ((values - self.mean) / self.scale) @ self.coefficients + self.intercept


def extract_gate_base_features(dataset: Any) -> np.ndarray:
    index = {str(name): position for position, name in enumerate(dataset.feature_names)}
    missing = [name for name in GATE_BASE_FEATURES if name not in index]
    if missing:
        raise KeyError(f"target-calibrated gate features are missing: {missing}")
    columns = np.asarray([index[name] for name in GATE_BASE_FEATURES], dtype=int)
    values = np.asarray(dataset.features[:, columns], dtype=float)
    if values.shape != (dataset.size, len(GATE_BASE_FEATURES)):
        raise ValueError("target-calibrated gate base feature shape changed")
    if not np.all(np.isfinite(values)):
        raise ValueError("target-calibrated gate base features are non-finite")
    return values


def aggregate_source_predictions(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build source-city permutation-invariant score and support features."""

    source_order = tuple(sorted(str(name) for name in predictions))
    if len(source_order) < 2:
        raise ValueError("source calibration requires at least two source groups")
    score_rows = []
    uncertainty_rows = []
    expected_size = None
    for name in source_order:
        score, uncertainty, trust = predictions[name]
        score = np.asarray(score, dtype=float).reshape(-1)
        uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
        trust = np.asarray(trust, dtype=float).reshape(-1)
        if expected_size is None:
            expected_size = int(score.size)
        if (
            score.size != expected_size
            or uncertainty.shape != score.shape
            or trust.shape != score.shape
            or not np.all(np.isfinite(score))
            or not np.all(np.isfinite(uncertainty))
            or not np.all(np.isfinite(trust))
            or np.any(uncertainty < 0.0)
        ):
            raise ValueError("source prediction arrays are not aligned")
        score_rows.append(score)
        uncertainty_rows.append(uncertainty)
    scores = np.vstack(score_rows)
    uncertainty = np.vstack(uncertainty_rows)
    upper = scores + uncertainty
    aggregates = np.column_stack(
        (
            np.mean(scores, axis=0),
            np.median(scores, axis=0),
            np.min(scores, axis=0),
            np.max(scores, axis=0),
            np.std(scores, axis=0),
            np.quantile(scores, 0.25, axis=0),
            np.quantile(scores, 0.75, axis=0),
            np.mean(scores < 0.0, axis=0),
            np.mean(upper, axis=0),
            np.max(upper, axis=0),
        )
    )
    if aggregates.shape != (int(expected_size), len(SOURCE_AGGREGATE_FEATURES)):
        raise ValueError("source aggregate feature shape changed")
    return aggregates, scores, source_order


def gate_feature_matrix(
    base_features: np.ndarray,
    source_features: np.ndarray,
    *,
    arm: str,
    base_feature_names: Sequence[str] = GATE_BASE_FEATURES,
) -> np.ndarray:
    base = np.asarray(base_features, dtype=float)
    source = np.asarray(source_features, dtype=float)
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown target-calibrated gate arm: {arm}")
    if (
        base.ndim != 2
        or source.ndim != 2
        or base.shape[0] != source.shape[0]
        or base.shape[1] != len(base_feature_names)
        or source.shape[1] != len(SOURCE_AGGREGATE_FEATURES)
    ):
        raise ValueError("target-calibrated gate feature blocks are incompatible")
    source_block = np.zeros_like(source) if arm == "target_only" else source
    combined = np.concatenate((base, source_block), axis=1)
    if not np.all(np.isfinite(combined)):
        raise ValueError("target-calibrated gate features are non-finite")
    return combined


def balanced_group_folds(
    groups: Sequence[str] | np.ndarray,
    *,
    fold_count: int = 5,
) -> np.ndarray:
    """Assign whole action groups to balanced deterministic folds."""

    values = np.asarray(groups, dtype=str)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("group folds require a nonempty one-dimensional array")
    unique = np.unique(values)
    actual_folds = min(int(fold_count), int(unique.size))
    if actual_folds < 2:
        raise ValueError("group cross-fitting requires at least two groups")
    strata: dict[str, list[str]] = {}
    for group in unique:
        text = str(group)
        scenario, separator, remainder = text.partition(":seed")
        seed = remainder.partition(":")[0] if separator else "unknown"
        strata.setdefault(f"{scenario}:seed{seed}", []).append(text)
    assignment: dict[str, int] = {}
    offset = 0
    for stratum in sorted(strata):
        for index, group in enumerate(sorted(strata[stratum])):
            assignment[group] = (offset + index) % actual_folds
        offset = (offset + len(strata[stratum])) % actual_folds
    row_folds = np.asarray([assignment[str(group)] for group in values], dtype=int)
    if set(row_folds.tolist()) != set(range(actual_folds)):
        raise ValueError("deterministic group folds are empty")
    return row_folds


def permute_source_features_by_group(
    source_features: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    is_reference: Sequence[bool] | np.ndarray,
    rank_signal: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Break source-target alignment while retaining block size and marginals."""

    source = np.asarray(source_features, dtype=float)
    group_values = np.asarray(groups, dtype=str)
    references = np.asarray(is_reference, dtype=bool)
    rank = np.asarray(rank_signal, dtype=float)
    if (
        source.ndim != 2
        or source.shape[0] != group_values.size
        or references.shape != (group_values.size,)
        or rank.shape != (group_values.size,)
    ):
        raise ValueError("placebo source features are not action-row aligned")
    result = np.zeros_like(source)
    strata: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for group in np.unique(group_values):
        rows = np.flatnonzero((group_values == group) & ~references)
        if rows.size == 0:
            raise ValueError("placebo group has no non-reference action")
        scenario = str(group).partition(":seed")[0]
        ordered = rows[np.lexsort((rows, rank[rows]))]
        strata.setdefault((scenario, int(rows.size)), []).append((str(group), ordered))
    changed = False
    for key in sorted(strata):
        blocks = sorted(strata[key], key=lambda row: row[0])
        if len(blocks) == 1:
            continue
        for index, (_, target_rows) in enumerate(blocks):
            donor_rows = blocks[(index + 1) % len(blocks)][1]
            result[target_rows] = source[donor_rows]
            changed = changed or not np.allclose(
                result[target_rows], source[target_rows], rtol=0.0, atol=0.0
            )
    if not changed:
        raise ValueError("placebo permutation did not break source alignment")
    return result


def _group_equal_weights(groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.zeros(groups.size, dtype=float)
    for group in np.unique(groups[mask]):
        rows = mask & (groups == group)
        weights[rows] = 1.0 / float(np.count_nonzero(rows))
    return weights


def _fit_standardized_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    feature_names: Sequence[str],
    alpha: float,
) -> FrozenStandardizedRidge:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if (
        x.ndim != 2
        or x.shape[0] != y.size
        or w.shape != y.shape
        or x.shape[1] != len(feature_names)
        or np.count_nonzero(w > 0.0) < 2
        or np.any(w < 0.0)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("ridge calibration inputs are invalid")
    selected = w > 0.0
    normalized_weights = w[selected] / np.sum(w[selected])
    mean = np.sum(x[selected] * normalized_weights[:, None], axis=0)
    variance = np.sum(
        np.square(x[selected] - mean) * normalized_weights[:, None], axis=0
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-10] = 1.0
    standardized = (x[selected] - mean) / scale
    estimator = Ridge(alpha=float(alpha), fit_intercept=True)
    estimator.fit(standardized, y[selected], sample_weight=w[selected])
    return FrozenStandardizedRidge(
        feature_names=tuple(str(name) for name in feature_names),
        mean=np.asarray(mean, dtype=float),
        scale=np.asarray(scale, dtype=float),
        coefficients=np.asarray(estimator.coef_, dtype=float).reshape(-1),
        intercept=float(estimator.intercept_),
        alpha=float(alpha),
    )


def _group_mean_squared_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
) -> float:
    values = []
    for group in np.unique(groups[mask]):
        rows = mask & (groups == group)
        values.append(float(np.mean(np.square(actual[rows] - predicted[rows]))))
    return float(np.mean(values))


def fit_architecture_matched_gate_arms(
    *,
    feature_matrices: Mapping[str, np.ndarray],
    targets: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    is_reference: Sequence[bool] | np.ndarray,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    fold_count: int = 5,
    base_feature_names: Sequence[str] = GATE_BASE_FEATURES,
) -> dict[str, Any]:
    """Cross-fit three equal-information-budget arms with one shared alpha."""

    if set(feature_matrices) != set(ARM_NAMES):
        raise ValueError("target-calibrated gate arm matrix is incomplete")
    y = np.asarray(targets, dtype=float).reshape(-1)
    group_values = np.asarray(groups, dtype=str)
    references = np.asarray(is_reference, dtype=bool)
    if y.shape != group_values.shape or references.shape != y.shape:
        raise ValueError("target-calibrated labels are not group aligned")
    trainable = ~references
    row_folds = balanced_group_folds(group_values, fold_count=fold_count)
    fold_values = tuple(sorted(int(value) for value in np.unique(row_folds)))
    base_names = tuple(str(value) for value in base_feature_names)
    if not base_names or len(set(base_names)) != len(base_names):
        raise ValueError("target-calibrated base feature names are invalid")
    names = (*base_names, *SOURCE_AGGREGATE_FEATURES)
    matrices = {name: np.asarray(feature_matrices[name], dtype=float) for name in ARM_NAMES}
    if any(matrix.shape != (y.size, len(names)) for matrix in matrices.values()):
        raise ValueError("target-calibrated arm architecture differs")
    weights = _group_equal_weights(group_values, trainable)
    alpha_diagnostics = {}
    alpha_candidates = tuple(float(value) for value in alpha_grid)
    if not alpha_candidates or any(value <= 0.0 for value in alpha_candidates):
        raise ValueError("ridge alpha grid must be positive")
    for alpha in alpha_candidates:
        arm_errors = {}
        for arm in ("target_only", "source_aligned"):
            prediction = np.full(y.size, np.nan, dtype=float)
            for fold in fold_values:
                fit_mask = trainable & (row_folds != fold)
                heldout = trainable & (row_folds == fold)
                model = _fit_standardized_ridge(
                    matrices[arm],
                    y,
                    np.where(fit_mask, weights, 0.0),
                    feature_names=names,
                    alpha=alpha,
                )
                prediction[heldout] = model.predict(matrices[arm][heldout])
            if not np.all(np.isfinite(prediction[trainable])):
                raise ValueError("shared alpha selection produced missing OOF rows")
            arm_errors[arm] = _group_mean_squared_error(
                y, prediction, group_values, trainable
            )
        alpha_diagnostics[str(alpha)] = {
            "target_only_group_mean_mse": float(arm_errors["target_only"]),
            "source_aligned_group_mean_mse": float(arm_errors["source_aligned"]),
            "paired_mean_group_mse": float(np.mean(list(arm_errors.values()))),
        }
    selected_alpha = min(
        alpha_candidates,
        key=lambda value: (
            alpha_diagnostics[str(value)]["paired_mean_group_mse"],
            -value,
        ),
    )
    models = {}
    oof_predictions = {}
    arm_diagnostics = {}
    for arm in ARM_NAMES:
        prediction = np.full(y.size, np.nan, dtype=float)
        for fold in fold_values:
            fit_mask = trainable & (row_folds != fold)
            heldout = trainable & (row_folds == fold)
            model = _fit_standardized_ridge(
                matrices[arm],
                y,
                np.where(fit_mask, weights, 0.0),
                feature_names=names,
                alpha=selected_alpha,
            )
            prediction[heldout] = model.predict(matrices[arm][heldout])
        models[arm] = _fit_standardized_ridge(
            matrices[arm],
            y,
            weights,
            feature_names=names,
            alpha=selected_alpha,
        )
        oof_predictions[arm] = prediction
        arm_diagnostics[arm] = {
            "group_mean_mse": _group_mean_squared_error(
                y, prediction, group_values, trainable
            ),
            "mean_absolute_error": float(
                np.mean(np.abs(y[trainable] - prediction[trainable]))
            ),
        }
    return {
        "feature_names": names,
        "selected_alpha": float(selected_alpha),
        "alpha_selection_arm": "paired_target_only_and_source_aligned",
        "alpha_diagnostics": alpha_diagnostics,
        "models": models,
        "oof_predictions": oof_predictions,
        "arm_diagnostics": arm_diagnostics,
        "row_folds": row_folds,
        "trainable_mask": trainable,
        "group_equal_weights": weights,
    }


def group_conformal_margins(
    *,
    actual: Sequence[float] | np.ndarray,
    oof_predictions: Mapping[str, np.ndarray],
    groups: Sequence[str] | np.ndarray,
    is_reference: Sequence[bool] | np.ndarray,
    quantiles: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Calibrate one-sided action-regret bounds at the action-group level."""

    y = np.asarray(actual, dtype=float).reshape(-1)
    group_values = np.asarray(groups, dtype=str)
    references = np.asarray(is_reference, dtype=bool)
    trainable = ~references
    result = {}
    for arm in ARM_NAMES:
        predicted = np.asarray(oof_predictions[arm], dtype=float).reshape(-1)
        if predicted.shape != y.shape or not np.all(np.isfinite(predicted[trainable])):
            raise ValueError("conformal prediction arrays changed")
        group_residual = np.asarray(
            [
                np.max(y[(group_values == group) & trainable] - predicted[(group_values == group) & trainable])
                for group in np.unique(group_values)
            ],
            dtype=float,
        )
        result[arm] = {
            str(float(quantile)): float(
                np.quantile(group_residual, float(quantile), method="higher")
            )
            for quantile in quantiles
        }
    return result
