"""Low-complexity state-conditioned utility gate for source proposals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from cf_h2o.traffic_signal.action_contrast import action_group_ids
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    SOURCE_EXPERT_GATE_FEATURE_NAMES,
    _source_expert_gate_records,
    _weighted_standardizer,
)
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    MECHANISM_ACTION_FEATURES,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


STATE_UTILITY_FEATURES = (
    "queue_concentration",
    "mean_occ",
    "mean_speed",
    "current_green_elapsed_norm",
    "graph_focal_to_neighbor_q_ratio",
    "neighbor_current_protected_ratio_mean",
    "neighbor_current_permissive_ratio_mean",
    "neighbor_switching_fraction",
    "neighbor_clearance_fraction_mean",
    "neighbor_phase_observation_ratio",
)
MECHANISM_BLOCKS = tuple(MECHANISM_ACTION_FEATURES)
UTILITY_FEATURE_NAMES = (
    *SOURCE_EXPERT_GATE_FEATURE_NAMES,
    *(f"state:{name}" for name in STATE_UTILITY_FEATURES),
    *(f"mechanism:{name}" for name in MECHANISM_BLOCKS),
)
PRESSURE_ALIGNED_CANDIDATE_PROTOCOL = (
    "phase-pressure-reference-or-exact-rigid-candidate-v1"
)
PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL = (
    "source-pairwise-phase-vs-rigid-or-exact-rigid-candidate-v1"
)
DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL = (
    "dense-phase-vs-rigid-source-evidence-records-v1"
)
DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES = (
    "rigid_phase_penalty_norm",
    "source_phase_preference_norm",
    "rigid_choice_margin_norm",
    "source_global_margin_norm",
    "source_correction_at_rigid_norm",
    "source_correction_at_phase_norm",
    "source_correction_range_norm",
    "log_action_count",
    *(f"state:{name}" for name in STATE_UTILITY_FEATURES),
    *(f"mechanism:{name}" for name in MECHANISM_BLOCKS),
)


@dataclass(frozen=True)
class UtilityGateConfig:
    ridge_alpha: float = 10.0
    error_quantile: float = 0.75
    minimum_records: int = 48
    minimum_predicted_gain: float = 0.0005


@dataclass(frozen=True)
class UtilityRecords:
    features: np.ndarray
    cost_delta: np.ndarray
    group_ids: np.ndarray
    candidate_keys: np.ndarray
    fold_ids: np.ndarray


@dataclass
class FittedUtilityGate:
    model: Ridge | None
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    error_margin: float
    config: UtilityGateConfig
    diagnostics: Mapping[str, Any]


def pressure_align_candidate_scores(
    dataset: MechanismDataset,
    rigid_score: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reject candidate actions that are neither rigid nor PhasePressure.

    The reference row is the PhasePressure action fixed when the contrast
    dataset is constructed. Rejected candidates are replaced group-wise by an
    exact copy of the rigid scores, so disabling source information preserves
    the target-only policy bit for bit.
    """

    rigid = np.asarray(rigid_score, dtype=float)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    references = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if (
        rigid.shape != (dataset.size,)
        or groups.shape != rigid.shape
        or references.shape != rigid.shape
        or not np.all(np.isfinite(rigid))
        or not candidate_scores
    ):
        raise ValueError("pressure-aligned candidates require aligned scores")

    ordered_groups = tuple(dict.fromkeys(groups.tolist()))
    reference_rows: dict[str, int] = {}
    rigid_rows: dict[str, int] = {}
    for group in ordered_groups:
        rows = np.flatnonzero(groups == group)
        reference = rows[references[rows]]
        if rows.size < 2 or reference.size != 1:
            raise ValueError(
                f"pressure-aligned group requires one reference action: {group}"
            )
        reference_rows[group] = int(reference[0])
        rigid_rows[group] = int(rows[np.argsort(rigid[rows], kind="stable")[0]])

    projected: dict[str, np.ndarray] = {}
    per_candidate: dict[str, dict[str, int]] = {}
    total_changed = 0
    total_aligned = 0
    total_rejected = 0
    for key in sorted(candidate_scores):
        candidate = np.asarray(candidate_scores[key], dtype=float)
        if candidate.shape != rigid.shape or not np.all(np.isfinite(candidate)):
            raise ValueError(f"pressure-aligned candidate is not aligned: {key}")
        constrained = candidate.copy()
        changed = 0
        aligned = 0
        rejected = 0
        for group in ordered_groups:
            rows = np.flatnonzero(groups == group)
            proposed = int(
                rows[np.argsort(candidate[rows], kind="stable")[0]]
            )
            rigid_selected = rigid_rows[group]
            if proposed == rigid_selected:
                continue
            changed += 1
            if proposed == reference_rows[group]:
                aligned += 1
                continue
            constrained[rows] = rigid[rows]
            rejected += 1
        projected[str(key)] = constrained
        per_candidate[str(key)] = {
            "changed_rigid_group_count": changed,
            "pressure_aligned_change_group_count": aligned,
            "off_reference_change_rejected_group_count": rejected,
        }
        total_changed += changed
        total_aligned += aligned
        total_rejected += rejected

    return projected, {
        "protocol": PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
        "candidate_count": len(projected),
        "action_group_count": len(ordered_groups),
        "candidate_group_count": len(projected) * len(ordered_groups),
        "changed_rigid_group_count": total_changed,
        "pressure_aligned_change_group_count": total_aligned,
        "off_reference_change_rejected_group_count": total_rejected,
        "per_candidate": per_candidate,
    }


def pressure_pairwise_candidate_scores(
    dataset: MechanismDataset,
    rigid_score: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert each source mechanism into a PhasePressure-vs-rigid vote.

    A source candidate can only replace the rigid action with the reference
    action. Its pairwise score margin determines the proposal strength; all
    other action scores remain an exact copy of the rigid model.
    """

    rigid = np.asarray(rigid_score, dtype=float)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    references = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if (
        rigid.shape != (dataset.size,)
        or groups.shape != rigid.shape
        or references.shape != rigid.shape
        or not np.all(np.isfinite(rigid))
        or not candidate_scores
    ):
        raise ValueError("pressure-pairwise candidates require aligned scores")

    ordered_groups = tuple(dict.fromkeys(groups.tolist()))
    reference_rows: dict[str, int] = {}
    rigid_rows: dict[str, int] = {}
    for group in ordered_groups:
        rows = np.flatnonzero(groups == group)
        reference = rows[references[rows]]
        if rows.size < 2 or reference.size != 1:
            raise ValueError(
                f"pressure-pairwise group requires one reference action: {group}"
            )
        reference_rows[group] = int(reference[0])
        rigid_rows[group] = int(rows[np.argsort(rigid[rows], kind="stable")[0]])

    projected: dict[str, np.ndarray] = {}
    per_candidate: dict[str, dict[str, float | int]] = {}
    total_eligible = 0
    total_proposed = 0
    margins: list[float] = []
    for key in sorted(candidate_scores):
        candidate = np.asarray(candidate_scores[key], dtype=float)
        if candidate.shape != rigid.shape or not np.all(np.isfinite(candidate)):
            raise ValueError(f"pressure-pairwise candidate is not aligned: {key}")
        constrained = rigid.copy()
        eligible = 0
        proposed = 0
        candidate_margins: list[float] = []
        for group in ordered_groups:
            reference = reference_rows[group]
            rigid_selected = rigid_rows[group]
            if reference == rigid_selected:
                continue
            eligible += 1
            margin = float(candidate[rigid_selected] - candidate[reference])
            if margin <= 1e-12:
                continue
            constrained[reference] = float(rigid[rigid_selected] - margin)
            proposed += 1
            candidate_margins.append(margin)
            margins.append(margin)
        projected[str(key)] = constrained
        per_candidate[str(key)] = {
            "eligible_phase_vs_rigid_group_count": eligible,
            "proposed_phase_group_count": proposed,
            "mean_positive_pairwise_margin": (
                float(np.mean(candidate_margins)) if candidate_margins else 0.0
            ),
        }
        total_eligible += eligible
        total_proposed += proposed

    return projected, {
        "protocol": PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
        "candidate_count": len(projected),
        "action_group_count": len(ordered_groups),
        "candidate_group_count": len(projected) * len(ordered_groups),
        "eligible_phase_vs_rigid_group_count": total_eligible,
        "proposed_phase_group_count": total_proposed,
        "mean_positive_pairwise_margin": (
            float(np.mean(margins)) if margins else 0.0
        ),
        "per_candidate": per_candidate,
    }


def _candidate_block(key: str) -> str:
    parts = str(key).split("|", 1)
    if len(parts) != 2 or parts[1] not in MECHANISM_BLOCKS:
        raise ValueError(f"invalid source-mechanism candidate key: {key}")
    return parts[1]


def _state_matrix(dataset: MechanismDataset) -> np.ndarray:
    feature_index = {
        str(name): index for index, name in enumerate(dataset.feature_names)
    }
    missing = [name for name in STATE_UTILITY_FEATURES if name not in feature_index]
    if missing:
        raise KeyError(f"state utility features are missing: {missing}")
    values = np.asarray(
        dataset.features[
            :, [feature_index[name] for name in STATE_UTILITY_FEATURES]
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("state utility features must be finite")
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        if not np.allclose(
            values[rows], values[rows[0]], rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"state utility feature changes across candidate actions: {group}"
            )
    return values


def build_utility_records(
    dataset: MechanismDataset,
    rigid_score: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
    *,
    actual: np.ndarray | None,
    fold_id: int,
) -> UtilityRecords:
    """Build one record for each candidate that changes a rigid action."""

    rigid = np.asarray(rigid_score, dtype=float)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    if rigid.shape != (dataset.size,) or not np.all(np.isfinite(rigid)):
        raise ValueError("utility rigid scores are not row aligned")
    if not candidate_scores:
        raise ValueError("utility records require source-mechanism candidates")
    labels = None if actual is None else np.asarray(actual, dtype=float)
    if labels is not None and (
        labels.shape != rigid.shape or not np.all(np.isfinite(labels))
    ):
        raise ValueError("utility labels are not row aligned")
    state = _state_matrix(dataset)

    feature_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    for key in sorted(candidate_scores):
        candidate = np.asarray(candidate_scores[key], dtype=float)
        if candidate.shape != rigid.shape or not np.all(np.isfinite(candidate)):
            raise ValueError(f"utility candidate scores are not aligned: {key}")
        block = _candidate_block(key)
        base = _source_expert_gate_records(
            rigid,
            candidate,
            groups,
            target=labels,
        )
        count = int(base["features"].shape[0])
        if count == 0:
            continue
        block_features = np.zeros((count, len(MECHANISM_BLOCKS)), dtype=float)
        block_features[:, MECHANISM_BLOCKS.index(block)] = 1.0
        core_rows = np.asarray(base["core_rows"], dtype=int)
        feature_parts.append(
            np.column_stack(
                (
                    np.asarray(base["features"], dtype=float),
                    state[core_rows],
                    block_features,
                )
            )
        )
        delta_parts.append(np.asarray(base["cost_delta"], dtype=float))
        group_parts.append(np.asarray(base["group_ids"], dtype=str))
        candidate_parts.append(np.full(count, str(key), dtype=object))
        fold_parts.append(np.full(count, int(fold_id), dtype=int))

    if not feature_parts:
        return UtilityRecords(
            features=np.zeros((0, len(UTILITY_FEATURE_NAMES)), dtype=float),
            cost_delta=np.zeros(0, dtype=float),
            group_ids=np.zeros(0, dtype=str),
            candidate_keys=np.zeros(0, dtype=object),
            fold_ids=np.zeros(0, dtype=int),
        )
    features = np.vstack(feature_parts)
    cost_delta = np.concatenate(delta_parts)
    if labels is None:
        cost_delta = np.zeros(features.shape[0], dtype=float)
    if features.shape[1] != len(UTILITY_FEATURE_NAMES):
        raise ValueError("state utility design width changed")
    return UtilityRecords(
        features=features,
        cost_delta=cost_delta,
        group_ids=np.concatenate(group_parts),
        candidate_keys=np.concatenate(candidate_parts),
        fold_ids=np.concatenate(fold_parts),
    )


def concatenate_utility_records(
    records: Sequence[UtilityRecords],
) -> UtilityRecords:
    rows = tuple(records)
    if not rows:
        raise ValueError("cannot concatenate an empty utility record sequence")
    if any(row.features.shape[1] != len(UTILITY_FEATURE_NAMES) for row in rows):
        raise ValueError("utility record schemas differ")
    return UtilityRecords(
        features=np.vstack([row.features for row in rows]),
        cost_delta=np.concatenate([row.cost_delta for row in rows]),
        group_ids=np.concatenate([row.group_ids for row in rows]),
        candidate_keys=np.concatenate([row.candidate_keys for row in rows]),
        fold_ids=np.concatenate([row.fold_ids for row in rows]),
    )


def source_blind_candidate_scores(
    rigid_score: np.ndarray,
    candidate_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    """Build a same-capacity candidate bank with no source-derived evidence."""

    rigid = np.asarray(rigid_score, dtype=float)
    if rigid.ndim != 1 or not np.all(np.isfinite(rigid)):
        raise ValueError("source-blind candidates require finite rigid scores")
    keys = tuple(sorted(str(key) for key in candidate_keys))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("source-blind candidate keys must be unique and nonempty")
    for key in keys:
        _candidate_block(key)
    return {key: rigid.copy() for key in keys}


def build_dense_pressure_pairwise_utility_records(
    dataset: MechanismDataset,
    rigid_score: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
    *,
    actual: np.ndarray | None,
    fold_id: int,
) -> UtilityRecords:
    """Build a PhasePressure-vs-rigid record for every eligible candidate.

    Unlike disagreement-only utility records, this design does not require a
    source model to select PhasePressure before its evidence can be calibrated.
    The label is always the observed cost of PhasePressure minus the observed
    cost of the rigid action in the same counterfactual action group.
    """

    rigid = np.asarray(rigid_score, dtype=float)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    references = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if (
        rigid.shape != (dataset.size,)
        or groups.shape != rigid.shape
        or references.shape != rigid.shape
        or not np.all(np.isfinite(rigid))
        or not candidate_scores
    ):
        raise ValueError("dense pressure-pairwise records require aligned scores")
    labels = None if actual is None else np.asarray(actual, dtype=float)
    if labels is not None and (
        labels.shape != rigid.shape or not np.all(np.isfinite(labels))
    ):
        raise ValueError("dense pressure-pairwise labels are not aligned")
    state = _state_matrix(dataset)
    ordered_group_rows = tuple(
        (group, np.flatnonzero(groups == group))
        for group in dict.fromkeys(groups.tolist())
    )

    feature_rows: list[list[float]] = []
    cost_delta: list[float] = []
    record_groups: list[str] = []
    record_candidates: list[str] = []
    for key in sorted(candidate_scores):
        candidate = np.asarray(candidate_scores[key], dtype=float)
        if candidate.shape != rigid.shape or not np.all(np.isfinite(candidate)):
            raise ValueError(f"dense pressure-pairwise candidate is invalid: {key}")
        block = _candidate_block(key)
        block_features = [0.0] * len(MECHANISM_BLOCKS)
        block_features[MECHANISM_BLOCKS.index(block)] = 1.0
        correction = candidate - rigid
        for group, rows in ordered_group_rows:
            reference_rows = rows[references[rows]]
            if rows.size < 2 or reference_rows.size != 1:
                raise ValueError(
                    f"dense pressure-pairwise group needs one reference: {group}"
                )
            rigid_order = rows[np.argsort(rigid[rows], kind="stable")]
            rigid_choice = int(rigid_order[0])
            reference = int(reference_rows[0])
            if rigid_choice == reference:
                continue
            source_order = rows[np.argsort(candidate[rows], kind="stable")]
            scale = max(
                float(np.ptp(rigid[rows])),
                float(np.ptp(candidate[rows])),
                0.05,
            )
            rigid_margin = float(
                rigid[rigid_order[1]] - rigid[rigid_order[0]]
            )
            source_margin = float(
                candidate[source_order[1]] - candidate[source_order[0]]
            )
            features = [
                float(rigid[reference] - rigid[rigid_choice]) / scale,
                float(candidate[rigid_choice] - candidate[reference]) / scale,
                rigid_margin / scale,
                source_margin / scale,
                float(correction[rigid_choice]) / scale,
                float(correction[reference]) / scale,
                float(np.ptp(correction[rows])) / scale,
                float(np.log(float(rows.size))),
                *state[rigid_choice].tolist(),
                *block_features,
            ]
            if len(features) != len(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES):
                raise RuntimeError("dense pressure-pairwise feature width changed")
            feature_rows.append(features)
            record_groups.append(str(group))
            record_candidates.append(str(key))
            if labels is not None:
                cost_delta.append(float(labels[reference] - labels[rigid_choice]))

    features = np.asarray(feature_rows, dtype=float)
    if not feature_rows:
        features = np.zeros(
            (0, len(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES)), dtype=float
        )
    targets = np.asarray(cost_delta, dtype=float)
    if labels is None:
        targets = np.zeros(features.shape[0], dtype=float)
    return UtilityRecords(
        features=features,
        cost_delta=targets,
        group_ids=np.asarray(record_groups, dtype=str),
        candidate_keys=np.asarray(record_candidates, dtype=object),
        fold_ids=np.full(features.shape[0], int(fold_id), dtype=int),
    )


def _group_balanced_weights(group_ids: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=str)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("utility weights require nonempty group ids")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(float)
    return weights * (weights.size / float(np.sum(weights)))


def _fold_and_group_balanced_weights(
    group_ids: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    """Give each fold equal total weight while retaining equal group weights."""

    groups = np.asarray(group_ids, dtype=str)
    folds = np.asarray(fold_ids, dtype=int)
    if groups.shape != folds.shape or groups.ndim != 1 or groups.size == 0:
        raise ValueError("fold-balanced utility weights require aligned records")
    weights = _group_balanced_weights(groups)
    unique_folds = np.unique(folds)
    for fold in unique_folds:
        rows = folds == fold
        total = float(np.sum(weights[rows]))
        if total <= 0.0:
            raise ValueError("utility fold has no positive record weight")
        weights[rows] /= total
    return weights * (weights.size / float(np.sum(weights)))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    data = np.asarray(values, dtype=float)
    mass = np.asarray(weights, dtype=float)
    if (
        data.ndim != 1
        or mass.shape != data.shape
        or data.size == 0
        or not np.all(np.isfinite(data))
        or not np.all(np.isfinite(mass))
        or np.any(mass < 0.0)
        or float(np.sum(mass)) <= 0.0
        or not 0.0 <= float(quantile) <= 1.0
    ):
        raise ValueError("weighted utility quantile inputs are invalid")
    order = np.argsort(data, kind="stable")
    cumulative = np.cumsum(mass[order])
    threshold = float(quantile) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), data.size - 1)
    return float(data[order[index]])


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> tuple[Ridge, np.ndarray, np.ndarray]:
    mean, scale = _weighted_standardizer(features, weights)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit((features - mean) / scale, target, sample_weight=weights)
    return model, mean, scale


def fit_utility_gate(
    records: UtilityRecords,
    *,
    config: UtilityGateConfig = UtilityGateConfig(),
    balance_folds: bool = False,
) -> FittedUtilityGate:
    """Fit a ridge utility model and calibrate its upper cost bound by fold."""

    features = np.asarray(records.features, dtype=float)
    target = np.asarray(records.cost_delta, dtype=float)
    folds = np.asarray(records.fold_ids, dtype=int)
    if (
        features.ndim != 2
        or features.shape[1] != len(UTILITY_FEATURE_NAMES)
        or target.shape != (features.shape[0],)
        or folds.shape != target.shape
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("utility gate records are invalid")
    unique_folds = np.unique(folds)
    if (
        features.shape[0] < int(config.minimum_records)
        or unique_folds.size < 3
    ):
        diagnostics = {
            "enabled": False,
            "reason": "insufficient_crossfitted_disagreement_records",
            "record_count": int(features.shape[0]),
            "fold_count": int(unique_folds.size),
            "feature_names": list(UTILITY_FEATURE_NAMES),
        }
        return FittedUtilityGate(
            model=None,
            feature_mean=np.zeros(len(UTILITY_FEATURE_NAMES), dtype=float),
            feature_scale=np.ones(len(UTILITY_FEATURE_NAMES), dtype=float),
            error_margin=0.0,
            config=config,
            diagnostics=diagnostics,
        )

    weights = (
        _fold_and_group_balanced_weights(records.group_ids, folds)
        if balance_folds
        else _group_balanced_weights(records.group_ids)
    )
    oof = np.zeros(target.shape, dtype=float)
    covered = np.zeros(target.shape, dtype=bool)
    minimum_fold_training_records = int(config.minimum_records) // 2
    fold_training_counts = {
        int(fold): int(np.count_nonzero(folds != fold)) for fold in unique_folds
    }
    if min(fold_training_counts.values()) < minimum_fold_training_records:
        diagnostics = {
            "enabled": False,
            "reason": "insufficient_fold_training_records",
            "record_count": int(features.shape[0]),
            "fold_count": int(unique_folds.size),
            "fold_training_record_counts": fold_training_counts,
            "minimum_fold_training_records": minimum_fold_training_records,
            "feature_names": list(UTILITY_FEATURE_NAMES),
        }
        return FittedUtilityGate(
            model=None,
            feature_mean=np.zeros(len(UTILITY_FEATURE_NAMES), dtype=float),
            feature_scale=np.ones(len(UTILITY_FEATURE_NAMES), dtype=float),
            error_margin=0.0,
            config=config,
            diagnostics=diagnostics,
        )
    for fold in unique_folds:
        heldout = folds == fold
        training = ~heldout
        model, mean, scale = _fit_ridge(
            features[training],
            target[training],
            weights[training],
            alpha=config.ridge_alpha,
        )
        oof[heldout] = model.predict((features[heldout] - mean) / scale)
        covered[heldout] = True
    if not np.all(covered):
        raise RuntimeError("utility gate cross-fitting left uncovered records")
    residual = target - oof
    error_margin = max(
        (
            _weighted_quantile(residual, weights, float(config.error_quantile))
            if balance_folds
            else float(np.quantile(residual, float(config.error_quantile)))
        ),
        0.0,
    )
    selected = oof + error_margin < -float(config.minimum_predicted_gain)
    model, mean, scale = _fit_ridge(
        features,
        target,
        weights,
        alpha=config.ridge_alpha,
    )
    diagnostics = {
        "enabled": True,
        "reason": "crossfitted_state_utility_model_available",
        "record_count": int(features.shape[0]),
        "group_count": int(np.unique(records.group_ids).size),
        "fold_count": int(unique_folds.size),
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "ridge_alpha": float(config.ridge_alpha),
        "error_quantile": float(config.error_quantile),
        "weighting_protocol": (
            "equal_fold_then_equal_group" if balance_folds else "equal_group"
        ),
        "one_sided_error_margin": error_margin,
        "oof_mae": float(
            np.average(np.abs(target - oof), weights=weights)
            if balance_folds
            else np.mean(np.abs(target - oof))
        ),
        "oof_selected_record_count": int(np.count_nonzero(selected)),
        "oof_selected_record_fraction": float(
            np.average(selected.astype(float), weights=weights)
            if balance_folds
            else np.mean(selected)
        ),
        "oof_selected_mean_cost_delta": (
            float(
                np.average(target[selected], weights=weights[selected])
                if balance_folds
                else np.mean(target[selected])
            )
            if np.any(selected)
            else 0.0
        ),
        "oof_upper_bound_coverage": float(
            np.average(
                (target <= oof + error_margin).astype(float), weights=weights
            )
            if balance_folds
            else np.mean(target <= oof + error_margin)
        ),
    }
    return FittedUtilityGate(
        model=model,
        feature_mean=mean,
        feature_scale=scale,
        error_margin=error_margin,
        config=config,
        diagnostics=diagnostics,
    )


def fit_dense_pressure_pairwise_utility_gate(
    records: UtilityRecords,
    *,
    config: UtilityGateConfig = UtilityGateConfig(),
    balance_folds: bool = False,
) -> FittedUtilityGate:
    """Fit the conservative utility gate on dense fixed-action-pair records."""

    if len(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES) != len(UTILITY_FEATURE_NAMES):
        raise RuntimeError("dense and disagreement utility widths diverged")
    gate = fit_utility_gate(
        records,
        config=config,
        balance_folds=balance_folds,
    )
    diagnostics = dict(gate.diagnostics)
    diagnostics["record_protocol"] = DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL
    diagnostics["feature_names"] = list(DENSE_PRESSURE_PAIRWISE_FEATURE_NAMES)
    if diagnostics.get("reason") == "insufficient_crossfitted_disagreement_records":
        diagnostics["reason"] = "insufficient_crossfitted_fixed_pair_records"
    elif diagnostics.get("reason") == "crossfitted_state_utility_model_available":
        diagnostics["reason"] = "crossfitted_dense_pairwise_utility_model_available"
    return replace(gate, diagnostics=diagnostics)


def apply_utility_gate(
    dataset: MechanismDataset,
    rigid_score: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
    gate: FittedUtilityGate,
    *,
    fold_id: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose at most one source-mechanism proposal per decision state."""

    rigid = np.asarray(rigid_score, dtype=float)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    result = rigid.copy()
    if gate.model is None:
        return result, {
            "gate_enabled": False,
            "selected_group_count": 0,
            "selected_group_fraction": 0.0,
            "selected_candidate_counts": {},
        }
    records = build_utility_records(
        dataset,
        rigid,
        candidate_scores,
        actual=None,
        fold_id=fold_id,
    )
    if records.features.shape[0] == 0:
        return result, {
            "gate_enabled": True,
            "selected_group_count": 0,
            "selected_group_fraction": 0.0,
            "selected_candidate_counts": {},
        }
    predicted = np.asarray(
        gate.model.predict(
            (records.features - gate.feature_mean) / gate.feature_scale
        ),
        dtype=float,
    )
    upper = predicted + float(gate.error_margin)
    threshold = -float(gate.config.minimum_predicted_gain)
    selected_counts: dict[str, int] = {}
    selected_upper: list[float] = []
    selected_groups = 0
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(records.group_ids == group)
        if rows.size == 0:
            continue
        best = int(rows[np.argmin(upper[rows])])
        if float(upper[best]) >= threshold:
            continue
        key = str(records.candidate_keys[best])
        result[groups == group] = np.asarray(candidate_scores[key], dtype=float)[
            groups == group
        ]
        selected_counts[key] = selected_counts.get(key, 0) + 1
        selected_upper.append(float(upper[best]))
        selected_groups += 1
    total_groups = int(np.unique(groups).size)
    return result, {
        "gate_enabled": True,
        "candidate_record_count": int(records.features.shape[0]),
        "selected_group_count": selected_groups,
        "selected_group_fraction": float(selected_groups / max(total_groups, 1)),
        "selected_candidate_counts": dict(sorted(selected_counts.items())),
        "selected_mean_predicted_upper_cost_delta": (
            float(np.mean(selected_upper)) if selected_upper else 0.0
        ),
    }


def apply_dense_pressure_pairwise_utility_gate(
    dataset: MechanismDataset,
    rigid_score: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
    gate: FittedUtilityGate,
    *,
    fold_id: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use source evidence to choose only PhasePressure or exact rigid."""

    rigid = np.asarray(rigid_score, dtype=float)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    references = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    result = rigid.copy()
    if gate.model is None:
        return result, {
            "gate_enabled": False,
            "candidate_record_count": 0,
            "eligible_group_count": 0,
            "selected_group_count": 0,
            "selected_group_fraction": 0.0,
            "selected_candidate_counts": {},
        }
    records = build_dense_pressure_pairwise_utility_records(
        dataset,
        rigid,
        candidate_scores,
        actual=None,
        fold_id=fold_id,
    )
    if records.features.shape[0] == 0:
        return result, {
            "gate_enabled": True,
            "candidate_record_count": 0,
            "eligible_group_count": 0,
            "selected_group_count": 0,
            "selected_group_fraction": 0.0,
            "selected_candidate_counts": {},
        }
    predicted = np.asarray(
        gate.model.predict(
            (records.features - gate.feature_mean) / gate.feature_scale
        ),
        dtype=float,
    )
    upper = predicted + float(gate.error_margin)
    threshold = -float(gate.config.minimum_predicted_gain)
    selected_counts: dict[str, int] = {}
    selected_upper: list[float] = []
    selected_groups = 0
    eligible_groups = tuple(dict.fromkeys(records.group_ids.tolist()))
    record_rows_by_group = {
        group: np.flatnonzero(records.group_ids == group)
        for group in eligible_groups
    }
    action_rows_by_group = {
        group: np.flatnonzero(groups == group) for group in eligible_groups
    }
    for group in eligible_groups:
        record_rows = record_rows_by_group[group]
        best = int(record_rows[np.argmin(upper[record_rows])])
        if float(upper[best]) >= threshold:
            continue
        action_rows = action_rows_by_group[group]
        reference_rows = action_rows[references[action_rows]]
        if reference_rows.size != 1:
            raise ValueError(f"dense utility application lost reference: {group}")
        result[action_rows] = rigid[action_rows]
        reference = int(reference_rows[0])
        result[reference] = np.nextafter(
            float(np.min(rigid[action_rows])),
            -np.inf,
        )
        key = str(records.candidate_keys[best])
        selected_counts[key] = selected_counts.get(key, 0) + 1
        selected_upper.append(float(upper[best]))
        selected_groups += 1
    return result, {
        "gate_enabled": True,
        "candidate_record_count": int(records.features.shape[0]),
        "eligible_group_count": len(eligible_groups),
        "selected_group_count": selected_groups,
        "selected_group_fraction": float(
            selected_groups / max(len(eligible_groups), 1)
        ),
        "selected_candidate_counts": dict(sorted(selected_counts.items())),
        "selected_mean_predicted_upper_cost_delta": (
            float(np.mean(selected_upper)) if selected_upper else 0.0
        ),
    }
