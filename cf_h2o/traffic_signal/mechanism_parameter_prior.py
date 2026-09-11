"""Mechanism-block parameter priors for target-city action adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.action_contrast import action_group_ids
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


MECHANISM_ACTION_FEATURES: Mapping[str, tuple[str, ...]] = {
    "queue_service": (
        "delta_green_q",
        "delta_red_q",
        "delta_service_pressure",
        "delta_red_pressure",
        "delta_movement_service_pressure",
        "delta_green_q_max",
    ),
    "spillback": (
        "delta_green_down_q",
        "delta_green_down_occ",
        "delta_green_down_q_max",
        "delta_green_down_occ_max",
        "delta_green_movement_down_occ_mean",
    ),
    "mobility": (
        "delta_green_speed_mean",
        "delta_red_speed_mean",
    ),
    "execution": (
        "phase_duration_norm",
        "delta_switch_indicator",
        "delta_clearance_fraction",
        "delta_current_phase_overlap",
        "delta_green_link_ratio",
        "delta_green_lane_ratio",
    ),
    "network_propagation": (
        "delta_green_signal_down_q_mean",
        "delta_green_signal_down_occ_mean",
        "delta_green_signal_reach_ratio",
        "delta_green_corridor_pressure",
    ),
    "right_of_way": (
        "delta_protected_green_ratio",
        "delta_permissive_green_ratio",
        "delta_green_shared_receiver_ratio",
        "delta_green_uncontrolled_major_merge_ratio",
        "delta_protected_uncontrolled_major_merge_ratio",
        "delta_permissive_uncontrolled_major_merge_ratio",
        "delta_green_receiver_competition_mean",
    ),
}

MECHANISM_STATE_CONDITIONERS: Mapping[str, str] = {
    "queue_service": "queue_concentration",
    "spillback": "mean_occ",
    "mobility": "mean_speed",
    "execution": "current_green_elapsed_norm",
    "network_propagation": "graph_focal_to_neighbor_q_ratio",
    "right_of_way": "neighbor_phase_observation_ratio",
}

CONDITIONAL_MECHANISM_STATE_FEATURES: Mapping[str, tuple[str, ...]] = {
    "queue_service": ("queue_concentration",),
    "spillback": ("mean_occ",),
    "mobility": ("mean_speed",),
    "execution": (
        "current_green_elapsed_norm",
        "neighbor_switching_fraction",
        "neighbor_clearance_fraction_mean",
    ),
    "network_propagation": (
        "graph_focal_to_neighbor_q_ratio",
        "neighbor_current_protected_ratio_mean",
        "neighbor_current_permissive_ratio_mean",
        "neighbor_switching_fraction",
    ),
    "right_of_way": (
        "neighbor_phase_observation_ratio",
        "neighbor_current_protected_ratio_mean",
        "neighbor_current_permissive_ratio_mean",
        "neighbor_current_green_elapsed_norm_mean",
        "neighbor_switching_fraction",
        "neighbor_clearance_fraction_mean",
    ),
}


@dataclass(frozen=True)
class MechanismParameterDesign:
    values: np.ndarray
    feature_names: tuple[str, ...]
    block_indices: Mapping[str, np.ndarray]


def mechanism_parameter_design(dataset: MechanismDataset) -> MechanismParameterDesign:
    """Build reference-centered main effects and one state interaction per block."""

    return _mechanism_parameter_design(
        dataset,
        {
            block: (conditioner,)
            for block, conditioner in MECHANISM_STATE_CONDITIONERS.items()
        },
    )


def conditional_mechanism_parameter_design(
    dataset: MechanismDataset,
) -> MechanismParameterDesign:
    """Add deterministic local right-of-way and execution interactions."""

    return _mechanism_parameter_design(dataset, CONDITIONAL_MECHANISM_STATE_FEATURES)


def _mechanism_parameter_design(
    dataset: MechanismDataset,
    state_features: Mapping[str, tuple[str, ...]],
) -> MechanismParameterDesign:
    if set(state_features) != set(MECHANISM_ACTION_FEATURES):
        raise ValueError("mechanism state-conditioner blocks changed")

    feature_index = {str(name): index for index, name in enumerate(dataset.feature_names)}
    requested = {
        name
        for values in MECHANISM_ACTION_FEATURES.values()
        for name in values
    } | {
        name
        for values in state_features.values()
        for name in values
    }
    missing = sorted(requested - set(feature_index))
    if missing:
        raise KeyError(f"mechanism parameter design is missing features: {missing}")

    columns: list[np.ndarray] = []
    names: list[str] = []
    blocks: dict[str, np.ndarray] = {}
    for block, action_names in MECHANISM_ACTION_FEATURES.items():
        start = len(columns)
        conditioners = {
            name: np.asarray(dataset.features[:, feature_index[name]], dtype=float)
            for name in state_features[block]
        }
        for action_name in action_names:
            action = np.asarray(
                dataset.features[:, feature_index[action_name]], dtype=float
            )
            columns.append(action)
            names.append(action_name)
            for conditioner_name, conditioner in conditioners.items():
                columns.append(action * conditioner)
                names.append(f"{action_name}*{conditioner_name}")
        blocks[block] = np.arange(start, len(columns), dtype=int)

    values = np.column_stack(columns)
    groups = np.asarray(action_group_ids(dataset), dtype=str)
    references = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if references.shape != (dataset.size,):
        raise ValueError("mechanism parameter design requires is_reference metadata")
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        reference_rows = rows[references[rows]]
        if reference_rows.size != 1:
            raise ValueError("each action group must contain one reference row")
        values[rows] -= values[int(reference_rows[0])]
    if not np.all(np.isfinite(values)):
        raise ValueError("mechanism parameter design contains non-finite values")
    if not np.allclose(values[references], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("reference-centered mechanism design is not zero at references")
    return MechanismParameterDesign(
        values=values,
        feature_names=tuple(names),
        block_indices=blocks,
    )


def equal_city_rms_scale(designs: Sequence[np.ndarray]) -> np.ndarray:
    """Compute a label-free feature scale with equal weight per city."""

    matrices = [np.asarray(values, dtype=float) for values in designs]
    if not matrices or any(values.ndim != 2 or values.shape[0] < 1 for values in matrices):
        raise ValueError("feature scaling requires nonempty two-dimensional city designs")
    width = matrices[0].shape[1]
    if any(values.shape[1] != width for values in matrices):
        raise ValueError("city mechanism designs have different widths")
    second_moment = np.mean(
        np.vstack([np.mean(values * values, axis=0) for values in matrices]),
        axis=0,
    )
    scale = np.sqrt(np.maximum(second_moment, 1e-12))
    return np.where(scale > 1e-6, scale, 1.0)


def group_balanced_weights(groups: Sequence[str]) -> np.ndarray:
    values = np.asarray(groups, dtype=str)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("group weights require a nonempty one-dimensional array")
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(float)
    return weights / float(np.sum(weights))


def ridge_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    *,
    ridge_l2: float,
    prior: np.ndarray | None = None,
    prior_indices: Sequence[int] = (),
    prior_strength: float = 0.0,
) -> np.ndarray:
    """Fit weighted ridge with an optional coefficient prior on one block."""

    features = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    if features.ndim != 2 or features.shape[0] != target.size:
        raise ValueError("ridge features and target are not row aligned")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
        raise ValueError("ridge inputs must be finite")
    if float(ridge_l2) < 0.0 or float(prior_strength) < 0.0:
        raise ValueError("ridge penalties must be nonnegative")
    weights = group_balanced_weights(groups)
    if weights.shape != target.shape:
        raise ValueError("ridge groups are not row aligned")
    width = features.shape[1]
    diagonal = np.full(width, float(ridge_l2), dtype=float)
    right = features.T @ (weights * target)
    indices = np.asarray(tuple(int(value) for value in prior_indices), dtype=int)
    if float(prior_strength) > 0.0:
        prior_values = np.asarray(prior, dtype=float).reshape(-1)
        if prior_values.shape != (width,):
            raise ValueError("coefficient prior width changed")
        if indices.size < 1 or np.any(indices < 0) or np.any(indices >= width):
            raise ValueError("coefficient prior indices are invalid")
        diagonal[indices] += float(prior_strength)
        right[indices] += float(prior_strength) * prior_values[indices]
    normal = features.T @ (weights[:, None] * features)
    normal.flat[:: width + 1] += diagonal
    try:
        return np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(normal, right, rcond=None)[0]


def permute_group_targets(
    target: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
    references: Sequence[bool] | None = None,
) -> np.ndarray:
    """Create a matched placebo by permuting complete equal-size action groups."""

    values = np.asarray(target, dtype=float).reshape(-1)
    group_values = np.asarray(groups, dtype=str)
    if values.shape != group_values.shape:
        raise ValueError("placebo target and groups are not row aligned")
    reference_mask = (
        None if references is None else np.asarray(references, dtype=bool)
    )
    if reference_mask is not None and reference_mask.shape != values.shape:
        raise ValueError("placebo references are not row aligned")
    ordered = tuple(dict.fromkeys(group_values.tolist()))
    rows = {group: np.flatnonzero(group_values == group) for group in ordered}
    buckets: dict[int, list[str]] = {}
    for group in ordered:
        buckets.setdefault(int(rows[group].size), []).append(group)
    output = np.empty_like(values)
    generator = np.random.default_rng(int(seed))
    for size, names in sorted(buckets.items()):
        if len(names) < 2:
            source_order = names
        else:
            permutation = generator.permutation(len(names))
            if np.array_equal(permutation, np.arange(len(names))):
                permutation = np.roll(permutation, 1)
            source_order = [names[int(index)] for index in permutation]
        for destination, source in zip(names, source_order, strict=True):
            if rows[destination].size != size or rows[source].size != size:
                raise ValueError("placebo action-group shape changed")
            output[rows[destination]] = values[rows[source]]
    if reference_mask is not None:
        for group in ordered:
            group_rows = rows[group]
            reference_rows = group_rows[reference_mask[group_rows]]
            if reference_rows.size != 1:
                raise ValueError("placebo action group must contain one reference")
            output[group_rows] -= output[int(reference_rows[0])]
    return output
