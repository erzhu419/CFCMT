"""Cross-fitted target-group selector for pairwise traffic-signal control."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    NUMERICAL_TOLERANCE,
    summarize_deployment_gate,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


CROSS_FIT_FOLD_COUNT = 5
CROSS_FIT_GROUP_COUNT = 60
CROSS_FIT_GROUPS_PER_FOLD = 12
MIN_MEAN_OOF_GAIN = 0.01
MIN_OOF_NONWORSE_FRACTION = 0.50
MIN_WORST_SEED_GAIN = -0.02
MIN_WORST_FOLD_GAIN = -0.03


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_seed_stratified_folds(
    group_records: Sequence[Mapping[str, Any]],
    *,
    expected_group_count: int = CROSS_FIT_GROUP_COUNT,
    fold_count: int = CROSS_FIT_FOLD_COUNT,
) -> dict[str, Any]:
    """Assign unique groups to deterministic, seed-balanced equal folds."""

    expected_group_count = int(expected_group_count)
    fold_count = int(fold_count)
    if expected_group_count <= 0 or fold_count < 2:
        raise ValueError("cross-fit group and fold counts must be positive")
    if expected_group_count % fold_count:
        raise ValueError("cross-fit groups must divide evenly across folds")
    groups_per_fold = expected_group_count // fold_count
    if len(group_records) != expected_group_count:
        raise ValueError(
            f"cross-fit selection requires exactly {expected_group_count} target groups"
        )
    normalized = []
    for row in group_records:
        normalized.append(
            {
                "group_id": str(row["group_id"]),
                "simulator_seed": int(row["simulator_seed"]),
                "snapshot_time_sec": float(row["snapshot_time_sec"]),
                "tls_id": str(row["tls_id"]),
            }
        )
    group_ids = [row["group_id"] for row in normalized]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("cross-fit target groups must be unique")
    if any(
        not np.isfinite(row["snapshot_time_sec"]) for row in normalized
    ):
        raise ValueError("cross-fit snapshot times must be finite")

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in normalized:
        by_seed.setdefault(row["simulator_seed"], []).append(row)
    if len(by_seed) < 2:
        raise ValueError("cross-fit groups must cover multiple simulator seeds")
    fold_counts = [0] * fold_count
    assignments: dict[str, int] = {}
    seed_fold_counts: dict[int, list[int]] = {}
    for seed in sorted(by_seed):
        rows = sorted(
            by_seed[seed],
            key=lambda row: (
                row["snapshot_time_sec"],
                row["tls_id"],
                _stable_digest(row["group_id"]),
            ),
        )
        candidates = []
        for offset in range(fold_count):
            additions = [0] * fold_count
            for index in range(len(rows)):
                additions[(index + offset) % fold_count] += 1
            resulting = [
                fold_counts[index] + additions[index]
                for index in range(fold_count)
            ]
            candidates.append(
                (
                    max(resulting) - min(resulting),
                    max(resulting),
                    sum(value * value for value in resulting),
                    tuple(resulting),
                    offset,
                    additions,
                )
            )
        _, _, _, _, selected_offset, additions = min(candidates)
        for index, row in enumerate(rows):
            assignments[row["group_id"]] = (
                index + selected_offset
            ) % fold_count
        fold_counts = [
            fold_counts[index] + additions[index]
            for index in range(fold_count)
        ]
        seed_fold_counts[seed] = additions
    if fold_counts != [groups_per_fold] * fold_count:
        raise ValueError(f"cross-fit fold balance failed: {fold_counts}")
    return {
        "protocol": (
            "seed-stratified-time-tls-balanced-five-fold-v1"
            if expected_group_count == CROSS_FIT_GROUP_COUNT
            and fold_count == CROSS_FIT_FOLD_COUNT
            else "seed-stratified-time-tls-balanced-equal-fold-v2"
        ),
        "fold_count": fold_count,
        "group_count": expected_group_count,
        "groups_per_fold": groups_per_fold,
        "assignments": assignments,
        "fold_group_counts": fold_counts,
        "seed_group_counts": {
            seed: len(rows) for seed, rows in sorted(by_seed.items())
        },
        "seed_fold_counts": seed_fold_counts,
    }


def target_group_records(
    dataset: MechanismDataset,
    *,
    selected_group_ids: Sequence[str],
    expected_group_count: int = CROSS_FIT_GROUP_COUNT,
) -> list[dict[str, Any]]:
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
    times = np.asarray(dataset.metadata.get("row_times", ()), dtype=float)
    tls_ids = np.asarray(dataset.metadata.get("row_tls", ()), dtype=str)
    if any(array.shape != (dataset.size,) for array in (groups, times, tls_ids)):
        raise ValueError("target fold construction requires row-aligned metadata")
    selected = tuple(sorted(str(value) for value in selected_group_ids))
    if len(selected) != int(expected_group_count) or len(set(selected)) != len(selected):
        raise ValueError(
            f"target fold construction requires {int(expected_group_count)} unique groups"
        )
    records = []
    for group in selected:
        rows = np.flatnonzero(groups == group)
        if rows.size == 0:
            raise ValueError(f"target group is absent from dataset: {group}")
        group_times = times[rows]
        group_tls = set(str(value) for value in tls_ids[rows])
        if not np.allclose(group_times, group_times[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"target group spans multiple snapshot times: {group}")
        if len(group_tls) != 1:
            raise ValueError(f"target group spans multiple TLS IDs: {group}")
        parts = group.split(":")
        seed_parts = [part for part in parts if part.startswith("seed")]
        if len(seed_parts) != 1:
            raise ValueError(f"target group has no unique seed: {group}")
        records.append(
            {
                "group_id": group,
                "simulator_seed": int(seed_parts[0][4:]),
                "snapshot_time_sec": float(group_times[0]),
                "tls_id": next(iter(group_tls)),
            }
        )
    return records


def select_cross_fitted_pairwise(
    oof_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(oof_rows) != CROSS_FIT_GROUP_COUNT:
        raise ValueError("cross-fit gate requires exactly 60 OOF rows")
    group_ids = [str(row["group_id"]) for row in oof_rows]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("every target group must have exactly one OOF row")
    folds = np.asarray([int(row["fold_index"]) for row in oof_rows], dtype=int)
    seeds = np.asarray([int(row["simulator_seed"]) for row in oof_rows], dtype=int)
    gains = np.asarray([float(row["paired_gain"]) for row in oof_rows], dtype=float)
    if not np.all(np.isfinite(gains)):
        raise ValueError("OOF paired gains must be finite")
    if set(folds.tolist()) != set(range(CROSS_FIT_FOLD_COUNT)):
        raise ValueError("OOF rows do not cover all frozen folds")
    fold_counts = {fold: int(np.sum(folds == fold)) for fold in sorted(set(folds))}
    if set(fold_counts.values()) != {CROSS_FIT_GROUPS_PER_FOLD}:
        raise ValueError(f"OOF fold counts changed: {fold_counts}")
    if len(set(seeds.tolist())) < 2:
        raise ValueError("OOF rows must cover multiple simulator seeds")

    mean_gain = float(np.mean(gains))
    nonworse_fraction = float(np.mean(gains >= -NUMERICAL_TOLERANCE))
    seed_mean_gains = {
        int(seed): float(np.mean(gains[seeds == seed])) for seed in sorted(set(seeds))
    }
    fold_mean_gains = {
        int(fold): float(np.mean(gains[folds == fold]))
        for fold in range(CROSS_FIT_FOLD_COUNT)
    }
    worst_seed_gain = min(seed_mean_gains.values())
    worst_fold_gain = min(fold_mean_gains.values())
    gate = {
        "exactly_sixty_unique_oof_groups": len(set(group_ids)) == 60,
        "mean_oof_gain_at_least_0_01": (
            mean_gain + NUMERICAL_TOLERANCE >= MIN_MEAN_OOF_GAIN
        ),
        "oof_nonworse_fraction_at_least_0_50": (
            nonworse_fraction + NUMERICAL_TOLERANCE
            >= MIN_OOF_NONWORSE_FRACTION
        ),
        "worst_seed_gain_at_least_minus_0_02": (
            worst_seed_gain + NUMERICAL_TOLERANCE >= MIN_WORST_SEED_GAIN
        ),
        "worst_fold_gain_at_least_minus_0_03": (
            worst_fold_gain + NUMERICAL_TOLERANCE >= MIN_WORST_FOLD_GAIN
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "five-fold-oof-seed-robust-pairwise-gate-v1",
        "selected_family": (
            PAIRWISE_PREFERENCE_FAMILY
            if gate["passed"]
            else GROUP_NORMALIZED_RIGID_FAMILY
        ),
        "oof_group_count": len(oof_rows),
        "mean_oof_paired_gain": mean_gain,
        "oof_nonworse_fraction": nonworse_fraction,
        "seed_mean_gains": seed_mean_gains,
        "worst_seed_gain": worst_seed_gain,
        "fold_mean_gains": fold_mean_gains,
        "worst_fold_gain": worst_fold_gain,
        "gate": gate,
    }


def summarize_cross_fitted_deployment(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    evaluation_regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    summary = summarize_deployment_gate(
        decisions=decisions,
        evaluation_regrets=evaluation_regrets,
    )
    summary["protocol"] = "frozen-six-city-cross-fitted-pairwise-selector-v1"
    summary["decision"] = (
        "freeze_cross_fitted_selector_for_salt_lake"
        if summary["passed"]
        else "reject_cross_fitted_pairwise_deployment"
    )
    return summary
