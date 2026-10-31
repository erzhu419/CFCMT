"""Audit whether unilateral TSC counterfactual labels retain actionable signal.

The current cache records a global system-cost target for a one-intersection
intervention.  This audit measures the within-action-group signal before any
model is fitted and compares it with local queue mechanisms.  It is intended
to catch network-size dilution and target/credit-assignment mismatch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PRIMARY_TARGET = "interval_cost"
LOCAL_COMPARISON_TARGETS = (
    "next_total_queue",
    "next_green_queue",
    "next_red_queue",
    "next_downstream_occupancy",
    "next_mean_speed",
    "terminal_system_load",
)


def _safe_quantile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, quantile)) if array.size else 0.0


def _best_set(values: np.ndarray) -> set[int]:
    values = np.asarray(values, dtype=float)
    minimum = float(np.min(values))
    tolerance = max(abs(minimum), 1.0) * 1e-12
    return set(np.flatnonzero(values <= minimum + tolerance).tolist())


def _target_signal(groups: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    ranges = []
    relative_ranges = []
    candidate_stds = []
    candidate_counts = []
    for values in groups.values():
        array = np.asarray(values, dtype=float)
        if array.size < 2:
            continue
        value_range = float(np.max(array) - np.min(array))
        level = max(abs(float(np.mean(array))), 1e-6)
        ranges.append(value_range)
        relative_ranges.append(value_range / level)
        candidate_stds.append(float(np.std(array)))
        candidate_counts.append(int(array.size))
    range_array = np.asarray(ranges, dtype=float)
    return {
        "group_count": int(range_array.size),
        "candidate_count_median": _safe_quantile(candidate_counts, 0.50),
        "within_group_range_median": _safe_quantile(ranges, 0.50),
        "within_group_range_q10": _safe_quantile(ranges, 0.10),
        "within_group_range_q90": _safe_quantile(ranges, 0.90),
        "within_group_range_max": float(np.max(range_array))
        if range_array.size
        else 0.0,
        "within_group_relative_range_median": _safe_quantile(
            relative_ranges, 0.50
        ),
        "candidate_std_median": _safe_quantile(candidate_stds, 0.50),
        "exact_zero_range_fraction": float(np.mean(range_array <= 1e-12))
        if range_array.size
        else 0.0,
        "range_below_1e_4_fraction": float(np.mean(range_array < 1e-4))
        if range_array.size
        else 0.0,
        "range_below_1e_3_fraction": float(np.mean(range_array < 1e-3))
        if range_array.size
        else 0.0,
    }


def _target_alignment(
    primary: Mapping[str, Sequence[float]],
    comparison: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    correlations = []
    best_overlap = []
    compared = 0
    for group in sorted(set(primary) & set(comparison)):
        first = np.asarray(primary[group], dtype=float)
        second = np.asarray(comparison[group], dtype=float)
        if first.shape != second.shape or first.size < 2:
            continue
        compared += 1
        best_overlap.append(bool(_best_set(first) & _best_set(second)))
        if float(np.std(first)) > 1e-12 and float(np.std(second)) > 1e-12:
            correlations.append(float(np.corrcoef(first, second)[0, 1]))
    return {
        "compared_group_count": int(compared),
        "nonconstant_group_count": len(correlations),
        "best_action_overlap_fraction": float(np.mean(best_overlap))
        if best_overlap
        else 0.0,
        "within_group_pearson_median": _safe_quantile(correlations, 0.50),
        "within_group_pearson_q10": _safe_quantile(correlations, 0.10),
        "within_group_pearson_q90": _safe_quantile(correlations, 0.90),
    }


def summarize_scenario_datasets(
    datasets: Iterable[MechanismDataset],
) -> dict[str, Any]:
    target_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scenario: str | None = None
    tls_counts: set[int] = set()
    rows = 0
    file_count = 0
    cost_scopes: set[str] = set()
    for dataset in datasets:
        file_count += 1
        rows += int(dataset.size)
        current_scenario = str(dataset.metadata.get("scenario", ""))
        if not current_scenario:
            raise ValueError("counterfactual dataset is missing its scenario")
        if scenario is None:
            scenario = current_scenario
        elif scenario != current_scenario:
            raise ValueError("scenario summary received mixed scenarios")
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
        if groups.shape != (dataset.size,):
            raise ValueError("counterfactual dataset has invalid action_group_ids")
        if "controllable_tls_count" in dataset.metadata:
            tls_counts.add(int(dataset.metadata["controllable_tls_count"]))
        if "counterfactual_cost_scope" in dataset.metadata:
            cost_scopes.add(str(dataset.metadata["counterfactual_cost_scope"]))
        for target, values in dataset.targets.items():
            array = np.asarray(values, dtype=float)
            if array.shape != (dataset.size,):
                raise ValueError(f"target {target!r} is not row aligned")
            for group, value in zip(groups, array, strict=True):
                target_groups[str(target)][str(group)].append(float(value))
    if scenario is None:
        raise ValueError("scenario summary requires at least one dataset")
    if len(tls_counts) > 1:
        raise ValueError(f"inconsistent controllable TLS counts: {sorted(tls_counts)}")
    if len(cost_scopes) > 1:
        raise ValueError(f"inconsistent cost scopes: {sorted(cost_scopes)}")
    if PRIMARY_TARGET not in target_groups:
        raise ValueError(f"counterfactual data is missing {PRIMARY_TARGET!r}")

    targets = {
        target: _target_signal(groups)
        for target, groups in sorted(target_groups.items())
    }
    alignments = {
        target: _target_alignment(
            target_groups[PRIMARY_TARGET], target_groups[target]
        )
        for target in LOCAL_COMPARISON_TARGETS
        if target in target_groups
    }
    tls_count = next(iter(tls_counts), 0)
    primary_range = float(targets[PRIMARY_TARGET]["within_group_range_median"])
    return {
        "scenario": scenario,
        "file_count": int(file_count),
        "row_count": int(rows),
        "action_group_count": len(target_groups[PRIMARY_TARGET]),
        "controllable_tls_count": int(tls_count),
        "counterfactual_cost_scope": next(iter(cost_scopes), None),
        "targets": targets,
        "primary_to_local_alignment": alignments,
        "tls_scaled_primary_range": float(primary_range * max(tls_count, 1)),
    }


def _dilution_summary(scenarios: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        (
            int(summary["controllable_tls_count"]),
            float(
                summary["targets"][PRIMARY_TARGET]["within_group_range_median"]
            ),
        )
        for summary in scenarios.values()
        if int(summary["controllable_tls_count"]) > 0
    ]
    if len(rows) < 2:
        return {
            "scenario_count": len(rows),
            "log_tls_vs_log_primary_range_pearson": 0.0,
            "log_log_slope": 0.0,
        }
    tls = np.log10(np.asarray([row[0] for row in rows], dtype=float))
    signal = np.log10(
        np.maximum(np.asarray([row[1] for row in rows], dtype=float), 1e-12)
    )
    return {
        "scenario_count": len(rows),
        "log_tls_vs_log_primary_range_pearson": float(np.corrcoef(tls, signal)[0, 1]),
        "log_log_slope": float(np.polyfit(tls, signal, 1)[0]),
        "interpretation": (
            "A strongly negative slope indicates that a unilateral action's "
            "global-cost label is diluted as network size grows."
        ),
    }


def audit_counterfactual_signal(cache_root: Path) -> dict[str, Any]:
    cache_root = Path(cache_root)
    files = sorted(cache_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no counterfactual NPZ files under {cache_root}")
    grouped: dict[str, list[MechanismDataset]] = defaultdict(list)
    for path in files:
        dataset = load_mechanism_dataset(path)
        grouped[str(dataset.metadata.get("scenario", ""))].append(dataset)
    scenarios = {
        scenario: summarize_scenario_datasets(datasets)
        for scenario, datasets in sorted(grouped.items())
    }
    return {
        "experiment": "traffic_signal_counterfactual_signal_audit",
        "cache_root": str(cache_root),
        "cache_file_count": len(files),
        "scenario_count": len(scenarios),
        "primary_target": PRIMARY_TARGET,
        "scenarios": scenarios,
        "network_size_dilution": _dilution_summary(scenarios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_counterfactual_signal(args.cache_root)
    atomic_write_json(args.out, result)
    print(
        f"audited {result['cache_file_count']} files across "
        f"{result['scenario_count']} scenarios"
    )


if __name__ == "__main__":
    main()
