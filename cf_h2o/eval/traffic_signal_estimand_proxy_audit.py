"""Audit whether local TSC labels improve source-city action transfer.

This is an offline diagnostic, not a closed-loop performance result. Each
estimand trains the same rigid causal action model on five source city groups;
the selected action is then scored against the held-out city's matched global
SUMO counterfactual cost.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.action_contrast import (
    build_action_contrast_dataset,
    select_source_only_reference_policy,
)
from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    ActionAdvantageConfig,
    PairwiseActionAdvantageRegressor,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import (
    atomic_write_json,
    load_mechanism_dataset,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


ESTIMAND_PROTOCOL = "source-city-loo-local-vs-global-proxy-v1"
DEFAULT_ESTIMANDS = (
    "global_interval_cost",
    "next_total_queue",
    "next_red_queue",
    "local_composite",
    "hybrid_global_local",
)
COMPONENT_NAMES = (
    "interval_cost",
    "next_total_queue",
    "next_red_queue",
    "next_downstream_occupancy",
    "next_mean_speed",
)
LOCAL_WEIGHTS = {
    "next_total_queue": 0.55,
    "next_red_queue": 0.20,
    "next_downstream_occupancy": 0.20,
    "next_mean_speed": -0.05,
}


def _subset_rows(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    mask = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(mask)
    aligned_metadata = {}
    for key in ("action_group_ids", "row_tls", "row_times", "candidate_states"):
        values = list(dataset.metadata.get(key, ()))
        if len(values) != dataset.size:
            raise ValueError(f"estimand audit requires row-aligned {key}")
        aligned_metadata[key] = [values[index] for index in rows]
    subset = dataset.subset(mask)
    return replace(subset, metadata=aligned_metadata)


def _merge_cached_datasets(
    paths: Sequence[Path],
    *,
    scenario_city_groups: Mapping[str, str],
) -> MechanismDataset:
    datasets = [load_mechanism_dataset(path) for path in paths]
    datasets = [dataset for dataset in datasets if dataset.size]
    if not datasets:
        raise ValueError("estimand audit cache contains no counterfactual rows")
    first = datasets[0]
    for dataset in datasets:
        if (
            dataset.feature_names != first.feature_names
            or dataset.context_names != first.context_names
            or set(dataset.priors) != set(first.priors)
            or set(dataset.targets) != set(first.targets)
        ):
            raise ValueError("estimand audit cache schemas are inconsistent")
        scenario = str(dataset.metadata.get("scenario", ""))
        if scenario not in scenario_city_groups:
            raise ValueError(f"cache scenario is absent from manifest: {scenario!r}")
    metadata = {}
    for key in ("action_group_ids", "row_tls", "row_times", "candidate_states"):
        metadata[key] = [
            value
            for dataset in datasets
            for value in dataset.metadata.get(key, ())
        ]
    domains = np.concatenate(
        [
            np.repeat(
                str(scenario_city_groups[str(dataset.metadata["scenario"])]),
                dataset.size,
            )
            for dataset in datasets
        ]
    )
    return MechanismDataset(
        feature_names=first.feature_names,
        features=np.vstack([dataset.features for dataset in datasets]),
        context_names=first.context_names,
        context=np.vstack([dataset.context for dataset in datasets]),
        priors={
            name: np.concatenate([dataset.priors[name] for dataset in datasets])
            for name in first.priors
        },
        targets={
            name: np.concatenate([dataset.targets[name] for dataset in datasets])
            for name in first.targets
        },
        domains=domains,
        metadata=metadata,
    )


def _median_positive_group_range(
    values: np.ndarray,
    groups: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if values.shape != groups.shape:
        raise ValueError("values and groups must be row aligned")
    ranges = []
    for group in np.unique(groups):
        group_range = float(np.ptp(values[groups == group]))
        if group_range > 1e-12:
            ranges.append(group_range)
    return max(float(np.median(ranges)) if ranges else 0.0, 1e-8)


def _component_scales(dataset: MechanismDataset) -> dict[str, float]:
    groups = np.asarray(dataset.metadata["action_group_ids"], dtype=str)
    return {
        name: _median_positive_group_range(dataset.targets[name], groups)
        for name in COMPONENT_NAMES
    }


def _estimand_values(
    dataset: MechanismDataset,
    estimand: str,
    *,
    source_scales: Mapping[str, float],
) -> np.ndarray:
    targets = dataset.targets
    local = sum(
        float(weight)
        * np.asarray(targets[name], dtype=float)
        / float(source_scales[name])
        for name, weight in LOCAL_WEIGHTS.items()
    )
    if estimand == "global_interval_cost":
        return np.asarray(targets["interval_cost"], dtype=float)
    if estimand == "next_total_queue":
        return np.asarray(targets["next_total_queue"], dtype=float)
    if estimand == "next_red_queue":
        return np.asarray(targets["next_red_queue"], dtype=float)
    if estimand == "local_composite":
        return np.asarray(local, dtype=float)
    if estimand == "hybrid_global_local":
        global_normalized = (
            np.asarray(targets["interval_cost"], dtype=float)
            / float(source_scales["interval_cost"])
        )
        return 0.25 * global_normalized + 0.75 * local
    raise ValueError(f"unknown estimand: {estimand}")


def _replace_interval_estimand(
    dataset: MechanismDataset,
    estimand: str,
    *,
    source_scales: Mapping[str, float],
) -> MechanismDataset:
    targets = {
        name: np.asarray(values, dtype=float).copy()
        for name, values in dataset.targets.items()
    }
    targets["interval_cost"] = _estimand_values(
        dataset,
        estimand,
        source_scales=source_scales,
    )
    return replace(dataset, targets=targets)


def _group_ranking_metrics(
    *,
    prediction: np.ndarray,
    global_cost_delta: np.ndarray,
    groups: np.ndarray,
    is_reference: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=float)
    global_cost_delta = np.asarray(global_cost_delta, dtype=float)
    groups = np.asarray(groups, dtype=str)
    is_reference = np.asarray(is_reference, dtype=bool)
    if not (
        prediction.shape
        == global_cost_delta.shape
        == groups.shape
        == is_reference.shape
    ):
        raise ValueError("ranking audit arrays must be row aligned")
    regrets = []
    reference_deltas = []
    oracle_actions = []
    oracle_cost_ties = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        references = rows[is_reference[rows]]
        if references.size != 1:
            raise ValueError(f"action group {group!r} has {references.size} references")
        selected = int(rows[int(np.argmin(prediction[rows]))])
        oracle = int(rows[int(np.argmin(global_cost_delta[rows]))])
        scale = max(float(np.ptp(global_cost_delta[rows])), 1e-8)
        regrets.append(
            (float(global_cost_delta[selected]) - float(global_cost_delta[oracle]))
            / scale
        )
        reference_deltas.append(
            (
                float(global_cost_delta[selected])
                - float(global_cost_delta[int(references[0])])
            )
            / scale
        )
        oracle_actions.append(selected == oracle)
        oracle_cost_ties.append(
            abs(float(global_cost_delta[selected]) - float(global_cost_delta[oracle]))
            <= 1e-12
        )
    return {
        "group_count": len(regrets),
        "normalized_regret": float(np.mean(regrets)),
        "normalized_delta_vs_reference": float(np.mean(reference_deltas)),
        "oracle_action_rate": float(np.mean(oracle_actions)),
        "oracle_cost_tie_rate": float(np.mean(oracle_cost_ties)),
    }


def audit_estimands(
    dataset: MechanismDataset,
    *,
    estimands: Sequence[str] = DEFAULT_ESTIMANDS,
    max_iter: int = 60,
) -> dict[str, Any]:
    estimands = tuple(dict.fromkeys(str(value) for value in estimands))
    unknown = set(estimands) - set(DEFAULT_ESTIMANDS)
    if unknown:
        raise ValueError(f"unknown estimands: {sorted(unknown)}")
    city_groups = tuple(sorted(str(value) for value in np.unique(dataset.domains)))
    folds = {estimand: [] for estimand in estimands}
    for heldout in city_groups:
        source = _subset_rows(dataset, dataset.domains != heldout)
        target = _subset_rows(dataset, dataset.domains == heldout)
        prior_policy, _ = select_source_only_reference_policy(
            source,
            target_name="interval_cost",
        )
        source_scales = _component_scales(source)
        global_contrast = build_action_contrast_dataset(
            target,
            reference_policy=prior_policy,
        )
        global_delta = np.asarray(
            global_contrast.targets["interval_cost"], dtype=float
        )
        for estimand in estimands:
            source_estimand = build_action_contrast_dataset(
                _replace_interval_estimand(
                    source,
                    estimand,
                    source_scales=source_scales,
                ),
                reference_policy=prior_policy,
            )
            target_estimand = build_action_contrast_dataset(
                _replace_interval_estimand(
                    target,
                    estimand,
                    source_scales=source_scales,
                ),
                reference_policy=prior_policy,
            )
            model = PairwiseActionAdvantageRegressor(
                causal=True,
                causal_feature_names=CAUSAL_RIGID_RANKING_PARENTS,
                config=ActionAdvantageConfig(max_iter=int(max_iter)),
            )
            fit = model.fit(source_estimand)
            prediction = model.predict(target_estimand)["control_cost"]["mean"]
            metrics = _group_ranking_metrics(
                prediction=prediction,
                global_cost_delta=global_delta,
                groups=np.asarray(global_contrast.metadata["action_group_ids"]),
                is_reference=np.asarray(global_contrast.metadata["is_reference"]),
            )
            folds[estimand].append(
                {
                    "heldout_city_group": heldout,
                    "source_city_groups": [
                        value for value in city_groups if value != heldout
                    ],
                    "reference_policy": prior_policy,
                    "source_component_scales": source_scales,
                    "fit": fit,
                    **metrics,
                }
            )
    summary = []
    for estimand in estimands:
        rows = folds[estimand]
        effects = np.asarray(
            [float(row["normalized_delta_vs_reference"]) for row in rows]
        )
        regrets = np.asarray([float(row["normalized_regret"]) for row in rows])
        improving = int(np.sum(effects < -1e-12))
        summary.append(
            {
                "estimand": estimand,
                "mean_city_normalized_regret": float(np.mean(regrets)),
                "median_city_normalized_regret": float(np.median(regrets)),
                "mean_city_normalized_delta_vs_reference": float(np.mean(effects)),
                "median_city_normalized_delta_vs_reference": float(np.median(effects)),
                "improving_city_groups": improving,
                "city_group_count": len(rows),
                "broadly_improving": bool(
                    float(np.mean(effects)) < -1e-12
                    and float(np.median(effects)) < -1e-12
                    and improving >= int(math.ceil(len(rows) / 2.0))
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            float(row["mean_city_normalized_regret"]),
            str(row["estimand"]),
        )
    )
    return {
        "experiment": "traffic_signal_estimand_proxy_audit",
        "protocol": ESTIMAND_PROTOCOL,
        "setting": {
            "rows": dataset.size,
            "action_groups": len(set(dataset.metadata["action_group_ids"])),
            "city_groups": list(city_groups),
            "estimands": list(estimands),
            "max_iter": int(max_iter),
            "model": "rigid_causal_pairwise_advantage",
            "evaluation_target": "heldout_global_interval_cost",
            "local_weights": dict(LOCAL_WEIGHTS),
            "claim_scope": "offline_estimand_proxy_only_not_closed_loop_control",
        },
        "summary": summary,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cf_h2o/config/traffic_signal_cross_city_v1.json"),
    )
    parser.add_argument("--estimands", nargs="+", default=list(DEFAULT_ESTIMANDS))
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_traffic_signal_manifest(args.manifest)
    paths = sorted(args.cache_root.rglob("*.npz"))
    dataset = _merge_cached_datasets(
        paths,
        scenario_city_groups=manifest.city_groups,
    )
    result = audit_estimands(
        dataset,
        estimands=args.estimands,
        max_iter=args.max_iter,
    )
    result["setting"]["cache_root"] = str(args.cache_root)
    result["setting"]["cache_file_count"] = len(paths)
    result["setting"]["manifest"] = str(args.manifest)
    atomic_write_json(args.out, result)
    print(
        f"audited {len(result['summary'])} estimands across "
        f"{len(result['setting']['city_groups'])} city groups -> {args.out}"
    )


if __name__ == "__main__":
    main()
