"""Matched LOSO model screen for candidate-specific arrival features."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    MODEL_FAMILY,
    _fit_state_conditioned_oof_fold,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (
    select_ranker_gate,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.movement_arrival_timeline import (
    MOVEMENT_ARRIVAL_FEATURE_NAMES,
    augment_dataset_with_movement_arrivals,
    build_movement_arrival_timeline_context,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3


RESULT_PROTOCOL = "tsc-v112-movement-arrival-matched-model-screen-v1"
BOOTSTRAP_SEED = 112002
MODEL_SPECS = (
    {
        "key": "state_action_k10_s2",
        "family": MODEL_FAMILY,
        "config": {
            "base": {
                "scope": "tls_time_phase_pair",
                "shrinkage": 20.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": "state_action",
            "neighbor_count": 10,
            "local_shrinkage": 2.0,
            "distance_bandwidth": 1.0,
            "uncertainty_quantile": 0.9,
        },
    },
    {
        "key": "state_action_arrival_k10_s2",
        "family": MODEL_FAMILY,
        "config": {
            "base": {
                "scope": "tls_time_phase_pair",
                "shrinkage": 20.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": "state_action_arrival",
            "neighbor_count": 10,
            "local_shrinkage": 2.0,
            "distance_bandwidth": 1.0,
            "uncertainty_quantile": 0.9,
        },
    },
)
SELECTION = {
    "bootstrap": {
        "replicates": 10000,
        "seed": BOOTSTRAP_SEED,
        "unit": "simulator_seed",
    },
    "gate_candidates": [
        {"minimum_context_trust": trust, "risk_multiplier": risk}
        for risk in (0.0, 0.25)
        for trust in (0.0, 0.1, 0.25, 0.5)
    ],
    "selection_rule": {
        "fallback": "retain_phase_pressure_if_no_candidate_is_feasible",
        "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
        "maximum_equal_seed_scenario_mean_delta": -0.0015,
        "maximum_harmful_group_fraction": 0.42,
        "maximum_worst_seed_mean_delta": 0.0075,
        "minimum_retained_groups_per_city": 500,
        "minimum_retained_seed_fraction": 1.0,
    },
}
_PROCESS_CONTEXT: dict[str, Any] | None = None


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1]
        / "traffic_signal"
        / "movement_arrival_timeline.py",
        Path(__file__).resolve().parents[1]
        / "traffic_signal"
        / "state_conditioned_spatiotemporal_latent.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fit_fold(seed: int) -> dict[str, Any]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("movement-arrival model process context is missing")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_state_conditioned_oof_fold(
            heldout_seed=int(seed),
            contrast=_PROCESS_CONTEXT["contrast"],
            model_specs=MODEL_SPECS,
            scenario=_PROCESS_CONTEXT["scenario"],
            expected_action_count=_PROCESS_CONTEXT["expected_action_count"],
            target_name=_PROCESS_CONTEXT["target_name"],
        )


def _bootstrap_seed_means(values: Mapping[int, float]) -> dict[str, Any]:
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(
        ordered, size=(10000, ordered.size), replace=True
    ).mean(axis=1)
    return {
        "observed": float(np.mean(ordered)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_nonnegative": float(np.mean(draws >= 0.0)),
        "replicates": 10000,
        "seed": BOOTSTRAP_SEED,
    }


def _ungated_summary(
    records: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    seed_means = {
        int(seed): float(
            np.mean(
                [
                    float(row["actual_group_normalized_delta"])
                    for row in records
                    if int(row["simulator_seed"]) == int(seed)
                ]
            )
        )
        for seed in seeds
    }
    changed = [row for row in records if bool(row["selected_differs"])]
    return {
        "group_count": len(records),
        "changed_group_count": len(changed),
        "changed_group_fraction": len(changed) / max(len(records), 1),
        "harmful_changed_group_fraction": float(
            np.mean(
                [float(row["actual_group_normalized_delta"]) > 0.0 for row in changed]
            )
        )
        if changed
        else 0.0,
        "equal_seed_mean_normalized_delta": float(np.mean(list(seed_means.values()))),
        "worst_seed_mean_normalized_delta": float(max(seed_means.values())),
        "selected_oracle_fraction": float(
            np.mean([bool(row["selected_is_oracle"]) for row in records])
        ),
        "mean_normalized_regret_to_oracle": float(
            np.mean([float(row["normalized_regret_to_oracle"]) for row in records])
        ),
        "seed_mean_normalized_delta": {
            str(key): value for key, value in sorted(seed_means.items())
        },
        "bootstrap": _bootstrap_seed_means(seed_means),
    }


def run_screen(
    *,
    cache_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    scenario: str,
    seeds: Sequence[int],
    collection_shards: int,
    target_name: str,
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(
        value for value in full_manifest.scenarios if value.scenario == scenario
    )
    if len(selected) != 1:
        raise ValueError("movement-arrival model scenario is not unique")
    manifest = replace(full_manifest, scenarios=selected)
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=tuple(int(value) for value in seeds),
        collection_shards=int(collection_shards),
        workers=int(cache_workers),
    )
    dataset = _merge_city_datasets(bank, (scenario,), city="jinan")
    if (
        dataset.metadata.get("counterfactual_cost_mode") != "halted_queue"
        or target_name not in dataset.targets
    ):
        raise ValueError("movement-arrival model screen requires waiting-aligned target")
    context = build_movement_arrival_timeline_context(
        Path(conversion_root) / scenario / f"{scenario}.sumocfg"
    )
    augmented = augment_dataset_with_movement_arrivals(dataset, context)
    contrast = build_action_contrast_dataset(
        augmented,
        reference_policy="phase_pressure",
        contrast_features=(*CONTRAST_FEATURES_V3, *MOVEMENT_ARRIVAL_FEATURE_NAMES),
    )
    groups = action_group_ids(contrast).astype(str)
    observed_seeds = {
        parse_action_group_seed(group) for group in np.unique(groups)
    }
    action_counts = {
        int(np.count_nonzero(groups == group)) for group in np.unique(groups)
    }
    if observed_seeds != set(int(value) for value in seeds) or len(action_counts) != 1:
        raise ValueError("movement-arrival model action-group contract changed")
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = {
        "contrast": contrast,
        "scenario": scenario,
        "expected_action_count": next(iter(action_counts)),
        "target_name": target_name,
    }
    workers = min(max(int(fold_workers), 1), 20, len(seeds))
    try:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("movement-arrival model screen requires Linux fork")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("fork")
        ) as pool:
            folds = list(pool.map(_fit_fold, (int(value) for value in seeds)))
    finally:
        _PROCESS_CONTEXT = None
    if not all(
        fold["training_validation_disjoint"] and fold["heldout_absent_from_training"]
        for fold in folds
    ):
        raise ValueError("movement-arrival model seed isolation failed")
    records = [
        row
        for fold in folds
        for model in fold["model_results"]
        for row in model["records"]
    ]
    ungated = {
        spec["key"]: _ungated_summary(
            [row for row in records if row["model_key"] == spec["key"]], seeds
        )
        for spec in MODEL_SPECS
    }
    gate = select_ranker_gate(
        records,
        model_specs=MODEL_SPECS,
        seeds=seeds,
        selection=SELECTION,
    )
    best_by_model = {}
    for spec in MODEL_SPECS:
        candidates = [
            row
            for row in gate["grid"]
            if row["model_key"] == spec["key"] and row["feasible"]
        ]
        best_by_model[spec["key"]] = min(
            candidates,
            key=lambda row: (
                row["robust_score"],
                row["mean_delta"],
                row["harmful_group_fraction"],
            ),
            default=None,
        )
    baseline = best_by_model["state_action_k10_s2"]
    arrival = best_by_model["state_action_arrival_k10_s2"]
    matched_gain = (
        float(baseline["mean_delta"] - arrival["mean_delta"])
        if baseline is not None and arrival is not None
        else None
    )
    arrival_gate_passed = bool(
        arrival is not None
        and matched_gain is not None
        and matched_gain >= 0.0005
        and gate["selected"] is not None
        and gate["selected"]["model_key"] == "state_action_arrival_k10_s2"
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "scientific_status": "disclosed_post-v111_development_gate_b_matched_model_screen",
        "source_sha256": _source_sha256(),
        "scenario": scenario,
        "seeds": [int(value) for value in seeds],
        "collection_shards": int(collection_shards),
        "target_name": target_name,
        "cache_audit": cache_audit,
        "movement_arrival_context": context.to_dict(),
        "model_specs": MODEL_SPECS,
        "selection": SELECTION,
        "fold_count": len(folds),
        "action_group_count": int(np.unique(groups).size),
        "action_count_per_group": next(iter(action_counts)),
        "ungated": ungated,
        "gate": gate,
        "best_feasible_by_model": best_by_model,
        "matched_arrival_gain": matched_gain,
        "arrival_feature_gate_passed": arrival_gate_passed,
        "decision": (
            "advance_arrival_features_to_nested_selection"
            if arrival_gate_passed
            else "reject_static_arrival_features_from_cfcmt"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--collection-shards", type=int, required=True)
    parser.add_argument("--target-name", default="prefix_mean_cost_450s")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_screen(
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        scenario=args.scenario,
        seeds=args.seeds,
        collection_shards=args.collection_shards,
        target_name=args.target_name,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "decision": result["decision"],
                "matched_arrival_gain": result["matched_arrival_gain"],
            }
        )
    )


if __name__ == "__main__":
    main()
