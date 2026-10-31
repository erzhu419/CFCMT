"""Freeze same-information external-city baselines before held-out collection."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    FOLD_COUNT,
    GROUPS_PER_FOLD,
    SELECTION_SEED,
    TARGET_GROUP_BUDGET,
    TRAINING_GROUPS_PER_FOLD,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _select_city_adaptation_groups,
    _sha256,
    _validate_protocol,
    assign_scenario_seed_stratified_folds,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    evaluate_target_action_groups,
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


SOURCE_SEEDS = (2027, 3037, 4047)
SOURCE_COLLECTION_SHARDS = 16
BENCHMARK_FAMILIES = (
    "simulator",
    "dense",
    "dense_advantage",
    "causal_rigid_advantage",
    "cfcmt_mechanism",
    "causal_target_only",
)
MODEL_ARTIFACT_PROTOCOL = "cfcmt-external-city-same-information-baselines-v1"
_BASELINE_FOLD_STATE: dict[str, Any] | None = None


def _equal_scenario_family_evaluation(
    validation,
    *,
    models,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    groups = set(str(value) for value in validation.metadata["action_group_ids"])
    scenario_results = {}
    for scenario in scenarios:
        scenario_groups = {
            group for group in groups if group.startswith(f"{scenario}:")
        }
        if not scenario_groups:
            raise ValueError(f"baseline OOF fold has no groups from {scenario}")
        subset = _group_subset_v3(
            validation,
            selected_groups=scenario_groups,
            metadata_updates={"target_data_role": "external_baseline_oof_scenario"},
        )
        scenario_results[str(scenario)] = evaluate_target_action_groups(
            subset,
            models=models,
            excluded_group_ids=(),
        )
    families = {}
    for family in BENCHMARK_FAMILIES:
        rows = [
            scenario_results[str(scenario)]["families"][family]
            for scenario in scenarios
        ]
        families[family] = {
            "equal_scenario_mean_normalized_action_regret": float(
                np.mean([row["mean_normalized_action_regret"] for row in rows])
            ),
            "equal_scenario_optimal_action_rate": float(
                np.mean([row["optimal_action_rate"] for row in rows])
            ),
            "scenario_group_counts": {
                str(scenario): int(
                    scenario_results[str(scenario)]["evaluation_group_count"]
                )
                for scenario in scenarios
            },
        }
    return {
        "protocol": "equal-scenario-weighted-external-baseline-oof-v1",
        "scenarios": scenario_results,
        "families": families,
    }


def _fit_baseline_fold(
    fold_index: int,
    *,
    fit_bank,
    target_key: str,
    city_groups,
    source_rule_costs,
    source_rule_specs,
    selected_group_ids: Sequence[str],
    fold_assignment: Mapping[str, Any],
    selected_dataset,
    scenarios: Sequence[str],
    source_scenarios: Sequence[str],
    source_city_groups: Sequence[str],
    external_cities: Sequence[str],
    city: str,
    city_static_context: Sequence[float],
) -> dict[str, Any]:
    fold_started = time.monotonic()
    validation_ids = tuple(
        sorted(
            group
            for group, fold in fold_assignment["assignments"].items()
            if int(fold) == int(fold_index)
        )
    )
    validation_set = set(validation_ids)
    training_ids = tuple(
        group for group in selected_group_ids if group not in validation_set
    )
    if (
        len(training_ids) != TRAINING_GROUPS_PER_FOLD
        or len(validation_ids) != GROUPS_PER_FOLD
    ):
        raise ValueError("external baseline fold sizes changed")
    models = fit_target_screening_models(
        fit_bank,
        target_key,
        fit_config=MechanismFitConfig(),
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
        model_families=BENCHMARK_FAMILIES,
        target_group_budget=len(training_ids),
        target_adaptation_selection_seed=SELECTION_SEED,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=5,
        scenario_city_groups=city_groups,
        target_adaptation_group_ids=training_ids,
        target_static_context=city_static_context,
    )
    diagnostics = models.diagnostics
    expected_domains = set(source_city_groups) | {city}
    if (
        diagnostics.get("heldout_city_scenarios") != [target_key]
        or set(diagnostics.get("source_scenarios", ())) != set(source_scenarios)
        or set(diagnostics.get("source_domains", ())) != expected_domains
        or any(
            external_city in set(diagnostics.get("source_domains", ()))
            for external_city in set(external_cities) - {city}
        )
    ):
        raise ValueError("external baseline target/source isolation failed")
    validation = _group_subset_v3(
        selected_dataset,
        selected_groups=validation_set,
        metadata_updates={
            "target_data_role": "external_baseline_cross_fit_oof",
            "cross_fit_fold_index": int(fold_index),
        },
    )
    oof_evaluation = _equal_scenario_family_evaluation(
        validation,
        models=models,
        scenarios=scenarios,
    )
    elapsed = float(time.monotonic() - fold_started)
    print(
        "CFCMT_EXTERNAL_BASELINE_FREEZE_PROGRESS "
        f"city={city} fold={int(fold_index) + 1}/{FOLD_COUNT} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "models": models,
        "fold_result": {
            "fold_index": int(fold_index),
            "training_group_ids": list(training_ids),
            "validation_group_ids": list(validation_ids),
            "model_diagnostics": diagnostics,
            "oof_evaluation": oof_evaluation,
            "elapsed_sec": elapsed,
        },
    }


def _baseline_fold_worker(fold_index: int) -> dict[str, Any]:
    if _BASELINE_FOLD_STATE is None:
        raise RuntimeError("baseline fold process state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_baseline_fold(int(fold_index), **_BASELINE_FOLD_STATE)


def run_external_city_baseline_freeze(
    *,
    source_cache_root: Path,
    source_baseline_result: Path,
    source_manifest_path: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_spec_path: Path,
    development_audit_path: Path,
    city: str,
    expected_source_cache_sha256: str,
    expected_source_baseline_sha256: str,
    expected_external_cache_sha256: str,
    expected_source_tree_sha256: str | None,
    workers: int,
    fold_workers: int,
    model_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol, _ = _validate_protocol(protocol_spec_path, development_audit_path)
    target_protocol = dict(protocol["target_protocol"])
    if tuple(
        str(value)
        for value in target_protocol.get("same_information_baseline_families", ())
    ) != BENCHMARK_FAMILIES:
        raise ValueError("external same-information baseline family set changed")
    if not bool(
        target_protocol.get(
            "same_information_baselines_frozen_before_evaluation_cache", False
        )
    ):
        raise ValueError("external pre-evaluation baseline-freeze rule is missing")
    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("external baseline source-tree SHA-256 mismatch")

    source_evidence = dict(protocol["source_evidence"])
    if (
        str(source_evidence["counterfactual_cache_sha256"])
        != str(expected_source_cache_sha256)
        or str(source_evidence["source_rule_baseline_sha256"])
        != str(expected_source_baseline_sha256)
        or _sha256(source_manifest_path) != str(source_evidence["manifest_sha256"])
        or _sha256(source_baseline_result) != str(expected_source_baseline_sha256)
    ):
        raise ValueError("external baseline source evidence changed")
    source_cache_sha, source_cache_files = aggregate_cache_sha256(source_cache_root)
    if (
        source_cache_sha != str(expected_source_cache_sha256)
        or source_cache_files
        != 18 * len(SOURCE_SEEDS) * SOURCE_COLLECTION_SHARDS
    ):
        raise ValueError("external baseline source cache identity changed")
    if _sha256(external_manifest_path) != str(
        protocol["counterfactual_cache"]["manifest_sha256"]
    ):
        raise ValueError("external baseline manifest SHA-256 mismatch")
    external_cache_sha, external_cache_files = aggregate_cache_sha256(
        external_cache_root
    )
    expected_external_files = (
        sum(
            len(names)
            for names in protocol["target_protocol"][
                "external_city_scenarios"
            ].values()
        )
        * len(ADAPTATION_SEEDS)
        * EXTERNAL_COLLECTION_SHARDS
    )
    if (
        external_cache_sha != str(expected_external_cache_sha256)
        or external_cache_files != expected_external_files
    ):
        raise ValueError("external baseline adaptation cache identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    city_scenarios = {
        str(name): tuple(str(value) for value in values)
        for name, values in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    if city not in city_scenarios:
        raise ValueError(f"unknown external baseline city: {city}")
    scenarios = city_scenarios[city]
    if set(scenarios) != {
        name for name, group in external_manifest.city_groups.items() if group == city
    }:
        raise ValueError("external baseline manifest city grouping changed")

    source_bank, source_cache_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        source_manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=SOURCE_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    external_bank, external_cache_audit = load_frozen_counterfactual_bank(
        external_cache_root,
        external_manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    selected_group_ids, selection_audit = _select_city_adaptation_groups(
        external_bank, city=city, scenarios=scenarios
    )
    full_city_dataset = _merge_city_datasets(external_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={"target_data_role": "external_baseline_b100_pool"},
    )
    scenario_static_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_static_context = np.mean(
        np.vstack([scenario_static_contexts[scenario] for scenario in scenarios]),
        axis=0,
    )
    records = target_group_records(
        selected_dataset,
        selected_group_ids=selected_group_ids,
        expected_group_count=TARGET_GROUP_BUDGET,
    )
    fold_assignment = assign_scenario_seed_stratified_folds(
        records,
        scenarios=scenarios,
        expected_group_count=TARGET_GROUP_BUDGET,
        fold_count=FOLD_COUNT,
    )
    target_key = f"external_city_{city}"
    fit_bank = dict(source_bank)
    fit_bank[target_key] = selected_dataset
    city_groups = dict(source_manifest.city_groups)
    city_groups[target_key] = city
    baseline = json.loads(source_baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}

    fold_state = {
        "fit_bank": fit_bank,
        "target_key": target_key,
        "city_groups": city_groups,
        "source_rule_costs": source_rule_costs,
        "source_rule_specs": source_rule_specs,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment,
        "selected_dataset": selected_dataset,
        "scenarios": scenarios,
        "source_scenarios": tuple(source_manifest.sumocfgs),
        "source_city_groups": tuple(set(source_manifest.city_groups.values())),
        "external_cities": tuple(city_scenarios),
        "city": city,
        "city_static_context": city_static_context,
    }
    actual_fold_workers = min(max(int(fold_workers), 1), FOLD_COUNT)
    if actual_fold_workers == 1:
        fitted_folds = [
            _fit_baseline_fold(fold_index, **fold_state)
            for fold_index in range(FOLD_COUNT)
        ]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("external baseline fold parallelism requires Linux fork")
        global _BASELINE_FOLD_STATE
        _BASELINE_FOLD_STATE = fold_state
        try:
            with ProcessPoolExecutor(
                max_workers=actual_fold_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {
                    fold_index: pool.submit(_baseline_fold_worker, fold_index)
                    for fold_index in range(FOLD_COUNT)
                }
                fitted_folds = [
                    futures[fold_index].result()
                    for fold_index in range(FOLD_COUNT)
                ]
        finally:
            _BASELINE_FOLD_STATE = None
    model_ensemble = [row["models"] for row in fitted_folds]
    fold_results = [row["fold_result"] for row in fitted_folds]
    gc.collect()

    oof_families = {
        family: {
            "fold_regrets": [
                float(
                    fold["oof_evaluation"]["families"][family][
                        "equal_scenario_mean_normalized_action_regret"
                    ]
                )
                for fold in fold_results
            ]
        }
        for family in BENCHMARK_FAMILIES
    }
    for row in oof_families.values():
        row["mean_regret"] = float(np.mean(row["fold_regrets"]))

    model_payload = {
        "protocol": MODEL_ARTIFACT_PROTOCOL,
        "city": city,
        "target_key": target_key,
        "scenarios": tuple(scenarios),
        "benchmark_families": BENCHMARK_FAMILIES,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment["assignments"],
        "model_ensemble": tuple(model_ensemble),
    }
    _atomic_bytes(
        model_out,
        pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    loaded = pickle.loads(model_out.read_bytes())
    if (
        loaded.get("protocol") != MODEL_ARTIFACT_PROTOCOL
        or loaded.get("city") != city
        or len(loaded.get("model_ensemble", ())) != FOLD_COUNT
        or any(
            set(models.family_models) != set(BENCHMARK_FAMILIES)
            for models in loaded.get("model_ensemble", ())
        )
    ):
        raise ValueError("external baseline model artifact failed round-trip audit")

    return {
        "protocol": "tsc-v42r38-external-city-baseline-b100-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "development_audit_sha256": _sha256(development_audit_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_cache_aggregate_sha256": source_cache_sha,
        "source_cache_file_count": source_cache_files,
        "source_cache_audit": source_cache_audit,
        "source_baseline_sha256": _sha256(source_baseline_result),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "external_cache_aggregate_sha256": external_cache_sha,
        "external_cache_file_count": external_cache_files,
        "external_cache_audit": external_cache_audit,
        "city": city,
        "target_key": target_key,
        "scenarios": list(scenarios),
        "source_scenarios": list(source_manifest.sumocfgs),
        "benchmark_families": list(BENCHMARK_FAMILIES),
        "benchmark_claim_labels": {
            "simulator": "simulator_only",
            "dense": "h2oplus_style_dense_residual_not_exact_h2oplus",
            "dense_advantage": "same_estimand_dense_advantage",
            "causal_rigid_advantage": "pre_v41_rigid_cfcmt",
            "cfcmt_mechanism": "pre_v41_full_cfcmt",
            "causal_target_only": "target_only_b80_per_fold",
        },
        "selected_group_ids": list(selected_group_ids),
        "selection_audit": selection_audit,
        "fold_assignment": fold_assignment,
        "fold_workers": actual_fold_workers,
        "city_static_context": {
            "protocol": "equal-scenario-mean-label-free-static-context-v1",
            "context_names": list(selected_dataset.context_names),
            "scenario_values": {
                scenario: scenario_static_contexts[scenario].tolist()
                for scenario in scenarios
            },
            "city_value": city_static_context.tolist(),
        },
        "fold_results": fold_results,
        "oof_families": oof_families,
        "model_artifact": {
            "protocol": MODEL_ARTIFACT_PROTOCOL,
            "path": str(Path(model_out).resolve()),
            "sha256": _sha256(model_out),
            "size_bytes": Path(model_out).stat().st_size,
            "round_trip_passed": True,
            "ensemble_size": FOLD_COUNT,
        },
        "freeze_gate": {
            "same_information_budget": True,
            "adaptation_only": True,
            "evaluation_seed_absent": True,
            "target_cities_isolated": True,
            "model_artifact_round_trip": True,
            "passed": True,
        },
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-baseline-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-baseline-sha256", required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fold-workers", type=int, default=5)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_external_city_baseline_freeze(
        source_cache_root=args.source_cache_root,
        source_baseline_result=args.source_baseline_result,
        source_manifest_path=args.source_manifest,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_spec_path=args.protocol_spec,
        development_audit_path=args.development_audit,
        city=args.city,
        expected_source_cache_sha256=args.expected_source_cache_sha256,
        expected_source_baseline_sha256=args.expected_source_baseline_sha256,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        workers=args.workers,
        fold_workers=args.fold_workers,
        model_out=args.model_out,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "city": payload["city"],
                "family_count": len(payload["benchmark_families"]),
                "model_sha256": payload["model_artifact"]["sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
