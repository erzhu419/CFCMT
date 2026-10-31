"""Select a causal source component from target offline action groups only."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    SOURCE_WEIGHT_GRID,
    evaluate_target_offline_guard_grid,
    evaluate_source_weight_grid_from_offline_groups,
    select_source_city_weight_with_familywise_control,
    select_target_offline_guard_with_familywise_control,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    FOLD_COUNT,
    SELECTION_SEED,
    TARGET_GROUP_BUDGET,
    _atomic_json,
    _merge_city_datasets,
    _select_city_adaptation_groups,
    _sha256,
    assign_scenario_seed_stratified_folds,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    _validate_fit_isolation,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    MODEL_PROTOCOL,
    PROTOCOL as REFIT_PROTOCOL,
    validate_repaired_cache_contract,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    COLLECTION_SHARDS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


PROTOCOL = "tsc-v110-target-offline-source-and-guard-selection-v2"
TARGET_ONLY_FAMILY = "causal_target_only_v2"
FIT_FAMILIES = (*BASE_FAMILIES, TARGET_ONLY_FAMILY)
_FIT_STATES: dict[str, Mapping[str, Any]] | None = None


def source_weight_candidate_key(source_city_group: str, weight: float) -> str:
    token = f"{float(weight):g}".replace(".", "p")
    return f"{str(source_city_group)}__w{token}"


def _fit_and_evaluate_one(
    source_city_group: str,
    fold_index: int,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    assignments = state["fold_assignment"]["assignments"]
    validation_ids = tuple(
        sorted(
            str(group)
            for group, fold in assignments.items()
            if int(fold) == int(fold_index)
        )
    )
    validation_set = set(validation_ids)
    training_ids = tuple(
        group
        for group in state["selected_group_ids"]
        if group not in validation_set
    )
    if not validation_ids or not training_ids:
        raise ValueError("offline source-selection fold is empty")
    models = fit_target_screening_models(
        state["fit_bank"],
        state["target_key"],
        fit_config=MechanismFitConfig(),
        source_rule_costs=state["source_rule_costs"],
        source_rule_specs=state["source_rule_specs"],
        model_families=FIT_FAMILIES,
        target_group_budget=len(training_ids),
        target_adaptation_selection_seed=SELECTION_SEED,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=5,
        scenario_city_groups=state["city_groups"],
        target_adaptation_group_ids=training_ids,
        target_static_context=state["city_static_context"],
        prior_policy_override=state["prior_policy_override"],
    )
    _validate_fit_isolation(
        models.diagnostics,
        target_key=state["target_key"],
        source_scenarios=state["source_scenarios"],
        expected_domains=set(state["source_city_groups"]) | {state["city"]},
        external_cities=state["external_cities"],
        city=state["city"],
        city_static_context=state["city_static_context"],
    )
    validation = _group_subset_v3(
        state["selected_dataset"],
        selected_groups=set(validation_ids),
        metadata_updates={
            "target_data_role": "source_selector_offline_oof_validation",
            "source_city_group": str(source_city_group),
            "cross_fit_fold_index": int(fold_index),
        },
    )
    selected_candidate = str(state["selected_candidate"])
    source_model = FrozenAnchoredBlendModel(
        anchor_model=models.family_models[ANCHOR_FAMILY],
        correction_model=models.family_models[CORRECTION_FAMILY],
        candidate=selected_candidate,
        anchor_objective_mode=models.objective_modes[ANCHOR_FAMILY],
        correction_objective_mode=models.objective_modes[CORRECTION_FAMILY],
    )
    evaluation = evaluate_source_weight_grid_from_offline_groups(
        validation,
        reference_policy=models.prior_spec,
        source_model=source_model,
        target_only_model=models.family_models[TARGET_ONLY_FAMILY],
        source_objective_mode="control_only",
        target_only_objective_mode=models.objective_modes[TARGET_ONLY_FAMILY],
        scenarios=state["scenarios"],
        weights=state["source_weight_grid"],
    )
    guard_evaluation_by_weight = {
        str(float(weight)): evaluate_target_offline_guard_grid(
            validation,
            reference_policy=models.prior_spec,
            source_model=source_model,
            target_only_model=models.family_models[TARGET_ONLY_FAMILY],
            source_objective_mode="control_only",
            target_only_objective_mode=models.objective_modes[
                TARGET_ONLY_FAMILY
            ],
            source_weight=float(weight),
            scenarios=state["scenarios"],
            min_context_trust=float(state["guard_min_context_trust"]),
        )
        for weight in state["source_weight_grid"]
    }
    diagnostics = models.diagnostics
    elapsed = float(time.monotonic() - started)
    print(
        "CFCMT_OFFLINE_SOURCE_OOF_PROGRESS "
        f"city={state['city']} source={source_city_group} "
        f"fold={int(fold_index) + 1}/{FOLD_COUNT} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "source_city_group": str(source_city_group),
        "fold_index": int(fold_index),
        "validation_seeds": sorted(
            {
                int(state["group_seed_by_id"][group])
                for group in validation_ids
            }
        ),
        "training_group_count": len(training_ids),
        "validation_group_count": len(validation_ids),
        "fit_elapsed_sec": elapsed,
        "prior_policy": str(getattr(models.prior_spec, "key", "")),
        "fitted_source_domains": list(diagnostics["source_domains"]),
        "evaluation": evaluation,
        "guard_evaluation_by_weight": guard_evaluation_by_weight,
    }


def _fit_and_evaluate_worker(
    source_city_group: str,
    fold_index: int,
) -> dict[str, Any]:
    if _FIT_STATES is None:
        raise RuntimeError("offline source selection state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_and_evaluate_one(
            source_city_group,
            fold_index,
            state=_FIT_STATES[source_city_group],
        )


def run_target_offline_source_selection(
    *,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_baseline_result: Path,
    source_repair_audit: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    external_repair_audit: Path,
    main_model_path: Path,
    expected_main_model_sha256: str,
    main_result_path: Path,
    expected_main_result_sha256: str,
    city: str,
    scenarios_by_city: Mapping[str, Sequence[str]],
    workers: int,
    fit_workers: int,
    result_out: Path,
    source_weight_grid: Sequence[float] = SOURCE_WEIGHT_GRID,
    minimum_mean_improvement: float = 0.005,
    maximum_fold_regression: float = 0.01,
    source_familywise_alpha: float = 0.025,
    guard_familywise_alpha: float = 0.025,
    guard_minimum_mean_improvement: float = 0.002,
    guard_maximum_fold_regression: float = 0.01,
    guard_min_context_trust: float = 0.1,
) -> dict[str, Any]:
    """Run seed-blocked target-offline source selection without rollouts."""

    started = time.monotonic()
    if (
        not 0.0 < float(source_familywise_alpha) < 1.0
        or not 0.0 < float(guard_familywise_alpha) < 1.0
        or float(source_familywise_alpha) + float(guard_familywise_alpha)
        > 0.05 + 1e-12
        or not 0.0 <= float(guard_min_context_trust) <= 1.0
    ):
        raise ValueError("target-offline family-wise error budget is invalid")
    if result_out.exists():
        raise FileExistsError("refusing to overwrite offline source selection")
    if _sha256(main_model_path) != str(expected_main_model_sha256):
        raise ValueError("main topology-repaired model identity changed")
    if _sha256(main_result_path) != str(expected_main_result_sha256):
        raise ValueError("main topology-repaired result identity changed")
    main_payload = pickle.loads(Path(main_model_path).read_bytes())
    main_result = json.loads(Path(main_result_path).read_text(encoding="utf-8"))
    if (
        main_payload.get("protocol") != MODEL_PROTOCOL
        or main_payload.get("city") != city
        or main_result.get("protocol") != REFIT_PROTOCOL
        or main_result.get("city") != city
        or main_result.get("model_artifact", {}).get("sha256")
        != str(expected_main_model_sha256)
    ):
        raise ValueError("main topology-repaired fit contract changed")

    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    source_repair = json.loads(Path(source_repair_audit).read_text(encoding="utf-8"))
    external_repair = json.loads(
        Path(external_repair_audit).read_text(encoding="utf-8")
    )
    validate_repaired_cache_contract(
        label="source",
        repair_audit=source_repair,
        manifest_scenarios=tuple(source_manifest.sumocfgs),
        seed_count=len(SOURCE_SEEDS),
        collection_shards=COLLECTION_SHARDS,
    )
    validate_repaired_cache_contract(
        label="external",
        repair_audit=external_repair,
        manifest_scenarios=tuple(external_manifest.sumocfgs),
        seed_count=len(ADAPTATION_SEEDS),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
    )
    source_bank, source_cache_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        source_manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    external_bank, external_cache_audit = load_frozen_counterfactual_bank(
        external_cache_root,
        external_manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    validate_repaired_cache_contract(
        label="source",
        repair_audit=source_repair,
        manifest_scenarios=tuple(source_manifest.sumocfgs),
        seed_count=len(SOURCE_SEEDS),
        collection_shards=COLLECTION_SHARDS,
        cache_audit=source_cache_audit,
    )
    validate_repaired_cache_contract(
        label="external",
        repair_audit=external_repair,
        manifest_scenarios=tuple(external_manifest.sumocfgs),
        seed_count=len(ADAPTATION_SEEDS),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        cache_audit=external_cache_audit,
    )
    if (
        main_result.get("source_cache_audit") != source_cache_audit
        or main_result.get("external_cache_audit") != external_cache_audit
    ):
        raise ValueError("offline selector did not receive the main fit cache banks")

    scenarios = tuple(str(value) for value in scenarios_by_city[city])
    if tuple(main_payload.get("scenarios", ())) != scenarios:
        raise ValueError("main model target scenarios changed")
    if set(scenarios) != {
        name
        for name, group in external_manifest.city_groups.items()
        if group == city
    }:
        raise ValueError("external city/scenario mapping changed")
    selected_group_ids, target_group_selection = _select_city_adaptation_groups(
        external_bank,
        city=city,
        scenarios=scenarios,
    )
    if not set(selected_group_ids) <= {
        str(value) for value in main_payload["selected_group_ids"]
    }:
        raise ValueError("B100 selector groups are outside the main adaptation pool")
    full_city_dataset = _merge_city_datasets(external_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "source_selector_balanced_b100_offline_pool"
        },
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
    group_seed_by_id = {
        str(row["group_id"]): int(row["simulator_seed"]) for row in records
    }
    scenario_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_context = np.mean(
        np.vstack([scenario_contexts[scenario] for scenario in scenarios]), axis=0
    )

    baseline = json.loads(Path(source_baseline_result).read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    if set(source_manifest.sumocfgs) - set(source_rule_costs):
        raise ValueError("source rule baseline does not cover the source manifest")
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    source_groups = tuple(sorted(set(source_manifest.city_groups.values())))
    target_key = f"external_city_{city}"
    prior_policy = str(main_payload["model"].prior_spec.key)
    states: dict[str, Mapping[str, Any]] = {}
    source_scenarios_by_group = {}
    for source_group in source_groups:
        group_scenarios = tuple(
            name
            for name, group in source_manifest.city_groups.items()
            if group == source_group
        )
        source_scenarios_by_group[source_group] = group_scenarios
        fit_bank = {name: source_bank[name] for name in group_scenarios}
        fit_bank[target_key] = selected_dataset
        city_groups = {
            name: source_manifest.city_groups[name] for name in group_scenarios
        }
        city_groups[target_key] = city
        states[source_group] = {
            "fit_bank": fit_bank,
            "target_key": target_key,
            "city_groups": city_groups,
            "source_rule_costs": source_rule_costs,
            "source_rule_specs": source_rule_specs,
            "selected_group_ids": selected_group_ids,
            "fold_assignment": fold_assignment,
            "selected_dataset": selected_dataset,
            "scenarios": scenarios,
            "source_scenarios": group_scenarios,
            "source_city_groups": (source_group,),
            "external_cities": tuple(scenarios_by_city),
            "city": city,
            "city_static_context": city_context,
            "model_families": FIT_FAMILIES,
            "prior_policy_override": prior_policy,
            "return_fold_models": True,
            "selected_candidate": str(main_payload["selected_candidate"]),
            "source_weight_grid": tuple(float(value) for value in source_weight_grid),
            "guard_min_context_trust": float(guard_min_context_trust),
            "group_seed_by_id": group_seed_by_id,
        }

    jobs = tuple(
        (source_group, fold_index)
        for source_group in source_groups
        for fold_index in range(FOLD_COUNT)
    )
    actual_fit_workers = min(max(int(fit_workers), 1), len(jobs))
    if actual_fit_workers == 1:
        fitted = [
            _fit_and_evaluate_one(source_group, fold_index, state=states[source_group])
            for source_group, fold_index in jobs
        ]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("offline source selection parallelism requires Linux fork")
        global _FIT_STATES
        _FIT_STATES = states
        try:
            with ProcessPoolExecutor(
                max_workers=actual_fit_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {
                    job: pool.submit(_fit_and_evaluate_worker, *job) for job in jobs
                }
                fitted = [futures[job].result() for job in jobs]
        finally:
            _FIT_STATES = None

    baseline_by_fold: dict[int, list[float]] = {
        fold: [] for fold in range(FOLD_COUNT)
    }
    matrix_rows = []
    for row in fitted:
        fold = int(row["fold_index"])
        source_group = str(row["source_city_group"])
        grid = row["evaluation"]["grid"]
        baseline_by_fold[fold].append(
            float(grid["0.0"]["equal_scenario_mean_normalized_action_regret"])
        )
        for weight in source_weight_grid:
            weight = float(weight)
            if weight == 0.0:
                continue
            metric = grid[str(weight)][
                "equal_scenario_mean_normalized_action_regret"
            ]
            matrix_rows.append(
                {
                    "fold_index": fold,
                    "candidate_key": source_weight_candidate_key(
                        source_group, weight
                    ),
                    "source_city_group": source_group,
                    "source_weight": weight,
                    "equal_scenario_mean_normalized_action_regret": float(metric),
                }
            )
    target_only_max_spread = {}
    for fold, values in baseline_by_fold.items():
        spread = float(max(values) - min(values))
        target_only_max_spread[str(fold)] = spread
        if spread > 1e-12:
            raise ValueError(
                "target-only OOF regret changed with the fitted source-city component"
            )
        matrix_rows.append(
            {
                "fold_index": fold,
                "candidate_key": "target_only",
                "source_city_group": None,
                "source_weight": 0.0,
                "equal_scenario_mean_normalized_action_regret": float(values[0]),
            }
        )
    matrix_rows.sort(key=lambda row: (row["fold_index"], row["candidate_key"]))
    selection = select_source_city_weight_with_familywise_control(
        matrix_rows,
        minimum_mean_improvement=float(minimum_mean_improvement),
        maximum_fold_regression=float(maximum_fold_regression),
        familywise_alpha=float(source_familywise_alpha),
    )
    selected_source_group = selection["selected_source_city_group"]
    selected_source_weight = float(selection["selected_source_weight"])
    guard_rows = []
    target_only_guard_by_fold: dict[int, list[Mapping[str, Any]]] = {
        fold: [] for fold in range(FOLD_COUNT)
    }
    for row in fitted:
        fold = int(row["fold_index"])
        source_group = str(row["source_city_group"])
        by_weight = row["guard_evaluation_by_weight"]
        target_only_guard_by_fold[fold].append(by_weight[str(0.0)]["grid"])
        if source_group != selected_source_group:
            continue
        selected_grid = by_weight[str(selected_source_weight)]["grid"]
        guard_rows.extend(
            {"fold_index": fold, **dict(profile)}
            for profile in selected_grid.values()
        )
    if selected_source_group is None:
        for fold, values in target_only_guard_by_fold.items():
            if not values or any(value != values[0] for value in values[1:]):
                raise ValueError(
                    "target-only guard evaluation changed with source component"
                )
            guard_rows.extend(
                {"fold_index": fold, **dict(profile)}
                for profile in values[0].values()
            )
    guard_selection = select_target_offline_guard_with_familywise_control(
        guard_rows,
        familywise_alpha=float(guard_familywise_alpha),
        minimum_mean_improvement=float(guard_minimum_mean_improvement),
        maximum_fold_regression=float(guard_maximum_fold_regression),
    )
    compact_fold_results = [
        {
            key: value
            for key, value in row.items()
            if key != "guard_evaluation_by_weight"
        }
        for row in fitted
    ]
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "scientific_status": "target-offline-source-and-guard-selection-development",
        "city": city,
        "scenarios": list(scenarios),
        "source_city_groups": list(source_groups),
        "source_scenarios_by_group": {
            key: list(value) for key, value in source_scenarios_by_group.items()
        },
        "target_data_contract": {
            "role": "target_offline_adaptation_not_zero_shot",
            "selected_group_count": len(selected_group_ids),
            "target_group_budget": TARGET_GROUP_BUDGET,
            "selection_protocol": target_group_selection["protocol"],
            "all_selected_groups_from_main_pool": True,
            "fold_protocol": fold_assignment["protocol"],
            "adaptation_seeds": list(ADAPTATION_SEEDS),
            "target_closed_loop_rollouts_used_for_selection": False,
            "familywise_alpha_budget": {
                "source_identity_and_weight": float(source_familywise_alpha),
                "deployment_guard": float(guard_familywise_alpha),
                "total": float(source_familywise_alpha)
                + float(guard_familywise_alpha),
            },
        },
        "fixed_parent_contract": {
            "main_model_sha256": str(expected_main_model_sha256),
            "main_result_sha256": str(expected_main_result_sha256),
            "selected_candidate": str(main_payload["selected_candidate"]),
            "prior_policy": prior_policy,
        },
        "source_cache_audit": source_cache_audit,
        "external_cache_audit": external_cache_audit,
        "fold_assignment": fold_assignment,
        "fit_families": list(FIT_FAMILIES),
        "fit_workers": actual_fit_workers,
        "source_weight_grid": [float(value) for value in source_weight_grid],
        "target_only_invariance_audit": {
            "passed": True,
            "maximum_regret_spread_by_fold": target_only_max_spread,
        },
        "fold_results": compact_fold_results,
        "selection_matrix": matrix_rows,
        "selection": selection,
        "guard_selection_matrix": guard_rows,
        "guard_selection": guard_selection,
        "elapsed_sec": float(time.monotonic() - started),
    }
    _atomic_json(result_out, payload)
    gc.collect()
    return payload
