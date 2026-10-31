"""Refit external CFCMT models after deterministic topology-context repair."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CANDIDATE_KEYS,
    COLLECTION_SHARDS,
    CORRECTION_FAMILY,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    FrozenCausalSourceWeightModel,
    SOURCE_WEIGHT_GRID,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    SELECTOR_SPECS,
    select_target_candidate,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    FROZEN_SELECTOR_KEY,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    COMBINED_FAMILIES,
    FOLD_COUNT,
    _fit_one_job,
    assign_leave_one_seed_out_folds,
    select_all_city_adaptation_groups,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_topology_context_repair import REPAIR_PROTOCOL
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
    evaluate_target_action_groups,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid


PROTOCOL = "tsc-v92-topology-repaired-external-refit-diagnostic-v1"
MODEL_PROTOCOL = "cfcmt-topology-repaired-source-weight-model-v1"
_FIT_STATE: dict[str, Any] | None = None


def validate_repaired_cache_contract(
    *,
    label: str,
    repair_audit: Mapping[str, Any],
    manifest_scenarios: Sequence[str],
    seed_count: int,
    collection_shards: int,
    cache_audit: Mapping[str, Any] | None = None,
) -> None:
    """Require the repair, manifest, and loaded cache to describe one bank."""

    expected_scenarios = {str(value) for value in manifest_scenarios}
    audited_scenarios = {
        str(value) for value in repair_audit.get("scenarios", {})
    }
    if repair_audit.get("protocol") != REPAIR_PROTOCOL:
        raise ValueError(f"{label} topology-repair protocol changed")
    if audited_scenarios != expected_scenarios:
        raise ValueError(
            f"{label} repair/manifest scenario mismatch: "
            f"missing={sorted(expected_scenarios - audited_scenarios)}, "
            f"extra={sorted(audited_scenarios - expected_scenarios)}"
        )
    expected_per_scenario = int(seed_count) * int(collection_shards)
    bad_counts = {
        scenario: int(repair_audit["scenarios"][scenario].get("file_count", -1))
        for scenario in sorted(expected_scenarios)
        if int(repair_audit["scenarios"][scenario].get("file_count", -1))
        != expected_per_scenario
    }
    expected_file_count = len(expected_scenarios) * expected_per_scenario
    if bad_counts or int(repair_audit.get("file_count", -1)) != expected_file_count:
        raise ValueError(
            f"{label} topology-repair audit is incomplete: "
            f"expected_file_count={expected_file_count}, "
            f"observed_file_count={repair_audit.get('file_count')}, "
            f"bad_scenario_counts={bad_counts}"
        )
    if cache_audit is None:
        return
    observed = {
        "file_count": int(cache_audit.get("file_count", -1)),
        "used_file_count": int(cache_audit.get("used_file_count", -1)),
        "scenario_count": int(cache_audit.get("scenario_count", -1)),
        "unique_identities": int(cache_audit.get("unique_identities", -1)),
    }
    expected = {
        "file_count": expected_file_count,
        "used_file_count": expected_file_count,
        "scenario_count": len(expected_scenarios),
        "unique_identities": expected_file_count,
    }
    if observed != expected:
        raise ValueError(
            f"{label} loaded-cache contract mismatch: "
            f"expected={expected}, observed={observed}"
        )


def _fit_worker(role: str) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("topology-repaired fit state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_one_job(role, state=_FIT_STATE)


def _evaluate_weight_grid(
    validation,
    *,
    models: OfflineScreeningModels,
    selected_candidate: str,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    source_model = FrozenAnchoredBlendModel(
        anchor_model=models.family_models[ANCHOR_FAMILY],
        correction_model=models.family_models[CORRECTION_FAMILY],
        candidate=selected_candidate,
        anchor_objective_mode=models.objective_modes[ANCHOR_FAMILY],
        correction_objective_mode=models.objective_modes[CORRECTION_FAMILY],
    )
    family_models = {}
    for weight in SOURCE_WEIGHT_GRID:
        key = f"source_weight_{str(weight).replace('.', 'p')}"
        family_models[key] = FrozenCausalSourceWeightModel(
            source_model=source_model,
            target_only_model=models.family_models["causal_target_only"],
            source_objective_mode="control_only",
            target_only_objective_mode=models.objective_modes[
                "causal_target_only"
            ],
            source_weight=weight,
        )
    wrapper = OfflineScreeningModels(
        family_models=family_models,
        prior_spec=models.prior_spec,
        objective_modes={name: "control_only" for name in family_models},
        diagnostics={"protocol": "topology-repaired-source-weight-oof-v1"},
    )
    groups = set(str(value) for value in validation.metadata["action_group_ids"])
    scenario_rows = {}
    for scenario in scenarios:
        selected = {
            group for group in groups if group.startswith(f"{scenario}:")
        }
        subset = _group_subset_v3(
            validation,
            selected_groups=selected,
            metadata_updates={"target_data_role": "source_weight_oof"},
        )
        scenario_rows[scenario] = evaluate_target_action_groups(
            subset,
            models=wrapper,
            excluded_group_ids=(),
        )
    return {
        str(weight): float(
            np.mean(
                [
                    scenario_rows[scenario]["families"][
                        f"source_weight_{str(weight).replace('.', 'p')}"
                    ]["mean_normalized_action_regret"]
                    for scenario in scenarios
                ]
            )
        )
        for weight in SOURCE_WEIGHT_GRID
    }


def run_topology_repaired_refit(
    *,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_baseline_result: Path,
    source_repair_audit: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    external_repair_audit: Path,
    city: str,
    scenarios_by_city: Mapping[str, Sequence[str]],
    workers: int,
    fit_workers: int,
    model_out: Path,
    result_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if model_out.exists() or result_out.exists():
        raise FileExistsError("refusing to overwrite topology-repaired refit")
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    source_repair = json.loads(source_repair_audit.read_text(encoding="utf-8"))
    external_repair = json.loads(external_repair_audit.read_text(encoding="utf-8"))
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
    scenarios = tuple(str(value) for value in scenarios_by_city[city])
    if set(scenarios) != {
        name
        for name, group in external_manifest.city_groups.items()
        if group == city
    }:
        raise ValueError("external city/scenario mapping changed")

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
    selected_group_ids, selection_audit = select_all_city_adaptation_groups(
        external_bank,
        city=city,
        scenarios=scenarios,
    )
    full_city_dataset = _merge_city_datasets(
        external_bank,
        scenarios,
        city=city,
    )
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "topology_repaired_external_adaptation"
        },
    )
    records = target_group_records(
        selected_dataset,
        selected_group_ids=selected_group_ids,
        expected_group_count=len(selected_group_ids),
    )
    fold_assignment = assign_leave_one_seed_out_folds(
        records,
        scenarios=scenarios,
    )
    scenario_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_context = np.mean(
        np.vstack([scenario_contexts[scenario] for scenario in scenarios]),
        axis=0,
    )
    target_key = f"external_city_{city}"
    fit_bank = dict(source_bank)
    fit_bank[target_key] = selected_dataset
    city_groups = dict(source_manifest.city_groups)
    city_groups[target_key] = city
    baseline = json.loads(source_baseline_result.read_text(encoding="utf-8"))
    fit_state = {
        "fit_bank": fit_bank,
        "target_key": target_key,
        "city_groups": city_groups,
        "source_rule_costs": baseline["source_rule_policy_costs"],
        "source_rule_specs": {
            spec.key: spec for spec in generalized_pressure_grid()
        },
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment,
        "selected_dataset": selected_dataset,
        "scenarios": scenarios,
        "source_scenarios": tuple(source_manifest.sumocfgs),
        "source_city_groups": tuple(set(source_manifest.city_groups.values())),
        "external_cities": tuple(scenarios_by_city),
        "city": city,
        "city_static_context": city_context,
        "model_families": COMBINED_FAMILIES,
        "return_fold_models": True,
    }
    roles = tuple(f"fold_{index}" for index in range(FOLD_COUNT)) + (
        "full_refit",
    )
    actual_fit_workers = min(max(int(fit_workers), 1), len(roles))
    if actual_fit_workers == 1:
        fitted = [_fit_one_job(role, state=fit_state) for role in roles]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("topology-repaired refit requires Linux fork")
        global _FIT_STATE
        _FIT_STATE = fit_state
        try:
            with ProcessPoolExecutor(
                max_workers=actual_fit_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {role: pool.submit(_fit_worker, role) for role in roles}
                fitted = [futures[role].result() for role in roles]
        finally:
            _FIT_STATE = None
    fold_results = sorted(
        (row for row in fitted if row["role"].startswith("fold_")),
        key=lambda row: int(row["fold_index"]),
    )
    full_refit = next(row for row in fitted if row["role"] == "full_refit")
    full_models = full_refit.pop("models")

    oof_candidates = {
        candidate: {
            "fold_regrets": [
                float(
                    fold["method_oof_evaluation"]["candidates"][candidate][
                        "equal_scenario_mean_normalized_action_regret"
                    ]
                )
                for fold in fold_results
            ],
            "mean_alpha": float(
                np.mean(
                    [
                        fold["method_oof_evaluation"]["candidates"][candidate][
                            "equal_scenario_mean_alpha"
                        ]
                        for fold in fold_results
                    ]
                )
            ),
        }
        for candidate in CANDIDATE_KEYS
    }
    for row in oof_candidates.values():
        row["mean_regret"] = float(np.mean(row["fold_regrets"]))
    target_selection = select_target_candidate(
        oof_candidates,
        expected_fold_count=FOLD_COUNT,
        **SELECTOR_SPECS[FROZEN_SELECTOR_KEY],
    )
    selected_candidate = str(target_selection["selected_candidate"])
    weight_oof = []
    for fold in fold_results:
        validation_ids = set(str(value) for value in fold["validation_group_ids"])
        validation = _group_subset_v3(
            selected_dataset,
            selected_groups=validation_ids,
            metadata_updates={"target_data_role": "topology_repaired_weight_oof"},
        )
        weight_oof.append(
            _evaluate_weight_grid(
                validation,
                models=fold.pop("models"),
                selected_candidate=selected_candidate,
                scenarios=scenarios,
            )
        )
    weight_summary = {
        str(weight): {
            "fold_regrets": [row[str(weight)] for row in weight_oof],
            "mean_regret": float(
                np.mean([row[str(weight)] for row in weight_oof])
            ),
        }
        for weight in SOURCE_WEIGHT_GRID
    }

    model_payload = {
        "protocol": MODEL_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "selected_candidate": selected_candidate,
        "source_weight_grid": SOURCE_WEIGHT_GRID,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment["assignments"],
        "model": full_models,
        "topology_repair_protocol": REPAIR_PROTOCOL,
    }
    _atomic_bytes(model_out, pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL))
    source_cache_sha, source_file_count = aggregate_cache_sha256(source_cache_root)
    external_cache_sha, external_file_count = aggregate_cache_sha256(external_cache_root)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "post_v91_redevelopment_diagnostic_not_confirmation",
        "city": city,
        "scenarios": list(scenarios),
        "source_cache_sha256": source_cache_sha,
        "source_cache_file_count": source_file_count,
        "external_cache_sha256": external_cache_sha,
        "external_cache_file_count": external_file_count,
        "source_cache_audit": source_cache_audit,
        "external_cache_audit": external_cache_audit,
        "source_repair_audit_sha256": _sha256(source_repair_audit),
        "external_repair_audit_sha256": _sha256(external_repair_audit),
        "source_baseline_sha256": _sha256(source_baseline_result),
        "selected_group_count": len(selected_group_ids),
        "selection_audit": selection_audit,
        "scenario_static_contexts": {
            key: value.tolist() for key, value in scenario_contexts.items()
        },
        "city_static_context": city_context.tolist(),
        "fold_assignment": fold_assignment,
        "selected_candidate": selected_candidate,
        "target_selection": target_selection,
        "oof_candidates": oof_candidates,
        "source_weight_oof": weight_summary,
        "deployment_refit": full_refit,
        "model_artifact": {
            "path": str(model_out.resolve()),
            "sha256": _sha256(model_out),
            "protocol": MODEL_PROTOCOL,
        },
        "fit_workers": actual_fit_workers,
        "elapsed_sec": float(time.monotonic() - started),
    }
    _atomic_json(result_out, payload)
    gc.collect()
    return payload
