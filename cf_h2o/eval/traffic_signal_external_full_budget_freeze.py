"""Freeze full-budget external-city models with seed-blocked OOF and refit.

The v42 B100 protocol used cross-validation estimators as deployment models.
This stage keeps model selection and deployment estimation separate: complete
adaptation seeds are the OOF folds, while the serialized deployment model is
refit once on every valid adaptation action group.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    BASE_FAMILIES,
    CANDIDATE_KEYS,
    COLLECTION_SHARDS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    SELECTOR_SPECS,
    select_target_candidate,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
    _equal_scenario_family_evaluation,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    FROZEN_SELECTOR_KEY,
    SELECTION_SEED,
    _atomic_bytes,
    _atomic_json,
    _equal_scenario_fold_evaluation,
    _merge_city_datasets,
    _sha256,
    _validate_protocol as _validate_parent_protocol,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_seed_v3,
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "tsc-v43r39-external-full-budget-seed-blocked-refit-confirmation-v1"
METHOD_FREEZE_PROTOCOL = "tsc-v43r39-external-city-full-oof-refit-freeze-v1"
BASELINE_FREEZE_PROTOCOL = (
    "tsc-v43r39-external-city-full-baseline-oof-refit-freeze-v1"
)
METHOD_MODEL_PROTOCOL = "cfcmt-external-city-full-refit-model-v2"
BASELINE_MODEL_PROTOCOL = "cfcmt-external-city-full-refit-baselines-v2"
V9_PROTOCOL = (
    "tsc-v54r50-external-v9-full-budget-seed-blocked-refit-confirmation-v1"
)
V9_PARENT_PROTOCOL = "tsc-v53r49-external-v9-network-repair-confirmation-v1"
V9_METHOD_FREEZE_PROTOCOL = (
    "tsc-v54r50-external-v9-city-full-oof-refit-freeze-v1"
)
V9_BASELINE_FREEZE_PROTOCOL = (
    "tsc-v54r50-external-v9-city-full-baseline-oof-refit-freeze-v1"
)
V9_METHOD_MODEL_PROTOCOL = "cfcmt-external-v9-city-full-refit-model-v1"
V9_BASELINE_MODEL_PROTOCOL = "cfcmt-external-v9-city-full-refit-baselines-v1"
FOLD_PROTOCOL = "leave-one-complete-adaptation-seed-out-v1"
REFIT_PROTOCOL = "all-valid-adaptation-groups-single-refit-v1"
EVALUATION_SEED = 8171
V9_EVALUATION_SEED = 9277
DIAGNOSTIC_ONLY_SEED = 7079
FOLD_COUNT = len(ADAPTATION_SEEDS)
COMBINED_FAMILIES = tuple(dict.fromkeys((*BASE_FAMILIES, *BENCHMARK_FAMILIES)))
_FIT_STATE: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _artifact_protocols(protocol: str) -> dict[str, str]:
    if protocol == PROTOCOL:
        return {
            "method_freeze": METHOD_FREEZE_PROTOCOL,
            "baseline_freeze": BASELINE_FREEZE_PROTOCOL,
            "method_model": METHOD_MODEL_PROTOCOL,
            "baseline_model": BASELINE_MODEL_PROTOCOL,
        }
    if protocol == V9_PROTOCOL:
        return {
            "method_freeze": V9_METHOD_FREEZE_PROTOCOL,
            "baseline_freeze": V9_BASELINE_FREEZE_PROTOCOL,
            "method_model": V9_METHOD_MODEL_PROTOCOL,
            "baseline_model": V9_BASELINE_MODEL_PROTOCOL,
        }
    raise ValueError(f"unsupported external full-budget protocol: {protocol}")


def _validate_v9_parent_protocol(
    parent_path: Path,
    development_audit_path: Path,
    *,
    failed_v52_audit_path: Path | None,
    network_admission_audit_path: Path | None,
    counterfactual_cache_audit_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = _read_json(parent_path)
    if parent.get("protocol") != V9_PARENT_PROTOCOL:
        raise ValueError("external v9 repair protocol changed")
    base_spec = dict(parent.get("base_protocol", {}))
    base_path = _resolve_project_path(str(base_spec.get("path", "")))
    if _sha256(base_path) != str(base_spec.get("sha256", "")):
        raise ValueError("external v9 base protocol SHA-256 mismatch")
    base, development_audit = _validate_parent_protocol(
        base_path, development_audit_path
    )
    for section in ("source_evidence", "target_protocol", "offline_confirmation_gate"):
        if dict(parent.get(section, {})) != dict(base.get(section, {})):
            raise ValueError(f"external v9 inherited section changed: {section}")

    conversion = dict(parent.get("conversion", {}))
    benchmark_manifest = _resolve_project_path(
        str(conversion.get("benchmark_manifest", ""))
    )
    if (
        conversion.get("protocol")
        != "cfcmt-cityflow-full-external-sumo-conversion-v9"
        or _sha256(benchmark_manifest)
        != str(conversion.get("benchmark_manifest_sha256", ""))
        or conversion.get("virtual_lane_link_protocol")
        != "cityflow-virtual-cartesian-lane-link-monotone-rank-v1"
        or int(conversion.get("post_reduction_crossing_road_link_count", -1)) != 0
        or int(conversion.get("source_lane_coverage_failures", -1)) != 0
        or int(conversion.get("target_lane_coverage_failures", -1)) != 0
    ):
        raise ValueError("external v9 conversion contract changed")

    method_freeze = dict(parent.get("method_freeze", {}))
    candidate_grid = _resolve_project_path(
        str(method_freeze.get("candidate_grid", ""))
    )
    if (
        _sha256(candidate_grid)
        != str(method_freeze.get("candidate_grid_sha256", ""))
        or not bool(
            method_freeze.get(
                "no_v9_efficacy_metric_may_change_the_algorithm_or_candidate_grid",
                False,
            )
        )
    ):
        raise ValueError("external v9 method-freeze contract changed")

    required_artifacts = (
        (
            failed_v52_audit_path,
            dict(
                parent.get("post_efficacy_integrity_amendment", {}).get(
                    "failed_v52_audit", {}
                )
            ),
            "reject_v52_and_preserve_new_validation_and_prospective_seeds",
        ),
        (
            network_admission_audit_path,
            dict(parent.get("network_admission", {})),
            "authorize_v9_counterfactual_cache_recollection",
        ),
        (
            counterfactual_cache_audit_path,
            {
                "sha256": parent.get("counterfactual_cache", {}).get(
                    "audit_sha256"
                ),
                "required_status": parent.get("counterfactual_cache", {}).get(
                    "required_audit_status"
                ),
                "required_decision": parent.get("counterfactual_cache", {}).get(
                    "required_audit_decision"
                ),
            },
            "authorize_v9_external_model_refit",
        ),
    )
    for path, expected, decision in required_artifacts:
        if path is None or not Path(path).is_file():
            raise ValueError("external v9 repair evidence path is required")
        payload = _read_json(Path(path))
        expected_sha = str(expected.get("sha256", expected.get("audit_sha256", "")))
        if (
            _sha256(Path(path)) != expected_sha
            or payload.get("status") != expected.get("required_status", "PASS")
            or payload.get("decision")
            != expected.get("required_decision", decision)
            or payload.get("decision") != decision
        ):
            raise ValueError(f"external v9 repair evidence changed: {path}")
    cache = dict(parent.get("counterfactual_cache", {}))
    execution = dict(cache.get("adaptation_cache_execution", {}))
    cache_audit = _read_json(Path(counterfactual_cache_audit_path))
    if (
        cache.get("adaptation_cache_status") != "PASS"
        or int(execution.get("cache_file_count", -1)) != 128
        or execution.get("cache_aggregate_sha256")
        != cache_audit.get("cache_sha256")
        or int(cache_audit.get("cache_file_count", -1)) != 128
        or not bool(cache_audit.get("integrity_gate", {}).get("passed", False))
    ):
        raise ValueError("external v9 counterfactual cache contract changed")
    return parent, development_audit


def _validate_full_protocol(
    protocol_spec_path: Path,
    development_audit_path: Path,
    diagnostic_failure_result_path: Path | None = None,
    failed_v52_audit_path: Path | None = None,
    network_admission_audit_path: Path | None = None,
    counterfactual_cache_audit_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _read_json(protocol_spec_path)
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name not in {PROTOCOL, V9_PROTOCOL}:
        raise ValueError("external full-budget protocol changed")

    parent_spec = dict(protocol.get("parent_protocol", {}))
    parent_path = _resolve_project_path(str(parent_spec.get("path", "")))
    if _sha256(parent_path) != str(parent_spec.get("sha256", "")):
        raise ValueError("external full-budget parent protocol SHA-256 mismatch")
    if protocol_name == PROTOCOL:
        parent, development_audit = _validate_parent_protocol(
            parent_path, development_audit_path
        )
    else:
        parent, development_audit = _validate_v9_parent_protocol(
            parent_path,
            development_audit_path,
            failed_v52_audit_path=failed_v52_audit_path,
            network_admission_audit_path=network_admission_audit_path,
            counterfactual_cache_audit_path=counterfactual_cache_audit_path,
        )

    diagnostic = dict(protocol.get("v42_diagnostic_failure", {}))
    diagnostic_path = (
        Path(diagnostic_failure_result_path)
        if diagnostic_failure_result_path is not None
        else _resolve_project_path(str(diagnostic.get("result", "")))
    )
    diagnostic_result = _read_json(diagnostic_path)
    if (
        _sha256(diagnostic_path) != str(diagnostic.get("sha256", ""))
        or int(diagnostic.get("seed", -1)) != DIAGNOSTIC_ONLY_SEED
        or not bool(
            diagnostic.get(
                "permanently_excluded_from_v43_selection_and_confirmation", False
            )
        )
        or diagnostic_result.get("decision")
        != "retain_offline_failure_and_prohibit_closed_loop_claim"
    ):
        raise ValueError("v42 diagnostic-failure boundary changed")

    correction = dict(protocol.get("methodological_correction", {}))
    required_corrections = (
        "uses_all_available_adaptation_groups",
        "removes_arbitrary_b100_cap",
        "oof_models_must_not_be_used_as_deployment_models",
        "method_and_baselines_share_exact_group_ids_and_refit_model_call",
        "seed_7079_metrics_must_not_change_any_v43_choice",
    )
    if not all(bool(correction.get(key, False)) for key in required_corrections):
        raise ValueError("external full-budget correction contract changed")
    if protocol_name == V9_PROTOCOL and (
        not bool(correction.get("v9_network_repair_must_not_change_model_family_or_selector"))
        or not bool(correction.get("v9_efficacy_unobserved_at_model_freeze"))
    ):
        raise ValueError("external v9 full-budget correction contract changed")

    target = dict(protocol.get("target_protocol", {}))
    parent_target = dict(parent["target_protocol"])
    exact_values = {
        "city_is_the_transfer_and_reporting_unit": True,
        "equal_scenario_weight_within_city": True,
        "target_group_budget_per_city": "all_available",
        "target_group_selection": "all_valid_groups_no_subsampling",
        "cross_fit_folds": FOLD_COUNT,
        "cross_fit_strategy": "leave_one_adaptation_seed_out",
        "deployment_refit_uses_all_adaptation_groups": True,
        "deployment_model_ensemble_size": 1,
        "selection_seed": SELECTION_SEED,
        "same_information_baselines_frozen_before_evaluation_cache": True,
        "evaluation_cache_must_be_collected_after_joint_freeze": True,
    }
    for key, expected in exact_values.items():
        if target.get(key) != expected:
            raise ValueError(f"external full-budget target protocol changed: {key}")
    if target.get("external_city_scenarios") != parent_target.get(
        "external_city_scenarios"
    ):
        raise ValueError("external full-budget city scenarios changed")
    if tuple(int(value) for value in target.get("adaptation_seeds", ())) != tuple(
        ADAPTATION_SEEDS
    ):
        raise ValueError("external full-budget adaptation seeds changed")
    if tuple(int(value) for value in target.get("diagnostic_only_seeds", ())) != (
        DIAGNOSTIC_ONLY_SEED,
    ):
        raise ValueError("external full-budget diagnostic-only seed changed")
    expected_evaluation_seed = (
        EVALUATION_SEED if protocol_name == PROTOCOL else V9_EVALUATION_SEED
    )
    if tuple(int(value) for value in target.get("evaluation_seeds", ())) != (
        expected_evaluation_seed,
    ):
        raise ValueError("external full-budget evaluation seed changed")
    if set(target["adaptation_seeds"]) & set(
        (*target["diagnostic_only_seeds"], *target["evaluation_seeds"])
    ):
        raise ValueError("external full-budget seed roles overlap")
    if tuple(str(value) for value in target.get("base_families", ())) != BASE_FAMILIES:
        raise ValueError("external full-budget base families changed")
    if tuple(
        str(value) for value in target.get("same_information_baseline_families", ())
    ) != BENCHMARK_FAMILIES:
        raise ValueError("external full-budget baseline families changed")
    if int(target.get("counterfactual_collection_shards_per_scenario_seed", -1)) != (
        EXTERNAL_COLLECTION_SHARDS
    ):
        raise ValueError("external full-budget collection shard count changed")
    if dict(protocol.get("offline_confirmation_gate", {})) != dict(
        parent.get("offline_confirmation_gate", {})
    ):
        raise ValueError("external full-budget confirmation gate changed")
    return protocol, parent, development_audit


def select_all_city_adaptation_groups(
    bank: Mapping[str, Any],
    *,
    city: str,
    scenarios: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    selected: list[str] = []
    scenario_rows: dict[str, Any] = {}
    seed_totals = {seed: 0 for seed in ADAPTATION_SEEDS}
    for scenario in scenarios:
        groups = sorted(
            set(
                str(value)
                for value in bank[str(scenario)].metadata.get("action_group_ids", ())
            )
        )
        observed_seeds = {_group_seed_v3(group) for group in groups}
        if observed_seeds != set(ADAPTATION_SEEDS):
            raise ValueError(
                f"full-budget adaptation seed coverage changed for {scenario}"
            )
        counts = {
            seed: sum(_group_seed_v3(group) == seed for group in groups)
            for seed in ADAPTATION_SEEDS
        }
        if any(count <= 0 for count in counts.values()):
            raise ValueError(f"empty adaptation seed stratum for {scenario}")
        selected.extend(groups)
        for seed, count in counts.items():
            seed_totals[seed] += count
        scenario_rows[str(scenario)] = {
            "available_group_count": len(groups),
            "selected_group_count": len(groups),
            "selected_group_count_by_seed": {
                str(seed): counts[seed] for seed in ADAPTATION_SEEDS
            },
            "selected_group_ids": groups,
        }
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("full-budget city groups are empty or duplicated")
    return tuple(sorted(selected)), {
        "protocol": "all-valid-city-adaptation-groups-no-subsampling-v1",
        "city": city,
        "selected_group_count": len(selected),
        "selected_group_count_by_seed": {
            str(seed): seed_totals[seed] for seed in ADAPTATION_SEEDS
        },
        "scenarios": scenario_rows,
    }


def assign_leave_one_seed_out_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    scenario_names = tuple(str(value) for value in scenarios)
    assignments: dict[str, int] = {}
    scenario_seed_counts = {
        scenario: {seed: 0 for seed in ADAPTATION_SEEDS}
        for scenario in scenario_names
    }
    for row in records:
        group = str(row["group_id"])
        seed = int(row["simulator_seed"])
        if seed not in ADAPTATION_SEEDS:
            raise ValueError(f"unexpected adaptation seed in full-budget fold: {seed}")
        matches = [
            scenario
            for scenario in scenario_names
            if group.startswith(f"{scenario}:")
        ]
        if len(matches) != 1 or group in assignments:
            raise ValueError(f"invalid full-budget action group identity: {group}")
        fold = ADAPTATION_SEEDS.index(seed)
        assignments[group] = fold
        scenario_seed_counts[matches[0]][seed] += 1
    if not assignments or any(
        scenario_seed_counts[scenario][seed] <= 0
        for scenario in scenario_names
        for seed in ADAPTATION_SEEDS
    ):
        raise ValueError("full-budget seed-blocked folds lack a scenario/seed stratum")
    fold_group_counts = [
        sum(fold == index for fold in assignments.values())
        for index in range(FOLD_COUNT)
    ]
    return {
        "protocol": FOLD_PROTOCOL,
        "fold_count": FOLD_COUNT,
        "group_count": len(assignments),
        "assignments": assignments,
        "validation_seed_by_fold": {
            str(index): seed for index, seed in enumerate(ADAPTATION_SEEDS)
        },
        "fold_group_counts": fold_group_counts,
        "scenario_seed_group_counts": {
            scenario: {
                str(seed): scenario_seed_counts[scenario][seed]
                for seed in ADAPTATION_SEEDS
            }
            for scenario in scenario_names
        },
    }


def _project_models(
    models: OfflineScreeningModels,
    families: Sequence[str],
) -> OfflineScreeningModels:
    names = tuple(str(value) for value in families)
    if not set(names) <= set(models.family_models):
        raise ValueError("cannot project absent external screening families")
    return OfflineScreeningModels(
        family_models={name: models.family_models[name] for name in names},
        prior_spec=models.prior_spec,
        objective_modes={name: models.objective_modes[name] for name in names},
        diagnostics=dict(models.diagnostics),
    )


def _validate_fit_isolation(
    diagnostics: Mapping[str, Any],
    *,
    target_key: str,
    source_scenarios: Sequence[str],
    expected_domains: set[str],
    external_cities: Sequence[str],
    city: str,
    city_static_context: Sequence[float],
) -> None:
    observed_domains = set(diagnostics.get("source_domains", ()))
    if (
        diagnostics.get("heldout_city_scenarios") != [target_key]
        or set(diagnostics.get("source_scenarios", ())) != set(source_scenarios)
        or observed_domains != expected_domains
        or any(
            external_city in observed_domains
            for external_city in set(external_cities) - {city}
        )
        or diagnostics.get("target_static_context_protocol")
        != "explicit-label-free-city-static-context-v1"
        or not np.allclose(
            np.asarray(diagnostics.get("target_static_context", ()), dtype=float),
            np.asarray(city_static_context, dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("full-budget external target/source isolation failed")


def _fit_one_job(role: str, *, state: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    selected_group_ids = tuple(state["selected_group_ids"])
    if role == "full_refit":
        training_ids = selected_group_ids
        validation_ids: tuple[str, ...] = ()
        fold_index = None
    else:
        fold_index = int(role.removeprefix("fold_"))
        validation_ids = tuple(
            sorted(
                group
                for group, fold in state["fold_assignment"]["assignments"].items()
                if int(fold) == fold_index
            )
        )
        validation_set = set(validation_ids)
        training_ids = tuple(
            group for group in selected_group_ids if group not in validation_set
        )
        if not validation_ids or not training_ids:
            raise ValueError("full-budget OOF fold is empty")
        validation_seed = int(
            state["fold_assignment"]["validation_seed_by_fold"][str(fold_index)]
        )
        if (
            {_group_seed_v3(group) for group in validation_ids}
            != {validation_seed}
            or {_group_seed_v3(group) for group in training_ids}
            != set(ADAPTATION_SEEDS) - {validation_seed}
        ):
            raise ValueError("full-budget OOF seed boundary failed")

    model_families = tuple(state.get("model_families", COMBINED_FAMILIES))
    models = fit_target_screening_models(
        state["fit_bank"],
        state["target_key"],
        fit_config=MechanismFitConfig(),
        source_rule_costs=state["source_rule_costs"],
        source_rule_specs=state["source_rule_specs"],
        model_families=model_families,
        target_group_budget=len(training_ids),
        target_adaptation_selection_seed=SELECTION_SEED,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=5,
        scenario_city_groups=state["city_groups"],
        target_adaptation_group_ids=training_ids,
        target_static_context=state["city_static_context"],
        prior_policy_override=state.get("prior_policy_override"),
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

    elapsed = float(time.monotonic() - started)
    if role == "full_refit":
        print(
            "CFCMT_EXTERNAL_FULL_REFIT_PROGRESS "
            f"city={state['city']} groups={len(training_ids)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return {
            "role": role,
            "models": models,
            "training_group_ids": list(training_ids),
            "model_diagnostics": models.diagnostics,
            "elapsed_sec": elapsed,
        }

    validation = _group_subset_v3(
        state["selected_dataset"],
        selected_groups=set(validation_ids),
        metadata_updates={
            "target_data_role": "external_full_seed_blocked_oof_validation",
            "cross_fit_fold_index": fold_index,
        },
    )
    method_evaluation = (
        _equal_scenario_fold_evaluation(
            validation,
            models=_project_models(models, BASE_FAMILIES),
            scenarios=state["scenarios"],
        )
        if set(BASE_FAMILIES) <= set(model_families)
        else None
    )
    baseline_evaluation = (
        _equal_scenario_family_evaluation(
            validation,
            models=_project_models(models, BENCHMARK_FAMILIES),
            scenarios=state["scenarios"],
        )
        if set(BENCHMARK_FAMILIES) <= set(model_families)
        else None
    )
    print(
        "CFCMT_EXTERNAL_FULL_OOF_PROGRESS "
        f"city={state['city']} fold={fold_index + 1}/{FOLD_COUNT} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    result = {
        "role": role,
        "fold_index": fold_index,
        "validation_seed": int(
            state["fold_assignment"]["validation_seed_by_fold"][str(fold_index)]
        ),
        "training_group_ids": list(training_ids),
        "validation_group_ids": list(validation_ids),
        "model_diagnostics": models.diagnostics,
        "method_oof_evaluation": method_evaluation,
        "baseline_oof_evaluation": baseline_evaluation,
        "elapsed_sec": elapsed,
    }
    if bool(state.get("return_fold_models", False)):
        result["models"] = models
    return result


def _fit_worker(role: str) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("full-budget fit process state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_one_job(role, state=_FIT_STATE)


def _write_model_artifact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    protocol: str,
    city: str,
    families: Sequence[str],
) -> dict[str, Any]:
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite frozen model: {path}")
    _atomic_bytes(path, pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL))
    loaded = pickle.loads(Path(path).read_bytes())
    models = tuple(loaded.get("model_ensemble", ()))
    if (
        loaded.get("protocol") != protocol
        or loaded.get("city") != city
        or len(models) != 1
        or any(set(model.family_models) != set(families) for model in models)
    ):
        raise ValueError(f"full-budget model artifact round-trip failed: {path}")
    return {
        "protocol": protocol,
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": Path(path).stat().st_size,
        "round_trip_passed": True,
        "ensemble_size": 1,
        "deployment_refit_protocol": REFIT_PROTOCOL,
    }


def run_external_full_budget_freeze(
    *,
    source_cache_root: Path,
    source_baseline_result: Path,
    source_manifest_path: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_spec_path: Path,
    development_audit_path: Path,
    diagnostic_failure_result_path: Path,
    failed_v52_audit_path: Path | None,
    network_admission_audit_path: Path | None,
    counterfactual_cache_audit_path: Path | None,
    city: str,
    expected_source_cache_sha256: str,
    expected_source_baseline_sha256: str,
    expected_external_cache_sha256: str,
    expected_source_tree_sha256: str | None,
    workers: int,
    fit_workers: int,
    method_model_out: Path,
    baseline_model_out: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    protocol, parent, development_audit = _validate_full_protocol(
        protocol_spec_path,
        development_audit_path,
        diagnostic_failure_result_path,
        failed_v52_audit_path,
        network_admission_audit_path,
        counterfactual_cache_audit_path,
    )
    artifact_protocols = _artifact_protocols(str(protocol["protocol"]))
    evaluation_seed = int(protocol["target_protocol"]["evaluation_seeds"][0])
    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("external full-budget source-tree SHA-256 mismatch")

    source_evidence = dict(parent["source_evidence"])
    if (
        str(source_evidence["counterfactual_cache_sha256"])
        != str(expected_source_cache_sha256)
        or str(source_evidence["source_rule_baseline_sha256"])
        != str(expected_source_baseline_sha256)
        or _sha256(source_manifest_path) != str(source_evidence["manifest_sha256"])
        or _sha256(source_baseline_result) != str(expected_source_baseline_sha256)
    ):
        raise ValueError("external full-budget source evidence changed")
    source_cache_sha, source_cache_files = aggregate_cache_sha256(source_cache_root)
    if (
        source_cache_sha != str(expected_source_cache_sha256)
        or source_cache_files != 18 * len(SOURCE_SEEDS) * COLLECTION_SHARDS
    ):
        raise ValueError("external full-budget source cache identity changed")

    parent_cache = dict(parent["counterfactual_cache"])
    if _sha256(external_manifest_path) != str(parent_cache["manifest_sha256"]):
        raise ValueError("external full-budget manifest SHA-256 mismatch")
    external_cache_sha, external_cache_files = aggregate_cache_sha256(
        external_cache_root
    )
    city_scenarios = {
        str(name): tuple(str(value) for value in values)
        for name, values in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    expected_external_files = (
        sum(len(values) for values in city_scenarios.values())
        * len(ADAPTATION_SEEDS)
        * EXTERNAL_COLLECTION_SHARDS
    )
    if (
        external_cache_sha != str(expected_external_cache_sha256)
        or external_cache_files != expected_external_files
    ):
        raise ValueError("external full-budget adaptation cache identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(
        Path(conversion_root).resolve()
    )
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    if city not in city_scenarios:
        raise ValueError(f"unknown full-budget external city: {city}")
    scenarios = city_scenarios[city]
    if set(scenarios) != {
        name for name, group in external_manifest.city_groups.items() if group == city
    }:
        raise ValueError("external full-budget manifest city grouping changed")

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
    selected_group_ids, selection_audit = select_all_city_adaptation_groups(
        external_bank, city=city, scenarios=scenarios
    )
    full_city_dataset = _merge_city_datasets(external_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "external_city_full_adaptation_pool",
            "target_group_selection": "all_valid_groups_no_subsampling",
        },
    )
    records = target_group_records(
        selected_dataset,
        selected_group_ids=selected_group_ids,
        expected_group_count=len(selected_group_ids),
    )
    fold_assignment = assign_leave_one_seed_out_folds(
        records, scenarios=scenarios
    )

    scenario_static_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_static_context = np.mean(
        np.vstack([scenario_static_contexts[scenario] for scenario in scenarios]),
        axis=0,
    )
    target_key = f"external_city_{city}"
    fit_bank = dict(source_bank)
    fit_bank[target_key] = selected_dataset
    city_groups = dict(source_manifest.city_groups)
    city_groups[target_key] = city
    if any(
        group in set(city_scenarios)
        for group in source_manifest.city_groups.values()
    ):
        raise ValueError("an external confirmation city entered the source bank")
    source_baseline = _read_json(source_baseline_result)
    source_rule_costs = source_baseline["source_rule_policy_costs"]
    if not set(source_manifest.sumocfgs) <= set(source_rule_costs):
        raise ValueError("source baseline does not cover all source scenarios")
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}

    fit_state = {
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
    roles = tuple(f"fold_{index}" for index in range(FOLD_COUNT)) + (
        "full_refit",
    )
    actual_fit_workers = min(max(int(fit_workers), 1), len(roles))
    if actual_fit_workers == 1:
        fitted = [_fit_one_job(role, state=fit_state) for role in roles]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("full-budget fit parallelism requires Linux fork")
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
    fold_results = [row for row in fitted if row["role"].startswith("fold_")]
    fold_results.sort(key=lambda row: int(row["fold_index"]))
    full_refit = next(row for row in fitted if row["role"] == "full_refit")
    full_models = full_refit.pop("models")
    if tuple(full_refit["training_group_ids"]) != selected_group_ids:
        raise ValueError("full-budget deployment refit did not use every group")

    oof_candidates: dict[str, Any] = {}
    for candidate in CANDIDATE_KEYS:
        regrets = [
            float(
                fold["method_oof_evaluation"]["candidates"][candidate][
                    "equal_scenario_mean_normalized_action_regret"
                ]
            )
            for fold in fold_results
        ]
        alphas = [
            float(
                fold["method_oof_evaluation"]["candidates"][candidate][
                    "equal_scenario_mean_alpha"
                ]
            )
            for fold in fold_results
        ]
        oof_candidates[candidate] = {
            "fold_regrets": regrets,
            "mean_regret": float(np.mean(regrets)),
            "mean_alpha": float(np.mean(alphas)),
        }
    target_selection = select_target_candidate(
        oof_candidates,
        expected_fold_count=FOLD_COUNT,
        **SELECTOR_SPECS[FROZEN_SELECTOR_KEY],
    )

    oof_families = {
        family: {
            "fold_regrets": [
                float(
                    fold["baseline_oof_evaluation"]["families"][family][
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

    method_model_payload = {
        "protocol": artifact_protocols["method_model"],
        "city": city,
        "target_key": target_key,
        "scenarios": tuple(scenarios),
        "selected_candidate": target_selection["selected_candidate"],
        "selector_key": FROZEN_SELECTOR_KEY,
        "base_families": BASE_FAMILIES,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment["assignments"],
        "deployment_refit_group_ids": selected_group_ids,
        "deployment_refit_protocol": REFIT_PROTOCOL,
        "model_ensemble": (_project_models(full_models, BASE_FAMILIES),),
    }
    baseline_model_payload = {
        "protocol": artifact_protocols["baseline_model"],
        "city": city,
        "target_key": target_key,
        "scenarios": tuple(scenarios),
        "benchmark_families": BENCHMARK_FAMILIES,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment["assignments"],
        "deployment_refit_group_ids": selected_group_ids,
        "deployment_refit_protocol": REFIT_PROTOCOL,
        "model_ensemble": (_project_models(full_models, BENCHMARK_FAMILIES),),
    }
    method_model_artifact = _write_model_artifact(
        method_model_out,
        method_model_payload,
        protocol=artifact_protocols["method_model"],
        city=city,
        families=BASE_FAMILIES,
    )
    baseline_model_artifact = _write_model_artifact(
        baseline_model_out,
        baseline_model_payload,
        protocol=artifact_protocols["baseline_model"],
        city=city,
        families=BENCHMARK_FAMILIES,
    )

    common = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "protocol_generation": str(protocol["protocol"]),
        "parent_protocol_sha256": protocol["parent_protocol"]["sha256"],
        "development_audit_sha256": _sha256(development_audit_path),
        "diagnostic_failure_result_sha256": _sha256(
            diagnostic_failure_result_path
        ),
        "repair_evidence": (
            {
                "failed_v52_audit_sha256": _sha256(failed_v52_audit_path),
                "network_admission_audit_sha256": _sha256(
                    network_admission_audit_path
                ),
                "counterfactual_cache_audit_sha256": _sha256(
                    counterfactual_cache_audit_path
                ),
            }
            if str(protocol["protocol"]) == V9_PROTOCOL
            else None
        ),
        "development_decision": development_audit["development_gate"]["decision"],
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
        "source_city_groups": sorted(set(source_manifest.city_groups.values())),
        "external_targets_must_not_source_each_other": True,
        "target_group_budget": "all_available",
        "target_group_count": len(selected_group_ids),
        "adaptation_seeds": list(ADAPTATION_SEEDS),
        "diagnostic_seed_not_loaded": DIAGNOSTIC_ONLY_SEED,
        "evaluation_seed_not_loaded": evaluation_seed,
        "selection_audit": selection_audit,
        "selected_group_ids": list(selected_group_ids),
        "city_static_context": {
            "protocol": "equal-scenario-mean-label-free-static-context-v1",
            "context_names": list(selected_dataset.context_names),
            "scenario_values": {
                scenario: scenario_static_contexts[scenario].tolist()
                for scenario in scenarios
            },
            "city_value": city_static_context.tolist(),
        },
        "fold_assignment": fold_assignment,
        "fit_workers": actual_fit_workers,
        "combined_fit_families": list(COMBINED_FAMILIES),
        "fold_results": fold_results,
        "deployment_refit": full_refit,
        "freeze_gate": {
            "all_adaptation_groups_used": True,
            "seed_blocked_oof": True,
            "diagnostic_seed_absent": True,
            "evaluation_seed_absent": True,
            "target_cities_isolated": True,
            "oof_models_not_deployed": True,
            "single_full_refit_shared_by_method_and_baselines": True,
            "model_artifact_round_trip": True,
            "passed": True,
        },
        "elapsed_sec": float(time.monotonic() - started),
    }
    method_result = {
        **common,
        "protocol": artifact_protocols["method_freeze"],
        "oof_candidates": oof_candidates,
        "frozen_selector_key": FROZEN_SELECTOR_KEY,
        "target_selection": target_selection,
        "selected_candidate": target_selection["selected_candidate"],
        "model_artifact": method_model_artifact,
    }
    baseline_result = {
        **common,
        "protocol": artifact_protocols["baseline_freeze"],
        "benchmark_families": list(BENCHMARK_FAMILIES),
        "benchmark_claim_labels": {
            "simulator": "simulator_only",
            "dense": "h2oplus_style_dense_residual_not_exact_h2oplus",
            "dense_advantage": "same_estimand_dense_advantage",
            "causal_rigid_advantage": "pre_v41_rigid_cfcmt",
            "cfcmt_mechanism": "pre_v41_full_cfcmt",
            "causal_target_only": "target_only_full_adaptation_refit",
        },
        "oof_families": oof_families,
        "model_artifact": baseline_model_artifact,
    }
    gc.collect()
    return method_result, baseline_result


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
    parser.add_argument(
        "--diagnostic-failure-result", type=Path, required=True
    )
    parser.add_argument("--failed-v52-audit", type=Path)
    parser.add_argument("--network-admission-audit", type=Path)
    parser.add_argument("--counterfactual-cache-audit", type=Path)
    parser.add_argument("--city", required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-baseline-sha256", required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fit-workers", type=int, default=3)
    parser.add_argument("--method-model-out", type=Path, required=True)
    parser.add_argument("--baseline-model-out", type=Path, required=True)
    parser.add_argument("--method-out", type=Path, required=True)
    parser.add_argument("--baseline-out", type=Path, required=True)
    args = parser.parse_args(argv)
    for output in (
        args.method_out,
        args.baseline_out,
        args.method_model_out,
        args.baseline_model_out,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite full-budget freeze: {output}")
    method, baseline = run_external_full_budget_freeze(
        source_cache_root=args.source_cache_root,
        source_baseline_result=args.source_baseline_result,
        source_manifest_path=args.source_manifest,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_spec_path=args.protocol_spec,
        development_audit_path=args.development_audit,
        diagnostic_failure_result_path=args.diagnostic_failure_result,
        failed_v52_audit_path=args.failed_v52_audit,
        network_admission_audit_path=args.network_admission_audit,
        counterfactual_cache_audit_path=args.counterfactual_cache_audit,
        city=args.city,
        expected_source_cache_sha256=args.expected_source_cache_sha256,
        expected_source_baseline_sha256=args.expected_source_baseline_sha256,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        workers=args.workers,
        fit_workers=args.fit_workers,
        method_model_out=args.method_model_out,
        baseline_model_out=args.baseline_model_out,
    )
    _atomic_json(args.method_out, method)
    _atomic_json(args.baseline_out, baseline)
    print(
        json.dumps(
            {
                "city": args.city,
                "target_group_count": method["target_group_count"],
                "selected_candidate": method["selected_candidate"],
                "method_model_sha256": method["model_artifact"]["sha256"],
                "baseline_model_sha256": baseline["model_artifact"]["sha256"],
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
