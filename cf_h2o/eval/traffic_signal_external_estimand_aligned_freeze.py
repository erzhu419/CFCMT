"""Refit the external-city CFCMT method with an estimand-aligned pressure anchor."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
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
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    FROZEN_SELECTOR_KEY,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    FOLD_COUNT,
    PROTOCOL,
    V9_PROTOCOL,
    _fit_one_job,
    _validate_full_protocol,
    assign_leave_one_seed_out_folds,
    select_all_city_adaptation_groups,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    MIN_OPERATIONAL_TOTAL_VEHICLES,
    accepted_oof_records,
    oof_action_records,
    select_oof_trust_region,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


RESULT_PROTOCOL = "tsc-v46r42-external-oof-validated-support-freeze-v1"
MODEL_PROTOCOL = "cfcmt-external-oof-validated-support-phase-anchor-model-v1"
V9_RESULT_PROTOCOL = "tsc-v55r51-external-v9-oof-validated-support-freeze-v1"
V9_MODEL_PROTOCOL = (
    "cfcmt-external-v9-oof-validated-support-phase-anchor-model-v1"
)
ANCHOR_POLICY = "phase_pressure"
_FIT_STATE: dict[str, Any] | None = None


def _fit_worker(role: str) -> dict[str, Any]:
    if _FIT_STATE is None:
        raise RuntimeError("estimand-aligned fit state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _fit_one_job(role, state=_FIT_STATE)


def _target_support(dataset, *, prior_spec) -> TargetActionSupport:
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    return TargetActionSupport.fit(contrast)


def _group_seeds(group_ids: Sequence[str]) -> set[int]:
    seeds: set[int] = set()
    for group_id in group_ids:
        matches = [
            part for part in str(group_id).split(":") if part.startswith("seed")
        ]
        if len(matches) != 1:
            raise ValueError(f"group has no unique simulator seed: {group_id}")
        seeds.add(int(matches[0][4:]))
    return seeds


def _write_model(
    path: Path,
    payload: Mapping[str, Any],
    *,
    city: str,
    model_protocol: str = MODEL_PROTOCOL,
) -> dict[str, Any]:
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite estimand-aligned model: {path}")
    _atomic_bytes(path, pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL))
    loaded = pickle.loads(Path(path).read_bytes())
    models = loaded.get("models")
    support = loaded.get("target_support")
    if (
        loaded.get("protocol") != model_protocol
        or loaded.get("city") != city
        or set(getattr(models, "family_models", {})) != set(BASE_FAMILIES)
        or not isinstance(support, TargetActionSupport)
        or str(getattr(models.prior_spec, "key", "")) != ANCHOR_POLICY
    ):
        raise ValueError("estimand-aligned model round-trip failed")
    return {
        "protocol": model_protocol,
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(Path(path).stat().st_size),
        "round_trip_passed": True,
    }


def run_estimand_aligned_freeze(
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
    parent_joint_audit_path: Path,
    city: str,
    expected_source_cache_sha256: str,
    expected_source_baseline_sha256: str,
    expected_external_cache_sha256: str,
    expected_parent_joint_audit_sha256: str,
    expected_source_tree_sha256: str | None,
    workers: int,
    fit_workers: int,
    model_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol, parent, _ = _validate_full_protocol(
        protocol_spec_path,
        development_audit_path,
        diagnostic_failure_result_path,
        failed_v52_audit_path,
        network_admission_audit_path,
        counterfactual_cache_audit_path,
    )
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name == PROTOCOL:
        result_protocol = RESULT_PROTOCOL
        model_protocol = MODEL_PROTOCOL
    elif protocol_name == V9_PROTOCOL:
        result_protocol = V9_RESULT_PROTOCOL
        model_protocol = V9_MODEL_PROTOCOL
    else:
        raise ValueError("estimand-aligned full-budget protocol changed")
    parent_joint = json.loads(Path(parent_joint_audit_path).read_text(encoding="utf-8"))
    if (
        _sha256(parent_joint_audit_path) != str(expected_parent_joint_audit_sha256)
        or parent_joint.get("status") != "PASS"
        or (
            protocol_name == V9_PROTOCOL
            and parent_joint.get("protocol")
            != "tsc-v54r50-external-v9-full-budget-joint-freeze-audit-v1"
        )
    ):
        raise ValueError("estimand-aligned parent evidence changed")
    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("estimand-aligned source-tree SHA-256 mismatch")

    source_evidence = dict(parent["source_evidence"])
    parent_cache = dict(parent["counterfactual_cache"])
    source_sha, source_files = aggregate_cache_sha256(source_cache_root)
    external_sha, external_files = aggregate_cache_sha256(external_cache_root)
    if (
        source_sha != str(expected_source_cache_sha256)
        or str(source_evidence["counterfactual_cache_sha256"])
        != str(expected_source_cache_sha256)
        or source_files != 18 * len(SOURCE_SEEDS) * COLLECTION_SHARDS
        or external_sha != str(expected_external_cache_sha256)
        or _sha256(source_baseline_result) != str(expected_source_baseline_sha256)
        or _sha256(source_manifest_path) != str(source_evidence["manifest_sha256"])
        or _sha256(external_manifest_path) != str(parent_cache["manifest_sha256"])
    ):
        raise ValueError("estimand-aligned cache or source evidence changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    source_manifest = load_traffic_signal_manifest(source_manifest_path)
    external_manifest = load_traffic_signal_manifest(external_manifest_path)
    city_scenarios = {
        str(name): tuple(str(value) for value in values)
        for name, values in protocol["target_protocol"]["external_city_scenarios"].items()
    }
    if city not in city_scenarios:
        raise ValueError(f"unknown estimand-aligned city: {city}")
    scenarios = city_scenarios[city]
    expected_external_files = (
        sum(len(values) for values in city_scenarios.values())
        * len(ADAPTATION_SEEDS)
        * EXTERNAL_COLLECTION_SHARDS
    )
    if external_files != expected_external_files:
        raise ValueError("estimand-aligned external cache file count changed")

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
        metadata_updates={"target_data_role": "estimand_aligned_adaptation_pool"},
    )
    group_records = target_group_records(
        selected_dataset,
        selected_group_ids=selected_group_ids,
        expected_group_count=len(selected_group_ids),
    )
    fold_assignment = assign_leave_one_seed_out_folds(group_records, scenarios=scenarios)
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
    source_baseline = json.loads(Path(source_baseline_result).read_text(encoding="utf-8"))
    state = {
        "fit_bank": fit_bank,
        "target_key": target_key,
        "city_groups": city_groups,
        "source_rule_costs": source_baseline["source_rule_policy_costs"],
        "source_rule_specs": {spec.key: spec for spec in generalized_pressure_grid()},
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment,
        "selected_dataset": selected_dataset,
        "scenarios": scenarios,
        "source_scenarios": tuple(source_manifest.sumocfgs),
        "source_city_groups": tuple(set(source_manifest.city_groups.values())),
        "external_cities": tuple(city_scenarios),
        "city": city,
        "city_static_context": city_static_context,
        "model_families": BASE_FAMILIES,
        "prior_policy_override": ANCHOR_POLICY,
        "return_fold_models": True,
    }
    roles = tuple(f"fold_{index}" for index in range(FOLD_COUNT)) + ("full_refit",)
    actual_workers = min(max(int(fit_workers), 1), len(roles))
    if actual_workers == 1:
        fitted = [_fit_one_job(role, state=state) for role in roles]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("estimand-aligned parallel fit requires Linux fork")
        global _FIT_STATE
        _FIT_STATE = state
        try:
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {role: pool.submit(_fit_worker, role) for role in roles}
                fitted = [futures[role].result() for role in roles]
        finally:
            _FIT_STATE = None
    folds = sorted(
        [row for row in fitted if row["role"].startswith("fold_")],
        key=lambda row: int(row["fold_index"]),
    )
    full = next(row for row in fitted if row["role"] == "full_refit")
    full_models = full.pop("models")
    if str(full_models.prior_spec.key) != ANCHOR_POLICY:
        raise ValueError("deployment refit did not use the phase-pressure estimand anchor")

    oof_candidates = {}
    for candidate in CANDIDATE_KEYS:
        regrets = [
            float(
                fold["method_oof_evaluation"]["candidates"][candidate][
                    "equal_scenario_mean_normalized_action_regret"
                ]
            )
            for fold in folds
        ]
        alphas = [
            float(
                fold["method_oof_evaluation"]["candidates"][candidate][
                    "equal_scenario_mean_alpha"
                ]
            )
            for fold in folds
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
    selected_candidate = str(target_selection["selected_candidate"])

    all_oof_records: list[dict[str, Any]] = []
    fold_results = []
    for fold in folds:
        fold_models = fold.pop("models")
        if str(fold_models.prior_spec.key) != ANCHOR_POLICY:
            raise ValueError("OOF fold did not use the phase-pressure estimand anchor")
        training = _group_subset_v3(
            selected_dataset,
            selected_groups=set(fold["training_group_ids"]),
            metadata_updates={"target_data_role": "estimand_aligned_support_fit"},
        )
        support = _target_support(training, prior_spec=fold_models.prior_spec)
        validation = _group_subset_v3(
            selected_dataset,
            selected_groups=set(fold["validation_group_ids"]),
            metadata_updates={"target_data_role": "estimand_aligned_oof_validation"},
        )
        records = oof_action_records(
            validation,
            models=fold_models,
            selected_candidate=selected_candidate,
            fold_index=int(fold["fold_index"]),
            validation_seed=int(fold["validation_seed"]),
            scenarios=scenarios,
            target_support=support,
        )
        all_oof_records.extend(records)
        fold_results.append(
            {
                **fold,
                "target_support": support.diagnostics(),
                "oof_action_record_count": len(records),
            }
        )
    guard, guard_selection = select_oof_trust_region(all_oof_records)
    adaptation_support = _target_support(
        selected_dataset, prior_spec=full_models.prior_spec
    )
    validated_records = accepted_oof_records(all_oof_records, guard)
    if guard.enabled:
        validated_folds = {int(row["fold_index"]) for row in validated_records}
        if validated_folds != set(range(FOLD_COUNT)):
            raise ValueError("validated OOF support does not cover every fold")
        deployment_support = TargetActionSupport.fit_validated_records(
            validated_records,
            minimum_total_vehicles=MIN_OPERATIONAL_TOTAL_VEHICLES,
        )
    else:
        deployment_support = adaptation_support
    guard_payload = {
        "enabled": guard.enabled,
        "risk_multiplier": guard.risk_multiplier,
        "min_context_trust": guard.min_context_trust,
        "margin": guard.margin,
        "max_relative_rule_gap": guard.max_relative_rule_gap,
    }
    model_payload = {
        "protocol": model_protocol,
        "city": city,
        "scenarios": tuple(scenarios),
        "anchor_policy": ANCHOR_POLICY,
        "selected_candidate": selected_candidate,
        "guard": guard,
        "models": full_models,
        "target_support": deployment_support,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": fold_assignment["assignments"],
    }
    model_artifact = _write_model(
        model_out,
        model_payload,
        city=city,
        model_protocol=model_protocol,
    )
    target_protocol = dict(protocol["target_protocol"])
    closed_loop_development = tuple(
        int(value)
        for value in target_protocol.get(
            "closed_loop_development_seeds",
            target_protocol.get("closed_loop_seeds", ()),
        )
    )
    closed_loop_validation = tuple(
        int(value)
        for value in target_protocol.get("closed_loop_validation_seeds", ())
    )
    prospective_confirmation = tuple(
        int(value)
        for value in target_protocol.get(
            "closed_loop_prospective_seeds", (10091, 11003, 12007)
        )
    )
    diagnostic_only = tuple(
        int(value) for value in target_protocol["diagnostic_only_seeds"]
    )
    offline_evaluation = tuple(
        int(value) for value in target_protocol["evaluation_seeds"]
    )
    adaptation_seed_set = set(ADAPTATION_SEEDS)
    selected_group_seed_set = _group_seeds(selected_group_ids)
    development_seed_set = set(closed_loop_development)
    adaptation_coupled_diagnostics = tuple(
        sorted(development_seed_set & adaptation_seed_set)
    )
    generalization_development = tuple(
        sorted(development_seed_set - adaptation_seed_set)
    )
    protected_seed_set = set(
        (*diagnostic_only, *offline_evaluation, *closed_loop_validation, *prospective_confirmation)
    )
    freeze_gate = {
        "all_adaptation_groups_used": True,
        "seed_blocked_oof_predictions": True,
        "target_support_fold_isolated": True,
        "counterfactual_and_deployment_anchor_aligned": True,
        "operational_domain_excludes_empty_local_states": bool(
            not guard.enabled
            or deployment_support.minimum_total_vehicles
            == MIN_OPERATIONAL_TOTAL_VEHICLES
        ),
        "deployment_support_uses_oof_safe_actions_only": bool(
            not guard.enabled
            or (
                not deployment_support.label_free
                and deployment_support.prototypes.shape[0]
                == len(validated_records)
            )
        ),
        "validated_support_covers_every_oof_fold": bool(
            not guard.enabled
            or {int(row["fold_index"]) for row in validated_records}
            == set(range(FOLD_COUNT))
        ),
        "selected_groups_use_adaptation_seeds_only": selected_group_seed_set
        == adaptation_seed_set,
        "protected_evaluation_seeds_absent": not (
            selected_group_seed_set & protected_seed_set
        ),
        "generalization_development_seeds_absent": not (
            selected_group_seed_set & set(generalization_development)
        ),
        "adaptation_coupled_diagnostics_declared": (
            set(adaptation_coupled_diagnostics)
            == (adaptation_seed_set if protocol_name == V9_PROTOCOL else set())
            and (selected_group_seed_set & development_seed_set)
            == set(adaptation_coupled_diagnostics)
        ),
        "model_artifact_round_trip": bool(model_artifact["round_trip_passed"]),
    }
    freeze_gate["passed"] = all(freeze_gate.values())
    return {
        "protocol": result_protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "claim_boundary": (
            "method refit and guard selection use adaptation seeds only; the "
            "estimand-aligned model, candidate, and guard do not use any v9 "
            "closed-loop efficacy outcome"
        ),
        "parent_protocol_sha256": _sha256(protocol_spec_path),
        "parent_joint_audit_sha256": _sha256(parent_joint_audit_path),
        "source_cache_aggregate_sha256": source_sha,
        "source_cache_file_count": source_files,
        "source_cache_audit": source_cache_audit,
        "external_cache_aggregate_sha256": external_sha,
        "external_cache_file_count": external_files,
        "external_cache_audit": external_cache_audit,
        "city": city,
        "scenarios": list(scenarios),
        "counterfactual_rollout_anchor": ANCHOR_POLICY,
        "deployment_fallback_anchor": ANCHOR_POLICY,
        "estimand_alignment": {
            "cache_behavior_policy": "phase_pressure",
            "cache_counterfactual_rollout_policy": "phase_pressure",
            "contrast_reference_policy": ANCHOR_POLICY,
            "deployment_fallback_policy": ANCHOR_POLICY,
            "passed": True,
        },
        "selected_group_ids": list(selected_group_ids),
        "selection_audit": selection_audit,
        "fold_assignment": fold_assignment,
        "fit_workers": actual_workers,
        "fold_results": fold_results,
        "oof_candidates": oof_candidates,
        "target_selection": target_selection,
        "selected_candidate": selected_candidate,
        "oof_action_record_count": len(all_oof_records),
        "oof_action_records": all_oof_records,
        "guard_selection": guard_selection,
        "deployment": {
            "policy": (
                "cfcmt_oof_validated_support_guard"
                if guard.enabled
                else ANCHOR_POLICY
            ),
            "guard": guard_payload,
            "target_support": deployment_support.diagnostics(),
            "adaptation_coverage_support": adaptation_support.diagnostics(),
            "validated_oof_group_ids": [
                str(row["group_id"]) for row in validated_records
            ],
            "validated_oof_fold_counts": {
                str(fold): sum(
                    int(row["fold_index"]) == fold for row in validated_records
                )
                for fold in range(FOLD_COUNT)
            },
            "coordination_mode": "sparse",
        },
        "deployment_refit": full,
        "model_artifact": model_artifact,
        "excluded_seed_roles": {
            "diagnostic_only": diagnostic_only,
            "offline_evaluation": offline_evaluation,
            "closed_loop_development": closed_loop_development,
            "closed_loop_adaptation_diagnostics": adaptation_coupled_diagnostics,
            "closed_loop_generalization_development": generalization_development,
            "closed_loop_validation": closed_loop_validation,
            "prospective_confirmation": prospective_confirmation,
        },
        "freeze_gate": freeze_gate,
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
    parser.add_argument("--diagnostic-failure-result", type=Path, required=True)
    parser.add_argument("--failed-v52-audit", type=Path)
    parser.add_argument("--network-admission-audit", type=Path)
    parser.add_argument("--counterfactual-cache-audit", type=Path)
    parser.add_argument("--parent-joint-audit", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-baseline-sha256", required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--expected-parent-joint-audit-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fit-workers", type=int, default=3)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.model_out.exists():
        raise FileExistsError("refusing to overwrite estimand-aligned freeze evidence")
    payload = run_estimand_aligned_freeze(
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
        parent_joint_audit_path=args.parent_joint_audit,
        city=args.city,
        expected_source_cache_sha256=args.expected_source_cache_sha256,
        expected_source_baseline_sha256=args.expected_source_baseline_sha256,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        expected_parent_joint_audit_sha256=args.expected_parent_joint_audit_sha256,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        workers=args.workers,
        fit_workers=args.fit_workers,
        model_out=args.model_out,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "city": args.city,
                "selected_candidate": payload["selected_candidate"],
                "guard_decision": payload["guard_selection"]["decision"],
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
