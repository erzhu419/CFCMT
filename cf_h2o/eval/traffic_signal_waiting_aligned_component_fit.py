"""Fit B0/B100 source mechanisms and an equal-information target-only model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
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
    CORRECTION_FAMILY,
    SOURCE_SEEDS,
    evaluate_anchored_ensemble_candidates,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
    summarize_anchored_development,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    TARGET_GROUP_BUDGET,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _select_city_adaptation_groups,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _fit_family_v3,
    counterfactual_cost_contract_v5,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _relabel_dataset_domain,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_source_waiting_aligned_cache_audit import (
    RESULT_PROTOCOL as SOURCE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_target_waiting_aligned_cache_audit import (
    RESULT_PROTOCOL as TARGET_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.action_ranker import TargetOnlyActionAdvantageRegressor
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


PROTOCOL = "tsc-v115-pure-waiting-mechanism-fit-v3"
RESULT_PROTOCOL = "tsc-v115-pure-waiting-source-component-fit-v3"
MODEL_BUNDLE_PROTOCOL = "cfcmt-pure-waiting-source-component-bundle-v3"
STRICT_TARGET_MODEL_PROTOCOL = (
    "cfcmt-pure-waiting-architecture-matched-target-only-v2"
)
LEGACY_STRICT_TARGET_MODEL_PROTOCOL = "cfcmt-pure-waiting-strict-target-only-v1"
H2OPLUS_MODEL_PROTOCOL = "cfcmt-pure-waiting-h2oplus-dense-v2"
TARGET_ONLY_FAMILY = "architecture_matched_target_only_pure_waiting_v2"
LEGACY_TARGET_ONLY_FAMILY = "causal_target_only_pure_waiting_v1"
H2OPLUS_FAMILY = "dense"
_FIT_STATES: dict[str, Mapping[str, Any]] | None = None
_LOCO_SELECTION_STATE: Mapping[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _contract_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_target_label_free_blend_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(payload.get("blend_candidate_selection", {}))
    selected = str(payload.get("selected_candidate", ""))
    passed = bool(contract.get("gate_passed", False))
    expected_decision = (
        "freeze_source_loco_candidate"
        if passed
        else "suppress_residual_and_freeze_rigid_anchor"
    )
    if (
        contract.get("protocol")
        != "pure-waiting-target-label-free-source-loco-blend-selection-v1"
        or int(contract.get("target_transition_labels_consumed", -1)) != 0
        or set(contract.get("source_city_groups", ()))
        != set(DEVELOPMENT_CITY_GROUPS)
        or int(contract.get("candidate_count", -1)) != len(CANDIDATE_KEYS)
        or contract.get("selected_candidate") != selected
        or selected not in CANDIDATE_KEYS
        or contract.get("fallback_candidate") != ANCHOR_KEY
        or contract.get("decision") != expected_decision
        or (not passed and selected != ANCHOR_KEY)
        or len(str(contract.get("selection_sha256", ""))) != 64
    ):
        raise ValueError("target-label-free source LOCO blend contract changed")
    return contract


def validate_architecture_matched_target_contract(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(payload.get("architecture_match", {}))
    diagnostics = dict(payload.get("fit_diagnostics", {}))
    if (
        payload.get("protocol") != STRICT_TARGET_MODEL_PROTOCOL
        or payload.get("family") != TARGET_ONLY_FAMILY
        or contract.get("protocol")
        != "pure-waiting-architecture-matched-target-only-v1"
        or tuple(contract.get("base_families", ())) != tuple(BASE_FAMILIES)
        or contract.get("selected_candidate")
        != payload.get("selected_candidate")
        or contract.get("blend_candidate_selection_sha256")
        != payload.get("blend_candidate_selection_sha256")
        or int(contract.get("source_transition_rows_consumed", -1)) != 0
        or int(diagnostics.get("source_row_count_consumed", -1)) != 0
        or int(diagnostics.get("target_group_count", -1)) != 100
    ):
        raise ValueError("architecture-matched target-only contract changed")
    return contract


def _source_shard_contract(
    protocol: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    scenarios: Sequence[str],
) -> tuple[int, dict[str, int]]:
    source_training = dict(protocol["source_training"])
    default_shards = int(source_training["collection_shards"])
    overrides = {
        str(name): int(value)
        for name, value in dict(
            source_training.get("scenario_collection_shards", {})
        ).items()
    }
    scenario_set = {str(value) for value in scenarios}
    if (
        default_shards < 1
        or set(overrides) - scenario_set
        or any(value < 1 for value in overrides.values())
    ):
        raise ValueError("source fit shard contract is invalid")
    expected = {
        scenario: int(overrides.get(scenario, default_shards))
        for scenario in scenario_set
    }
    observed = {
        str(name): int(value)
        for name, value in dict(
            source_audit.get("shards_per_scenario_seed", {})
        ).items()
    }
    if observed != expected:
        raise ValueError(
            "source cache shard partition differs from the fit contract: "
            f"{observed} != {expected}"
        )
    return default_shards, overrides


def _assert_waiting_aligned_bank(
    bank: Mapping[str, Any],
    *,
    target_name: str,
    tolerance: float,
) -> None:
    for scenario, dataset in bank.items():
        if (
            dataset.metadata.get("counterfactual_cost_mode") != "halted_queue"
            or dataset.metadata.get("counterfactual_estimand")
            != WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6
            or dataset.metadata.get("counterfactual_cost_population")
            != "sumo_last_step_halting_number_on_controlled_lanes"
            or target_name not in dataset.targets
        ):
            raise ValueError(f"{scenario}: pure waiting-aligned target is absent")
        interval_cost = np.asarray(dataset.targets["interval_cost"], dtype=float)
        waiting_target = np.asarray(dataset.targets[target_name], dtype=float)
        if (
            interval_cost.shape != waiting_target.shape
            or not np.all(np.isfinite(interval_cost))
            or not np.allclose(
                interval_cost,
                waiting_target,
                rtol=0.0,
                atol=float(tolerance),
            )
        ):
            raise ValueError(f"{scenario}: training and 450-second targets differ")


def _fit_component_budget(
    state: Mapping[str, Any],
    *,
    target_group_ids: Sequence[str],
) -> Any:
    budget = len(target_group_ids)
    return fit_target_screening_models(
        state["fit_bank"],
        state["target_key"],
        fit_config=MechanismFitConfig(),
        source_rule_costs={},
        source_rule_specs=state["source_rule_specs"],
        model_families=BASE_FAMILIES,
        target_group_budget=budget,
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=1,
        scenario_city_groups=state["city_groups"],
        target_adaptation_group_ids=tuple(target_group_ids),
        target_static_context=state["target_static_context"],
        prior_policy_override=state["prior_policy"],
    )


def _fit_component_worker(source_city_group: str) -> dict[str, Any]:
    if _FIT_STATES is None:
        raise RuntimeError("waiting-aligned component fit state is missing")
    from threadpoolctl import threadpool_limits

    state = _FIT_STATES[source_city_group]
    started = time.monotonic()
    with threadpool_limits(limits=1):
        b0 = _fit_component_budget(state, target_group_ids=())
        b100 = _fit_component_budget(
            state,
            target_group_ids=state["selected_group_ids"],
        )
    b0_domains = set(b0.diagnostics["source_domains"])
    b100_domains = set(b100.diagnostics["source_domains"])
    expected_b0 = {source_city_group}
    expected_b100 = {source_city_group, str(state["city"])}
    if b0_domains != expected_b0 or b100_domains != expected_b100:
        raise ValueError(
            f"{source_city_group}: B0/B100 information boundary failed: "
            f"{b0_domains}, {b100_domains}"
        )
    return {
        "source_city_group": source_city_group,
        "b0": b0,
        "b100": b100,
        "diagnostics": {
            "b0": b0.diagnostics,
            "b100": b100.diagnostics,
            "elapsed_sec": float(time.monotonic() - started),
        },
    }


def _fit_h2oplus_budget(
    *,
    fit_bank: Mapping[str, Any],
    target_key: str,
    city_groups: Mapping[str, str],
    target_group_ids: Sequence[str],
    target_static_context: Sequence[float],
    prior_policy: str,
    source_rule_specs: Mapping[str, Any],
) -> Any:
    """Fit the dense H2O+-style comparator on the exact CFCMT information set."""

    return fit_target_screening_models(
        fit_bank,
        target_key,
        fit_config=MechanismFitConfig(),
        source_rule_costs={},
        source_rule_specs=source_rule_specs,
        model_families=(H2OPLUS_FAMILY,),
        target_group_budget=len(target_group_ids),
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=1,
        scenario_city_groups=city_groups,
        target_adaptation_group_ids=tuple(target_group_ids),
        target_static_context=target_static_context,
        prior_policy_override=prior_policy,
    )


def _evaluate_source_loco_heldout(heldout: str) -> dict[str, Any]:
    if _LOCO_SELECTION_STATE is None:
        raise RuntimeError("source LOCO selection state is missing")
    from threadpoolctl import threadpool_limits

    state = _LOCO_SELECTION_STATE
    scenarios = tuple(state["source_scenarios_by_group"][heldout])
    heldout_dataset = _merge_city_datasets(
        state["source_bank"],
        scenarios,
        city=heldout,
    )
    model_ensemble = tuple(
        state["b0_models"][group]
        for group in DEVELOPMENT_CITY_GROUPS
        if group != heldout
    )
    with threadpool_limits(limits=1):
        evaluation = evaluate_anchored_ensemble_candidates(
            heldout_dataset,
            model_ensemble=model_ensemble,
            excluded_group_ids=(),
        )
    candidates = dict(evaluation["candidates"])
    if set(candidates) != set(CANDIDATE_KEYS):
        raise ValueError(f"{heldout}: source LOCO candidate grid changed")
    return {
        "heldout": heldout,
        "city_rows": {
            candidate: {
                "regret": float(row["mean_normalized_action_regret"]),
                "mean_alpha": float(row["mean_alpha"]),
            }
            for candidate, row in candidates.items()
        },
        "audit": {
            "scenario_count": len(scenarios),
            "evaluation_group_count": int(evaluation["evaluation_group_count"]),
            "model_ensemble_size": int(evaluation["model_ensemble_size"]),
            "excluded_training_city": heldout,
        },
    }


def _select_source_loco_candidate(
    *,
    source_bank: Mapping[str, Any],
    source_scenarios_by_group: Mapping[str, Sequence[str]],
    b0_models: Mapping[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    """Select residual strength without consuming target-city labels."""

    expected_groups = set(DEVELOPMENT_CITY_GROUPS)
    if (
        set(source_scenarios_by_group) != expected_groups
        or set(b0_models) != expected_groups
        or any(not tuple(source_scenarios_by_group[group]) for group in expected_groups)
    ):
        raise ValueError("source LOCO candidate-selection city groups changed")

    actual_workers = min(max(int(workers), 1), len(DEVELOPMENT_CITY_GROUPS))
    global _LOCO_SELECTION_STATE
    _LOCO_SELECTION_STATE = {
        "source_bank": source_bank,
        "source_scenarios_by_group": source_scenarios_by_group,
        "b0_models": b0_models,
    }
    try:
        if actual_workers == 1:
            rows = [
                _evaluate_source_loco_heldout(group)
                for group in DEVELOPMENT_CITY_GROUPS
            ]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("parallel source LOCO selection requires Linux fork")
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                rows = list(
                    pool.map(
                        _evaluate_source_loco_heldout,
                        DEVELOPMENT_CITY_GROUPS,
                    )
                )
    finally:
        _LOCO_SELECTION_STATE = None
    city_rows = {row["heldout"]: row["city_rows"] for row in rows}
    evaluation_audit = {row["heldout"]: row["audit"] for row in rows}

    selection = summarize_anchored_development(city_rows)
    passed = bool(selection["passed"])
    selected = (
        str(selection["global_selection"]["selected_candidate"])
        if passed
        else ANCHOR_KEY
    )
    if selected not in CANDIDATE_KEYS:
        raise ValueError("source LOCO selected an unknown blend candidate")
    return {
        "protocol": "pure-waiting-target-label-free-source-loco-blend-selection-v1",
        "target_transition_labels_consumed": 0,
        "source_city_groups": list(DEVELOPMENT_CITY_GROUPS),
        "candidate_count": len(CANDIDATE_KEYS),
        "evaluation_workers": actual_workers,
        "gate_passed": passed,
        "selected_candidate": selected,
        "fallback_candidate": ANCHOR_KEY,
        "decision": (
            "freeze_source_loco_candidate"
            if passed
            else "suppress_residual_and_freeze_rigid_anchor"
        ),
        "evaluation_audit": evaluation_audit,
        "selection": selection,
    }


def run_waiting_aligned_component_fit(
    *,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    target_cache_root: Path,
    target_manifest_path: Path,
    target_cache_audit_path: Path,
    city: str,
    cache_workers: int,
    fit_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite component fit: {output_root}")
    protocol = _read_json(fit_protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("waiting-aligned component fit protocol changed")
    if protocol.get("required_counterfactual_cost_contract") != (
        counterfactual_cost_contract_v5("halted_queue")
    ):
        raise ValueError("pure halted-queue fit contract changed")
    if tuple(protocol.get("same_information_baseline_families", ())) != (
        H2OPLUS_FAMILY,
    ):
        raise ValueError("pure-waiting H2O+ baseline family changed")
    if tuple(protocol.get("source_component_families", ())) != tuple(
        BASE_FAMILIES
    ):
        raise ValueError("pure-waiting source component families changed")
    source_audit = _read_json(source_cache_audit_path)
    target_audit = _read_json(target_cache_audit_path)
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or not source_audit.get("gate", {}).get("passed", False)
    ):
        raise ValueError("source cache audit does not authorize mechanism fit")
    if (
        target_audit.get("protocol") != TARGET_AUDIT_PROTOCOL
        or target_audit.get("status") != "PASS"
        or not target_audit.get("gate", {}).get("passed", False)
    ):
        raise ValueError("target cache audit does not authorize B100 fit")

    source_manifest_full = load_traffic_signal_manifest(source_manifest_path)
    source_collection_shards, source_shard_overrides = _source_shard_contract(
        protocol,
        source_audit,
        tuple(source_manifest_full.sumocfgs),
    )
    admitted_scenarios = tuple(str(value) for value in source_audit["training_scenarios"])
    source_manifest = replace(
        source_manifest_full,
        scenarios=tuple(
            row
            for row in source_manifest_full.scenarios
            if row.scenario in set(admitted_scenarios)
        ),
    )
    if (
        len(source_manifest.scenarios)
        != int(protocol["source_training"]["admitted_scenario_count"])
        or set(source_manifest.sumocfgs) != set(admitted_scenarios)
        or len(set(source_manifest.city_groups.values()))
        != int(protocol["source_training"]["source_city_group_count"])
    ):
        raise ValueError("source admission manifest changed before fit")
    target_manifest = load_traffic_signal_manifest(target_manifest_path)
    if city not in set(target_manifest.city_groups.values()):
        raise ValueError(f"unknown target city: {city}")

    source_bank, source_bank_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        source_manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=source_collection_shards,
        scenario_collection_shards={
            name: count
            for name, count in source_shard_overrides.items()
            if name in source_manifest.sumocfgs
        },
        workers=max(int(cache_workers), 1),
    )
    target_bank, target_bank_audit = load_frozen_counterfactual_bank(
        target_cache_root,
        target_manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(cache_workers), 1),
    )
    target_name = str(protocol["target_name"])
    tolerance = float(protocol["required_target_equivalence_tolerance"])
    _assert_waiting_aligned_bank(
        source_bank,
        target_name=target_name,
        tolerance=tolerance,
    )
    _assert_waiting_aligned_bank(
        target_bank,
        target_name=target_name,
        tolerance=tolerance,
    )

    scenarios = tuple(
        name for name, group in target_manifest.city_groups.items() if group == city
    )
    selected_group_ids, selection_audit = _select_city_adaptation_groups(
        target_bank,
        city=city,
        scenarios=scenarios,
    )
    if len(selected_group_ids) != TARGET_GROUP_BUDGET:
        raise ValueError("waiting-aligned target adaptation is not B100")
    full_target_dataset = _merge_city_datasets(target_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_target_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "waiting_aligned_equal_information_b100"
        },
    )
    target_context = np.mean(
        np.vstack([_static_context_from_dataset(target_bank[name]) for name in scenarios]),
        axis=0,
    )
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    prior_policy = str(protocol["prior_policy"])
    prior_spec = source_rule_specs[prior_policy]
    target_contrast = build_action_contrast_dataset(
        _relabel_dataset_domain(selected_dataset, city),
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    legacy_strict_model = TargetOnlyActionAdvantageRegressor(target_domain=city)
    legacy_strict_diagnostics = legacy_strict_model.fit(target_contrast)
    if (
        int(legacy_strict_diagnostics.get("source_row_count_consumed", -1)) != 0
        or int(legacy_strict_diagnostics.get("target_group_count", -1))
        != TARGET_GROUP_BUDGET
    ):
        raise ValueError("legacy strict target-only fit consumed wrong rows")

    source_groups = tuple(sorted(set(source_manifest.city_groups.values())))
    target_key = f"waiting_aligned_target_{city}"
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
        fit_bank[target_key] = full_target_dataset
        city_groups = {
            name: source_manifest.city_groups[name] for name in group_scenarios
        }
        city_groups[target_key] = city
        states[source_group] = {
            "fit_bank": fit_bank,
            "target_key": target_key,
            "city_groups": city_groups,
            "selected_group_ids": selected_group_ids,
            "target_static_context": target_context,
            "prior_policy": prior_policy,
            "source_rule_specs": source_rule_specs,
            "city": city,
        }

    actual_fit_workers = min(max(int(fit_workers), 1), len(source_groups))
    if actual_fit_workers == 1:
        global _FIT_STATES
        _FIT_STATES = states
        try:
            fitted = [_fit_component_worker(group) for group in source_groups]
        finally:
            _FIT_STATES = None
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("waiting-aligned component fit requires Linux fork")
        _FIT_STATES = states
        try:
            with ProcessPoolExecutor(
                max_workers=actual_fit_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                futures = {
                    group: pool.submit(_fit_component_worker, group)
                    for group in source_groups
                }
                fitted = [futures[group].result() for group in source_groups]
        finally:
            _FIT_STATES = None

    b0_models = {row["source_city_group"]: row["b0"] for row in fitted}
    b100_models = {row["source_city_group"]: row["b100"] for row in fitted}
    component_diagnostics = {
        row["source_city_group"]: row["diagnostics"] for row in fitted
    }
    h2oplus_bank = dict(source_bank)
    h2oplus_bank[target_key] = full_target_dataset
    h2oplus_city_groups = {
        name: source_manifest.city_groups[name] for name in source_bank
    }
    h2oplus_city_groups[target_key] = city
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        h2oplus_b0 = _fit_h2oplus_budget(
            fit_bank=h2oplus_bank,
            target_key=target_key,
            city_groups=h2oplus_city_groups,
            target_group_ids=(),
            target_static_context=target_context,
            prior_policy=prior_policy,
            source_rule_specs=source_rule_specs,
        )
        h2oplus_b100 = _fit_h2oplus_budget(
            fit_bank=h2oplus_bank,
            target_key=target_key,
            city_groups=h2oplus_city_groups,
            target_group_ids=selected_group_ids,
            target_static_context=target_context,
            prior_policy=prior_policy,
            source_rule_specs=source_rule_specs,
        )
    expected_h2oplus_domains = set(source_groups)
    if (
        set(h2oplus_b0.family_models) != {H2OPLUS_FAMILY}
        or set(h2oplus_b100.family_models) != {H2OPLUS_FAMILY}
        or set(h2oplus_b0.diagnostics["source_domains"])
        != expected_h2oplus_domains
        or set(h2oplus_b100.diagnostics["source_domains"])
        != expected_h2oplus_domains | {city}
    ):
        raise ValueError("pure-waiting H2O+ information boundary failed")
    selection_spec = dict(protocol.get("blend_candidate_selection", {}))
    if (
        selection_spec.get("protocol")
        != "pure-waiting-target-label-free-source-loco-blend-selection-v1"
        or int(selection_spec.get("target_transition_labels_consumed", -1)) != 0
        or selection_spec.get("fallback_candidate") != ANCHOR_KEY
    ):
        raise ValueError("target-label-free blend selection contract changed")
    candidate_selection = _select_source_loco_candidate(
        source_bank=source_bank,
        source_scenarios_by_group=source_scenarios_by_group,
        b0_models=b0_models,
        workers=actual_fit_workers,
    )
    candidate = str(candidate_selection["selected_candidate"])
    candidate_selection_sha256 = _contract_sha256(candidate_selection)
    matched_target_family_models: dict[str, Any] = {}
    matched_target_family_diagnostics: dict[str, Any] = {}
    with threadpool_limits(limits=1):
        for family in BASE_FAMILIES:
            model, diagnostics = _fit_family_v3(
                target_contrast,
                family,
                MechanismFitConfig(),
                target_domain=city,
            )
            matched_target_family_models[family] = model
            matched_target_family_diagnostics[family] = diagnostics
    target_group_count = int(
        np.unique(np.asarray(target_contrast.metadata["action_group_ids"], dtype=str)).size
    )
    target_domains = set(np.asarray(target_contrast.domains, dtype=str).tolist())
    if target_group_count != TARGET_GROUP_BUDGET or target_domains != {city}:
        raise ValueError(
            "architecture-matched target-only fit did not consume exactly B100 target rows"
        )
    matched_target_model = FrozenAnchoredBlendModel(
        anchor_model=matched_target_family_models[ANCHOR_FAMILY],
        correction_model=matched_target_family_models[CORRECTION_FAMILY],
        candidate=candidate,
        anchor_objective_mode="control_only",
        correction_objective_mode="control_only",
    )
    candidate_selection_contract = {
        name: candidate_selection[name]
        for name in (
            "protocol",
            "target_transition_labels_consumed",
            "source_city_groups",
            "candidate_count",
            "gate_passed",
            "selected_candidate",
            "fallback_candidate",
            "decision",
        )
    }
    candidate_selection_contract["selection_sha256"] = (
        candidate_selection_sha256
    )
    adaptation_contract = {
        "protocol": "waiting-aligned-equal-information-adaptation-contract-v1",
        "city": city,
        "scenarios": list(scenarios),
        "selected_group_ids": list(selected_group_ids),
        "selection_audit": selection_audit,
        "prior_policy": prior_policy,
        "target_name": target_name,
        "blend_candidate": candidate,
        "blend_candidate_selection_sha256": candidate_selection_sha256,
        "blend_candidate_target_transition_labels": 0,
        "target_only_comparator": (
            "same_anchor_same_correction_same_blend_b100_without_source_rows"
        ),
        "source_cache_audit_sha256": _sha256(source_cache_audit_path),
        "target_cache_audit_sha256": _sha256(target_cache_audit_path),
    }
    adaptation_sha256 = _contract_sha256(adaptation_contract)
    output_root.mkdir(parents=True, exist_ok=False)
    strict_path = output_root / "strict_target_b100.pkl"
    legacy_strict_path = output_root / "legacy_strict_target_b100.pkl"
    b0_path = output_root / "source_components_b0.pkl"
    b100_path = output_root / "source_components_b100.pkl"
    h2oplus_b0_path = output_root / "h2oplus_dense_b0.pkl"
    h2oplus_b100_path = output_root / "h2oplus_dense_b100.pkl"
    matched_target_diagnostics = {
        "estimator": "architecture_matched_target_only_pure_waiting_v2",
        "target_domain": city,
        "target_group_count": target_group_count,
        "target_row_count": int(target_contrast.size),
        "source_row_count_consumed": 0,
        "selected_candidate": candidate,
        "families": matched_target_family_diagnostics,
    }
    architecture_match = {
        "protocol": "pure-waiting-architecture-matched-target-only-v1",
        "base_families": list(BASE_FAMILIES),
        "selected_candidate": candidate,
        "blend_candidate_selection_sha256": candidate_selection_sha256,
        "same_family_hyperparameters_as_source_components": True,
        "same_b100_group_ids_as_source_components": True,
        "source_transition_rows_consumed": 0,
    }
    strict_payload = {
        "protocol": STRICT_TARGET_MODEL_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "family": TARGET_ONLY_FAMILY,
        "objective_mode": "control_only",
        "selected_group_ids": selected_group_ids,
        "prior_policy": prior_policy,
        "target_name": target_name,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "main_model_sha256": adaptation_sha256,
        "adaptation_contract_sha256": adaptation_sha256,
        "selected_candidate": candidate,
        "blend_candidate_selection_sha256": candidate_selection_sha256,
        "architecture_match": architecture_match,
        "model": matched_target_model,
        "fit_diagnostics": matched_target_diagnostics,
    }
    legacy_strict_payload = {
        "protocol": LEGACY_STRICT_TARGET_MODEL_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "family": LEGACY_TARGET_ONLY_FAMILY,
        "objective_mode": "control_only",
        "selected_group_ids": selected_group_ids,
        "prior_policy": prior_policy,
        "target_name": target_name,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "main_model_sha256": adaptation_sha256,
        "adaptation_contract_sha256": adaptation_sha256,
        "model": legacy_strict_model,
        "fit_diagnostics": legacy_strict_diagnostics,
    }
    validate_architecture_matched_target_contract(strict_payload)
    common_bundle = {
        "protocol": MODEL_BUNDLE_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "source_city_groups": source_groups,
        "source_scenarios_by_group": source_scenarios_by_group,
        "selected_candidate": candidate,
        "candidate_selection_sha256": candidate_selection_sha256,
        "blend_candidate_selection": candidate_selection_contract,
        "prior_policy": prior_policy,
        "target_name": target_name,
        "main_model_sha256": adaptation_sha256,
        "adaptation_contract_sha256": adaptation_sha256,
    }
    b0_payload = {
        **common_bundle,
        "target_group_budget": 0,
        "selected_group_ids": (),
        "component_models": b0_models,
    }
    b100_payload = {
        **common_bundle,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": selected_group_ids,
        "component_models": b100_models,
    }
    validate_target_label_free_blend_contract(b0_payload)
    validate_target_label_free_blend_contract(b100_payload)
    h2oplus_common = {
        "protocol": H2OPLUS_MODEL_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "family": H2OPLUS_FAMILY,
        "objective_mode": "control_only",
        "source_city_groups": source_groups,
        "prior_policy": prior_policy,
        "target_name": target_name,
        "adaptation_contract_sha256": adaptation_sha256,
    }
    h2oplus_b0_payload = {
        **h2oplus_common,
        "target_group_budget": 0,
        "selected_group_ids": (),
        "model": h2oplus_b0.family_models[H2OPLUS_FAMILY],
        "prior_spec": h2oplus_b0.prior_spec,
        "fit_diagnostics": h2oplus_b0.diagnostics,
    }
    h2oplus_b100_payload = {
        **h2oplus_common,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": selected_group_ids,
        "model": h2oplus_b100.family_models[H2OPLUS_FAMILY],
        "prior_spec": h2oplus_b100.prior_spec,
        "fit_diagnostics": h2oplus_b100.diagnostics,
    }
    _atomic_bytes(
        strict_path,
        pickle.dumps(strict_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    _atomic_bytes(
        legacy_strict_path,
        pickle.dumps(legacy_strict_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    _atomic_bytes(b0_path, pickle.dumps(b0_payload, protocol=pickle.HIGHEST_PROTOCOL))
    _atomic_bytes(
        b100_path,
        pickle.dumps(b100_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    _atomic_bytes(
        h2oplus_b0_path,
        pickle.dumps(h2oplus_b0_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    _atomic_bytes(
        h2oplus_b100_path,
        pickle.dumps(h2oplus_b100_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    round_trip = [
        pickle.loads(path.read_bytes())
        for path in (
            strict_path,
            legacy_strict_path,
            b0_path,
            b100_path,
            h2oplus_b0_path,
            h2oplus_b100_path,
        )
    ]
    if (
        round_trip[0].get("protocol") != STRICT_TARGET_MODEL_PROTOCOL
        or round_trip[1].get("protocol") != LEGACY_STRICT_TARGET_MODEL_PROTOCOL
        or any(
            row.get("protocol") != MODEL_BUNDLE_PROTOCOL
            for row in round_trip[2:4]
        )
        or set(round_trip[3].get("component_models", {})) != set(source_groups)
        or any(
            row.get("protocol") != H2OPLUS_MODEL_PROTOCOL
            or row.get("family") != H2OPLUS_FAMILY
            for row in round_trip[4:]
        )
    ):
        raise ValueError("waiting-aligned model artifact round trip failed")
    validate_architecture_matched_target_contract(round_trip[0])
    payload = {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "pure-waiting-development-fit_not_confirmation",
        "city": city,
        "scenarios": list(scenarios),
        "source_city_groups": list(source_groups),
        "source_scenarios_by_group": {
            key: list(value) for key, value in source_scenarios_by_group.items()
        },
        "target_name": target_name,
        "prior_policy": prior_policy,
        "frozen_blend_candidate": candidate,
        "blend_candidate_selection": candidate_selection,
        "blend_candidate_selection_sha256": candidate_selection_sha256,
        "information_budget": {
            "zero_shot_target_transition_labels": 0,
            "blend_candidate_target_transition_labels": 0,
            "blend_candidate_selection_protocol": candidate_selection[
                "protocol"
            ],
            "few_shot_target_action_groups": TARGET_GROUP_BUDGET,
            "target_only_action_groups": TARGET_GROUP_BUDGET,
            "target_only_architecture_matched": True,
            "target_only_source_rows_consumed": 0,
            "same_b100_group_ids": True,
            "selected_group_ids": list(selected_group_ids),
            "selection_audit": selection_audit,
        },
        "adaptation_contract": adaptation_contract,
        "adaptation_contract_sha256": adaptation_sha256,
        "architecture_matched_target_diagnostics": matched_target_diagnostics,
        "legacy_strict_target_diagnostics": legacy_strict_diagnostics,
        "component_diagnostics": component_diagnostics,
        "h2oplus_diagnostics": {
            "b0": h2oplus_b0.diagnostics,
            "b100": h2oplus_b100.diagnostics,
        },
        "source_cache_bank_audit": source_bank_audit,
        "target_cache_bank_audit": target_bank_audit,
        "input_audits": {
            "source": {
                "path": str(Path(source_cache_audit_path).resolve()),
                "sha256": _sha256(source_cache_audit_path),
            },
            "target": {
                "path": str(Path(target_cache_audit_path).resolve()),
                "sha256": _sha256(target_cache_audit_path),
            },
        },
        "model_artifacts": {
            "strict_target_b100": {
                "path": str(strict_path.resolve()),
                "sha256": _sha256(strict_path),
                "size_bytes": strict_path.stat().st_size,
            },
            "legacy_strict_target_b100": {
                "path": str(legacy_strict_path.resolve()),
                "sha256": _sha256(legacy_strict_path),
                "size_bytes": legacy_strict_path.stat().st_size,
            },
            "source_components_b0": {
                "path": str(b0_path.resolve()),
                "sha256": _sha256(b0_path),
                "size_bytes": b0_path.stat().st_size,
            },
            "source_components_b100": {
                "path": str(b100_path.resolve()),
                "sha256": _sha256(b100_path),
                "size_bytes": b100_path.stat().st_size,
            },
            "h2oplus_dense_b0": {
                "path": str(h2oplus_b0_path.resolve()),
                "sha256": _sha256(h2oplus_b0_path),
                "size_bytes": h2oplus_b0_path.stat().st_size,
            },
            "h2oplus_dense_b100": {
                "path": str(h2oplus_b100_path.resolve()),
                "sha256": _sha256(h2oplus_b100_path),
                "size_bytes": h2oplus_b100_path.stat().st_size,
            },
        },
        "fit_workers": actual_fit_workers,
        "elapsed_sec": float(time.monotonic() - started),
        "claim_boundary": protocol["claim_boundary"],
    }
    _atomic_json(output_root / "result.json", payload)
    gc.collect()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fit-workers", type=int, default=7)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_waiting_aligned_component_fit(
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        target_cache_root=args.target_cache_root,
        target_manifest_path=args.target_manifest,
        target_cache_audit_path=args.target_cache_audit,
        city=args.city,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "city": result["city"],
                "source_city_group_count": len(result["source_city_groups"]),
                "b100_group_count": len(
                    result["information_budget"]["selected_group_ids"]
                ),
                "result": str(args.output_root / "result.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
