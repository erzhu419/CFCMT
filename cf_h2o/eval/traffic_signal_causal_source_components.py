"""Fit source-city causal components around a fixed target-only anchor."""

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

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import BASE_FAMILIES
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    _fit_one_job,
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
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    COLLECTION_SHARDS,
    SOURCE_SEEDS,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid


PROTOCOL = "tsc-v93-source-city-causal-component-fit-v1"
MODEL_BUNDLE_PROTOCOL = "cfcmt-source-city-causal-component-bundle-v1"
_COMPONENT_STATES: dict[str, Mapping[str, Any]] | None = None


def _fit_component_worker(source_city_group: str) -> dict[str, Any]:
    if _COMPONENT_STATES is None:
        raise RuntimeError("source-component fit state was not initialized")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        row = _fit_one_job(
            "full_refit",
            state=_COMPONENT_STATES[source_city_group],
        )
    row["source_city_group"] = source_city_group
    return row


def run_source_city_component_fit(
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
    model_out: Path,
    result_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if model_out.exists() or result_out.exists():
        raise FileExistsError("refusing to overwrite source-component fit")
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
    if main_result.get("source_cache_audit") != source_cache_audit:
        raise ValueError("main fit did not consume the same complete source bank")

    scenarios = tuple(str(value) for value in scenarios_by_city[city])
    if tuple(main_payload.get("scenarios", ())) != scenarios:
        raise ValueError("main model target scenarios changed")
    if set(scenarios) != {
        name
        for name, group in external_manifest.city_groups.items()
        if group == city
    }:
        raise ValueError("external city/scenario mapping changed")
    selected_group_ids = tuple(
        str(value) for value in main_payload["selected_group_ids"]
    )
    full_city_dataset = _merge_city_datasets(external_bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "source_city_component_fixed_target_adaptation"
        },
    )
    scenario_contexts = {
        scenario: _static_context_from_dataset(external_bank[scenario])
        for scenario in scenarios
    }
    city_context = np.mean(
        np.vstack([scenario_contexts[scenario] for scenario in scenarios]),
        axis=0,
    )
    baseline = json.loads(Path(source_baseline_result).read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    if set(source_manifest.sumocfgs) - set(source_rule_costs):
        raise ValueError("source rule baseline does not cover the source manifest")
    source_groups = tuple(sorted(set(source_manifest.city_groups.values())))
    target_key = f"external_city_{city}"
    main_offline = main_payload["model"]
    prior_policy = str(main_offline.prior_spec.key)
    source_rule_specs = {
        spec.key: spec for spec in generalized_pressure_grid()
    }
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
            "selected_dataset": selected_dataset,
            "scenarios": scenarios,
            "source_scenarios": group_scenarios,
            "source_city_groups": (source_group,),
            "external_cities": tuple(scenarios_by_city),
            "city": city,
            "city_static_context": city_context,
            "model_families": BASE_FAMILIES,
            "prior_policy_override": prior_policy,
        }

    actual_fit_workers = min(max(int(fit_workers), 1), len(source_groups))
    if actual_fit_workers == 1:
        fitted = []
        for source_group in source_groups:
            row = _fit_one_job("full_refit", state=states[source_group])
            row["source_city_group"] = source_group
            fitted.append(row)
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("source-component fit requires Linux fork")
        global _COMPONENT_STATES
        _COMPONENT_STATES = states
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
            _COMPONENT_STATES = None

    component_models = {}
    component_diagnostics = {}
    for row in fitted:
        source_group = str(row.pop("source_city_group"))
        component_models[source_group] = row.pop("models")
        component_diagnostics[source_group] = row
    model_payload = {
        "protocol": MODEL_BUNDLE_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "source_city_groups": source_groups,
        "source_scenarios_by_group": source_scenarios_by_group,
        "selected_candidate": str(main_payload["selected_candidate"]),
        "selected_group_ids": selected_group_ids,
        "prior_policy": prior_policy,
        "main_model_sha256": str(expected_main_model_sha256),
        "component_models": component_models,
        "topology_repair_protocol": main_payload["topology_repair_protocol"],
    }
    _atomic_bytes(
        model_out,
        pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    loaded = pickle.loads(Path(model_out).read_bytes())
    if (
        loaded.get("protocol") != MODEL_BUNDLE_PROTOCOL
        or set(loaded.get("component_models", {})) != set(source_groups)
    ):
        raise ValueError("source-component model bundle round trip failed")
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "post-v92-source-city-component-development",
        "city": city,
        "scenarios": list(scenarios),
        "source_city_groups": list(source_groups),
        "source_scenarios_by_group": {
            key: list(value) for key, value in source_scenarios_by_group.items()
        },
        "fixed_target_contract": {
            "main_model_sha256": str(expected_main_model_sha256),
            "main_result_sha256": str(expected_main_result_sha256),
            "selected_candidate": str(main_payload["selected_candidate"]),
            "selected_group_count": len(selected_group_ids),
            "prior_policy": prior_policy,
        },
        "source_cache_audit": source_cache_audit,
        "external_cache_audit": external_cache_audit,
        "component_diagnostics": component_diagnostics,
        "model_artifact": {
            "path": str(Path(model_out).resolve()),
            "sha256": _sha256(model_out),
            "protocol": MODEL_BUNDLE_PROTOCOL,
            "size_bytes": Path(model_out).stat().st_size,
        },
        "fit_workers": actual_fit_workers,
        "elapsed_sec": float(time.monotonic() - started),
    }
    _atomic_json(result_out, payload)
    gc.collect()
    return payload
