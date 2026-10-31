"""Fit a deployable target-only action model with no source-row consumption."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _relabel_dataset_domain,
)
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    MODEL_PROTOCOL as MAIN_MODEL_PROTOCOL,
    PROTOCOL as MAIN_RESULT_PROTOCOL,
    validate_repaired_cache_contract,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.action_ranker import (
    TargetOnlyActionAdvantageRegressor,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


PROTOCOL = "tsc-v98-strict-target-only-fit-v1"
MODEL_PROTOCOL = "cfcmt-strict-target-only-action-model-v1"
TARGET_ONLY_FAMILY = "causal_target_only_v2"


def run_strict_target_only_fit(
    *,
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
    model_out: Path,
    result_out: Path,
) -> dict[str, Any]:
    """Fit the strict target-only comparator on the parent's complete target pool."""

    started = time.monotonic()
    if model_out.exists() or result_out.exists():
        raise FileExistsError("refusing to overwrite strict target-only fit")
    if _sha256(main_model_path) != str(expected_main_model_sha256):
        raise ValueError("main topology-repaired model identity changed")
    if _sha256(main_result_path) != str(expected_main_result_sha256):
        raise ValueError("main topology-repaired result identity changed")
    main_payload = pickle.loads(Path(main_model_path).read_bytes())
    main_result = json.loads(Path(main_result_path).read_text(encoding="utf-8"))
    if (
        main_payload.get("protocol") != MAIN_MODEL_PROTOCOL
        or main_payload.get("city") != city
        or main_result.get("protocol") != MAIN_RESULT_PROTOCOL
        or main_result.get("city") != city
        or main_result.get("model_artifact", {}).get("sha256")
        != str(expected_main_model_sha256)
    ):
        raise ValueError("main topology-repaired fit contract changed")

    manifest = load_traffic_signal_manifest(external_manifest_path)
    repair = json.loads(Path(external_repair_audit).read_text(encoding="utf-8"))
    validate_repaired_cache_contract(
        label="external",
        repair_audit=repair,
        manifest_scenarios=tuple(manifest.sumocfgs),
        seed_count=len(ADAPTATION_SEEDS),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
    )
    bank, cache_audit = load_frozen_counterfactual_bank(
        external_cache_root,
        manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    validate_repaired_cache_contract(
        label="external",
        repair_audit=repair,
        manifest_scenarios=tuple(manifest.sumocfgs),
        seed_count=len(ADAPTATION_SEEDS),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        cache_audit=cache_audit,
    )
    if main_result.get("external_cache_audit") != cache_audit:
        raise ValueError("strict target-only fit received a different target cache")

    scenarios = tuple(str(value) for value in scenarios_by_city[city])
    if tuple(main_payload.get("scenarios", ())) != scenarios:
        raise ValueError("main model target scenarios changed")
    if set(scenarios) != {
        name for name, group in manifest.city_groups.items() if group == city
    }:
        raise ValueError("external city/scenario mapping changed")
    selected_group_ids = tuple(
        str(value) for value in main_payload["selected_group_ids"]
    )
    full_city_dataset = _merge_city_datasets(bank, scenarios, city=city)
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "strict_target_only_complete_adaptation_pool"
        },
    )
    prior_spec = main_payload["model"].prior_spec
    target_dataset = _relabel_dataset_domain(selected_dataset, city)
    contrast = build_action_contrast_dataset(
        target_dataset,
        reference_policy=prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    domains = set(str(value) for value in np.unique(contrast.domains))
    observed_groups = set(
        str(value) for value in contrast.metadata["action_group_ids"]
    )
    if domains != {city} or observed_groups != set(selected_group_ids):
        raise ValueError("strict target-only training matrix is not target-exclusive")

    model = TargetOnlyActionAdvantageRegressor(target_domain=city)
    diagnostics = model.fit(contrast)
    if (
        int(diagnostics.get("source_row_count_consumed", -1)) != 0
        or int(diagnostics.get("target_group_count", -1))
        != len(selected_group_ids)
    ):
        raise ValueError("strict target-only estimator consumption audit failed")
    model_payload = {
        "protocol": MODEL_PROTOCOL,
        "city": city,
        "scenarios": scenarios,
        "family": TARGET_ONLY_FAMILY,
        "objective_mode": "control_only",
        "selected_group_ids": selected_group_ids,
        "prior_policy": str(prior_spec.key),
        "main_model_sha256": str(expected_main_model_sha256),
        "model": model,
        "fit_diagnostics": diagnostics,
    }
    _atomic_bytes(
        model_out,
        pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    loaded = pickle.loads(Path(model_out).read_bytes())
    if (
        loaded.get("protocol") != MODEL_PROTOCOL
        or loaded.get("city") != city
        or loaded.get("family") != TARGET_ONLY_FAMILY
        or int(
            loaded.get("fit_diagnostics", {}).get(
                "source_row_count_consumed", -1
            )
        )
        != 0
    ):
        raise ValueError("strict target-only model round trip failed")

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "strict-comparator-fit-before-closed-loop-evaluation",
        "city": city,
        "scenarios": list(scenarios),
        "family": TARGET_ONLY_FAMILY,
        "prior_policy": str(prior_spec.key),
        "selected_group_count": len(selected_group_ids),
        "selected_group_ids": list(selected_group_ids),
        "training_data_audit": {
            "domains": sorted(domains),
            "target_rows": int(contrast.size),
            "source_rows_consumed": 0,
            "target_group_count": len(observed_groups),
        },
        "fit_diagnostics": diagnostics,
        "external_cache_audit": cache_audit,
        "parent_model": {
            "path": str(Path(main_model_path).resolve()),
            "sha256": str(expected_main_model_sha256),
        },
        "model_artifact": {
            "path": str(Path(model_out).resolve()),
            "sha256": _sha256(model_out),
            "protocol": MODEL_PROTOCOL,
            "size_bytes": Path(model_out).stat().st_size,
        },
        "elapsed_sec": float(time.monotonic() - started),
    }
    _atomic_json(result_out, payload)
    gc.collect()
    return payload
