"""Cross-fitted B100 source and target predictions for stacked gate training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    TARGET_GROUP_BUDGET,
    _atomic_bytes,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _fit_family_v3,
    _group_adjusted_scores,
    counterfactual_cost_contract_v5,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    PROTOCOL as COMPONENT_FIT_PROTOCOL,
    _assert_waiting_aligned_bank,
    _fit_component_budget,
    _source_shard_contract,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _fit_result_contract,
    _target_cache_audit_contract,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig
from cf_h2o.traffic_signal.target_calibrated_source_gate import (
    GATE_BASE_FEATURES,
    balanced_group_folds,
)


RESULT_PROTOCOL = "tsc-v120-cross-fitted-b100-source-predictions-v1"
ARTIFACT_PROTOCOL = "cfcmt-cross-fitted-b100-source-predictions-v1"
TARGET_MODEL_FEATURES = (
    "target_model_score",
    "target_model_uncertainty",
)
STACKED_GATE_BASE_FEATURES = (*GATE_BASE_FEATURES, *TARGET_MODEL_FEATURES)
FOLD_COUNT = 5
_CROSSFIT_STATE: Mapping[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_inputs(
    *,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    cache_workers: int,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, tuple[str, ...]],
    Mapping[str, Any],
]:
    protocol = _read_json(fit_protocol_path)
    source_audit = _read_json(source_cache_audit_path)
    if (
        protocol.get("protocol") != COMPONENT_FIT_PROTOCOL
        or protocol.get("required_counterfactual_cost_contract")
        != counterfactual_cost_contract_v5("halted_queue")
        or source_audit.get("status") != "PASS"
        or not source_audit.get("gate", {}).get("passed", False)
    ):
        raise ValueError("source cache does not authorize V120 cross-fitting")
    manifest_full = load_traffic_signal_manifest(source_manifest_path)
    default_shards, shard_overrides = _source_shard_contract(
        protocol,
        source_audit,
        tuple(manifest_full.sumocfgs),
    )
    admitted = tuple(str(value) for value in source_audit["training_scenarios"])
    manifest = replace(
        manifest_full,
        scenarios=tuple(
            row for row in manifest_full.scenarios if row.scenario in set(admitted)
        ),
    )
    if (
        set(manifest.sumocfgs) != set(admitted)
        or len(set(manifest.city_groups.values()))
        != int(protocol["source_training"]["source_city_group_count"])
    ):
        raise ValueError("V120 admitted source scenarios changed")
    bank, bank_audit = load_frozen_counterfactual_bank(
        source_cache_root,
        manifest,
        seeds=tuple(int(value) for value in protocol["source_training"]["seeds"]),
        collection_shards=default_shards,
        scenario_collection_shards={
            name: count for name, count in shard_overrides.items() if name in admitted
        },
        workers=max(int(cache_workers), 1),
    )
    _assert_waiting_aligned_bank(
        bank,
        target_name=str(protocol["target_name"]),
        tolerance=float(protocol["required_target_equivalence_tolerance"]),
    )
    scenarios_by_group = {
        group: tuple(
            name for name, value in manifest.city_groups.items() if value == group
        )
        for group in sorted(set(manifest.city_groups.values()))
    }
    return protocol, bank, scenarios_by_group, bank_audit


def _adaptation_inputs(
    *,
    cache_root: Path,
    manifest_path: Path,
    cache_workers: int,
    selected_group_ids: Sequence[str],
) -> tuple[Any, np.ndarray, dict[str, Any], tuple[str, ...]]:
    manifest = load_traffic_signal_manifest(manifest_path)
    scenarios = tuple(
        name for name, group in manifest.city_groups.items() if group == "jinan"
    )
    if not scenarios:
        raise ValueError("Jinan target scenarios are absent from the manifest")
    bank, audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(cache_workers), 1),
    )
    full = _merge_city_datasets(bank, scenarios, city="jinan")
    selected = _group_subset_v3(
        full,
        selected_groups=set(str(value) for value in selected_group_ids),
        metadata_updates={"target_data_role": "v120_crossfit_b100"},
    )
    _assert_pure_waiting_selector_dataset(
        selected, target_name="prefix_mean_cost_450s"
    )
    groups = np.asarray(action_group_ids(selected), dtype=str)
    if set(np.unique(groups).tolist()) != set(selected_group_ids):
        raise ValueError("V120 B100 group identity changed")
    target_context = np.mean(
        np.vstack([_static_context_from_dataset(bank[name]) for name in scenarios]),
        axis=0,
    )
    return selected, target_context, audit, scenarios


def _target_fold_model(
    *,
    adaptation_dataset: Any,
    train_groups: Sequence[str],
    candidate: str,
) -> FrozenAnchoredBlendModel:
    train_dataset = _group_subset_v3(
        adaptation_dataset,
        selected_groups=set(str(value) for value in train_groups),
        metadata_updates={"target_data_role": "v120_crossfit_training_fold"},
    )
    contrast = build_action_contrast_dataset(
        train_dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    models = {}
    for family in BASE_FAMILIES:
        model, _ = _fit_family_v3(
            contrast,
            family,
            MechanismFitConfig(),
            target_domain="jinan",
        )
        models[family] = model
    return FrozenAnchoredBlendModel(
        anchor_model=models[ANCHOR_FAMILY],
        correction_model=models[CORRECTION_FAMILY],
        candidate=candidate,
        anchor_objective_mode="control_only",
        correction_objective_mode="control_only",
    )


def _crossfit_worker(task: tuple[str, int, str | None]) -> dict[str, Any]:
    if _CROSSFIT_STATE is None:
        raise RuntimeError("V120 cross-fit state is missing")
    from threadpoolctl import threadpool_limits

    arm, fold, source_group = task
    state = _CROSSFIT_STATE
    row_folds = np.asarray(state["row_folds"], dtype=int)
    heldout_rows = np.flatnonzero(row_folds == int(fold))
    train_groups = tuple(
        sorted(
            set(
                np.asarray(state["row_groups"], dtype=str)[row_folds != int(fold)]
            )
        )
    )
    if len(train_groups) != 80 or heldout_rows.size == 0:
        raise ValueError("V120 fold information budget changed")
    started = time.monotonic()
    with threadpool_limits(limits=1):
        if arm == "target_only":
            if source_group is not None:
                raise ValueError("target-only cross-fit task has a source group")
            model = _target_fold_model(
                adaptation_dataset=state["adaptation_dataset"],
                train_groups=train_groups,
                candidate=str(state["candidate"]),
            )
            diagnostics = {"source_transition_rows_consumed": 0}
        elif arm == "source_aligned":
            if source_group not in state["source_states"]:
                raise ValueError("unknown source group in V120 cross-fit task")
            fitted = _fit_component_budget(
                state["source_states"][str(source_group)],
                target_group_ids=train_groups,
            )
            expected_domains = {str(source_group), "jinan"}
            if set(fitted.diagnostics["source_domains"]) != expected_domains:
                raise ValueError("V120 source fold crossed its information boundary")
            model = FrozenAnchoredBlendModel(
                anchor_model=fitted.family_models[ANCHOR_FAMILY],
                correction_model=fitted.family_models[CORRECTION_FAMILY],
                candidate=str(state["candidate"]),
                anchor_objective_mode=fitted.objective_modes[ANCHOR_FAMILY],
                correction_objective_mode=fitted.objective_modes[CORRECTION_FAMILY],
            )
            diagnostics = {
                "source_domains": sorted(expected_domains),
                "target_adaptation_groups": int(
                    fitted.diagnostics["target_adaptation_groups"]
                ),
            }
        else:
            raise ValueError(f"unknown V120 cross-fit arm: {arm}")
        contrast = state["adaptation_contrast"]
        score, uncertainty, trust, _ = _group_adjusted_scores(
            contrast,
            model.predict(contrast),
            objective_mode="control_only",
        )
    return {
        "arm": arm,
        "fold": int(fold),
        "source_group": source_group,
        "heldout_rows": heldout_rows,
        "score": np.asarray(score[heldout_rows], dtype=float),
        "uncertainty": np.asarray(uncertainty[heldout_rows], dtype=float),
        "trust": np.asarray(trust[heldout_rows], dtype=float),
        "training_group_count": len(train_groups),
        "heldout_group_count": int(
            np.unique(np.asarray(state["row_groups"], dtype=str)[heldout_rows]).size
        ),
        "diagnostics": diagnostics,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def assemble_crossfit_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_count: int,
    source_groups: Sequence[str],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    target_arrays = [np.full(row_count, np.nan, dtype=float) for _ in range(3)]
    source_arrays = {
        str(group): [np.full(row_count, np.nan, dtype=float) for _ in range(3)]
        for group in source_groups
    }
    occupied = {
        "target_only": np.zeros(row_count, dtype=bool),
        **{
            f"source_aligned:{group}": np.zeros(row_count, dtype=bool)
            for group in source_groups
        },
    }
    for row in rows:
        arm = str(row["arm"])
        source_group = row.get("source_group")
        key = "target_only" if arm == "target_only" else f"source_aligned:{source_group}"
        if key not in occupied:
            raise ValueError("V120 cross-fit result has an unknown arm")
        indices = np.asarray(row["heldout_rows"], dtype=int)
        if (
            indices.ndim != 1
            or indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= row_count)
            or np.any(occupied[key][indices])
        ):
            raise ValueError("V120 cross-fit rows overlap or are invalid")
        destination = target_arrays if arm == "target_only" else source_arrays[str(source_group)]
        for index, name in enumerate(("score", "uncertainty", "trust")):
            values = np.asarray(row[name], dtype=float)
            if values.shape != indices.shape or not np.all(np.isfinite(values)):
                raise ValueError("V120 cross-fit prediction values are invalid")
            destination[index][indices] = values
        occupied[key][indices] = True
    if any(not np.all(mask) for mask in occupied.values()):
        raise ValueError("V120 cross-fit prediction coverage is incomplete")
    return (
        tuple(target_arrays),
        {group: tuple(values) for group, values in source_arrays.items()},
    )


def run_crossfit(
    *,
    fit_result_path: Path,
    expected_fit_result_sha256: str,
    fit_protocol_path: Path,
    source_cache_root: Path,
    source_manifest_path: Path,
    source_cache_audit_path: Path,
    target_cache_root: Path,
    target_manifest_path: Path,
    target_cache_audit_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fit_workers: int,
    artifact_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if artifact_path.exists():
        raise FileExistsError(f"refusing to overwrite V120 artifact: {artifact_path}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    fit_result = _fit_result_contract(
        fit_result_path, expected_sha256=expected_fit_result_sha256
    )
    _target_cache_audit_contract(target_cache_audit_path, fit_result)
    expected_source_audit = fit_result.get("input_audits", {}).get("source", {})
    if (
        str(Path(source_cache_audit_path)) != str(expected_source_audit.get("path"))
        or _sha256(source_cache_audit_path) != str(expected_source_audit.get("sha256"))
    ):
        raise ValueError("source cache audit differs from V115")
    selected_group_ids = tuple(
        str(value)
        for value in fit_result["information_budget"]["selected_group_ids"]
    )
    adaptation, target_context, adaptation_audit, adaptation_scenarios = _adaptation_inputs(
        cache_root=target_cache_root,
        manifest_path=target_manifest_path,
        cache_workers=cache_workers,
        selected_group_ids=selected_group_ids,
    )
    protocol, source_bank, source_scenarios, source_bank_audit = _source_inputs(
        fit_protocol_path=fit_protocol_path,
        source_cache_root=source_cache_root,
        source_manifest_path=source_manifest_path,
        source_cache_audit_path=source_cache_audit_path,
        cache_workers=cache_workers,
    )
    candidate = str(fit_result["frozen_blend_candidate"])
    adaptation_contrast = build_action_contrast_dataset(
        adaptation,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    row_groups = np.asarray(action_group_ids(adaptation_contrast), dtype=str)
    row_folds = balanced_group_folds(row_groups, fold_count=FOLD_COUNT)
    if np.unique(row_groups).size != TARGET_GROUP_BUDGET:
        raise ValueError("V120 adaptation group count changed")
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    target_key = "waiting_aligned_target_jinan_v120"
    source_states = {}
    for source_group, scenarios in source_scenarios.items():
        fit_bank = {name: source_bank[name] for name in scenarios}
        fit_bank[target_key] = adaptation
        city_groups = {name: source_group for name in scenarios}
        city_groups[target_key] = "jinan"
        source_states[source_group] = {
            "fit_bank": fit_bank,
            "target_key": target_key,
            "city_groups": city_groups,
            "target_static_context": target_context,
            "prior_policy": "phase_pressure",
            "source_rule_specs": source_rule_specs,
            "city": "jinan",
        }
    tasks = [
        ("target_only", fold, None) for fold in range(FOLD_COUNT)
    ] + [
        ("source_aligned", fold, source_group)
        for fold in range(FOLD_COUNT)
        for source_group in sorted(source_states)
    ]
    actual_workers = min(max(int(fit_workers), 1), len(tasks))
    global _CROSSFIT_STATE
    _CROSSFIT_STATE = {
        "adaptation_dataset": adaptation,
        "adaptation_contrast": adaptation_contrast,
        "row_groups": row_groups,
        "row_folds": row_folds,
        "candidate": candidate,
        "source_states": source_states,
    }
    try:
        if actual_workers == 1:
            fitted_rows = [_crossfit_worker(task) for task in tasks]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("V120 parallel cross-fit requires Linux fork")
            with ProcessPoolExecutor(
                max_workers=actual_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                fitted_rows = list(pool.map(_crossfit_worker, tasks))
    finally:
        _CROSSFIT_STATE = None
    target_predictions, source_predictions = assemble_crossfit_predictions(
        fitted_rows,
        row_count=adaptation_contrast.size,
        source_groups=tuple(sorted(source_states)),
    )
    artifact = {
        "protocol": ARTIFACT_PROTOCOL,
        "city": "jinan",
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "target_group_budget": TARGET_GROUP_BUDGET,
        "fold_count": FOLD_COUNT,
        "row_group_ids": row_groups,
        "row_folds": row_folds,
        "selected_group_ids": selected_group_ids,
        "source_group_order": tuple(sorted(source_states)),
        "target_predictions": target_predictions,
        "source_predictions": source_predictions,
        "frozen_blend_candidate": candidate,
        "fit_result_sha256": expected_fit_result_sha256,
        "source_cache_audit_sha256": _sha256(source_cache_audit_path),
        "target_cache_audit_sha256": _sha256(target_cache_audit_path),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        artifact_path, pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    )
    round_trip = pickle.loads(artifact_path.read_bytes())
    if (
        round_trip.get("protocol") != ARTIFACT_PROTOCOL
        or set(round_trip.get("source_predictions", {})) != set(source_states)
        or len(round_trip.get("target_predictions", ())) != 3
    ):
        raise ValueError("V120 artifact round trip failed")
    fold_diagnostics = [
        {
            key: row[key]
            for key in (
                "arm",
                "fold",
                "source_group",
                "training_group_count",
                "heldout_group_count",
                "diagnostics",
                "elapsed_seconds",
            )
        }
        for row in fitted_rows
    ]
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "cross_fitted_training_artifact_not_selector_result",
        "city": "jinan",
        "estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "information_budget": {
            "total_target_groups": TARGET_GROUP_BUDGET,
            "training_target_groups_per_fold": 80,
            "heldout_target_groups_per_fold": 20,
            "fold_count": FOLD_COUNT,
            "source_city_group_count": len(source_states),
            "source_fold_fit_count": FOLD_COUNT * len(source_states),
            "target_only_fold_fit_count": FOLD_COUNT,
            "selector_target_labels_consumed": 0,
        },
        "crossfit": {
            "fit_workers": actual_workers,
            "task_count": len(tasks),
            "row_count": int(adaptation_contrast.size),
            "group_count": int(np.unique(row_groups).size),
            "complete_target_prediction_rows": int(
                np.count_nonzero(np.isfinite(target_predictions[0]))
            ),
            "complete_source_prediction_rows": {
                group: int(np.count_nonzero(np.isfinite(values[0])))
                for group, values in source_predictions.items()
            },
            "fold_diagnostics": fold_diagnostics,
        },
        "inputs": {
            "fit_result": {
                "path": str(Path(fit_result_path).resolve()),
                "sha256": expected_fit_result_sha256,
            },
            "fit_protocol": {
                "path": str(Path(fit_protocol_path).resolve()),
                "sha256": _sha256(fit_protocol_path),
                "protocol": protocol["protocol"],
            },
            "source_cache_audit": {
                "path": str(Path(source_cache_audit_path).resolve()),
                "sha256": _sha256(source_cache_audit_path),
            },
            "target_cache_audit": {
                "path": str(Path(target_cache_audit_path).resolve()),
                "sha256": _sha256(target_cache_audit_path),
            },
        },
        "data_audits": {
            "source_bank": source_bank_audit,
            "adaptation_bank": adaptation_audit,
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "protocol": ARTIFACT_PROTOCOL,
        },
        "runtime_seconds": float(time.monotonic() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--fit-result-sha256", required=True)
    parser.add_argument("--fit-protocol", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-audit", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-audit", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V120 cache workers must be in [1, 20]")
    if not 1 <= int(args.fit_workers) <= 20:
        raise ValueError("V120 fit workers must be in [1, 20]")
    result = run_crossfit(
        fit_result_path=args.fit_result,
        expected_fit_result_sha256=args.fit_result_sha256,
        fit_protocol_path=args.fit_protocol,
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_cache_audit_path=args.source_cache_audit,
        target_cache_root=args.target_cache_root,
        target_manifest_path=args.target_manifest,
        target_cache_audit_path=args.target_cache_audit,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
        artifact_path=args.artifact,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
