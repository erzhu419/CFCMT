"""Run the v77 seed-blocked multihorizon latent development screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_multihorizon_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    rollout_prefix_target_name,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    BASE_LATENT_MODEL_FAMILY,
    MODEL_FAMILY as STATE_MODEL_FAMILY,
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
from cf_h2o.traffic_signal.spatiotemporal_action_latent import LATENT_PROTOCOL
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_CONDITIONED_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_latent_screen import (
    PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (
    MANIFEST_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PARTITION_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-latent-screen-result-v1"
AUTHORIZATION_DECISION = "authorize_nested_multihorizon_latent_selection"
REJECTION_DECISION = "retain_phase_pressure_and_reject_multihorizon_latent_screen"
_PROCESS_CONTEXT: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _expanded_specs(training: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = []
    for horizon in training["rollout_prefix_horizons_sec"]:
        target_name = rollout_prefix_target_name(int(horizon))
        for architecture in training["model_candidates"]:
            values.append(
                {
                    **dict(architecture),
                    "key": f"{architecture['key']}::h{int(horizon)}s",
                    "architecture_key": str(architecture["key"]),
                    "horizon_sec": int(horizon),
                    "target_name": target_name,
                }
            )
    return tuple(values)


def _fit_fold(
    *,
    heldout_seed: int,
    contrast: MechanismDataset,
    expanded_specs: Sequence[Mapping[str, Any]],
    scenario: str,
    expected_action_count: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for spec in expanded_specs:
        grouped.setdefault(str(spec["target_name"]), []).append(spec)
    results = []
    training_group_count = None
    validation_group_count = None
    training_validation_disjoint = True
    heldout_absent_from_training = True
    for target_name, specs in grouped.items():
        fold = _fit_state_conditioned_oof_fold(
            heldout_seed=int(heldout_seed),
            contrast=contrast,
            model_specs=specs,
            scenario=scenario,
            expected_action_count=int(expected_action_count),
            target_name=target_name,
        )
        if training_group_count is None:
            training_group_count = int(fold["training_group_count"])
            validation_group_count = int(fold["validation_group_count"])
        elif (
            int(fold["training_group_count"]) != training_group_count
            or int(fold["validation_group_count"]) != validation_group_count
        ):
            raise RuntimeError("multihorizon folds disagree on row partition")
        results.extend(fold["model_results"])
        training_validation_disjoint = bool(
            training_validation_disjoint
            and fold["training_validation_disjoint"]
        )
        heldout_absent_from_training = bool(
            heldout_absent_from_training
            and fold["heldout_absent_from_training"]
        )
    return {
        "heldout_seed": int(heldout_seed),
        "training_group_count": int(training_group_count or 0),
        "validation_group_count": int(validation_group_count or 0),
        "training_validation_disjoint": training_validation_disjoint,
        "heldout_absent_from_training": heldout_absent_from_training,
        "model_results": results,
    }


def _process_fold(heldout_seed: int) -> dict[str, Any]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("multihorizon process context is not initialized")
    return _fit_fold(
        heldout_seed=int(heldout_seed),
        contrast=_PROCESS_CONTEXT["contrast"],
        expanded_specs=_PROCESS_CONTEXT["expanded_specs"],
        scenario=_PROCESS_CONTEXT["scenario"],
        expected_action_count=_PROCESS_CONTEXT["expected_action_count"],
    )


def _atomic_oof_npz(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_columns = (
        "model_key",
        "model_family",
        "target_name",
        "scenario",
        "group_id",
        "selected_candidate_state",
        "reference_candidate_state",
        "oracle_candidate_state",
    )
    int_columns = (
        "heldout_seed",
        "simulator_seed",
        "action_count",
        "selected_latent_level",
        "selected_support_count",
    )
    bool_columns = (
        "selected_differs",
        "selected_is_oracle",
    )
    float_columns = (
        "predicted_delta",
        "selected_uncertainty",
        "selected_context_trust",
        "selected_context_distance",
        "actual_group_normalized_delta",
        "actual_raw_delta",
        "oracle_group_normalized_delta",
        "normalized_regret_to_oracle",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in string_columns:
        arrays[name] = np.asarray([str(row[name]) for row in records], dtype=str)
    for name in int_columns:
        arrays[name] = np.asarray([int(row[name]) for row in records], dtype=np.int64)
    for name in bool_columns:
        arrays[name] = np.asarray([bool(row[name]) for row in records], dtype=bool)
    for name in float_columns:
        arrays[name] = np.asarray([float(row[name]) for row in records], dtype=float)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_screen(
    *,
    diagnostic_protocol_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    partition_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fold_workers: int,
    oof_out: Path,
) -> dict[str, Any]:
    protocol = _read_json(diagnostic_protocol_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest_document = _read_json(manifest_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    evidence_ok = bool(
        protocol.get("protocol") == PROTOCOL
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("decision")
        == "authorize_multihorizon_nested_seed_oof_development"
        and cache_audit.get("gate", {}).get("passed") is True
        and partition.get("protocol") == PARTITION_PROTOCOL
        and manifest_document.get("protocol") == MANIFEST_PROTOCOL
        and _sha256(cache_protocol_path) == frozen.get("cache_protocol_sha256")
        and _sha256(cache_audit_path) == frozen.get("cache_audit_sha256")
        and _sha256(partition_path) == frozen.get("partition_sha256")
        and _sha256(manifest_path) == frozen.get("manifest_sha256")
    )
    if not evidence_ok:
        raise ValueError("multihorizon screen evidence changed")
    training = dict(protocol["training"])
    scenario = str(training["scenarios"][0])
    seeds = tuple(int(value) for value in training["seeds"])
    sealed = {
        int(value)
        for value in (
            list(training["sealed_confirmatory_seeds"])
            + list(training["sealed_prospective_seeds"])
        )
    }
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if (
        set(seeds) != {int(value) for value in partition["redevelopment"]["seeds"]}
        or set(seeds).intersection(sealed)
        or any(path.exists() for path in sealed_roots)
    ):
        raise ValueError("multihorizon screen partition changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("multihorizon screen scenario is not unique")
    from dataclasses import replace

    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(int(cache_workers), 1),
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=str(training["city"]))
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=str(training["reference_policy"]),
        contrast_features=CONTRAST_FEATURES_V3,
    )
    groups = action_group_ids(contrast).astype(str)
    observed_seeds = {parse_action_group_seed(group) for group in np.unique(groups)}
    expanded_specs = _expanded_specs(training)
    expected_targets = {
        rollout_prefix_target_name(int(value))
        for value in training["rollout_prefix_horizons_sec"]
    }
    if (
        observed_seeds != set(seeds)
        or len(expanded_specs) != int(training["expanded_candidate_count"])
        or not expected_targets.issubset(contrast.targets)
        or len({str(spec["key"]) for spec in expanded_specs}) != len(expanded_specs)
    ):
        raise ValueError("multihorizon screen candidate or data contract changed")
    workers = min(max(int(fold_workers), 1), 8, len(seeds))
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = {
        "contrast": contrast,
        "expanded_specs": expanded_specs,
        "scenario": scenario,
        "expected_action_count": int(training["expected_action_count_per_group"]),
    }
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            folds = list(pool.map(_process_fold, seeds))
    finally:
        _PROCESS_CONTEXT = None
    oof_records = [
        row
        for fold in folds
        for model_result in fold["model_results"]
        for row in model_result["records"]
    ]
    expected_groups = int(cache_audit["total_groups"])
    expected_records = expected_groups * len(expanded_specs)
    unique_keys = {
        (str(row["model_key"]), str(row["group_id"])) for row in oof_records
    }
    fit_contract = True
    for fold in folds:
        for result in fold["model_results"]:
            diagnostics = result["fit_diagnostics"]
            family = str(result["model_family"])
            target_name = str(diagnostics.get("target_name", ""))
            if family == BASE_LATENT_MODEL_FAMILY:
                valid = diagnostics.get("protocol") == LATENT_PROTOCOL
            elif family == STATE_MODEL_FAMILY:
                valid = bool(
                    diagnostics.get("protocol") == STATE_CONDITIONED_PROTOCOL
                    and diagnostics.get("calibration_mode")
                    == "leave_self_out_within_cell"
                )
            else:
                valid = False
            fit_contract = fit_contract and valid and target_name in expected_targets
    integrity = {
        "frozen_evidence_chain": evidence_ok,
        "development_seed_set_exact": observed_seeds == set(seeds),
        "sealed_seed_overlap_absent": not bool(set(seeds).intersection(sealed)),
        "sealed_roots_absent": not any(path.exists() for path in sealed_roots),
        "fold_count_exact": len(folds) == len(seeds),
        "fold_seed_partition_disjoint": all(
            bool(fold["training_validation_disjoint"])
            and bool(fold["heldout_absent_from_training"])
            for fold in folds
        ),
        "validation_groups_exact": sum(
            int(fold["validation_group_count"]) for fold in folds
        )
        == expected_groups,
        "oof_record_count_exact": len(oof_records) == expected_records,
        "oof_group_model_keys_unique": len(unique_keys) == expected_records,
        "expected_actions_per_group": {
            int(row["action_count"]) for row in oof_records
        }
        == {int(training["expected_action_count_per_group"])},
        "target_specific_fit_contract_exact": fit_contract,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    selection = select_ranker_gate(
        oof_records,
        model_specs=expanded_specs,
        seeds=seeds,
        selection=protocol["selection"],
    )
    horizon_screens = {}
    for horizon in training["rollout_prefix_horizons_sec"]:
        specs = [
            spec for spec in expanded_specs if int(spec["horizon_sec"]) == int(horizon)
        ]
        keys = {str(spec["key"]) for spec in specs}
        horizon_screens[str(int(horizon))] = select_ranker_gate(
            [row for row in oof_records if str(row["model_key"]) in keys],
            model_specs=specs,
            seeds=seeds,
            selection=protocol["selection"],
        )
    _atomic_oof_npz(oof_out, oof_records)
    advance = bool(integrity["passed"] and selection["selected"] is not None)
    fold_summaries = [
        {
            key: value for key, value in fold.items() if key != "model_results"
        }
        | {
            "model_results": [
                {
                    "model_key": result["model_key"],
                    "model_family": result["model_family"],
                    "fit_diagnostics": result["fit_diagnostics"],
                    "validation_record_count": len(result["records"]),
                }
                for result in fold["model_results"]
            ]
        }
        for fold in folds
    ]
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": AUTHORIZATION_DECISION if advance else REJECTION_DECISION,
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "bank_audit": bank_audit,
        "dataset_rows": int(contrast.size),
        "action_group_count": expected_groups,
        "architecture_count": len(training["model_candidates"]),
        "horizon_count": len(training["rollout_prefix_horizons_sec"]),
        "expanded_candidate_count": len(expanded_specs),
        "fold_workers": workers,
        "folds": fold_summaries,
        "oof_artifact": {
            "path": str(Path(oof_out).resolve()),
            "sha256": _file_sha256(oof_out),
            "record_count": len(oof_records),
            "format": "compressed_columnar_npz_no_pickle",
        },
        "gate_selection": selection,
        "horizon_screens": horizon_screens,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": advance,
            "selected_candidate": selection["selected"],
            "next_stage": (
                "nested_leave_one_seed_out_joint_selection"
                if advance
                else "none"
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--oof-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.oof_out.exists():
        raise FileExistsError("refusing to overwrite multihorizon screen outputs")
    result = run_screen(
        diagnostic_protocol_path=args.diagnostic_protocol,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        partition_path=args.partition,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
        oof_out=args.oof_out,
    )
    atomic_write_json(args.out, result)
    selected = result["advance_gate"]["selected_candidate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "groups": result["action_group_count"],
                "candidates": result["expanded_candidate_count"],
                "selected": selected.get("key") if selected else None,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
