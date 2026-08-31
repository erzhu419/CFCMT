"""Seed-blocked OOF diagnostic for state-conditioned action latents."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (
    _argmin_with_reference_tie_break,
    _row_subset,
    select_ranker_gate,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_diagnostic import (
    AUTHORIZATION_DECISION as V70_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as V70_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_ranker import group_normalized_action_target
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.distributional_action_state_latent import (
    DistributionalActionStateConfig,
    DistributionalActionStateLatent,
)
from cf_h2o.traffic_signal.hierarchical_action_state_residual import (
    HierarchicalActionStateResidual,
    HierarchicalActionStateResidualConfig,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
    TargetSpatiotemporalActionLatent,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_CONDITIONED_PROTOCOL,
    StateConditionedLatentConfig,
    StateConditionedSpatiotemporalActionLatent,
)
from scripts.cluster.audit_tsc_external_waiting_aligned_latent_closed_loop import (
    AUDIT_PROTOCOL as V72_AUDIT_PROTOCOL,
    REJECTION_DECISION as V72_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (
    MANIFEST_PROTOCOL,
    PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_latent_closed_loop_development import (
    PROTOCOL as V72_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PARTITION_PROTOCOL,
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_diagnostic import (
    PROTOCOL as V70_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-state-conditioned-latent-result-v1"
AUTHORIZATION_DECISION = "authorize_state_conditioned_latent_closed_loop_development"
REJECTION_DECISION = "retain_phase_pressure_and_reject_state_conditioned_latent"
MODEL_FAMILY = "target_state_conditioned_spatiotemporal_latent"
BASE_LATENT_MODEL_FAMILY = "target_spatiotemporal_action_latent"
DISTRIBUTIONAL_MODEL_FAMILY = "distributional_action_state_latent"
HIERARCHICAL_RESIDUAL_MODEL_FAMILY = "hierarchical_action_state_residual"
_PROCESS_FOLD_CONTEXT: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _state_config(raw: Mapping[str, Any]) -> StateConditionedLatentConfig:
    values = dict(raw)
    base = SpatiotemporalActionLatentConfig(**dict(values.pop("base")))
    return StateConditionedLatentConfig(base=base, **values)


def _build_model(
    spec: Mapping[str, Any],
    *,
    target_name: str = "interval_cost",
) -> Any:
    family = str(spec.get("family"))
    raw = dict(spec["config"])
    if family == BASE_LATENT_MODEL_FAMILY:
        return TargetSpatiotemporalActionLatent(
            SpatiotemporalActionLatentConfig(**raw),
            target_name=target_name,
        )
    if family == MODEL_FAMILY:
        return StateConditionedSpatiotemporalActionLatent(
            _state_config(raw),
            target_name=target_name,
        )
    if target_name != "interval_cost":
        raise ValueError(
            f"model family {family!r} does not support target {target_name!r}"
        )
    if family == DISTRIBUTIONAL_MODEL_FAMILY:
        state_model = _state_config(dict(raw.pop("state_model")))
        return DistributionalActionStateLatent(
            DistributionalActionStateConfig(state_model=state_model, **raw)
        )
    if family == HIERARCHICAL_RESIDUAL_MODEL_FAMILY:
        base = SpatiotemporalActionLatentConfig(**dict(raw.pop("base")))
        return HierarchicalActionStateResidual(
            HierarchicalActionStateResidualConfig(base=base, **raw)
        )
    raise ValueError(f"unsupported state-conditioned model family: {family!r}")


def _fit_state_conditioned_oof_fold(
    *,
    heldout_seed: int,
    contrast: MechanismDataset,
    model_specs: Sequence[Mapping[str, Any]],
    scenario: str,
    expected_action_count: int,
    target_name: str = "interval_cost",
) -> dict[str, Any]:
    groups = action_group_ids(contrast).astype(str)
    seed_by_group = {
        str(group): parse_action_group_seed(str(group)) for group in np.unique(groups)
    }
    training_mask = np.asarray(
        [seed_by_group[str(group)] != int(heldout_seed) for group in groups],
        dtype=bool,
    )
    validation_mask = ~training_mask
    training = _row_subset(contrast, training_mask)
    validation = _row_subset(contrast, validation_mask)
    training_groups = {
        str(value) for value in training.metadata["action_group_ids"]
    }
    validation_groups = {
        str(value) for value in validation.metadata["action_group_ids"]
    }
    if (
        not training_groups
        or not validation_groups
        or training_groups.intersection(validation_groups)
        or any(
            parse_action_group_seed(group) == int(heldout_seed)
            for group in training_groups
        )
    ):
        raise RuntimeError("state-conditioned OOF seed partition leaked")

    validation_group_ids = action_group_ids(validation).astype(str)
    is_reference = np.asarray(validation.metadata["is_reference"], dtype=bool)
    candidate_states = np.asarray(
        validation.metadata["candidate_states"], dtype=str
    )
    actual_raw = np.asarray(validation.targets[target_name], dtype=float)
    actual_normalized, _ = group_normalized_action_target(
        validation, target_name
    )
    model_results = []
    for spec in model_specs:
        model = _build_model(spec, target_name=target_name)
        fit = model.fit(training)
        prediction = model.predict(validation)["control_cost"]
        score = np.asarray(prediction["mean"], dtype=float)
        uncertainty = np.asarray(prediction["uncertainty"], dtype=float)
        trust = np.asarray(prediction["context_trust"], dtype=float)
        distance = np.asarray(prediction["context_distance"], dtype=float)
        level = np.asarray(prediction["latent_level"], dtype=float)
        support = np.asarray(prediction["latent_support_count"], dtype=float)
        benefit_raw = np.asarray(
            prediction.get("benefit_probability_raw", np.ones(validation.size)),
            dtype=float,
        )
        benefit_calibrated = np.asarray(
            prediction.get(
                "benefit_probability_calibrated", np.ones(validation.size)
            ),
            dtype=float,
        )
        benefit_support = np.asarray(
            prediction.get("benefit_effective_support", np.zeros(validation.size)),
            dtype=float,
        )
        if not all(
            np.isfinite(values).all()
            for values in (
                score,
                uncertainty,
                trust,
                distance,
                level,
                support,
                benefit_raw,
                benefit_calibrated,
                benefit_support,
            )
        ):
            raise ValueError("state-conditioned latent emitted non-finite predictions")
        records = []
        for group in sorted(validation_groups):
            rows = np.flatnonzero(validation_group_ids == group)
            references = rows[is_reference[rows]]
            if rows.size != int(expected_action_count) or references.size != 1:
                raise ValueError(
                    f"state-conditioned action group {group!r} violates the action contract"
                )
            reference = int(references[0])
            selected = _argmin_with_reference_tie_break(
                rows,
                score,
                reference=reference,
                candidate_states=candidate_states,
            )
            oracle = _argmin_with_reference_tie_break(
                rows,
                actual_raw,
                reference=reference,
                candidate_states=candidate_states,
            )
            records.append(
                {
                    "model_key": str(spec["key"]),
                    "model_family": str(spec["family"]),
                    "target_name": str(target_name),
                    "heldout_seed": int(heldout_seed),
                    "simulator_seed": int(heldout_seed),
                    "scenario": str(scenario),
                    "group_id": str(group),
                    "action_count": int(rows.size),
                    "selected_candidate_state": str(candidate_states[selected]),
                    "reference_candidate_state": str(candidate_states[reference]),
                    "oracle_candidate_state": str(candidate_states[oracle]),
                    "selected_differs": bool(selected != reference),
                    "selected_is_oracle": bool(selected == oracle),
                    "predicted_delta": float(score[selected]),
                    "selected_uncertainty": float(uncertainty[selected]),
                    "selected_context_trust": float(trust[selected]),
                    "selected_context_distance": float(distance[selected]),
                    "selected_benefit_probability_raw": float(
                        benefit_raw[selected]
                    ),
                    "selected_benefit_probability_calibrated": float(
                        benefit_calibrated[selected]
                    ),
                    "selected_benefit_effective_support": float(
                        benefit_support[selected]
                    ),
                    "selected_latent_level": int(level[selected]),
                    "selected_support_count": int(support[selected]),
                    "actual_group_normalized_delta": float(
                        actual_normalized[selected] - actual_normalized[reference]
                    ),
                    "actual_raw_delta": float(
                        actual_raw[selected] - actual_raw[reference]
                    ),
                    "oracle_group_normalized_delta": float(
                        actual_normalized[oracle] - actual_normalized[reference]
                    ),
                    "normalized_regret_to_oracle": float(
                        actual_normalized[selected] - actual_normalized[oracle]
                    ),
                }
            )
        model_results.append(
            {
                "model_key": str(spec["key"]),
                "model_family": str(spec["family"]),
                "fit_diagnostics": fit,
                "records": records,
            }
        )
    return {
        "heldout_seed": int(heldout_seed),
        "training_group_count": len(training_groups),
        "validation_group_count": len(validation_groups),
        "training_validation_disjoint": not bool(
            training_groups.intersection(validation_groups)
        ),
        "heldout_absent_from_training": all(
            parse_action_group_seed(group) != int(heldout_seed)
            for group in training_groups
        ),
        "model_results": model_results,
    }


def _process_fold(heldout_seed: int) -> dict[str, Any]:
    if _PROCESS_FOLD_CONTEXT is None:
        raise RuntimeError("state-conditioned process context is not initialized")
    return _fit_state_conditioned_oof_fold(
        heldout_seed=int(heldout_seed),
        contrast=_PROCESS_FOLD_CONTEXT["contrast"],
        model_specs=_PROCESS_FOLD_CONTEXT["model_specs"],
        scenario=_PROCESS_FOLD_CONTEXT["scenario"],
        expected_action_count=_PROCESS_FOLD_CONTEXT["expected_action_count"],
        target_name=_PROCESS_FOLD_CONTEXT.get("target_name", "interval_cost"),
    )


def run_diagnostic(
    *,
    diagnostic_protocol_path: Path,
    v70_protocol_path: Path,
    v70_result_path: Path,
    v72_protocol_path: Path,
    v72_audit_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    partition_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    cache_workers: int,
    fold_workers: int,
) -> dict[str, Any]:
    protocol = _read_json(diagnostic_protocol_path)
    v70_protocol = _read_json(v70_protocol_path)
    v70_result = _read_json(v70_result_path)
    v72_protocol = _read_json(v72_protocol_path)
    v72_audit = _read_json(v72_audit_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest_document = _read_json(manifest_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    evidence_ok = (
        protocol.get("protocol") == PROTOCOL
        and v70_protocol.get("protocol") == V70_PROTOCOL
        and v70_result.get("protocol") == V70_RESULT_PROTOCOL
        and v70_result.get("status") == "PASS"
        and v70_result.get("decision") == V70_AUTHORIZATION_DECISION
        and v70_result.get("diagnostic_protocol_sha256")
        == _sha256(v70_protocol_path)
        and v72_protocol.get("protocol") == V72_PROTOCOL
        and v72_audit.get("protocol") == V72_AUDIT_PROTOCOL
        and v72_audit.get("status") == "PASS"
        and v72_audit.get("decision") == V72_REJECTION_DECISION
        and v72_audit.get("integrity_gate", {}).get("passed") is True
        and v72_audit.get("protocol_sha256") == _sha256(v72_protocol_path)
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("gate", {}).get("passed") is True
        and partition.get("protocol") == PARTITION_PROTOCOL
        and manifest_document.get("protocol") == MANIFEST_PROTOCOL
        and _sha256(v70_protocol_path) == frozen.get("v70_protocol_sha256")
        and _sha256(v70_result_path) == frozen.get("v70_result_sha256")
        and _sha256(v72_protocol_path) == frozen.get("v72_protocol_sha256")
        and _sha256(v72_audit_path) == frozen.get("v72_audit_sha256")
        and _sha256(cache_protocol_path) == frozen.get("cache_protocol_sha256")
        and _sha256(cache_audit_path) == frozen.get("cache_audit_sha256")
        and _sha256(partition_path) == frozen.get("partition_sha256")
        and _sha256(manifest_path) == frozen.get("manifest_sha256")
    )
    if not evidence_ok:
        raise ValueError("state-conditioned latent evidence changed")

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
        raise ValueError("state-conditioned development partition changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("state-conditioned scenario is not unique")
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
    if observed_seeds != set(seeds):
        raise ValueError("state-conditioned contrast seed set changed")

    model_specs = tuple(training["model_candidates"])
    if (
        len(model_specs) != 8
        or len({str(spec["key"]) for spec in model_specs}) != len(model_specs)
        or {str(spec["family"]) for spec in model_specs} != {MODEL_FAMILY}
    ):
        raise ValueError("state-conditioned model grid changed")
    workers = min(max(int(fold_workers), 1), 8, len(seeds))
    global _PROCESS_FOLD_CONTEXT
    _PROCESS_FOLD_CONTEXT = {
        "contrast": contrast,
        "model_specs": model_specs,
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
        _PROCESS_FOLD_CONTEXT = None

    oof_records = [
        row
        for fold in folds
        for model_result in fold["model_results"]
        for row in model_result["records"]
    ]
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
    expected_groups = int(cache_audit["total_groups"])
    expected_records = expected_groups * len(model_specs)
    unique_keys = {
        (str(row["model_key"]), str(row["group_id"])) for row in oof_records
    }
    fit_contract = all(
        result["fit_diagnostics"].get("protocol") == STATE_CONDITIONED_PROTOCOL
        and result["fit_diagnostics"].get("calibration_mode")
        == "leave_self_out_within_cell"
        and result["fit_diagnostics"].get("uses_target_labels_at_fit") is True
        and result["fit_diagnostics"].get("uses_future_outcomes_at_prediction")
        is False
        for fold in folds
        for result in fold["model_results"]
    )
    sealed_roots_absent = not any(path.exists() for path in sealed_roots)
    integrity = {
        "frozen_evidence_chain": evidence_ok,
        "development_seed_set_exact": observed_seeds == set(seeds),
        "sealed_seed_overlap_absent": not bool(set(seeds).intersection(sealed)),
        "sealed_roots_absent": sealed_roots_absent,
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
        "state_conditioned_fit_contract_exact": fit_contract,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    selection = select_ranker_gate(
        oof_records,
        model_specs=model_specs,
        seeds=seeds,
        selection=protocol["selection"],
    )
    advance = bool(integrity["passed"] and selection["selected"] is not None)
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": AUTHORIZATION_DECISION if advance else REJECTION_DECISION,
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "parent_v70_result_sha256": _sha256(v70_result_path),
        "parent_v72_audit_sha256": _sha256(v72_audit_path),
        "bank_audit": bank_audit,
        "dataset_rows": int(contrast.size),
        "action_group_count": expected_groups,
        "model_count": len(model_specs),
        "fold_workers": workers,
        "folds": fold_summaries,
        "oof_record_count": len(oof_records),
        "oof_records": oof_records,
        "offline_comparators": dict(v70_result.get("offline_comparators", {})),
        "parent_closed_loop_failure": {
            "decision": v72_audit["decision"],
            "candidate_summaries": v72_audit["candidate_summaries"],
        },
        "gate_selection": selection,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": advance,
            "selected_candidate": selection["selected"],
        },
        "adaptive_status": protocol["adaptive_status"],
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--v70-protocol", type=Path, required=True)
    parser.add_argument("--v70-result", type=Path, required=True)
    parser.add_argument("--v72-protocol", type=Path, required=True)
    parser.add_argument("--v72-audit", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            f"refusing to overwrite state-conditioned result: {args.out}"
        )
    result = run_diagnostic(
        diagnostic_protocol_path=args.diagnostic_protocol,
        v70_protocol_path=args.v70_protocol,
        v70_result_path=args.v70_result,
        v72_protocol_path=args.v72_protocol,
        v72_audit_path=args.v72_audit,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        partition_path=args.partition,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
    )
    atomic_write_json(args.out, result)
    selected = result["advance_gate"]["selected_candidate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "groups": result["action_group_count"],
                "models": result["model_count"],
                "selected": selected.get("key") if selected else None,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
