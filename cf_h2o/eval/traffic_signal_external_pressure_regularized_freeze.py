"""Freeze an OOF-selected pressure-regularized direct controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    EXTERNAL_COLLECTION_SHARDS,
    _atomic_bytes,
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    ALIGNED_AUDIT_DECISION,
    ALIGNED_AUDIT_PROTOCOL,
    ALIGNED_PROTOCOL,
    V9_ALIGNED_AUDIT_DECISION,
    V9_ALIGNED_AUDIT_PROTOCOL,
    V9_ALIGNED_PROTOCOL,
    _load_aligned_artifacts,
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (
    ANCHOR_POLICY,
    MODEL_PROTOCOL as PARENT_MODEL_PROTOCOL,
    RESULT_PROTOCOL as PARENT_RESULT_PROTOCOL,
    V9_MODEL_PROTOCOL as V9_PARENT_MODEL_PROTOCOL,
    V9_RESULT_PROTOCOL as V9_PARENT_RESULT_PROTOCOL,
    _group_seeds,
    _target_support,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    PRESSURE_REGULARIZER_SELECTION_PROTOCOL,
    select_oof_pressure_regularizer,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import PriorRegularizationConfig
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


RESULT_PROTOCOL = "tsc-v47r43-external-oof-pressure-regularized-freeze-v1"
MODEL_PROTOCOL = "cfcmt-external-oof-pressure-regularized-direct-model-v1"
V9_RESULT_PROTOCOL = "tsc-v56r52-external-v9-oof-pressure-regularized-freeze-v1"
V9_MODEL_PROTOCOL = "cfcmt-external-v9-oof-pressure-regularized-direct-model-v1"
DEPLOYMENT_POLICY = "cfcmt_oof_pressure_regularized_direct_mpc"


def _write_model(
    path: Path,
    payload: Mapping[str, Any],
    *,
    city: str,
    model_protocol: str = MODEL_PROTOCOL,
    parent_model_protocol: str = PARENT_MODEL_PROTOCOL,
) -> dict[str, Any]:
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite pressure-regularized model: {path}")
    _atomic_bytes(path, pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL))
    loaded = pickle.loads(Path(path).read_bytes())
    if (
        loaded.get("protocol") != model_protocol
        or loaded.get("city") != city
        or loaded.get("anchor_policy") != ANCHOR_POLICY
        or loaded.get("parent_model_protocol") != parent_model_protocol
        or not isinstance(loaded.get("models"), OfflineScreeningModels)
        or not isinstance(loaded.get("target_support"), TargetActionSupport)
        or not isinstance(loaded.get("regularizer"), PriorRegularizationConfig)
    ):
        raise ValueError("pressure-regularized model round-trip failed")
    return {
        "protocol": model_protocol,
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(Path(path).stat().st_size),
        "round_trip_passed": True,
    }


def run_pressure_regularized_freeze(
    *,
    aligned_protocol_path: Path,
    aligned_joint_audit_path: Path,
    expected_aligned_joint_audit_sha256: str,
    aligned_freeze_root: Path,
    external_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    expected_external_cache_sha256: str,
    expected_source_tree_sha256: str | None,
    city: str,
    workers: int,
    model_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _read_json(aligned_protocol_path)
    joint = _read_json(aligned_joint_audit_path)
    joint_sha = _sha256(aligned_joint_audit_path)
    aligned_protocol = str(protocol.get("protocol", ""))
    if aligned_protocol == ALIGNED_PROTOCOL:
        expected_audit_protocol = ALIGNED_AUDIT_PROTOCOL
        expected_audit_decision = ALIGNED_AUDIT_DECISION
        parent_result_protocol = PARENT_RESULT_PROTOCOL
        parent_model_protocol = PARENT_MODEL_PROTOCOL
        result_protocol = RESULT_PROTOCOL
        model_protocol = MODEL_PROTOCOL
    elif aligned_protocol == V9_ALIGNED_PROTOCOL:
        expected_audit_protocol = V9_ALIGNED_AUDIT_PROTOCOL
        expected_audit_decision = V9_ALIGNED_AUDIT_DECISION
        parent_result_protocol = V9_PARENT_RESULT_PROTOCOL
        parent_model_protocol = V9_PARENT_MODEL_PROTOCOL
        result_protocol = V9_RESULT_PROTOCOL
        model_protocol = V9_MODEL_PROTOCOL
    else:
        raise ValueError("pressure-regularized aligned protocol changed")
    if (
        joint_sha != str(expected_aligned_joint_audit_sha256)
        or joint_sha != str(protocol["estimand_aligned_joint_audit"]["sha256"])
        or joint.get("protocol") != expected_audit_protocol
        or joint.get("status") != "PASS"
        or joint.get("decision") != expected_audit_decision
    ):
        raise ValueError("pressure-regularized freeze lacks a valid aligned parent audit")
    if city not in protocol["city_artifacts"]:
        raise ValueError(f"unknown pressure-regularized city: {city}")
    parent_certificate, parent_model = _load_aligned_artifacts(
        protocol=protocol,
        joint=joint,
        aligned_freeze_root=aligned_freeze_root,
        city=city,
    )
    if (
        parent_certificate.get("protocol") != parent_result_protocol
        or parent_model.get("protocol") != parent_model_protocol
    ):
        raise ValueError("pressure-regularized aligned parent artifact changed")

    runtime = _runtime_metadata()
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != str(
        expected_source_tree_sha256
    ):
        raise ValueError("pressure-regularized source-tree SHA-256 mismatch")
    cache_sha, cache_files = aggregate_cache_sha256(external_cache_root)
    city_scenarios = {
        str(name): tuple(str(value) for value in values)
        for name, values in protocol["prospective_confirmation_reservation"][
            "city_scenarios"
        ].items()
    }
    expected_files = (
        sum(len(values) for values in city_scenarios.values())
        * len(ADAPTATION_SEEDS)
        * EXTERNAL_COLLECTION_SHARDS
    )
    if (
        cache_sha != str(expected_external_cache_sha256)
        or cache_sha != str(parent_certificate["external_cache_aggregate_sha256"])
        or cache_files != expected_files
    ):
        raise ValueError("pressure-regularized external cache identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        external_cache_root,
        manifest,
        seeds=ADAPTATION_SEEDS,
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    scenarios = city_scenarios[city]
    full_city_dataset = _merge_city_datasets(bank, scenarios, city=city)
    selected_group_ids = tuple(str(value) for value in parent_model["selected_group_ids"])
    selected_dataset = _group_subset_v3(
        full_city_dataset,
        selected_groups=set(selected_group_ids),
        metadata_updates={
            "target_data_role": "pressure_regularized_adaptation_coverage"
        },
    )
    models: OfflineScreeningModels = parent_model["models"]
    adaptation_support = _target_support(
        selected_dataset, prior_spec=models.prior_spec
    )
    regularizer, regularizer_selection = select_oof_pressure_regularizer(
        parent_certificate["oof_action_records"]
    )
    regularizer_payload = {
        "enabled": regularizer.enabled,
        "blend_weight": regularizer.blend_weight,
        "risk_multiplier": regularizer.risk_multiplier,
        "min_context_trust": regularizer.min_context_trust,
    }
    parent_certificate_sha = str(
        protocol["city_artifacts"][city]["certificate"]["sha256"]
    )
    parent_model_sha = str(protocol["city_artifacts"][city]["model"]["sha256"])
    model_payload = {
        "protocol": model_protocol,
        "city": city,
        "scenarios": scenarios,
        "anchor_policy": ANCHOR_POLICY,
        "selected_candidate": str(parent_model["selected_candidate"]),
        "regularizer": regularizer,
        "regularizer_selection_protocol": PRESSURE_REGULARIZER_SELECTION_PROTOCOL,
        "models": models,
        "target_support": adaptation_support,
        "selected_group_ids": selected_group_ids,
        "fold_assignment": dict(parent_model["fold_assignment"]),
        "parent_model_protocol": parent_model_protocol,
        "parent_certificate_sha256": parent_certificate_sha,
        "parent_model_sha256": parent_model_sha,
    }
    model_artifact = _write_model(
        model_out,
        model_payload,
        city=city,
        model_protocol=model_protocol,
        parent_model_protocol=parent_model_protocol,
    )
    deployment_policy = DEPLOYMENT_POLICY if regularizer.enabled else ANCHOR_POLICY
    selected_group_seed_set = _group_seeds(selected_group_ids)
    adaptation_seed_set = set(ADAPTATION_SEEDS)
    development_spec = dict(protocol.get("development", {}))
    adaptation_diagnostics = tuple(
        int(value)
        for value in development_spec.get(
            "adaptation_coupled_diagnostic_seeds", ()
        )
    )
    generalization_development = tuple(
        int(value)
        for value in development_spec.get("generalization_selection_seeds", ())
    )
    validation_seeds = tuple(
        int(value)
        for value in protocol.get("validation_reservation", {}).get(
            "closed_loop_seeds", ()
        )
    )
    prospective_seeds = tuple(
        int(value)
        for value in protocol["prospective_confirmation_reservation"][
            "closed_loop_seeds"
        ]
    )
    freeze_gate = {
        "aligned_parent_artifacts_verified": True,
        "external_cache_hash_verified": True,
        "all_adaptation_groups_reused_without_refit": set(selected_group_ids)
        == set(parent_certificate["selected_group_ids"]),
        "seed_blocked_oof_selection_reused": bool(
            parent_certificate["freeze_gate"]["seed_blocked_oof_predictions"]
        ),
        "adaptation_support_is_label_free": bool(adaptation_support.label_free),
        "pressure_anchor_preserved": str(models.prior_spec.key) == ANCHOR_POLICY,
        "model_artifact_round_trip": bool(model_artifact["round_trip_passed"]),
        "selected_groups_use_adaptation_seeds_only": selected_group_seed_set
        == adaptation_seed_set,
        "adaptation_diagnostic_overlap_declared": set(adaptation_diagnostics)
        == adaptation_seed_set,
        "generalization_validation_and_prospective_seeds_absent": not (
            selected_group_seed_set
            & set(
                (
                    *generalization_development,
                    *validation_seeds,
                    *prospective_seeds,
                )
            )
        ),
    }
    freeze_gate["passed"] = all(freeze_gate.values())
    development_seeds = protocol.get("development", {}).get(
        "closed_loop_seeds",
        protocol.get("development", {}).get("closed_loop_development_seeds", ()),
    )
    return {
        "protocol": result_protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "claim_boundary": (
            "the pressure regularizer and coverage support use only adaptation OOF "
            "records and label-free adaptation features; no development or prospective "
            "closed-loop outcome selects this artifact"
        ),
        "city": city,
        "scenarios": list(scenarios),
        "aligned_protocol_sha256": _sha256(aligned_protocol_path),
        "aligned_joint_audit_sha256": joint_sha,
        "parent_certificate_sha256": parent_certificate_sha,
        "parent_model_sha256": parent_model_sha,
        "external_cache_aggregate_sha256": cache_sha,
        "external_cache_file_count": cache_files,
        "external_cache_audit": cache_audit,
        "selected_group_ids": list(selected_group_ids),
        "selected_candidate": str(parent_model["selected_candidate"]),
        "anchor_policy": ANCHOR_POLICY,
        "regularizer_selection": regularizer_selection,
        "deployment": {
            "policy": deployment_policy,
            "coordination_mode": "direct" if regularizer.enabled else "phase_pressure",
            "regularizer": regularizer_payload,
            "target_support": adaptation_support.diagnostics(),
            "target_support_role": (
                "label-free adaptation coverage only; OOF labels select regularizer "
                "hyperparameters but do not define support prototypes"
            ),
            "operational_minimum_total_vehicles": 1.0,
        },
        "model_artifact": model_artifact,
        "excluded_seed_roles": {
            "closed_loop_development": list(development_seeds),
            "closed_loop_adaptation_diagnostics": list(adaptation_diagnostics),
            "closed_loop_generalization_development": list(
                generalization_development
            ),
            "closed_loop_validation": list(validation_seeds),
            "prospective_confirmation": list(prospective_seeds),
        },
        "freeze_gate": freeze_gate,
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-protocol", type=Path, required=True)
    parser.add_argument("--aligned-joint-audit", type=Path, required=True)
    parser.add_argument("--expected-aligned-joint-audit-sha256", required=True)
    parser.add_argument("--aligned-freeze-root", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-cache-sha256", required=True)
    parser.add_argument("--expected-source-tree-sha256")
    parser.add_argument("--city", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.model_out.exists():
        raise FileExistsError("refusing to overwrite pressure-regularized freeze evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_pressure_regularized_freeze(
        aligned_protocol_path=args.aligned_protocol,
        aligned_joint_audit_path=args.aligned_joint_audit,
        expected_aligned_joint_audit_sha256=args.expected_aligned_joint_audit_sha256,
        aligned_freeze_root=args.aligned_freeze_root,
        external_cache_root=args.external_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        expected_external_cache_sha256=args.expected_external_cache_sha256,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        city=args.city,
        workers=args.workers,
        model_out=args.model_out,
    )
    _atomic_json(args.out, payload)
    print(
        {
            "city": args.city,
            "policy": payload["deployment"]["policy"],
            "model_sha256": payload["model_artifact"]["sha256"],
            "out": str(args.out),
        },
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
