"""Fit and freeze the selected waiting-aligned spatiotemporal action latent."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_diagnostic import (
    AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as DIAGNOSTIC_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
    TargetSpatiotemporalActionLatent,
)
from cf_h2o.traffic_signal.waiting_aligned_action_originator import (
    WaitingAlignedActionOriginator,
    WaitingAlignedActionOriginatorConfig,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_artifact import (
    ADAPTATION_CLASSIFICATION,
    ARTIFACT_ROLE,
    PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_diagnostic import (
    PROTOCOL as DIAGNOSTIC_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-waiting-aligned-latent-freeze-result-v1"
ARTIFACT_PROTOCOL = "waiting-aligned-target-spatiotemporal-latent-artifact-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_waiting_aligned_originator(
    artifact_root: Path,
) -> tuple[WaitingAlignedActionOriginator, dict[str, Any]]:
    root = Path(artifact_root)
    model_path = root / "model.pkl"
    certificate_path = root / "freeze.json"
    certificate = _read_json(certificate_path)
    if (
        certificate.get("protocol") != RESULT_PROTOCOL
        or certificate.get("status") != "PASS"
        or certificate.get("model_sha256") != _sha256(model_path)
    ):
        raise ValueError("waiting-aligned latent artifact certificate changed")
    artifact = pickle.loads(model_path.read_bytes())
    if (
        artifact.get("protocol") != ARTIFACT_PROTOCOL
        or artifact.get("artifact_role") != ARTIFACT_ROLE
        or artifact.get("adaptation_classification") != ADAPTATION_CLASSIFICATION
        or artifact.get("zero_shot") is not False
    ):
        raise ValueError("waiting-aligned latent artifact contract changed")
    originator = WaitingAlignedActionOriginator(
        model=artifact["model"],
        config=WaitingAlignedActionOriginatorConfig(
            **dict(artifact["originator_config"])
        ),
    )
    return originator, certificate


def run_freeze(
    *,
    artifact_protocol_path: Path,
    diagnostic_protocol_path: Path,
    diagnostic_result_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    partition_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    artifact_protocol = _read_json(artifact_protocol_path)
    diagnostic_protocol = _read_json(diagnostic_protocol_path)
    diagnostic_result = _read_json(diagnostic_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    frozen = dict(artifact_protocol.get("frozen_inputs", {}))
    evidence_ok = (
        artifact_protocol.get("protocol") == PROTOCOL
        and diagnostic_protocol.get("protocol") == DIAGNOSTIC_PROTOCOL
        and diagnostic_result.get("protocol") == DIAGNOSTIC_RESULT_PROTOCOL
        and diagnostic_result.get("status") == "PASS"
        and diagnostic_result.get("decision") == AUTHORIZATION_DECISION
        and diagnostic_result.get("integrity_gate", {}).get("passed") is True
        and diagnostic_result.get("advance_gate", {}).get("passed") is True
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("gate", {}).get("passed") is True
        and _sha256(diagnostic_protocol_path)
        == frozen.get("diagnostic_protocol_sha256")
        and _sha256(diagnostic_result_path) == frozen.get("diagnostic_result_sha256")
        and _sha256(cache_protocol_path) == frozen.get("cache_protocol_sha256")
        and _sha256(cache_audit_path) == frozen.get("cache_audit_sha256")
        and _sha256(partition_path) == frozen.get("partition_sha256")
        and _sha256(manifest_path) == frozen.get("manifest_sha256")
    )
    if not evidence_ok:
        raise ValueError("waiting-aligned latent freeze evidence changed")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite latent artifact: {output_root}")

    training = dict(artifact_protocol["training"])
    seeds = tuple(int(value) for value in training["seeds"])
    sealed = {
        int(value)
        for value in (
            list(training["sealed_confirmatory_seeds"])
            + list(training["sealed_prospective_seeds"])
        )
    }
    if set(seeds).intersection(sealed):
        raise ValueError("latent artifact training overlaps sealed seeds")
    scenario = str(training["scenarios"][0])
    city = str(training["city"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("latent artifact scenario changed")
    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(1, int(workers)),
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=city)
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=str(training["reference_policy"]),
        contrast_features=CONTRAST_FEATURES_V3,
    )
    selected = dict(artifact_protocol["selected"])
    model_spec = dict(selected["model_spec"])
    model = TargetSpatiotemporalActionLatent(
        SpatiotemporalActionLatentConfig(**dict(model_spec["config"]))
    )
    fit_diagnostics = model.fit(contrast)
    originator_config = WaitingAlignedActionOriginatorConfig(
        risk_multiplier=float(selected["risk_multiplier"]),
        minimum_context_trust=float(selected["minimum_context_trust"]),
        rollout_value_horizon_sec=int(
            artifact_protocol["artifact_contract"]["rollout_value_horizon_sec"]
        ),
    )
    artifact = {
        "protocol": ARTIFACT_PROTOCOL,
        "artifact_role": ARTIFACT_ROLE,
        "adaptation_classification": ADAPTATION_CLASSIFICATION,
        "zero_shot": False,
        "proposal_origin_changed": True,
        "city": city,
        "scenarios": (scenario,),
        "training_seeds": seeds,
        "training_seed_count": len(seeds),
        "reference_policy": str(training["reference_policy"]),
        "counterfactual_target": str(training["target"]),
        "counterfactual_cost_mode": "halted_queue",
        "model_key": str(selected["model_key"]),
        "model": model,
        "model_fit_diagnostics": fit_diagnostics,
        "originator_config": {
            "enabled": bool(originator_config.enabled),
            "risk_multiplier": float(originator_config.risk_multiplier),
            "minimum_context_trust": float(
                originator_config.minimum_context_trust
            ),
            "rollout_value_horizon_sec": int(
                originator_config.rollout_value_horizon_sec
            ),
            "objective_name": str(originator_config.objective_name),
        },
        "uses_target_labels_at_fit": True,
        "uses_future_outcomes_at_prediction": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    model_path = output_root / "model.pkl"
    model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    loaded = pickle.loads(model_path.read_bytes())
    loaded_originator = WaitingAlignedActionOriginator(
        model=loaded["model"],
        config=WaitingAlignedActionOriginatorConfig(
            **dict(loaded["originator_config"])
        ),
    )
    model_sha256 = _sha256(model_path)
    groups = action_group_ids(contrast).astype(str)
    expected_rows = int(cache_audit["total_rows"])
    expected_groups = int(cache_audit["total_groups"])
    integrity = {
        "frozen_evidence_chain": evidence_ok,
        "development_seed_set_exact": len(seeds) == 22,
        "sealed_seed_overlap_absent": not bool(set(seeds).intersection(sealed)),
        "training_rows_exact": int(contrast.size) == expected_rows,
        "training_groups_exact": int(len(set(groups.tolist()))) == expected_groups,
        "selected_model_exact": str(selected["model_key"])
        == "tls_time_300s_s20",
        "selected_gate_exact": float(originator_config.risk_multiplier) == 0.0
        and float(originator_config.minimum_context_trust) == 0.75,
        "artifact_reload_exact": loaded_originator.diagnostics()["config"]
        == artifact["originator_config"],
        "adaptation_classification_exact": loaded["adaptation_classification"]
        == ADAPTATION_CLASSIFICATION
        and loaded["zero_shot"] is False,
        "future_outcomes_absent_at_prediction": loaded[
            "uses_future_outcomes_at_prediction"
        ]
        is False,
        "proposal_origin_changed": loaded["proposal_origin_changed"] is True,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    certificate = {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "model_sha256": model_sha256,
        "artifact_protocol_sha256": _sha256(artifact_protocol_path),
        "city": city,
        "scenarios": [scenario],
        "artifact_role": ARTIFACT_ROLE,
        "adaptation_classification": ADAPTATION_CLASSIFICATION,
        "zero_shot": False,
        "training_seed_count": len(seeds),
        "training_row_count": int(contrast.size),
        "training_group_count": int(len(set(groups.tolist()))),
        "selected_candidate": selected,
        "fit_diagnostics": fit_diagnostics,
        "originator_config": artifact["originator_config"],
        "integrity_gate": integrity,
    }
    certificate_path = output_root / "freeze.json"
    atomic_write_json(certificate_path, certificate)
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": certificate["status"],
        "decision": (
            "authorize_waiting_aligned_latent_closed_loop_redevelopment"
            if integrity["passed"]
            else "reject_waiting_aligned_latent_artifact"
        ),
        "artifact_protocol_sha256": _sha256(artifact_protocol_path),
        "diagnostic_result_sha256": _sha256(diagnostic_result_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "bank_audit": bank_audit,
        "artifact": {
            "root": str(output_root.resolve()),
            "model": {"path": str(model_path.resolve()), "sha256": model_sha256},
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
        },
        "integrity_gate": integrity,
        "claim_boundary": artifact_protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite latent freeze result: {args.out}")
    result = run_freeze(
        artifact_protocol_path=args.artifact_protocol,
        diagnostic_protocol_path=args.diagnostic_protocol,
        diagnostic_result_path=args.diagnostic_result,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        partition_path=args.partition,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "artifact": result["artifact"]["root"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
