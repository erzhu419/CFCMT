"""Fit and freeze the v79 multihorizon state-conditioned action latents."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (
    RESULT_PROTOCOL as SCREEN_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_nested_selection_audit import (
    AUTHORIZATION_DECISION as NESTED_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as NESTED_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    SpatiotemporalActionLatentConfig,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_CONDITIONED_PROTOCOL,
    StateConditionedLatentConfig,
    StateConditionedSpatiotemporalActionLatent,
)
from cf_h2o.traffic_signal.waiting_aligned_action_originator import (
    WaitingAlignedActionOriginator,
    WaitingAlignedActionOriginatorConfig,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_latent_screen import (
    PROTOCOL as SCREEN_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_nested_selection import (
    PROTOCOL as NESTED_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (
    ADAPTATION_CLASSIFICATION,
    ARTIFACT_KEYS,
    ARTIFACT_ROLE,
    PRIMARY_ARTIFACT_KEY,
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


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-state-latent-freeze-result-v1"
ARTIFACT_PROTOCOL = "target-multihorizon-state-conditioned-latent-artifact-v1"
AUTHORIZATION_DECISION = "authorize_multihorizon_state_latent_artifact_audit"
REJECTION_DECISION = "reject_multihorizon_state_latent_artifacts"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _atomic_pickle(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _state_config(value: Mapping[str, Any]) -> StateConditionedLatentConfig:
    payload = dict(value)
    base = SpatiotemporalActionLatentConfig(**dict(payload.pop("base")))
    return StateConditionedLatentConfig(base=base, **payload)


def _prediction_digest(
    model: StateConditionedSpatiotemporalActionLatent,
    dataset: MechanismDataset,
) -> str:
    prediction = model.predict(dataset)["control_cost"]
    digest = hashlib.sha256()
    digest.update(str(model.target_name).encode("utf-8"))
    digest.update(b"\0")
    groups = action_group_ids(dataset).astype(str)
    for group in groups:
        digest.update(str(group).encode("utf-8"))
        digest.update(b"\0")
    for name in (
        "mean",
        "uncertainty",
        "context_trust",
        "context_distance",
        "latent_support_count",
        "latent_level",
    ):
        values = np.ascontiguousarray(prediction[name], dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def load_multihorizon_state_originator(
    artifact_root: Path,
    artifact_key: str,
) -> tuple[WaitingAlignedActionOriginator, dict[str, Any], dict[str, Any]]:
    key = str(artifact_key)
    if key not in ARTIFACT_KEYS:
        raise ValueError(f"unknown multihorizon artifact key: {key!r}")
    root = Path(artifact_root) / key
    model_path = root / "model.pkl"
    certificate_path = root / "freeze.json"
    certificate = _read_json(certificate_path)
    if (
        certificate.get("protocol") != RESULT_PROTOCOL
        or certificate.get("status") != "PASS"
        or certificate.get("artifact_key") != key
        or certificate.get("model_sha256") != _sha256(model_path)
    ):
        raise ValueError("multihorizon state-latent certificate changed")
    artifact = pickle.loads(model_path.read_bytes())
    if (
        artifact.get("protocol") != ARTIFACT_PROTOCOL
        or artifact.get("artifact_key") != key
        or artifact.get("artifact_role") != ARTIFACT_ROLE
        or artifact.get("adaptation_classification") != ADAPTATION_CLASSIFICATION
        or artifact.get("zero_shot") is not False
        or artifact.get("uses_future_outcomes_at_prediction") is not False
        or int(artifact.get("prediction_horizon_sec", -1)) != 60
        or str(artifact.get("target_name")) != "prefix_mean_cost_60s"
    ):
        raise ValueError("multihorizon state-latent artifact contract changed")
    originator = WaitingAlignedActionOriginator(
        model=artifact["model"],
        config=WaitingAlignedActionOriginatorConfig(
            **dict(artifact["originator_config"])
        ),
    )
    return originator, certificate, artifact


def fit_artifact_models(
    *,
    contrast: MechanismDataset,
    artifact_specs: Sequence[Mapping[str, Any]],
    city: str,
    scenario: str,
    training_seeds: Sequence[int],
    reference_policy: str,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite artifact root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, Any]] = {}
    for spec_value in artifact_specs:
        spec = dict(spec_value)
        key = str(spec["artifact_key"])
        if key not in ARTIFACT_KEYS or key in records:
            raise ValueError("multihorizon artifact keys changed")
        model = StateConditionedSpatiotemporalActionLatent(
            _state_config(spec["model_config"]),
            target_name=str(spec["target_name"]),
        )
        fit_diagnostics = model.fit(contrast)
        originator_config = WaitingAlignedActionOriginatorConfig(
            risk_multiplier=float(spec["risk_multiplier"]),
            minimum_context_trust=float(spec["minimum_context_trust"]),
            rollout_value_horizon_sec=int(spec["prediction_horizon_sec"]),
        )
        artifact = {
            "protocol": ARTIFACT_PROTOCOL,
            "artifact_key": key,
            "artifact_role": ARTIFACT_ROLE,
            "candidate_role": str(spec["role"]),
            "selection_eligible_closed_loop": bool(
                spec["selection_eligible_closed_loop"]
            ),
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
            "proposal_origin_changed": True,
            "city": str(city),
            "scenarios": (str(scenario),),
            "training_seeds": tuple(int(value) for value in training_seeds),
            "training_seed_count": len(training_seeds),
            "reference_policy": str(reference_policy),
            "target_name": str(spec["target_name"]),
            "prediction_horizon_sec": int(spec["prediction_horizon_sec"]),
            "model_key": str(spec["model_key"]),
            "model_family": str(spec["model_family"]),
            "model": model,
            "model_fit_diagnostics": fit_diagnostics,
            "originator_config": {
                "enabled": bool(originator_config.enabled),
                "risk_multiplier": float(originator_config.risk_multiplier),
                "minimum_context_trust": float(
                    originator_config.minimum_context_trust
                ),
                "minimum_benefit_probability": float(
                    originator_config.minimum_benefit_probability
                ),
                "benefit_probability_kind": str(
                    originator_config.benefit_probability_kind
                ),
                "rollout_value_horizon_sec": int(
                    originator_config.rollout_value_horizon_sec
                ),
                "objective_name": str(originator_config.objective_name),
            },
            "uses_target_labels_at_fit": True,
            "uses_future_outcomes_at_prediction": False,
        }
        artifact_dir = output_root / key
        artifact_dir.mkdir(parents=False, exist_ok=False)
        model_path = artifact_dir / "model.pkl"
        _atomic_pickle(model_path, artifact)
        model_sha256 = _sha256(model_path)
        prediction_sha256 = _prediction_digest(model, contrast)
        certificate = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "artifact_key": key,
            "candidate_role": str(spec["role"]),
            "selection_eligible_closed_loop": bool(
                spec["selection_eligible_closed_loop"]
            ),
            "model_sha256": model_sha256,
            "prediction_sha256": prediction_sha256,
            "target_name": str(spec["target_name"]),
            "prediction_horizon_sec": int(spec["prediction_horizon_sec"]),
            "training_seed_count": len(training_seeds),
            "training_row_count": int(contrast.size),
            "training_group_count": int(
                np.unique(action_group_ids(contrast).astype(str)).size
            ),
            "fit_diagnostics": fit_diagnostics,
            "originator_config": artifact["originator_config"],
        }
        certificate_path = artifact_dir / "freeze.json"
        atomic_write_json(certificate_path, certificate)
        loaded_originator, loaded_certificate, loaded_artifact = (
            load_multihorizon_state_originator(output_root, key)
        )
        records[key] = {
            "artifact_key": key,
            "candidate_role": str(spec["role"]),
            "selection_eligible_closed_loop": bool(
                spec["selection_eligible_closed_loop"]
            ),
            "model": {"path": str(model_path.resolve()), "sha256": model_sha256},
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
            "prediction_sha256": prediction_sha256,
            "reload_prediction_sha256": _prediction_digest(
                loaded_originator.model, contrast
            ),
            "reload_certificate_exact": loaded_certificate == certificate,
            "reload_model_key": str(loaded_artifact["model_key"]),
            "fit_diagnostics": fit_diagnostics,
        }
    if tuple(records) != ARTIFACT_KEYS:
        raise ValueError("multihorizon artifact fit order changed")
    return records


def run_freeze(
    *,
    artifact_protocol_path: Path,
    nested_protocol_path: Path,
    nested_result_path: Path,
    nested_deployment_oof_path: Path,
    screen_protocol_path: Path,
    screen_result_path: Path,
    screen_oof_path: Path,
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
    nested_protocol = _read_json(nested_protocol_path)
    nested_result = _read_json(nested_result_path)
    screen_protocol = _read_json(screen_protocol_path)
    screen_result = _read_json(screen_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest_document = _read_json(manifest_path)
    frozen = dict(artifact_protocol.get("frozen_inputs", {}))
    paths = {
        "nested_protocol": nested_protocol_path,
        "nested_result": nested_result_path,
        "nested_deployment_oof": nested_deployment_oof_path,
        "screen_protocol": screen_protocol_path,
        "screen_result": screen_result_path,
        "screen_oof": screen_oof_path,
        "cache_protocol": cache_protocol_path,
        "cache_audit": cache_audit_path,
        "partition": partition_path,
        "manifest": manifest_path,
    }
    evidence_ok = bool(
        artifact_protocol.get("protocol") == PROTOCOL
        and nested_protocol.get("protocol") == NESTED_PROTOCOL
        and nested_result.get("protocol") == NESTED_RESULT_PROTOCOL
        and nested_result.get("status") == "PASS"
        and nested_result.get("decision") == NESTED_AUTHORIZATION_DECISION
        and nested_result.get("integrity_gate", {}).get("passed") is True
        and screen_protocol.get("protocol") == SCREEN_PROTOCOL
        and screen_result.get("protocol") == SCREEN_RESULT_PROTOCOL
        and screen_result.get("status") == "PASS"
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and partition.get("protocol") == PARTITION_PROTOCOL
        and manifest_document.get("protocol") == MANIFEST_PROTOCOL
        and all(
            frozen.get(key, {}).get("sha256") == _sha256(path)
            for key, path in paths.items()
        )
    )
    if not evidence_ok:
        raise ValueError("multihorizon state-latent freeze evidence changed")
    training = dict(artifact_protocol["training"])
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
    if set(seeds).intersection(sealed) or any(path.exists() for path in sealed_roots):
        raise ValueError("multihorizon artifact fit touched sealed evidence")

    scenario = str(training["scenarios"][0])
    city = str(training["city"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("multihorizon artifact scenario changed")
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
    groups = action_group_ids(contrast).astype(str)
    target_name = str(artifact_protocol["artifact_contract"]["target_name"])
    observed_actions = np.unique(groups, return_counts=True)[1]
    dataset_ok = bool(
        int(contrast.size) == int(training["expected_rows"])
        and int(np.unique(groups).size) == int(training["expected_groups"])
        and target_name in contrast.targets
        and set(observed_actions.tolist())
        == {int(training["expected_action_count_per_group"])}
    )
    if not dataset_ok:
        raise ValueError("multihorizon artifact training dataset changed")

    artifact_records = fit_artifact_models(
        contrast=contrast,
        artifact_specs=artifact_protocol["artifacts"],
        city=city,
        scenario=scenario,
        training_seeds=seeds,
        reference_policy=str(training["reference_policy"]),
        output_root=output_root,
    )
    artifact_contract_ok = all(
        row["prediction_sha256"] == row["reload_prediction_sha256"]
        and row["reload_certificate_exact"] is True
        and row["fit_diagnostics"].get("protocol")
        == STATE_CONDITIONED_PROTOCOL
        and row["fit_diagnostics"].get("target_name") == target_name
        and row["fit_diagnostics"].get("calibration_mode")
        == "leave_self_out_within_cell"
        for row in artifact_records.values()
    )
    integrity = {
        "frozen_evidence_chain": evidence_ok,
        "development_seed_set_exact": len(seeds) == 22,
        "sealed_seed_overlap_absent": not bool(set(seeds).intersection(sealed)),
        "sealed_roots_absent": not any(path.exists() for path in sealed_roots),
        "training_dataset_exact": dataset_ok,
        "artifact_key_set_exact": tuple(artifact_records) == ARTIFACT_KEYS,
        "one_primary_one_diagnostic": bool(
            artifact_records[PRIMARY_ARTIFACT_KEY][
                "selection_eligible_closed_loop"
            ]
            and sum(
                bool(row["selection_eligible_closed_loop"])
                for row in artifact_records.values()
            )
            == 1
        ),
        "artifact_reload_prediction_exact": artifact_contract_ok,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": (
            AUTHORIZATION_DECISION if integrity["passed"] else REJECTION_DECISION
        ),
        "artifact_protocol_sha256": _sha256(artifact_protocol_path),
        "nested_result_sha256": _sha256(nested_result_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "bank_audit": bank_audit,
        "training_row_count": int(contrast.size),
        "training_group_count": int(np.unique(groups).size),
        "artifacts": artifact_records,
        "integrity_gate": integrity,
        "claim_boundary": artifact_protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--nested-protocol", type=Path, required=True)
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--nested-deployment-oof", type=Path, required=True)
    parser.add_argument("--screen-protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-oof", type=Path, required=True)
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
        raise FileExistsError(f"refusing to overwrite artifact result: {args.out}")
    result = run_freeze(
        artifact_protocol_path=args.artifact_protocol,
        nested_protocol_path=args.nested_protocol,
        nested_result_path=args.nested_result,
        nested_deployment_oof_path=args.nested_deployment_oof,
        screen_protocol_path=args.screen_protocol,
        screen_result_path=args.screen_result,
        screen_oof_path=args.screen_oof,
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
                "artifacts": list(result["artifacts"]),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
