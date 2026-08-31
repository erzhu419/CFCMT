"""Fit and freeze the v82 uncertainty-aware h60/h120 originators."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (
    _atomic_pickle,
    _prediction_digest,
    _state_config,
    load_multihorizon_state_originator,
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
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    StateConditionedSpatiotemporalActionLatent,
)
from cf_h2o.traffic_signal.waiting_aligned_action_originator import (
    WaitingAlignedActionOriginator,
    WaitingAlignedActionOriginatorConfig,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_artifacts import (
    ADAPTATION_CLASSIFICATION,
    ARTIFACT_KEYS,
    ARTIFACT_ROLE,
    MINIMUM_CONTEXT_TRUST,
    PROTOCOL,
    RISK_MULTIPLIER,
)


RESULT_PROTOCOL = "tsc-v83r79-uncertainty-horizon-artifact-freeze-result-v1"
ARTIFACT_PROTOCOL = "target-uncertainty-horizon-originator-artifact-v1"
AUTHORIZATION_DECISION = "authorize_uncertainty_horizon_artifact_audit"
REJECTION_DECISION = "reject_uncertainty_horizon_artifacts"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_uncertainty_horizon_originator(
    artifact_root: Path, artifact_key: str
) -> tuple[WaitingAlignedActionOriginator, dict[str, Any], dict[str, Any]]:
    key = str(artifact_key)
    if key not in ARTIFACT_KEYS:
        raise ValueError(f"unknown uncertainty-horizon artifact: {key!r}")
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
        raise ValueError("uncertainty-horizon certificate changed")
    artifact = pickle.loads(model_path.read_bytes())
    if not (
        artifact.get("protocol") == ARTIFACT_PROTOCOL
        and artifact.get("artifact_key") == key
        and artifact.get("artifact_role") == ARTIFACT_ROLE
        and artifact.get("adaptation_classification")
        == ADAPTATION_CLASSIFICATION
        and artifact.get("zero_shot") is False
        and artifact.get("uses_future_outcomes_at_prediction") is False
        and int(artifact.get("prediction_horizon_sec", -1)) in {60, 120}
        and str(artifact.get("target_name"))
        == f"prefix_mean_cost_{int(artifact['prediction_horizon_sec'])}s"
    ):
        raise ValueError("uncertainty-horizon artifact contract changed")
    config = WaitingAlignedActionOriginatorConfig(
        **dict(artifact["originator_config"])
    )
    if not (
        float(config.risk_multiplier) == RISK_MULTIPLIER
        and float(config.minimum_context_trust) == MINIMUM_CONTEXT_TRUST
        and int(config.rollout_value_horizon_sec)
        == int(artifact["prediction_horizon_sec"])
    ):
        raise ValueError("uncertainty-horizon originator gate changed")
    return (
        WaitingAlignedActionOriginator(model=artifact["model"], config=config),
        certificate,
        artifact,
    )


def _originator_config(spec: Mapping[str, Any]) -> WaitingAlignedActionOriginatorConfig:
    return WaitingAlignedActionOriginatorConfig(
        risk_multiplier=float(spec["risk_multiplier"]),
        minimum_context_trust=float(spec["minimum_context_trust"]),
        rollout_value_horizon_sec=int(spec["prediction_horizon_sec"]),
    )


def fit_artifacts(
    *,
    contrast: MechanismDataset,
    artifact_specs: Sequence[Mapping[str, Any]],
    v79_artifact_root: Path,
    city: str,
    scenario: str,
    training_seeds: Sequence[int],
    reference_policy: str,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite v82 artifact root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, Any]] = {}
    group_count = int(np.unique(action_group_ids(contrast).astype(str)).size)
    for spec_value in artifact_specs:
        spec = dict(spec_value)
        key = str(spec["artifact_key"])
        if key not in ARTIFACT_KEYS or key in records:
            raise ValueError("v82 artifact key order changed")
        source = dict(spec["model_source"])
        if source["kind"] == "reuse_v79_frozen_model":
            _, _, source_artifact = load_multihorizon_state_originator(
                v79_artifact_root, str(source["artifact_key"])
            )
            model = source_artifact["model"]
            if (
                str(source_artifact["model_key"]) != str(spec["model_key"])
                or str(source_artifact["target_name"]) != str(spec["target_name"])
            ):
                raise ValueError(f"v82 reused model identity changed for {key}")
            fit_diagnostics = dict(source_artifact["model_fit_diagnostics"])
            fit_mode = "reused_v79_audited_model"
        elif source["kind"] == "fit_full_development_cache":
            model = StateConditionedSpatiotemporalActionLatent(
                _state_config(spec["model_config"]),
                target_name=str(spec["target_name"]),
            )
            fit_diagnostics = model.fit(contrast)
            fit_mode = "fit_full_v82_development_cache"
        else:
            raise ValueError(f"unknown v82 model source: {source['kind']!r}")
        originator_config = _originator_config(spec)
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
            "city": str(city),
            "scenarios": (str(scenario),),
            "training_seeds": tuple(int(value) for value in training_seeds),
            "training_seed_count": len(training_seeds),
            "reference_policy": str(reference_policy),
            "target_name": str(spec["target_name"]),
            "prediction_horizon_sec": int(spec["prediction_horizon_sec"]),
            "model_key": str(spec["model_key"]),
            "model_family": str(spec["model_family"]),
            "model_source": source,
            "fit_mode": fit_mode,
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
        model_sha = _sha256(model_path)
        prediction_sha = _prediction_digest(model, contrast)
        certificate = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "artifact_key": key,
            "candidate_role": str(spec["role"]),
            "selection_eligible_closed_loop": bool(
                spec["selection_eligible_closed_loop"]
            ),
            "model_sha256": model_sha,
            "prediction_sha256": prediction_sha,
            "target_name": str(spec["target_name"]),
            "prediction_horizon_sec": int(spec["prediction_horizon_sec"]),
            "training_seed_count": len(training_seeds),
            "training_row_count": int(contrast.size),
            "training_group_count": group_count,
            "fit_mode": fit_mode,
            "fit_diagnostics": fit_diagnostics,
            "originator_config": artifact["originator_config"],
        }
        certificate_path = artifact_dir / "freeze.json"
        atomic_write_json(certificate_path, certificate)
        loaded, loaded_certificate, loaded_artifact = (
            load_uncertainty_horizon_originator(output_root, key)
        )
        records[key] = {
            "artifact_key": key,
            "fit_mode": fit_mode,
            "model": {"path": str(model_path.resolve()), "sha256": model_sha},
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
            "prediction_sha256": prediction_sha,
            "reload_prediction_sha256": _prediction_digest(
                loaded.model, contrast
            ),
            "reload_certificate_exact": loaded_certificate == certificate,
            "reload_model_key": str(loaded_artifact["model_key"]),
            "fit_diagnostics": fit_diagnostics,
        }
    if tuple(records) != ARTIFACT_KEYS:
        raise ValueError("v82 artifact fit order changed")
    return records


def run_freeze(
    *,
    artifact_protocol_path: Path,
    frozen_input_paths: Mapping[str, Path],
    v79_artifact_root: Path,
    cache_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(artifact_protocol_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    if protocol.get("protocol") != PROTOCOL or any(
        _sha256(path) != frozen[key]["sha256"]
        for key, path in frozen_input_paths.items()
    ):
        raise ValueError("v82 artifact freeze evidence changed")
    partition = _read_json(frozen_input_paths["partition"])
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if any(path.exists() for path in sealed_roots):
        raise ValueError("v82 artifact freeze found unsealed holdout roots")
    for spec in protocol["artifacts"]:
        source = dict(spec["model_source"])
        if source["kind"] == "reuse_v79_frozen_model":
            key = str(source["artifact_key"])
            if (
                _sha256(Path(v79_artifact_root) / key / "model.pkl")
                != source["model"]["sha256"]
                or _sha256(Path(v79_artifact_root) / key / "freeze.json")
                != source["certificate"]["sha256"]
            ):
                raise ValueError(f"v82 reused source {key!r} changed")

    training = dict(protocol["training"])
    scenario = str(training["scenarios"][0])
    seeds = tuple(int(value) for value in training["seeds"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected) != 1:
        raise ValueError("v82 artifact scenario is not unique")
    manifest = replace(full_manifest, scenarios=selected)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(int(workers), 1),
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=str(training["city"]))
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=str(training["reference_policy"]),
        contrast_features=CONTRAST_FEATURES_V3,
    )
    group_count = int(np.unique(action_group_ids(contrast).astype(str)).size)
    artifacts = fit_artifacts(
        contrast=contrast,
        artifact_specs=protocol["artifacts"],
        v79_artifact_root=v79_artifact_root,
        city=str(training["city"]),
        scenario=scenario,
        training_seeds=seeds,
        reference_policy=str(training["reference_policy"]),
        output_root=output_root,
    )
    integrity = {
        "artifact_protocol_exact": True,
        "frozen_inputs_exact": True,
        "seed_count_exact": len(seeds) == 22,
        "row_count_exact": int(contrast.size) == int(training["expected_rows"]),
        "group_count_exact": group_count == int(training["expected_groups"]),
        "bank_rows_exact": int(bank_audit["rows_by_scenario"][scenario])
        == int(training["expected_rows"]),
        "artifact_keys_exact": tuple(artifacts) == ARTIFACT_KEYS,
        "reload_predictions_exact": all(
            row["prediction_sha256"] == row["reload_prediction_sha256"]
            for row in artifacts.values()
        ),
        "reload_certificates_exact": all(
            row["reload_certificate_exact"] for row in artifacts.values()
        ),
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
        "training_row_count": int(contrast.size),
        "training_group_count": group_count,
        "training_seed_count": len(seeds),
        "artifacts": artifacts,
        "integrity_gate": integrity,
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    for key in (
        "v81-protocol",
        "v81-audit",
        "v79-protocol",
        "v79-result",
        "v79-audit",
        "screen-protocol",
        "screen-result",
        "screen-oof",
        "cache-protocol",
        "cache-audit",
        "partition",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--v79-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v82 freeze result: {args.out}")
    frozen_inputs = {
        "v81_protocol": args.v81_protocol,
        "v81_audit": args.v81_audit,
        "v79_protocol": args.v79_protocol,
        "v79_result": args.v79_result,
        "v79_audit": args.v79_audit,
        "screen_protocol": args.screen_protocol,
        "screen_result": args.screen_result,
        "screen_oof": args.screen_oof,
        "cache_protocol": args.cache_protocol,
        "cache_audit": args.cache_audit,
        "partition": args.partition,
        "manifest": args.manifest,
    }
    result = run_freeze(
        artifact_protocol_path=args.artifact_protocol,
        frozen_input_paths=frozen_inputs,
        v79_artifact_root=args.v79_artifact_root,
        cache_root=args.cache_root,
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
