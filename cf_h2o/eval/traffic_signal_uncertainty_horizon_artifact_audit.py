"""Independently audit the frozen v82 uncertainty-horizon artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (
    _prediction_digest,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_uncertainty_horizon_artifact_freeze import (
    AUTHORIZATION_DECISION as FREEZE_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
    _read_json,
    load_uncertainty_horizon_originator,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_artifacts import (
    ARTIFACT_KEYS,
    MINIMUM_CONTEXT_TRUST,
    PROTOCOL,
    RISK_MULTIPLIER,
)


RESULT_PROTOCOL = "tsc-v83r79-uncertainty-horizon-artifact-audit-v1"
AUTHORIZATION_DECISION = "authorize_uncertainty_horizon_closed_loop_development"
REJECTION_DECISION = "reject_uncertainty_horizon_artifact_deployment"


def audit_artifacts(
    *,
    artifact_protocol_path: Path,
    artifact_result_path: Path,
    frozen_input_paths: Mapping[str, Path],
    artifact_root: Path,
    cache_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(artifact_protocol_path)
    result = _read_json(artifact_result_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    if not (
        protocol.get("protocol") == PROTOCOL
        and result.get("protocol") == FREEZE_RESULT_PROTOCOL
        and result.get("status") == "PASS"
        and result.get("decision") == FREEZE_AUTHORIZATION_DECISION
        and result.get("integrity_gate", {}).get("passed") is True
        and result.get("artifact_protocol_sha256")
        == _sha256(artifact_protocol_path)
        and all(
            _sha256(path) == frozen[key]["sha256"]
            for key, path in frozen_input_paths.items()
        )
    ):
        raise ValueError("v82 artifact audit evidence changed")
    partition = _read_json(frozen_input_paths["partition"])
    if any(
        Path(partition[key]["root"]).exists()
        for key in ("confirmatory", "prospective")
    ):
        raise ValueError("v82 artifact audit found unsealed holdout roots")

    training = dict(protocol["training"])
    scenario = str(training["scenarios"][0])
    seeds = tuple(int(value) for value in training["seeds"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected) != 1:
        raise ValueError("v82 artifact audit scenario is not unique")
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        replace(full_manifest, scenarios=selected),
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
    specs = {str(row["artifact_key"]): dict(row) for row in protocol["artifacts"]}
    rows = {}
    for key in ARTIFACT_KEYS:
        originator, certificate, artifact = load_uncertainty_horizon_originator(
            artifact_root, key
        )
        record = dict(result["artifacts"][key])
        model_path = Path(artifact_root) / key / "model.pkl"
        certificate_path = Path(artifact_root) / key / "freeze.json"
        prediction_sha = _prediction_digest(originator.model, contrast)
        spec = specs[key]
        exact = bool(
            record["model"]["sha256"] == _sha256(model_path)
            and record["certificate"]["sha256"] == _sha256(certificate_path)
            and record["prediction_sha256"] == prediction_sha
            and certificate["prediction_sha256"] == prediction_sha
            and str(artifact["model_key"]) == str(spec["model_key"])
            and str(artifact["target_name"]) == str(spec["target_name"])
            and int(artifact["prediction_horizon_sec"])
            == int(spec["prediction_horizon_sec"])
            and float(originator.config.risk_multiplier) == RISK_MULTIPLIER
            and float(originator.config.minimum_context_trust)
            == MINIMUM_CONTEXT_TRUST
            and int(originator.config.rollout_value_horizon_sec)
            == int(spec["prediction_horizon_sec"])
            and int(certificate["training_row_count"]) == int(contrast.size)
            and int(certificate["training_group_count"]) == group_count
            and int(certificate["training_seed_count"]) == len(seeds)
        )
        rows[key] = {
            "artifact_key": key,
            "model_sha256": _sha256(model_path),
            "certificate_sha256": _sha256(certificate_path),
            "prediction_sha256": prediction_sha,
            "fit_mode": str(artifact["fit_mode"]),
            "prediction_horizon_sec": int(artifact["prediction_horizon_sec"]),
            "risk_multiplier": float(originator.config.risk_multiplier),
            "minimum_context_trust": float(
                originator.config.minimum_context_trust
            ),
            "exact": exact,
        }
    integrity = {
        "freeze_chain_exact": True,
        "frozen_inputs_exact": True,
        "row_count_exact": int(contrast.size) == int(training["expected_rows"]),
        "group_count_exact": group_count == int(training["expected_groups"]),
        "bank_rows_exact": int(bank_audit["rows_by_scenario"][scenario])
        == int(training["expected_rows"]),
        "artifact_keys_exact": tuple(rows) == ARTIFACT_KEYS,
        "all_model_predictions_recomputed_exact": all(
            row["exact"] for row in rows.values()
        ),
        "h60_models_reused": all(
            rows[key]["fit_mode"] == "reused_v79_audited_model"
            for key in ARTIFACT_KEYS
            if "h60_" in key
        ),
        "h120_models_refit": all(
            rows[key]["fit_mode"] == "fit_full_v82_development_cache"
            for key in ARTIFACT_KEYS
            if "h120_" in key
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
        "artifact_result_sha256": _sha256(artifact_result_path),
        "artifact_root": str(Path(artifact_root).resolve()),
        "dataset_rows": int(contrast.size),
        "action_groups": group_count,
        "artifacts": rows,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": bool(integrity["passed"]),
            "next_stage": "v83_uncertainty_horizon_closed_loop_development",
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v82 audit: {args.out}")
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
    payload = audit_artifacts(
        artifact_protocol_path=args.artifact_protocol,
        artifact_result_path=args.artifact_result,
        frozen_input_paths=frozen_inputs,
        artifact_root=args.artifact_root,
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        workers=args.workers,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "artifacts": list(payload["artifacts"]),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
