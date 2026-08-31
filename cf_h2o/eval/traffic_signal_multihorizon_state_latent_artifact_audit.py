"""Independently audit the frozen v79 multihorizon state-latent artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (
    AUTHORIZATION_DECISION as FREEZE_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as FREEZE_RESULT_PROTOCOL,
    _prediction_digest,
    load_multihorizon_state_originator,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    FEATURE_NAMES_V3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (
    ARTIFACT_KEYS,
    PRIMARY_ARTIFACT_KEY,
    PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-state-latent-artifact-audit-v2"
AUTHORIZATION_DECISION = "authorize_multihorizon_state_latent_closed_loop_development"
REJECTION_DECISION = "reject_multihorizon_state_latent_artifact_audit"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def online_contrast_feature_names() -> set[str]:
    """Mirror the feature names produced by make_target_action_contrasts."""

    return set(FEATURE_NAMES_V3).union(
        f"reference_{name}" for name in CONTRAST_FEATURES_V3
    ).union(f"delta_{name}" for name in CONTRAST_FEATURES_V3)


def audit_artifacts(
    *,
    artifact_protocol_path: Path,
    artifact_result_path: Path,
    artifact_root: Path,
    cache_root: Path,
    manifest_path: Path,
    partition_path: Path,
    conversion_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(artifact_protocol_path)
    freeze_result = _read_json(artifact_result_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or freeze_result.get("protocol") != FREEZE_RESULT_PROTOCOL
        or freeze_result.get("status") != "PASS"
        or freeze_result.get("decision") != FREEZE_AUTHORIZATION_DECISION
        or freeze_result.get("integrity_gate", {}).get("passed") is not True
        or freeze_result.get("artifact_protocol_sha256")
        != _sha256(artifact_protocol_path)
    ):
        raise ValueError("multihorizon artifact freeze evidence changed")
    training = dict(protocol["training"])
    seeds = tuple(int(value) for value in training["seeds"])
    scenario = str(training["scenarios"][0])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("multihorizon artifact audit scenario changed")
    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(1, int(workers)),
    )
    dataset = _merge_city_datasets(
        bank, (scenario,), city=str(training["city"])
    )
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=str(training["reference_policy"]),
        contrast_features=CONTRAST_FEATURES_V3,
    )
    groups = action_group_ids(contrast).astype(str)
    specs = {str(row["artifact_key"]): dict(row) for row in protocol["artifacts"]}
    online_feature_names = online_contrast_feature_names()
    rows = []
    for key in ARTIFACT_KEYS:
        originator, certificate, artifact = load_multihorizon_state_originator(
            artifact_root, key
        )
        result_record = dict(freeze_result["artifacts"][key])
        spec = specs[key]
        prediction_sha256 = _prediction_digest(originator.model, contrast)
        model_diagnostics = originator.model.diagnostics()
        row = {
            "artifact_key": key,
            "candidate_role": str(spec["role"]),
            "selection_eligible_closed_loop": bool(
                spec["selection_eligible_closed_loop"]
            ),
            "model_sha256": _sha256(Path(artifact_root) / key / "model.pkl"),
            "certificate_sha256": _sha256(
                Path(artifact_root) / key / "freeze.json"
            ),
            "prediction_sha256": prediction_sha256,
            "prediction_exact": bool(
                prediction_sha256 == certificate["prediction_sha256"]
                == result_record["prediction_sha256"]
            ),
            "model_key_exact": str(artifact["model_key"])
            == str(spec["model_key"]),
            "model_config_exact": model_diagnostics["config"]
            == {
                **dict(spec["model_config"]),
                "base": dict(spec["model_config"]["base"]),
            },
            "target_exact": bool(
                model_diagnostics["target_name"] == spec["target_name"]
                == "prefix_mean_cost_60s"
                and int(artifact["prediction_horizon_sec"]) == 60
            ),
            "online_feature_contract_exact": set(
                model_diagnostics["feature_names"]
            ).issubset(online_feature_names),
            "originator_gate_exact": bool(
                originator.config.risk_multiplier
                == float(spec["risk_multiplier"])
                == 0.0
                and originator.config.minimum_context_trust
                == float(spec["minimum_context_trust"])
                == 0.25
                and originator.config.rollout_value_horizon_sec == 60
            ),
        }
        row["passed"] = all(
            bool(value)
            for name, value in row.items()
            if name.endswith("_exact") or name == "prediction_exact"
        )
        rows.append(row)
    partition = _read_json(partition_path)
    if (
        protocol["frozen_inputs"]["partition"]["sha256"]
        != _sha256(partition_path)
    ):
        raise ValueError("multihorizon artifact audit partition changed")
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    integrity = {
        "freeze_evidence_chain_exact": True,
        "artifact_keys_exact": tuple(row["artifact_key"] for row in rows)
        == ARTIFACT_KEYS,
        "artifact_predictions_reconstructed_exact": all(
            row["prediction_exact"] for row in rows
        ),
        "artifact_contracts_exact": all(row["passed"] for row in rows),
        "training_rows_exact": int(contrast.size) == int(training["expected_rows"]),
        "training_groups_exact": int(np.unique(groups).size)
        == int(training["expected_groups"]),
        "only_primary_is_selection_eligible": [
            row["artifact_key"]
            for row in rows
            if row["selection_eligible_closed_loop"]
        ]
        == [PRIMARY_ARTIFACT_KEY],
        "sealed_roots_absent": not any(path.exists() for path in sealed_roots),
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
        "bank_audit": bank_audit,
        "training_row_count": int(contrast.size),
        "training_group_count": int(np.unique(groups).size),
        "artifact_audits": rows,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": bool(integrity["passed"]),
            "next_stage": "multihorizon_state_latent_closed_loop_development",
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-protocol", type=Path, required=True)
    parser.add_argument("--artifact-result", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite artifact audit: {args.out}")
    payload = audit_artifacts(
        artifact_protocol_path=args.artifact_protocol,
        artifact_result_path=args.artifact_result,
        artifact_root=args.artifact_root,
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        partition_path=args.partition,
        conversion_root=args.conversion_root,
        workers=args.workers,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "artifacts": len(payload["artifact_audits"]),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
