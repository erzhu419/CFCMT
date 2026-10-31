#!/usr/bin/env python3
"""Freeze the v79 deployable multihorizon state-latent artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_multihorizon_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (  # noqa: E402
    RESULT_PROTOCOL as SCREEN_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_nested_selection_audit import (  # noqa: E402
    AUTHORIZATION_DECISION as NESTED_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as NESTED_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (  # noqa: E402
    rollout_prefix_target_name,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (  # noqa: E402
    MODEL_FAMILY as STATE_MODEL_FAMILY,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_multihorizon_latent_screen import (  # noqa: E402
    PROTOCOL as SCREEN_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_nested_selection import (  # noqa: E402
    PROTOCOL as NESTED_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (  # noqa: E402
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (  # noqa: E402
    MANIFEST_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-multihorizon-state-latent-artifacts-v1"
ARTIFACT_ROLE = "target_multihorizon_state_action_originator"
ADAPTATION_CLASSIFICATION = "target_simulator_labeled_few_shot_adaptation"
PRIMARY_ARTIFACT_KEY = "compact_h60"
SECONDARY_ARTIFACT_KEY = "state_action_h60"
ARTIFACT_KEYS = (PRIMARY_ARTIFACT_KEY, SECONDARY_ARTIFACT_KEY)
PRIMARY_SELECTION_KEY = "compact_k10_s2::h60s::r0p0::t0p25"
SECONDARY_SELECTION_KEY = "state_action_k10_s2::h60s::r0p0::t0p25"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def _model_architecture(
    screen_protocol: Mapping[str, Any], architecture_key: str
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in screen_protocol["training"]["model_candidates"]
        if str(row["key"]) == str(architecture_key)
    ]
    if len(matches) != 1:
        raise ValueError(f"model architecture {architecture_key!r} is not unique")
    model = matches[0]
    if str(model.get("family")) != STATE_MODEL_FAMILY:
        raise ValueError("multihorizon deployable artifact is not state-conditioned")
    return model


def _screen_candidate(
    screen_result: Mapping[str, Any], selection_key: str
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in screen_result["gate_selection"]["grid"]
        if str(row["key"]) == str(selection_key)
    ]
    if len(matches) != 1 or matches[0].get("feasible") is not True:
        raise ValueError(f"screen candidate {selection_key!r} is not uniquely feasible")
    return matches[0]


def _artifact_candidate(
    *,
    artifact_key: str,
    role: str,
    selected: Mapping[str, Any],
    architecture: Mapping[str, Any],
    selection_eligible: bool,
) -> dict[str, Any]:
    model_key = str(selected["model_key"])
    architecture_key, separator, horizon = model_key.partition("::h")
    if separator != "::h" or not horizon.endswith("s"):
        raise ValueError(f"invalid multihorizon model key: {model_key!r}")
    horizon_sec = int(horizon[:-1])
    if architecture_key != str(architecture["key"]) or horizon_sec != 60:
        raise ValueError("multihorizon artifact architecture or horizon changed")
    return {
        "artifact_key": str(artifact_key),
        "role": str(role),
        "selection_eligible_closed_loop": bool(selection_eligible),
        "selection_key": str(selected["key"]),
        "model_key": model_key,
        "architecture_key": architecture_key,
        "model_family": str(architecture["family"]),
        "model_config": dict(architecture["config"]),
        "target_name": rollout_prefix_target_name(horizon_sec),
        "prediction_horizon_sec": horizon_sec,
        "risk_multiplier": float(selected["risk_multiplier"]),
        "minimum_context_trust": float(selected["minimum_context_trust"]),
        "development_screen_summary": dict(selected),
    }


def freeze_protocol(
    *,
    nested_protocol_path: Path,
    nested_result_path: Path,
    nested_deployment_oof_path: Path,
    screen_protocol_path: Path,
    screen_result_path: Path,
    screen_oof_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    remote_artifact_root: Path,
    local_artifact_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v79 protocol: {output_path}")
    nested_protocol = _read_json(nested_protocol_path)
    nested_result = _read_json(nested_result_path)
    screen_protocol = _read_json(screen_protocol_path)
    screen_result = _read_json(screen_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    nested_frozen = dict(nested_protocol.get("frozen_inputs", {}))
    nested_oof_record = dict(nested_result.get("deployment_oof_artifact", {}))
    primary = dict(nested_result.get("final_full_development_candidate", {}))
    stability = dict(nested_result.get("selection_stability", {}))
    evidence_ok = bool(
        nested_protocol.get("protocol") == NESTED_PROTOCOL
        and nested_result.get("protocol") == NESTED_RESULT_PROTOCOL
        and nested_result.get("status") == "PASS"
        and nested_result.get("decision") == NESTED_AUTHORIZATION_DECISION
        and nested_result.get("integrity_gate", {}).get("passed") is True
        and nested_result.get("advance_gate", {}).get("passed") is True
        and nested_result.get("nested_protocol_sha256")
        == _sha256(nested_protocol_path)
        and nested_oof_record.get("sha256") == _sha256(nested_deployment_oof_path)
        and primary.get("key") == PRIMARY_SELECTION_KEY
        and primary.get("feasible") is True
        and stability.get("candidate_counts", {}).get(SECONDARY_SELECTION_KEY) == 21
        and stability.get("fallback_outer_fold_count") == 0
        and screen_protocol.get("protocol") == SCREEN_PROTOCOL
        and screen_result.get("protocol") == SCREEN_RESULT_PROTOCOL
        and screen_result.get("status") == "PASS"
        and screen_result.get("integrity_gate", {}).get("passed") is True
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("gate", {}).get("passed") is True
        and partition.get("protocol") == PARTITION_PROTOCOL
        and manifest.get("protocol") == MANIFEST_PROTOCOL
        and nested_frozen.get("screen_protocol_sha256")
        == _sha256(screen_protocol_path)
        and nested_frozen.get("screen_result_sha256")
        == _sha256(screen_result_path)
        and nested_frozen.get("screen_oof_sha256") == _sha256(screen_oof_path)
        and nested_frozen.get("cache_protocol_sha256")
        == _sha256(cache_protocol_path)
        and nested_frozen.get("cache_audit_sha256") == _sha256(cache_audit_path)
        and nested_frozen.get("partition_sha256") == _sha256(partition_path)
        and nested_frozen.get("manifest_sha256") == _sha256(manifest_path)
    )
    if not evidence_ok:
        raise ValueError("v79 multihorizon artifact parent evidence changed")

    training = dict(screen_protocol["training"])
    development_seeds = tuple(int(value) for value in training["seeds"])
    confirmatory_seeds = tuple(
        int(value) for value in training["sealed_confirmatory_seeds"]
    )
    prospective_seeds = tuple(
        int(value) for value in training["sealed_prospective_seeds"]
    )
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if (
        len(development_seeds) != 22
        or len(confirmatory_seeds) != 32
        or len(prospective_seeds) != 8
        or set(development_seeds) != {
            int(value) for value in partition["redevelopment"]["seeds"]
        }
        or set(development_seeds).intersection(confirmatory_seeds)
        or set(development_seeds).intersection(prospective_seeds)
        or set(confirmatory_seeds).intersection(prospective_seeds)
        or any(path.exists() for path in sealed_roots)
    ):
        raise ValueError("v79 development partition or sealed roots changed")

    primary_architecture = _model_architecture(screen_protocol, "compact_k10_s2")
    secondary_architecture = _model_architecture(
        screen_protocol, "state_action_k10_s2"
    )
    secondary = _screen_candidate(screen_result, SECONDARY_SELECTION_KEY)
    artifacts = (
        _artifact_candidate(
            artifact_key=PRIMARY_ARTIFACT_KEY,
            role="preregistered_primary",
            selected=primary,
            architecture=primary_architecture,
            selection_eligible=True,
        ),
        _artifact_candidate(
            artifact_key=SECONDARY_ARTIFACT_KEY,
            role="predeclared_stability_diagnostic",
            selected=secondary,
            architecture=secondary_architecture,
            selection_eligible=False,
        ),
    )
    if (
        tuple(row["artifact_key"] for row in artifacts) != ARTIFACT_KEYS
        or any(float(row["risk_multiplier"]) != 0.0 for row in artifacts)
        or any(float(row["minimum_context_trust"]) != 0.25 for row in artifacts)
    ):
        raise ValueError("v79 artifact gate contract changed")

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v78_nested_pre_full_development_artifact_fit",
        "frozen_inputs": {
            "nested_protocol": _path_record(nested_protocol_path),
            "nested_result": _path_record(nested_result_path),
            "nested_deployment_oof": _path_record(nested_deployment_oof_path),
            "screen_protocol": _path_record(screen_protocol_path),
            "screen_result": _path_record(screen_result_path),
            "screen_oof": _path_record(screen_oof_path),
            "cache_protocol": _path_record(cache_protocol_path),
            "cache_audit": _path_record(cache_audit_path),
            "partition": _path_record(partition_path),
            "manifest": _path_record(manifest_path),
        },
        "training": {
            "city": str(training["city"]),
            "scenarios": list(training["scenarios"]),
            "seeds": list(development_seeds),
            "sealed_confirmatory_seeds": list(confirmatory_seeds),
            "sealed_prospective_seeds": list(prospective_seeds),
            "reference_policy": str(training["reference_policy"]),
            "collection_shards": int(training["collection_shards"]),
            "expected_rows": int(cache_audit["total_rows"]),
            "expected_groups": int(cache_audit["total_groups"]),
            "expected_action_count_per_group": int(
                training["expected_action_count_per_group"]
            ),
            "counterfactual_cost_mode": "halted_queue",
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
        },
        "artifacts": list(artifacts),
        "artifact_keys": list(ARTIFACT_KEYS),
        "primary_artifact_key": PRIMARY_ARTIFACT_KEY,
        "secondary_artifact_key": SECONDARY_ARTIFACT_KEY,
        "artifact_contract": {
            "artifact_role": ARTIFACT_ROLE,
            "objective_name": "control_cost",
            "target_name": rollout_prefix_target_name(60),
            "prediction_horizon_sec": 60,
            "uses_target_labels_at_fit": True,
            "uses_future_outcomes_at_prediction": False,
            "proposal_origin_changed": True,
        },
        "output_roots": {
            "remote_artifact_root": str(remote_artifact_root),
            "local_artifact_root": str(Path(local_artifact_root).resolve()),
        },
        "advance_contract": {
            "independent_artifact_audit_required": True,
            "closed_loop_primary_selection_uses_compact_only": True,
            "state_action_is_diagnostic_only": True,
            "confirmatory_and_prospective_roots_remain_sealed": True,
        },
        "claim_boundary": (
            "Both deployable models are fitted only on the disclosed 22-seed "
            "target-simulator development cache. Compact h60 is the frozen primary; "
            "state-action h60 is a predeclared stability diagnostic and cannot "
            "authorize confirmation. This is target-labeled few-shot adaptation, "
            "not zero-shot transfer, and v84/v85 remain sealed."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-protocol", type=Path, required=True)
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--nested-deployment-oof", type=Path, required=True)
    parser.add_argument("--screen-protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-oof", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", type=Path, required=True)
    parser.add_argument("--local-artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        nested_protocol_path=args.nested_protocol,
        nested_result_path=args.nested_result,
        nested_deployment_oof_path=args.nested_deployment_oof,
        screen_protocol_path=args.screen_protocol,
        screen_result_path=args.screen_result,
        screen_oof_path=args.screen_oof,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        partition_path=args.partition,
        manifest_path=args.manifest,
        remote_artifact_root=args.remote_artifact_root,
        local_artifact_root=args.local_artifact_root,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "artifact_keys": payload["artifact_keys"],
                "training_seeds": len(payload["training"]["seeds"]),
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
