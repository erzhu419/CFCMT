#!/usr/bin/env python3
"""Freeze v82 uncertainty-aware h60/h120 successor artifacts."""

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
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_audit import (  # noqa: E402
    AUTHORIZATION_DECISION as V79_AUDIT_DECISION,
    RESULT_PROTOCOL as V79_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_artifact_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION as V79_FREEZE_DECISION,
    RESULT_PROTOCOL as V79_FREEZE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (  # noqa: E402
    rollout_prefix_target_name,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (  # noqa: E402
    MODEL_FAMILY as STATE_MODEL_FAMILY,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_multihorizon_estimand_aligned_closed_loop import (  # noqa: E402
    AUDIT_PROTOCOL as V81_AUDIT_PROTOCOL,
    REJECTION_DECISION as V81_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_estimand_aligned_closed_loop_development import (  # noqa: E402
    PROTOCOL as V81_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_latent_screen import (  # noqa: E402
    PROTOCOL as SCREEN_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (  # noqa: E402
    PROTOCOL as V79_ARTIFACT_PROTOCOL,
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


PROTOCOL = "tsc-v83r79-external-v9-uncertainty-horizon-artifacts-v1"
ARTIFACT_ROLE = "adaptive_uncertainty_horizon_action_originator"
ADAPTATION_CLASSIFICATION = "target_simulator_labeled_few_shot_adaptation"
ARTIFACT_SPECS = (
    ("compact_h60_r025_t010", "compact_k10_s2", 60, "compact_h60"),
    (
        "state_action_h60_r025_t010",
        "state_action_k10_s2",
        60,
        "state_action_h60",
    ),
    ("compact_h120_r025_t010", "compact_k10_s2", 120, None),
    ("state_action_h120_r025_t010", "state_action_k10_s2", 120, None),
)
ARTIFACT_KEYS = tuple(row[0] for row in ARTIFACT_SPECS)
RISK_MULTIPLIER = 0.25
MINIMUM_CONTEXT_TRUST = 0.10


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def _architecture(
    screen_protocol: Mapping[str, Any], architecture_key: str
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in screen_protocol["training"]["model_candidates"]
        if str(row["key"]) == str(architecture_key)
    ]
    if len(rows) != 1 or rows[0].get("family") != STATE_MODEL_FAMILY:
        raise ValueError(f"invalid state-conditioned architecture {architecture_key!r}")
    return rows[0]


def _screen_row(
    screen_result: Mapping[str, Any], architecture_key: str, horizon_sec: int
) -> dict[str, Any]:
    key = (
        f"{architecture_key}::h{int(horizon_sec)}s::r0p25::t0p1"
    )
    rows = [
        dict(row)
        for row in screen_result["gate_selection"]["grid"]
        if str(row["key"]) == key
    ]
    if len(rows) != 1:
        raise ValueError(f"OOF candidate {key!r} is not unique")
    row = rows[0]
    if not (
        float(row["risk_multiplier"]) == RISK_MULTIPLIER
        and float(row["minimum_context_trust"]) == MINIMUM_CONTEXT_TRUST
        and int(row["retained_seed_count"]) == 22
        and float(row["retained_seed_fraction"]) == 1.0
        and float(row["bootstrap"]["ci95"][1]) < 0.0
        and float(row["mean_delta"]) < 0.0
    ):
        raise ValueError(f"OOF candidate {key!r} lacks frozen directional support")
    return row


def freeze_protocol(
    *,
    v81_protocol_path: Path,
    v81_audit_path: Path,
    v79_protocol_path: Path,
    v79_result_path: Path,
    v79_audit_path: Path,
    v79_artifact_root: Path,
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
        raise FileExistsError(f"refusing to overwrite v82 protocol: {output_path}")
    v81_protocol = _read_json(v81_protocol_path)
    v81_audit = _read_json(v81_audit_path)
    v79_protocol = _read_json(v79_protocol_path)
    v79_result = _read_json(v79_result_path)
    v79_audit = _read_json(v79_audit_path)
    screen_protocol = _read_json(screen_protocol_path)
    screen_result = _read_json(screen_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    screen_frozen = dict(screen_protocol.get("frozen_inputs", {}))
    if not (
        v81_protocol.get("protocol") == V81_PROTOCOL
        and v81_audit.get("protocol") == V81_AUDIT_PROTOCOL
        and v81_audit.get("status") == "PASS"
        and v81_audit.get("decision") == V81_REJECTION_DECISION
        and v81_audit.get("integrity_gate", {}).get("passed") is True
        and v81_audit.get("protocol_sha256") == _sha256(v81_protocol_path)
        and v79_protocol.get("protocol") == V79_ARTIFACT_PROTOCOL
        and v79_result.get("protocol") == V79_FREEZE_PROTOCOL
        and v79_result.get("status") == "PASS"
        and v79_result.get("decision") == V79_FREEZE_DECISION
        and v79_result.get("artifact_protocol_sha256")
        == _sha256(v79_protocol_path)
        and v79_audit.get("protocol") == V79_AUDIT_PROTOCOL
        and v79_audit.get("status") == "PASS"
        and v79_audit.get("decision") == V79_AUDIT_DECISION
        and v79_audit.get("integrity_gate", {}).get("passed") is True
        and v79_audit.get("artifact_protocol_sha256")
        == _sha256(v79_protocol_path)
        and v79_audit.get("artifact_result_sha256") == _sha256(v79_result_path)
        and screen_protocol.get("protocol") == SCREEN_PROTOCOL
        and screen_result.get("protocol") == SCREEN_RESULT_PROTOCOL
        and screen_result.get("status") == "PASS"
        and screen_result.get("integrity_gate", {}).get("passed") is True
        and screen_result.get("diagnostic_protocol_sha256")
        == _sha256(screen_protocol_path)
        and screen_result.get("oof_artifact", {}).get("sha256")
        == _sha256(screen_oof_path)
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and cache_audit.get("gate", {}).get("passed") is True
        and partition.get("protocol") == PARTITION_PROTOCOL
        and manifest.get("protocol") == MANIFEST_PROTOCOL
        and screen_frozen.get("cache_protocol_sha256")
        == _sha256(cache_protocol_path)
        and screen_frozen.get("cache_audit_sha256") == _sha256(cache_audit_path)
        and screen_frozen.get("partition_sha256") == _sha256(partition_path)
        and screen_frozen.get("manifest_sha256") == _sha256(manifest_path)
    ):
        raise ValueError("v82 artifact parent evidence changed")

    training = dict(screen_protocol["training"])
    seeds = tuple(int(value) for value in training["seeds"])
    confirmatory = tuple(int(value) for value in training["sealed_confirmatory_seeds"])
    prospective = tuple(int(value) for value in training["sealed_prospective_seeds"])
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if (
        len(seeds) != 22
        or len(confirmatory) != 32
        or len(prospective) != 8
        or set(seeds).intersection(confirmatory)
        or set(seeds).intersection(prospective)
        or set(confirmatory).intersection(prospective)
        or any(path.exists() for path in sealed_roots)
    ):
        raise ValueError("v82 artifact partition or sealed roots changed")

    artifacts = []
    for artifact_key, architecture_key, horizon_sec, source_key in ARTIFACT_SPECS:
        architecture = _architecture(screen_protocol, architecture_key)
        screen_row = _screen_row(screen_result, architecture_key, horizon_sec)
        source = None
        if source_key is not None:
            source_model = Path(v79_artifact_root) / source_key / "model.pkl"
            source_certificate = Path(v79_artifact_root) / source_key / "freeze.json"
            source_result = dict(v79_result["artifacts"][source_key])
            if (
                source_result["model"]["sha256"] != _sha256(source_model)
                or source_result["certificate"]["sha256"]
                != _sha256(source_certificate)
            ):
                raise ValueError(f"v82 source artifact {source_key!r} changed")
            source = {
                "kind": "reuse_v79_frozen_model",
                "artifact_key": source_key,
                "model": _path_record(source_model),
                "certificate": _path_record(source_certificate),
            }
        else:
            source = {"kind": "fit_full_development_cache"}
        artifacts.append(
            {
                "artifact_key": artifact_key,
                "role": "adaptive_successor_candidate",
                "selection_eligible_closed_loop": True,
                "architecture_key": architecture_key,
                "model_key": f"{architecture_key}::h{horizon_sec}s",
                "model_family": str(architecture["family"]),
                "model_config": dict(architecture["config"]),
                "target_name": rollout_prefix_target_name(horizon_sec),
                "prediction_horizon_sec": int(horizon_sec),
                "risk_multiplier": RISK_MULTIPLIER,
                "minimum_context_trust": MINIMUM_CONTEXT_TRUST,
                "development_screen_summary": screen_row,
                "model_source": source,
            }
        )

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v81_rejection_pre_v82_successor_artifact_fit",
        "frozen_inputs": {
            "v81_protocol": _path_record(v81_protocol_path),
            "v81_audit": _path_record(v81_audit_path),
            "v79_protocol": _path_record(v79_protocol_path),
            "v79_result": _path_record(v79_result_path),
            "v79_audit": _path_record(v79_audit_path),
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
            "seeds": list(seeds),
            "sealed_confirmatory_seeds": list(confirmatory),
            "sealed_prospective_seeds": list(prospective),
            "reference_policy": str(training["reference_policy"]),
            "collection_shards": int(training["collection_shards"]),
            "expected_rows": int(cache_audit["total_rows"]),
            "expected_groups": int(cache_audit["total_groups"]),
            "expected_action_count_per_group": int(
                training["expected_action_count_per_group"]
            ),
            "adaptation_classification": ADAPTATION_CLASSIFICATION,
            "zero_shot": False,
        },
        "artifacts": artifacts,
        "artifact_keys": list(ARTIFACT_KEYS),
        "artifact_contract": {
            "artifact_role": ARTIFACT_ROLE,
            "objective_name": "control_cost",
            "prediction_horizons_sec": [60, 120],
            "risk_multiplier": RISK_MULTIPLIER,
            "minimum_context_trust": MINIMUM_CONTEXT_TRUST,
            "uses_target_labels_at_fit": True,
            "uses_future_outcomes_at_prediction": False,
        },
        "output_roots": {
            "remote_artifact_root": str(remote_artifact_root),
            "local_artifact_root": str(Path(local_artifact_root).resolve()),
        },
        "advance_contract": {
            "independent_artifact_audit_required": True,
            "all_four_candidates_are_adaptive_development_only": True,
            "untouched_v84_required_after_v83_selection": True,
            "confirmatory_and_prospective_roots_remain_sealed": True,
        },
        "claim_boundary": (
            "The four uncertainty-aware h60/h120 artifacts are adaptive successor "
            "candidates after the v81 rejection. H60 reuses the audited v79 models; "
            "h120 is fitted on the same disclosed 22-seed target-simulator cache. "
            "All use risk=0.25 and trust=0.10 fixed from the pre-existing v77 OOF "
            "screen. No efficacy claim is authorized, and v84/v85 remain sealed."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v81-protocol", type=Path, required=True)
    parser.add_argument("--v81-audit", type=Path, required=True)
    parser.add_argument("--v79-protocol", type=Path, required=True)
    parser.add_argument("--v79-result", type=Path, required=True)
    parser.add_argument("--v79-audit", type=Path, required=True)
    parser.add_argument("--v79-artifact-root", type=Path, required=True)
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
        v81_protocol_path=args.v81_protocol,
        v81_audit_path=args.v81_audit,
        v79_protocol_path=args.v79_protocol,
        v79_result_path=args.v79_result,
        v79_audit_path=args.v79_audit,
        v79_artifact_root=args.v79_artifact_root,
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
                "artifacts": payload["artifact_keys"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
