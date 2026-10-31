#!/usr/bin/env python3
"""Freeze the post-v74 hierarchical action-state residual OOF diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_distributional_action_state_diagnostic import (  # noqa: E402
    REJECTION_DECISION as V74_REJECTION_DECISION,
    RESULT_PROTOCOL as V74_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_distributional_action_state_diagnostic import (  # noqa: E402
    PROTOCOL as V74_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (  # noqa: E402
    MANIFEST_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-hierarchical-action-state-residual-diagnostic-v1"
MODEL_FAMILY = "hierarchical_action_state_residual"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _model_spec(
    *,
    feature_set: str,
    residual_scale: float,
    capacity: str,
) -> dict[str, Any]:
    if capacity == "regularized":
        boosting = {
            "max_iter": 120,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "l2_regularization": 20.0,
        }
    elif capacity == "expanded":
        boosting = {
            "max_iter": 160,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 10.0,
        }
    else:
        raise ValueError(f"unknown hierarchical residual capacity: {capacity!r}")
    encoded_scale = str(float(residual_scale)).replace(".", "p")
    return {
        "key": f"hier_{feature_set}_{capacity}_s{encoded_scale}",
        "family": MODEL_FAMILY,
        "config": {
            "base": {
                "scope": "tls_time_phase_pair",
                "shrinkage": 20.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": feature_set,
            "residual_scale": float(residual_scale),
            "residual_clip": 1.0,
            "learning_rate": 0.05,
            **boosting,
            "uncertainty_quantile": 0.9,
            "random_state": 20260830,
        },
    }


def freeze_protocol(
    *,
    v74_protocol_path: Path,
    v74_result_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite hierarchical residual diagnostic: {output_path}"
        )
    v74_protocol = _read_json(v74_protocol_path)
    v74_result = _read_json(v74_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    if (
        v74_protocol.get("protocol") != V74_PROTOCOL
        or v74_result.get("protocol") != V74_RESULT_PROTOCOL
        or v74_result.get("status") != "PASS"
        or v74_result.get("decision") != V74_REJECTION_DECISION
        or v74_result.get("integrity_gate", {}).get("passed") is not True
        or v74_result.get("diagnostic_protocol_sha256")
        != _sha256(v74_protocol_path)
        or cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("gate", {}).get("passed") is not True
        or partition.get("protocol") != PARTITION_PROTOCOL
        or manifest.get("protocol") != MANIFEST_PROTOCOL
    ):
        raise ValueError("hierarchical residual authorization changed")
    training = dict(v74_protocol["training"])
    if tuple(training["seeds"]) != tuple(partition["redevelopment"]["seeds"]):
        raise ValueError("hierarchical residual seed partition changed")
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if any(path.exists() for path in sealed_roots):
        raise ValueError("confirmatory or prospective root is no longer sealed")

    models = [
        _model_spec(
            feature_set=feature_set,
            residual_scale=scale,
            capacity="regularized",
        )
        for feature_set in ("compact_state_action", "causal_rigid")
        for scale in (0.25, 0.50, 1.0)
    ]
    models.extend(
        _model_spec(
            feature_set=feature_set,
            residual_scale=0.50,
            capacity="expanded",
        )
        for feature_set in ("compact_state_action", "causal_rigid")
    )
    gates = [
        {
            "risk_multiplier": risk,
            "minimum_context_trust": trust,
        }
        for risk in (0.0, 0.25, 0.50)
        for trust in (0.0, 0.10, 0.25, 0.50)
    ]
    parent_rule = dict(v74_protocol["selection"]["selection_rule"])
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v74_rejection_pre_hierarchical_residual_oof_outcomes",
        "frozen_inputs": {
            "v74_protocol_path": str(v74_protocol_path.resolve()),
            "v74_protocol_sha256": _sha256(v74_protocol_path),
            "v74_result_path": str(v74_result_path.resolve()),
            "v74_result_sha256": _sha256(v74_result_path),
            "cache_protocol_path": str(cache_protocol_path.resolve()),
            "cache_protocol_sha256": _sha256(cache_protocol_path),
            "cache_audit_path": str(cache_audit_path.resolve()),
            "cache_audit_sha256": _sha256(cache_audit_path),
            "partition_path": str(partition_path.resolve()),
            "partition_sha256": _sha256(partition_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "training": {
            **training,
            "cross_fitting": "leave_one_simulator_seed_out_22_folds",
            "model_candidates": models,
            "fixed_effect": "target_tls_300_second_reference_candidate_phase_pair",
            "pooled_target": "group_normalized_cost_minus_fixed_effect",
            "within_cell_feature_centering": True,
        },
        "selection": {
            "gate_candidates": gates,
            "selection_rule": parent_rule,
            "bootstrap": dict(v74_protocol["selection"]["bootstrap"]),
        },
        "adaptive_status": {
            "classification": "post_v74_rejection_third_stage_adaptive_development",
            "trigger": V74_REJECTION_DECISION,
            "failed_local_distributional_route": {
                "feasible_candidate_count": v74_result["gate_selection"][
                    "feasible_candidate_count"
                ],
                "reason": (
                    "probability gates reduced harm only by collapsing coverage; "
                    "shared state response across cells is required"
                ),
            },
            "design_change": (
                "retain the selected target TLS-time-phase fixed effect and learn "
                "a pooled causal state-action residual from within-cell deviations"
            ),
            "selection_thresholds_unchanged_from_v73": True,
        },
        "claim_boundary": (
            "This is third-stage adaptive target-offline redevelopment. It cannot "
            "support confirmatory claims; v84 and v85 remain sealed."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v74-protocol", type=Path, required=True)
    parser.add_argument("--v74-result", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v74_protocol_path=args.v74_protocol,
        v74_result_path=args.v74_result,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        partition_path=args.partition,
        manifest_path=args.manifest,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "models": len(payload["training"]["model_candidates"]),
                "gates": len(payload["selection"]["gate_candidates"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
