#!/usr/bin/env python3
"""Freeze the post-v73 distributional action-state OOF diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (  # noqa: E402
    REJECTION_DECISION as V73_REJECTION_DECISION,
    RESULT_PROTOCOL as V73_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_waiting_aligned_latent_closed_loop import (  # noqa: E402
    AUDIT_PROTOCOL as V72_AUDIT_PROTOCOL,
    REJECTION_DECISION as V72_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (  # noqa: E402
    MANIFEST_PROTOCOL,
    PROTOCOL as V73_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_latent_closed_loop_development import (  # noqa: E402
    PROTOCOL as V72_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-distributional-action-state-diagnostic-v1"
MODEL_FAMILY = "distributional_action_state_latent"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _model_spec(
    *,
    feature_set: str,
    neighbors: int,
    shrinkage: float,
    bandwidth: float,
) -> dict[str, Any]:
    key = (
        f"dist_{feature_set}_k{neighbors}_"
        f"s{str(float(shrinkage)).replace('.', 'p')}_"
        f"b{str(float(bandwidth)).replace('.', 'p')}"
    )
    return {
        "key": key,
        "family": MODEL_FAMILY,
        "config": {
            "state_model": {
                "base": {
                    "scope": "tls_time_phase_pair",
                    "shrinkage": 20.0,
                    "time_bin_sec": 300,
                    "uncertainty_quantile": 0.9,
                },
                "feature_set": feature_set,
                "neighbor_count": int(neighbors),
                "local_shrinkage": float(shrinkage),
                "distance_bandwidth": float(bandwidth),
                "uncertainty_quantile": 0.9,
            },
            "benefit_prior_alpha": 1.0,
            "benefit_prior_beta": 1.0,
        },
    }


def freeze_protocol(
    *,
    v73_protocol_path: Path,
    v73_result_path: Path,
    v72_protocol_path: Path,
    v72_audit_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite distributional diagnostic: {output_path}"
        )
    v73_protocol = _read_json(v73_protocol_path)
    v73_result = _read_json(v73_result_path)
    v72_protocol = _read_json(v72_protocol_path)
    v72_audit = _read_json(v72_audit_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    if (
        v73_protocol.get("protocol") != V73_PROTOCOL
        or v73_result.get("protocol") != V73_RESULT_PROTOCOL
        or v73_result.get("status") != "PASS"
        or v73_result.get("decision") != V73_REJECTION_DECISION
        or v73_result.get("integrity_gate", {}).get("passed") is not True
        or v73_result.get("diagnostic_protocol_sha256")
        != _sha256(v73_protocol_path)
        or v72_protocol.get("protocol") != V72_PROTOCOL
        or v72_audit.get("protocol") != V72_AUDIT_PROTOCOL
        or v72_audit.get("status") != "PASS"
        or v72_audit.get("decision") != V72_REJECTION_DECISION
        or v72_audit.get("integrity_gate", {}).get("passed") is not True
        or v72_audit.get("protocol_sha256") != _sha256(v72_protocol_path)
        or cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("gate", {}).get("passed") is not True
        or partition.get("protocol") != PARTITION_PROTOCOL
        or manifest.get("protocol") != MANIFEST_PROTOCOL
    ):
        raise ValueError("distributional diagnostic authorization changed")
    training = dict(v73_protocol["training"])
    if tuple(training["seeds"]) != tuple(partition["redevelopment"]["seeds"]):
        raise ValueError("distributional diagnostic seed partition changed")
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if any(path.exists() for path in sealed_roots):
        raise ValueError("confirmatory or prospective root is no longer sealed")

    models = (
        _model_spec(
            feature_set="compact_state",
            neighbors=5,
            shrinkage=1.0,
            bandwidth=0.75,
        ),
        _model_spec(
            feature_set="compact_state",
            neighbors=10,
            shrinkage=2.0,
            bandwidth=1.0,
        ),
        _model_spec(
            feature_set="state_action",
            neighbors=5,
            shrinkage=1.0,
            bandwidth=0.75,
        ),
        _model_spec(
            feature_set="state_action",
            neighbors=10,
            shrinkage=2.0,
            bandwidth=1.0,
        ),
    )
    gates = [
        {
            "benefit_probability_kind": kind,
            "minimum_benefit_probability": probability,
            "risk_multiplier": risk,
            "minimum_context_trust": trust,
        }
        for kind in ("raw", "calibrated")
        for probability in (0.55, 0.60, 0.65, 0.70)
        for risk in (0.0, 0.25)
        for trust in (0.25, 0.50)
    ]
    parent_rule = dict(v73_protocol["selection"]["selection_rule"])
    near_misses = [
        {
            "key": row["key"],
            "retained_group_count": row["retained_group_count"],
            "mean_delta": row["mean_delta"],
            "bootstrap_ci95": row["bootstrap"]["ci95"],
            "worst_seed_delta": row["worst_seed_delta"],
            "harmful_group_fraction": row["harmful_group_fraction"],
            "failed_gates": [
                key
                for key, passed in row["gates"].items()
                if key != "passed" and not bool(passed)
            ],
        }
        for row in v73_result["gate_selection"]["grid"]
        if int(row["retained_group_count"])
        >= int(parent_rule["minimum_retained_groups_per_city"])
        and row["gates"].get("mean_delta") is True
        and row["gates"].get("bootstrap_upper") is True
        and row["gates"].get("worst_seed") is True
        and row["gates"].get("minimum_retained_seed_fraction") is True
        and row["gates"].get("harmful_fraction") is False
    ]
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v73_rejection_pre_distributional_action_state_oof_outcomes",
        "frozen_inputs": {
            "v73_protocol_path": str(v73_protocol_path.resolve()),
            "v73_protocol_sha256": _sha256(v73_protocol_path),
            "v73_result_path": str(v73_result_path.resolve()),
            "v73_result_sha256": _sha256(v73_result_path),
            "v72_protocol_path": str(v72_protocol_path.resolve()),
            "v72_protocol_sha256": _sha256(v72_protocol_path),
            "v72_audit_path": str(v72_audit_path.resolve()),
            "v72_audit_sha256": _sha256(v72_audit_path),
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
            "model_candidates": list(models),
            "distributional_target": "probability_group_normalized_delta_below_zero",
            "probability_calibration": (
                "isotonic_on_leave_self_out_training_local_probabilities"
            ),
        },
        "selection": {
            "gate_candidates": gates,
            "selection_rule": parent_rule,
            "bootstrap": dict(v73_protocol["selection"]["bootstrap"]),
        },
        "adaptive_status": {
            "classification": "post_v73_rejection_second_stage_adaptive_development",
            "trigger": V73_REJECTION_DECISION,
            "parent_near_miss_count": len(near_misses),
            "parent_near_misses": near_misses,
            "design_change": (
                "estimate and calibrate training-only local action benefit probability "
                "and require it in addition to mean, uncertainty, and context trust"
            ),
            "selection_thresholds_unchanged_from_v73": True,
        },
        "claim_boundary": (
            "This diagnostic is adaptive target-offline redevelopment after v73. "
            "It cannot support confirmatory claims; v84 and v85 remain sealed."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v73-protocol", type=Path, required=True)
    parser.add_argument("--v73-result", type=Path, required=True)
    parser.add_argument("--v72-protocol", type=Path, required=True)
    parser.add_argument("--v72-audit", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v73_protocol_path=args.v73_protocol,
        v73_result_path=args.v73_result,
        v72_protocol_path=args.v72_protocol,
        v72_audit_path=args.v72_audit,
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
                "parent_near_misses": payload["adaptive_status"][
                    "parent_near_miss_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
