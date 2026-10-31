#!/usr/bin/env python3
"""Freeze the post-v72 state-conditioned latent seed-LOO diagnostic."""

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
from cf_h2o.eval.traffic_signal_waiting_aligned_spatiotemporal_latent_diagnostic import (  # noqa: E402
    RESULT_PROTOCOL as V70_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (  # noqa: E402
    TLS_TIME_PHASE_PAIR_SCOPE,
)
from scripts.cluster.audit_tsc_external_waiting_aligned_latent_closed_loop import (  # noqa: E402
    AUDIT_PROTOCOL as V72_AUDIT_PROTOCOL,
    REJECTION_DECISION as V72_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_spatiotemporal_latent_diagnostic import (  # noqa: E402
    PROTOCOL as V70_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-state-conditioned-latent-diagnostic-v1"
MANIFEST_PROTOCOL = "external-la-jinan-v9-monotone-virtual-lane-safe-full-network-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _model_spec(
    *, feature_set: str, neighbors: int, shrinkage: float, bandwidth: float
) -> dict[str, Any]:
    key = (
        f"{feature_set}_k{neighbors}_s{str(float(shrinkage)).replace('.', 'p')}_"
        f"b{str(float(bandwidth)).replace('.', 'p')}"
    )
    return {
        "key": key,
        "family": "target_state_conditioned_spatiotemporal_latent",
        "config": {
            "base": {
                "scope": TLS_TIME_PHASE_PAIR_SCOPE,
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
    }


def freeze_protocol(
    *,
    v70_protocol_path: Path,
    v70_result_path: Path,
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
            f"refusing to overwrite state-conditioned diagnostic: {output_path}"
        )
    v70_protocol = _read_json(v70_protocol_path)
    v70_result = _read_json(v70_result_path)
    v72_protocol = _read_json(v72_protocol_path)
    v72_audit = _read_json(v72_audit_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    if (
        v70_protocol.get("protocol") != V70_PROTOCOL
        or v70_result.get("protocol") != V70_RESULT_PROTOCOL
        or v70_result.get("status") != "PASS"
        or v70_result.get("decision")
        != "authorize_waiting_aligned_spatiotemporal_latent_freeze"
        or v70_result.get("diagnostic_protocol_sha256")
        != _sha256(v70_protocol_path)
        or v72_protocol.get("protocol")
        != "tsc-v83r79-external-v9-waiting-aligned-latent-closed-loop-v1"
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
        raise ValueError("state-conditioned diagnostic authorization changed")
    training = dict(v70_protocol["training"])
    if tuple(training["seeds"]) != tuple(partition["redevelopment"]["seeds"]):
        raise ValueError("state-conditioned diagnostic seed partition changed")
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if any(path.exists() for path in sealed_roots):
        raise ValueError("confirmatory or prospective root is no longer sealed")
    candidates = (
        _model_spec(feature_set="compact_state", neighbors=5, shrinkage=1.0, bandwidth=0.75),
        _model_spec(feature_set="compact_state", neighbors=10, shrinkage=2.0, bandwidth=1.0),
        _model_spec(feature_set="compact_state", neighbors=10, shrinkage=5.0, bandwidth=1.0),
        _model_spec(feature_set="compact_state", neighbors=20, shrinkage=5.0, bandwidth=1.0),
        _model_spec(feature_set="state_action", neighbors=5, shrinkage=1.0, bandwidth=0.75),
        _model_spec(feature_set="state_action", neighbors=10, shrinkage=2.0, bandwidth=1.0),
        _model_spec(feature_set="state_action", neighbors=10, shrinkage=5.0, bandwidth=1.0),
        _model_spec(feature_set="state_action", neighbors=20, shrinkage=5.0, bandwidth=1.0),
    )
    gate_candidates = [
        {
            "risk_multiplier": risk,
            "minimum_context_trust": trust,
        }
        for risk in (0.0, 0.25)
        for trust in (0.0, 0.10, 0.25, 0.50)
    ]
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v72_rejection_pre_state_conditioned_oof_outcomes",
        "frozen_inputs": {
            "v70_protocol_path": str(v70_protocol_path.resolve()),
            "v70_protocol_sha256": _sha256(v70_protocol_path),
            "v70_result_path": str(v70_result_path.resolve()),
            "v70_result_sha256": _sha256(v70_result_path),
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
            "model_candidates": list(candidates),
            "latent_hierarchy": [
                "transferable_phase_pair_prior",
                "target_tls_residual",
                "target_tls_300_second_residual",
                "same_cell_deterministic_local_state_neighbors",
            ],
            "state_conditioning_is_local_to_cell": True,
        },
        "selection": {
            "gate_candidates": gate_candidates,
            "selection_rule": {
                "minimum_retained_groups_per_city": 500,
                "minimum_retained_seed_fraction": 1.0,
                "maximum_equal_seed_scenario_mean_delta": -0.0015,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.0075,
                "maximum_harmful_group_fraction": 0.42,
                "fallback": "retain_phase_pressure_if_no_candidate_is_feasible",
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "unit": "simulator_seed",
            },
        },
        "adaptive_status": {
            "classification": "post_closed_loop_failure_adaptive_development",
            "trigger": V72_REJECTION_DECISION,
            "closed_loop_failure": {
                row["policy"]: {
                    "mean_relative_delta": row["mean_relative_delta"],
                    "bootstrap_ci95": row["bootstrap"]["ci95"],
                }
                for row in v72_audit["candidate_summaries"]
            },
            "design_change": (
                "condition the previously selected TLS-time-phase latent on "
                "deterministic local state within each cell"
            ),
        },
        "claim_boundary": (
            "This diagnostic reuses only disclosed redevelopment simulator labels. "
            "It is adaptive target offline development after v72 failed, not "
            "confirmatory evidence. v84 and v85 remain sealed."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v70-protocol", type=Path, required=True)
    parser.add_argument("--v70-result", type=Path, required=True)
    parser.add_argument("--v72-protocol", type=Path, required=True)
    parser.add_argument("--v72-audit", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v70_protocol_path=args.v70_protocol,
        v70_result_path=args.v70_result,
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
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
