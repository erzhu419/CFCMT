#!/usr/bin/env python3
"""Freeze the v77 multihorizon latent OOF development screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (  # noqa: E402
    TLS_TIME_PHASE_PAIR_SCOPE,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (  # noqa: E402
    BASE_LATENT_MODEL_FAMILY,
    MODEL_FAMILY as STATE_MODEL_FAMILY,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (  # noqa: E402
    PREFIX_HORIZONS_SEC,
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (  # noqa: E402
    MANIFEST_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-multihorizon-latent-screen-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _base_spec(key: str, *, shrinkage: float, time_bin_sec: int) -> dict[str, Any]:
    return {
        "key": key,
        "family": BASE_LATENT_MODEL_FAMILY,
        "config": {
            "scope": TLS_TIME_PHASE_PAIR_SCOPE,
            "shrinkage": float(shrinkage),
            "time_bin_sec": int(time_bin_sec),
            "uncertainty_quantile": 0.9,
        },
    }


def _state_spec(key: str, *, feature_set: str) -> dict[str, Any]:
    return {
        "key": key,
        "family": STATE_MODEL_FAMILY,
        "config": {
            "base": {
                "scope": TLS_TIME_PHASE_PAIR_SCOPE,
                "shrinkage": 20.0,
                "time_bin_sec": 300,
                "uncertainty_quantile": 0.9,
            },
            "feature_set": feature_set,
            "neighbor_count": 10,
            "local_shrinkage": 2.0,
            "distance_bandwidth": 1.0,
            "uncertainty_quantile": 0.9,
        },
    }


def freeze_protocol(
    *,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite multihorizon screen: {output_path}"
        )
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    frozen_cache = cache_protocol.get("frozen_inputs", {})
    if (
        cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision")
        != "authorize_multihorizon_nested_seed_oof_development"
        or cache_audit.get("gate", {}).get("passed") is not True
        or tuple(cache_audit.get("rollout_prefix_horizons_sec", ()))
        != PREFIX_HORIZONS_SEC
        or partition.get("protocol") != PARTITION_PROTOCOL
        or manifest.get("protocol") != MANIFEST_PROTOCOL
        or frozen_cache.get("partition", {}).get("sha256")
        != _sha256(partition_path)
        or frozen_cache.get("manifest", {}).get("sha256")
        != _sha256(manifest_path)
    ):
        raise ValueError("multihorizon screen parent evidence changed")
    seeds = tuple(int(value) for value in partition["redevelopment"]["seeds"])
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if len(seeds) != 22 or any(path.exists() for path in sealed_roots):
        raise ValueError("multihorizon screen development partition changed")
    model_candidates = (
        _base_spec("base_t300_s20", shrinkage=20.0, time_bin_sec=300),
        _base_spec("base_t300_s5", shrinkage=5.0, time_bin_sec=300),
        _base_spec("base_t600_s20", shrinkage=20.0, time_bin_sec=600),
        _state_spec("state_action_k10_s2", feature_set="state_action"),
        _state_spec("compact_k10_s2", feature_set="compact_state"),
    )
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_cache_pre_multihorizon_oof_outcomes",
        "frozen_inputs": {
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
            "city": "jinan",
            "scenarios": ["jinan_3x4_real"],
            "seeds": list(seeds),
            "collection_shards": int(
                cache_protocol["collection"]["collection_shards_per_seed"]
            ),
            "reference_policy": "phase_pressure",
            "expected_action_count_per_group": 8,
            "cross_fitting": "leave_one_simulator_seed_out_22_folds",
            "rollout_prefix_horizons_sec": list(PREFIX_HORIZONS_SEC),
            "model_candidates": list(model_candidates),
            "expanded_candidate_count": len(model_candidates)
            * len(PREFIX_HORIZONS_SEC),
            "sealed_confirmatory_seeds": list(partition["confirmatory"]["seeds"]),
            "sealed_prospective_seeds": list(partition["prospective"]["seeds"]),
        },
        "selection": {
            "gate_candidates": [
                {
                    "risk_multiplier": risk,
                    "minimum_context_trust": trust,
                }
                for risk in (0.0, 0.25)
                for trust in (0.0, 0.10, 0.25, 0.50)
            ],
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
        "advance_contract": {
            "screen_is_not_deployment_evidence": True,
            "required_next_stage": (
                "nested_leave_one_seed_out_joint_horizon_model_gate_selection"
            ),
            "closed_loop_forbidden_before_nested_stage": True,
        },
        "claim_boundary": (
            "This is an adaptive 22-seed redevelopment screen. A feasible result "
            "only authorizes leakage-free nested selection; it does not authorize "
            "closed-loop or confirmatory claims. v84 and v85 remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        partition_path=args.partition,
        manifest_path=args.manifest,
        output_path=args.out,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "expanded_candidates": payload["training"][
                    "expanded_candidate_count"
                ],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
