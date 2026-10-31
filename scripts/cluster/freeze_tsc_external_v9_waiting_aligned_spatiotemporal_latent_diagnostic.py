#!/usr/bin/env python3
"""Freeze the post-v69 target spatiotemporal action-latent diagnostic."""

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
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (  # noqa: E402
    REJECTION_DECISION as RANKER_REJECTION_DECISION,
    RESULT_PROTOCOL as RANKER_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (  # noqa: E402
    PHASE_PAIR_SCOPE,
    TLS_PHASE_PAIR_SCOPE,
    TLS_TIME_PHASE_PAIR_SCOPE,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_action_ranker_diagnostic import (  # noqa: E402
    PARTITION_PROTOCOL,
    PROTOCOL as RANKER_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-spatiotemporal-latent-diagnostic-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _model(
    key: str,
    *,
    scope: str,
    shrinkage: float,
    time_bin_sec: int | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "family": "target_spatiotemporal_action_latent",
        "config": {
            "scope": scope,
            "shrinkage": float(shrinkage),
            "time_bin_sec": time_bin_sec,
            "uncertainty_quantile": 0.9,
        },
    }


def build_protocol(
    *,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    ranker_protocol_path: Path,
    ranker_result_path: Path,
    partition_path: Path,
    manifest_path: Path,
    diagnostic_result_root: Path,
    artifact_root: Path,
    closed_loop_root: Path,
    confirmatory_root: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    cache = _read_json(cache_protocol_path)
    audit = _read_json(cache_audit_path)
    ranker_protocol = _read_json(ranker_protocol_path)
    ranker_result = _read_json(ranker_result_path)
    partition = _read_json(partition_path)
    if (
        cache.get("protocol") != CACHE_PROTOCOL
        or audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("gate", {}).get("passed") is not True
        or ranker_protocol.get("protocol") != RANKER_PROTOCOL
        or ranker_result.get("protocol") != RANKER_RESULT_PROTOCOL
        or ranker_result.get("status") != "PASS"
        or ranker_result.get("decision") != RANKER_REJECTION_DECISION
        or ranker_result.get("integrity_gate", {}).get("passed") is not True
        or ranker_result.get("diagnostic_protocol_sha256")
        != _sha256(ranker_protocol_path)
        or partition.get("protocol") != PARTITION_PROTOCOL
    ):
        raise ValueError("spatiotemporal latent parent evidence changed")
    collection = dict(cache["collection"])
    seeds = [int(value) for value in partition["redevelopment"]["seeds"]]
    if (
        collection.get("counterfactual_cost_mode") != "halted_queue"
        or collection.get("scenario") != "jinan_3x4_real"
        or [int(value) for value in collection["seeds"]] != seeds
        or int(ranker_result.get("action_group_count", 0))
        != int(audit.get("total_groups", -1))
    ):
        raise ValueError("spatiotemporal latent dataset changed")
    if (
        Path(confirmatory_root).resolve()
        != Path(partition["confirmatory"]["root"]).resolve()
        or Path(prospective_root).resolve()
        != Path(partition["prospective"]["root"]).resolve()
    ):
        raise ValueError("spatiotemporal sealed roots changed")
    future_roots = [
        Path(diagnostic_result_root),
        Path(artifact_root),
        Path(closed_loop_root),
        Path(confirmatory_root),
        Path(prospective_root),
    ]
    if any(path.exists() for path in future_roots):
        raise ValueError("spatiotemporal latent or sealed output already exists")

    models = [
        _model("phase_pair", scope=PHASE_PAIR_SCOPE, shrinkage=0.0),
        _model("tls_phase_pair_s20", scope=TLS_PHASE_PAIR_SCOPE, shrinkage=20.0),
        _model(
            "tls_time_300s_s20",
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            shrinkage=20.0,
            time_bin_sec=300,
        ),
        _model(
            "tls_time_600s_s20",
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            shrinkage=20.0,
            time_bin_sec=600,
        ),
        _model(
            "tls_time_900s_s20",
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            shrinkage=20.0,
            time_bin_sec=900,
        ),
        _model(
            "tls_time_600s_s5",
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            shrinkage=5.0,
            time_bin_sec=600,
        ),
        _model(
            "tls_time_600s_s100",
            scope=TLS_TIME_PHASE_PAIR_SCOPE,
            shrinkage=100.0,
            time_bin_sec=600,
        ),
    ]
    gates = [
        {"risk_multiplier": risk, "minimum_context_trust": trust}
        for risk in (0.0, 0.25, 0.5, 1.0)
        for trust in (0.0, 0.25, 0.5, 0.75)
    ]
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v69_adaptive_target_spatiotemporal_mechanism_latent",
        "adaptive_status": {
            "trigger": RANKER_REJECTION_DECISION,
            "pre_registered_before_v67_collection": False,
            "classification": "target_offline_adaptation_few_shot",
            "disclosure": (
                "The target latent was designed after v69 showed no invariant "
                "state-only signal. Selection remains development-only and requires "
                "fresh v84 confirmation."
            ),
        },
        "frozen_inputs": {
            "cache_protocol_path": str(Path(cache_protocol_path).resolve()),
            "cache_protocol_sha256": _sha256(cache_protocol_path),
            "cache_audit_path": str(Path(cache_audit_path).resolve()),
            "cache_audit_sha256": _sha256(cache_audit_path),
            "ranker_protocol_path": str(Path(ranker_protocol_path).resolve()),
            "ranker_protocol_sha256": _sha256(ranker_protocol_path),
            "ranker_result_path": str(Path(ranker_result_path).resolve()),
            "ranker_result_sha256": _sha256(ranker_result_path),
            "partition_path": str(Path(partition_path).resolve()),
            "partition_sha256": _sha256(partition_path),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "training": {
            "city": "jinan",
            "scenarios": [str(collection["scenario"])],
            "seeds": seeds,
            "sealed_confirmatory_seeds": [
                int(value) for value in partition["confirmatory"]["seeds"]
            ],
            "sealed_prospective_seeds": [
                int(value) for value in partition["prospective"]["seeds"]
            ],
            "collection_shards": int(collection["collection_shards_per_seed"]),
            "expected_action_count_per_group": 8,
            "cross_fitting": "leave_one_simulator_seed_out_22_folds",
            "reference_policy": "phase_pressure",
            "target": "450_second_global_halted_queue_integral_per_controlled_lane",
            "fit_target": "within_action_group_range_normalized_cost_difference",
            "latent_hierarchy": [
                "transferable_reference_phase_to_candidate_phase_prior",
                "target_tls_residual",
                "target_tls_time_bin_residual",
            ],
            "model_candidates": models,
        },
        "selection": {
            "gate_candidates": gates,
            "selection_rule": {
                "minimum_retained_groups_per_city": 500,
                "minimum_retained_seed_fraction": 1.0,
                "maximum_equal_seed_scenario_mean_delta": -0.001,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.01,
                "maximum_harmful_group_fraction": 0.45,
                "fallback": "phase_pressure_if_no_latent_gate_is_feasible",
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "unit": "simulator_seed",
            },
        },
        "execution": {
            "parallel_oof_folds": 8,
            "executor": "linux_fork_process_pool",
            "blas_threads_per_process": 1,
        },
        "future_roots_absent_at_freeze": [
            str(path.resolve()) for path in future_roots
        ],
        "claim_boundary": (
            "The latent uses target simulator labels and is few-shot adaptation, "
            "not zero-shot transfer. Only the 22 redevelopment seeds may be read "
            "before a successor is frozen."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--ranker-protocol", type=Path, required=True)
    parser.add_argument("--ranker-result", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-result-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--closed-loop-root", type=Path, required=True)
    parser.add_argument("--confirmatory-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite latent diagnostic: {args.out}")
    payload = build_protocol(
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        ranker_protocol_path=args.ranker_protocol,
        ranker_result_path=args.ranker_result,
        partition_path=args.partition,
        manifest_path=args.manifest,
        diagnostic_result_root=args.diagnostic_result_root,
        artifact_root=args.artifact_root,
        closed_loop_root=args.closed_loop_root,
        confirmatory_root=args.confirmatory_root,
        prospective_root=args.prospective_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "folds": len(payload["training"]["seeds"]),
                "models": len(payload["training"]["model_candidates"]),
                "gate_candidates": len(payload["selection"]["gate_candidates"]),
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
