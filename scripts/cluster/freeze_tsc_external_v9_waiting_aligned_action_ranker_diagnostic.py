#!/usr/bin/env python3
"""Freeze the post-veto waiting-aligned direct action-ranker diagnostic."""

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
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_veto_diagnostic import (  # noqa: E402
    RESULT_PROTOCOL as VETO_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.action_ranker import (  # noqa: E402
    CAUSAL_CORE_RANKING_PARENTS,
    PAIRWISE_CAUSAL_ACTION_PARENTS,
    PAIRWISE_CAUSAL_STATE_PARENTS,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_veto_diagnostic import (  # noqa: E402
    PROTOCOL as VETO_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-action-ranker-diagnostic-v1"
PARTITION_PROTOCOL = "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"
PARENT_PROTOCOL = "tsc-v64r60-external-v9-hierarchical-closed-loop-development-v2"
VETO_REJECTION_DECISION = "retain_phase_pressure_and_reject_waiting_aligned_veto"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def build_protocol(
    *,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    veto_protocol_path: Path,
    veto_result_path: Path,
    partition_path: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    manifest_path: Path,
    diagnostic_result_root: Path,
    artifact_root: Path,
    closed_loop_root: Path,
    confirmatory_root: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    cache = _read_json(cache_protocol_path)
    audit = _read_json(cache_audit_path)
    veto_protocol = _read_json(veto_protocol_path)
    veto_result = _read_json(veto_result_path)
    partition = _read_json(partition_path)
    parent = _read_json(proposal_parent_protocol_path)
    parent_audit = _read_json(hierarchical_freeze_audit_path)
    if (
        cache.get("protocol") != CACHE_PROTOCOL
        or audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision")
        != "authorize_waiting_aligned_seed_blocked_oof_training"
        or audit.get("gate", {}).get("passed") is not True
        or veto_protocol.get("protocol") != VETO_PROTOCOL
        or veto_result.get("protocol") != VETO_RESULT_PROTOCOL
        or veto_result.get("status") != "PASS"
        or veto_result.get("decision") != VETO_REJECTION_DECISION
        or veto_result.get("diagnostic_protocol_sha256")
        != _sha256(veto_protocol_path)
        or partition.get("protocol") != PARTITION_PROTOCOL
        or parent.get("protocol") != PARENT_PROTOCOL
        or parent_audit.get("status") != "PASS"
    ):
        raise ValueError("waiting-aligned action-ranker parent evidence changed")

    collection = dict(cache["collection"])
    redevelopment_seeds = [
        int(value) for value in partition["redevelopment"]["seeds"]
    ]
    if (
        collection.get("scenario") != "jinan_3x4_real"
        or collection.get("counterfactual_cost_mode") != "halted_queue"
        or [int(value) for value in collection["seeds"]] != redevelopment_seeds
        or int(audit.get("total_groups", 0)) != 19851
        or int(audit.get("total_rows", 0)) != 158808
    ):
        raise ValueError("waiting-aligned action-ranker collection changed")
    if (
        Path(confirmatory_root).resolve()
        != Path(partition["confirmatory"]["root"]).resolve()
        or Path(prospective_root).resolve()
        != Path(partition["prospective"]["root"]).resolve()
    ):
        raise ValueError("sealed validation roots differ from the seed partition")

    future_roots = [
        Path(diagnostic_result_root),
        Path(artifact_root),
        Path(closed_loop_root),
        Path(confirmatory_root),
        Path(prospective_root),
    ]
    if any(path.exists() for path in future_roots):
        raise ValueError("action-ranker diagnostic or sealed output root already exists")

    model_candidates = [
        {
            "key": "causal_antisymmetric_pairwise",
            "family": "antisymmetric_pairwise",
            "role": "primary_all_action_ranker",
            "config": {
                "learning_rate": 0.05,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 10.0,
                "uncertainty_quantile": 0.9,
                "random_state": 20260803,
            },
            "state_feature_names": list(PAIRWISE_CAUSAL_STATE_PARENTS),
            "action_feature_names": list(PAIRWISE_CAUSAL_ACTION_PARENTS),
        },
        {
            "key": "causal_group_normalized_advantage",
            "family": "group_normalized_advantage",
            "role": "reference_delta_ablation",
            "config": {
                "learning_rate": 0.05,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 12,
                "l2_regularization": 10.0,
                "target_clip": 5.0,
                "uncertainty_quantile": 0.9,
                "random_state": 20260803,
                "candidate_only": False,
                "balance_candidate_signs": False,
                "max_sign_weight_multiplier": 4.0,
                "target_normalization_protocol": "group_action_range_v1",
            },
            "causal_feature_names": list(CAUSAL_CORE_RANKING_PARENTS),
        },
    ]
    gate_candidates = [
        {
            "risk_multiplier": risk,
            "minimum_context_trust": trust,
        }
        for risk in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
        for trust in (0.0, 0.5, 0.75, 0.9)
    ]
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_veto_adaptive_development_seed_blocked_direct_action_ranking",
        "adaptive_status": {
            "trigger": VETO_REJECTION_DECISION,
            "pre_registered_before_v67_collection": False,
            "disclosure": (
                "The direct ranker was designed after the proposal-veto diagnostic "
                "failed. It is development evidence only; final efficacy inference "
                "must use the untouched v84 seeds."
            ),
        },
        "frozen_inputs": {
            "cache_protocol_path": str(Path(cache_protocol_path).resolve()),
            "cache_protocol_sha256": _sha256(cache_protocol_path),
            "cache_audit_path": str(Path(cache_audit_path).resolve()),
            "cache_audit_sha256": _sha256(cache_audit_path),
            "veto_protocol_path": str(Path(veto_protocol_path).resolve()),
            "veto_protocol_sha256": _sha256(veto_protocol_path),
            "veto_result_path": str(Path(veto_result_path).resolve()),
            "veto_result_sha256": _sha256(veto_result_path),
            "partition_path": str(Path(partition_path).resolve()),
            "partition_sha256": _sha256(partition_path),
            "proposal_parent_protocol_path": str(
                Path(proposal_parent_protocol_path).resolve()
            ),
            "proposal_parent_protocol_sha256": _sha256(
                proposal_parent_protocol_path
            ),
            "hierarchical_freeze_audit_path": str(
                Path(hierarchical_freeze_audit_path).resolve()
            ),
            "hierarchical_freeze_audit_sha256": _sha256(
                hierarchical_freeze_audit_path
            ),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "training": {
            "city": "jinan",
            "scenarios": [str(collection["scenario"])],
            "seeds": redevelopment_seeds,
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
            "action_originator": "direct_ranker_over_all_eight_candidate_actions",
            "target": "450_second_global_halted_queue_integral_per_controlled_lane",
            "fit_target": "within_action_group_range_normalized_cost_difference",
            "evaluation_target": "selected_minus_phase_pressure_group_normalized_cost",
            "model_candidates": model_candidates,
        },
        "selection": {
            "gate": (
                "execute_ranker_argmin_only_if_predicted_delta_plus_risk_times_"
                "uncertainty_is_negative_and_context_trust_passes"
            ),
            "gate_candidates": gate_candidates,
            "selection_rule": {
                "minimum_retained_groups_per_city": 500,
                "minimum_retained_seed_fraction": 1.0,
                "maximum_equal_seed_scenario_mean_delta": -0.001,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.01,
                "maximum_harmful_group_fraction": 0.45,
                "fallback": "phase_pressure_if_no_model_gate_is_feasible",
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "unit": "simulator_seed",
            },
            "offline_comparators": [
                "phase_pressure_reference",
                "immutable_v60_hierarchical_cfcmt",
                "counterfactual_oracle_best_of_eight_diagnostic_only",
                "ungated_direct_ranker",
            ],
        },
        "execution": {
            "maximum_parallel_oof_folds": 8,
            "blas_threads_per_fold": 1,
        },
        "future_roots_absent_at_freeze": [
            str(path.resolve()) for path in future_roots
        ],
        "claim_boundary": (
            "This post-veto OOF run may select one successor specification using "
            "only the 22 disclosed redevelopment seeds. The 32 confirmatory and "
            "8 prospective seeds remain unread until a successor is frozen."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--veto-protocol", type=Path, required=True)
    parser.add_argument("--veto-result", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-result-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--closed-loop-root", type=Path, required=True)
    parser.add_argument("--confirmatory-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            f"refusing to overwrite action-ranker diagnostic protocol: {args.out}"
        )
    payload = build_protocol(
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        veto_protocol_path=args.veto_protocol,
        veto_result_path=args.veto_result,
        partition_path=args.partition,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
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
