#!/usr/bin/env python3
"""Freeze the post-v64 long-horizon target-veto successor protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256  # noqa: E402
from scripts.cluster.audit_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    AUDIT_PROTOCOL as V64_AUDIT_PROTOCOL,
    REJECTION_DECISION as V64_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_closed_loop_protocol import (  # noqa: E402
    PROTOCOL as PARENT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


PROTOCOL = "tsc-v65r61-external-v9-long-horizon-target-veto-cache-v1"
MODEL_FAMILY = "causal_target_only"
REFERENCE_POLICY = "phase_pressure"
COLLECTION_SEEDS = (80314, 88625, 27178, 77524, 63775, 50945)
RISK_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0)
MIN_CONTEXT_TRUSTS = (0.0, 0.25, 0.5, 0.75)
CLOSED_LOOP_COOLDOWN_INTERVALS = (11, 29, 44)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260817
CONVERSION_MANIFEST_SHA256 = (
    "b140f2deea63653efd225c53ad306bf910ecf1ee149fb894c202227d7d454f56"
)


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _file_count(path: Path) -> int:
    root = Path(path)
    return sum(item.is_file() for item in root.rglob("*")) if root.exists() else 0


def veto_gate_candidates() -> list[dict[str, float | str]]:
    rows = [
        {
            "key": f"r{str(risk).replace('.', 'p')}_t{str(trust).replace('.', 'p')}",
            "risk_multiplier": float(risk),
            "min_context_trust": float(trust),
        }
        for risk in RISK_MULTIPLIERS
        for trust in MIN_CONTEXT_TRUSTS
    ]
    if len(rows) != 20 or len({str(row["key"]) for row in rows}) != 20:
        raise RuntimeError("long-horizon veto gate grid is not unique")
    return rows


def closed_loop_candidates() -> list[dict[str, Any]]:
    rows = [
        {
            "key": f"cfcmt_target_veto_cd{(intervals + 1) * 10}",
            "runtime_policy": "causal_group_normalized_rigid_advantage_contrast_hierarchical_guard",
            "coordination_mode": "direct",
            "cooldown_intervals": int(intervals),
            "max_simultaneous_overrides": 1,
        }
        for intervals in CLOSED_LOOP_COOLDOWN_INTERVALS
    ]
    if len(rows) != 3 or len({str(row["key"]) for row in rows}) != 3:
        raise RuntimeError("long-horizon closed-loop grid is not unique")
    return rows


def build_protocol(
    *,
    parent_path: Path,
    v64_audit_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
    future_cache_root: Path,
    future_training_root: Path,
    future_closed_loop_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    parent = _read_json(parent_path)
    v64_audit = _read_json(v64_audit_path)
    if (
        parent.get("protocol") != PARENT_PROTOCOL
        or v64_audit.get("protocol") != V64_AUDIT_PROTOCOL
        or v64_audit.get("status") != "PASS"
        or v64_audit.get("decision") != V64_REJECTION_DECISION
        or not bool(v64_audit.get("integrity_gate", {}).get("passed", False))
        or v64_audit.get("development_protocol_sha256") != _sha256(parent_path)
        or sorted(int(value) for value in v64_audit["untouched_validation_seeds"])
        != sorted(int(value) for value in parent["validation"]["closed_loop_seeds"])
        or sorted(int(value) for value in v64_audit["untouched_prospective_seeds"])
        != sorted(
            int(value)
            for value in parent["prospective_confirmation"]["closed_loop_seeds"]
        )
    ):
        raise ValueError("v64 falsification evidence or seed seal changed")
    future_counts = {
        "cache": _file_count(future_cache_root),
        "training": _file_count(future_training_root),
        "closed_loop": _file_count(future_closed_loop_root),
    }
    if any(future_counts.values()):
        raise ValueError(f"v65 future output already exists: {future_counts}")
    if _sha256(external_manifest_path) != str(
        parent["environment"]["external_manifest"]["sha256"]
    ):
        raise ValueError("v65 external manifest changed")
    conversion_manifest_sha = _sha256(conversion_manifest_path)
    conversion_tree_sha = _tree_sha256(conversion_root)
    if (
        conversion_manifest_sha != CONVERSION_MANIFEST_SHA256
        or conversion_manifest_sha
        != str(parent["environment"]["conversion_manifest"]["sha256"])
        or conversion_tree_sha
        != str(parent["environment"]["conversion_tree_sha256"])
    ):
        raise ValueError("v65 conversion evidence changed")
    scenarios = {
        str(city): [str(value) for value in values]
        for city, values in parent["development"]["city_scenarios"].items()
    }
    flat_scenarios = [value for values in scenarios.values() for value in values]
    gate_grid = veto_gate_candidates()
    rollout_grid = closed_loop_candidates()
    development_seeds = [int(value) for value in parent["development"]["seeds"]]
    development_matrix_size = (
        len(development_seeds)
        * len(flat_scenarios)
        * (len(rollout_grid) + 1)
    )
    if (
        len(COLLECTION_SEEDS) != 6
        or len(set(COLLECTION_SEEDS)) != 6
        or set(COLLECTION_SEEDS) & set(development_seeds)
        or set(COLLECTION_SEEDS)
        & {int(value) for value in parent["validation"]["closed_loop_seeds"]}
        or set(COLLECTION_SEEDS)
        & {
            int(value)
            for value in parent["prospective_confirmation"]["closed_loop_seeds"]
        }
        or len(flat_scenarios) != 4
        or development_matrix_size != 128
    ):
        raise ValueError("v65 seed or scenario partition changed")

    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "scientific_status": {
            "stage": "post_v64_falsification_pre_v65_outcome_successor_freeze",
            "v64_failure_interpretation": (
                "A 60-second action-level estimand did not survive repeated "
                "3600-second deployment; observable execution thresholds were "
                "insufficient to repair the horizon mismatch."
            ),
            "successor_hypothesis": (
                "Retain the cross-city short-horizon CFCMT proposal, but permit "
                "execution only when a target-city 300-second recovery-value head "
                "predicts nonpositive cost with a frozen uncertainty/support gate."
            ),
            "no_v65_cache_training_or_closed_loop_outcome_exists": True,
            "future_file_counts_at_freeze": future_counts,
        },
        "parent_protocol": {
            "path": _reference(parent_path),
            "sha256": _sha256(parent_path),
        },
        "v64_falsification_audit": {
            "path": _reference(v64_audit_path),
            "sha256": _sha256(v64_audit_path),
            "protocol": V64_AUDIT_PROTOCOL,
            "decision": V64_REJECTION_DECISION,
        },
        "environment": {
            **dict(parent["environment"]),
            "external_manifest": {
                "path": _reference(external_manifest_path),
                "sha256": _sha256(external_manifest_path),
            },
            "conversion_root": str(Path(conversion_root).resolve()),
            "conversion_manifest": {
                "path": _reference(conversion_manifest_path),
                "sha256": conversion_manifest_sha,
            },
            "conversion_tree_sha256": conversion_tree_sha,
        },
        "city_scenarios": scenarios,
        "proposal": {
            "method": "frozen_v60_hierarchical_cfcmt",
            "target_labels_used_by_proposal": False,
            "city_artifacts": parent["city_artifacts"],
            "prediction_horizon_sec": int(
                parent["environment"]["prediction_horizon_sec"]
            ),
            "immutability": (
                "Model, target-support objects, regularizer, conformal guard, and "
                "hierarchical layer order are loaded byte-for-byte from v60."
            ),
        },
        "long_horizon_collection": {
            "seeds": list(COLLECTION_SEEDS),
            "duration_sec": 3600,
            "warmup_sec": 60,
            "control_interval_sec": 10,
            "counterfactual_horizon_intervals": 30,
            "rollout_value_horizon_sec": 300,
            "max_focal_tls": 4,
            "collection_shards_per_scenario_seed": 16,
            "workers_per_seed": 64,
            "behavior_policy": REFERENCE_POLICY,
            "reference_action": REFERENCE_POLICY,
            "scenario_count": len(flat_scenarios),
            "expected_cache_file_count": (
                len(COLLECTION_SEEDS) * len(flat_scenarios) * 16
            ),
            "reporting_unit": "city",
            "scenario_weighting_within_city": "equal",
            "seed_weighting_within_city": "equal",
            "full_network_and_full_benchmark_horizon": True,
        },
        "target_veto_training": {
            "classification": "target_simulator_labeled_few_shot_adaptation",
            "zero_shot_claim_permitted": False,
            "model_family": MODEL_FAMILY,
            "target_head": "dedicated_city_specific_pairwise_causal_advantage",
            "target_head_capacity": {
                "max_iter": 80,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 4,
                "l2_regularization": 16.0,
            },
            "training_target": (
                "300-second system-vehicle cost after one candidate action "
                "followed by phase-pressure recovery"
            ),
            "cross_fitting": "leave_one_simulator_seed_out_within_target_city",
            "selection_scope": (
                "The target head may veto an action proposed by frozen CFCMT; it "
                "may neither propose a new action nor replace phase pressure."
            ),
            "gate_candidates": gate_grid,
            "gate_candidate_count": len(gate_grid),
            "gate_rule": (
                "accept iff proposal action predicted_delta + risk_multiplier * "
                "uncertainty < 0 and context_trust >= min_context_trust"
            ),
            "selection_unit": "paired_action_group_equalized_within_seed_scenario",
            "city_specific_gate_permitted": True,
            "selection_rule": {
                "minimum_retained_groups_per_city": 12,
                "minimum_retained_seed_fraction": 0.6666666666666666,
                "maximum_equal_seed_scenario_mean_delta": -0.001,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.01,
                "maximum_harmful_group_fraction": 0.45,
                "robust_score": (
                    "mean_delta + 0.5 * seed_delta_std + "
                    "max(worst_seed_delta, 0)"
                ),
                "tie_break": (
                    "lower robust score, lower mean delta, lower harmful fraction, "
                    "more retained seeds, more retained groups, higher trust, "
                    "higher risk multiplier, lexical key"
                ),
                "fallback": "always_veto_to_phase_pressure_if_no_gate_is_feasible",
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "unit": "simulator_seed_with_equal_scenario_weighting_within_city",
            },
            "final_fit": "refit_selected_city_gate_head_on_all_six_collection_seeds",
        },
        "closed_loop_development": {
            "seeds": development_seeds,
            "city_scenarios": scenarios,
            "phase_policy": REFERENCE_POLICY,
            "method_candidates": rollout_grid,
            "candidate_count": len(rollout_grid),
            "policy_count": len(rollout_grid) + 1,
            "matrix_size": development_matrix_size,
            "duration_sec": 3600,
            "primary_metric": "mean_tripinfo_waiting_time",
            "reuse_v64_phase_pressure_rollouts_by_exact_identity_and_hash": True,
            "reporting_unit": "city",
            "scenario_weighting_within_city": "equal",
            "seed_weighting": "equal",
            "selection_rule": parent["development"]["selection_rule"],
            "fallback": REFERENCE_POLICY,
        },
        "validation": parent["validation"],
        "prospective_confirmation": parent["prospective_confirmation"],
        "claim_boundary": (
            "The target veto is few-shot because it consumes target-city simulator "
            "counterfactual labels. This protocol can authorize cache collection, "
            "OOF veto training, and development-seed selection only; validation "
            "and prospective claims remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--v64-audit", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--future-cache-root", type=Path, required=True)
    parser.add_argument("--future-training-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v65 protocol: {args.out}")
    payload = build_protocol(
        parent_path=args.parent,
        v64_audit_path=args.v64_audit,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
        future_cache_root=args.future_cache_root,
        future_training_root=args.future_training_root,
        future_closed_loop_root=args.future_closed_loop_root,
        frozen_at_utc=args.frozen_at_utc or datetime.now(timezone.utc).isoformat(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "cache_files": payload["long_horizon_collection"][
                    "expected_cache_file_count"
                ],
                "gate_candidates": payload["target_veto_training"][
                    "gate_candidate_count"
                ],
                "closed_loop_matrix": payload["closed_loop_development"][
                    "matrix_size"
                ],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
