#!/usr/bin/env python3
"""Freeze v70 training after the v69 450-second cache passes audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_demand_conditioned_veto_freeze import (  # noqa: E402
    TRAINING_PROTOCOL,
    combined_feature_names,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


CACHE_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v51_external_v9_"
    "estimand_aligned_450s_target_veto_cache.json"
)
CACHE_PROTOCOL = "tsc-v69r65-external-v9-estimand-aligned-450s-veto-cache-v1"
CACHE_PROTOCOL_SHA256 = "e36b86166a52841f7edf615b76af390f5736ec7603d7dd3265afd67312cb874e"
CACHE_AUDIT_PROTOCOL = "tsc-v69r65-external-v9-450s-veto-cache-audit-v1"
CACHE_AUDIT_DECISION = (
    "authorize_demand_conditioned_450s_veto_oof_training_only"
)
PARENT_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v50_external_v9_"
    "proposal_conditional_closed_loop_development.json"
)
PARENT_SHA256 = "1baf64dae82b4b1b6dcd6edbb6928f983b06ac79be65f16f964a62d558f3e0ea"


def _file_count(path: Path) -> int:
    return sum(1 for item in Path(path).rglob("*") if item.is_file()) if path.exists() else 0


def build_training_protocol(
    *, cache_audit_path: Path, output_root: Path, future_closed_loop_root: Path
) -> dict[str, Any]:
    cache_protocol = _read_json(CACHE_PROTOCOL_PATH)
    cache_audit = _read_json(cache_audit_path)
    parent = _read_json(PARENT_PATH)
    proposal_parent_path = Path(parent["proposal_parent_protocol"]["path"])
    freeze_audit_path = Path(parent["hierarchical_freeze_audit"]["path"])
    if (
        cache_protocol.get("protocol") != CACHE_PROTOCOL
        or _sha256(CACHE_PROTOCOL_PATH) != CACHE_PROTOCOL_SHA256
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision") != CACHE_AUDIT_DECISION
        or not bool(cache_audit.get("integrity_gate", {}).get("passed", False))
        or cache_audit.get("frozen_protocol_sha256") != CACHE_PROTOCOL_SHA256
        or _sha256(PARENT_PATH) != PARENT_SHA256
        or _sha256(proposal_parent_path)
        != parent["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != parent["hierarchical_freeze_audit"]["sha256"]
    ):
        raise ValueError("v70 training parent evidence changed")
    future_counts = {
        "output_root": _file_count(output_root),
        "closed_loop": _file_count(future_closed_loop_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v70 future outputs already exist: {future_counts}")
    executable_sources = (
        "cf_h2o/eval/traffic_signal_external_demand_conditioned_veto_freeze.py",
        "cf_h2o/traffic_signal/demand_conditioned_target_veto.py",
        "cf_h2o/traffic_signal/demand_schedule_context.py",
        "cf_h2o/traffic_signal/proposal_conditional_target_veto.py",
        "scripts/cluster/run_tsc_external_demand_conditioned_450s_veto_training.py",
    )
    return {
        "protocol": TRAINING_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v69_cache_audit_pre_v70_oof_training_outcome_freeze",
        "cache_protocol": {
            "path": str(CACHE_PROTOCOL_PATH),
            "sha256": _sha256(CACHE_PROTOCOL_PATH),
        },
        "evidence": {
            "cache_audit": {
                "path": str(cache_audit_path),
                "sha256": _sha256(cache_audit_path),
                "decision": cache_audit["decision"],
                "cache_sha256": cache_audit["cache_sha256"],
                "cache_file_count": cache_audit["cache_file_count"],
            },
            "v68_falsification_audit": cache_protocol["v68_falsification_audit"],
        },
        "proposal_parent_protocol": {
            "path": str(proposal_parent_path),
            "sha256": _sha256(proposal_parent_path),
        },
        "hierarchical_freeze_audit": {
            "path": str(freeze_audit_path),
            "sha256": _sha256(freeze_audit_path),
        },
        "runtime_executable_sources": {
            path: _sha256(Path(path)) for path in executable_sources
        },
        "training": {
            "classification": "target_simulator_labeled_few_shot_adaptation",
            "zero_shot_claim_permitted": False,
            "seeds": cache_protocol["long_horizon_collection"]["seeds"],
            "city_scenarios": cache_protocol["city_scenarios"],
            "cross_fitting": "leave_one_simulator_seed_out",
            "target": "450_second_group_normalized_candidate_minus_phase_cost",
            "proposal_scope": (
                "veto_only_after_frozen_v60_CFCMT_proposes_never_action_originator"
            ),
            "candidate_feature_names": list(combined_feature_names()[:16]),
            "demand_feature_names": list(combined_feature_names()[16:]),
            "combined_feature_names": list(combined_feature_names()),
            "combined_feature_count": len(combined_feature_names()),
            "scenario_id_excluded": True,
            "acceptance_quantiles": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
            "fallback_quantile": 0.1,
            "model": {
                "learning_rate": 0.05,
                "max_iter": 120,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 12,
                "l2_regularization": 24.0,
                "random_state": 20260803,
                "uncertainty_quantile": 0.9,
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260820,
                "unit": "simulator_seed_equal_scenario_mean",
            },
            "selection_rule": {
                "minimum_retained_groups_per_city": 12,
                "minimum_retained_seed_fraction": 2.0 / 3.0,
                "maximum_equal_seed_scenario_mean_delta": -0.001,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.01,
                "maximum_harmful_group_fraction": 0.45,
                "fallback": "always_veto_to_phase_pressure_if_no_gate_is_feasible",
                "tie_break": (
                    "lower_robust_score_then_mean_then_harm_then_more_groups_then_quantile"
                ),
            },
            "final_fit": (
                "refit_all_six_adaptation_seeds_after_OOF_quantile_selection"
            ),
        },
        "demand_schedule_context": cache_protocol["demand_schedule_context"],
        "closed_loop_development": cache_protocol["closed_loop_development"],
        "validation": cache_protocol["validation"],
        "prospective_confirmation": cache_protocol["prospective_confirmation"],
        "future_file_counts_at_freeze": future_counts,
        "claim_boundary": (
            "OOF target-simulator few-shot training only. Passing this stage authorizes "
            "development-seed closed-loop evaluation, not validation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "cf_h2o/config/traffic_signal_tsc_v52_external_v9_"
            "demand_conditioned_450s_veto_training.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v70 training protocol: {args.out}")
    payload = build_training_protocol(
        cache_audit_path=args.cache_audit,
        output_root=args.output_root,
        future_closed_loop_root=args.future_closed_loop_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "feature_count": payload["training"]["combined_feature_count"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
