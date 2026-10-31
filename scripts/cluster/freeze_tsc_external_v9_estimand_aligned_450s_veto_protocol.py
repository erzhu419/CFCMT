#!/usr/bin/env python3
"""Freeze the v69 450-second, demand-conditioned target-veto cache protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from cf_h2o.traffic_signal.demand_schedule_context import (  # noqa: E402
    DEMAND_SCHEDULE_FEATURE_NAMES,
    DEMAND_SCHEDULE_PROTOCOL,
    build_demand_schedule_profile,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


PROTOCOL = "tsc-v69r65-external-v9-estimand-aligned-450s-veto-cache-v1"
PARENT_PROTOCOL = (
    "tsc-v68r64-external-v9-proposal-conditional-closed-loop-development-v1"
)
PARENT_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v50_external_v9_"
    "proposal_conditional_closed_loop_development.json"
)
PARENT_SHA256 = "1baf64dae82b4b1b6dcd6edbb6928f983b06ac79be65f16f964a62d558f3e0ea"
V68_AUDIT_PATH = Path(
    "cf_h2o/results/cluster/tsc_v68r64_external_v9_"
    "proposal_conditional_closed_loop_development_20260810/"
    "development_selection_audit_v1.json"
)
V68_AUDIT_SHA256 = "6bce1e012fabfeadaaee50d87401adf1c5545601e2fd9567bcd566dc1618056d"
V68_AUDIT_PROTOCOL = (
    "tsc-v68r64-external-v9-proposal-conditional-closed-loop-"
    "development-selection-audit-v1"
)
V68_REJECTION = (
    "reject_target_veto_development_and_preserve_validation_and_prospective_seeds"
)
RESULT_ROOT = Path(
    "cf_h2o/results/cluster/tsc_v69r65_external_v9_"
    "estimand_aligned_450s_target_veto_20260810"
)


def _file_count(path: Path) -> int:
    return sum(1 for item in Path(path).rglob("*") if item.is_file()) if path.exists() else 0


def _profile_payload(parent: dict[str, Any]) -> dict[str, Any]:
    conversion_root = Path(parent["environment"]["conversion_root"])
    profiles = {}
    for city, scenarios in sorted(parent["development"]["city_scenarios"].items()):
        for scenario in scenarios:
            sumocfg = conversion_root / scenario / f"{scenario}.sumocfg"
            profile = build_demand_schedule_profile(sumocfg, horizon_sec=3600)
            profiles[str(scenario)] = {
                "city": str(city),
                "sumocfg": str(sumocfg.resolve()),
                **profile.to_dict(),
            }
    return profiles


def build_protocol(
    *,
    parent_path: Path,
    v68_audit_path: Path,
    future_cache_root: Path,
    future_training_root: Path,
    future_closed_loop_root: Path,
) -> dict[str, Any]:
    parent = _read_json(parent_path)
    v68_audit = _read_json(v68_audit_path)
    if (
        parent.get("protocol") != PARENT_PROTOCOL
        or _sha256(parent_path) != PARENT_SHA256
        or v68_audit.get("protocol") != V68_AUDIT_PROTOCOL
        or _sha256(v68_audit_path) != V68_AUDIT_SHA256
        or v68_audit.get("status") != "PASS"
        or v68_audit.get("decision") != V68_REJECTION
        or not bool(v68_audit.get("integrity_gate", {}).get("passed", False))
    ):
        raise ValueError("v69 parent evidence changed")
    validation = parent["validation"]
    prospective = parent["prospective_confirmation"]
    if (
        sorted(int(value) for value in v68_audit["untouched_validation_seeds"])
        != sorted(int(value) for value in validation["closed_loop_seeds"])
        or sorted(int(value) for value in v68_audit["untouched_prospective_seeds"])
        != sorted(int(value) for value in prospective["closed_loop_seeds"])
    ):
        raise ValueError("v69 sealed seed sets changed")
    future_counts = {
        "cache": _file_count(future_cache_root),
        "training": _file_count(future_training_root),
        "closed_loop": _file_count(future_closed_loop_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v69 future outputs already exist: {future_counts}")

    if parent["city_target_veto_artifacts"]["jinan"]["selected_gate"] is None:
        raise ValueError("v69 requires the active Jinan proposal-conditional gate")
    candidate_features = tuple(
        str(value)
        for value in parent["proposal"]["city_artifacts"]["jinan"]["deployment"]
        ["target_support"]["pressure_support"]["feature_names"]
    )
    if len(candidate_features) != 16:
        raise ValueError("v69 candidate feature contract changed")
    profiles = _profile_payload(parent)
    selected_v68 = next(
        row
        for row in v68_audit["selector"]["grid"]
        if row["key"] == "cfcmt_target_veto_cd450"
    )
    collection_seeds = tuple(
        int(value)
        for value in _read_json(Path(parent["collection_protocol"]["path"]))
        ["long_horizon_collection"]["seeds"]
    )
    shards = 32
    scenarios = sum(
        (list(values) for _, values in sorted(parent["development"]["city_scenarios"].items())),
        [],
    )
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v68_rejection_pre_450s_counterfactual_outcome_freeze",
        "parent_protocol": {
            "path": str(parent_path),
            "sha256": _sha256(parent_path),
        },
        "v68_falsification_audit": {
            "path": str(v68_audit_path),
            "sha256": _sha256(v68_audit_path),
            "decision": v68_audit["decision"],
        },
        "scientific_rationale": {
            "observed_failure": (
                "The 300-second target veto with 450-second execution spacing improved "
                "Jinan by 0.3023% but missed the frozen 0.5% development threshold."
            ),
            "failure_localization": (
                "The sustained-arrival Jinan demand profile was the only scenario-level "
                "regression; total scheduled volume was not monotone with failure."
            ),
            "successor_hypothesis": (
                "Align target labels to the 450-second execution spacing and condition "
                "the veto on label-free exogenous demand-schedule parents."
            ),
            "outcome_informed_design_disclosure": (
                "The 450-second horizon and demand-schedule parent family were selected "
                "after inspecting v68 development outcomes. They require a new development "
                "pass and cannot be called confirmatory until untouched validation."
            ),
            "v68_cd450_snapshot": selected_v68,
        },
        "environment": parent["environment"],
        "city_scenarios": parent["development"]["city_scenarios"],
        "proposal": parent["proposal"],
        "demand_schedule_context": {
            "protocol": DEMAND_SCHEDULE_PROTOCOL,
            "label_free": True,
            "scenario_id_is_not_a_feature": True,
            "horizon_sec": 3600,
            "feature_names": list(DEMAND_SCHEDULE_FEATURE_NAMES),
            "feature_count": len(DEMAND_SCHEDULE_FEATURE_NAMES),
            "profiles": profiles,
        },
        "long_horizon_collection": {
            "behavior_policy": "phase_pressure",
            "duration_sec": 3600,
            "control_interval_sec": 10,
            "warmup_sec": 60,
            "counterfactual_horizon_intervals": 45,
            "rollout_value_horizon_sec": 450,
            "max_focal_tls": 4,
            "collection_shards_per_scenario_seed": shards,
            "workers_per_seed": len(scenarios) * shards,
            "expected_cache_file_count": len(scenarios) * len(collection_seeds) * shards,
            "seeds": list(collection_seeds),
            "scenario_count": len(scenarios),
            "reference_action": "phase_pressure",
            "full_network_and_full_benchmark_horizon": True,
            "reporting_unit": "city",
            "seed_weighting_within_city": "equal",
            "scenario_weighting_within_city": "equal",
        },
        "target_veto_training": {
            "classification": "target_simulator_labeled_few_shot_adaptation",
            "zero_shot_claim_permitted": False,
            "cross_fitting": "leave_one_simulator_seed_out",
            "target": "450_second_group_normalized_candidate_minus_phase_cost",
            "scope": "veto_only_after_frozen_v60_CFCMT_proposes",
            "candidate_feature_names": list(candidate_features),
            "demand_feature_names": list(DEMAND_SCHEDULE_FEATURE_NAMES),
            "combined_feature_count": len(candidate_features)
            + len(DEMAND_SCHEDULE_FEATURE_NAMES),
            "scenario_id_excluded": True,
            "acceptance_quantiles": [0.025, 0.05, 0.075, 0.1, 0.125],
            "model": {
                "learning_rate": 0.05,
                "max_iter": 120,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 12,
                "l2_regularization": 24.0,
                "random_state": 20260803,
                "uncertainty_quantile": 0.9,
            },
            "selection_rule": {
                "minimum_retained_groups_per_city": 12,
                "minimum_retained_seed_fraction": 2.0 / 3.0,
                "maximum_equal_seed_scenario_mean_delta": -0.001,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.01,
                "maximum_harmful_group_fraction": 0.45,
                "fallback": "always_veto_to_phase_pressure_if_no_gate_is_feasible",
            },
        },
        "closed_loop_development": {
            "seeds": parent["development"]["seeds"],
            "city_scenarios": parent["development"]["city_scenarios"],
            "active_cities": parent["development"]["active_cities"],
            "fallback_cities": parent["development"]["fallback_cities"],
            "candidate": {
                "key": "cfcmt_demand_conditioned_veto_cd450",
                "cooldown_intervals": 44,
                "max_simultaneous_overrides": 1,
                "coordination_mode": "direct",
                "runtime_policy": (
                    "causal_group_normalized_rigid_advantage_contrast_hierarchical_guard"
                ),
            },
            "selection_rule": parent["development"]["selection_rule"],
            "phase_references": parent["development"]["phase_references"],
        },
        "validation": validation,
        "prospective_confirmation": prospective,
        "future_file_counts_at_freeze": future_counts,
        "claim_boundary": (
            "Target-simulator-labeled few-shot successor development only. Demand "
            "profiles are label-free, but the 450-second veto labels are not. Validation "
            "and prospective seeds remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT_PATH)
    parser.add_argument("--v68-audit", type=Path, default=V68_AUDIT_PATH)
    parser.add_argument(
        "--future-cache-root", type=Path, default=RESULT_ROOT / "long_horizon_cache"
    )
    parser.add_argument("--future-training-root", type=Path, default=RESULT_ROOT / "training")
    parser.add_argument("--future-closed-loop-root", type=Path, default=RESULT_ROOT / "closed_loop")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "cf_h2o/config/traffic_signal_tsc_v51_external_v9_"
            "estimand_aligned_450s_target_veto_cache.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v69 protocol: {args.out}")
    payload = build_protocol(
        parent_path=args.parent,
        v68_audit_path=args.v68_audit,
        future_cache_root=args.future_cache_root,
        future_training_root=args.future_training_root,
        future_closed_loop_root=args.future_closed_loop_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "cache_files": payload["long_horizon_collection"][
                    "expected_cache_file_count"
                ],
                "workers_per_seed": payload["long_horizon_collection"][
                    "workers_per_seed"
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
