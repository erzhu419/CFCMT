#!/usr/bin/env python3
"""Freeze the outcome-independent v63 hierarchical closed-loop protocol."""

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
from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (  # noqa: E402
    ANCHOR_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (  # noqa: E402
    AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as OFFLINE_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_target_support_equivalence_audit import (  # noqa: E402
    AUDIT_DECISION as SUPPORT_AUDIT_DECISION,
    RESULT_PROTOCOL as SUPPORT_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_protocol import (  # noqa: E402
    PROTOCOL as PARENT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


PROTOCOL = "tsc-v64r60-external-v9-hierarchical-closed-loop-development-v2"
ENVIRONMENT_PROTOCOL = "tsc-v54r50-external-v9-full-budget-seed-blocked-refit-confirmation-v1"
ENVIRONMENT_PARENT_PROTOCOL = "tsc-v53r49-external-v9-network-repair-confirmation-v1"
DEVELOPMENT_RESULT_PROTOCOL = (
    "tsc-v64r60-external-v9-hierarchical-closed-loop-development-rollout-v2"
)
PHASE_POLICY = "phase_pressure"
METHOD_POLICY_PREFIX = "cfcmt_hierarchical"
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_REPLICATES = 10000


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _file_count(path: Path) -> int:
    root = Path(path)
    return sum(item.is_file() for item in root.rglob("*")) if root.exists() else 0


def execution_candidates(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    inherited = tuple(
        dict(value)
        for value in parent["closed_loop_development"]["candidate_grid"][
            "execution_trust_regions"
        ]
    )
    candidates = [
        {
            "key": f"{METHOD_POLICY_PREFIX}_{row['key']}",
            "runtime_policy": f"{ANCHOR_FAMILY}_contrast_hierarchical_guard",
            "coordination_mode": "direct",
            "cooldown_intervals": int(row["global_cooldown_intervals"]),
            "execution_trust_region": row,
        }
        for row in inherited
    ]
    keys = [str(row["key"]) for row in candidates]
    regions = [
        json.dumps(row["execution_trust_region"], sort_keys=True)
        for row in candidates
    ]
    if (
        len(candidates) != 12
        or len(set(keys)) != len(candidates)
        or len(set(regions)) != len(candidates)
        or any(
            not bool(row["execution_trust_region"]["enabled"])
            or int(row["execution_trust_region"]["max_simultaneous_overrides"])
            != 1
            or int(row["execution_trust_region"]["global_cooldown_intervals"])
            <= 0
            for row in candidates
        )
    ):
        raise ValueError("hierarchical execution-only candidate grid changed")
    return candidates


def build_protocol(
    *,
    parent_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    freeze_audit_path: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
    future_results_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    parent = _read_json(parent_path)
    offline = _read_json(offline_result_path)
    support = _read_json(support_audit_path)
    freeze_audit = _read_json(freeze_audit_path)
    environment = _read_json(environment_protocol_path)
    environment_parent = _read_json(environment_parent_path)
    if (
        parent.get("protocol") != PARENT_PROTOCOL
        or offline.get("protocol") != OFFLINE_RESULT_PROTOCOL
        or offline.get("decision") != AUTHORIZATION_DECISION
        or not bool(offline.get("offline_confirmation_gate", {}).get("passed", False))
        or offline.get("frozen_protocol_sha256") != _sha256(parent_path)
        or support.get("protocol") != SUPPORT_AUDIT_PROTOCOL
        or support.get("status") != "PASS"
        or support.get("decision") != SUPPORT_AUDIT_DECISION
        or support.get("offline_result_sha256") != _sha256(offline_result_path)
        or environment.get("protocol") != ENVIRONMENT_PROTOCOL
        or environment_parent.get("protocol") != ENVIRONMENT_PARENT_PROTOCOL
        or _sha256(environment_parent_path)
        != str(environment["parent_protocol"]["sha256"])
        or _sha256(external_manifest_path)
        != str(environment_parent["counterfactual_cache"]["manifest_sha256"])
        or _sha256(freeze_audit_path)
        != str(parent["hierarchical_joint_audit"]["sha256"])
        or freeze_audit.get("status") != "PASS"
        or not bool(freeze_audit.get("integrity_gate", {}).get("passed", False))
        or _file_count(future_results_root) != 0
    ):
        raise ValueError("hierarchical closed-loop amendment evidence changed")
    conversion_manifest_sha = _sha256(conversion_manifest_path)
    conversion_tree_sha = _tree_sha256(conversion_root)
    if conversion_manifest_sha != "b140f2deea63653efd225c53ad306bf910ecf1ee149fb894c202227d7d454f56":
        raise ValueError("hierarchical closed-loop conversion manifest changed")
    candidates = execution_candidates(parent)
    scenarios = {
        str(city): [str(value) for value in values]
        for city, values in parent["city_scenarios"].items()
    }
    development = dict(parent["closed_loop_development"])
    seeds = tuple(int(value) for value in development["seeds"])
    scenario_count = sum(len(values) for values in scenarios.values())
    policy_count = len(candidates) + 1
    matrix_size = len(seeds) * scenario_count * policy_count
    if matrix_size != 416:
        raise ValueError("hierarchical development matrix changed")
    environment_cache = dict(environment_parent["counterfactual_cache"])
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "scientific_amendment": {
            "status": "pre_closed_loop_outcome_independent_protocol_correction",
            "closed_loop_result_file_count_at_freeze": 0,
            "reason": (
                "The inherited two base-candidate labels encoded alternative "
                "pressure blend weights. Replaying them would mutate the numeric "
                "gate confirmed by v62. They are removed before any closed-loop "
                "outcome is generated; the 12 already-declared observable-state "
                "execution regions are retained without numeric change."
            ),
            "removed_semantic_duplicates": list(
                parent["closed_loop_development"]["candidate_grid"][
                    "base_candidates"
                ]
            ),
            "model_gate_support_and_thresholds_immutable": True,
            "no_development_validation_or_prospective_outcome_inspected": True,
        },
        "parent_protocol": {
            "path": _reference(parent_path),
            "sha256": _sha256(parent_path),
        },
        "offline_authorization": {
            "path": _reference(offline_result_path),
            "sha256": _sha256(offline_result_path),
            "decision": AUTHORIZATION_DECISION,
        },
        "support_runtime_equivalence": {
            "path": _reference(support_audit_path),
            "sha256": _sha256(support_audit_path),
            "decision": SUPPORT_AUDIT_DECISION,
        },
        "hierarchical_freeze_audit": {
            "path": _reference(freeze_audit_path),
            "sha256": _sha256(freeze_audit_path),
        },
        "city_artifacts": parent["city_artifacts"],
        "environment": {
            "protocol": {
                "path": _reference(environment_protocol_path),
                "sha256": _sha256(environment_protocol_path),
            },
            "network_parent_protocol": {
                "path": _reference(environment_parent_path),
                "sha256": _sha256(environment_parent_path),
            },
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
            "sumo_version": "1.22.0",
            "duration_sec": int(development["duration_sec"]),
            "control_interval_sec": int(environment_cache["control_interval_sec"]),
            "warmup_sec": int(environment_cache["warmup_sec"]),
            "prediction_horizon_sec": int(
                environment_cache["control_interval_sec"]
                * environment_cache["counterfactual_horizon_intervals"]
            ),
        },
        "development": {
            "seeds": list(seeds),
            "city_scenarios": scenarios,
            "policies": [PHASE_POLICY, *[row["key"] for row in candidates]],
            "phase_policy": PHASE_POLICY,
            "method_candidates": candidates,
            "candidate_count": len(candidates),
            "policy_count": policy_count,
            "matrix_size": matrix_size,
            "reporting_unit": "city",
            "scenario_weighting_within_city": "equal",
            "seed_weighting": "equal",
            "common_execution_candidate_across_cities": True,
            "selection_rule": {
                "minimum_city_mean_relative_improvement": 0.005,
                "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.01,
                "maximum_seed_level_city_mean_relative_regression": 0.01,
                "minimum_improved_seed_fraction": 0.5,
                "minimum_executed_interventions_per_scenario_seed": 1,
                "method_teleports_must_not_exceed_paired_phase_pressure": True,
                "method_collision_incidents_must_not_exceed_paired_phase_pressure": True,
                "robust_score": (
                    "city_mean_relative_delta + 0.5 * seed_delta_std + "
                    "max(worst_seed_delta, 0)"
                ),
                "global_selector": "lowest_equal_city_mean_robust_score_among_all_city_feasible_candidates",
                "tie_break": (
                    "lower global robust score, lower equal-city mean delta, "
                    "fewer interventions, longer cooldown, lexical key"
                ),
                "fallback": "phase_pressure_if_no_common_candidate_is_feasible",
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired simulator seed with equal scenario weighting within city",
            },
        },
        "validation": parent["closed_loop_validation"],
        "prospective_confirmation": parent["prospective_confirmation"],
        "claim_boundary": (
            "This protocol can select one execution-only trust region on disclosed "
            "development seeds. It cannot support validation, prospective, or "
            "general efficacy claims."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--future-results-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite closed-loop protocol: {args.out}")
    payload = build_protocol(
        parent_path=args.parent,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        freeze_audit_path=args.freeze_audit,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
        future_results_root=args.future_results_root,
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
                "candidate_count": payload["development"]["candidate_count"],
                "matrix_size": payload["development"]["matrix_size"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
