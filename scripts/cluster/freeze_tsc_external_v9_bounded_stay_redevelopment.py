#!/usr/bin/env python3
"""Freeze v88 bounded-stay adaptive redevelopment."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (  # noqa: E402
    candidate_payload,
    redevelopment_candidates,
    stay_aware_execution_candidate,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_phase_transition_aware_redevelopment import (  # noqa: E402
    AUDIT_PROTOCOL as V87_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (  # noqa: E402
    PHASE_POLICY,
    PROTOCOL as V86_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_phase_transition_aware_redevelopment import (  # noqa: E402
    ACTION_AWARE_POLICY,
    PROTOCOL as V87_PROTOCOL,
)


PROTOCOL = "tsc-v88r84-external-v9-bounded-stay-redevelopment-v1"
V86_AUDIT_PROTOCOL = "tsc-v86r82-interference-aware-redevelopment-audit-v1"
V86_REJECTION = "retain_phase_pressure_and_reject_v86_redevelopment"
V87_DECISION = "freeze_selected_candidate_for_fresh_independent_confirmation"
LEGACY_POLICY = "uh_global_h120_m1"
LOCAL_REFERENCE_POLICY = "uh_local_h120_m1"
BOUNDED_STAY_POLICY = "uh_global_h120_stay_budget1"
BOUNDED_STAY_ROLE = "bounded_stay_budget_adaptive_candidate"
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def freeze_protocol(
    *,
    v86_protocol_path: Path,
    v86_audit_path: Path,
    v87_protocol_path: Path,
    v87_audit_path: Path,
    remote_results_root: Path,
    local_results_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v88 protocol: {output_path}")
    v86 = _read_json(v86_protocol_path)
    v86_audit = _read_json(v86_audit_path)
    v87 = _read_json(v87_protocol_path)
    v87_audit = _read_json(v87_audit_path)
    if not (
        v86.get("protocol") == V86_PROTOCOL
        and v86_audit.get("protocol") == V86_AUDIT_PROTOCOL
        and v86_audit.get("status") == "PASS"
        and v86_audit.get("decision") == V86_REJECTION
        and v86_audit.get("protocol_sha256") == _sha256(v86_protocol_path)
        and v86_audit.get("selection_gate", {}).get("passed") is False
        and v86_audit.get("selection_gate", {}).get("selected_candidate") is None
        and v87.get("protocol") == V87_PROTOCOL
        and v87_audit.get("protocol") == V87_AUDIT_PROTOCOL
        and v87_audit.get("status") == "PASS"
        and v87_audit.get("decision") == V87_DECISION
        and v87_audit.get("protocol_sha256") == _sha256(v87_protocol_path)
        and v87_audit.get("selection_gate", {}).get("selected_candidate")
        == LEGACY_POLICY
        and ACTION_AWARE_POLICY
        not in v87_audit.get("selection_gate", {}).get(
            "eligible_candidates", ()
        )
    ):
        raise ValueError("v88 redevelopment authorization chain changed")

    seeds = tuple(int(value) for value in v87["development"]["seeds"])
    prospective = tuple(
        int(value) for value in v87["development"]["sealed_prospective_seeds"]
    )
    if not (
        len(seeds) == len(set(seeds)) == 54
        and len(prospective) == 8
        and not set(seeds).intersection(prospective)
        and v86_audit.get("scientific_status", {}).get(
            "v85_prospective_evidence_used"
        )
        is False
    ):
        raise ValueError("v88 seed partition or v85 seal changed")

    base_candidates = {
        candidate.key: candidate for candidate in redevelopment_candidates(v86)
    }
    base = base_candidates[LEGACY_POLICY]
    candidate = stay_aware_execution_candidate(
        base,
        key=BOUNDED_STAY_POLICY,
        candidate_role=BOUNDED_STAY_ROLE,
        max_stay_overrides_per_global_cooldown=1,
    )
    if not (
        candidate.execution_trust_region.allow_stay_during_global_cooldown
        and candidate.execution_trust_region.max_stay_overrides_per_global_cooldown
        == 1
        and candidate.execution_trust_region.global_cooldown_intervals == 11
        and candidate.execution_trust_region.max_simultaneous_overrides == 1
        and candidate.coordination_mode == "direct"
        and candidate.cooldown_intervals == 0
        and candidate.artifact_key == "state_action_h120_r025_t010"
        and candidate.prediction_horizon_sec == 120
    ):
        raise ValueError("v88 bounded-stay candidate changed")

    payload = copy.deepcopy(v87)
    payload.update(
        {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "post_v87_bounded_stay_adaptive_redevelopment",
            "redevelopment_parent": {
                "v86_protocol": _path_record(v86_protocol_path),
                "v86_independent_audit": _path_record(v86_audit_path),
                "v86_decision": V86_REJECTION,
                "v87_protocol": _path_record(v87_protocol_path),
                "v87_independent_audit": _path_record(v87_audit_path),
                "v87_decision": V87_DECISION,
                "v87_selected_candidate": LEGACY_POLICY,
                "v87_rejected_candidate": ACTION_AWARE_POLICY,
                "v86_results_root": str(v86_audit["results_root"]),
            },
            "development": {
                "classification": "adaptive_development_only_never_confirmatory",
                "city_scenarios": {"jinan": ["jinan_3x4_real"]},
                "seeds": list(seeds),
                "sealed_prospective_seeds": list(prospective),
                "rollout_policies": [BOUNDED_STAY_POLICY],
                "reference_policies": [
                    PHASE_POLICY,
                    LEGACY_POLICY,
                    LOCAL_REFERENCE_POLICY,
                    ACTION_AWARE_POLICY,
                ],
                "selection_eligible_policies": [
                    LEGACY_POLICY,
                    BOUNDED_STAY_POLICY,
                ],
                "phase_policy": PHASE_POLICY,
                "legacy_policy": LEGACY_POLICY,
                "bounded_stay_policy": BOUNDED_STAY_POLICY,
                "unbounded_stay_reference_policy": ACTION_AWARE_POLICY,
                "method_candidate": candidate_payload(candidate),
                "candidate_count": 1,
                "new_rollout_count": len(seeds),
                "pairing_unit": "same_scenario_same_simulator_seed",
                "primary_metric": "mean_tripinfo_waiting_time",
                "selection_rule": {
                    "maximum_equal_seed_mean_relative_delta_vs_phase": 0.0,
                    "maximum_bootstrap_95pct_upper_mean_relative_delta_vs_phase": 0.0,
                    "maximum_worst_seed_relative_delta_vs_phase": 0.05,
                    "minimum_improved_seed_fraction_vs_phase": 0.5,
                    "minimum_active_seed_fraction": 0.75,
                    "method_teleports_must_not_exceed_phase_in_aggregate": True,
                    "method_collision_incidents_must_not_exceed_phase_in_aggregate": True,
                    "phase_execution_accounting_must_close": True,
                    "discordant_collision_pairs_reported_not_seedwise_dominance_gate": True,
                    "fallback": "phase_pressure_if_no_candidate_is_feasible",
                    "selector": (
                        "lowest_mean_relative_delta_then_lower_bootstrap_upper_"
                        "then_fewer_effective_overrides"
                    ),
                },
                "safety_estimand": {
                    "primary_unit": "aggregate_incidents_over_equal_seed_demand_matrix",
                    "paired_diagnostic": "baseline_only_and_method_only_seed_incidents",
                    "rationale": (
                        "A control intervention changes microscopic arrival and vehicle "
                        "trajectories, so incident identity within one seed is not a stable "
                        "unit-level counterfactual. Aggregate incidence remains comparable "
                        "under the equal-seed demand matrix."
                    ),
                    "v86_original_seedwise_gate_is_not_reinterpreted": True,
                },
                "bootstrap": {
                    "replicates": 10000,
                    "seed": 20260830,
                    "unit": "paired_simulator_seed",
                    "shared_indices_across_candidates": True,
                },
            },
            "output_roots": {
                "remote_results_root": str(remote_results_root),
                "local_results_root": str(Path(local_results_root).resolve()),
            },
            "scientific_status": {
                "classification": "adaptive_redevelopment_after_v86_rejection",
                "v86_outcomes_used_for_design": True,
                "v87_outcomes_used_for_design": True,
                "v86_remains_rejected_under_its_original_gate": True,
                "v88_can_never_support_confirmation": True,
                "new_independent_confirmation_required": True,
                "v85_prospective_remains_sealed": True,
                "artifact_refit": False,
                "changed_modules": [
                    "one_stay_override_budget_per_active_global_cooldown",
                ],
                "unchanged_modules": [
                    "state_action_h120_originator",
                    "risk_multiplier",
                    "minimum_context_trust",
                    "phase_pressure_prior",
                    "safe_phase_executor_clearance_logic",
                    "sumo_network_and_demand",
                    "phase_transition_only_global_cooldown",
                    "aggregate_collision_safety_estimand",
                ],
            },
            "claim_boundary": (
                "V88 is adaptive redevelopment on the same 54 outcomes already exposed "
                "through v83/v84/v86/v87. It tests one continuation veto per active "
                "global cooldown after unlimited stay bypass failed in v87. It cannot "
                "support confirmation, and the eight v85 prospective seeds remain sealed."
            ),
        }
    )
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v86-audit", type=Path, required=True)
    parser.add_argument("--v87-protocol", type=Path, required=True)
    parser.add_argument("--v87-audit", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v86_protocol_path=args.v86_protocol,
        v86_audit_path=args.v86_audit,
        v87_protocol_path=args.v87_protocol,
        v87_audit_path=args.v87_audit,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "seed_count": len(payload["development"]["seeds"]),
                "new_rollout_count": payload["development"]["new_rollout_count"],
                "policy": payload["development"]["bounded_stay_policy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
