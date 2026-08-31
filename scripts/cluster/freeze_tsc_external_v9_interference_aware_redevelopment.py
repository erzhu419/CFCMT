#!/usr/bin/env python3
"""Freeze post-v84 interference-aware adaptive redevelopment."""

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
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    HIERARCHICAL_RUNTIME_POLICY,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402


PROTOCOL = "tsc-v86r82-external-v9-interference-aware-redevelopment-v1"
V83_PROTOCOL = "tsc-v83r79-external-v9-uncertainty-horizon-closed-loop-v1"
V83_AUDIT_PROTOCOL = "tsc-v83r79-uncertainty-horizon-closed-loop-audit-v2"
V84_PROTOCOL = "tsc-v84r80-external-v9-uncertainty-horizon-confirmation-v1"
V84_AUDIT_PROTOCOL = "tsc-v84r80-uncertainty-horizon-confirmation-audit-v2"
V83_AUTHORIZATION = "authorize_uncertainty_horizon_confirmatory_freeze"
V84_REJECTION = "retain_phase_pressure_and_reject_v84_confirmation"
PHASE_POLICY = "phase_pressure"
ARTIFACT_KEY = "state_action_h120_r025_t010"
CANDIDATE_SPECS = (
    # key, coordination mode, local cooldown, global cooldown, maximum simultaneous
    ("uh_global_h120_m1", "direct", 0, 11, 1),
    ("uh_local_h120_m1", "sparse", 11, 0, 1),
    ("uh_local_h120_m2", "sparse", 11, 0, 2),
    ("uh_local_h120_graph", "sparse", 11, 0, None),
)
METHOD_KEYS = tuple(row[0] for row in CANDIDATE_SPECS)
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def freeze_protocol(
    *,
    v83_protocol_path: Path,
    v83_audit_path: Path,
    v84_protocol_path: Path,
    v84_audit_path: Path,
    remote_results_root: Path,
    local_results_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v86 protocol: {output_path}")
    v83 = _read_json(v83_protocol_path)
    v83_audit = _read_json(v83_audit_path)
    v84 = _read_json(v84_protocol_path)
    v84_audit = _read_json(v84_audit_path)
    if not (
        v83.get("protocol") == V83_PROTOCOL
        and v83_audit.get("protocol") == V83_AUDIT_PROTOCOL
        and v83_audit.get("status") == "PASS"
        and v83_audit.get("decision") == V83_AUTHORIZATION
        and v83_audit.get("protocol_sha256") == _sha256(v83_protocol_path)
        and v83_audit.get("advance_gate", {}).get("selected_policy")
        == "uh_state_action_h120"
        and v84.get("protocol") == V84_PROTOCOL
        and v84_audit.get("protocol") == V84_AUDIT_PROTOCOL
        and v84_audit.get("status") == "PASS"
        and v84_audit.get("decision") == V84_REJECTION
        and v84_audit.get("protocol_sha256") == _sha256(v84_protocol_path)
        and v84_audit.get("confirmation_gate", {}).get("passed") is False
    ):
        raise ValueError("v83/v84 adaptive redevelopment authorization changed")

    development_seeds = tuple(int(value) for value in v83["development"]["seeds"])
    failed_confirmation_seeds = tuple(
        int(value) for value in v84["confirmation"]["seeds"]
    )
    prospective_seeds = tuple(
        int(value) for value in v83["development"]["sealed_prospective_seeds"]
    )
    redevelopment_seeds = (*development_seeds, *failed_confirmation_seeds)
    prospective_root = Path(
        v83["output_roots"]["local_results_root"]
    ).parent / "tsc_v85r81_external_v9_runtime_queue_trust_prospective_20260830"
    if not (
        len(development_seeds) == 22
        and len(failed_confirmation_seeds) == 32
        and len(redevelopment_seeds) == len(set(redevelopment_seeds)) == 54
        and len(prospective_seeds) == 8
        and not set(redevelopment_seeds).intersection(prospective_seeds)
        and not prospective_root.exists()
    ):
        raise ValueError("v86 redevelopment or sealed prospective partition changed")

    artifact = dict(v83["artifact"]["artifacts"][ARTIFACT_KEY])
    if not (
        artifact["prediction_horizon_sec"] == 120
        and artifact["risk_multiplier"] == 0.25
        and artifact["minimum_context_trust"] == 0.10
        and artifact["selection_eligible_closed_loop"] is True
    ):
        raise ValueError("v86 selected h120 artifact changed")

    candidates = []
    for key, mode, local_cooldown, global_cooldown, maximum in CANDIDATE_SPECS:
        candidates.append(
            {
                "key": key,
                "artifact_key": ARTIFACT_KEY,
                "candidate_role": "post_v84_partial_interference_adaptive_candidate",
                "selection_eligible_closed_loop": True,
                "prediction_horizon_sec": 120,
                "runtime_policy": HIERARCHICAL_RUNTIME_POLICY,
                "coordination_mode": mode,
                "cooldown_intervals": local_cooldown,
                "execution_trust_region": {
                    "enabled": True,
                    "min_priority": 0.0,
                    "min_total_queue": 0.0,
                    "min_total_vehicles": 1.0,
                    "max_mean_speed": None,
                    "max_simultaneous_overrides": maximum,
                    "global_cooldown_intervals": global_cooldown,
                },
            }
        )

    payload = copy.deepcopy(v83)
    payload.update(
        {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "post_v84_rejection_partial_interference_adaptive_redevelopment",
            "redevelopment_parent": {
                "v83_protocol": _path_record(v83_protocol_path),
                "v83_independent_audit": _path_record(v83_audit_path),
                "v83_decision": V83_AUTHORIZATION,
                "v84_protocol": _path_record(v84_protocol_path),
                "v84_independent_audit": _path_record(v84_audit_path),
                "v84_decision": V84_REJECTION,
            },
            "development": {
                "classification": "adaptive_development_only_never_confirmatory",
                "city_scenarios": {"jinan": ["jinan_3x4_real"]},
                "seeds": list(redevelopment_seeds),
                "original_development_seeds": list(development_seeds),
                "failed_confirmation_seeds": list(failed_confirmation_seeds),
                "sealed_prospective_seeds": list(prospective_seeds),
                "policies": [PHASE_POLICY, *METHOD_KEYS],
                "phase_policy": PHASE_POLICY,
                "method_candidates": candidates,
                "primary_method_keys": list(METHOD_KEYS),
                "diagnostic_method_keys": [],
                "candidate_count": len(candidates),
                "policy_count": 1 + len(candidates),
                "matrix_size": len(redevelopment_seeds) * (1 + len(candidates)),
                "pairing_unit": "same_scenario_same_simulator_seed",
                "primary_metric": "mean_tripinfo_waiting_time",
                "progressive_ablation_order": list(METHOD_KEYS),
                "selection_rule": {
                    "maximum_equal_seed_mean_relative_delta": 0.0,
                    "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.0,
                    "maximum_worst_seed_relative_delta": 0.05,
                    "minimum_improved_seed_fraction": 0.5,
                    "minimum_active_seed_fraction": 0.75,
                    "method_teleports_must_not_exceed_paired_phase_pressure": True,
                    "method_collision_incidents_must_not_exceed_paired_phase_pressure": True,
                    "intervention_alignment_must_pass": True,
                    "fallback": "phase_pressure_if_no_candidate_is_feasible",
                    "selector": (
                        "lowest_mean_relative_delta_then_lower_bootstrap_upper_"
                        "then_fewer_overrides_then_progressive_order"
                    ),
                },
                "bootstrap": {
                    "replicates": 10000,
                    "seed": 20260830,
                    "unit": "paired_simulator_seed",
                },
            },
            "output_roots": {
                "remote_results_root": str(remote_results_root),
                "local_results_root": str(Path(local_results_root).resolve()),
            },
            "scientific_status": {
                "classification": "adaptive_redevelopment_after_failed_confirmation",
                "v84_outcomes_used_for_design": True,
                "v84_no_longer_eligible_as_confirmation": True,
                "new_independent_confirmation_required": True,
                "v85_prospective_remains_sealed": True,
                "artifact_refit": False,
                "changed_module": "partial_interference_execution_coordination_only",
                "unchanged_modules": [
                    "state_action_h120_originator",
                    "risk_multiplier",
                    "minimum_context_trust",
                    "phase_pressure_prior",
                    "safe_phase_executor",
                ],
            },
            "claim_boundary": (
                "V86 is adaptive redevelopment after the failed v84 confirmation. "
                "It compares the frozen h120 originator under network-global and "
                "conflict-graph-local horizon alignment. The 54 seeds combine prior "
                "development and failed-confirmation outcomes and can never support "
                "confirmation. V85 prospective evidence remains sealed."
            ),
        }
    )
    if payload["development"]["matrix_size"] != 270:
        raise RuntimeError("v86 matrix size changed")
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v83-protocol", type=Path, required=True)
    parser.add_argument("--v83-audit", type=Path, required=True)
    parser.add_argument("--v84-protocol", type=Path, required=True)
    parser.add_argument("--v84-audit", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v83_protocol_path=args.v83_protocol,
        v83_audit_path=args.v83_audit,
        v84_protocol_path=args.v84_protocol,
        v84_audit_path=args.v84_audit,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "seed_count": len(payload["development"]["seeds"]),
                "matrix_size": payload["development"]["matrix_size"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
