#!/usr/bin/env python3
"""Freeze v89 fresh-seed confirmation after adaptive redevelopment closes."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (  # noqa: E402
    candidate_payload,
    redevelopment_candidates,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_bounded_stay_redevelopment import (  # noqa: E402
    AUDIT_PROTOCOL as V88_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_bounded_stay_redevelopment import (  # noqa: E402
    BOUNDED_STAY_POLICY,
    LEGACY_POLICY as METHOD_POLICY,
    PROTOCOL as V88_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (  # noqa: E402
    PHASE_POLICY,
    PROTOCOL as V86_PROTOCOL,
)


PROTOCOL = "tsc-v89r85-external-v9-fresh-independent-confirmation-v1"
V88_DECISION = "freeze_selected_candidate_for_fresh_independent_confirmation"
METHOD_ROLE = "frozen_v88_selected_legacy_global_candidate"
SEED_GENERATOR_SEED = 20260831
CONFIRMATION_SEED_COUNT = 64
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256(path)}


def generate_confirmation_seeds(
    *, excluded: Iterable[int], count: int = CONFIRMATION_SEED_COUNT
) -> tuple[int, ...]:
    excluded_set = {int(value) for value in excluded}
    rng = random.Random(SEED_GENERATOR_SEED)
    selected: list[int] = []
    while len(selected) < int(count):
        candidate = rng.randrange(100_000, 1_000_000)
        if candidate not in excluded_set and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def freeze_protocol(
    *,
    v86_protocol_path: Path,
    v88_protocol_path: Path,
    v88_audit_path: Path,
    remote_results_root: Path,
    local_results_root: Path,
    sealed_v85_local_root: Path,
    sealed_v85_remote_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite v89 protocol: {output_path}")
    v86 = _read_json(v86_protocol_path)
    v88 = _read_json(v88_protocol_path)
    v88_audit = _read_json(v88_audit_path)
    if not (
        v86.get("protocol") == V86_PROTOCOL
        and v88.get("protocol") == V88_PROTOCOL
        and v88_audit.get("protocol") == V88_AUDIT_PROTOCOL
        and v88_audit.get("status") == "PASS"
        and v88_audit.get("decision") == V88_DECISION
        and v88_audit.get("protocol_sha256") == _sha256(v88_protocol_path)
        and v88_audit.get("selection_gate", {}).get("passed") is True
        and v88_audit.get("selection_gate", {}).get("selected_candidate")
        == METHOD_POLICY
        and BOUNDED_STAY_POLICY
        not in v88_audit.get("selection_gate", {}).get(
            "eligible_candidates", ()
        )
    ):
        raise ValueError("v89 confirmation authorization chain changed")

    candidates = {
        candidate.key: candidate for candidate in redevelopment_candidates(v86)
    }
    method = candidates[METHOD_POLICY]
    if not (
        method.artifact_key == "state_action_h120_r025_t010"
        and method.prediction_horizon_sec == 120
        and method.coordination_mode == "direct"
        and method.cooldown_intervals == 0
        and method.execution_trust_region.enabled
        and method.execution_trust_region.global_cooldown_intervals == 11
        and method.execution_trust_region.max_simultaneous_overrides == 1
        and not method.execution_trust_region.allow_stay_during_global_cooldown
        and method.execution_trust_region.max_stay_overrides_per_global_cooldown
        is None
    ):
        raise ValueError("v89 selected method changed")

    disclosed = tuple(int(value) for value in v88["development"]["seeds"])
    sealed = tuple(
        int(value)
        for value in v88["development"]["sealed_prospective_seeds"]
    )
    seeds = generate_confirmation_seeds(excluded=(*disclosed, *sealed))
    if not (
        len(disclosed) == len(set(disclosed)) == 54
        and len(sealed) == len(set(sealed)) == 8
        and len(seeds) == len(set(seeds)) == CONFIRMATION_SEED_COUNT
        and not set(seeds).intersection(disclosed)
        and not set(seeds).intersection(sealed)
        and not sealed_v85_local_root.exists()
    ):
        raise ValueError("v89 fresh confirmation partition is not sealed")

    payload = copy.deepcopy(v88)
    payload.update(
        {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "post_v88_pre_outcome_fresh_independent_confirmation",
            "confirmation_parent": {
                "v86_protocol": _path_record(v86_protocol_path),
                "v88_protocol": _path_record(v88_protocol_path),
                "v88_independent_audit": _path_record(v88_audit_path),
                "v88_decision": V88_DECISION,
                "selected_policy": METHOD_POLICY,
            },
            "confirmation": {
                "classification": "fresh_independent_seed_confirmation",
                "city_scenarios": {"jinan": ["jinan_3x4_real"]},
                "policies": [PHASE_POLICY, METHOD_POLICY],
                "phase_policy": PHASE_POLICY,
                "method_policy": METHOD_POLICY,
                "method_role": METHOD_ROLE,
                "method_candidate": candidate_payload(method),
                "seeds": list(seeds),
                "seed_count": len(seeds),
                "matrix_size": len(seeds) * 2,
                "pairing_unit": "same_scenario_same_fresh_simulator_seed",
                "primary_metric": "mean_tripinfo_waiting_time",
                "secondary_metrics": [
                    "system_vehicle_hours",
                    "halted_vehicle_hours",
                    "completion_ratio",
                    "mean_queue",
                ],
                "seed_generation": {
                    "algorithm": "python_random_mt19937_unique_randrange_100000_1000000",
                    "generator_seed": SEED_GENERATOR_SEED,
                    "generated_before_any_v89_rollout": True,
                    "excluded_adaptive_seed_count": len(disclosed),
                    "excluded_sealed_v85_seed_count": len(sealed),
                },
                "preregistered_gate": {
                    "maximum_equal_seed_mean_relative_delta_vs_phase": 0.0,
                    "maximum_paired_bootstrap_95pct_upper_mean_relative_delta_vs_phase": 0.0,
                    "maximum_worst_seed_relative_delta_vs_phase": 0.05,
                    "minimum_improved_seed_fraction_vs_phase": 0.5,
                    "maximum_one_sided_exact_sign_test_p": 0.05,
                    "minimum_active_seed_fraction": 0.75,
                    "method_teleports_must_not_exceed_phase_in_aggregate": True,
                    "method_collision_incidents_must_not_exceed_phase_in_aggregate": True,
                    "phase_execution_accounting_must_close": True,
                    "all_frozen_identities_must_complete": True,
                    "no_failed_seed_exclusion": True,
                    "joint_rule": "all_gates_must_pass",
                },
                "bootstrap": {
                    "replicates": 20000,
                    "seed": 20260831,
                    "unit": "paired_fresh_simulator_seed",
                },
                "safety_estimand": dict(
                    v88["development"]["safety_estimand"]
                ),
                "adaptive_seeds_excluded": list(disclosed),
                "sealed_v85_seeds_excluded": list(sealed),
                "sealed_v85_local_root": str(sealed_v85_local_root.resolve()),
                "sealed_v85_remote_root": str(sealed_v85_remote_root),
            },
            "output_roots": {
                "remote_results_root": str(remote_results_root),
                "local_results_root": str(Path(local_results_root).resolve()),
            },
            "scientific_status": {
                "classification": "fresh_independent_confirmation",
                "v89_outcomes_available_at_freeze": False,
                "adaptive_candidate_search_closed": True,
                "artifact_refit": False,
                "target_labels_added": False,
                "v85_prospective_remains_sealed": True,
                "cross_network_generalization_claim_authorized_by_v89": False,
                "same_network_fresh_seed_confirmation_only": True,
            },
            "claim_boundary": (
                "V89 is a fresh-seed confirmation of the single v88-selected legacy "
                "global CFCMT execution rule against PhasePressure on Jinan. Its 64 "
                "seeds are disjoint from all adaptive outcomes and the sealed v85 "
                "partition. It can confirm out-of-seed performance and aggregate "
                "safety on this target network, but not cross-network generalization "
                "or zero-shot transfer."
            ),
        }
    )
    payload.pop("development", None)
    payload.pop("redevelopment_parent", None)
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v88-protocol", type=Path, required=True)
    parser.add_argument("--v88-audit", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--sealed-v85-local-root", type=Path, required=True)
    parser.add_argument("--sealed-v85-remote-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v86_protocol_path=args.v86_protocol,
        v88_protocol_path=args.v88_protocol,
        v88_audit_path=args.v88_audit,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        sealed_v85_local_root=args.sealed_v85_local_root,
        sealed_v85_remote_root=args.sealed_v85_remote_root,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "seed_count": payload["confirmation"]["seed_count"],
                "matrix_size": payload["confirmation"]["matrix_size"],
                "policies": payload["confirmation"]["policies"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
