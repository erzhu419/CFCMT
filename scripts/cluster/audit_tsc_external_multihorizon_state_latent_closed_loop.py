#!/usr/bin/env python3
"""Independently audit and select the frozen v80 closed-loop matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    HIERARCHICAL_RUNTIME_POLICY,
)
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_closed_loop_development import (  # noqa: E402
    ANALYSIS_STAGE,
    RESULT_PROTOCOL,
    all_rollout_identities,
    development_candidates,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (  # noqa: E402
    PRIMARY_ARTIFACT_KEY,
    SECONDARY_ARTIFACT_KEY,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_closed_loop_development import (  # noqa: E402
    DIAGNOSTIC_METHOD_KEYS,
    METHOD_KEYS,
    NODES,
    PHASE_POLICY,
    PRIMARY_METHOD_KEYS,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_multihorizon_state_latent_closed_loop import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_multihorizon_state_latent_closed_loop_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v83r79-multihorizon-state-latent-closed-loop-audit-v1"
AUTHORIZATION_DECISION = "authorize_multihorizon_state_latent_confirmatory_freeze"
DIAGNOSTIC_ONLY_DECISION = "retain_phase_pressure_and_require_new_primary_development"
REJECTION_DECISION = "retain_phase_pressure_and_reject_multihorizon_state_latent"


def _paired_bootstrap(
    values: Sequence[float], *, replicates: int, seed: int
) -> dict[str, Any]:
    observed = np.asarray(values, dtype=float)
    if observed.ndim != 1 or observed.size < 2 or not np.isfinite(observed).all():
        raise ValueError("paired bootstrap requires at least two finite seed deltas")
    rng = np.random.default_rng(int(seed))
    sampled = observed[
        rng.integers(0, observed.size, size=(int(replicates), observed.size))
    ].mean(axis=1)
    return {
        "protocol": "paired-simulator-seed-bootstrap-v1",
        "unit": "paired_simulator_seed",
        "seed": int(seed),
        "replicates": int(replicates),
        "observed": float(np.mean(observed)),
        "ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "probability_nonnegative": float(np.mean(sampled >= 0.0)),
    }


def _load_rollout(
    *,
    results_root: Path,
    identity: tuple[str, str, int, str],
    protocol_sha256: str,
) -> dict[str, Any]:
    city, scenario, seed, policy = identity
    root = results_root / city / scenario / f"seed_{seed}" / policy
    result_path = root / "result.json"
    tripinfo_path = root / "tripinfo.xml"
    result = _read_json(result_path)
    if not (
        result.get("protocol") == RESULT_PROTOCOL
        and result.get("analysis_stage") == ANALYSIS_STAGE
        and result.get("city") == city
        and result.get("scenario") == scenario
        and int(result.get("seed", -1)) == seed
        and result.get("policy") == policy
        and result.get("protocol_sha256") == protocol_sha256
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and bool(result.get("metrics", {}).get("ok", False))
    ):
        raise ValueError(f"v80 rollout changed: {result_path}")
    return result


def _candidate_summary(
    *,
    policy: str,
    pairs: Sequence[Mapping[str, Any]],
    candidate: Any,
    selection_rule: Mapping[str, Any],
    bootstrap_config: Mapping[str, Any],
) -> dict[str, Any]:
    relative = np.asarray([float(row["relative_delta"]) for row in pairs])
    absolute = np.asarray([float(row["absolute_delta"]) for row in pairs])
    overrides = np.asarray([int(row["executed_overrides"]) for row in pairs])
    bootstrap = _paired_bootstrap(
        relative,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
    )
    teleport_violations = sum(
        int(row["method_teleports"]) > int(row["baseline_teleports"])
        for row in pairs
    )
    collision_violations = sum(
        int(row["method_collisions"]) > int(row["baseline_collisions"])
        for row in pairs
    )
    mean_relative = float(np.mean(relative))
    improved_fraction = float(np.mean(relative < 0.0))
    active_fraction = float(np.mean(overrides > 0))
    gates = {
        "mean_relative_delta": mean_relative
        <= float(selection_rule["maximum_equal_seed_mean_relative_delta"]),
        "bootstrap_upper": float(bootstrap["ci95"][1])
        <= float(selection_rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed": float(np.max(relative))
        <= float(selection_rule["maximum_worst_seed_relative_delta"]),
        "improved_seed_fraction": improved_fraction
        >= float(selection_rule["minimum_improved_seed_fraction"]),
        "active_seed_fraction": active_fraction
        >= float(selection_rule["minimum_active_seed_fraction"]),
        "teleport_safety": teleport_violations == 0,
        "collision_safety": collision_violations == 0,
    }
    gates["passed"] = all(gates.values())
    return {
        "policy": str(policy),
        "artifact_key": str(candidate.artifact_key),
        "candidate_role": str(candidate.candidate_role),
        "selection_eligible_closed_loop": bool(
            candidate.selection_eligible_closed_loop
        ),
        "cooldown_intervals": int(candidate.cooldown_intervals),
        "seed_count": len(pairs),
        "mean_baseline_waiting_time": float(
            np.mean([float(row["baseline_waiting_time"]) for row in pairs])
        ),
        "mean_method_waiting_time": float(
            np.mean([float(row["method_waiting_time"]) for row in pairs])
        ),
        "mean_absolute_delta_sec": float(np.mean(absolute)),
        "mean_relative_delta": mean_relative,
        "median_relative_delta": float(np.median(relative)),
        "seed_delta_std": float(np.std(relative)),
        "worst_seed_relative_delta": float(np.max(relative)),
        "best_seed_relative_delta": float(np.min(relative)),
        "improved_seed_fraction": improved_fraction,
        "active_seed_fraction": active_fraction,
        "total_executed_overrides": int(np.sum(overrides)),
        "mean_executed_overrides": float(np.mean(overrides)),
        "teleport_violation_count": int(teleport_violations),
        "collision_violation_count": int(collision_violations),
        "bootstrap": bootstrap,
        "gates": gates,
        "feasible": bool(gates["passed"]),
        "seed_rows": [dict(row) for row in pairs],
    }


def select_primary_candidate(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    feasible = [
        dict(row)
        for row in summaries
        if bool(row["selection_eligible_closed_loop"])
        and bool(row["feasible"])
    ]
    return min(
        feasible,
        key=lambda row: (
            float(row["mean_relative_delta"]),
            float(row["bootstrap"]["ci95"][1]),
            int(row["total_executed_overrides"]),
            -int(row["cooldown_intervals"]),
            str(row["policy"]),
        ),
        default=None,
    )


def select_diagnostic_candidate(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    feasible = [
        dict(row)
        for row in summaries
        if not bool(row["selection_eligible_closed_loop"])
        and bool(row["feasible"])
    ]
    return min(
        feasible,
        key=lambda row: (
            float(row["mean_relative_delta"]),
            float(row["bootstrap"]["ci95"][1]),
            int(row["total_executed_overrides"]),
            -int(row["cooldown_intervals"]),
            str(row["policy"]),
        ),
        default=None,
    )


def audit_matrix(
    *, protocol_path: Path, launch_path: Path, results_root: Path
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    launch = _read_json(launch_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or launch.get("protocol") != LAUNCH_PROTOCOL
        or launch.get("submitted") is not True
        or int(launch.get("matrix_size", -1)) != 154
        or launch.get("remote_preflight", {}).get("passed") is not True
        or launch.get("remote_postflight", {}).get("passed") is not True
        or not all(
            int(row.get("returncode", -1)) == 0
            for row in launch.get("direct_results", ())
        )
    ):
        raise ValueError("v80 launch evidence changed")
    protocol_sha = _sha256(protocol_path)
    identities = all_rollout_identities(protocol)
    results = {
        identity: _load_rollout(
            results_root=results_root,
            identity=identity,
            protocol_sha256=protocol_sha,
        )
        for identity in identities
    }
    shard_rows = []
    for shard_index in range(len(NODES)):
        path = results_root / "_shards" / f"shard_{shard_index}.json"
        shard = _read_json(path)
        if not (
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("protocol_sha256") == protocol_sha
            and int(shard.get("shard_index", -1)) == shard_index
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("workers", -1)) == 21
        ):
            raise ValueError(f"v80 shard changed: {path}")
        shard_rows.append(
            {
                "shard_index": shard_index,
                "hostname": shard["hostname"],
                "task_count": int(shard["task_count"]),
                "workers": int(shard["workers"]),
                "elapsed_sec": float(shard["elapsed_sec"]),
                "sha256": _sha256(path),
            }
        )

    candidates = {row.key: row for row in development_candidates(protocol)}
    seeds = tuple(int(value) for value in protocol["development"]["seeds"])
    candidate_pairs: dict[str, list[dict[str, Any]]] = {
        policy: [] for policy in METHOD_KEYS
    }
    contract_ok = True
    for seed in seeds:
        baseline = results[("jinan", "jinan_3x4_real", seed, PHASE_POLICY)]
        baseline_metrics = dict(baseline["metrics"])
        baseline_waiting = float(baseline_metrics["mean_tripinfo_waiting_time"])
        baseline_teleports = int(baseline_metrics["starting_teleports"]) + int(
            baseline_metrics["ending_teleports"]
        )
        baseline_collisions = int(baseline_metrics["collision_events"])
        contract_ok = contract_ok and (
            baseline.get("proposal_origin") == "phase_pressure"
            and baseline.get("runtime_policy") == PHASE_POLICY
            and baseline.get("target_labels_used_at_fit") is False
            and baseline.get("selection_eligible_evidence") is True
        )
        for policy in METHOD_KEYS:
            candidate = candidates[policy]
            method = results[("jinan", "jinan_3x4_real", seed, policy)]
            metrics = dict(method["metrics"])
            guard = dict(metrics["guard_audit"])
            method_waiting = float(metrics["mean_tripinfo_waiting_time"])
            method_teleports = int(metrics["starting_teleports"]) + int(
                metrics["ending_teleports"]
            )
            method_collisions = int(metrics["collision_events"])
            expected_origin = (
                "target_multihorizon_compact_state_latent"
                if candidate.artifact_key == PRIMARY_ARTIFACT_KEY
                else "target_multihorizon_state_action_latent"
            )
            contract_ok = contract_ok and (
                method.get("proposal_origin") == expected_origin
                and method.get("artifact_key") == candidate.artifact_key
                and method.get("runtime_policy") == HIERARCHICAL_RUNTIME_POLICY
                and method.get("target_labels_used_at_fit") is True
                and method.get("target_outcomes_used_online") is False
                and int(method.get("prediction_horizon_sec", -1)) == 60
                and method.get("selection_eligible_evidence")
                is candidate.selection_eligible_closed_loop
                and method.get("diagnostic_only_evidence")
                is (not candidate.selection_eligible_closed_loop)
                and int(guard.get("target_originator_decisions", 0)) > 0
                and int(metrics["departed"]) == int(baseline_metrics["departed"])
            )
            candidate_pairs[policy].append(
                {
                    "seed": int(seed),
                    "baseline_waiting_time": baseline_waiting,
                    "method_waiting_time": method_waiting,
                    "absolute_delta": method_waiting - baseline_waiting,
                    "relative_delta": (method_waiting - baseline_waiting)
                    / max(baseline_waiting, 1e-12),
                    "executed_overrides": int(guard.get("executed_overrides", 0)),
                    "originator_decisions": int(
                        guard.get("target_originator_decisions", 0)
                    ),
                    "selected_originator_overrides": int(
                        guard.get("selected_target_originators", 0)
                    ),
                    "baseline_teleports": baseline_teleports,
                    "method_teleports": method_teleports,
                    "baseline_collisions": baseline_collisions,
                    "method_collisions": method_collisions,
                    "baseline_arrived": int(baseline_metrics["arrived"]),
                    "method_arrived": int(metrics["arrived"]),
                }
            )
    summaries = [
        _candidate_summary(
            policy=policy,
            pairs=candidate_pairs[policy],
            candidate=candidates[policy],
            selection_rule=protocol["development"]["selection_rule"],
            bootstrap_config=protocol["development"]["bootstrap"],
        )
        for policy in METHOD_KEYS
    ]
    primary_selected = select_primary_candidate(summaries)
    diagnostic_selected = select_diagnostic_candidate(summaries)
    primary_feasible = [
        row
        for row in summaries
        if row["selection_eligible_closed_loop"] and row["feasible"]
    ]
    diagnostic_feasible = [
        row
        for row in summaries
        if not row["selection_eligible_closed_loop"] and row["feasible"]
    ]
    sealed = set(protocol["development"]["sealed_confirmatory_seeds"]) | set(
        protocol["development"]["sealed_prospective_seeds"]
    )
    observed_seeds = {identity[2] for identity in identities}
    integrity = {
        "protocol_and_durable_launch_chain_exact": True,
        "matrix_size_exact": len(results) == 154,
        "identity_set_exact": set(results) == set(identities),
        "six_shards_exact": len(shard_rows) == 6,
        "twenty_one_workers_per_shard": all(
            row["workers"] == 21 for row in shard_rows
        ),
        "runtime_contract_exact": bool(contract_ok),
        "primary_and_diagnostic_roles_exact": bool(
            tuple(protocol["development"]["primary_method_keys"])
            == PRIMARY_METHOD_KEYS
            and tuple(protocol["development"]["diagnostic_method_keys"])
            == DIAGNOSTIC_METHOD_KEYS
        ),
        "sealed_seed_overlap_absent": not bool(observed_seeds.intersection(sealed)),
        "teleports_zero": all(
            int(result["metrics"]["starting_teleports"]) == 0
            and int(result["metrics"]["ending_teleports"]) == 0
            for result in results.values()
        ),
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    advance = bool(integrity["passed"] and primary_selected is not None)
    if advance:
        decision = AUTHORIZATION_DECISION
    elif integrity["passed"] and diagnostic_selected is not None:
        decision = DIAGNOSTIC_ONLY_DECISION
    else:
        decision = REJECTION_DECISION
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": decision,
        "protocol_sha256": protocol_sha,
        "launch_sha256": _sha256(launch_path),
        "results_root": str(results_root.resolve()),
        "matrix_size": len(results),
        "shards": shard_rows,
        "candidate_summaries": summaries,
        "primary_feasible_candidate_count": len(primary_feasible),
        "diagnostic_feasible_candidate_count": len(diagnostic_feasible),
        "selected_primary_candidate": primary_selected,
        "best_diagnostic_candidate": diagnostic_selected,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": advance,
            "selected_policy": (
                primary_selected["policy"] if primary_selected else None
            ),
            "next_stage": (
                "untouched_v84_confirmatory_freeze" if advance else None
            ),
        },
        "diagnostic_gate": {
            "passed": diagnostic_selected is not None,
            "selected_policy": (
                diagnostic_selected["policy"] if diagnostic_selected else None
            ),
            "cannot_authorize_confirmation": True,
        },
        "scientific_status": {
            "classification": "adaptive_development_only",
            "efficacy_claim_authorized": False,
            "untouched_v84_required": advance,
            "v85_remains_sealed": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Multihorizon State-Latent Closed-Loop Development Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Matrix: `{payload['matrix_size']}` paired rollouts",
        f"- Primary feasible: `{payload['primary_feasible_candidate_count']}`",
        f"- Diagnostic feasible: `{payload['diagnostic_feasible_candidate_count']}`",
        "",
        "| Policy | Role | Cooldown | Mean relative delta | 95% CI | Improved seeds | Active seeds | Overrides | Feasible |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["candidate_summaries"]:
        ci = row["bootstrap"]["ci95"]
        lines.append(
            f"| {row['policy']} | {row['candidate_role']} | "
            f"{row['cooldown_intervals']} | {row['mean_relative_delta']:.4%} | "
            f"[{ci[0]:.4%}, {ci[1]:.4%}] | "
            f"{row['improved_seed_fraction']:.1%} | "
            f"{row['active_seed_fraction']:.1%} | "
            f"{row['total_executed_overrides']} | "
            f"{'yes' if row['feasible'] else 'no'} |"
        )
    lines.extend(["", "## Claim Boundary", "", str(payload["claim_boundary"]), ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite v80 audit")
    payload = audit_matrix(
        protocol_path=args.protocol,
        launch_path=args.launch,
        results_root=args.results_root,
    )
    atomic_write_json(args.out, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "primary": payload["advance_gate"]["selected_policy"],
                "diagnostic": payload["diagnostic_gate"]["selected_policy"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
