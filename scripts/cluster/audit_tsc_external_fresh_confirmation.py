#!/usr/bin/env python3
"""Audit the preregistered v89 fresh-seed confirmation on the server."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (  # noqa: E402
    _read_json,
)
from cf_h2o.eval.traffic_signal_fresh_confirmation import (  # noqa: E402
    ALIGNMENT_PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.audit_tsc_external_bounded_stay_redevelopment import (  # noqa: E402
    AUDIT_PROTOCOL as V88_AUDIT_PROTOCOL,
)
from scripts.cluster.audit_tsc_external_interference_aware_redevelopment import (  # noqa: E402
    _bootstrap_indices,
)
from scripts.cluster.freeze_tsc_external_v9_bounded_stay_redevelopment import (  # noqa: E402
    BOUNDED_STAY_POLICY,
    PROTOCOL as V88_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_fresh_confirmation import (  # noqa: E402
    METHOD_POLICY,
    PHASE_POLICY,
    PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (  # noqa: E402
    PROTOCOL as V86_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_fresh_confirmation import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_fresh_confirmation_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v89r85-fresh-independent-confirmation-audit-v1"


def _result_paths(root: Path, seed: int, policy: str) -> tuple[Path, Path]:
    base = root / "jinan" / "jinan_3x4_real" / f"seed_{seed}" / policy
    return base / "result.json", base / "tripinfo.xml"


def _phase_accounting_closed(metrics: Mapping[str, Any]) -> bool:
    audit = dict(metrics.get("phase_execution_audit", {}))
    requests = int(audit.get("requests", -1))
    accounted = sum(
        int(audit.get(key, 0))
        for key in (
            "accepted_requests",
            "same_phase_requests",
            "rejected_busy",
            "rejected_min_green",
        )
    )
    return bool(
        requests >= 0
        and requests == accounted
        and int(audit.get("switches", -1))
        == int(audit.get("accepted_requests", -2))
        and all(
            float(audit.get(key, -1.0)) >= 0.0
            for key in (
                "green_seconds",
                "yellow_seconds",
                "all_red_seconds",
                "occupancy_clearance_extension_seconds",
            )
        )
    )


def _load_result(
    *, root: Path, seed: int, policy: str, protocol_sha256: str
) -> dict[str, Any]:
    result_path, tripinfo_path = _result_paths(root, seed, policy)
    result = _read_json(result_path)
    metrics = dict(result.get("metrics", {}))
    alignment = result.get("intervention_alignment")
    policy_evidence_valid = bool(
        policy == PHASE_POLICY and alignment is None
    ) or bool(
        policy == METHOD_POLICY
        and isinstance(alignment, dict)
        and alignment.get("protocol") == ALIGNMENT_PROTOCOL
        and alignment.get("passed") is True
    )
    if not (
        tripinfo_path.is_file()
        and result.get("protocol") == RESULT_PROTOCOL
        and result.get("city") == "jinan"
        and result.get("scenario") == "jinan_3x4_real"
        and int(result.get("seed", -1)) == int(seed)
        and result.get("policy") == policy
        and result.get("protocol_sha256") == protocol_sha256
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and result.get("identity_authorization")
        == "external_frozen_protocol_exact_identity"
        and result.get("confirmation_evidence") is True
        and result.get("prospective_evidence") is True
        and result.get("untouched_validation_evidence") is True
        and bool(metrics.get("ok", False))
        and int(metrics.get("starting_teleports", -1)) == 0
        and int(metrics.get("ending_teleports", -1)) == 0
        and _phase_accounting_closed(metrics)
        and policy_evidence_valid
    ):
        raise ValueError(f"invalid v89 result evidence: {result_path}")
    return result


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials)
    )


def _paired_summary(
    values: np.ndarray, bootstrap_indices: np.ndarray, *, unit: str
) -> dict[str, Any]:
    bootstrap = values[bootstrap_indices].mean(axis=1)
    return {
        "unit": unit,
        "observed_mean": float(np.mean(values)),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _confirmation_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    deltas = np.asarray([float(row["relative_delta"]) for row in rows])
    bootstrap = deltas[bootstrap_indices].mean(axis=1)
    improved = int(np.sum(deltas < 0.0))
    non_ties = int(np.sum(np.abs(deltas) > 1e-15))
    sign_p = _binomial_upper_tail(improved, non_ties)
    baseline_collisions = sum(
        int(row["baseline_collision_incidents"]) for row in rows
    )
    method_collisions = sum(
        int(row["method_collision_incidents"]) for row in rows
    )
    baseline_only = sum(
        int(row["baseline_collision_incidents"] > 0)
        and not int(row["method_collision_incidents"] > 0)
        for row in rows
    )
    method_only = sum(
        int(row["method_collision_incidents"] > 0)
        and not int(row["baseline_collision_incidents"] > 0)
        for row in rows
    )
    both = sum(
        int(row["baseline_collision_incidents"] > 0)
        and int(row["method_collision_incidents"] > 0)
        for row in rows
    )
    gates = {
        "mean_relative_delta": float(np.mean(deltas))
        <= float(rule["maximum_equal_seed_mean_relative_delta_vs_phase"]),
        "paired_bootstrap_upper": float(np.quantile(bootstrap, 0.975))
        <= float(
            rule[
                "maximum_paired_bootstrap_95pct_upper_mean_relative_delta_vs_phase"
            ]
        ),
        "worst_seed": float(np.max(deltas))
        <= float(rule["maximum_worst_seed_relative_delta_vs_phase"]),
        "improved_seed_fraction": improved / len(rows)
        >= float(rule["minimum_improved_seed_fraction_vs_phase"]),
        "one_sided_exact_sign_test": sign_p
        <= float(rule["maximum_one_sided_exact_sign_test_p"]),
        "active_seed_fraction": float(
            np.mean([int(row["effective_overrides"]) > 0 for row in rows])
        )
        >= float(rule["minimum_active_seed_fraction"]),
        "teleport_safety": sum(int(row["method_teleports"]) for row in rows)
        <= sum(int(row["baseline_teleports"]) for row in rows),
        "aggregate_collision_safety": method_collisions <= baseline_collisions,
        "phase_execution_accounting": all(
            bool(row["phase_execution_accounting_closed"]) for row in rows
        ),
        "all_frozen_identities_complete": len(rows) == 64,
    }
    gates["passed"] = all(gates.values())

    secondary = {}
    for metric in (
        "system_vehicle_hours",
        "halted_vehicle_hours",
        "mean_queue",
    ):
        relative = np.asarray(
            [
                (float(row[f"method_{metric}"]) - float(row[f"baseline_{metric}"]))
                / max(abs(float(row[f"baseline_{metric}"])), 1e-12)
                for row in rows
            ]
        )
        secondary[metric] = _paired_summary(
            relative, bootstrap_indices, unit="relative_method_minus_phase"
        )
    completion = np.asarray(
        [
            float(row["method_completion_ratio"])
            - float(row["baseline_completion_ratio"])
            for row in rows
        ]
    )
    secondary["completion_ratio"] = _paired_summary(
        completion, bootstrap_indices, unit="absolute_method_minus_phase"
    )
    return {
        "seed_count": len(rows),
        "mean_relative_delta": float(np.mean(deltas)),
        "mean_absolute_delta_sec": float(
            np.mean([float(row["absolute_delta"]) for row in rows])
        ),
        "median_relative_delta": float(np.median(deltas)),
        "improved_seed_fraction": improved / len(rows),
        "active_seed_fraction": float(
            np.mean([int(row["effective_overrides"]) > 0 for row in rows])
        ),
        "worst_seed_relative_delta": float(np.max(deltas)),
        "best_seed_relative_delta": float(np.min(deltas)),
        "total_effective_overrides": sum(
            int(row["effective_overrides"]) for row in rows
        ),
        "exact_sign_test": {
            "improved": improved,
            "non_ties": non_ties,
            "one_sided_p": sign_p,
        },
        "bootstrap": {
            "protocol": "paired-fresh-simulator-seed-bootstrap-v1",
            "observed": float(np.mean(deltas)),
            "ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "replicates": int(bootstrap_indices.shape[0]),
            "seed": 20260831,
        },
        "safety": {
            "baseline_collision_incidents": baseline_collisions,
            "method_collision_incidents": method_collisions,
            "baseline_teleports": sum(
                int(row["baseline_teleports"]) for row in rows
            ),
            "method_teleports": sum(
                int(row["method_teleports"]) for row in rows
            ),
            "collision_discordance": {
                "baseline_only_seed_count": baseline_only,
                "method_only_seed_count": method_only,
                "both_seed_count": both,
                "neither_seed_count": len(rows) - baseline_only - method_only - both,
                "one_sided_exact_p_for_method_excess": _binomial_upper_tail(
                    method_only, baseline_only + method_only
                ),
            },
        },
        "secondary_metrics": secondary,
        "gates": gates,
        "seed_rows": list(rows),
    }


def audit(
    *,
    protocol_path: Path,
    v86_protocol_path: Path,
    v88_protocol_path: Path,
    v88_audit_path: Path,
    launch_path: Path,
    results_root: Path,
    sealed_v85_root: Path,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    v86 = _read_json(v86_protocol_path)
    v88 = _read_json(v88_protocol_path)
    v88_audit = _read_json(v88_audit_path)
    launch = _read_json(launch_path)
    confirmation = dict(protocol["confirmation"])
    seeds = tuple(int(value) for value in confirmation["seeds"])
    protocol_sha = _sha256(protocol_path)
    identities = all_rollout_identities(protocol)
    summaries = tuple(
        _read_json(results_root / "_shards" / f"shard_{index}.json")
        for index in range(6)
    )
    expected_shard_sizes = (22, 22, 21, 21, 21, 21)
    summary_identities = {
        (
            str(row["city"]),
            str(row["scenario"]),
            int(row["seed"]),
            str(row["policy"]),
        )
        for summary in summaries
        for row in summary["rows"]
    }
    integrity = {
        "protocol_exact": protocol.get("protocol") == PROTOCOL,
        "v86_parent_exact": v86.get("protocol") == V86_PROTOCOL,
        "v88_parent_exact": bool(
            v88.get("protocol") == V88_PROTOCOL
            and v88_audit.get("protocol") == V88_AUDIT_PROTOCOL
            and v88_audit.get("status") == "PASS"
            and v88_audit.get("decision")
            == "freeze_selected_candidate_for_fresh_independent_confirmation"
            and v88_audit.get("selection_gate", {}).get("selected_candidate")
            == METHOD_POLICY
            and BOUNDED_STAY_POLICY
            not in v88_audit.get("selection_gate", {}).get(
                "eligible_candidates", ()
            )
            and protocol["confirmation_parent"]["v88_independent_audit"][
                "sha256"
            ]
            == _sha256(v88_audit_path)
        ),
        "launch_complete": bool(
            launch.get("protocol") == LAUNCH_PROTOCOL
            and launch.get("complete") is True
            and int(launch.get("matrix_size", -1)) == 128
            and int(launch.get("remote_file_count", -1)) == 262
            and launch.get("protocol_sha256") == protocol_sha
        ),
        "six_shards_exact": bool(
            len(summaries) == 6
            and all(
                summary.get("protocol") == SHARD_PROTOCOL
                and summary.get("passed") is True
                and int(summary.get("task_count", -1))
                == expected_shard_sizes[index]
                and int(summary.get("workers", -1))
                == expected_shard_sizes[index]
                and summary.get("protocol_sha256") == protocol_sha
                for index, summary in enumerate(summaries)
            )
        ),
        "identity_set_exact": bool(
            len(identities) == len(set(identities)) == 128
            and summary_identities == set(identities)
        ),
        "all_rollouts_fresh": all(
            row.get("reused") is False
            for summary in summaries
            for row in summary["rows"]
        ),
        "seed_partition_disjoint": bool(
            len(seeds) == len(set(seeds)) == 64
            and not set(seeds).intersection(
                confirmation["adaptive_seeds_excluded"]
            )
            and not set(seeds).intersection(
                confirmation["sealed_v85_seeds_excluded"]
            )
        ),
        "v85_remains_sealed": not sealed_v85_root.exists(),
    }

    baseline_results = {
        seed: _load_result(
            root=results_root,
            seed=seed,
            policy=PHASE_POLICY,
            protocol_sha256=protocol_sha,
        )
        for seed in seeds
    }
    method_results = {
        seed: _load_result(
            root=results_root,
            seed=seed,
            policy=METHOD_POLICY,
            protocol_sha256=protocol_sha,
        )
        for seed in seeds
    }
    rows = []
    for seed in seeds:
        baseline_metrics = dict(baseline_results[seed]["metrics"])
        method_metrics = dict(method_results[seed]["metrics"])
        baseline_waiting = float(baseline_metrics["mean_tripinfo_waiting_time"])
        method_waiting = float(method_metrics["mean_tripinfo_waiting_time"])
        guard = dict(method_metrics["guard_audit"])
        rows.append(
            {
                "seed": seed,
                "baseline_waiting_time": baseline_waiting,
                "method_waiting_time": method_waiting,
                "absolute_delta": method_waiting - baseline_waiting,
                "relative_delta": (method_waiting - baseline_waiting)
                / baseline_waiting,
                "effective_overrides": int(guard["effective_overrides"]),
                "baseline_teleports": int(baseline_metrics["starting_teleports"])
                + int(baseline_metrics["ending_teleports"]),
                "method_teleports": int(method_metrics["starting_teleports"])
                + int(method_metrics["ending_teleports"]),
                "baseline_collision_incidents": int(
                    baseline_metrics["collision_incidents"]
                ),
                "method_collision_incidents": int(
                    method_metrics["collision_incidents"]
                ),
                "phase_execution_accounting_closed": bool(
                    _phase_accounting_closed(baseline_metrics)
                    and _phase_accounting_closed(method_metrics)
                ),
                **{
                    f"baseline_{metric}": float(baseline_metrics[metric])
                    for metric in (
                        "system_vehicle_hours",
                        "halted_vehicle_hours",
                        "mean_queue",
                        "completion_ratio",
                    )
                },
                **{
                    f"method_{metric}": float(method_metrics[metric])
                    for metric in (
                        "system_vehicle_hours",
                        "halted_vehicle_hours",
                        "mean_queue",
                        "completion_ratio",
                    )
                },
            }
        )

    bootstrap = dict(confirmation["bootstrap"])
    bootstrap_indices = _bootstrap_indices(
        count=len(seeds),
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    confirmation_summary = _confirmation_summary(
        rows=rows,
        rule=dict(confirmation["preregistered_gate"]),
        bootstrap_indices=bootstrap_indices,
    )
    integrity["passed"] = all(integrity.values())
    if not integrity["passed"]:
        raise ValueError(f"v89 integrity gate failed: {integrity}")
    confirmed = confirmation_summary["gates"]["passed"] is True
    development_reference = next(
        row
        for row in v88_audit["candidate_summaries"]
        if row["policy"] == METHOD_POLICY
    )
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "v86_protocol_sha256": _sha256(v86_protocol_path),
        "v88_protocol_sha256": _sha256(v88_protocol_path),
        "v88_audit_sha256": _sha256(v88_audit_path),
        "launch_sha256": _sha256(launch_path),
        "integrity_gate": integrity,
        "confirmation_gate": {
            "passed": confirmed,
            "policy": METHOD_POLICY,
            "baseline": PHASE_POLICY,
            "joint_rule": "all_preregistered_gates_must_pass",
            "component_gates": confirmation_summary["gates"],
        },
        "confirmation_summary": confirmation_summary,
        "adaptive_development_reference": {
            "mean_relative_delta": development_reference[
                "mean_relative_delta"
            ],
            "bootstrap_ci95": development_reference["bootstrap"]["ci95"],
            "seed_count": development_reference["seed_count"],
            "not_pooled_with_confirmation": True,
        },
        "decision": (
            "confirm_same_network_fresh_seed_cfcmt_advantage"
            if confirmed
            else "reject_preregistered_same_network_confirmation"
        ),
        "scientific_status": {
            "classification": "fresh_independent_confirmation",
            "adaptive_candidate_search_closed": True,
            "v85_prospective_evidence_used": False,
            "cross_network_generalization_claim_authorized": False,
            "zero_shot_claim_authorized": False,
            "same_network_fresh_seed_claim_authorized": confirmed,
        },
        "claim_boundary": str(protocol["claim_boundary"]),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["confirmation_summary"]
    ci = summary["bootstrap"]["ci95"]
    safety = summary["safety"]
    lines = [
        "# V89 fresh independent confirmation audit",
        "",
        f"Evidence status: **{payload['status']}**",
        f"Decision: `{payload['decision']}`",
        f"Preregistered joint gate: **{payload['confirmation_gate']['passed']}**",
        "",
        "| Seeds | Mean delta | 95% CI | Improved | Sign-test p | Collisions phase/method |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary['seed_count']} | {100*summary['mean_relative_delta']:.3f}% | "
            f"[{100*ci[0]:.3f}%, {100*ci[1]:.3f}%] | "
            f"{100*summary['improved_seed_fraction']:.1f}% | "
            f"{summary['exact_sign_test']['one_sided_p']:.4g} | "
            f"{safety['baseline_collision_incidents']}/"
            f"{safety['method_collision_incidents']} |"
        ),
        "",
        "This confirmation supports only same-network fresh-seed evaluation. It does "
        "not by itself establish cross-network or zero-shot generalization.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v88-protocol", type=Path, required=True)
    parser.add_argument("--v88-audit", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--sealed-v85-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.markdown_out.exists():
        raise FileExistsError("refusing to overwrite v89 audit")
    payload = audit(
        protocol_path=args.protocol,
        v86_protocol_path=args.v86_protocol,
        v88_protocol_path=args.v88_protocol,
        v88_audit_path=args.v88_audit,
        launch_path=args.launch,
        results_root=args.results_root,
        sealed_v85_root=args.sealed_v85_root,
    )
    _atomic_json(args.out, payload)
    args.markdown_out.write_text(_markdown(payload), encoding="utf-8")
    summary = payload["confirmation_summary"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "confirmation_passed": payload["confirmation_gate"]["passed"],
                "mean_relative_delta": summary["mean_relative_delta"],
                "ci95": summary["bootstrap"]["ci95"],
                "improved_seed_fraction": summary["improved_seed_fraction"],
                "sign_test_p": summary["exact_sign_test"]["one_sided_p"],
                "collisions": [
                    summary["safety"]["baseline_collision_incidents"],
                    summary["safety"]["method_collision_incidents"],
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
