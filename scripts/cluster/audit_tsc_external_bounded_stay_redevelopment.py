#!/usr/bin/env python3
"""Audit v88 against frozen v86 phase/global references on the server."""

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
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (  # noqa: E402
    RESULT_PROTOCOL as V86_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_bounded_stay_redevelopment import (  # noqa: E402
    ALIGNMENT_PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.audit_tsc_external_interference_aware_redevelopment import (  # noqa: E402
    _bootstrap_indices,
)
from scripts.cluster.audit_tsc_external_phase_transition_aware_redevelopment import (  # noqa: E402
    AUDIT_PROTOCOL as V87_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (  # noqa: E402
    PHASE_POLICY,
    PROTOCOL as V86_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_bounded_stay_redevelopment import (  # noqa: E402
    BOUNDED_STAY_POLICY,
    LEGACY_POLICY,
    PROTOCOL,
    V86_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_phase_transition_aware_redevelopment import (  # noqa: E402
    ACTION_AWARE_POLICY,
    PROTOCOL as V87_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_bounded_stay_redevelopment import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_bounded_stay_redevelopment_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v88r84-bounded-stay-redevelopment-audit-v1"


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


def _effective_override_count(result: Mapping[str, Any]) -> int:
    guard = dict(result["metrics"].get("guard_audit", {}))
    if "effective_overrides" in guard:
        return int(guard["effective_overrides"])
    count = 0
    for row in result["metrics"].get("accepted_intervention_trace", ()):
        request_accepted = bool(row.get("request_accepted", False))
        stay_effective = bool(
            not request_accepted
            and str(row.get("selected_phase_state"))
            == str(row.get("current_green_state_before"))
            and str(row.get("selected_phase_state"))
            != str(row.get("prior_phase_state"))
        )
        count += int(request_accepted or stay_effective)
    return count


def _load_result(
    *,
    root: Path,
    seed: int,
    policy: str,
    expected_protocol: str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    result_path, tripinfo_path = _result_paths(root, seed, policy)
    result = _read_json(result_path)
    if not (
        tripinfo_path.is_file()
        and result.get("protocol") == expected_protocol
        and int(result.get("seed", -1)) == int(seed)
        and result.get("policy") == policy
        and result.get("protocol_sha256") == expected_protocol_sha256
        and result.get("tripinfo_evidence", {}).get("sha256")
        == _sha256(tripinfo_path)
        and bool(result.get("metrics", {}).get("ok", False))
        and _phase_accounting_closed(result["metrics"])
    ):
        raise ValueError(f"invalid result evidence: {result_path}")
    if policy == BOUNDED_STAY_POLICY and not (
        result.get("intervention_alignment", {}).get("protocol")
        == ALIGNMENT_PROTOCOL
        and result.get("intervention_alignment", {}).get("passed") is True
    ):
        raise ValueError(f"invalid v88 alignment evidence: {result_path}")
    return result


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials)
    )


def _candidate_summary(
    *,
    policy: str,
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    deltas = np.asarray([float(row["relative_delta"]) for row in rows])
    bootstrap = deltas[bootstrap_indices].mean(axis=1)
    baseline_collisions = sum(int(row["baseline_collision_incidents"]) for row in rows)
    method_collisions = sum(int(row["method_collision_incidents"]) for row in rows)
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
        int(row["method_collision_incidents"] > 0)
        and int(row["baseline_collision_incidents"] > 0)
        for row in rows
    )
    gates = {
        "mean_relative_delta": float(np.mean(deltas))
        <= float(rule["maximum_equal_seed_mean_relative_delta_vs_phase"]),
        "bootstrap_upper": float(np.quantile(bootstrap, 0.975))
        <= float(
            rule[
                "maximum_bootstrap_95pct_upper_mean_relative_delta_vs_phase"
            ]
        ),
        "worst_seed": float(np.max(deltas))
        <= float(rule["maximum_worst_seed_relative_delta_vs_phase"]),
        "improved_seed_fraction": float(np.mean(deltas < 0.0))
        >= float(rule["minimum_improved_seed_fraction_vs_phase"]),
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
    }
    gates["passed"] = all(gates.values())
    baseline_departed = sum(int(row["baseline_departed"]) for row in rows)
    method_departed = sum(int(row["method_departed"]) for row in rows)
    return {
        "policy": policy,
        "seed_count": len(rows),
        "mean_relative_delta": float(np.mean(deltas)),
        "mean_absolute_delta_sec": float(
            np.mean([float(row["absolute_delta"]) for row in rows])
        ),
        "improved_seed_fraction": float(np.mean(deltas < 0.0)),
        "active_seed_fraction": float(
            np.mean([int(row["effective_overrides"]) > 0 for row in rows])
        ),
        "worst_seed_relative_delta": float(np.max(deltas)),
        "total_effective_overrides": sum(
            int(row["effective_overrides"]) for row in rows
        ),
        "baseline_collision_incidents": baseline_collisions,
        "method_collision_incidents": method_collisions,
        "baseline_collision_incidents_per_departed": float(
            baseline_collisions / max(baseline_departed, 1)
        ),
        "method_collision_incidents_per_departed": float(
            method_collisions / max(method_departed, 1)
        ),
        "collision_discordance": {
            "baseline_only_seed_count": int(baseline_only),
            "method_only_seed_count": int(method_only),
            "both_seed_count": int(both),
            "neither_seed_count": int(len(rows) - baseline_only - method_only - both),
            "one_sided_exact_p_for_method_excess": _binomial_upper_tail(
                method_only, baseline_only + method_only
            ),
            "reported_not_used_as_seedwise_dominance_gate": True,
        },
        "bootstrap": {
            "protocol": "shared-index-paired-simulator-seed-bootstrap-v1",
            "observed": float(np.mean(deltas)),
            "ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "replicates": int(bootstrap_indices.shape[0]),
            "seed": 20260830,
        },
        "gates": gates,
        "seed_rows": list(rows),
    }


def audit(
    *,
    protocol_path: Path,
    v86_protocol_path: Path,
    v86_audit_path: Path,
    v87_protocol_path: Path,
    v87_audit_path: Path,
    launch_path: Path,
    v86_results_root: Path,
    results_root: Path,
    sealed_v85_root: Path,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    v86_protocol = _read_json(v86_protocol_path)
    v86_audit = _read_json(v86_audit_path)
    v87_protocol = _read_json(v87_protocol_path)
    v87_audit = _read_json(v87_audit_path)
    launch = _read_json(launch_path)
    seeds = tuple(int(value) for value in protocol["development"]["seeds"])
    protocol_sha = _sha256(protocol_path)
    v86_protocol_sha = _sha256(v86_protocol_path)
    identities = all_rollout_identities(protocol)
    summaries = tuple(
        _read_json(results_root / "_shards" / f"shard_{index}.json")
        for index in range(6)
    )
    integrity = {
        "protocol_exact": protocol.get("protocol") == PROTOCOL,
        "v86_parent_exact": bool(
            v86_protocol.get("protocol") == V86_PROTOCOL
            and v86_audit.get("protocol") == V86_AUDIT_PROTOCOL
            and v86_audit.get("status") == "PASS"
            and v86_audit.get("decision")
            == "retain_phase_pressure_and_reject_v86_redevelopment"
        ),
        "v87_parent_exact": bool(
            v87_protocol.get("protocol") == V87_PROTOCOL
            and v87_audit.get("protocol") == V87_AUDIT_PROTOCOL
            and v87_audit.get("status") == "PASS"
            and v87_audit.get("decision")
            == "freeze_selected_candidate_for_fresh_independent_confirmation"
            and v87_audit.get("selection_gate", {}).get("selected_candidate")
            == LEGACY_POLICY
            and ACTION_AWARE_POLICY
            not in v87_audit.get("selection_gate", {}).get(
                "eligible_candidates", ()
            )
        ),
        "launch_complete": bool(
            launch.get("protocol") == LAUNCH_PROTOCOL
            and launch.get("complete") is True
            and int(launch.get("matrix_size", -1)) == 54
            and int(launch.get("remote_file_count", -1)) == 114
        ),
        "six_shards_exact": bool(
            len(summaries) == 6
            and all(
                summary.get("protocol") == SHARD_PROTOCOL
                and summary.get("passed") is True
                and int(summary.get("task_count", -1)) == 9
                and int(summary.get("workers", -1)) == 9
                and summary.get("protocol_sha256") == protocol_sha
                for summary in summaries
            )
        ),
        "identity_set_exact": len(identities) == len(set(identities)) == 54,
        "v85_remains_sealed": not sealed_v85_root.exists(),
    }

    baseline_results = {
        seed: _load_result(
            root=v86_results_root,
            seed=seed,
            policy=PHASE_POLICY,
            expected_protocol=V86_RESULT_PROTOCOL,
            expected_protocol_sha256=v86_protocol_sha,
        )
        for seed in seeds
    }
    method_results = {
        LEGACY_POLICY: {
            seed: _load_result(
                root=v86_results_root,
                seed=seed,
                policy=LEGACY_POLICY,
                expected_protocol=V86_RESULT_PROTOCOL,
                expected_protocol_sha256=v86_protocol_sha,
            )
            for seed in seeds
        },
        BOUNDED_STAY_POLICY: {
            seed: _load_result(
                root=results_root,
                seed=seed,
                policy=BOUNDED_STAY_POLICY,
                expected_protocol=RESULT_PROTOCOL,
                expected_protocol_sha256=protocol_sha,
            )
            for seed in seeds
        },
    }
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for policy, results in method_results.items():
        rows = []
        for seed in seeds:
            baseline = baseline_results[seed]
            method = results[seed]
            baseline_metrics = dict(baseline["metrics"])
            method_metrics = dict(method["metrics"])
            baseline_waiting = float(baseline_metrics["mean_tripinfo_waiting_time"])
            method_waiting = float(method_metrics["mean_tripinfo_waiting_time"])
            rows.append(
                {
                    "seed": seed,
                    "baseline_waiting_time": baseline_waiting,
                    "method_waiting_time": method_waiting,
                    "absolute_delta": method_waiting - baseline_waiting,
                    "relative_delta": (method_waiting - baseline_waiting)
                    / baseline_waiting,
                    "effective_overrides": _effective_override_count(method),
                    "baseline_teleports": int(
                        baseline_metrics["starting_teleports"]
                    )
                    + int(baseline_metrics["ending_teleports"]),
                    "method_teleports": int(method_metrics["starting_teleports"])
                    + int(method_metrics["ending_teleports"]),
                    "baseline_collision_incidents": int(
                        baseline_metrics["collision_incidents"]
                    ),
                    "method_collision_incidents": int(
                        method_metrics["collision_incidents"]
                    ),
                    "baseline_departed": int(baseline_metrics["departed"]),
                    "method_departed": int(method_metrics["departed"]),
                    "phase_execution_accounting_closed": bool(
                        _phase_accounting_closed(baseline_metrics)
                        and _phase_accounting_closed(method_metrics)
                    ),
                }
            )
        rows_by_policy[policy] = rows

    bootstrap_indices = _bootstrap_indices(
        count=len(seeds), replicates=10000, seed=20260830
    )
    rule = dict(protocol["development"]["selection_rule"])
    summaries_by_policy = {
        policy: _candidate_summary(
            policy=policy,
            rows=rows,
            rule=rule,
            bootstrap_indices=bootstrap_indices,
        )
        for policy, rows in rows_by_policy.items()
    }
    v86_record = next(
        row
        for row in v86_audit["candidate_summaries"]
        if row["policy"] == LEGACY_POLICY
    )
    integrity["legacy_effect_reproduced"] = bool(
        abs(
            summaries_by_policy[LEGACY_POLICY]["mean_relative_delta"]
            - float(v86_record["mean_relative_delta"])
        )
        <= 1e-15
    )
    integrity["passed"] = all(integrity.values())
    if not integrity["passed"]:
        raise ValueError(f"v88 integrity gate failed: {integrity}")

    eligible = [
        summary
        for summary in summaries_by_policy.values()
        if summary["gates"]["passed"] is True
    ]
    eligible.sort(
        key=lambda summary: (
            float(summary["mean_relative_delta"]),
            float(summary["bootstrap"]["ci95"][1]),
            int(summary["total_effective_overrides"]),
        )
    )
    selected = eligible[0]["policy"] if eligible else None
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "v86_protocol_sha256": v86_protocol_sha,
        "v86_audit_sha256": _sha256(v86_audit_path),
        "v87_protocol_sha256": _sha256(v87_protocol_path),
        "v87_audit_sha256": _sha256(v87_audit_path),
        "launch_sha256": _sha256(launch_path),
        "integrity_gate": integrity,
        "candidate_summaries": [
            summaries_by_policy[LEGACY_POLICY],
            summaries_by_policy[BOUNDED_STAY_POLICY],
        ],
        "selection_gate": {
            "passed": selected is not None,
            "eligible_candidates": [row["policy"] for row in eligible],
            "selected_candidate": selected,
            "fresh_independent_confirmation_required": selected is not None,
            "v85_prospective_remains_sealed": True,
        },
        "decision": (
            "freeze_selected_candidate_for_fresh_independent_confirmation"
            if selected is not None
            else "retain_phase_pressure_and_continue_adaptive_redevelopment"
        ),
        "scientific_status": {
            "classification": "adaptive_development_only_never_confirmatory",
            "v86_original_rejection_unchanged": True,
            "aggregate_safety_estimand_applied_prospectively_to_v88_selection": True,
            "cross_network_generalization_claim_authorized": False,
            "v85_prospective_evidence_used": False,
        },
        "v87_unbounded_stay_reference": next(
            row
            for row in v87_audit["candidate_summaries"]
            if row["policy"] == ACTION_AWARE_POLICY
        ),
        "claim_boundary": str(protocol["claim_boundary"]),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V88 bounded-stay redevelopment audit",
        "",
        f"Status: **{payload['status']}**",
        f"Decision: `{payload['decision']}`",
        f"Selected: `{payload['selection_gate']['selected_candidate']}`",
        "",
        "| Candidate | Mean delta | 95% CI | Improved | Collisions phase/method | Passed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidate_summaries"]:
        ci = row["bootstrap"]["ci95"]
        lines.append(
            f"| `{row['policy']}` | {100*row['mean_relative_delta']:.3f}% | "
            f"[{100*ci[0]:.3f}%, {100*ci[1]:.3f}%] | "
            f"{100*row['improved_seed_fraction']:.1f}% | "
            f"{row['baseline_collision_incidents']}/{row['method_collision_incidents']} | "
            f"{row['gates']['passed']} |"
        )
    lines.extend(
        [
            "",
            "V88 is adaptive evidence only. V86 remains rejected under its original "
            "seed-wise gate, v87 unlimited stay bypass remains rejected, and v85 "
            "remains sealed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v86-protocol", type=Path, required=True)
    parser.add_argument("--v86-audit", type=Path, required=True)
    parser.add_argument("--v87-protocol", type=Path, required=True)
    parser.add_argument("--v87-audit", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--v86-results-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--sealed-v85-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.markdown_out.exists():
        raise FileExistsError("refusing to overwrite v88 audit")
    payload = audit(
        protocol_path=args.protocol,
        v86_protocol_path=args.v86_protocol,
        v86_audit_path=args.v86_audit,
        v87_protocol_path=args.v87_protocol,
        v87_audit_path=args.v87_audit,
        launch_path=args.launch,
        v86_results_root=args.v86_results_root,
        results_root=args.results_root,
        sealed_v85_root=args.sealed_v85_root,
    )
    _atomic_json(args.out, payload)
    args.markdown_out.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "selected_candidate": payload["selection_gate"][
                    "selected_candidate"
                ],
                "candidate_summaries": [
                    {
                        "policy": row["policy"],
                        "mean_relative_delta": row["mean_relative_delta"],
                        "ci95": row["bootstrap"]["ci95"],
                        "improved_seed_fraction": row["improved_seed_fraction"],
                        "collisions": [
                            row["baseline_collision_incidents"],
                            row["method_collision_incidents"],
                        ],
                        "passed": row["gates"]["passed"],
                    }
                    for row in payload["candidate_summaries"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
