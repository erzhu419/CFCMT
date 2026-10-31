#!/usr/bin/env python3
"""Audit the adaptive-only v86 partial-interference redevelopment matrix."""

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
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (  # noqa: E402
    ANALYSIS_STAGE,
    RESULT_PROTOCOL,
    all_rollout_identities,
    candidate_payload,
    redevelopment_candidates,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (  # noqa: E402
    ARTIFACT_KEY,
    METHOD_KEYS,
    NODES,
    PHASE_POLICY,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_interference_aware_redevelopment import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_interference_aware_redevelopment_shard import (  # noqa: E402
    SHARD_PROTOCOL,
    _validate_existing,
    shard_identities,
)


AUDIT_PROTOCOL = "tsc-v86r82-interference-aware-redevelopment-audit-v1"
ADVANCE_DECISION = "advance_v86_interference_candidate_to_fresh_confirmation"
REJECTION_DECISION = "retain_phase_pressure_and_reject_v86_redevelopment"


def _bootstrap_indices(*, count: int, replicates: int, seed: int) -> np.ndarray:
    if count < 2 or replicates < 1:
        raise ValueError("paired bootstrap dimensions are invalid")
    return np.random.default_rng(int(seed)).integers(
        0, int(count), size=(int(replicates), int(count))
    )


def _paired_bootstrap(
    values: Sequence[float], *, indices: np.ndarray, seed: int
) -> dict[str, Any]:
    observed = np.asarray(values, dtype=float)
    if (
        observed.ndim != 1
        or observed.size < 2
        or not np.isfinite(observed).all()
        or indices.ndim != 2
        or indices.shape[1] != observed.size
    ):
        raise ValueError("paired bootstrap input changed")
    sampled = observed[indices].mean(axis=1)
    return {
        "protocol": "shared-index-paired-simulator-seed-bootstrap-v1",
        "unit": "paired_simulator_seed",
        "seed": int(seed),
        "replicates": int(indices.shape[0]),
        "observed": float(np.mean(observed)),
        "ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "probability_nonnegative": float(np.mean(sampled >= 0.0)),
    }


def _candidate_summary(
    *,
    policy: str,
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    bootstrap_indices: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    relative = np.asarray([float(row["relative_delta"]) for row in rows])
    overrides = np.asarray([int(row["executed_overrides"]) for row in rows])
    bootstrap = _paired_bootstrap(
        relative, indices=bootstrap_indices, seed=bootstrap_seed
    )
    gates = {
        "mean_relative_delta": float(np.mean(relative))
        <= float(rule["maximum_equal_seed_mean_relative_delta"]),
        "bootstrap_upper": float(bootstrap["ci95"][1])
        <= float(rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed": float(np.max(relative))
        <= float(rule["maximum_worst_seed_relative_delta"]),
        "improved_seed_fraction": float(np.mean(relative < 0.0))
        >= float(rule["minimum_improved_seed_fraction"]),
        "active_seed_fraction": float(np.mean(overrides > 0))
        >= float(rule["minimum_active_seed_fraction"]),
        "teleport_safety": all(
            int(row["method_teleports"]) <= int(row["baseline_teleports"])
            for row in rows
        ),
        "collision_safety": all(
            int(row["method_collision_incidents"])
            <= int(row["baseline_collision_incidents"])
            for row in rows
        ),
        "intervention_alignment": all(
            bool(row["intervention_alignment_passed"]) for row in rows
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "policy": policy,
        "seed_count": len(rows),
        "mean_baseline_waiting_time": float(
            np.mean([float(row["baseline_waiting_time"]) for row in rows])
        ),
        "mean_method_waiting_time": float(
            np.mean([float(row["method_waiting_time"]) for row in rows])
        ),
        "mean_absolute_delta_sec": float(
            np.mean([float(row["absolute_delta"]) for row in rows])
        ),
        "mean_relative_delta": float(np.mean(relative)),
        "median_relative_delta": float(np.median(relative)),
        "seed_delta_std": float(np.std(relative)),
        "worst_seed_relative_delta": float(np.max(relative)),
        "best_seed_relative_delta": float(np.min(relative)),
        "improved_seed_fraction": float(np.mean(relative < 0.0)),
        "active_seed_fraction": float(np.mean(overrides > 0)),
        "total_executed_overrides": int(np.sum(overrides)),
        "total_independent_overlap_pairs": int(
            sum(int(row["independent_overlap_pair_count"]) for row in rows)
        ),
        "baseline_collision_incidents": int(
            sum(int(row["baseline_collision_incidents"]) for row in rows)
        ),
        "method_collision_incidents": int(
            sum(int(row["method_collision_incidents"]) for row in rows)
        ),
        "bootstrap": bootstrap,
        "gates": gates,
        "seed_rows": list(rows),
    }


def _load_and_validate_results(
    *, protocol: Mapping[str, Any], protocol_sha: str, results_root: Path
) -> tuple[
    dict[tuple[str, str, int, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    identities = all_rollout_identities(protocol)
    identity_set = set(identities)
    requested_workers = 19
    results: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    shard_rows = []
    observed_summary_identities = set()
    for shard_index in range(len(NODES)):
        path = results_root / "_shards" / f"shard_{shard_index}.json"
        shard = _read_json(path)
        expected = shard_identities(
            protocol, shard_index=shard_index, shard_count=len(NODES)
        )
        rows = tuple(shard.get("rows", ()))
        row_index = {
            (
                str(row.get("city")),
                str(row.get("scenario")),
                int(row.get("seed", -1)),
                str(row.get("policy")),
            ): row
            for row in rows
        }
        if not (
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("protocol_sha256") == protocol_sha
            and int(shard.get("shard_index", -1)) == shard_index
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == len(expected)
            and int(shard.get("workers", -1))
            == min(requested_workers, len(expected))
            and shard.get("fresh_process_per_rollout") is True
            and len(rows) == len(row_index) == len(expected)
            and set(row_index) == set(expected)
        ):
            raise ValueError(f"v86 shard evidence changed: {path}")
        for identity in expected:
            city, scenario, seed, policy = identity
            root = results_root / city / scenario / f"seed_{seed}" / policy
            result_path = root / "result.json"
            tripinfo_path = root / "tripinfo.xml"
            result = _validate_existing(
                result_path=result_path,
                tripinfo_path=tripinfo_path,
                identity=identity,
                protocol_sha256=protocol_sha,
            )
            row = row_index[identity]
            metrics = dict(result["metrics"])
            guard = dict(metrics.get("guard_audit", {}))
            if not (
                row.get("result_sha256") == _sha256(result_path)
                and row.get("tripinfo_sha256") == _sha256(tripinfo_path)
                and float(row.get("mean_waiting_time"))
                == float(metrics["mean_tripinfo_waiting_time"])
                and int(row.get("executed_overrides", -1))
                == int(guard.get("executed_overrides", 0))
                and int(row.get("collision_incidents", -1))
                == int(metrics["collision_incidents"])
                and int(row.get("starting_teleports", -1))
                == int(metrics["starting_teleports"])
                and int(row.get("ending_teleports", -1))
                == int(metrics["ending_teleports"])
                and row.get("intervention_alignment")
                == result.get("intervention_alignment")
            ):
                raise ValueError(f"v86 shard/result mismatch: {result_path}")
            results[identity] = result
            observed_summary_identities.add(identity)
        shard_rows.append(
            {
                "shard_index": shard_index,
                "hostname": shard["hostname"],
                "spawn_working_directory": shard["spawn_working_directory"],
                "task_count": int(shard["task_count"]),
                "workers": int(shard["workers"]),
                "elapsed_sec": float(shard["elapsed_sec"]),
                "sha256": _sha256(path),
            }
        )
    if set(results) != identity_set or observed_summary_identities != identity_set:
        raise ValueError("v86 result identity coverage changed")
    return results, shard_rows


def audit_redevelopment(
    *, protocol_path: Path, launch_path: Path, results_root: Path
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    launch = _read_json(launch_path)
    protocol_sha = _sha256(protocol_path)
    identities = all_rollout_identities(protocol)
    direct_nodes = tuple(
        str(row.get("node")) for row in launch.get("direct_results", ())
    )
    if not (
        protocol.get("protocol") == PROTOCOL
        and launch.get("protocol") == LAUNCH_PROTOCOL
        and launch.get("complete") is True
        and launch.get("full_rollouts_remain_remote") is True
        and launch.get("local_materialization")
        == "six_shard_summary_json_files_only"
        and int(launch.get("matrix_size", -1)) == 270
        and launch.get("input_hashes", {}).get("protocol") == protocol_sha
        and launch.get("remote_preflight", {}).get("passed") is True
        and launch.get("remote_postflight", {}).get("passed") is True
        and int(
            launch.get("remote_postflight", {}).get("result_file_count", -1)
        )
        == 546
        and direct_nodes == NODES
        and all(
            int(row.get("returncode", -1)) == 0
            for row in launch.get("direct_results", ())
        )
        and launch.get("sealed_v85_local", {}).get("state") == "ABSENT"
        and launch.get("sealed_v85_remote", {}).get("state") == "ABSENT"
    ):
        raise ValueError("v86 launch evidence changed")

    results, shards = _load_and_validate_results(
        protocol=protocol, protocol_sha=protocol_sha, results_root=results_root
    )
    candidates = {row.key: row for row in redevelopment_candidates(protocol)}
    artifact = protocol["artifact"]["artifacts"][ARTIFACT_KEY]
    expected_hashes = {
        "v83_protocol_sha256": protocol["redevelopment_parent"]["v83_protocol"][
            "sha256"
        ],
        "v83_audit_sha256": protocol["redevelopment_parent"][
            "v83_independent_audit"
        ]["sha256"],
        "v84_protocol_sha256": protocol["redevelopment_parent"]["v84_protocol"][
            "sha256"
        ],
        "v84_audit_sha256": protocol["redevelopment_parent"][
            "v84_independent_audit"
        ]["sha256"],
        "parent_runtime_protocol_sha256": protocol["runtime_parent"]["protocol"][
            "sha256"
        ],
        "hierarchical_parent_sha256": protocol["runtime_parent"][
            "hierarchical_parent"
        ]["sha256"],
        "environment_protocol_sha256": protocol["environment"]["protocol"][
            "sha256"
        ],
        "environment_parent_sha256": protocol["environment"][
            "network_parent_protocol"
        ]["sha256"],
        "external_manifest_sha256": protocol["environment"]["external_manifest"][
            "sha256"
        ],
        "conversion_manifest_sha256": protocol["environment"][
            "conversion_manifest"
        ]["sha256"],
    }
    contract_ok = True
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        key: [] for key in METHOD_KEYS
    }
    for seed in (int(value) for value in protocol["development"]["seeds"]):
        baseline = results[("jinan", "jinan_3x4_real", seed, PHASE_POLICY)]
        baseline_metrics = dict(baseline["metrics"])
        contract_ok = contract_ok and bool(
            baseline.get("analysis_stage") == ANALYSIS_STAGE
            and baseline.get("proposal_origin") == "phase_pressure"
            and baseline.get("runtime_policy") == PHASE_POLICY
            and baseline.get("selection_eligible_evidence") is True
            and baseline.get("target_labels_used_at_fit") is False
            and baseline.get("target_outcomes_used_online") is False
            and baseline.get("prediction_horizon_sec") is None
            and all(baseline.get(key) == value for key, value in expected_hashes.items())
        )
        baseline_waiting = float(baseline_metrics["mean_tripinfo_waiting_time"])
        baseline_teleports = int(baseline_metrics["starting_teleports"]) + int(
            baseline_metrics["ending_teleports"]
        )
        for policy in METHOD_KEYS:
            method = results[("jinan", "jinan_3x4_real", seed, policy)]
            metrics = dict(method["metrics"])
            guard = dict(metrics["guard_audit"])
            alignment = dict(method["intervention_alignment"])
            originator_config = dict(
                method.get("originator_diagnostics", {}).get("config", {})
            )
            candidate = candidates[policy]
            contract_ok = contract_ok and bool(
                method.get("analysis_stage") == ANALYSIS_STAGE
                and method.get("proposal_origin")
                == "target_uncertainty_aware_state_action_latent"
                and method.get("artifact_key") == ARTIFACT_KEY
                and method.get("artifact_model_sha256")
                == artifact["model"]["sha256"]
                and method.get("artifact_certificate_sha256")
                == artifact["certificate"]["sha256"]
                and method.get("runtime_policy") == HIERARCHICAL_RUNTIME_POLICY
                and method.get("selection_eligible_evidence") is True
                and method.get("target_labels_used_at_fit") is True
                and method.get("target_outcomes_used_online") is False
                and int(method.get("prediction_horizon_sec", -1)) == 120
                and method.get("execution_candidate") == candidate_payload(candidate)
                and float(originator_config.get("risk_multiplier", -1.0)) == 0.25
                and float(originator_config.get("minimum_context_trust", -1.0))
                == 0.10
                and int(originator_config.get("rollout_value_horizon_sec", -1))
                == 120
                and all(method.get(key) == value for key, value in expected_hashes.items())
                and int(guard.get("target_originator_decisions", 0)) > 0
                and int(metrics["departed"]) == int(baseline_metrics["departed"])
                and alignment.get("passed") is True
                and int(alignment.get("expected_minimum_gap_sec", -1)) == 120
                and int(alignment.get("accepted_request_count", -1))
                == int(guard.get("executed_overrides", 0))
            )
            method_waiting = float(metrics["mean_tripinfo_waiting_time"])
            candidate_rows[policy].append(
                {
                    "seed": seed,
                    "baseline_waiting_time": baseline_waiting,
                    "method_waiting_time": method_waiting,
                    "absolute_delta": method_waiting - baseline_waiting,
                    "relative_delta": (method_waiting - baseline_waiting)
                    / max(baseline_waiting, 1e-12),
                    "executed_overrides": int(
                        guard.get("executed_overrides", 0)
                    ),
                    "independent_overlap_pair_count": int(
                        alignment.get("independent_overlap_pair_count", 0)
                    ),
                    "intervention_alignment_passed": bool(alignment["passed"]),
                    "baseline_teleports": baseline_teleports,
                    "method_teleports": int(metrics["starting_teleports"])
                    + int(metrics["ending_teleports"]),
                    "baseline_collision_incidents": int(
                        baseline_metrics["collision_incidents"]
                    ),
                    "method_collision_incidents": int(
                        metrics["collision_incidents"]
                    ),
                }
            )

    bootstrap_spec = dict(protocol["development"]["bootstrap"])
    indices = _bootstrap_indices(
        count=len(protocol["development"]["seeds"]),
        replicates=int(bootstrap_spec["replicates"]),
        seed=int(bootstrap_spec["seed"]),
    )
    rule = dict(protocol["development"]["selection_rule"])
    summaries = [
        _candidate_summary(
            policy=policy,
            rows=candidate_rows[policy],
            rule=rule,
            bootstrap_indices=indices,
            bootstrap_seed=int(bootstrap_spec["seed"]),
        )
        for policy in METHOD_KEYS
    ]
    order = {policy: index for index, policy in enumerate(METHOD_KEYS)}
    eligible = [row for row in summaries if row["gates"]["passed"]]
    eligible.sort(
        key=lambda row: (
            float(row["mean_relative_delta"]),
            float(row["bootstrap"]["ci95"][1]),
            int(row["total_executed_overrides"]),
            order[str(row["policy"])],
        )
    )
    selected_candidate = str(eligible[0]["policy"]) if eligible else None
    observed_seeds = {identity[2] for identity in identities}
    sealed_seeds = {
        int(value) for value in protocol["development"]["sealed_prospective_seeds"]
    }
    integrity = {
        "protocol_and_durable_launch_chain_exact": True,
        "matrix_size_exact": len(results) == 270,
        "identity_set_exact": set(results) == set(identities),
        "six_shards_exact": len(shards) == 6,
        "shard_workers_exact": all(row["workers"] == 19 for row in shards),
        "runtime_contract_exact": bool(contract_ok),
        "sealed_v85_seed_overlap_absent": not bool(
            observed_seeds.intersection(sealed_seeds)
        ),
        "teleports_zero": all(
            int(result["metrics"]["starting_teleports"]) == 0
            and int(result["metrics"]["ending_teleports"]) == 0
            for result in results.values()
        ),
        "v86_is_adaptive_only": True,
    }
    integrity["passed"] = all(integrity.values())
    advance = bool(integrity["passed"] and selected_candidate is not None)
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": ADVANCE_DECISION if advance else REJECTION_DECISION,
        "protocol_sha256": protocol_sha,
        "launch_sha256": _sha256(launch_path),
        "results_root": str(results_root.resolve()),
        "matrix_size": len(results),
        "shards": shards,
        "candidate_summaries": summaries,
        "integrity_gate": integrity,
        "selection_gate": {
            "passed": advance,
            "eligible_candidates": [str(row["policy"]) for row in eligible],
            "selected_candidate": selected_candidate,
            "deployment_fallback": (
                selected_candidate if selected_candidate else PHASE_POLICY
            ),
            "fresh_independent_confirmation_required": advance,
        },
        "scientific_status": {
            "classification": "adaptive_development_only_never_confirmatory",
            "partial_interference_candidate_selected": advance,
            "v85_prospective_evidence_used": False,
            "cross_network_generalization_claim_authorized": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V86 Interference-Aware Redevelopment Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Matrix: `{payload['matrix_size']}` adaptive-development rollouts",
        f"- Selected candidate: `{payload['selection_gate']['selected_candidate']}`",
        f"- Deployment fallback: `{payload['selection_gate']['deployment_fallback']}`",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Mean delta | 95% CI | Improved | Active | Collisions | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["candidate_summaries"]:
        ci = row["bootstrap"]["ci95"]
        lines.append(
            f"| `{row['policy']}` | {row['mean_relative_delta']:.4%} | "
            f"[{ci[0]:.4%}, {ci[1]:.4%}] | "
            f"{row['improved_seed_fraction']:.1%} | "
            f"{row['active_seed_fraction']:.1%} | "
            f"{row['method_collision_incidents']} | "
            f"{'PASS' if row['gates']['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
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
        raise FileExistsError("refusing to overwrite v86 audit")
    payload = audit_redevelopment(
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
                "selected_candidate": payload["selection_gate"][
                    "selected_candidate"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
