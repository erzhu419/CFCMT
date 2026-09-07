#!/usr/bin/env python3
"""Independently audit the untouched v84 uncertainty-horizon confirmation."""

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
from cf_h2o.eval.traffic_signal_uncertainty_horizon_confirmation import (  # noqa: E402
    ANALYSIS_STAGE,
    RESULT_PROTOCOL,
    all_rollout_identities,
    candidate_payload,
    confirmation_candidates,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_confirmation import (  # noqa: E402
    ARTIFACT_KEY,
    METHOD_KEY,
    NODES,
    PHASE_POLICY,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_uncertainty_horizon_confirmation import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_uncertainty_horizon_confirmation_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v84r80-uncertainty-horizon-confirmation-audit-v2"
AUTHORIZATION_DECISION = "authorize_uncertainty_horizon_prospective_freeze"
REJECTION_DECISION = "retain_phase_pressure_and_reject_v84_confirmation"


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
        and result.get("untouched_validation_evidence") is True
        and result.get("prospective_evidence") is False
        and (
            policy == PHASE_POLICY
            or result.get("intervention_alignment", {}).get("passed") is True
        )
    ):
        raise ValueError(f"v84 rollout changed: {result_path}")
    return result


def audit_confirmation(
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
        and launch.get("submitted") is True
        and int(launch.get("matrix_size", -1)) == 64
        and launch.get("input_hashes", {}).get("protocol") == protocol_sha
        and launch.get("remote_preflight", {}).get("passed") is True
        and launch.get("remote_postflight", {}).get("passed") is True
        and direct_nodes == NODES
        and all(
            int(row.get("returncode", -1)) == 0
            for row in launch.get("direct_results", ())
        )
    ):
        raise ValueError("v84 launch evidence changed")
    results = {
        identity: _load_rollout(
            results_root=results_root,
            identity=identity,
            protocol_sha256=protocol_sha,
        )
        for identity in identities
    }
    expected_parent_hashes = {
        "v83_protocol_sha256": protocol["confirmation_parent"]["v83_protocol"][
            "sha256"
        ],
        "v83_audit_sha256": protocol["confirmation_parent"][
            "v83_independent_audit"
        ]["sha256"],
        "artifact_protocol_sha256": protocol["artifact"]["protocol"]["sha256"],
        "artifact_result_sha256": protocol["artifact"]["freeze_result"][
            "sha256"
        ],
        "artifact_audit_sha256": protocol["artifact"]["independent_audit"][
            "sha256"
        ],
    }
    shard_rows = []
    requested_workers = int(launch["workers_per_shard"])
    for shard_index in range(len(NODES)):
        path = results_root / "_shards" / f"shard_{shard_index}.json"
        shard = _read_json(path)
        task_count = sum(
            index % len(NODES) == shard_index for index in range(len(identities))
        )
        if not (
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("protocol_sha256") == protocol_sha
            and int(shard.get("shard_index", -1)) == shard_index
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == task_count
            and int(shard.get("workers", -1)) == min(requested_workers, task_count)
            and all(
                shard.get(key) == value
                for key, value in expected_parent_hashes.items()
            )
            and shard.get("artifact_model_sha256")
            == protocol["artifact"]["model"]["sha256"]
        ):
            raise ValueError(f"v84 shard changed: {path}")
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

    candidate = confirmation_candidates(protocol)[0]
    rows = []
    contract_ok = True
    for seed in protocol["confirmation"]["seeds"]:
        seed = int(seed)
        baseline = results[("jinan", "jinan_3x4_real", seed, PHASE_POLICY)]
        method = results[("jinan", "jinan_3x4_real", seed, METHOD_KEY)]
        baseline_metrics = dict(baseline["metrics"])
        metrics = dict(method["metrics"])
        guard = dict(metrics["guard_audit"])
        alignment = dict(method["intervention_alignment"])
        baseline_waiting = float(baseline_metrics["mean_tripinfo_waiting_time"])
        method_waiting = float(metrics["mean_tripinfo_waiting_time"])
        baseline_teleports = int(baseline_metrics["starting_teleports"]) + int(
            baseline_metrics["ending_teleports"]
        )
        method_teleports = int(metrics["starting_teleports"]) + int(
            metrics["ending_teleports"]
        )
        originator_config = dict(
            method.get("originator_diagnostics", {}).get("config", {})
        )
        contract_ok = contract_ok and bool(
            baseline.get("proposal_origin") == "phase_pressure"
            and baseline.get("runtime_policy") == PHASE_POLICY
            and baseline.get("target_labels_used_at_fit") is False
            and baseline.get("selection_eligible_evidence") is False
            and baseline.get("prediction_horizon_sec") is None
            and method.get("proposal_origin")
            == "target_uncertainty_aware_state_action_latent"
            and method.get("artifact_key") == ARTIFACT_KEY
            and method.get("artifact_model_sha256")
            == protocol["artifact"]["model"]["sha256"]
            and method.get("artifact_certificate_sha256")
            == protocol["artifact"]["certificate"]["sha256"]
            and method.get("runtime_policy") == HIERARCHICAL_RUNTIME_POLICY
            and method.get("target_labels_used_at_fit") is True
            and method.get("target_outcomes_used_online") is False
            and int(method.get("prediction_horizon_sec", -1)) == 120
            and method.get("selection_eligible_evidence") is False
            and method.get("execution_candidate") == candidate_payload(candidate)
            and float(originator_config.get("risk_multiplier", -1.0)) == 0.25
            and float(originator_config.get("minimum_context_trust", -1.0))
            == 0.10
            and int(originator_config.get("rollout_value_horizon_sec", -1)) == 120
            and all(
                baseline.get(key) == value and method.get(key) == value
                for key, value in expected_parent_hashes.items()
            )
            and int(guard.get("target_originator_decisions", 0)) > 0
            and int(metrics["departed"]) == int(baseline_metrics["departed"])
            and alignment.get("passed") is True
            and int(alignment.get("expected_minimum_gap_sec", -1)) == 120
            and int(alignment.get("accepted_request_count", -1))
            == int(guard.get("executed_overrides", 0))
        )
        rows.append(
            {
                "seed": seed,
                "baseline_waiting_time": baseline_waiting,
                "method_waiting_time": method_waiting,
                "absolute_delta": method_waiting - baseline_waiting,
                "relative_delta": (method_waiting - baseline_waiting)
                / max(baseline_waiting, 1e-12),
                "executed_overrides": int(guard.get("executed_overrides", 0)),
                "originator_decisions": int(
                    guard.get("target_originator_decisions", 0)
                ),
                "minimum_observed_gap_sec": alignment.get(
                    "minimum_observed_gap_sec"
                ),
                "baseline_teleports": baseline_teleports,
                "method_teleports": method_teleports,
                "baseline_collisions": int(baseline_metrics["collision_incidents"]),
                "method_collisions": int(metrics["collision_incidents"]),
                "baseline_arrived": int(baseline_metrics["arrived"]),
                "method_arrived": int(metrics["arrived"]),
            }
        )

    relative = np.asarray([float(row["relative_delta"]) for row in rows])
    overrides = np.asarray([int(row["executed_overrides"]) for row in rows])
    rule = dict(protocol["confirmation"]["confirmation_rule"])
    bootstrap = _paired_bootstrap(
        relative,
        replicates=int(protocol["confirmation"]["bootstrap"]["replicates"]),
        seed=int(protocol["confirmation"]["bootstrap"]["seed"]),
    )
    mean_relative = float(np.mean(relative))
    improved_fraction = float(np.mean(relative < 0.0))
    active_fraction = float(np.mean(overrides > 0))
    gates = {
        "mean_relative_delta": mean_relative
        <= float(rule["maximum_equal_seed_mean_relative_delta"]),
        "bootstrap_upper": float(bootstrap["ci95"][1])
        <= float(rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed": float(np.max(relative))
        <= float(rule["maximum_worst_seed_relative_delta"]),
        "improved_seed_fraction": improved_fraction
        >= float(rule["minimum_improved_seed_fraction"]),
        "active_seed_fraction": active_fraction
        >= float(rule["minimum_active_seed_fraction"]),
        "teleport_safety": all(
            row["method_teleports"] <= row["baseline_teleports"] for row in rows
        ),
        "collision_safety": all(
            row["method_collisions"] <= row["baseline_collisions"] for row in rows
        ),
    }
    gates["passed"] = all(gates.values())
    observed_gaps = [
        float(row["minimum_observed_gap_sec"])
        for row in rows
        if row["minimum_observed_gap_sec"] is not None
    ]
    summary = {
        "policy": METHOD_KEY,
        "artifact_key": ARTIFACT_KEY,
        "prediction_horizon_sec": 120,
        "seed_count": len(rows),
        "mean_baseline_waiting_time": float(
            np.mean([row["baseline_waiting_time"] for row in rows])
        ),
        "mean_method_waiting_time": float(
            np.mean([row["method_waiting_time"] for row in rows])
        ),
        "mean_absolute_delta_sec": float(
            np.mean([row["absolute_delta"] for row in rows])
        ),
        "mean_relative_delta": mean_relative,
        "median_relative_delta": float(np.median(relative)),
        "seed_delta_std": float(np.std(relative)),
        "worst_seed_relative_delta": float(np.max(relative)),
        "best_seed_relative_delta": float(np.min(relative)),
        "improved_seed_fraction": improved_fraction,
        "active_seed_fraction": active_fraction,
        "total_executed_overrides": int(np.sum(overrides)),
        "minimum_observed_intervention_gap_sec": (
            min(observed_gaps) if observed_gaps else None
        ),
        "bootstrap": bootstrap,
        "gates": gates,
        "seed_rows": rows,
    }
    prospective_seeds = set(
        _read_json(Path(protocol["partition"]["path"]))["prospective"]["seeds"]
    )
    observed_seeds = {identity[2] for identity in identities}
    integrity = {
        "protocol_and_durable_launch_chain_exact": True,
        "matrix_size_exact": len(results) == 64,
        "identity_set_exact": set(results) == set(identities),
        "six_shards_exact": len(shard_rows) == 6,
        "shard_workers_match_task_counts": all(
            row["workers"] == row["task_count"] for row in shard_rows
        ),
        "runtime_contract_exact": bool(contract_ok),
        "prospective_seed_overlap_absent": not bool(
            observed_seeds.intersection(prospective_seeds)
        ),
        "teleports_zero": all(
            int(result["metrics"]["starting_teleports"]) == 0
            and int(result["metrics"]["ending_teleports"]) == 0
            for result in results.values()
        ),
        "prospective_data_used": False,
    }
    integrity["passed"] = all(
        value for key, value in integrity.items() if key != "prospective_data_used"
    ) and not integrity["prospective_data_used"]
    confirmed = bool(integrity["passed"] and gates["passed"])
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": AUTHORIZATION_DECISION if confirmed else REJECTION_DECISION,
        "protocol_sha256": protocol_sha,
        "launch_sha256": _sha256(launch_path),
        "results_root": str(results_root.resolve()),
        "matrix_size": len(results),
        "shards": shard_rows,
        "confirmation_summary": summary,
        "integrity_gate": integrity,
        "confirmation_gate": {
            "passed": confirmed,
            "selected_policy": METHOD_KEY,
            "next_stage": "untouched_v85_prospective_freeze" if confirmed else None,
        },
        "scientific_status": {
            "classification": "untouched_confirmatory",
            "simulation_efficacy_confirmed": confirmed,
            "cross_network_generalization_claim_authorized": False,
            "v85_required_for_prospective_replication": confirmed,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _report(payload: Mapping[str, Any]) -> str:
    row = payload["confirmation_summary"]
    ci = row["bootstrap"]["ci95"]
    return "\n".join(
        [
            "# Untouched V84 Uncertainty-Horizon Confirmation",
            "",
            f"- Status: `{payload['status']}`",
            f"- Decision: `{payload['decision']}`",
            f"- Matrix: `{payload['matrix_size']}` rollouts",
            f"- Mean relative delta: `{row['mean_relative_delta']:.4%}`",
            f"- Paired bootstrap 95% CI: `[{ci[0]:.4%}, {ci[1]:.4%}]`",
            f"- Improved seeds: `{row['improved_seed_fraction']:.1%}`",
            f"- Active seeds: `{row['active_seed_fraction']:.1%}`",
            f"- Executed overrides: `{row['total_executed_overrides']}`",
            f"- Minimum intervention gap: `{row['minimum_observed_intervention_gap_sec']}` s",
            "",
            "## Claim Boundary",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite v84 audit")
    payload = audit_confirmation(
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
                "confirmed": payload["confirmation_gate"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
