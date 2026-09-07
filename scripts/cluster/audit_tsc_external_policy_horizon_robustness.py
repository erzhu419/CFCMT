#!/usr/bin/env python3
"""Audit the frozen v49 untouched robustness matrix and its advance gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_policy_horizon_robustness import (  # noqa: E402
    ADAPTATION_AUDIT_DECISION,
    METHOD,
    PHASE_POLICY,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_policy_horizon_robustness import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_policy_horizon_robustness_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v49r45-policy-horizon-robustness-audit-v1"
ADVANCE_DECISION = "authorize_v49_prospective_analysis_freeze"
REJECTION_DECISION = "reject_v49_and_preserve_prospective_seeds"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _tripinfo_semantic_sha256(path: Path) -> str:
    root = ET.parse(path).getroot()
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def bootstrap_mean_interval(
    values: Sequence[float], *, replicates: int, seed: int, confidence: float
) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("bootstrap requires at least two seed-level values")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(data, size=(int(replicates), data.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def audit_policy_horizon_robustness(
    *,
    results_root: Path,
    launch_manifest: Path,
    robustness_protocol_path: Path,
    adaptation_selection_audit_path: Path,
    pressure_joint_audit_path: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(robustness_protocol_path)
    selection = _read_json(adaptation_selection_audit_path)
    identities = all_rollout_identities(protocol)
    expected = {(scenario, seed, policy) for _, scenario, seed, policy in identities}
    identity_index = {
        (scenario, seed, policy): index
        for index, (_, scenario, seed, policy) in enumerate(identities)
    }
    specs = tuple(launch.get("specs", ()))
    spec_by_shard = {
        int(str(spec["signature"]).rsplit("-", 1)[-1]): spec for spec in specs
    }
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol") == RESULT_PROTOCOL,
        "robustness_protocol_matches": protocol.get("protocol") == PROTOCOL,
        "selection_audit_passed": selection.get("status") == "PASS",
        "selection_decision_matches": selection.get("decision")
        == ADAPTATION_AUDIT_DECISION,
        "selection_hash_matches": _sha256(adaptation_selection_audit_path)
        == str(protocol["adaptation_selection_audit"]["sha256"]),
        "submitted": bool(launch.get("submitted", False)),
        "scheduler_returncode_zero": int(launch.get("scheduler_returncode", -1))
        == 0,
        "six_unique_node_shards": set(spec_by_shard) == set(range(len(NODES)))
        and {str(spec["require_node"]) for spec in specs} == set(NODES),
        "matrix_size_matches": int(launch.get("matrix_size", -1)) == len(expected),
    }
    launch_gate["passed"] = all(launch_gate.values())

    shard_rows = []
    shard_identities: set[tuple[str, int, str]] = set()
    for shard_index in range(len(NODES)):
        path = Path(results_root) / "_shards" / f"shard_{shard_index}.json"
        if not path.is_file():
            shard_rows.append({"shard_index": shard_index, "passed": False})
            continue
        shard = _read_json(path)
        spec = spec_by_shard[shard_index]
        rows = tuple(shard.get("rows", ()))
        observed = {
            (str(row["scenario"]), int(row["seed"]), str(row["policy"]))
            for row in rows
        }
        assigned = {
            key
            for key, index in identity_index.items()
            if index % len(NODES) == shard_index
        }
        passed = bool(
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("hostname") == spec.get("require_node")
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == len(assigned)
            and observed == assigned
        )
        shard_rows.append(
            {
                "shard_index": shard_index,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "hostname": shard.get("hostname"),
                "task_count": len(rows),
                "passed": passed,
            }
        )
        shard_identities.update(observed)

    errors: list[str] = []
    rows = []
    robustness_sha = _sha256(robustness_protocol_path)
    selection_sha = _sha256(adaptation_selection_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    for city, scenario, seed, policy in identities:
        root = Path(results_root) / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        if not result_path.is_file() or not tripinfo_path.is_file():
            errors.append(f"missing v49 result: {scenario}/{seed}/{policy}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        shard_index = identity_index[(scenario, seed, policy)] % len(NODES)
        expected_host = str(spec_by_shard[shard_index]["require_node"])
        tripinfo_sha = _sha256(tripinfo_path)
        expected_candidate = protocol["frozen_city_deployment"][city]
        observed_candidate = result.get("frozen_city_candidate", {})
        row_gate = {
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "stage_matches": result.get("analysis_stage") == "untouched_robustness",
            "identity_matches": (
                result.get("city"),
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (city, scenario, seed, policy),
            "physical_host_matches_shard": result.get("hostname") == expected_host,
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "robustness_protocol_hash_matches": result.get(
                "policy_horizon_robustness_protocol_sha256"
            )
            == robustness_sha,
            "selection_hash_matches": result.get(
                "adaptation_selection_audit_sha256"
            )
            == selection_sha,
            "joint_hash_matches": result.get(
                "pressure_regularized_joint_audit_sha256"
            )
            == joint_sha,
            "candidate_matches_freeze": all(
                observed_candidate.get(key) == expected_candidate.get(key)
                for key in (
                    "key",
                    "family",
                    "blend_weight",
                    "risk_multiplier",
                    "min_context_trust",
                    "cooldown_intervals",
                )
            ),
            "fallback_flag_matches": bool(result.get("exact_phase_fallback", False))
            == bool(policy == METHOD and expected_candidate.get("fallback", False)),
            "tripinfo_hash_matches": result.get("tripinfo_evidence", {}).get(
                "sha256"
            )
            == tripinfo_sha,
            "tripinfo_size_matches": int(
                result.get("tripinfo_evidence", {}).get("size_bytes", -1)
            )
            == tripinfo_path.stat().st_size,
            "sumo_succeeded": bool(metrics.get("ok", False)),
            "not_prospective": result.get("prospective_evidence") is False,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"v49 row gate failed: {scenario}/{seed}/{policy}")
        rows.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "policy": policy,
                "hostname": result.get("hostname"),
                "result_path": str(result_path.resolve()),
                "result_sha256": _sha256(result_path),
                "tripinfo_sha256": tripinfo_sha,
                "tripinfo_semantic_sha256": _tripinfo_semantic_sha256(
                    tripinfo_path
                ),
                "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
                "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
                "executed_overrides": int(
                    metrics.get("guard_audit", {}).get(
                        "executed_overrides",
                        metrics.get("guard_audit", {}).get("accepted_overrides", 0),
                    )
                ),
                "safety": {
                    key: int(metrics.get(key, -1))
                    for key in (
                        "starting_teleports",
                        "ending_teleports",
                        "collision_events",
                        "collision_incidents",
                    )
                },
                "metrics": metrics,
                "gate": row_gate,
            }
        )

    keyed = {(row["scenario"], row["seed"], row["policy"]): row for row in rows}
    paired_rows = []
    for city, scenarios in protocol["city_scenarios"].items():
        for scenario in scenarios:
            for seed in protocol["untouched_robustness"]["seeds"]:
                method = keyed.get((scenario, int(seed), METHOD))
                phase = keyed.get((scenario, int(seed), PHASE_POLICY))
                if method is None or phase is None:
                    continue
                paired_rows.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "seed": int(seed),
                        "method_mean_waiting_time": method["mean_waiting_time"],
                        "phase_mean_waiting_time": phase["mean_waiting_time"],
                        "relative_delta": (
                            float(method["mean_waiting_time"])
                            - float(phase["mean_waiting_time"])
                        )
                        / max(float(phase["mean_waiting_time"]), 1e-12),
                        "executed_overrides": method["executed_overrides"],
                        "method_safety": method["safety"],
                        "phase_safety": phase["safety"],
                        "metrics_exact_match": method["metrics"] == phase["metrics"],
                        "tripinfo_semantic_exact_match": method[
                            "tripinfo_semantic_sha256"
                        ]
                        == phase["tripinfo_semantic_sha256"],
                    }
                )

    robustness = dict(protocol["untouched_robustness"])
    seeds = tuple(int(value) for value in robustness["seeds"])
    bootstrap = dict(robustness["bootstrap"])
    city_summary = {}
    for city, scenarios in protocol["city_scenarios"].items():
        city_rows = [row for row in paired_rows if row["city"] == city]
        seed_rows = {}
        for seed in seeds:
            selected = [row for row in city_rows if int(row["seed"]) == seed]
            method_mean = statistics.fmean(
                float(row["method_mean_waiting_time"]) for row in selected
            )
            phase_mean = statistics.fmean(
                float(row["phase_mean_waiting_time"]) for row in selected
            )
            seed_rows[str(seed)] = {
                "method_mean_waiting_time": method_mean,
                "phase_mean_waiting_time": phase_mean,
                "relative_delta": (method_mean - phase_mean)
                / max(phase_mean, 1e-12),
                "executed_overrides": sum(
                    int(row["executed_overrides"]) for row in selected
                ),
            }
        deltas = [float(value["relative_delta"]) for value in seed_rows.values()]
        city_summary[str(city)] = {
            "scenario_count": len(scenarios),
            "seed_count": len(seed_rows),
            "seed_rows": seed_rows,
            "mean_relative_delta": statistics.fmean(deltas),
            "bootstrap_mean_relative_delta": bootstrap_mean_interval(
                deltas,
                replicates=int(bootstrap["replicates"]),
                seed=int(bootstrap["seed"]),
                confidence=float(bootstrap["confidence"]),
            ),
            "worst_scenario_seed_relative_delta": max(
                float(row["relative_delta"]) for row in city_rows
            ),
            "intervened_seed_fraction": statistics.fmean(
                float(value["executed_overrides"] > 0)
                for value in seed_rows.values()
            ),
            "all_metrics_exact_match": all(
                row["metrics_exact_match"] for row in city_rows
            ),
            "all_tripinfo_semantic_exact_match": all(
                row["tripinfo_semantic_exact_match"] for row in city_rows
            ),
        }

    prospective = {
        int(value)
        for value in protocol["prospective_confirmation_reservation"]["seeds"]
    }
    prospective_paths = [
        str(path.resolve())
        for seed in sorted(prospective)
        for path in Path(results_root).rglob(f"seed_{seed}")
    ]
    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_shards_passed": len(shard_rows) == len(NODES)
        and all(row["passed"] for row in shard_rows),
        "shard_rows_cover_matrix": shard_identities == expected,
        "all_expected_results_present": len(rows) == len(expected),
        "all_result_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "all_pairs_present": len(paired_rows)
        == sum(len(values) for values in protocol["city_scenarios"].values())
        * len(seeds),
        "prospective_results_absent": not prospective_paths,
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    threshold = dict(robustness["advance_gate"])
    active_cities = [
        city
        for city, candidate in protocol["frozen_city_deployment"].items()
        if not bool(candidate["fallback"])
    ]
    fallback_cities = [
        city
        for city, candidate in protocol["frozen_city_deployment"].items()
        if bool(candidate["fallback"])
    ]
    advance_gate = {
        "each_city_mean_noninferior": all(
            float(summary["mean_relative_delta"])
            <= float(
                threshold[
                    "each_city_mean_relative_regression_vs_phase_pressure_maximum"
                ]
            )
            for summary in city_summary.values()
        ),
        "each_city_bootstrap_upper_within_limit": all(
            float(summary["bootstrap_mean_relative_delta"]["upper"])
            <= float(
                threshold[
                    "each_city_bootstrap_95pct_upper_mean_relative_delta_maximum"
                ]
            )
            for summary in city_summary.values()
        ),
        "worst_pair_within_limit": max(
            float(row["relative_delta"]) for row in paired_rows
        )
        <= float(
            threshold[
                "worst_scenario_seed_relative_regression_vs_phase_pressure_maximum"
            ]
        ),
        "active_city_intervention_fraction": all(
            float(city_summary[city]["intervened_seed_fraction"])
            >= float(
                threshold[
                    "minimum_fraction_of_active_city_seeds_with_interventions"
                ]
            )
            for city in active_cities
        ),
        "fallback_cities_exact": all(
            bool(city_summary[city]["all_metrics_exact_match"])
            and bool(city_summary[city]["all_tripinfo_semantic_exact_match"])
            for city in fallback_cities
        ),
        "teleports_noninferior": all(
            int(row["method_safety"]["ending_teleports"])
            <= int(row["phase_safety"]["ending_teleports"])
            for row in paired_rows
        ),
        "collision_incidents_noninferior": all(
            int(row["method_safety"]["collision_incidents"])
            <= int(row["phase_safety"]["collision_incidents"])
            for row in paired_rows
        ),
        "result_and_tripinfo_hash_integrity": bool(integrity_gate["passed"]),
    }
    advance_gate["passed"] = all(advance_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": ADVANCE_DECISION if advance_gate["passed"] else REJECTION_DECISION,
        "claim_boundary": (
            "These are untouched robustness-development seeds. They can "
            "authorize a prospective analysis freeze but cannot confirm efficacy."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "robustness_protocol_sha256": robustness_sha,
        "adaptation_selection_audit_sha256": selection_sha,
        "pressure_joint_audit_sha256": joint_sha,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "shards": shard_rows,
        "rows": [{key: value for key, value in row.items() if key != "metrics"} for row in rows],
        "paired_rows": paired_rows,
        "city_summary": city_summary,
        "active_cities": active_cities,
        "fallback_cities": fallback_cities,
        "integrity_gate": integrity_gate,
        "advance_gate": advance_gate,
        "prospective_confirmation_seeds_remain_unrun": sorted(prospective),
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--robustness-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-selection-audit", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v49 robustness audit: {args.out}")
    payload = audit_policy_horizon_robustness(
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        robustness_protocol_path=args.robustness_protocol,
        adaptation_selection_audit_path=args.adaptation_selection_audit,
        pressure_joint_audit_path=args.pressure_joint_audit,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "advance_gate_passed": payload["advance_gate"]["passed"],
                "city_summary": {
                    city: {
                        "mean_relative_delta": row["mean_relative_delta"],
                        "bootstrap_upper": row["bootstrap_mean_relative_delta"][
                            "upper"
                        ],
                    }
                    for city, row in payload["city_summary"].items()
                },
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
