#!/usr/bin/env python3
"""Audit the frozen v81 untouched-seed target-regime validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import hashlib
import json
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    HIERARCHICAL_RUNTIME_POLICY,
)
from cf_h2o.eval.traffic_signal_external_target_regime_trust_validation import (  # noqa: E402
    CANDIDATE_KEY,
    PHASE_POLICY,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_target_regime_trust_validation import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_target_regime_trust_validation_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v81r77-target-regime-trust-validation-audit-v1"
PASS_DECISION = "authorize_prospective_analysis_and_baseline_suite_freeze_only"
FAIL_DECISION = "reject_target_regime_successor_and_preserve_prospective_seeds"
EXPECTED_NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safety(metrics: Mapping[str, Any]) -> dict[str, int]:
    audit = dict(metrics.get("safety_audit", {}))
    return {
        "collision_incidents": int(audit.get("unique_collision_incidents", 0)),
        "starting_teleports": int(metrics.get("starting_teleports", -1)),
        "ending_teleports": int(metrics.get("ending_teleports", -1)),
    }


def _bootstrap(seed_values: Mapping[int, float], *, replicates: int, seed: int) -> dict[str, Any]:
    values = np.asarray([seed_values[key] for key in sorted(seed_values)], dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(replicates), values.size))
    draws = values[indices].mean(axis=1)
    return {
        "protocol": "paired-simulator-seed-bootstrap-v1",
        "replicates": int(replicates),
        "seed": int(seed),
        "observed": float(np.mean(values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_nonnegative": float(np.mean(draws >= 0.0)),
    }


def _active_summary(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenario: str,
    rule: Mapping[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    indexed = {(int(row["seed"]), str(row["scenario"])): row for row in pairs}
    expected = {(int(seed), str(scenario)) for seed in seeds}
    if set(indexed) != expected:
        raise ValueError("v81 active validation matrix changed")
    seed_delta = {int(seed): float(indexed[(int(seed), scenario)]["relative_delta"]) for seed in seeds}
    values = np.asarray(list(seed_delta.values()), dtype=float)
    bootstrap = _bootstrap(seed_delta, replicates=bootstrap_replicates, seed=bootstrap_seed)
    mean_delta = float(np.mean(values))
    improved_fraction = float(np.mean(values < 0.0))
    gates = {
        "regime_gate_active_in_every_method_row": all(bool(row["method_regime_active"]) for row in pairs),
        "hierarchical_runtime_active_in_every_method_row": all(row["method_runtime_policy"] == HIERARCHICAL_RUNTIME_POLICY for row in pairs),
        "minimum_interventions_per_seed": all(int(row["executed_overrides"]) >= int(rule["minimum_executed_interventions_per_seed"]) for row in pairs),
        "minimum_mean_improvement": mean_delta <= -float(rule["minimum_mean_relative_improvement"]),
        "bootstrap_upper_within_limit": float(bootstrap["ci95"][1]) <= float(rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed_within_limit": float(np.max(values)) <= float(rule["maximum_seed_relative_regression"]),
        "minimum_improved_seed_fraction": improved_fraction >= float(rule["minimum_improved_seed_fraction"]),
    }
    gates["passed"] = all(gates.values())
    return {
        "scenario": scenario,
        "paired_seed_count": len(pairs),
        "mean_relative_delta": mean_delta,
        "seed_delta_std": float(np.std(values)),
        "worst_seed_relative_delta": float(np.max(values)),
        "improved_seed_fraction": improved_fraction,
        "total_executed_overrides": int(sum(int(row["executed_overrides"]) for row in pairs)),
        "seed_relative_deltas": {str(key): value for key, value in sorted(seed_delta.items())},
        "bootstrap": bootstrap,
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def _fallback_summary(
    pairs: Sequence[Mapping[str, Any]], *, expected_units: int, rule: Mapping[str, Any]
) -> dict[str, Any]:
    waiting_atol = float(rule["mean_waiting_time_equal_atol"])
    travel_atol = float(rule["mean_travel_time_equal_atol"])
    gates = {
        "expected_pair_count": len(pairs) == int(expected_units),
        "regime_gate_inactive": all(not bool(row["method_regime_active"]) for row in pairs),
        "phase_runtime_exact_fallback": all(row["method_runtime_policy"] == PHASE_POLICY for row in pairs),
        "zero_interventions": all(int(row["executed_overrides"]) == 0 for row in pairs),
        "waiting_time_exact_fallback": all(abs(float(row["method_mean_waiting_time"]) - float(row["phase_mean_waiting_time"])) <= waiting_atol for row in pairs),
        "travel_time_exact_fallback": all(abs(float(row["method_mean_travel_time"]) - float(row["phase_mean_travel_time"])) <= travel_atol for row in pairs),
        "safety_exact_fallback": all(row["method_safety"] == row["phase_safety"] for row in pairs),
    }
    gates["passed"] = all(gates.values())
    return {
        "paired_seed_scenario_units": len(pairs),
        "scenarios": sorted({str(row["scenario"]) for row in pairs}),
        "mean_relative_delta": float(np.mean([float(row["relative_delta"]) for row in pairs])) if pairs else None,
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def _city_summary(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    rule: Mapping[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    indexed = {(int(row["seed"]), str(row["scenario"])): row for row in pairs}
    expected = {(int(seed), str(scenario)) for seed in seeds for scenario in scenarios}
    if set(indexed) != expected:
        raise ValueError("v81 Jinan city validation matrix changed")
    seed_delta = {
        int(seed): float(np.mean([indexed[(int(seed), str(scenario))]["relative_delta"] for scenario in scenarios]))
        for seed in seeds
    }
    values = np.asarray(list(seed_delta.values()), dtype=float)
    bootstrap = _bootstrap(seed_delta, replicates=bootstrap_replicates, seed=bootstrap_seed)
    mean_delta = float(np.mean(values))
    improved_fraction = float(np.mean(values < 0.0))
    gates = {
        "mean_delta_nonpositive": mean_delta <= float(rule["maximum_mean_relative_delta"]),
        "bootstrap_upper_within_limit": float(bootstrap["ci95"][1]) <= float(rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed_within_limit": float(np.max(values)) <= float(rule["maximum_seed_mean_relative_regression"]),
        "minimum_improved_seed_fraction": improved_fraction >= float(rule["minimum_improved_seed_fraction"]),
    }
    gates["passed"] = all(gates.values())
    return {
        "city": "jinan",
        "mean_relative_delta": mean_delta,
        "worst_seed_relative_delta": float(np.max(values)),
        "improved_seed_fraction": improved_fraction,
        "seed_mean_relative_deltas": {str(key): value for key, value in sorted(seed_delta.items())},
        "bootstrap": bootstrap,
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit(*, protocol_path: Path, launch_path: Path, results_root: Path) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    launch = _read_json(launch_path)
    identities = all_rollout_identities(protocol)
    identity_order = {identity: index for index, identity in enumerate(identities)}
    rows = []
    errors = []
    for identity in identities:
        city, scenario, seed, policy = identity
        root = Path(results_root) / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        try:
            result = _read_json(result_path)
            ET.parse(tripinfo_path)
            metrics = dict(result["metrics"])
            expected_node = EXPECTED_NODES[identity_order[identity] % len(EXPECTED_NODES)]
            declared_active = bool(protocol["validation"]["scenario_gate"][scenario])
            is_method = policy == CANDIDATE_KEY
            gate = {
                "protocol": result.get("protocol") == RESULT_PROTOCOL,
                "identity": (result.get("city"), result.get("scenario"), int(result.get("seed", -1)), result.get("policy")) == identity,
                "untouched_validation": result.get("untouched_validation_evidence") is True and result.get("selection_eligible_evidence") is False,
                "protocol_hash": result.get("protocol_sha256") == _sha256(protocol_path),
                "runtime_sources": bool(result.get("runtime_executable_source_gate")) and all(result["runtime_executable_source_gate"].values()),
                "tripinfo_hash": result.get("tripinfo_evidence", {}).get("sha256") == _sha256(tripinfo_path),
                "metrics_ok": metrics.get("ok") is True,
                "no_teleports": int(metrics.get("starting_teleports", -1)) == 0 and int(metrics.get("ending_teleports", -1)) == 0,
                "physical_shard": str(result.get("hostname")) == expected_node,
                "no_refit": result.get("controller_refit") is False,
                "pre_simulation_artifact_decision": result.get("regime_trust_evidence", {}).get("heldout_or_validation_metrics_used") is False,
                "method_gate_matches_freeze": (not is_method and result.get("regime_trust_active") is False) or (is_method and bool(result.get("regime_trust_active")) == declared_active),
            }
            gate["passed"] = all(gate.values())
            guard = dict(metrics.get("guard_audit", {}))
            rows.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "seed": int(seed),
                    "policy": policy,
                    "hostname": result.get("hostname"),
                    "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
                    "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
                    "executed_overrides": int(guard.get("executed_overrides", 0)),
                    "regime_trust_active": bool(result["regime_trust_active"]),
                    "runtime_policy": str(result["runtime_policy"]),
                    "safety": _safety(metrics),
                    "result_path": str(result_path.resolve()),
                    "result_sha256": _sha256(result_path),
                    "tripinfo_path": str(tripinfo_path.resolve()),
                    "tripinfo_sha256": _sha256(tripinfo_path),
                    "gate": gate,
                }
            )
        except Exception as exc:
            errors.append({"identity": list(identity), "error": repr(exc)})
    index = {(row["city"], row["scenario"], int(row["seed"]), row["policy"]): row for row in rows}
    paired = []
    for city, scenarios in protocol["validation"]["city_scenarios"].items():
        for scenario in scenarios:
            for seed in protocol["validation"]["seeds"]:
                phase = index.get((city, scenario, int(seed), PHASE_POLICY))
                method = index.get((city, scenario, int(seed), CANDIDATE_KEY))
                if phase is None or method is None:
                    continue
                denominator = max(abs(float(phase["mean_waiting_time"])), 1e-12)
                paired.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "seed": int(seed),
                        "relative_delta": (float(method["mean_waiting_time"]) - float(phase["mean_waiting_time"])) / denominator,
                        "method_mean_waiting_time": method["mean_waiting_time"],
                        "phase_mean_waiting_time": phase["mean_waiting_time"],
                        "method_mean_travel_time": method["mean_travel_time"],
                        "phase_mean_travel_time": phase["mean_travel_time"],
                        "executed_overrides": method["executed_overrides"],
                        "method_regime_active": method["regime_trust_active"],
                        "method_runtime_policy": method["runtime_policy"],
                        "method_safety": method["safety"],
                        "phase_safety": phase["safety"],
                    }
                )
    analysis = protocol["validation"]["analysis"]
    active_scenarios = [scenario for scenario, active in protocol["validation"]["scenario_gate"].items() if active]
    if len(active_scenarios) != 1:
        raise ValueError("v81 audit requires one active scenario")
    active_pairs = [row for row in paired if row["scenario"] == active_scenarios[0]]
    fallback_pairs = [row for row in paired if row["scenario"] != active_scenarios[0]]
    active_summary = _active_summary(
        active_pairs,
        seeds=protocol["validation"]["seeds"],
        scenario=active_scenarios[0],
        rule=analysis["active_scenario"],
        bootstrap_replicates=int(analysis["bootstrap_replicates"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]),
    )
    fallback_summary = _fallback_summary(
        fallback_pairs,
        expected_units=len(protocol["validation"]["seeds"]) * (len(protocol["validation"]["scenario_gate"]) - 1),
        rule=analysis["fallback"],
    )
    jinan_scenarios = protocol["validation"]["city_scenarios"]["jinan"]
    city_summary = _city_summary(
        [row for row in paired if row["city"] == "jinan"],
        seeds=protocol["validation"]["seeds"],
        scenarios=jinan_scenarios,
        rule=analysis["jinan_all_scenarios"],
        bootstrap_replicates=int(analysis["bootstrap_replicates"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]) + 1,
    )
    method_collisions = sum(int(row["method_safety"]["collision_incidents"]) for row in paired)
    phase_collisions = sum(int(row["phase_safety"]["collision_incidents"]) for row in paired)
    safety_gate = {
        "collisions_noninferior_to_phase": method_collisions <= phase_collisions,
        "method_teleports_zero": all(int(row["method_safety"]["starting_teleports"]) == 0 and int(row["method_safety"]["ending_teleports"]) == 0 for row in paired),
        "phase_teleports_zero": all(int(row["phase_safety"]["starting_teleports"]) == 0 and int(row["phase_safety"]["ending_teleports"]) == 0 for row in paired),
    }
    safety_gate["passed"] = all(safety_gate.values())
    shards = [_read_json(path) for path in sorted((Path(results_root) / "_shards").glob("shard_*.json"))]
    observed_files = [path for path in Path(results_root).rglob("*") if path.is_file()]
    prospective_root = Path(protocol["outputs"]["prospective_root"])
    integrity = {
        "protocol": protocol.get("protocol") == PROTOCOL,
        "no_errors": not errors,
        "all_rows_present": len(rows) == len(identities) == len(index),
        "all_row_gates": bool(rows) and all(row["gate"]["passed"] for row in rows),
        "paired_matrix_complete": len(paired) == len(identities) // 2,
        "all_six_shards": len(shards) == len(EXPECTED_NODES) and {row.get("hostname") for row in shards} == set(EXPECTED_NODES) and all(row.get("protocol") == SHARD_PROTOCOL and row.get("passed") for row in shards),
        "result_file_count": len(observed_files) == 2 * len(identities) + len(EXPECTED_NODES),
        "launch_submitted": launch.get("protocol") == LAUNCH_PROTOCOL and launch.get("submitted") is True,
        "launch_protocol_hash": launch.get("frozen_protocol_sha256") == _sha256(protocol_path),
        "six_physical_nodes": set(launch.get("physical_nodes", ())) == set(EXPECTED_NODES),
        "prospective_output_absent": not prospective_root.exists(),
    }
    integrity["passed"] = all(integrity.values())
    advance = {
        "integrity_passed": bool(integrity["passed"]),
        "active_scenario_passed": bool(active_summary["feasible"]),
        "fallback_contract_passed": bool(fallback_summary["feasible"]),
        "jinan_all_scenarios_passed": bool(city_summary["feasible"]),
        "safety_passed": bool(safety_gate["passed"]),
    }
    advance["passed"] = all(advance.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": PASS_DECISION if advance["passed"] else FAIL_DECISION,
        "claim_boundary": protocol["claim_boundary"],
        "input_hashes": {"protocol": _sha256(protocol_path), "launch": _sha256(launch_path)},
        "active_scenario_summary": active_summary,
        "fallback_summary": fallback_summary,
        "jinan_all_scenarios_summary": city_summary,
        "safety_gate": safety_gate,
        "integrity_gate": integrity,
        "advance_gate": advance,
        "rows": rows,
        "paired_rows": paired,
        "errors": errors,
        "result_file_count": len(observed_files),
        "result_tree_sha256": _tree_sha256(Path(results_root)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v81 audit: {args.out}")
    payload = audit(protocol_path=args.protocol, launch_path=args.launch, results_root=args.results_root)
    _atomic_json(args.out, payload)
    print(json.dumps({"status": payload["status"], "decision": payload["decision"], "advance": payload["advance_gate"]["passed"], "active_mean_delta": payload["active_scenario_summary"]["mean_relative_delta"], "jinan_mean_delta": payload["jinan_all_scenarios_summary"]["mean_relative_delta"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
