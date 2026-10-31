#!/usr/bin/env python3
"""Audit v64 closed-loop development and freeze one common execution candidate."""

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

from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (  # noqa: E402
    HIERARCHICAL_RUNTIME_POLICY,
    PHASE_POLICY,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
    candidate_payload,
    development_candidates,
)
from scripts.cluster.launch_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_hierarchical_closed_loop_development_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = (
    "tsc-v64r60-external-v9-hierarchical-closed-loop-development-selection-audit-v1"
)
ADVANCE_DECISION = (
    "authorize_frozen_hierarchical_execution_candidate_for_untouched_validation"
)
REJECTION_DECISION = (
    "reject_hierarchical_development_and_preserve_validation_and_prospective_seeds"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bootstrap_mean_interval(
    values: Sequence[float], *, replicates: int, seed: int, confidence: float = 0.95
) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("bootstrap requires at least two finite seed-level values")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(data, size=(int(replicates), data.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def summarize_candidate_city(
    *,
    city: str,
    candidate: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    key = str(candidate["key"])
    scenario_set = {str(value) for value in scenarios}
    seed_set = {int(value) for value in seeds}
    selected = [
        row
        for row in paired_rows
        if str(row["city"]) == str(city)
        and str(row["candidate"]) == key
        and str(row["scenario"]) in scenario_set
        and int(row["seed"]) in seed_set
    ]
    if len(selected) != len(scenario_set) * len(seed_set):
        raise ValueError(f"candidate pairing incomplete for {city}/{key}")
    seed_rows: dict[str, Any] = {}
    for seed in seeds:
        rows = [row for row in selected if int(row["seed"]) == int(seed)]
        if {str(row["scenario"]) for row in rows} != scenario_set:
            raise ValueError(f"scenario pairing incomplete for {city}/{key}/{seed}")
        seed_rows[str(seed)] = {
            "scenario_relative_deltas": {
                str(row["scenario"]): float(row["relative_delta"]) for row in rows
            },
            "city_mean_relative_delta": statistics.fmean(
                float(row["relative_delta"]) for row in rows
            ),
            "equal_scenario_method_mean_waiting_time": statistics.fmean(
                float(row["method_mean_waiting_time"]) for row in rows
            ),
            "equal_scenario_phase_mean_waiting_time": statistics.fmean(
                float(row["phase_mean_waiting_time"]) for row in rows
            ),
            "executed_overrides": sum(int(row["executed_overrides"]) for row in rows),
        }
    deltas = [
        float(seed_rows[str(seed)]["city_mean_relative_delta"]) for seed in seeds
    ]
    mean_delta = statistics.fmean(deltas)
    std_delta = statistics.pstdev(deltas)
    worst_seed = max(deltas)
    interval = bootstrap_mean_interval(
        deltas,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    improved_fraction = statistics.fmean(float(value < 0.0) for value in deltas)
    minimum_interventions = int(
        selection_rule["minimum_executed_interventions_per_scenario_seed"]
    )
    intervention_gate = all(
        int(row["executed_overrides"]) >= minimum_interventions for row in selected
    )
    teleport_gate = all(
        int(row["method_safety"]["starting_teleports"])
        <= int(row["phase_safety"]["starting_teleports"])
        and int(row["method_safety"]["ending_teleports"])
        <= int(row["phase_safety"]["ending_teleports"])
        for row in selected
    )
    collision_gate = all(
        int(row["method_safety"]["collision_incidents"])
        <= int(row["phase_safety"]["collision_incidents"])
        for row in selected
    )
    gates = {
        "minimum_city_mean_improvement": mean_delta
        <= -float(selection_rule["minimum_city_mean_relative_improvement"]),
        "bootstrap_upper_within_limit": interval["upper"]
        <= float(
            selection_rule[
                "maximum_bootstrap_95pct_upper_mean_relative_delta"
            ]
        ),
        "worst_seed_regression_within_limit": worst_seed
        <= float(
            selection_rule[
                "maximum_seed_level_city_mean_relative_regression"
            ]
        ),
        "minimum_improved_seed_fraction": improved_fraction
        >= float(selection_rule["minimum_improved_seed_fraction"]),
        "every_scenario_seed_intervened": intervention_gate,
        "teleports_noninferior_to_paired_phase": teleport_gate,
        "collisions_noninferior_to_paired_phase": collision_gate,
    }
    gates["passed"] = all(gates.values())
    return {
        **dict(candidate),
        "paired_scenario_seed_units": len(selected),
        "seed_rows": seed_rows,
        "city_mean_relative_delta": mean_delta,
        "seed_delta_std": std_delta,
        "bootstrap_mean_relative_delta": interval,
        "worst_seed_relative_delta": worst_seed,
        "improved_seed_fraction": improved_fraction,
        "total_executed_overrides": sum(
            int(row["executed_overrides"]) for row in selected
        ),
        "robust_score": mean_delta + 0.5 * std_delta + max(worst_seed, 0.0),
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def select_common_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    city_scenarios: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    grid = []
    for candidate in candidates:
        city_summaries = {
            str(city): summarize_candidate_city(
                city=str(city),
                candidate=candidate,
                paired_rows=paired_rows,
                scenarios=tuple(str(value) for value in scenarios),
                seeds=seeds,
                selection_rule=selection_rule,
                bootstrap=bootstrap,
            )
            for city, scenarios in city_scenarios.items()
        }
        row = {
            **dict(candidate),
            "city_summaries": city_summaries,
            "all_cities_feasible": all(
                bool(value["feasible"]) for value in city_summaries.values()
            ),
            "equal_city_mean_robust_score": statistics.fmean(
                float(value["robust_score"]) for value in city_summaries.values()
            ),
            "equal_city_mean_relative_delta": statistics.fmean(
                float(value["city_mean_relative_delta"])
                for value in city_summaries.values()
            ),
            "total_executed_overrides": sum(
                int(value["total_executed_overrides"])
                for value in city_summaries.values()
            ),
        }
        grid.append(row)
    feasible = [row for row in grid if bool(row["all_cities_feasible"])]
    selected = (
        min(
            feasible,
            key=lambda row: (
                float(row["equal_city_mean_robust_score"]),
                float(row["equal_city_mean_relative_delta"]),
                int(row["total_executed_overrides"]),
                -int(row["cooldown_intervals"]),
                str(row["key"]),
            ),
        )
        if feasible
        else {
            "key": PHASE_POLICY,
            "runtime_policy": PHASE_POLICY,
            "coordination_mode": "direct",
            "cooldown_intervals": 0,
            "execution_trust_region": None,
            "fallback": True,
        }
    )
    selected_config = {
        key: selected[key]
        for key in (
            "key",
            "runtime_policy",
            "coordination_mode",
            "cooldown_intervals",
            "execution_trust_region",
        )
    }
    return {
        "decision": (
            "freeze_common_active_hierarchical_candidate"
            if feasible
            else "retain_phase_pressure_fallback"
        ),
        "selected": selected,
        "selected_config": selected_config,
        "selected_config_sha256": _canonical_sha256(selected_config),
        "feasible_common_candidate_count": len(feasible),
        "grid": grid,
    }


def _tripinfo_is_valid(path: Path) -> bool:
    try:
        return ET.parse(path).getroot().tag == "tripinfos"
    except (ET.ParseError, OSError):
        return False


def audit_hierarchical_closed_loop_development(
    *,
    results_root: Path,
    launch_manifest_path: Path,
    protocol_path: Path,
    parent_protocol_path: Path,
    offline_result_path: Path,
    support_audit_path: Path,
    freeze_audit_path: Path,
    environment_protocol_path: Path,
    environment_parent_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
) -> dict[str, Any]:
    results_root = Path(results_root)
    launch = _read_json(launch_manifest_path)
    protocol = _read_json(protocol_path)
    identities = all_rollout_identities(protocol)
    identity_index = {identity: index for index, identity in enumerate(identities)}
    expected = set(identities)
    protocol_sha = _sha256(protocol_path)
    parent_sha = _sha256(parent_protocol_path)
    offline_sha = _sha256(offline_result_path)
    support_sha = _sha256(support_audit_path)
    freeze_sha = _sha256(freeze_audit_path)
    environment_sha = _sha256(environment_protocol_path)
    environment_parent_sha = _sha256(environment_parent_path)
    manifest_sha = _sha256(external_manifest_path)
    conversion_manifest_sha = _sha256(conversion_manifest_path)
    expected_tree_sha = str(protocol["environment"]["conversion_tree_sha256"])
    observed_tree_sha = _tree_sha256(Path(conversion_root))
    specs = tuple(launch.get("specs", ()))
    spec_by_shard = {
        int(str(spec.get("signature", "")).rsplit("-", 1)[-1]): spec
        for spec in specs
        if str(spec.get("signature", "")).rsplit("-", 1)[-1].isdigit()
    }
    direct_rows = tuple(launch.get("direct_results", ()))
    input_hashes = dict(launch.get("input_hashes", {}))
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol") == RESULT_PROTOCOL,
        "development_protocol_matches": protocol.get("protocol") == PROTOCOL,
        "protocol_hash_matches": input_hashes.get("protocol") == protocol_sha,
        "parent_hash_matches": input_hashes.get("parent_protocol")
        == parent_sha
        == str(protocol["parent_protocol"]["sha256"]),
        "offline_hash_matches": input_hashes.get("offline_result")
        == offline_sha
        == str(protocol["offline_authorization"]["sha256"]),
        "support_hash_matches": input_hashes.get("support_audit")
        == support_sha
        == str(protocol["support_runtime_equivalence"]["sha256"]),
        "freeze_hash_matches": input_hashes.get("freeze_audit")
        == freeze_sha
        == str(protocol["hierarchical_freeze_audit"]["sha256"]),
        "environment_hash_matches": input_hashes.get("environment_protocol")
        == environment_sha
        == str(protocol["environment"]["protocol"]["sha256"]),
        "environment_parent_hash_matches": input_hashes.get("environment_parent")
        == environment_parent_sha
        == str(protocol["environment"]["network_parent_protocol"]["sha256"]),
        "manifest_hash_matches": input_hashes.get("external_manifest")
        == manifest_sha
        == str(protocol["environment"]["external_manifest"]["sha256"]),
        "conversion_manifest_hash_matches": conversion_manifest_sha
        == str(protocol["environment"]["conversion_manifest"]["sha256"]),
        "submitted_directly": launch.get("submitted") is True
        and launch.get("execution_mode") == "direct_six_physical_nodes",
        "six_unique_physical_node_specs": set(spec_by_shard) == set(range(len(NODES)))
        and {str(spec.get("require_node")) for spec in specs} == set(NODES),
        "six_direct_results_succeeded": len(direct_rows) == len(NODES)
        and {str(row.get("node")) for row in direct_rows} == set(NODES)
        and all(int(row.get("returncode", -1)) == 0 for row in direct_rows),
        "matrix_size_matches": int(launch.get("matrix_size", -1)) == len(expected),
        "snapshot_identifiers_present": bool(launch.get("snapshot_sha256"))
        and bool(launch.get("source_tree_sha256")),
        "remote_preflight_passed": launch.get("remote_preflight", {}).get("passed")
        is True,
        "remote_postflight_passed": launch.get("remote_postflight", {}).get("passed")
        is True,
        "remote_conversion_tree_unchanged": launch.get("remote_preflight", {}).get(
            "conversion_tree_sha256"
        )
        == launch.get("remote_postflight", {}).get("conversion_tree_sha256")
        == expected_tree_sha,
        "remote_result_file_count_matches": int(
            launch.get("remote_postflight", {}).get("result_file_count", -1)
        )
        == 2 * len(expected) + len(NODES),
    }
    launch_gate["passed"] = all(launch_gate.values())

    errors: list[str] = []
    shard_summaries = []
    shard_rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for shard_index in range(len(NODES)):
        path = results_root / "_shards" / f"shard_{shard_index}.json"
        assigned = {
            identity
            for identity, index in identity_index.items()
            if index % len(NODES) == shard_index
        }
        if not path.is_file():
            shard_summaries.append(
                {"shard_index": shard_index, "path": str(path), "passed": False}
            )
            errors.append(f"missing shard summary: {path}")
            continue
        shard = _read_json(path)
        rows = tuple(shard.get("rows", ()))
        observed = {
            (
                str(row.get("city")),
                str(row.get("scenario")),
                int(row.get("seed", -1)),
                str(row.get("policy")),
            )
            for row in rows
        }
        expected_node = str(spec_by_shard.get(shard_index, {}).get("require_node"))
        passed = bool(
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and str(shard.get("hostname")) == expected_node
            and int(shard.get("shard_index", -1)) == shard_index
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == len(assigned)
            and shard.get("protocol_sha256") == protocol_sha
            and shard.get("parent_protocol_sha256") == parent_sha
            and shard.get("offline_result_sha256") == offline_sha
            and shard.get("support_audit_sha256") == support_sha
            and shard.get("freeze_audit_sha256") == freeze_sha
            and observed == assigned
        )
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "hostname": shard.get("hostname"),
                "task_count": len(rows),
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"shard summary gate failed: {shard_index}")
        for row in rows:
            identity = (
                str(row.get("city")),
                str(row.get("scenario")),
                int(row.get("seed", -1)),
                str(row.get("policy")),
            )
            if identity in shard_rows:
                errors.append(f"duplicate shard row: {identity}")
            shard_rows[identity] = dict(row)

    candidates = {
        candidate.key: candidate_payload(candidate)
        for candidate in development_candidates(protocol)
    }
    expected_files = {
        (results_root / "_shards" / f"shard_{index}.json").resolve()
        for index in range(len(NODES))
    }
    result_rows = []
    phase_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    method_rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for city, scenario, seed, policy in identities:
        root = results_root / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        expected_files.update({result_path.resolve(), tripinfo_path.resolve()})
        if not result_path.is_file() or not tripinfo_path.is_file():
            errors.append(f"missing rollout evidence: {city}/{scenario}/{seed}/{policy}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        shard_row = shard_rows.get((city, scenario, seed, policy), {})
        expected_node = str(
            spec_by_shard[identity_index[(city, scenario, seed, policy)] % len(NODES)][
                "require_node"
            ]
        )
        candidate = candidates.get(policy)
        tripinfo_evidence = dict(result.get("tripinfo_evidence", {}))
        safety = {
            key: int(metrics.get(key, -1))
            for key in (
                "starting_teleports",
                "ending_teleports",
                "collision_events",
                "collision_incidents",
            )
        }
        guard = dict(metrics.get("guard_audit", {}))
        executed_overrides = int(
            guard.get("executed_overrides", guard.get("accepted_overrides", 0))
        )
        result_sha = _sha256(result_path)
        tripinfo_sha = _sha256(tripinfo_path)
        expected_suffix = (
            f"/{city}/{scenario}/seed_{seed}/{policy}/result.json"
        )
        row_gate = {
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "stage_matches": result.get("analysis_stage")
            == "hierarchical_closed_loop_development",
            "identity_matches": (
                result.get("city"),
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (city, scenario, seed, policy),
            "selection_scope_matches": result.get("selection_eligible_evidence") is True
            and result.get("untouched_validation_evidence") is False
            and result.get("prospective_evidence") is False,
            "candidate_config_matches": result.get("execution_candidate") == candidate,
            "runtime_policy_matches": result.get("runtime_policy")
            == (PHASE_POLICY if policy == PHASE_POLICY else HIERARCHICAL_RUNTIME_POLICY),
            "controller_not_refit": result.get("controller_refit") is False,
            "physical_host_matches_shard": result.get("hostname") == expected_node
            and result.get("runtime", {}).get("hostname") == expected_node,
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "protocol_hash_matches": result.get("protocol_sha256") == protocol_sha,
            "parent_hash_matches": result.get("parent_protocol_sha256") == parent_sha,
            "offline_hash_matches": result.get("offline_authorization_sha256")
            == offline_sha,
            "support_hash_matches": result.get("support_equivalence_audit_sha256")
            == support_sha,
            "freeze_hash_matches": result.get("hierarchical_freeze_audit_sha256")
            == freeze_sha,
            "environment_hash_matches": result.get("environment_protocol_sha256")
            == environment_sha,
            "environment_parent_hash_matches": result.get(
                "environment_parent_sha256"
            )
            == environment_parent_sha,
            "manifest_hash_matches": result.get("external_manifest_sha256")
            == manifest_sha,
            "conversion_manifest_hash_matches": result.get(
                "conversion_manifest_sha256"
            )
            == conversion_manifest_sha,
            "conversion_tree_matches": result.get("conversion_tree_sha256")
            == expected_tree_sha,
            "sumo_version_matches": result.get("sumo_version")
            == str(protocol["environment"]["sumo_version"])
            and result.get("runtime", {}).get("libsumo_version")
            == str(protocol["environment"]["sumo_version"]),
            "frozen_city_artifacts_match": (
                result.get("model_sha256"), result.get("certificate_sha256")
            )
            == (
                (protocol["city_artifacts"][city]["model"]["sha256"] if candidate else None),
                (
                    protocol["city_artifacts"][city]["certificate"]["sha256"]
                    if candidate
                    else None
                ),
            ),
            "anchored_model_present_only_for_method": (
                result.get("anchored_blend_audit") is not None
            )
            == (candidate is not None),
            "sumo_succeeded": bool(metrics.get("ok", False)),
            "no_teleports": safety["starting_teleports"] == 0
            and safety["ending_teleports"] == 0,
            "tripinfo_hash_matches_result": tripinfo_evidence.get("sha256")
            == tripinfo_sha
            and int(tripinfo_evidence.get("size_bytes", -1))
            == tripinfo_path.stat().st_size,
            "tripinfo_xml_valid": _tripinfo_is_valid(tripinfo_path),
            "result_hash_matches_shard": shard_row.get("result_sha256") == result_sha,
            "tripinfo_hash_matches_shard": shard_row.get("tripinfo_sha256")
            == tripinfo_sha,
            "remote_result_path_matches": str(shard_row.get("result_path", "")).endswith(
                expected_suffix
            ),
            "shard_host_matches": shard_row.get("hostname") == expected_node,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"rollout row gate failed: {city}/{scenario}/{seed}/{policy}")
        row = {
            "city": city,
            "scenario": scenario,
            "seed": seed,
            "policy": policy,
            "hostname": result.get("hostname"),
            "result_path": str(result_path.resolve()),
            "result_sha256": result_sha,
            "tripinfo_path": str(tripinfo_path.resolve()),
            "tripinfo_sha256": tripinfo_sha,
            "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
            "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
            "executed_overrides": executed_overrides,
            "safety": safety,
            "gate": row_gate,
        }
        result_rows.append(row)
        if policy == PHASE_POLICY:
            phase_rows[(city, scenario, seed)] = row
        else:
            method_rows[(city, scenario, seed, policy)] = row

    paired_rows = []
    for city, scenario, seed, policy in identities:
        if policy == PHASE_POLICY:
            continue
        phase = phase_rows.get((city, scenario, seed))
        method = method_rows.get((city, scenario, seed, policy))
        if phase is None or method is None:
            errors.append(f"paired result missing: {city}/{scenario}/{seed}/{policy}")
            continue
        phase_wait = float(phase["mean_waiting_time"])
        paired_rows.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "candidate": policy,
                "method_mean_waiting_time": method["mean_waiting_time"],
                "phase_mean_waiting_time": phase_wait,
                "relative_delta": (
                    float(method["mean_waiting_time"]) - phase_wait
                )
                / max(phase_wait, 1e-12),
                "method_mean_travel_time": method["mean_travel_time"],
                "phase_mean_travel_time": phase["mean_travel_time"],
                "executed_overrides": method["executed_overrides"],
                "method_safety": method["safety"],
                "phase_safety": phase["safety"],
                "method_result_sha256": method["result_sha256"],
                "phase_result_sha256": phase["result_sha256"],
            }
        )

    development = dict(protocol["development"])
    selector = select_common_candidate(
        candidates=[candidates[key] for key in candidates],
        paired_rows=paired_rows,
        city_scenarios=development["city_scenarios"],
        seeds=tuple(int(value) for value in development["seeds"]),
        selection_rule=development["selection_rule"],
        bootstrap=development["bootstrap"],
    )
    validation_seeds = {
        int(value) for value in protocol["validation"]["closed_loop_seeds"]
    }
    prospective_seeds = {
        int(value)
        for value in protocol["prospective_confirmation"]["closed_loop_seeds"]
    }
    forbidden_paths = sorted(
        str(path.resolve())
        for seed in validation_seeds | prospective_seeds
        for path in results_root.rglob(f"seed_{seed}")
    )
    observed_files = {path.resolve() for path in results_root.rglob("*") if path.is_file()}
    unexpected_files = sorted(str(path) for path in observed_files - expected_files)
    missing_expected_files = sorted(str(path) for path in expected_files - observed_files)
    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "local_conversion_tree_matches": observed_tree_sha == expected_tree_sha,
        "all_six_shards_passed": len(shard_summaries) == len(NODES)
        and all(bool(row["passed"]) for row in shard_summaries),
        "shard_rows_cover_exact_matrix": set(shard_rows) == expected,
        "all_rollout_results_present": len(result_rows) == len(expected),
        "all_rollout_row_gates_passed": bool(result_rows)
        and all(bool(row["gate"]["passed"]) for row in result_rows),
        "all_phase_references_present": len(phase_rows)
        == sum(len(values) for values in development["city_scenarios"].values())
        * len(development["seeds"]),
        "all_method_pairs_present": len(paired_rows)
        == (len(expected) - len(phase_rows)),
        "validation_and_prospective_results_absent": not forbidden_paths,
        "result_tree_contains_only_expected_files": not unexpected_files
        and not missing_expected_files,
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    active_selected = selector["decision"] == "freeze_common_active_hierarchical_candidate"
    advance_gate = {
        "integrity_passed": bool(integrity_gate["passed"]),
        "common_active_candidate_selected": active_selected,
        "selected_candidate_feasible_in_every_city": active_selected
        and all(
            bool(value["feasible"])
            for value in selector["selected"].get("city_summaries", {}).values()
        ),
        "validation_seeds_still_untouched": not forbidden_paths,
    }
    advance_gate["passed"] = all(advance_gate.values())
    status = "PASS" if integrity_gate["passed"] else "FAIL"
    decision = ADVANCE_DECISION if advance_gate["passed"] else REJECTION_DECISION
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": decision,
        "claim_boundary": (
            "Disclosed eight-seed closed-loop development evidence only. It may "
            "freeze one common execution candidate for untouched validation, but "
            "it is neither validation nor prospective confirmation."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest_path),
        "development_protocol_sha256": protocol_sha,
        "parent_protocol_sha256": parent_sha,
        "offline_result_sha256": offline_sha,
        "support_audit_sha256": support_sha,
        "freeze_audit_sha256": freeze_sha,
        "environment_protocol_sha256": environment_sha,
        "environment_parent_sha256": environment_parent_sha,
        "external_manifest_sha256": manifest_sha,
        "conversion_manifest_sha256": conversion_manifest_sha,
        "conversion_tree_sha256": observed_tree_sha,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "shards": shard_summaries,
        "rows": result_rows,
        "paired_rows": paired_rows,
        "selector": selector,
        "integrity_gate": integrity_gate,
        "advance_gate": advance_gate,
        "untouched_validation_seeds": sorted(validation_seeds),
        "untouched_prospective_seeds": sorted(prospective_seeds),
        "forbidden_paths": forbidden_paths,
        "unexpected_files": unexpected_files,
        "missing_expected_files": missing_expected_files,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--offline-result", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--environment-protocol", type=Path, required=True)
    parser.add_argument("--environment-parent", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v64 selection audit: {args.out}")
    payload = audit_hierarchical_closed_loop_development(
        results_root=args.results_root,
        launch_manifest_path=args.launch_manifest,
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        offline_result_path=args.offline_result,
        support_audit_path=args.support_audit,
        freeze_audit_path=args.freeze_audit,
        environment_protocol_path=args.environment_protocol,
        environment_parent_path=args.environment_parent,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "selected": payload["selector"]["selected_config"],
                "feasible_common_candidates": payload["selector"][
                    "feasible_common_candidate_count"
                ],
                "integrity": payload["integrity_gate"]["passed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
