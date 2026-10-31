#!/usr/bin/env python3
"""Audit v67 target-veto development and freeze one common cooldown candidate."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_target_veto_closed_loop_development import (  # noqa: E402
    ANALYSIS_STAGE,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_method_rollout_identities,
    candidate_payload,
    development_candidates,
)
from scripts.cluster.audit_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    bootstrap_mean_interval,
)
from scripts.cluster.launch_tsc_external_target_veto_closed_loop_development import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_target_veto_closed_loop_development_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = (
    "tsc-v68r64-external-v9-proposal-conditional-closed-loop-development-selection-audit-v1"
)
ADVANCE_DECISION = (
    "authorize_proposal_conditional_candidate_for_untouched_validation"
)
REJECTION_DECISION = (
    "reject_target_veto_development_and_preserve_validation_and_prospective_seeds"
)
PHASE_POLICY = "phase_pressure"


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


def _tripinfo_is_valid(path: Path) -> bool:
    try:
        return ET.parse(path).getroot().tag == "tripinfos"
    except (ET.ParseError, OSError):
        return False


def _seed_rows(
    *,
    selected: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> tuple[dict[str, Any], list[float]]:
    scenario_set = {str(value) for value in scenarios}
    rows: dict[str, Any] = {}
    for seed in seeds:
        paired = [row for row in selected if int(row["seed"]) == int(seed)]
        if {str(row["scenario"]) for row in paired} != scenario_set:
            raise ValueError(f"scenario pairing incomplete for seed {seed}")
        rows[str(seed)] = {
            "scenario_relative_deltas": {
                str(row["scenario"]): float(row["relative_delta"]) for row in paired
            },
            "city_mean_relative_delta": statistics.fmean(
                float(row["relative_delta"]) for row in paired
            ),
            "executed_overrides": sum(
                int(row["executed_overrides"]) for row in paired
            ),
            "rejected_long_horizon_veto": sum(
                int(row["rejected_long_horizon_veto"]) for row in paired
            ),
        }
    return rows, [float(rows[str(seed)]["city_mean_relative_delta"]) for seed in seeds]


def summarize_active_city(
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
    selected = [
        row
        for row in paired_rows
        if str(row["city"]) == city and str(row["candidate"]) == key
    ]
    if len(selected) != len(scenarios) * len(seeds):
        raise ValueError(f"active candidate pairing incomplete for {city}/{key}")
    seed_rows, deltas = _seed_rows(
        selected=selected, scenarios=scenarios, seeds=seeds
    )
    mean_delta = statistics.fmean(deltas)
    std_delta = statistics.pstdev(deltas)
    worst_seed = max(deltas)
    interval = bootstrap_mean_interval(
        deltas,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    improved_fraction = statistics.fmean(float(value < 0.0) for value in deltas)
    rule = dict(selection_rule["active_city"])
    gates = {
        "minimum_city_mean_improvement": mean_delta
        <= -float(rule["minimum_city_mean_relative_improvement"]),
        "bootstrap_upper_within_limit": interval["upper"]
        <= float(rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed_regression_within_limit": worst_seed
        <= float(rule["maximum_seed_level_city_mean_relative_regression"]),
        "minimum_improved_seed_fraction": improved_fraction
        >= float(rule["minimum_improved_seed_fraction"]),
        "every_scenario_seed_intervened": all(
            int(row["executed_overrides"])
            >= int(rule["minimum_executed_interventions_per_scenario_seed"])
            for row in selected
        ),
        "target_veto_active_in_every_row": all(
            bool(row["target_veto_active"]) for row in selected
        ),
        "teleports_noninferior_to_phase": all(
            int(row["method_safety"]["starting_teleports"])
            <= int(row["phase_safety"]["starting_teleports"])
            and int(row["method_safety"]["ending_teleports"])
            <= int(row["phase_safety"]["ending_teleports"])
            for row in selected
        ),
        "collisions_noninferior_to_phase": all(
            int(row["method_safety"]["collision_incidents"])
            <= int(row["phase_safety"]["collision_incidents"])
            for row in selected
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "role": "oof_active_improvement",
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
        "total_rejected_long_horizon_veto": sum(
            int(row["rejected_long_horizon_veto"]) for row in selected
        ),
        "robust_score": mean_delta + 0.5 * std_delta + max(worst_seed, 0.0),
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def summarize_fallback_city(
    *,
    city: str,
    candidate: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
) -> dict[str, Any]:
    key = str(candidate["key"])
    selected = [
        row
        for row in paired_rows
        if str(row["city"]) == city and str(row["candidate"]) == key
    ]
    if len(selected) != len(scenarios) * len(seeds):
        raise ValueError(f"fallback candidate pairing incomplete for {city}/{key}")
    seed_rows, deltas = _seed_rows(
        selected=selected, scenarios=scenarios, seeds=seeds
    )
    rule = dict(selection_rule["fallback_city"])
    atol = float(rule["mean_waiting_time_must_equal_phase_atol"])
    travel_atol = float(rule["mean_travel_time_must_equal_phase_atol"])
    gates = {
        "zero_executed_interventions": all(
            int(row["executed_overrides"])
            == int(rule["executed_interventions_must_equal"])
            for row in selected
        ),
        "waiting_time_exact_safe_abstention": all(
            abs(
                float(row["method_mean_waiting_time"])
                - float(row["phase_mean_waiting_time"])
            )
            <= atol
            for row in selected
        ),
        "travel_time_exact_safe_abstention": all(
            abs(
                float(row["method_mean_travel_time"])
                - float(row["phase_mean_travel_time"])
            )
            <= travel_atol
            for row in selected
        ),
        "safety_exact_safe_abstention": all(
            row["method_safety"] == row["phase_safety"] for row in selected
        ),
        "target_veto_disabled_in_every_row": all(
            not bool(row["target_veto_active"]) for row in selected
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "role": "oof_fallback_exact_safe_abstention",
        **dict(candidate),
        "paired_scenario_seed_units": len(selected),
        "seed_rows": seed_rows,
        "city_mean_relative_delta": statistics.fmean(deltas),
        "total_executed_overrides": sum(
            int(row["executed_overrides"]) for row in selected
        ),
        "total_rejected_long_horizon_veto": sum(
            int(row["rejected_long_horizon_veto"]) for row in selected
        ),
        "robust_score": 0.0,
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def select_common_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    city_scenarios: Mapping[str, Sequence[str]],
    active_cities: Sequence[str],
    fallback_cities: Sequence[str],
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    active = {str(value) for value in active_cities}
    fallback = {str(value) for value in fallback_cities}
    if not active or active & fallback or active | fallback != set(city_scenarios):
        raise ValueError("active/fallback city partition changed")
    grid = []
    for candidate in candidates:
        city_summaries = {}
        for city, scenarios in city_scenarios.items():
            city_summaries[str(city)] = (
                summarize_active_city(
                    city=str(city),
                    candidate=candidate,
                    paired_rows=paired_rows,
                    scenarios=scenarios,
                    seeds=seeds,
                    selection_rule=selection_rule,
                    bootstrap=bootstrap,
                )
                if str(city) in active
                else summarize_fallback_city(
                    city=str(city),
                    candidate=candidate,
                    paired_rows=paired_rows,
                    scenarios=scenarios,
                    seeds=seeds,
                    selection_rule=selection_rule,
                )
            )
        row = {
            **dict(candidate),
            "city_summaries": city_summaries,
            "all_cities_feasible": all(
                bool(value["feasible"]) for value in city_summaries.values()
            ),
            "equal_active_city_mean_robust_score": statistics.fmean(
                float(city_summaries[city]["robust_score"]) for city in active
            ),
            "equal_active_city_mean_relative_delta": statistics.fmean(
                float(city_summaries[city]["city_mean_relative_delta"])
                for city in active
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
                float(row["equal_active_city_mean_robust_score"]),
                float(row["equal_active_city_mean_relative_delta"]),
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
            "freeze_common_target_veto_candidate"
            if feasible
            else "retain_phase_pressure_fallback"
        ),
        "selected": selected,
        "selected_config": selected_config,
        "selected_config_sha256": _canonical_sha256(selected_config),
        "feasible_common_candidate_count": len(feasible),
        "grid": grid,
    }


def _safety(metrics: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(metrics.get(key, -1))
        for key in (
            "starting_teleports",
            "ending_teleports",
            "collision_events",
            "collision_incidents",
        )
    }


def _load_phase_rows(protocol: Mapping[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    for reference in protocol["development"]["phase_references"]:
        identity = (
            str(reference["city"]),
            str(reference["scenario"]),
            int(reference["seed"]),
        )
        result_path = PROJECT_ROOT / str(reference["result"]["path"])
        tripinfo_path = PROJECT_ROOT / str(reference["tripinfo"]["path"])
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        if not (
            _sha256(result_path) == reference["result"]["sha256"]
            and _sha256(tripinfo_path) == reference["tripinfo"]["sha256"]
            and result.get("policy") == PHASE_POLICY
            and (
                str(result.get("city")),
                str(result.get("scenario")),
                int(result.get("seed", -1)),
            )
            == identity
            and bool(metrics.get("ok", False))
            and _tripinfo_is_valid(tripinfo_path)
        ):
            raise ValueError(f"phase reference changed: {identity}")
        rows[identity] = {
            "city": identity[0],
            "scenario": identity[1],
            "seed": identity[2],
            "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
            "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
            "safety": _safety(metrics),
            "result_sha256": reference["result"]["sha256"],
            "tripinfo_sha256": reference["tripinfo"]["sha256"],
        }
    if len(rows) != int(protocol["development"]["reused_phase_matrix_size"]):
        raise ValueError("phase reference matrix changed")
    return rows


def audit_target_veto_development(
    *,
    results_root: Path,
    launch_manifest_path: Path,
    protocol_path: Path,
    parent_protocol_path: Path,
    collection_protocol_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    veto_joint_audit_path: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
) -> dict[str, Any]:
    results_root = Path(results_root)
    launch = _read_json(launch_manifest_path)
    protocol = _read_json(protocol_path)
    veto_joint_audit = _read_json(veto_joint_audit_path)
    identities = all_method_rollout_identities(protocol)
    identity_index = {identity: index for index, identity in enumerate(identities)}
    expected = set(identities)
    hashes = {
        "protocol": _sha256(protocol_path),
        "parent_protocol": _sha256(parent_protocol_path),
        "collection_protocol": _sha256(collection_protocol_path),
        "proposal_parent_protocol": _sha256(proposal_parent_protocol_path),
        "freeze_audit": _sha256(freeze_audit_path),
        "veto_joint_audit": _sha256(veto_joint_audit_path),
        "external_manifest": _sha256(external_manifest_path),
        "conversion_manifest": _sha256(conversion_manifest_path),
    }
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
        "all_input_hashes_match": input_hashes == hashes,
        "parent_hash_matches": hashes["parent_protocol"]
        == protocol["parent_protocol"]["sha256"],
        "collection_hash_matches": hashes["collection_protocol"]
        == protocol["collection_protocol"]["sha256"],
        "proposal_parent_hash_matches": hashes["proposal_parent_protocol"]
        == protocol["proposal_parent_protocol"]["sha256"],
        "freeze_hash_matches": hashes["freeze_audit"]
        == protocol["hierarchical_freeze_audit"]["sha256"],
        "veto_hash_matches": hashes["veto_joint_audit"]
        == protocol["target_veto_joint_audit"]["sha256"],
        "manifest_hash_matches": hashes["external_manifest"]
        == protocol["environment"]["external_manifest"]["sha256"],
        "conversion_manifest_hash_matches": hashes["conversion_manifest"]
        == protocol["environment"]["conversion_manifest"]["sha256"],
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
            errors.append(f"missing shard summary: {path}")
            shard_summaries.append({"shard_index": shard_index, "passed": False})
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
            and shard.get("protocol_sha256") == hashes["protocol"]
            and shard.get("parent_protocol_sha256") == hashes["parent_protocol"]
            and shard.get("collection_protocol_sha256")
            == hashes["collection_protocol"]
            and shard.get("proposal_parent_protocol_sha256")
            == hashes["proposal_parent_protocol"]
            and shard.get("freeze_audit_sha256") == hashes["freeze_audit"]
            and shard.get("veto_joint_audit_sha256") == hashes["veto_joint_audit"]
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
    method_rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    result_rows = []
    for identity in identities:
        city, scenario, seed, policy = identity
        root = results_root / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        expected_files.update({result_path.resolve(), tripinfo_path.resolve()})
        if not result_path.is_file() or not tripinfo_path.is_file():
            errors.append(f"missing rollout evidence: {identity}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        guard = dict(metrics.get("guard_audit", {}))
        shard_row = shard_rows.get(identity, {})
        expected_node = str(
            spec_by_shard[identity_index[identity] % len(NODES)]["require_node"]
        )
        result_sha = _sha256(result_path)
        tripinfo_sha = _sha256(tripinfo_path)
        row_gate = {
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "stage_matches": result.get("analysis_stage") == ANALYSIS_STAGE,
            "identity_matches": (
                result.get("city"),
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == identity,
            "candidate_matches": result.get("execution_candidate")
            == candidates[policy],
            "host_matches_physical_shard": result.get("hostname") == expected_node,
            "shard_hash_matches": shard_row.get("result_sha256") == result_sha
            and shard_row.get("tripinfo_sha256") == tripinfo_sha,
            "evidence_chain_matches": result.get("protocol_sha256")
            == hashes["protocol"]
            and result.get("parent_protocol_sha256") == hashes["parent_protocol"]
            and result.get("collection_protocol_sha256")
            == hashes["collection_protocol"]
            and result.get("proposal_parent_protocol_sha256")
            == hashes["proposal_parent_protocol"]
            and result.get("hierarchical_freeze_audit_sha256")
            == hashes["freeze_audit"]
            and result.get("target_veto_joint_audit_sha256")
            == hashes["veto_joint_audit"]
            and result.get("external_manifest_sha256")
            == hashes["external_manifest"]
            and result.get("conversion_manifest_sha256")
            == hashes["conversion_manifest"],
            "environment_matches": result.get("conversion_tree_sha256")
            == expected_tree_sha
            and result.get("sumo_version")
            == protocol["environment"]["sumo_version"],
            "runtime_executable_sources_match": bool(
                result.get("runtime_executable_source_gate", {})
            )
            and all(
                bool(value)
                for value in result.get(
                    "runtime_executable_source_gate", {}
                ).values()
            )
            and set(result.get("runtime_executable_source_gate", {}))
            == set(protocol["runtime_executable_sources"]),
            "no_refit": result.get("controller_refit") is False,
            "classification_matches_city_freeze": result.get(
                "target_veto_classification"
            )
            == "target_simulator_labeled_few_shot_adaptation",
            "active_flag_matches_protocol": bool(result.get("target_veto_active"))
            == bool(protocol["city_target_veto_artifacts"][city]["active"])
            == bool(veto_joint_audit["cities"][city]["active"]),
            "metrics_valid": bool(metrics.get("ok", False)),
            "no_teleports": int(metrics.get("starting_teleports", -1)) == 0
            and int(metrics.get("ending_teleports", -1)) == 0,
            "tripinfo_hash_matches": result.get("tripinfo_evidence", {}).get(
                "sha256"
            )
            == tripinfo_sha,
            "tripinfo_xml_valid": _tripinfo_is_valid(tripinfo_path),
        }
        row_gate["passed"] = all(row_gate.values())
        row = {
            "city": city,
            "scenario": scenario,
            "seed": seed,
            "candidate": policy,
            "hostname": result.get("hostname"),
            "target_veto_active": bool(result.get("target_veto_active")),
            "result_path": str(result_path.resolve()),
            "result_sha256": result_sha,
            "tripinfo_path": str(tripinfo_path.resolve()),
            "tripinfo_sha256": tripinfo_sha,
            "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
            "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
            "executed_overrides": int(
                guard.get("executed_overrides", guard.get("accepted_overrides", 0))
            ),
            "rejected_long_horizon_veto": int(
                guard.get("rejected_long_horizon_veto", 0)
            ),
            "safety": _safety(metrics),
            "gate": row_gate,
        }
        result_rows.append(row)
        method_rows[identity] = row

    phase_rows = _load_phase_rows(protocol)
    paired_rows = []
    for identity in identities:
        city, scenario, seed, policy = identity
        method = method_rows.get(identity)
        phase = phase_rows.get((city, scenario, seed))
        if method is None or phase is None:
            errors.append(f"paired result missing: {identity}")
            continue
        phase_wait = float(phase["mean_waiting_time"])
        paired_rows.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "candidate": policy,
                "target_veto_active": method["target_veto_active"],
                "method_mean_waiting_time": method["mean_waiting_time"],
                "phase_mean_waiting_time": phase_wait,
                "relative_delta": (float(method["mean_waiting_time"]) - phase_wait)
                / max(phase_wait, 1e-12),
                "method_mean_travel_time": method["mean_travel_time"],
                "phase_mean_travel_time": phase["mean_travel_time"],
                "executed_overrides": method["executed_overrides"],
                "rejected_long_horizon_veto": method[
                    "rejected_long_horizon_veto"
                ],
                "method_safety": method["safety"],
                "phase_safety": phase["safety"],
                "method_result_sha256": method["result_sha256"],
                "phase_result_sha256": phase["result_sha256"],
            }
        )

    development = dict(protocol["development"])
    selector = (
        select_common_candidate(
            candidates=[candidates[key] for key in candidates],
            paired_rows=paired_rows,
            city_scenarios=development["city_scenarios"],
            active_cities=development["active_cities"],
            fallback_cities=development["fallback_cities"],
            seeds=tuple(int(value) for value in development["seeds"]),
            selection_rule=development["selection_rule"],
            bootstrap=development["bootstrap"],
        )
        if len(paired_rows) == len(expected)
        else {
            "decision": "incomplete_matrix",
            "selected": {},
            "selected_config": {},
            "selected_config_sha256": None,
            "feasible_common_candidate_count": 0,
            "grid": [],
        }
    )
    validation_seeds = {int(value) for value in protocol["validation"]["closed_loop_seeds"]}
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
        "shard_rows_cover_exact_method_matrix": set(shard_rows) == expected,
        "all_method_results_present": len(result_rows) == len(expected),
        "all_method_row_gates_passed": bool(result_rows)
        and all(bool(row["gate"]["passed"]) for row in result_rows),
        "all_phase_references_present": len(phase_rows)
        == int(development["reused_phase_matrix_size"]),
        "all_method_pairs_present": len(paired_rows) == len(expected),
        "validation_and_prospective_results_absent": not forbidden_paths,
        "result_tree_contains_only_expected_files": not unexpected_files
        and not missing_expected_files,
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    active_selected = selector["decision"] == "freeze_common_target_veto_candidate"
    advance_gate = {
        "integrity_passed": bool(integrity_gate["passed"]),
        "common_target_veto_candidate_selected": active_selected,
        "selected_candidate_satisfies_every_city_contract": active_selected
        and all(
            bool(value["feasible"])
            for value in selector["selected"].get("city_summaries", {}).values()
        ),
        "at_least_one_oof_active_city": bool(development["active_cities"]),
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
            "Disclosed eight-seed target-veto development only. OOF-active cities "
            "test improvement; OOF-fallback cities test exact safe abstention."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest_path),
        "input_hashes": hashes,
        "conversion_tree_sha256": observed_tree_sha,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "shards": shard_summaries,
        "rows": result_rows,
        "phase_references": list(phase_rows.values()),
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
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--veto-joint-audit", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v67 selection audit: {args.out}")
    payload = audit_target_veto_development(
        results_root=args.results_root,
        launch_manifest_path=args.launch_manifest,
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        collection_protocol_path=args.collection_protocol,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        freeze_audit_path=args.freeze_audit,
        veto_joint_audit_path=args.veto_joint_audit,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        conversion_manifest_path=args.conversion_manifest,
    )
    from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json

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
