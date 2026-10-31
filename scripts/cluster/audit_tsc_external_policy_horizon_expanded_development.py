#!/usr/bin/env python3
"""Audit v50 development and freeze a ten-seed city selector."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from cf_h2o.eval.traffic_signal_external_policy_horizon_adaptation import (  # noqa: E402
    PHASE_POLICY,
    policy_candidates,
)
from cf_h2o.eval.traffic_signal_external_policy_horizon_expanded_development import (  # noqa: E402
    FAILED_V49_DECISION,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.launch_tsc_external_policy_horizon_expanded_development import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.run_tsc_external_policy_horizon_expanded_development_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v50r46-policy-horizon-expanded-development-selection-audit-v1"
ADVANCE_DECISION = "authorize_frozen_v50_selector_for_new_untouched_validation"
REJECTION_DECISION = "reject_v50_and_preserve_new_validation_and_prospective_seeds"
PREVIEW_DECISION = "preview_only_no_validation_authorization"
V48_AUDIT_PROTOCOL = "tsc-v48r44-policy-horizon-adaptation-selection-audit-v1"
V48_ADVANCE_DECISION = "authorize_frozen_v48_selector_for_untouched_robustness"


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


def summarize_candidate(
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
        if row["city"] == city
        and row["candidate"] == key
        and int(row["seed"]) in set(seeds)
    ]
    expected_count = len(scenarios) * len(seeds)
    if len(selected) != expected_count:
        raise ValueError(f"candidate pairing incomplete for {city}/{key}")
    seed_rows = {}
    for seed in seeds:
        rows = [row for row in selected if int(row["seed"]) == int(seed)]
        method_mean = statistics.fmean(
            float(row["method_mean_waiting_time"]) for row in rows
        )
        phase_mean = statistics.fmean(
            float(row["phase_mean_waiting_time"]) for row in rows
        )
        seed_rows[str(seed)] = {
            "method_mean_waiting_time": method_mean,
            "phase_mean_waiting_time": phase_mean,
            "relative_delta": (method_mean - phase_mean) / max(phase_mean, 1e-12),
            "executed_overrides": sum(int(row["executed_overrides"]) for row in rows),
        }
    deltas = [float(row["relative_delta"]) for row in seed_rows.values()]
    mean_delta = statistics.fmean(deltas)
    std_delta = statistics.pstdev(deltas)
    worst_seed = max(deltas)
    interval = bootstrap_mean_interval(
        deltas,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        confidence=float(bootstrap["confidence"]),
    )
    improved_fraction = statistics.fmean(float(value < 0.0) for value in deltas)
    intervened_fraction = statistics.fmean(
        float(int(row["executed_overrides"]) > 0) for row in seed_rows.values()
    )
    safety_ok = all(
        int(row["method_safety"]["ending_teleports"])
        <= int(row["phase_safety"]["ending_teleports"])
        and int(row["method_safety"]["collision_incidents"])
        <= int(row["phase_safety"]["collision_incidents"])
        for row in selected
    )
    feasible = bool(
        mean_delta
        <= -float(selection_rule["minimum_city_mean_relative_improvement"])
        and interval["upper"]
        <= float(
            selection_rule[
                "maximum_bootstrap_95pct_upper_mean_relative_delta"
            ]
        )
        and worst_seed
        <= float(
            selection_rule[
                "maximum_seed_level_city_mean_relative_regression"
            ]
        )
        and improved_fraction
        >= float(selection_rule["minimum_improved_seed_fraction"])
        and intervened_fraction
        >= float(selection_rule["minimum_intervened_seed_fraction"])
        and safety_ok
    )
    return {
        **dict(candidate),
        "paired_units": len(selected),
        "seed_rows": seed_rows,
        "city_mean_relative_delta": mean_delta,
        "seed_delta_std": std_delta,
        "bootstrap_mean_relative_delta": interval,
        "worst_seed_relative_delta": worst_seed,
        "improved_seed_fraction": improved_fraction,
        "intervened_seed_fraction": intervened_fraction,
        "total_executed_overrides": sum(
            int(row["executed_overrides"]) for row in selected
        ),
        "safety_noninferior": safety_ok,
        "robust_score": mean_delta + 0.5 * std_delta + max(worst_seed, 0.0),
        "feasible": feasible,
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    blend = row.get("blend_weight")
    return (
        float(row["robust_score"]),
        float(row["city_mean_relative_delta"]),
        int(row["total_executed_overrides"]),
        float(blend) if blend is not None else float("inf"),
        -int(row["cooldown_intervals"]),
        str(row["key"]),
    )


def select_city(
    *,
    city: str,
    candidates: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    grid = [
        summarize_candidate(
            city=city,
            candidate=candidate,
            paired_rows=paired_rows,
            scenarios=scenarios,
            seeds=seeds,
            selection_rule=selection_rule,
            bootstrap=bootstrap,
        )
        for candidate in candidates
    ]
    feasible = [row for row in grid if bool(row["feasible"])]
    selected = min(feasible, key=_selection_key) if feasible else {
        "key": PHASE_POLICY,
        "family": PHASE_POLICY,
        "blend_weight": None,
        "risk_multiplier": None,
        "min_context_trust": None,
        "cooldown_intervals": 0,
        "feasible": True,
        "fallback": True,
    }
    return {
        "decision": (
            "deploy_active_cfcmt_candidate"
            if feasible
            else "deploy_phase_pressure_fallback"
        ),
        "selected": selected,
        "feasible_active_candidate_count": len(feasible),
        "grid": grid,
    }


def _verify_prior_audit_files(audit: Mapping[str, Any]) -> list[str]:
    errors = []
    for row in audit.get("rows", ()):
        result_path = Path(str(row["result_path"]))
        tripinfo_path = result_path.parent / "tripinfo.xml"
        if not result_path.is_file() or _sha256(result_path) != row.get("result_sha256"):
            errors.append(f"prior result hash failed: {result_path}")
        if not tripinfo_path.is_file() or _sha256(tripinfo_path) != row.get(
            "tripinfo_sha256"
        ):
            errors.append(f"prior tripinfo hash failed: {tripinfo_path}")
    return errors


def _reference_row(row: Mapping[str, Any], *, candidate: str) -> dict[str, Any]:
    return {
        "city": str(row["city"]),
        "scenario": str(row["scenario"]),
        "seed": int(row["seed"]),
        "candidate": candidate,
        "mean_waiting_time": float(row["mean_waiting_time"]),
        "mean_travel_time": float(row["mean_travel_time"]),
        "executed_overrides": int(row["executed_overrides"]),
        "safety": dict(row["safety"]),
        "result_sha256": str(row["result_sha256"]),
        "tripinfo_sha256": str(row["tripinfo_sha256"]),
        "evidence_source": "prior_hash_audited_local_artifact",
    }


def audit_expanded_development(
    *,
    expanded_results_root: Path,
    launch_manifest: Path,
    expanded_protocol_path: Path,
    adaptation_protocol_path: Path,
    pressure_protocol_path: Path,
    adaptation_audit_path: Path,
    failed_v49_audit_path: Path,
    pressure_joint_audit_path: Path,
    preview: bool,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(expanded_protocol_path)
    adaptation = _read_json(adaptation_protocol_path)
    pressure = _read_json(pressure_protocol_path)
    adaptation_audit = _read_json(adaptation_audit_path)
    failed = _read_json(failed_v49_audit_path)
    identities = all_rollout_identities(protocol, adaptation)
    expected = {(scenario, seed, candidate) for _, scenario, seed, candidate in identities}
    identity_index = {
        (scenario, seed, candidate): index
        for index, (_, scenario, seed, candidate) in enumerate(identities)
    }
    specs = tuple(launch.get("specs", ()))
    spec_by_shard = {
        int(str(spec["signature"]).rsplit("-", 1)[-1]): spec for spec in specs
    }
    expanded_sha = _sha256(expanded_protocol_path)
    adaptation_sha = _sha256(adaptation_protocol_path)
    pressure_sha = _sha256(pressure_protocol_path)
    failed_sha = _sha256(failed_v49_audit_path)
    joint_sha = _sha256(pressure_joint_audit_path)
    command_text = "\n".join(str(spec.get("cmd", "")) for spec in specs)
    validation_seeds = {
        int(value) for value in protocol["new_untouched_validation"]["seeds"]
    }
    prospective_seeds = {
        int(value)
        for value in protocol["prospective_confirmation_reservation"]["seeds"]
    }
    launch_gate = {
        "launch_protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "result_protocol_matches": launch.get("result_protocol") == RESULT_PROTOCOL,
        "expanded_protocol_matches": protocol.get("protocol") == PROTOCOL,
        "expanded_protocol_hash_matches": launch.get("input_hashes", {}).get(
            "expanded_protocol"
        )
        == expanded_sha,
        "adaptation_protocol_hash_matches": adaptation_sha
        == str(protocol["candidate_grid_protocol"]["sha256"])
        == str(launch.get("input_hashes", {}).get("adaptation_protocol")),
        "pressure_protocol_hash_matches": pressure_sha
        == str(adaptation["parent_pressure_protocol"]["sha256"])
        == str(launch.get("input_hashes", {}).get("pressure_protocol")),
        "failed_v49_audit_matches": failed.get("status") == "PASS"
        and failed.get("decision") == FAILED_V49_DECISION
        and failed_sha == str(protocol["failed_v49_robustness_audit"]["sha256"])
        == str(launch.get("input_hashes", {}).get("failed_v49_audit")),
        "pressure_joint_hash_matches": joint_sha
        == str(adaptation["model_artifacts"]["pressure_joint_audit_sha256"])
        == str(launch.get("input_hashes", {}).get("pressure_joint_audit")),
        "adaptation_audit_valid": adaptation_audit.get("protocol")
        == V48_AUDIT_PROTOCOL
        and adaptation_audit.get("status") == "PASS"
        and adaptation_audit.get("decision") == V48_ADVANCE_DECISION
        and adaptation_audit.get("policy_protocol_sha256") == adaptation_sha,
        "submitted": bool(launch.get("submitted", False)),
        "scheduler_returncode_zero": int(launch.get("scheduler_returncode", -1))
        == 0,
        "six_unique_node_shards": set(spec_by_shard) == set(range(len(NODES)))
        and {str(spec["require_node"]) for spec in specs} == set(NODES),
        "matrix_size_matches": int(launch.get("matrix_size", -1)) == len(expected),
        "sealed_seed_literals_absent_from_commands": not any(
            str(seed) in command_text for seed in validation_seeds | prospective_seeds
        ),
    }
    launch_gate["passed"] = all(launch_gate.values())

    shard_summaries = []
    shard_identity_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for shard_index in range(len(NODES)):
        path = Path(expanded_results_root) / "_shards" / f"shard_{shard_index}.json"
        if not path.is_file():
            shard_summaries.append({"shard_index": shard_index, "passed": False})
            continue
        shard = _read_json(path)
        assigned = {
            key
            for key, index in identity_index.items()
            if index % len(NODES) == shard_index
        }
        rows = tuple(shard.get("rows", ()))
        observed = {
            (str(row["scenario"]), int(row["seed"]), str(row["candidate"]))
            for row in rows
        }
        passed = bool(
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and shard.get("hostname") == spec_by_shard[shard_index].get("require_node")
            and int(shard.get("shard_count", -1)) == len(NODES)
            and int(shard.get("task_count", -1)) == len(assigned)
            and shard.get("expanded_protocol_sha256") == expanded_sha
            and shard.get("adaptation_protocol_sha256") == adaptation_sha
            and shard.get("failed_v49_audit_sha256") == failed_sha
            and shard.get("pressure_joint_audit_sha256") == joint_sha
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
        for row in rows:
            key = (str(row["scenario"]), int(row["seed"]), str(row["candidate"]))
            if key in shard_identity_rows:
                raise ValueError(f"duplicate v50 shard identity: {key}")
            shard_identity_rows[key] = dict(row)

    errors = _verify_prior_audit_files(adaptation_audit)
    errors.extend(_verify_prior_audit_files(failed))
    expanded_rows = []
    candidate_specs = {
        (city, candidate.key): asdict(candidate)
        for city in protocol["city_scenarios"]
        for candidate in policy_candidates(adaptation, str(city))
        if candidate.key != PHASE_POLICY
    }
    for city, scenario, seed, candidate in identities:
        root = Path(expanded_results_root) / scenario / f"seed_{seed}" / candidate
        result_path = root / "result.json"
        shard_row = shard_identity_rows.get((scenario, seed, candidate), {})
        if not result_path.is_file():
            errors.append(f"missing v50 result: {scenario}/{seed}/{candidate}")
            continue
        result = _read_json(result_path)
        metrics = dict(result.get("metrics", {}))
        result_sha = _sha256(result_path)
        tripinfo_evidence = dict(result.get("tripinfo_evidence", {}))
        expected_host = str(
            spec_by_shard[identity_index[(scenario, seed, candidate)] % len(NODES)][
                "require_node"
            ]
        )
        expected_spec = candidate_specs[(city, candidate)]
        expected_relative = f"/{scenario}/seed_{seed}/{candidate}/result.json"
        row_gate = {
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "stage_matches": result.get("analysis_stage")
            == "expanded_policy_horizon_development",
            "identity_matches": (
                result.get("city"),
                result.get("scenario"),
                int(result.get("seed", -1)),
                result.get("policy"),
            )
            == (city, scenario, seed, candidate),
            "candidate_spec_matches": result.get("policy_horizon_candidate")
            == expected_spec,
            "physical_host_matches_shard": result.get("hostname") == expected_host,
            "source_tree_matches_snapshot": result.get("runtime", {}).get(
                "source_tree_sha256"
            )
            == launch.get("source_tree_sha256"),
            "expanded_protocol_hash_matches": result.get(
                "expanded_development_protocol_sha256"
            )
            == expanded_sha,
            "candidate_grid_hash_matches": result.get(
                "candidate_grid_protocol_sha256"
            )
            == adaptation_sha,
            "robustness_protocol_hash_matches": result.get(
                "failed_robustness_protocol_sha256"
            )
            == str(protocol["parent_robustness_protocol"]["sha256"]),
            "failed_v49_hash_matches": result.get(
                "failed_v49_robustness_audit_sha256"
            )
            == failed_sha,
            "joint_hash_matches": result.get(
                "pressure_regularized_joint_audit_sha256"
            )
            == joint_sha,
            "certificate_hash_matches": result.get(
                "pressure_regularized_certificate_sha256"
            )
            == pressure["city_artifacts"][city]["certificate"]["sha256"],
            "model_hash_matches": result.get("pressure_regularized_model_sha256")
            == pressure["city_artifacts"][city]["model"]["sha256"],
            "result_hash_matches_shard": shard_row.get("result_sha256")
            == result_sha,
            "tripinfo_hash_matches_shard": shard_row.get("tripinfo_sha256")
            == tripinfo_evidence.get("sha256"),
            "remote_result_path_matches": str(shard_row.get("result_path", "")).endswith(
                expected_relative
            ),
            "sumo_succeeded": bool(metrics.get("ok", False)),
            "selection_eligible": bool(result.get("selection_eligible_evidence")),
            "not_prospective": result.get("prospective_evidence") is False,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"v50 row gate failed: {scenario}/{seed}/{candidate}")
        expanded_rows.append(
            {
                "city": city,
                "scenario": scenario,
                "seed": seed,
                "candidate": candidate,
                "candidate_spec": expected_spec,
                "hostname": result.get("hostname"),
                "result_path": str(result_path.resolve()),
                "result_sha256": result_sha,
                "remote_tripinfo_path": shard_row.get("tripinfo_path"),
                "tripinfo_sha256": tripinfo_evidence.get("sha256"),
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
                "gate": row_gate,
            }
        )

    original_seeds = {
        int(value)
        for value in protocol["expanded_development_grid"]["original_adaptation_seeds"]
    }
    additional_seeds = {
        int(value)
        for value in protocol["expanded_development_grid"]["additional_seeds"]
    }
    active_rows: dict[tuple[str, int, str], dict[str, Any]] = {
        (row["scenario"], int(row["seed"]), row["candidate"]): row
        for row in expanded_rows
    }
    phase_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in adaptation_audit.get("rows", ()):
        seed = int(row["seed"])
        candidate = str(row["candidate"])
        if seed not in original_seeds:
            continue
        normalized = _reference_row(row, candidate=candidate)
        if candidate == PHASE_POLICY:
            phase_rows[(str(row["scenario"]), seed)] = normalized
        else:
            active_rows[(str(row["scenario"]), seed, candidate)] = normalized
    for row in failed.get("rows", ()):
        seed = int(row["seed"])
        if seed in additional_seeds and str(row["policy"]) == PHASE_POLICY:
            phase_rows[(str(row["scenario"]), seed)] = _reference_row(
                row, candidate=PHASE_POLICY
            )

    all_seeds = tuple(
        int(value)
        for value in protocol["expanded_development_grid"]["learning_curve_seed_order"]
    )
    paired_rows = []
    for city, scenarios in protocol["city_scenarios"].items():
        for scenario in scenarios:
            for seed in all_seeds:
                phase = phase_rows.get((str(scenario), seed))
                for candidate in policy_candidates(adaptation, str(city)):
                    if candidate.key == PHASE_POLICY:
                        continue
                    method = active_rows.get((str(scenario), seed, candidate.key))
                    if phase is None or method is None:
                        errors.append(
                            f"combined pairing missing: {scenario}/{seed}/{candidate.key}"
                        )
                        continue
                    paired_rows.append(
                        {
                            "city": str(city),
                            "scenario": str(scenario),
                            "seed": seed,
                            "candidate": candidate.key,
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
                            "method_evidence_source": method["evidence_source"]
                            if "evidence_source" in method
                            else "v50_remote_tripinfo_hash_bound_result",
                            "phase_evidence_source": phase["evidence_source"],
                        }
                    )

    expanded = dict(protocol["expanded_development_grid"])
    selection_rule = dict(expanded["selection_rule"])
    bootstrap = dict(expanded["bootstrap"])
    candidates_by_city = {
        str(city): [
            asdict(candidate)
            for candidate in policy_candidates(adaptation, str(city))
            if candidate.key != PHASE_POLICY
        ]
        for city in protocol["city_scenarios"]
    }
    city_selection = {
        str(city): select_city(
            city=str(city),
            candidates=candidates_by_city[str(city)],
            paired_rows=paired_rows,
            scenarios=tuple(str(value) for value in scenarios),
            seeds=all_seeds,
            selection_rule=selection_rule,
            bootstrap=bootstrap,
        )
        for city, scenarios in protocol["city_scenarios"].items()
    }

    learning_curve = {}
    for city, scenarios in protocol["city_scenarios"].items():
        city_curve = []
        for budget in expanded["learning_curve_budgets"]:
            budget = int(budget)
            prefix = all_seeds[:budget]
            selection = select_city(
                city=str(city),
                candidates=candidates_by_city[str(city)],
                paired_rows=paired_rows,
                scenarios=tuple(str(value) for value in scenarios),
                seeds=prefix,
                selection_rule=selection_rule,
                bootstrap=bootstrap,
            )
            selected_key = str(selection["selected"]["key"])
            next_seeds = all_seeds[budget : min(budget + 2, len(all_seeds))]
            next_block = None
            if next_seeds:
                if selected_key == PHASE_POLICY:
                    next_block = {
                        "seeds": list(next_seeds),
                        "mean_relative_delta": 0.0,
                        "fallback": True,
                    }
                else:
                    diagnostic = summarize_candidate(
                        city=str(city),
                        candidate=next(
                            candidate
                            for candidate in candidates_by_city[str(city)]
                            if candidate["key"] == selected_key
                        ),
                        paired_rows=paired_rows,
                        scenarios=tuple(str(value) for value in scenarios),
                        seeds=next_seeds,
                        selection_rule=selection_rule,
                        bootstrap=bootstrap,
                    )
                    next_block = {
                        "seeds": list(next_seeds),
                        "mean_relative_delta": diagnostic[
                            "city_mean_relative_delta"
                        ],
                        "worst_seed_relative_delta": diagnostic[
                            "worst_seed_relative_delta"
                        ],
                        "fallback": False,
                    }
            city_curve.append(
                {
                    "budget": budget,
                    "seeds": list(prefix),
                    "selected_key": selected_key,
                    "feasible_active_candidate_count": selection[
                        "feasible_active_candidate_count"
                    ],
                    "selected_in_prefix_summary": selection["selected"],
                    "next_two_seed_development_diagnostic": next_block,
                }
            )
        learning_curve[str(city)] = city_curve

    active_cities = [
        city
        for city, selection in city_selection.items()
        if selection["decision"] == "deploy_active_cfcmt_candidate"
    ]
    selected_tripinfo = []
    selected_tripinfo_ok = True
    for city in active_cities:
        candidate = str(city_selection[city]["selected"]["key"])
        for scenario in protocol["city_scenarios"][city]:
            for seed in sorted(additional_seeds):
                row = active_rows[(str(scenario), seed, candidate)]
                path = (
                    Path(expanded_results_root)
                    / str(scenario)
                    / f"seed_{seed}"
                    / candidate
                    / "tripinfo.xml"
                )
                exists = path.is_file()
                digest = _sha256(path) if exists else None
                passed = exists and digest == row["tripinfo_sha256"]
                semantic = _tripinfo_semantic_sha256(path) if passed else None
                selected_tripinfo_ok = selected_tripinfo_ok and passed
                selected_tripinfo.append(
                    {
                        "city": city,
                        "scenario": str(scenario),
                        "seed": seed,
                        "candidate": candidate,
                        "path": str(path.resolve()),
                        "sha256": digest,
                        "semantic_sha256": semantic,
                        "passed": passed,
                    }
                )

    forbidden_paths = [
        str(path.resolve())
        for seed in sorted(validation_seeds | prospective_seeds)
        for path in Path(expanded_results_root).rglob(f"seed_{seed}")
    ]
    expected_pair_count = sum(
        len(scenarios) for scenarios in protocol["city_scenarios"].values()
    ) * len(all_seeds) * int(
        protocol["candidate_grid_protocol"]["active_candidate_count_per_city"]
    )
    duplicate_consistency = []
    frozen_v49_la_key = str(
        _read_json(
            Path(protocol["parent_robustness_protocol"]["path"])
        )["frozen_city_deployment"]["los_angeles"]["key"]
    )
    prior_method = {
        (str(row["scenario"]), int(row["seed"])): row
        for row in failed.get("rows", ())
        if str(row.get("city")) == "los_angeles"
        and str(row.get("policy")) == "cfcmt_policy_horizon_selected_mpc"
    }
    for seed in sorted(additional_seeds):
        current = active_rows.get(("la_1x4", seed, frozen_v49_la_key))
        prior = prior_method.get(("la_1x4", seed))
        current_tripinfo = (
            Path(expanded_results_root)
            / "la_1x4"
            / f"seed_{seed}"
            / frozen_v49_la_key
            / "tripinfo.xml"
        )
        current_semantic_sha = (
            _tripinfo_semantic_sha256(current_tripinfo)
            if current_tripinfo.is_file()
            else None
        )
        passed = bool(
            current
            and prior
            and float(current["mean_waiting_time"]) == float(prior["mean_waiting_time"])
            and float(current["mean_travel_time"]) == float(prior["mean_travel_time"])
            and int(current["executed_overrides"]) == int(prior["executed_overrides"])
            and current["safety"] == prior["safety"]
            and current_semantic_sha == prior["tripinfo_semantic_sha256"]
        )
        duplicate_consistency.append(
            {
                "seed": seed,
                "candidate": frozen_v49_la_key,
                "current_raw_tripinfo_sha256": current["tripinfo_sha256"]
                if current
                else None,
                "prior_raw_tripinfo_sha256": prior["tripinfo_sha256"]
                if prior
                else None,
                "current_semantic_tripinfo_sha256": current_semantic_sha,
                "prior_semantic_tripinfo_sha256": prior[
                    "tripinfo_semantic_sha256"
                ]
                if prior
                else None,
                "raw_hash_expected_to_differ_due_to_sumo_generation_header": True,
                "passed": passed,
            }
        )

    core_integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_shards_passed": len(shard_summaries) == len(NODES)
        and all(row["passed"] for row in shard_summaries),
        "shard_rows_cover_matrix": set(shard_identity_rows) == expected,
        "all_expanded_results_present": len(expanded_rows) == len(expected),
        "all_expanded_result_gates_passed": bool(expanded_rows)
        and all(row["gate"]["passed"] for row in expanded_rows),
        "prior_artifacts_still_match_frozen_hashes": not any(
            error.startswith("prior ") for error in errors
        ),
        "all_combined_pairs_present": len(paired_rows) == expected_pair_count,
        "v49_duplicate_candidate_reproduced_exactly": bool(duplicate_consistency)
        and all(row["passed"] for row in duplicate_consistency),
        "sealed_seed_results_absent": not forbidden_paths,
        "no_errors": not errors,
    }
    core_integrity_gate["passed"] = all(core_integrity_gate.values())
    integrity_gate = {
        **core_integrity_gate,
        "selected_expanded_tripinfo_locally_verified": selected_tripinfo_ok,
    }
    integrity_gate["passed"] = bool(core_integrity_gate["passed"]) and (
        preview or selected_tripinfo_ok
    )
    advance_gate = {
        "final_not_preview": not preview,
        "integrity_passed": bool(integrity_gate["passed"]),
        "at_least_one_city_selects_active_cfcmt": bool(active_cities),
        "every_city_has_frozen_deployment": len(city_selection)
        == len(protocol["city_scenarios"]),
    }
    advance_gate["passed"] = all(advance_gate.values())
    if preview:
        status = "PREVIEW"
        decision = PREVIEW_DECISION
    else:
        status = "PASS" if integrity_gate["passed"] else "FAIL"
        decision = ADVANCE_DECISION if advance_gate["passed"] else REJECTION_DECISION
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": decision,
        "claim_boundary": (
            "Post-v49 development evidence over ten disclosed development seeds. "
            "It may freeze a selector for new untouched validation, but it is "
            "neither untouched validation nor prospective confirmation."
        ),
        "preview": preview,
        "launch_manifest_sha256": _sha256(launch_manifest),
        "expanded_protocol_sha256": expanded_sha,
        "adaptation_protocol_sha256": adaptation_sha,
        "pressure_protocol_sha256": pressure_sha,
        "adaptation_audit_sha256": _sha256(adaptation_audit_path),
        "failed_v49_audit_sha256": failed_sha,
        "pressure_joint_audit_sha256": joint_sha,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "shards": shard_summaries,
        "expanded_rows": expanded_rows,
        "paired_rows": paired_rows,
        "city_selection": city_selection,
        "learning_curve": learning_curve,
        "active_cities": active_cities,
        "selected_tripinfo": selected_tripinfo,
        "v49_duplicate_candidate_consistency": duplicate_consistency,
        "core_integrity_gate": core_integrity_gate,
        "integrity_gate": integrity_gate,
        "advance_gate": advance_gate,
        "new_untouched_validation_seeds_remain_unrun": sorted(validation_seeds),
        "prospective_confirmation_seeds_remain_unrun": sorted(prospective_seeds),
        "forbidden_paths": forbidden_paths,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-protocol", type=Path, required=True)
    parser.add_argument("--pressure-protocol", type=Path, required=True)
    parser.add_argument("--adaptation-audit", type=Path, required=True)
    parser.add_argument("--failed-v49-audit", type=Path, required=True)
    parser.add_argument("--pressure-joint-audit", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v50 selection audit: {args.out}")
    payload = audit_expanded_development(
        expanded_results_root=args.expanded_results_root,
        launch_manifest=args.launch_manifest,
        expanded_protocol_path=args.expanded_protocol,
        adaptation_protocol_path=args.adaptation_protocol,
        pressure_protocol_path=args.pressure_protocol,
        adaptation_audit_path=args.adaptation_audit,
        failed_v49_audit_path=args.failed_v49_audit,
        pressure_joint_audit_path=args.pressure_joint_audit,
        preview=bool(args.preview),
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": payload["active_cities"],
                "selected": {
                    city: value["selected"]["key"]
                    for city, value in payload["city_selection"].items()
                },
                "core_integrity": payload["core_integrity_gate"]["passed"],
                "selected_tripinfo_verified": payload["integrity_gate"][
                    "selected_expanded_tripinfo_locally_verified"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if args.preview:
        return 0 if payload["core_integrity_gate"]["passed"] else 2
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
