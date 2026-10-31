"""Audit and summarize a zero-shot/few-shot CFCMT TSC budget matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_HIERARCHY_PROTOCOL_V3,
    COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
    STRICT_SAFETY_MONITORING_V3,
)
from cf_h2o.traffic_signal.action_ranker import ACTION_TARGET_SCALE_PROTOCOL


PRIMARY_METRIC = "mean_system_vehicles_per_controlled_lane"
REPORT_METRICS = (
    PRIMARY_METRIC,
    "mean_queue_per_lane",
    "p90_queue_per_lane",
    "completion_ratio",
)
SAFETY_METRICS = (
    "collision_events",
    "collision_incidents",
    "starting_teleports",
    "ending_teleports",
)
COLLISION_AUDIT_METRICS = ("collision_events", "collision_incidents")
TELEPORT_METRICS = ("starting_teleports", "ending_teleports")
UNGUARDED_DIAGNOSTIC_POLICY_SUFFIX = "_contrast_mpc"
TARGET_DEPLOYMENT_FIT_PROTOCOL = (
    "adaptation_only_model_disjoint_calibration_guard_v1"
)
TARGET_GROUP_SELECTION_PROTOCOL = (
    "coverage-first-per-tls-disjoint-seed-role-v1"
)
RULE_POLICIES = (
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
    "selected_source_prior",
)
FIXED_PROTOCOL_POLICIES = (
    "fixed_program",
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
    "selected_source_prior",
)
CORE_COMPARATORS = (
    "selected_source_prior",
    "simulator_contrast_guard",
    "dense_contrast_guard",
    "dense_balanced_advantage_contrast_guard",
    "causal_core_advantage_contrast_guard",
    "causal_balanced_advantage_contrast_guard",
    "causal_rigid_advantage_contrast_guard",
    "cfcmt_mechanism_contrast_regularized",
    "cfcmt_mechanism_contrast_guard",
    "causal_target_adapter_contrast_regularized",
    "causal_target_adapter_contrast_guard",
    "causal_target_only_contrast_regularized",
    "causal_target_only_contrast_guard",
    "cfcmt_fused_contrast_regularized",
    "cfcmt_fused_contrast_guard",
    "cfcmt_fused_contrast_mpc",
    "cfcmt_fused_rigid_contrast_guard",
    "cfcmt_fused_rigid_contrast_mpc",
)
FUSED_PRIMARY_POLICY = "cfcmt_fused_contrast_guard"
TARGET_ONLY_COMPARATOR = "causal_target_only_contrast_guard"
FUSED_RIGID_COMPARATOR = "cfcmt_fused_rigid_contrast_guard"
FUSED_EFFICACY_BUDGETS = (60, 120)
FUSED_CONTRIBUTION_BUDGETS = (16, 32, 60, 120)
RUNTIME_PROTOCOL_KEYS = (
    "git_commit",
    "source_tree_sha256",
    "platform",
    "python_version",
    "python_executable",
    "sumo_binary",
    "sumo_version",
    "libsumo_version",
    "numpy_version",
    "scikit_learn_version",
    "thread_limits",
    "determinism_environment",
)
VARIABLE_SETTING_KEYS = {
    "target_group_budget",
    "target_information_budget",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _is_unguarded_diagnostic_policy(policy: str) -> bool:
    """Return whether a policy is an explicitly unsafe controller ablation."""

    return str(policy).endswith(UNGUARDED_DIAGNOSTIC_POLICY_SUFFIX)


def _protocol_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    setting = {
        str(key): value
        for key, value in dict(result.get("setting", {})).items()
        if key not in VARIABLE_SETTING_KEYS
    }
    runtime = dict(result.get("runtime", {}))
    return {
        "experiment": result.get("experiment"),
        "setting": setting,
        "runtime": {key: runtime.get(key) for key in RUNTIME_PROTOCOL_KEYS},
    }


def _city_first_metric(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    metric: str,
    city_groups: Mapping[str, str],
) -> float:
    network_values: dict[str, list[float]] = {}
    for row in seed_results:
        if str(row.get("policy")) != policy:
            continue
        target = str(row.get("target"))
        network_values.setdefault(target, []).append(float(row["metrics"][metric]))
    grouped: dict[str, list[float]] = {}
    for target, values in network_values.items():
        grouped.setdefault(str(city_groups[target]), []).append(sum(values) / len(values))
    city_values = [sum(values) / len(values) for values in grouped.values()]
    return sum(city_values) / len(city_values)


def _result_row_map(result: Mapping[str, Any], policy: str) -> dict[tuple[str, int], float]:
    return {
        (str(row["target"]), int(row["seed"])): float(row["metrics"][PRIMARY_METRIC])
        for row in result["seed_results"]
        if str(row["policy"]) == policy
    }


def _target_group_ids(result: Mapping[str, Any]) -> dict[str, set[str]] | None:
    collected: dict[str, set[str]] = {}
    for target, diagnostics in dict(result.get("model_diagnostics", {})).items():
        adaptation = dict(diagnostics.get("target_adaptation", {}))
        if "selected_group_ids" not in adaptation:
            return None
        collected[str(target)] = {str(value) for value in adaptation["selected_group_ids"]}
    return collected


def _action_group_seed(group_id: str) -> int | None:
    match = re.search(r":seed(-?\d+):", str(group_id))
    return int(match.group(1)) if match else None


def _audit_single_result(
    result: Mapping[str, Any],
    *,
    expected_source_tree_sha256: str | None = None,
    expected_sumo_version: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    setting = dict(result.get("setting", {}))
    try:
        budget = int(setting["target_group_budget"])
        scenarios = tuple(str(value) for value in setting["scenarios"])
        policies = tuple(str(value) for value in setting["policies"])
        seeds = tuple(int(value) for value in setting["seeds"])
        city_groups = {
            str(key): str(value)
            for key, value in dict(setting["scenario_city_groups"]).items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "passed": False,
            "budget": None,
            "errors": [f"malformed setting: {exc}"],
            "warnings": [],
        }

    if budget < 0:
        errors.append("target budget is negative")
    if len(scenarios) != len(set(scenarios)) or not scenarios:
        errors.append("scenario list is empty or contains duplicates")
    if len(policies) != len(set(policies)) or not policies:
        errors.append("policy list is empty or contains duplicates")
    if len(seeds) != len(set(seeds)) or not seeds:
        errors.append("evaluation seed list is empty or contains duplicates")
    if set(city_groups) != set(scenarios):
        errors.append("scenario-city mapping does not exactly cover the scenario list")
    if set(seeds) & {int(value) for value in setting.get("source_seeds", ())}:
        errors.append("source and evaluation seeds overlap")
    if set(seeds) & {int(value) for value in setting.get("target_calibration_seeds", ())}:
        errors.append("target-calibration and evaluation seeds overlap")
    if str(setting.get("fold_protocol")) != "strict_leave_one_city_group_out":
        errors.append("fold protocol is not strict leave-one-city-group-out")
    if setting.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
        errors.append("SUMO execution protocol is not the audited no-teleport protocol")
    deployment_fit_protocol = setting.get("target_deployment_fit_protocol")
    if deployment_fit_protocol is not None and deployment_fit_protocol != (
        TARGET_DEPLOYMENT_FIT_PROTOCOL
    ):
        errors.append("target deployment fit protocol is not calibration-disjoint")
    action_scale_protocol = setting.get("action_target_scale_protocol")
    if action_scale_protocol is not None and action_scale_protocol != (
        ACTION_TARGET_SCALE_PROTOCOL
    ):
        errors.append("action target scale protocol mismatch")
    hierarchy_protocol = setting.get("cfcmt_hierarchy_protocol")
    if hierarchy_protocol is not None and hierarchy_protocol != (
        CFCMT_HIERARCHY_PROTOCOL_V3
    ):
        errors.append("CFCMT hierarchy protocol mismatch")
    target_group_selection_protocol = setting.get(
        "target_group_selection_protocol"
    )
    if (
        target_group_selection_protocol is not None
        and target_group_selection_protocol != TARGET_GROUP_SELECTION_PROTOCOL
    ):
        errors.append("target group selection protocol mismatch")

    runtime = dict(result.get("runtime", {}))
    if expected_source_tree_sha256 and runtime.get("source_tree_sha256") != expected_source_tree_sha256:
        errors.append("runtime source-tree hash does not match the staged snapshot")
    if expected_sumo_version and str(runtime.get("libsumo_version")) != str(
        expected_sumo_version
    ):
        errors.append("libsumo version does not match the matrix specification")

    source_rule_specs = tuple(
        dict(value) for value in setting.get("source_rule_specs", ())
    )
    source_rule_keys = tuple(str(value.get("key", "")) for value in source_rule_specs)
    source_policy_seeds = tuple(
        int(value) for value in setting.get("source_policy_seeds", ())
    )
    if (
        not source_rule_specs
        or any(not key for key in source_rule_keys)
        or len(source_rule_keys) != len(set(source_rule_keys))
    ):
        errors.append("source pressure-rule specification grid is missing or duplicated")
    if not source_policy_seeds or len(source_policy_seeds) != len(
        set(source_policy_seeds)
    ):
        errors.append("source pressure-rule seed list is missing or duplicated")
    if set(source_policy_seeds) & set(seeds):
        errors.append("source pressure-rule and evaluation seeds overlap")

    source_rule_rows = list(result.get("source_rule_seed_results", ()))
    expected_source_rule_keys = set(
        product(scenarios, source_policy_seeds, source_rule_keys)
    )
    observed_source_rule_keys: set[tuple[str, int, str]] = set()
    duplicate_source_rule_keys: set[tuple[str, int, str]] = set()
    source_rule_failures = 0
    source_rule_protocol_mismatches = 0
    source_rule_safety_mismatches = 0
    source_rule_teleports = 0
    source_rule_values: dict[tuple[str, str], list[float]] = {}
    for row in source_rule_rows:
        try:
            key = (
                str(row["scenario"]),
                int(row["seed"]),
                str(row["policy"]),
            )
            metrics = dict(row["metrics"])
            pressure_spec = dict(row["pressure_spec"])
        except (KeyError, TypeError, ValueError):
            source_rule_failures += 1
            continue
        if key in observed_source_rule_keys:
            duplicate_source_rule_keys.add(key)
        observed_source_rule_keys.add(key)
        if metrics.get("ok") is not True or pressure_spec.get("key") != key[2]:
            source_rule_failures += 1
        if metrics.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
            source_rule_protocol_mismatches += 1
        if metrics.get("strict_safety_monitoring") != STRICT_SAFETY_MONITORING_V3:
            source_rule_safety_mismatches += 1
        for metric in TELEPORT_METRICS:
            value = metrics.get(metric)
            if not _finite(value):
                source_rule_failures += 1
            else:
                source_rule_teleports += int(value)
        value = metrics.get(PRIMARY_METRIC)
        if not _finite(value):
            source_rule_failures += 1
        else:
            source_rule_values.setdefault((key[0], key[2]), []).append(
                float(value)
            )
    if duplicate_source_rule_keys:
        errors.append(
            "duplicate source pressure-rule rollouts: "
            f"{len(duplicate_source_rule_keys)}"
        )
    missing_source_rule_keys = expected_source_rule_keys - observed_source_rule_keys
    extra_source_rule_keys = observed_source_rule_keys - expected_source_rule_keys
    if missing_source_rule_keys:
        errors.append(
            "missing source pressure-rule rollouts: "
            f"{len(missing_source_rule_keys)}"
        )
    if extra_source_rule_keys:
        errors.append(
            "unexpected source pressure-rule rollouts: "
            f"{len(extra_source_rule_keys)}"
        )
    if source_rule_failures:
        errors.append(
            "failed or malformed source pressure-rule rows: "
            f"{source_rule_failures}"
        )
    if source_rule_protocol_mismatches:
        errors.append(
            "source pressure-rule rollouts not reporting the audited SUMO "
            f"protocol: {source_rule_protocol_mismatches}"
        )
    if source_rule_safety_mismatches:
        errors.append(
            "source pressure-rule rollouts not reporting benchmark-aware "
            f"safety: {source_rule_safety_mismatches}"
        )
    if source_rule_teleports:
        errors.append(
            f"source pressure-rule teleport events are nonzero: {source_rule_teleports}"
        )
    source_rule_costs = dict(result.get("source_rule_policy_costs", {}))
    if set(source_rule_costs) != set(scenarios):
        errors.append("source pressure-rule costs do not cover every scenario")
    for scenario in scenarios:
        reported = dict(source_rule_costs.get(scenario, {}))
        if set(reported) != set(source_rule_keys):
            errors.append(
                f"source pressure-rule cost grid is incomplete for {scenario}"
            )
            continue
        for policy in source_rule_keys:
            values = source_rule_values.get((scenario, policy), ())
            if len(values) != len(source_policy_seeds):
                continue
            recomputed = sum(values) / len(values)
            if not _finite(reported.get(policy)) or not math.isclose(
                recomputed,
                float(reported[policy]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"source pressure-rule cost mismatch for {scenario}/{policy}"
                )

    rows = list(result.get("seed_results", ()))
    expected_keys = set(product(scenarios, seeds, policies))
    observed_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: set[tuple[str, int, str]] = set()
    failed_rows = 0
    nonfinite_rows = 0
    protocol_mismatch_rows = 0
    safety_protocol_mismatch_rows = 0
    safety_totals = {metric: 0.0 for metric in SAFETY_METRICS}
    deployable_safety_totals = {metric: 0.0 for metric in SAFETY_METRICS}
    unguarded_diagnostic_safety_totals = {
        metric: 0.0 for metric in SAFETY_METRICS
    }
    emergency_stops = 0.0
    for row in rows:
        try:
            key = (str(row["target"]), int(row["seed"]), str(row["policy"]))
            metrics = dict(row["metrics"])
        except (KeyError, TypeError, ValueError):
            failed_rows += 1
            continue
        if key in observed_keys:
            duplicate_keys.add(key)
        observed_keys.add(key)
        if metrics.get("ok") is not True:
            failed_rows += 1
        if metrics.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
            protocol_mismatch_rows += 1
        if metrics.get("strict_safety_monitoring") != STRICT_SAFETY_MONITORING_V3:
            safety_protocol_mismatch_rows += 1
        if any(not _finite(metrics.get(metric)) for metric in REPORT_METRICS):
            nonfinite_rows += 1
        for metric in SAFETY_METRICS:
            value = metrics.get(metric)
            if not _finite(value):
                nonfinite_rows += 1
            else:
                numeric = float(value)
                safety_totals[metric] += numeric
                safety_partition = (
                    unguarded_diagnostic_safety_totals
                    if _is_unguarded_diagnostic_policy(key[2])
                    else deployable_safety_totals
                )
                safety_partition[metric] += numeric
        value = metrics.get("emergency_stops", 0.0)
        if _finite(value):
            emergency_stops += float(value)

    if duplicate_keys:
        errors.append(f"duplicate evaluation rollouts: {len(duplicate_keys)}")
    missing_keys = expected_keys - observed_keys
    extra_keys = observed_keys - expected_keys
    if missing_keys:
        errors.append(f"missing evaluation rollouts: {len(missing_keys)}")
    if extra_keys:
        errors.append(f"unexpected evaluation rollouts: {len(extra_keys)}")
    if failed_rows:
        errors.append(f"failed or malformed evaluation rows: {failed_rows}")
    if protocol_mismatch_rows:
        errors.append(
            "rollouts not reporting the audited SUMO protocol: "
            f"{protocol_mismatch_rows}"
        )
    if safety_protocol_mismatch_rows:
        errors.append(
            "rollouts not reporting the benchmark-aware safety protocol: "
            f"{safety_protocol_mismatch_rows}"
        )
    if nonfinite_rows:
        errors.append(f"rows with non-finite required metrics: {nonfinite_rows}")
    nonzero_deployable_teleports = {
        key: value
        for key, value in deployable_safety_totals.items()
        if key in TELEPORT_METRICS and value != 0.0
    }
    if nonzero_deployable_teleports:
        errors.append(
            "deployable-policy teleport events are nonzero: "
            f"{nonzero_deployable_teleports}"
        )
    nonzero_deployable_collisions = {
        key: value
        for key, value in deployable_safety_totals.items()
        if key in COLLISION_AUDIT_METRICS and value != 0.0
    }
    if nonzero_deployable_collisions:
        warnings.append(
            "raw benchmark junction collisions were observed and retained for "
            f"paired incident analysis: {nonzero_deployable_collisions}"
        )
    nonzero_unguarded_safety = {
        key: value
        for key, value in unguarded_diagnostic_safety_totals.items()
        if value != 0.0
    }
    if nonzero_unguarded_safety:
        warnings.append(
            "explicitly unguarded MPC ablations produced collision/teleport "
            f"events: {nonzero_unguarded_safety}"
        )
    primary_policy = str(setting.get("primary_reference_policy", ""))
    collision_noninferiority: dict[str, Any] | None = None
    if (
        primary_policy
        and primary_policy in policies
        and "selected_source_prior" in policies
        and not missing_keys
        and not extra_keys
        and not duplicate_keys
        and not failed_rows
    ):
        paired_incidents = {
            policy: {
                (str(row["target"]), int(row["seed"])): int(
                    row["metrics"]["collision_incidents"]
                )
                for row in rows
                if str(row.get("policy")) == policy
            }
            for policy in (primary_policy, "selected_source_prior")
        }
        primary_total = sum(paired_incidents[primary_policy].values())
        reference_total = sum(
            paired_incidents["selected_source_prior"].values()
        )
        collision_noninferiority = {
            "metric": "unique_collision_incidents",
            "primary_policy": primary_policy,
            "reference_policy": "selected_source_prior",
            "primary_total": int(primary_total),
            "reference_total": int(reference_total),
            "paired_rollouts": len(paired_incidents[primary_policy]),
            "absolute_margin_incidents": 0,
            "passed": bool(primary_total <= reference_total),
        }
        if not collision_noninferiority["passed"]:
            errors.append(
                "primary policy fails paired collision-incident noninferiority "
                f"against selected_source_prior: {primary_total} > {reference_total}"
            )
    if emergency_stops:
        warnings.append(f"SUMO emergency-stop events are nonzero: {emergency_stops:g}")

    aggregate = dict(result.get("aggregate", {}))
    for policy in policies:
        metrics = dict(aggregate.get(policy, {}))
        if not _finite(metrics.get(PRIMARY_METRIC)):
            errors.append(f"aggregate primary metric is missing/non-finite for {policy}")
            continue
        if not missing_keys and not extra_keys and not duplicate_keys and not failed_rows:
            recomputed = _city_first_metric(
                rows,
                policy=policy,
                metric=PRIMARY_METRIC,
                city_groups=city_groups,
            )
            if not math.isclose(
                recomputed,
                float(metrics[PRIMARY_METRIC]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                errors.append(f"city-first aggregate mismatch for {policy}")

    source_diagnostics = dict(result.get("source_counterfactual_diagnostics", {}))
    if set(source_diagnostics) != set(scenarios):
        errors.append("source counterfactual diagnostics do not cover every scenario")
    coverage_floor = float(setting.get("min_source_tls_coverage", 0.0))
    for scenario, diagnostics in source_diagnostics.items():
        coverage = diagnostics.get("tls_coverage_fraction")
        if not _finite(coverage) or float(coverage) + 1e-12 < coverage_floor:
            errors.append(f"counterfactual TLS coverage failed for {scenario}")
        replay = diagnostics.get("counterfactual_replay_audit")
        if not isinstance(replay, Mapping) or replay.get("passed") is not True:
            errors.append(f"counterfactual replay audit failed for {scenario}")
        if (
            diagnostics.get("strict_safety_monitoring")
            != STRICT_SAFETY_MONITORING_V3
        ):
            errors.append(
                f"counterfactual strict safety monitoring is absent for {scenario}"
            )
        behavior_safety = dict(diagnostics.get("behavior_safety_audit", {}))
        if (
            behavior_safety.get("no_teleport_passed") is not True
            or int(behavior_safety.get("starting_teleports", -1)) != 0
            or int(behavior_safety.get("ending_teleports", -1)) != 0
        ):
            errors.append(f"counterfactual behavior teleport audit failed for {scenario}")
        counterfactual_safety = dict(
            diagnostics.get("counterfactual_safety_audit", {})
        )
        if (
            counterfactual_safety.get("protocol")
            != COUNTERFACTUAL_SAFETY_PROTOCOL_V3
            or counterfactual_safety.get("no_teleport_passed") is not True
            or counterfactual_safety.get(
                "symmetric_group_censoring_passed"
            )
            is not True
        ):
            errors.append(
                f"counterfactual symmetric safety censor audit failed for {scenario}"
            )
        retained_rows = int(diagnostics.get("rows", -1))
        retained_groups = int(diagnostics.get("groups", -1))
        if int(counterfactual_safety.get("retained_branches", -2)) != retained_rows:
            errors.append(
                f"counterfactual retained-row audit mismatch for {scenario}"
            )
        if int(counterfactual_safety.get("retained_groups", -2)) != retained_groups:
            errors.append(
                f"counterfactual retained-group audit mismatch for {scenario}"
            )
        candidate_groups = int(counterfactual_safety.get("candidate_groups", -1))
        censored_groups = int(counterfactual_safety.get("censored_groups", -1))
        if candidate_groups != retained_groups + censored_groups:
            errors.append(
                f"counterfactual candidate-group accounting mismatch for {scenario}"
            )

    model_diagnostics = dict(result.get("model_diagnostics", {}))
    if set(model_diagnostics) != set(scenarios):
        errors.append("model diagnostics do not cover every target scenario")
    selected_group_counts: list[int] = []
    for target in scenarios:
        diagnostics = dict(model_diagnostics.get(target, {}))
        adaptation = dict(diagnostics.get("target_adaptation", {}))
        target_city = city_groups.get(target)
        expected_heldout = sorted(
            scenario for scenario in scenarios if city_groups.get(scenario) == target_city
        )
        expected_source = sorted(set(scenarios) - set(expected_heldout))
        expected_source_cities = sorted(
            {str(city_groups[scenario]) for scenario in expected_source}
        )
        source_scenarios = [str(value) for value in adaptation.get("source_scenarios", ())]
        reported_source_cities = sorted(
            str(value) for value in adaptation.get("source_city_groups", ())
        )
        heldout = sorted(str(value) for value in adaptation.get("heldout_city_scenarios", ()))
        source_cities = {city_groups.get(source) for source in source_scenarios}
        if target in source_scenarios or target_city in source_cities:
            errors.append(f"target-city leakage in source scenarios for {target}")
        if heldout != expected_heldout:
            errors.append(f"held-out city fold is incomplete for {target}")
        if sorted(source_scenarios) != expected_source:
            errors.append(f"source scenario fold is incomplete for {target}")
        if reported_source_cities != expected_source_cities:
            errors.append(f"source city-group fold is incomplete for {target}")
        if adaptation.get("target_context_transition_labels_used") is not False:
            errors.append(f"target static context label budget is not explicit for {target}")
        if adaptation.get("target_context_provenance") != (
            "static_sumocfg_network_phase_and_route_summary"
        ):
            errors.append(f"target static context provenance mismatch for {target}")
        if adaptation.get("label_type") != (
            "matched_target_simulator_counterfactual_action_outcomes"
        ):
            errors.append(f"target adaptation label type mismatch for {target}")
        if int(adaptation.get("requested_group_budget", -1)) != budget:
            errors.append(f"target budget diagnostic mismatch for {target}")
        selected = int(adaptation.get("selected_groups", -1))
        adapted = int(adaptation.get("adaptation_groups", -1))
        calibrated = int(adaptation.get("calibration_groups", -1))
        selected_group_counts.append(selected)
        if selected < 0 or selected > budget or selected != adapted + calibrated:
            errors.append(f"invalid target group allocation for {target}")
        if adaptation.get("evaluation_seed_overlap") is not False:
            errors.append(f"adaptation/evaluation overlap is not explicitly false for {target}")
        fitted_domains = {str(value) for value in diagnostics.get("source_domains", ())}
        if budget == 0:
            if selected != 0 or target_city in fitted_domains:
                errors.append(f"zero-shot target labels entered model fitting for {target}")
        elif selected > 0 and target_city not in fitted_domains:
            errors.append(f"few-shot target labels are absent from fitted domains for {target}")
        if "selected_group_ids" in adaptation:
            selected_ids = {str(value) for value in adaptation["selected_group_ids"]}
            adaptation_ids = {str(value) for value in adaptation.get("adaptation_group_ids", ())}
            calibration_ids = {str(value) for value in adaptation.get("calibration_group_ids", ())}
            if selected_ids != adaptation_ids | calibration_ids:
                errors.append(f"target group IDs do not match role partitions for {target}")
            if adaptation_ids & calibration_ids:
                errors.append(f"target adaptation/calibration group IDs overlap for {target}")
            if len(selected_ids) != selected:
                errors.append(f"target selected group ID count mismatch for {target}")
            calibration_seed_set = {
                int(value) for value in setting.get("target_calibration_seeds", ())
            }
            evaluation_seed_set = set(seeds)
            for role, ids in (
                ("adaptation", adaptation_ids),
                ("calibration", calibration_ids),
            ):
                parsed_seeds = {_action_group_seed(value) for value in ids}
                if None in parsed_seeds:
                    errors.append(
                        f"target {role} group seed is not auditable for {target}"
                    )
                    continue
                if parsed_seeds & evaluation_seed_set:
                    errors.append(
                        f"target {role} group overlaps evaluation seeds for {target}"
                    )
                if role == "adaptation" and parsed_seeds & calibration_seed_set:
                    errors.append(
                        f"target adaptation group uses calibration seed for {target}"
                    )
                if role == "calibration" and not parsed_seeds <= calibration_seed_set:
                    errors.append(
                        f"target calibration group uses a non-calibration seed for {target}"
                    )
            if deployment_fit_protocol is not None:
                deployment_ids = {
                    str(value)
                    for value in adaptation.get("deployment_model_group_ids", ())
                }
                if adaptation.get("deployment_fit_protocol") != (
                    TARGET_DEPLOYMENT_FIT_PROTOCOL
                ):
                    errors.append(
                        f"target deployment fit protocol mismatch for {target}"
                    )
                if adaptation.get(
                    "calibration_groups_refit_into_deployment_model"
                ) is not False:
                    errors.append(
                        f"target calibration labels entered deployment fitting for {target}"
                    )
                if deployment_ids != adaptation_ids:
                    errors.append(
                        f"deployment model groups differ from adaptation groups for {target}"
                    )
                if int(adaptation.get("deployment_model_groups", -1)) != len(
                    deployment_ids
                ):
                    errors.append(
                        f"deployment model group count mismatch for {target}"
                    )
            if "selected_tls_ids" in adaptation:
                available_tls_ids = {
                    str(value)
                    for value in adaptation.get("available_target_tls_ids", ())
                }
                selected_tls_ids = {
                    str(value) for value in adaptation.get("selected_tls_ids", ())
                }
                adaptation_tls_ids = {
                    str(value) for value in adaptation.get("adaptation_tls_ids", ())
                }
                calibration_tls_ids = {
                    str(value) for value in adaptation.get("calibration_tls_ids", ())
                }
                if selected_tls_ids != adaptation_tls_ids | calibration_tls_ids:
                    errors.append(
                        f"target TLS IDs do not match role partitions for {target}"
                    )
                if not selected_tls_ids <= available_tls_ids:
                    errors.append(f"selected unknown target TLS IDs for {target}")
                for role, ids in (
                    ("available_target", available_tls_ids),
                    ("selected", selected_tls_ids),
                    ("adaptation", adaptation_tls_ids),
                    ("calibration", calibration_tls_ids),
                ):
                    if int(adaptation.get(f"{role}_tls_count", -1)) != len(ids):
                        errors.append(
                            f"target {role} TLS count mismatch for {target}"
                        )
                if adaptation.get("target_group_selection_protocol") != (
                    TARGET_GROUP_SELECTION_PROTOCOL
                ):
                    errors.append(
                        f"target group selection protocol mismatch for {target}"
                    )
        else:
            warnings.append(f"target group IDs are absent for {target}; nesting cannot be audited")

        uncertainty = dict(diagnostics.get("uncertainty_calibration", {}))
        for family, calibration in uncertainty.items():
            calibration = dict(calibration)
            if calibration.get("protocol") != (
                "domain-normalized-worst-source-city-quantile-inflation-v2"
            ):
                errors.append(
                    f"uncertainty calibration protocol mismatch for {target}/{family}"
                )
            if calibration.get("action_unit") != (
                "within-domain-median-action-range"
            ):
                errors.append(
                    f"uncertainty calibration unit mismatch for {target}/{family}"
                )

        fitted_families = {
            str(value) for value in setting.get("fitted_model_families", ())
        }
        for family in (
            "cfcmt_mechanism",
            "cfcmt_fused",
            "cfcmt_fused_rigid",
        ):
            if family not in fitted_families:
                continue
            cfcmt_source = dict(
                dict(diagnostics.get("fit", {})).get(family, {}).get("source", {})
            )
            source_protocol = cfcmt_source.get("source_selection_protocol")
            allowed_source_protocols = {
                "nested_outer_city_inner_city_cross_fit_v2"
            }
            if family == "cfcmt_fused_rigid":
                allowed_source_protocols.add("exact_rigid_core_fast_path_v1")
            if source_protocol not in allowed_source_protocols:
                errors.append(
                    f"CFCMT nested source selection protocol mismatch for {target}/{family}"
                )
            if family == "cfcmt_fused":
                family_fit = dict(
                    dict(diagnostics.get("fit", {})).get(family, {})
                )
                target_fit = dict(family_fit.get("target", {}))
                min_target_groups = int(
                    dict(family_fit.get("config", {})).get(
                        "min_target_groups",
                        6,
                    )
                )
                allowed_target_protocols = {
                    "target-action-group-oof-source-mechanism-specialist-fusion-v1",
                    "target-action-group-oof-joint-mechanism-gate-specialist-fusion-v2",
                }
                if (
                    int(adaptation.get("adaptation_groups", 0))
                    >= min_target_groups
                    and target_fit.get("selection_protocol")
                    not in allowed_target_protocols
                ):
                    errors.append(
                        f"CFCMT target fusion selection protocol mismatch for {target}"
                    )
            if family == "cfcmt_fused_rigid":
                ablation_weight = cfcmt_source.get("stack_weight")
                if (
                    cfcmt_source.get("ablation")
                    != "source_mechanism_stack_forced_off"
                    or not _finite(ablation_weight)
                    or not math.isclose(
                        float(ablation_weight),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                ):
                    errors.append(
                        f"CFCMT fused-rigid ablation mismatch for {target}"
                    )
                if source_protocol == "exact_rigid_core_fast_path_v1" and (
                    cfcmt_source.get("full_stack_fit_performed") is not False
                ):
                    errors.append(
                        f"CFCMT fused-rigid fast-path audit mismatch for {target}"
                    )

    return {
        "passed": not errors,
        "budget": budget,
        "protocol_sha256": _sha256_json(_protocol_payload(result)),
        "rollout_count": len(rows),
        "expected_rollout_count": len(expected_keys),
        "scenario_count": len(scenarios),
        "city_group_count": len(set(city_groups.values())),
        "policy_count": len(policies),
        "evaluation_seed_count": len(seeds),
        "source_rule_rollout_count": len(source_rule_rows),
        "expected_source_rule_rollout_count": len(expected_source_rule_keys),
        "source_rule_policy_count": len(source_rule_keys),
        "source_rule_seed_count": len(source_policy_seeds),
        "protocol_mismatch_rows": protocol_mismatch_rows,
        "safety_protocol_mismatch_rows": safety_protocol_mismatch_rows,
        "safety_totals": safety_totals,
        "deployable_safety_totals": deployable_safety_totals,
        "unguarded_diagnostic_safety_totals": (
            unguarded_diagnostic_safety_totals
        ),
        "collision_noninferiority": collision_noninferiority,
        "emergency_stops": emergency_stops,
        "selected_group_count_min": min(selected_group_counts, default=0),
        "selected_group_count_max": max(selected_group_counts, default=0),
        "errors": errors,
        "warnings": warnings,
    }


def _paired_city_effect(
    result: Mapping[str, Any],
    *,
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    city_values: dict[str, list[float]] = {}
    network_effects: dict[str, float] = {}
    for target in result["targets"]:
        candidate_value = float(target["policy_metrics"][candidate][PRIMARY_METRIC])
        reference_value = float(target["policy_metrics"][reference][PRIMARY_METRIC])
        effect = (candidate_value - reference_value) / max(abs(reference_value), 1e-12)
        target_name = str(target["target"])
        network_effects[target_name] = effect
        city_values.setdefault(str(target["city_group"]), []).append(effect)
    city_effects = {
        city: sum(values) / len(values) for city, values in sorted(city_values.items())
    }
    ordered = list(city_effects.values())
    median = sorted(ordered)[len(ordered) // 2]
    if len(ordered) % 2 == 0:
        middle = len(ordered) // 2
        median = (sorted(ordered)[middle - 1] + sorted(ordered)[middle]) / 2.0
    gains = {
        city: max(-float(effect), 0.0)
        for city, effect in city_effects.items()
    }
    total_gain = float(sum(gains.values()))
    top_gain_city = max(gains, key=gains.__getitem__) if total_gain > 0.0 else None
    top_gain_share = (
        float(gains[top_gain_city] / total_gain)
        if top_gain_city is not None
        else 0.0
    )
    effective_gain_city_count = (
        float(total_gain**2 / sum(value**2 for value in gains.values()))
        if total_gain > 0.0
        else 0.0
    )
    win_count = sum(value < -1e-12 for value in ordered)
    tie_count = sum(abs(value) <= 1e-12 for value in ordered)
    mean_effect = sum(ordered) / len(ordered)
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": PRIMARY_METRIC,
        "mean_relative_effect": mean_effect,
        "median_relative_effect": median,
        "win_city_groups": win_count,
        "tie_city_groups": tie_count,
        "city_group_count": len(ordered),
        "worst_city_relative_effect": max(ordered),
        "best_city_relative_effect": min(ordered),
        "top_gain_city": top_gain_city,
        "top_gain_share": top_gain_share,
        "effective_gain_city_count": effective_gain_city_count,
        "broadly_improving": bool(
            mean_effect < -1e-12
            and median < -1e-12
            and win_count >= int(math.ceil(len(ordered) / 2.0))
        ),
        "city_relative_effect": city_effects,
        "network_relative_effect": network_effects,
    }


def _paired_policy_ranking(
    result: Mapping[str, Any],
    *,
    policies: Sequence[str],
    reference: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    effects: dict[str, dict[str, Any]] = {}
    city_group_count = len(
        {str(target["city_group"]) for target in result.get("targets", ())}
    )
    for policy in policies:
        if policy == reference:
            effects[policy] = {
                "candidate": policy,
                "reference": reference,
                "metric": PRIMARY_METRIC,
                "mean_relative_effect": 0.0,
                "median_relative_effect": 0.0,
                "win_city_groups": 0,
                "tie_city_groups": city_group_count,
                "city_group_count": city_group_count,
                "worst_city_relative_effect": 0.0,
                "best_city_relative_effect": 0.0,
                "top_gain_city": None,
                "top_gain_share": 0.0,
                "effective_gain_city_count": 0.0,
                "broadly_improving": False,
            }
        else:
            effects[policy] = _paired_city_effect(
                result, candidate=policy, reference=reference
            )
    order = {policy: index for index, policy in enumerate(policies)}
    ranking = sorted(
        policies,
        key=lambda policy: (
            float(effects[policy]["mean_relative_effect"]),
            order[policy],
        ),
    )
    return ranking, effects


def _guard_audit(result: Mapping[str, Any], policy: str) -> dict[str, Any]:
    totals = {"decisions": 0, "proposed_overrides": 0, "accepted_overrides": 0}
    for target in result.get("targets", ()):
        audit = target.get("policy_metrics", {}).get(policy, {}).get("guard_audit", {})
        for key in totals:
            totals[key] += int(audit.get(key, 0))
    decisions = max(totals["decisions"], 1)
    proposed = max(totals["proposed_overrides"], 1)
    return {
        **totals,
        "proposal_rate": totals["proposed_overrides"] / decisions,
        "acceptance_rate": totals["accepted_overrides"] / proposed,
    }


def _uncertainty_summary(result: Mapping[str, Any], family: str) -> dict[str, Any]:
    scales = []
    for diagnostics in result.get("model_diagnostics", {}).values():
        calibration = diagnostics.get("uncertainty_calibration", {}).get(family, {})
        value = calibration.get("scale")
        if _finite(value):
            scales.append(float(value))
    return {
        "target_count": len(scales),
        "mean": sum(scales) / len(scales) if scales else None,
        "min": min(scales) if scales else None,
        "max": max(scales) if scales else None,
    }


def _fused_target_weight_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    by_target: dict[str, float] = {}
    for target, diagnostics in result.get("model_diagnostics", {}).items():
        target_fit = (
            diagnostics.get("fit", {})
            .get("cfcmt_fused", {})
            .get("target", {})
        )
        value = target_fit.get("target_specialist_weight")
        if _finite(value):
            by_target[str(target)] = float(value)
    weights = list(by_target.values())
    return {
        "target_count": len(weights),
        "positive_count": sum(value > 1e-12 for value in weights),
        "intermediate_count": sum(
            1e-12 < value < 1.0 - 1e-12 for value in weights
        ),
        "mean": sum(weights) / len(weights) if weights else None,
        "min": min(weights) if weights else None,
        "max": max(weights) if weights else None,
        "by_target": by_target,
    }


def _fused_source_stack_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    by_target: dict[str, float] = {}
    raw_by_target: dict[str, float] = {}
    gate_by_target: dict[str, float] = {}
    city_groups = {
        str(key): str(value)
        for key, value in result.get("setting", {})
        .get("scenario_city_groups", {})
        .items()
    }
    for target, diagnostics in result.get("model_diagnostics", {}).items():
        source_fit = (
            diagnostics.get("fit", {})
            .get("cfcmt_fused", {})
            .get("source", {})
        )
        target_fit = (
            diagnostics.get("fit", {})
            .get("cfcmt_fused", {})
            .get("target", {})
        )
        raw_value = source_fit.get("stack_weight")
        effective_value = target_fit.get("effective_mechanism_stack_weight")
        gate_value = target_fit.get("target_mechanism_gate")
        if _finite(raw_value):
            raw_by_target[str(target)] = float(raw_value)
        if _finite(gate_value):
            gate_by_target[str(target)] = float(gate_value)
        if _finite(effective_value):
            by_target[str(target)] = float(effective_value)
        elif _finite(raw_value):
            by_target[str(target)] = float(raw_value)
    positive_targets = {
        target for target, value in by_target.items() if value > 1e-12
    }
    positive_cities = sorted(
        {city_groups[target] for target in positive_targets if target in city_groups}
    )
    weights = list(by_target.values())
    return {
        "target_count": len(weights),
        "positive_count": len(positive_targets),
        "positive_city_group_count": len(positive_cities),
        "positive_city_groups": positive_cities,
        "mean": sum(weights) / len(weights) if weights else None,
        "min": min(weights) if weights else None,
        "max": max(weights) if weights else None,
        "by_target": by_target,
        "raw_by_target": raw_by_target,
        "target_mechanism_gate_by_target": gate_by_target,
    }


def _fused_method_acceptance(
    indexed: Mapping[int, Mapping[str, Any]],
    *,
    primary_policy: str,
    acceptance_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_spec = dict(acceptance_spec or {})
    contract = {
        "efficacy_budgets": tuple(
            int(value)
            for value in raw_spec.get(
                "efficacy_budgets", FUSED_EFFICACY_BUDGETS
            )
        ),
        "contribution_budgets": tuple(
            int(value)
            for value in raw_spec.get(
                "contribution_budgets", FUSED_CONTRIBUTION_BUDGETS
            )
        ),
        "primary_policy": str(
            raw_spec.get("primary_policy", FUSED_PRIMARY_POLICY)
        ),
        "selected_source_prior": str(
            raw_spec.get("selected_source_prior", "selected_source_prior")
        ),
        "target_only_comparator": str(
            raw_spec.get("target_only_comparator", TARGET_ONLY_COMPARATOR)
        ),
        "exact_mechanism_ablation": str(
            raw_spec.get("exact_mechanism_ablation", FUSED_RIGID_COMPARATOR)
        ),
        "mechanism_primary_policy": str(
            raw_spec.get(
                "mechanism_primary_policy",
                raw_spec.get("primary_policy", FUSED_PRIMARY_POLICY),
            )
        ),
        "mechanism_target_only_comparator": str(
            raw_spec.get(
                "mechanism_target_only_comparator",
                raw_spec.get("target_only_comparator", TARGET_ONLY_COMPARATOR),
            )
        ),
        "exact_mechanism_model_ablation": str(
            raw_spec.get(
                "exact_mechanism_model_ablation",
                raw_spec.get("exact_mechanism_ablation", FUSED_RIGID_COMPARATOR),
            )
        ),
        "minimum_mean_relative_gain_vs_each_adaptation_comparator": float(
            raw_spec.get(
                "minimum_mean_relative_gain_vs_each_adaptation_comparator",
                0.001,
            )
        ),
        "minimum_positive_source_mechanism_city_groups": int(
            raw_spec.get(
                "minimum_positive_source_mechanism_city_groups", 2
            )
        ),
        "require_joint_non_degenerate_source_target_fusion": bool(
            raw_spec.get(
                "require_joint_non_degenerate_source_target_fusion", True
            )
        ),
        "minimum_efficacy_city_win_fraction": float(
            raw_spec.get("minimum_efficacy_city_win_fraction", 2.0 / 3.0)
        ),
        "maximum_worst_city_relative_harm": float(
            raw_spec.get("maximum_worst_city_relative_harm", 0.005)
        ),
        "maximum_top_gain_share": float(
            raw_spec.get("maximum_top_gain_share", 0.90)
        ),
    }
    if not contract["efficacy_budgets"]:
        raise ValueError("method acceptance efficacy_budgets is empty")
    if not contract["contribution_budgets"]:
        raise ValueError("method acceptance contribution_budgets is empty")
    if any(value < 0 for value in contract["efficacy_budgets"]):
        raise ValueError("method acceptance efficacy budgets must be nonnegative")
    if any(value < 0 for value in contract["contribution_budgets"]):
        raise ValueError("method acceptance contribution budgets must be nonnegative")
    if contract[
        "minimum_mean_relative_gain_vs_each_adaptation_comparator"
    ] < 0.0:
        raise ValueError("method acceptance minimum gain must be nonnegative")
    if contract["minimum_positive_source_mechanism_city_groups"] < 0:
        raise ValueError("method acceptance minimum city count must be nonnegative")
    for key in (
        "minimum_efficacy_city_win_fraction",
        "maximum_worst_city_relative_harm",
        "maximum_top_gain_share",
    ):
        if not _finite(contract[key]):
            raise ValueError(f"method acceptance {key} is not finite")
    if not 0.0 <= contract["minimum_efficacy_city_win_fraction"] <= 1.0:
        raise ValueError("method acceptance city win fraction is outside [0, 1]")
    if not 0.0 < contract["maximum_top_gain_share"] <= 1.0:
        raise ValueError("method acceptance maximum top-gain share is outside (0, 1]")

    configured_primary = contract["primary_policy"]
    selected_prior = contract["selected_source_prior"]
    target_only = contract["target_only_comparator"]
    exact_ablation = contract["exact_mechanism_ablation"]
    mechanism_primary = contract["mechanism_primary_policy"]
    mechanism_target_only = contract["mechanism_target_only_comparator"]
    mechanism_exact_ablation = contract["exact_mechanism_model_ablation"]
    policies = {
        str(value)
        for result in indexed.values()
        for value in result.get("setting", {}).get("policies", ())
    }
    applicable = (
        primary_policy == configured_primary
        and target_only in policies
        and exact_ablation in policies
        and mechanism_primary in policies
        and mechanism_target_only in policies
        and mechanism_exact_ablation in policies
        and selected_prior in policies
    )
    if not applicable:
        return {
            "applicable": False,
            "passed": False,
            "reason": "fused primary or required comparators are absent",
            "contract": contract,
            "efficacy": {},
            "mechanism_contribution": {},
        }

    efficacy: dict[str, Any] = {}
    efficacy_passed = True
    for budget in contract["efficacy_budgets"]:
        if budget not in indexed:
            efficacy[str(budget)] = {
                "available": False,
                "passed": False,
                "reason": "required preregistered budget is absent",
            }
            efficacy_passed = False
            continue
        effect = _paired_city_effect(
            indexed[budget],
            candidate=primary_policy,
            reference=selected_prior,
        )
        required_wins = int(
            math.ceil(
                contract["minimum_efficacy_city_win_fraction"]
                * effect["city_group_count"]
            )
        )
        checks = {
            "negative_mean": effect["mean_relative_effect"] < -1e-12,
            "negative_median": effect["median_relative_effect"] < -1e-12,
            "city_breadth": effect["win_city_groups"] >= required_wins,
            "worst_city_within_preregistered_harm": (
                effect["worst_city_relative_effect"]
                <= contract["maximum_worst_city_relative_harm"] + 1e-12
            ),
            "top_gain_share_within_preregistered_limit": (
                effect["top_gain_share"]
                < contract["maximum_top_gain_share"] - 1e-12
            ),
        }
        passed = all(checks.values())
        efficacy[str(budget)] = {
            "available": True,
            "passed": passed,
            "required_win_city_groups": required_wins,
            "checks": checks,
            "effect_vs_selected_source_prior": effect,
        }
        efficacy_passed &= passed

    contribution_budgets: dict[str, Any] = {}
    contribution_candidates: list[int] = []
    intermediate_selections: list[dict[str, Any]] = []
    for budget in contract["contribution_budgets"]:
        if budget not in indexed:
            continue
        result = indexed[budget]
        fused_vs_target = _paired_city_effect(
            result,
            candidate=mechanism_primary,
            reference=mechanism_target_only,
        )
        fused_vs_rigid = _paired_city_effect(
            result,
            candidate=mechanism_primary,
            reference=mechanism_exact_ablation,
        )
        fused_vs_prior = _paired_city_effect(
            result,
            candidate=mechanism_primary,
            reference=selected_prior,
        )
        target_vs_prior = _paired_city_effect(
            result,
            candidate=mechanism_target_only,
            reference=selected_prior,
        )
        rigid_vs_prior = _paired_city_effect(
            result,
            candidate=mechanism_exact_ablation,
            reference=selected_prior,
        )
        source_stack_summary = _fused_source_stack_summary(result)
        target_weight_summary = _fused_target_weight_summary(result)
        joint_targets = sorted(
            target
            for target, stack_weight in source_stack_summary["by_target"].items()
            if stack_weight > 1e-12
            and 1e-12
            < target_weight_summary["by_target"].get(target, 0.0)
            < 1.0 - 1e-12
        )
        checks = {
            "gain_vs_target_only_reaches_preregistered_minimum": (
                fused_vs_target["mean_relative_effect"]
                <= -contract[
                    "minimum_mean_relative_gain_vs_each_adaptation_comparator"
                ]
                + 1e-12
            ),
            "gain_vs_exact_ablation_reaches_preregistered_minimum": (
                fused_vs_rigid["mean_relative_effect"]
                <= -contract[
                    "minimum_mean_relative_gain_vs_each_adaptation_comparator"
                ]
                + 1e-12
            ),
            "no_fewer_winning_city_groups": (
                fused_vs_prior["win_city_groups"]
                >= max(
                    target_vs_prior["win_city_groups"],
                    rigid_vs_prior["win_city_groups"],
                )
            ),
            "worst_city_gate": (
                fused_vs_prior["worst_city_relative_effect"]
                <= contract["maximum_worst_city_relative_harm"] + 1e-12
            ),
            "source_mechanism_selected_in_required_city_groups": (
                source_stack_summary["positive_city_group_count"]
                >= contract["minimum_positive_source_mechanism_city_groups"]
            ),
            "joint_non_degenerate_fusion_selected": (
                bool(joint_targets)
                if contract[
                    "require_joint_non_degenerate_source_target_fusion"
                ]
                else True
            ),
        }
        passed = all(checks.values())
        if passed:
            contribution_candidates.append(budget)
        intermediate_selections.extend(
            {
                "budget": budget,
                "target": target,
                "target_specialist_weight": weight,
            }
            for target, weight in target_weight_summary["by_target"].items()
            if 1e-12 < weight < 1.0 - 1e-12
        )
        contribution_budgets[str(budget)] = {
            "passed": passed,
            "checks": checks,
            "fused_vs_target_only": fused_vs_target,
            "fused_vs_fused_rigid": fused_vs_rigid,
            "fused_vs_selected_source_prior": fused_vs_prior,
            "target_only_vs_selected_source_prior": target_vs_prior,
            "fused_rigid_vs_selected_source_prior": rigid_vs_prior,
            "source_mechanism_stack": source_stack_summary,
            "fusion_weights": target_weight_summary,
            "joint_fusion_targets": joint_targets,
        }

    contribution_passed = bool(contribution_candidates) and (
        bool(intermediate_selections)
        if contract["require_joint_non_degenerate_source_target_fusion"]
        else True
    )
    return {
        "applicable": True,
        "passed": bool(efficacy_passed and contribution_passed),
        "contract": contract,
        "efficacy_passed": bool(efficacy_passed),
        "efficacy": efficacy,
        "mechanism_contribution": {
            "passed": contribution_passed,
            "qualifying_budgets": contribution_candidates,
            "intermediate_fusion_selection_count": len(intermediate_selections),
            "intermediate_fusion_selections": intermediate_selections,
            "budgets": contribution_budgets,
        },
    }


def build_budget_matrix(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_budgets: Sequence[int] | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_sumo_version: str | None = None,
    expected_primary_policy: str | None = None,
    method_acceptance_spec: Mapping[str, Any] | None = None,
    input_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a strict protocol audit and city-first budget summary."""

    errors: list[str] = []
    warnings: list[str] = []
    audited = [
        _audit_single_result(
            result,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_sumo_version=expected_sumo_version,
        )
        for result in results
    ]
    for audit in audited:
        errors.extend(f"budget {audit.get('budget')}: {value}" for value in audit["errors"])
        warnings.extend(
            f"budget {audit.get('budget')}: {value}" for value in audit["warnings"]
        )
    indexed: dict[int, Mapping[str, Any]] = {}
    for result, audit in zip(results, audited):
        budget = audit.get("budget")
        if budget is None:
            continue
        if int(budget) in indexed:
            errors.append(f"duplicate budget result: {budget}")
        indexed[int(budget)] = result
    observed_budgets = sorted(indexed)
    if expected_budgets is not None:
        expected = sorted({int(value) for value in expected_budgets})
        if observed_budgets != expected:
            errors.append(
                f"budget coverage mismatch: observed={observed_budgets}, expected={expected}"
            )
    if not observed_budgets:
        errors.append("budget matrix contains no valid result payloads")
        return {
            "experiment": "traffic_signal_tsc_budget_matrix_v1",
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "individual_audits": audited,
        }

    protocol_hashes = {audit.get("protocol_sha256") for audit in audited if audit.get("budget") is not None}
    if len(protocol_hashes) != 1:
        errors.append(f"protocol fingerprint drift across budgets: {sorted(protocol_hashes)}")

    reference_result = indexed[observed_budgets[0]]
    reference_policies = tuple(reference_result["setting"]["policies"])
    primary_policy = str(reference_result["setting"].get("primary_reference_policy", ""))
    if primary_policy not in reference_policies:
        errors.append("declared primary CFCMT policy is missing from the policy list")
        primary_policy = next(
            (policy for policy in CORE_COMPARATORS if policy in reference_policies),
            reference_policies[0],
        )
    if expected_primary_policy is not None:
        expected_primary = str(expected_primary_policy)
        for budget in observed_budgets:
            declared = str(
                indexed[budget]["setting"].get("primary_reference_policy", "")
            )
            if declared != expected_primary:
                errors.append(
                    "primary policy does not match the matrix specification at "
                    f"budget {budget}: declared={declared!r}, "
                    f"expected={expected_primary!r}"
                )

    for policy in FIXED_PROTOCOL_POLICIES:
        if policy not in reference_policies:
            continue
        reference_rows = _result_row_map(reference_result, policy)
        for budget in observed_budgets[1:]:
            candidate_rows = _result_row_map(indexed[budget], policy)
            if reference_rows.keys() != candidate_rows.keys() or any(
                not math.isclose(
                    reference_rows[key], candidate_rows[key], rel_tol=0.0, abs_tol=1e-12
                )
                for key in reference_rows.keys() & candidate_rows.keys()
            ):
                errors.append(f"budget-invariant policy changed at budget {budget}: {policy}")
    reference_prior = {
        "policy": reference_result.get("selected_prior_policy"),
        "spec": reference_result.get("selected_prior_spec"),
    }
    reference_source_evidence = {
        "counterfactual": reference_result.get("source_counterfactual_diagnostics"),
        "rule_costs": reference_result.get("source_rule_policy_costs"),
        "rule_rows": reference_result.get("source_rule_seed_results"),
    }
    for budget in observed_budgets[1:]:
        candidate_prior = {
            "policy": indexed[budget].get("selected_prior_policy"),
            "spec": indexed[budget].get("selected_prior_spec"),
        }
        if _canonical_json(candidate_prior) != _canonical_json(reference_prior):
            errors.append(f"source-only prior selection changed at budget {budget}")
        candidate_source_evidence = {
            "counterfactual": indexed[budget].get(
                "source_counterfactual_diagnostics"
            ),
            "rule_costs": indexed[budget].get("source_rule_policy_costs"),
            "rule_rows": indexed[budget].get("source_rule_seed_results"),
        }
        if _canonical_json(candidate_source_evidence) != _canonical_json(
            reference_source_evidence
        ):
            errors.append(f"source training/selector evidence changed at budget {budget}")

    previous_ids: dict[str, set[str]] | None = None
    for budget in observed_budgets:
        current_ids = _target_group_ids(indexed[budget])
        if current_ids is None:
            warnings.append(f"budget {budget}: target group nesting is not auditable")
            previous_ids = None
            continue
        if previous_ids is not None:
            for target, values in current_ids.items():
                if not previous_ids.get(target, set()) <= values:
                    errors.append(
                        f"target adaptation groups are not nested at budget {budget}: {target}"
                    )
        previous_ids = current_ids

    summary_rows = []
    paired_effects: dict[str, Any] = {}
    policy_curves = {policy: [] for policy in reference_policies}
    for budget in observed_budgets:
        result = indexed[budget]
        aggregate = result["aggregate"]
        policy_costs = {
            policy: float(aggregate[policy][PRIMARY_METRIC]) for policy in reference_policies
        }
        for policy, value in policy_costs.items():
            policy_curves[policy].append({"budget": budget, "value": value})
        available_rules = [policy for policy in RULE_POLICIES if policy in policy_costs]
        strongest_rule_absolute = min(available_rules, key=policy_costs.__getitem__)
        best_absolute_policy = min(reference_policies, key=policy_costs.__getitem__)
        paired_reference = (
            "selected_source_prior"
            if "selected_source_prior" in reference_policies
            else strongest_rule_absolute
        )
        paired_ranking, paired_ranking_effects = _paired_policy_ranking(
            result,
            policies=reference_policies,
            reference=paired_reference,
        )
        best_policy = paired_ranking[0]
        paired_rule_ranking, _ = _paired_policy_ranking(
            result,
            policies=available_rules,
            reference=paired_reference,
        )
        strongest_rule = paired_rule_ranking[0]
        primary_value = policy_costs[primary_policy]
        strongest_value = policy_costs[strongest_rule]
        selected_prior_value = policy_costs.get("selected_source_prior")
        primary_vs_rule = (
            0.0
            if primary_policy == strongest_rule
            else float(
                _paired_city_effect(
                    result, candidate=primary_policy, reference=strongest_rule
                )["mean_relative_effect"]
            )
        )
        primary_vs_prior = float(
            paired_ranking_effects[primary_policy]["mean_relative_effect"]
        )
        primary_breadth = paired_ranking_effects[primary_policy]
        guard_policy = (
            primary_policy
            if primary_policy.endswith("_guard")
            else "cfcmt_mechanism_contrast_guard"
        )
        guard_audit = _guard_audit(result, guard_policy) if guard_policy in policy_costs else {}
        selected_counts = [
            int(value.get("target_adaptation", {}).get("selected_groups", 0))
            for value in result.get("model_diagnostics", {}).values()
        ]
        fusion_weights = _fused_target_weight_summary(result)
        source_stack = _fused_source_stack_summary(result)
        paired_effects[str(budget)] = {}
        for candidate in dict.fromkeys((primary_policy, guard_policy, *CORE_COMPARATORS)):
            if candidate not in policy_costs:
                continue
            for reference in dict.fromkeys(("selected_source_prior", strongest_rule, "phase_pressure")):
                if reference not in policy_costs or candidate == reference:
                    continue
                key = f"{candidate}__vs__{reference}"
                paired_effects[str(budget)][key] = _paired_city_effect(
                    result, candidate=candidate, reference=reference
                )
        summary_rows.append(
            {
                "budget": budget,
                "information_regime": "zero-shot" if budget == 0 else "target-simulator-few-shot",
                "primary_policy": primary_policy,
                "primary_cost": primary_value,
                "best_policy": best_policy,
                "best_policy_cost": policy_costs[best_policy],
                "best_policy_paired_relative_to_selected_source_prior": float(
                    paired_ranking_effects[best_policy]["mean_relative_effect"]
                ),
                "best_absolute_policy": best_absolute_policy,
                "best_absolute_policy_cost": policy_costs[best_absolute_policy],
                "primary_rank": paired_ranking.index(primary_policy) + 1,
                "primary_absolute_rank": 1
                + sum(value < primary_value - 1e-12 for value in policy_costs.values()),
                "strongest_rule": strongest_rule,
                "strongest_rule_cost": strongest_value,
                "strongest_rule_absolute": strongest_rule_absolute,
                "strongest_rule_absolute_cost": policy_costs[strongest_rule_absolute],
                "primary_relative_to_strongest_rule": primary_vs_rule,
                "selected_source_prior_cost": selected_prior_value,
                "primary_relative_to_selected_source_prior": primary_vs_prior,
                "primary_median_relative_to_selected_source_prior": float(
                    primary_breadth["median_relative_effect"]
                ),
                "primary_win_city_groups": int(
                    primary_breadth["win_city_groups"]
                ),
                "primary_city_group_count": int(
                    primary_breadth["city_group_count"]
                ),
                "primary_worst_city_relative_to_selected_source_prior": float(
                    primary_breadth["worst_city_relative_effect"]
                ),
                "primary_top_gain_city": primary_breadth["top_gain_city"],
                "primary_top_gain_share": float(
                    primary_breadth["top_gain_share"]
                ),
                "primary_effective_gain_city_count": float(
                    primary_breadth["effective_gain_city_count"]
                ),
                "primary_broadly_improving": bool(
                    primary_breadth["broadly_improving"]
                ),
                "selected_target_groups_mean": (
                    sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
                ),
                "guard_proposal_rate": guard_audit.get("proposal_rate"),
                "guard_acceptance_rate": guard_audit.get("acceptance_rate"),
                "cfcmt_uncertainty_scale": _uncertainty_summary(
                    result,
                    "cfcmt_fused"
                    if primary_policy.startswith("cfcmt_fused_")
                    else "cfcmt_mechanism",
                ),
                "fused_target_weight_mean": fusion_weights["mean"],
                "fused_target_weight_positive_count": fusion_weights[
                    "positive_count"
                ],
                "fused_target_weight_intermediate_count": fusion_weights[
                    "intermediate_count"
                ],
                "fused_source_stack_weight_mean": source_stack["mean"],
                "fused_source_stack_positive_count": source_stack[
                    "positive_count"
                ],
                "fused_source_stack_positive_city_group_count": source_stack[
                    "positive_city_group_count"
                ],
                "safety_totals": next(
                    audit["safety_totals"] for audit in audited if audit.get("budget") == budget
                ),
            }
        )

    monotonicity = {}
    for policy, curve in policy_curves.items():
        inversions = [
            {
                "from_budget": left["budget"],
                "to_budget": right["budget"],
                "increase": right["value"] - left["value"],
            }
            for left, right in zip(curve, curve[1:])
            if right["value"] > left["value"] + 1e-12
        ]
        monotonicity[policy] = {
            "nonincreasing": not inversions,
            "inversion_count": len(inversions),
            "inversions": inversions,
        }
    if not monotonicity.get(primary_policy, {}).get("nonincreasing", True):
        warnings.append("declared primary policy is non-monotonic across target budgets")
    for row in summary_rows:
        if row["primary_relative_to_selected_source_prior"] >= -1e-12:
            continue
        if not row["primary_broadly_improving"]:
            warnings.append(
                f"budget {row['budget']}: primary mean gain is not broad across city groups"
            )
        if row["primary_top_gain_share"] > 0.75:
            warnings.append(
                f"budget {row['budget']}: {row['primary_top_gain_share']:.1%} of "
                f"primary improvement is concentrated in {row['primary_top_gain_city']}"
            )

    try:
        method_acceptance = _fused_method_acceptance(
            indexed,
            primary_policy=primary_policy,
            acceptance_spec=method_acceptance_spec,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid method acceptance specification: {exc}")
        method_acceptance = {
            "applicable": False,
            "passed": False,
            "reason": str(exc),
            "contract": dict(method_acceptance_spec or {}),
            "efficacy": {},
            "mechanism_contribution": {},
        }
    if method_acceptance.get("applicable") and not method_acceptance.get("passed"):
        warnings.append(
            "fused CFCMT does not pass all preregistered efficacy and mechanism-contribution gates"
        )

    return {
        "experiment": "traffic_signal_tsc_budget_matrix_v1",
        "passed": not errors,
        "protocol_sha256": next(iter(protocol_hashes)) if len(protocol_hashes) == 1 else None,
        "expected_source_tree_sha256": expected_source_tree_sha256,
        "expected_sumo_version": expected_sumo_version,
        "expected_primary_policy": expected_primary_policy,
        "method_acceptance_spec": dict(method_acceptance_spec or {}),
        "input_paths": list(input_paths or ()),
        "budgets": observed_budgets,
        "primary_metric": PRIMARY_METRIC,
        "primary_policy": primary_policy,
        "integrity_passed": not errors,
        "method_acceptance": method_acceptance,
        "submission_claim_supported": bool(
            not errors and method_acceptance.get("passed", False)
        ),
        "summary": summary_rows,
        "policy_curves": policy_curves,
        "monotonicity": monotonicity,
        "paired_city_effects": paired_effects,
        "selected_prior_policy": reference_result.get("selected_prior_policy", {}),
        "selected_prior_spec": reference_result.get("selected_prior_spec", {}),
        "individual_audits": audited,
        "errors": errors,
        "warnings": warnings,
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    return output.getvalue()


def write_budget_matrix_artifacts(
    matrix: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    summary_csv_path: Path,
    policy_csv_path: Path,
) -> None:
    _atomic_write_text(
        json_path,
        json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    summary_fields = (
        "budget",
        "information_regime",
        "primary_policy",
        "primary_cost",
        "best_policy",
        "best_policy_cost",
        "best_policy_paired_relative_to_selected_source_prior",
        "best_absolute_policy",
        "best_absolute_policy_cost",
        "primary_rank",
        "primary_absolute_rank",
        "strongest_rule",
        "strongest_rule_cost",
        "strongest_rule_absolute",
        "strongest_rule_absolute_cost",
        "primary_relative_to_strongest_rule",
        "selected_source_prior_cost",
        "primary_relative_to_selected_source_prior",
        "primary_median_relative_to_selected_source_prior",
        "primary_win_city_groups",
        "primary_city_group_count",
        "primary_worst_city_relative_to_selected_source_prior",
        "primary_top_gain_city",
        "primary_top_gain_share",
        "primary_effective_gain_city_count",
        "primary_broadly_improving",
        "selected_target_groups_mean",
        "guard_proposal_rate",
        "guard_acceptance_rate",
        "fused_target_weight_mean",
        "fused_target_weight_positive_count",
        "fused_target_weight_intermediate_count",
        "fused_source_stack_weight_mean",
        "fused_source_stack_positive_count",
        "fused_source_stack_positive_city_group_count",
    )
    _atomic_write_text(summary_csv_path, _csv_text(matrix.get("summary", ()), summary_fields))
    policy_rows = [
        {"policy": policy, **point}
        for policy, curve in matrix.get("policy_curves", {}).items()
        for point in curve
    ]
    _atomic_write_text(policy_csv_path, _csv_text(policy_rows, ("budget", "policy", "value")))

    lines = [
        "# TSC CFCMT Budget Matrix Audit",
        "",
        f"Audit: **{'PASS' if matrix.get('passed') else 'FAIL'}**  ",
        f"Protocol SHA-256: `{matrix.get('protocol_sha256')}`  ",
        f"Primary metric: `{matrix.get('primary_metric')}`  ",
        f"Declared primary policy: `{matrix.get('primary_policy')}`",
        "",
        "| Budget | Regime | Primary cost | Best paired policy | Best absolute policy | Strongest paired rule | Delta vs rule | Paired rank |",
        "|---:|---|---:|---|---|---|---:|---:|",
    ]
    for row in matrix.get("summary", ()):
        lines.append(
            f"| {row['budget']} | {row['information_regime']} | {row['primary_cost']:.6f} | "
            f"{row['best_policy']} | {row['best_absolute_policy']} | {row['strongest_rule']} | "
            f"{100.0 * row['primary_relative_to_strongest_rule']:.2f}% | {row['primary_rank']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-city breadth",
            "",
            "| Budget | Mean vs prior | Median vs prior | Winning cities | Worst city | Top-gain city share | Broad |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matrix.get("summary", ()):
        lines.append(
            f"| {row['budget']} | "
            f"{100.0 * row['primary_relative_to_selected_source_prior']:.2f}% | "
            f"{100.0 * row['primary_median_relative_to_selected_source_prior']:.2f}% | "
            f"{row['primary_win_city_groups']}/{row['primary_city_group_count']} | "
            f"{100.0 * row['primary_worst_city_relative_to_selected_source_prior']:.2f}% | "
            f"{100.0 * row['primary_top_gain_share']:.1f}% | "
            f"{row['primary_broadly_improving']} |"
        )
    acceptance = matrix.get("method_acceptance", {})
    if acceptance.get("applicable"):
        lines.extend(
            [
                "",
                "## Preregistered method acceptance",
                "",
                f"Overall method gate: **{'PASS' if acceptance.get('passed') else 'FAIL'}**  ",
                f"Efficacy gate: **{'PASS' if acceptance.get('efficacy_passed') else 'FAIL'}**  ",
                "Mechanism contribution gate: "
                f"**{'PASS' if acceptance.get('mechanism_contribution', {}).get('passed') else 'FAIL'}**",
            ]
        )
    lines.extend(["", "## Monotonicity", "", "| Policy | Non-increasing | Inversions |", "|---|---:|---:|"])
    for policy, diagnostics in matrix.get("monotonicity", {}).items():
        if policy in CORE_COMPARATORS or policy == matrix.get("primary_policy"):
            lines.append(
                f"| {policy} | {diagnostics['nonincreasing']} | {diagnostics['inversion_count']} |"
            )
    if matrix.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {value}" for value in matrix["errors"])
    if matrix.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {value}" for value in matrix["warnings"])
    _atomic_write_text(markdown_path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--policy-csv", type=Path)
    parser.add_argument("--allow-failed-audit", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.results_root.glob("budget_*/result.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    expected_budgets = spec["development"]["target_group_budgets"]
    expected_sumo = str(spec["expected_sumo_version"])
    expected_primary = spec["development"].get("primary_policy")
    expected_source = None
    if args.snapshot_manifest:
        snapshot = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
        expected_source = str(snapshot["source_tree_sha256"])
    matrix = build_budget_matrix(
        payloads,
        expected_budgets=expected_budgets,
        expected_source_tree_sha256=expected_source,
        expected_sumo_version=expected_sumo,
        expected_primary_policy=expected_primary,
        method_acceptance_spec=spec.get("method_acceptance"),
        input_paths=[str(path) for path in paths],
    )
    write_budget_matrix_artifacts(
        matrix,
        json_path=args.out or args.results_root / "budget_matrix_audit.json",
        markdown_path=args.md_out or args.results_root / "budget_matrix_audit.md",
        summary_csv_path=args.summary_csv or args.results_root / "budget_matrix_summary.csv",
        policy_csv_path=args.policy_csv or args.results_root / "budget_matrix_policy_curve.csv",
    )
    if not matrix["passed"] and not args.allow_failed_audit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
