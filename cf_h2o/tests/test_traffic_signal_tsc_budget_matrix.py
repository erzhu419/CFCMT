from __future__ import annotations

import copy
from itertools import product

from cf_h2o.eval.traffic_signal_tsc_budget_matrix import (
    PRIMARY_METRIC,
    build_budget_matrix,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_HIERARCHY_PROTOCOL_V3,
    COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
    STRICT_SAFETY_MONITORING_V3,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL
from cf_h2o.traffic_signal.action_ranker import ACTION_TARGET_SCALE_PROTOCOL


SCENARIOS = ("city_a_net", "city_b_net", "city_c_net")
CITIES = {scenario: scenario.split("_")[1] for scenario in SCENARIOS}
SEEDS = (11, 13)
SOURCE_RULE_SEEDS = (101, 103)
SOURCE_RULE_SPECS = (
    {
        "key": "phase_pressure",
        "downstream_queue_weight": 0.0,
        "downstream_occupancy_weight": 0.0,
        "switch_penalty": 0.0,
        "mode": "linear",
    },
    {
        "key": "max_pressure",
        "downstream_queue_weight": 1.0,
        "downstream_occupancy_weight": 0.0,
        "switch_penalty": 0.0,
        "mode": "linear",
    },
)
POLICIES = (
    "fixed_program",
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
    "selected_source_prior",
    "simulator_contrast_guard",
    "dense_contrast_guard",
    "causal_core_advantage_contrast_guard",
    "cfcmt_mechanism_contrast_regularized",
    "cfcmt_mechanism_contrast_guard",
    "causal_target_adapter_contrast_guard",
    "causal_target_only_contrast_guard",
)


def _cost(policy: str, budget: int) -> float:
    fixed = {
        "fixed_program": 15.0,
        "max_pressure": 10.0,
        "phase_pressure": 9.5,
        "spillback_pressure": 9.8,
        "selected_source_prior": 9.2,
    }
    if policy in fixed:
        return fixed[policy]
    offsets = {
        "simulator_contrast_guard": 1.0,
        "dense_contrast_guard": 0.8,
        "causal_core_advantage_contrast_guard": 0.4,
        "cfcmt_mechanism_contrast_regularized": 0.0,
        "cfcmt_mechanism_contrast_guard": 0.2,
        "causal_target_adapter_contrast_guard": 0.3,
        "causal_target_only_contrast_guard": 0.6,
    }
    return 9.0 + offsets[policy] - 0.002 * budget


def _metrics(value: float) -> dict[str, object]:
    return {
        "ok": True,
        "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
        "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
        PRIMARY_METRIC: value,
        "mean_queue_per_lane": value / 2.0,
        "p90_queue_per_lane": value,
        "completion_ratio": 0.9,
        "collision_events": 0,
        "collision_incidents": 0,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "emergency_stops": 0,
        "guard_audit": {
            "decisions": 10,
            "proposed_overrides": 4,
            "accepted_overrides": 2,
        },
    }


def _result(budget: int, *, source_hash: str = "source-hash") -> dict[str, object]:
    rows = []
    for target, seed, policy in product(SCENARIOS, SEEDS, POLICIES):
        rows.append(
            {
                "target": target,
                "seed": seed,
                "policy": policy,
                "metrics": _metrics(_cost(policy, budget)),
            }
        )
    targets = []
    for target in SCENARIOS:
        targets.append(
            {
                "target": target,
                "city_group": CITIES[target],
                "policy_metrics": {
                    policy: {
                        **_metrics(_cost(policy, budget)),
                        "guard_audit": {
                            "decisions": 20,
                            "proposed_overrides": 8,
                            "accepted_overrides": 4,
                        },
                    }
                    for policy in POLICIES
                },
            }
        )
    model_diagnostics = {}
    for target in SCENARIOS:
        target_city = CITIES[target]
        split = budget // 2
        adaptation_ids = [
            f"{target}:seed101:adaptation:{index:03d}" for index in range(split)
        ]
        calibration_ids = [
            f"{target}:seed103:calibration:{index:03d}"
            for index in range(budget - split)
        ]
        selected_ids = [*adaptation_ids, *calibration_ids]
        source_domains = sorted(set(CITIES.values()) - {target_city})
        if budget:
            source_domains.append(target_city)
        model_diagnostics[target] = {
            "source_domains": source_domains,
            "uncertainty_calibration": {
                "cfcmt_mechanism": {
                    "scale": 1.5,
                    "protocol": (
                        "domain-normalized-worst-source-city-quantile-inflation-v2"
                    ),
                    "action_unit": "within-domain-median-action-range",
                }
            },
            "fit": {
                "cfcmt_mechanism": {
                    "source": {
                        "source_selection_protocol": (
                            "nested_outer_city_inner_city_cross_fit_v2"
                        )
                    }
                }
            },
            "target_adaptation": {
                "target": target,
                "target_city_group": target_city,
                "heldout_city_scenarios": [target],
                "source_scenarios": [value for value in SCENARIOS if value != target],
                "source_city_groups": sorted(set(CITIES.values()) - {target_city}),
                "target_context_provenance": (
                    "static_sumocfg_network_phase_and_route_summary"
                ),
                "target_context_transition_labels_used": False,
                "label_type": (
                    "matched_target_simulator_counterfactual_action_outcomes"
                ),
                "requested_group_budget": budget,
                "selected_groups": budget,
                "adaptation_groups": split,
                "calibration_groups": budget - split,
                "selected_group_ids": selected_ids,
                "adaptation_group_ids": adaptation_ids,
                "calibration_group_ids": calibration_ids,
                "deployment_fit_protocol": (
                    "adaptation_only_model_disjoint_calibration_guard_v1"
                ),
                "deployment_model_groups": split,
                "deployment_model_group_ids": adaptation_ids,
                "calibration_groups_refit_into_deployment_model": False,
                "evaluation_seed_overlap": False,
            },
        }
    return {
        "experiment": "synthetic_tsc_matrix",
        "runtime": {
            "git_commit": "abc",
            "source_tree_sha256": source_hash,
            "platform": "linux",
            "python_version": "3.10",
            "python_executable": "/python",
            "sumo_binary": "/sumo",
            "sumo_version": "SUMO 1.22.0",
            "libsumo_version": "1.22.0",
            "numpy_version": "2",
            "scikit_learn_version": "1",
            "thread_limits": {"OMP_NUM_THREADS": "1"},
            "determinism_environment": {"PYTHONHASHSEED": "0"},
        },
        "setting": {
            "target_group_budget": budget,
            "target_information_budget": "zero" if budget == 0 else "few",
            "scenarios": list(SCENARIOS),
            "policies": list(POLICIES),
            "seeds": list(SEEDS),
            "source_seeds": [101],
            "target_calibration_seeds": [103],
            "source_policy_seeds": list(SOURCE_RULE_SEEDS),
            "source_rule_specs": list(SOURCE_RULE_SPECS),
            "scenario_city_groups": CITIES,
            "fold_protocol": "strict_leave_one_city_group_out",
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "target_deployment_fit_protocol": (
                "adaptation_only_model_disjoint_calibration_guard_v1"
            ),
            "action_target_scale_protocol": ACTION_TARGET_SCALE_PROTOCOL,
            "cfcmt_hierarchy_protocol": CFCMT_HIERARCHY_PROTOCOL_V3,
            "target_group_selection_protocol": (
                "coverage-first-per-tls-disjoint-seed-role-v1"
            ),
            "min_source_tls_coverage": 1.0,
            "fitted_model_families": ["cfcmt_mechanism"],
            "primary_reference_policy": "cfcmt_mechanism_contrast_regularized",
        },
        "source_counterfactual_diagnostics": {
            scenario: {
                "tls_coverage_fraction": 1.0,
                "counterfactual_replay_audit": {"passed": True},
                "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
                "behavior_safety_audit": {
                    "starting_teleports": 0,
                    "ending_teleports": 0,
                    "no_teleport_passed": True,
                },
                "counterfactual_safety_audit": {
                    "protocol": COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
                    "candidate_groups": 2,
                    "retained_groups": 2,
                    "censored_groups": 0,
                    "retained_branches": 4,
                    "no_teleport_passed": True,
                    "symmetric_group_censoring_passed": True,
                },
                "groups": 2,
                "rows": 4,
            }
            for scenario in SCENARIOS
        },
        "source_rule_policy_costs": {
            scenario: {
                spec["key"]: 8.0 + index
                for index, spec in enumerate(SOURCE_RULE_SPECS)
            }
            for scenario in SCENARIOS
        },
        "source_rule_seed_results": [
            {
                "scenario": scenario,
                "seed": seed,
                "policy": spec["key"],
                "pressure_spec": dict(spec),
                "metrics": _metrics(8.0 + index),
            }
            for scenario, seed, (index, spec) in product(
                SCENARIOS,
                SOURCE_RULE_SEEDS,
                tuple(enumerate(SOURCE_RULE_SPECS)),
            )
        ],
        "model_diagnostics": model_diagnostics,
        "aggregate": {
            policy: {PRIMARY_METRIC: _cost(policy, budget)} for policy in POLICIES
        },
        "targets": targets,
        "seed_results": rows,
        "selected_prior_policy": {scenario: "selected_source_prior" for scenario in SCENARIOS},
        "selected_prior_spec": {scenario: {"key": "selected_source_prior"} for scenario in SCENARIOS},
    }


def _fused_result(budget: int) -> dict[str, object]:
    result = _result(budget)
    policy = "cfcmt_fused_contrast_guard"
    rigid_policy = "cfcmt_fused_rigid_contrast_guard"
    fused_cost = 9.2 if budget == 0 else 8.8
    rigid_cost = 9.2 if budget == 0 else 9.0
    result["setting"]["policies"].extend((policy, rigid_policy))
    result["setting"]["fitted_model_families"].extend(
        ("cfcmt_fused", "cfcmt_fused_rigid")
    )
    result["setting"]["primary_reference_policy"] = policy
    result["aggregate"][policy] = {PRIMARY_METRIC: fused_cost}
    result["aggregate"][rigid_policy] = {PRIMARY_METRIC: rigid_cost}
    for target in result["targets"]:
        target["policy_metrics"][policy] = _metrics(fused_cost)
        target["policy_metrics"][rigid_policy] = _metrics(rigid_cost)
    for target, seed, candidate in product(
        SCENARIOS,
        SEEDS,
        (policy, rigid_policy),
    ):
        cost = fused_cost if candidate == policy else rigid_cost
        result["seed_results"].append(
            {
                "target": target,
                "seed": seed,
                "policy": candidate,
                "metrics": _metrics(cost),
            }
        )
    for diagnostics in result["model_diagnostics"].values():
        diagnostics["fit"]["cfcmt_fused"] = {
            "source": {
                "source_selection_protocol": (
                    "nested_outer_city_inner_city_cross_fit_v2"
                ),
                "stack_weight": 0.25,
            },
            "target": {
                "selection_protocol": (
                    "target-action-group-oof-source-mechanism-specialist-fusion-v1"
                ),
                "target_specialist_weight": 0.0 if budget == 0 else 0.25,
            },
        }
        diagnostics["fit"]["cfcmt_fused_rigid"] = {
            "source": {
                "source_selection_protocol": (
                    "nested_outer_city_inner_city_cross_fit_v2"
                ),
                "stack_weight": 0.0,
                "ablation": "source_mechanism_stack_forced_off",
            },
            "target": {
                "selection_protocol": (
                    "target-action-group-oof-source-mechanism-specialist-fusion-v1"
                ),
                "target_specialist_weight": 0.0 if budget == 0 else 0.25,
            },
        }
        diagnostics["uncertainty_calibration"]["cfcmt_fused"] = {
            "scale": 1.5,
            "protocol": (
                "domain-normalized-worst-source-city-quantile-inflation-v2"
            ),
            "action_unit": "within-domain-median-action-range",
        }
        diagnostics["uncertainty_calibration"]["cfcmt_fused_rigid"] = {
            "scale": 1.5,
            "protocol": (
                "domain-normalized-worst-source-city-quantile-inflation-v2"
            ),
            "action_unit": "within-domain-median-action-range",
        }
    return result


def _dual_layer_fused_result(budget: int) -> dict[str, object]:
    result = _fused_result(budget)
    policy_costs = {
        "cfcmt_fused_contrast_mpc": 8.5,
        "cfcmt_fused_rigid_contrast_mpc": 8.9,
        "causal_target_only_contrast_mpc": 8.8,
    }
    result["setting"]["policies"].extend(policy_costs)
    for policy, cost in policy_costs.items():
        result["aggregate"][policy] = {PRIMARY_METRIC: cost}
        for target in result["targets"]:
            target["policy_metrics"][policy] = _metrics(cost)
        for target, seed in product(SCENARIOS, SEEDS):
            result["seed_results"].append(
                {
                    "target": target,
                    "seed": seed,
                    "policy": policy,
                    "metrics": _metrics(cost),
                }
            )
    return result


def test_budget_matrix_passes_strict_protocol_and_nested_groups():
    matrix = build_budget_matrix(
        [_result(0), _result(8)],
        expected_budgets=[0, 8],
        expected_source_tree_sha256="source-hash",
        expected_sumo_version="1.22.0",
    )
    assert matrix["passed"] is True
    assert matrix["summary"][0]["strongest_rule"] == "selected_source_prior"
    assert matrix["summary"][1]["primary_relative_to_strongest_rule"] < 0.0
    assert matrix["summary"][1]["primary_broadly_improving"] is True
    assert matrix["summary"][1]["primary_win_city_groups"] == 3
    assert matrix["summary"][1]["primary_top_gain_share"] < 0.34
    assert matrix["monotonicity"][matrix["primary_policy"]]["nonincreasing"] is True


def test_budget_matrix_separates_integrity_from_fused_method_acceptance():
    budgets = [0, 16, 32, 60, 120]
    matrix = build_budget_matrix(
        [_fused_result(budget) for budget in budgets],
        expected_budgets=budgets,
        expected_primary_policy="cfcmt_fused_contrast_guard",
    )

    assert matrix["passed"] is True
    assert matrix["integrity_passed"] is True
    assert matrix["method_acceptance"]["passed"] is True
    assert matrix["method_acceptance"]["efficacy_passed"] is True
    contribution = matrix["method_acceptance"]["mechanism_contribution"]
    assert contribution["passed"] is True
    assert contribution["qualifying_budgets"]
    assert contribution["intermediate_fusion_selection_count"] > 0
    assert matrix["submission_claim_supported"] is True


def test_budget_matrix_separates_model_ablation_from_guarded_deployment():
    budgets = [0, 16, 32, 60, 120]
    contract = {
        "efficacy_budgets": [60, 120],
        "contribution_budgets": [16, 32, 60, 120],
        "primary_policy": "cfcmt_fused_contrast_guard",
        "selected_source_prior": "selected_source_prior",
        "target_only_comparator": "causal_target_only_contrast_guard",
        "exact_mechanism_ablation": "cfcmt_fused_rigid_contrast_guard",
        "mechanism_primary_policy": "cfcmt_fused_contrast_mpc",
        "mechanism_target_only_comparator": "causal_target_only_contrast_mpc",
        "exact_mechanism_model_ablation": "cfcmt_fused_rigid_contrast_mpc",
        "minimum_mean_relative_gain_vs_each_adaptation_comparator": 0.001,
        "minimum_positive_source_mechanism_city_groups": 2,
        "require_joint_non_degenerate_source_target_fusion": True,
        "minimum_efficacy_city_win_fraction": 2.0 / 3.0,
        "maximum_worst_city_relative_harm": 0.005,
        "maximum_top_gain_share": 0.9,
    }

    matrix = build_budget_matrix(
        [_dual_layer_fused_result(budget) for budget in budgets],
        expected_budgets=budgets,
        expected_primary_policy="cfcmt_fused_contrast_guard",
        method_acceptance_spec=contract,
    )

    assert matrix["integrity_passed"] is True
    contribution = matrix["method_acceptance"]["mechanism_contribution"]
    assert contribution["passed"] is True
    budget = contribution["budgets"]["16"]
    assert budget["fused_vs_fused_rigid"]["candidate"] == (
        "cfcmt_fused_contrast_mpc"
    )
    assert budget["fused_vs_fused_rigid"]["reference"] == (
        "cfcmt_fused_rigid_contrast_mpc"
    )


def test_budget_matrix_accepts_verified_exact_rigid_fast_path():
    result = _fused_result(16)
    for diagnostics in result["model_diagnostics"].values():
        source = diagnostics["fit"]["cfcmt_fused_rigid"]["source"]
        source["source_selection_protocol"] = "exact_rigid_core_fast_path_v1"
        source["full_stack_fit_performed"] = False

    matrix = build_budget_matrix([result], expected_budgets=[16])

    assert matrix["integrity_passed"] is True


def test_budget_matrix_reports_valid_but_unsupported_fused_method():
    budgets = [0, 16, 32, 60, 120]
    results = [_fused_result(budget) for budget in budgets]
    for result in results:
        target_only = float(
            result["aggregate"]["causal_target_only_contrast_guard"][PRIMARY_METRIC]
        )
        result["aggregate"]["cfcmt_fused_contrast_guard"][PRIMARY_METRIC] = target_only
        for row in result["seed_results"]:
            if row["policy"] == "cfcmt_fused_contrast_guard":
                row["metrics"] = _metrics(target_only)
        for target in result["targets"]:
            target["policy_metrics"]["cfcmt_fused_contrast_guard"] = _metrics(
                target_only
            )
        for diagnostics in result["model_diagnostics"].values():
            diagnostics["fit"]["cfcmt_fused"]["target"][
                "target_specialist_weight"
            ] = 0.0

    matrix = build_budget_matrix(
        results,
        expected_budgets=budgets,
        expected_primary_policy="cfcmt_fused_contrast_guard",
    )

    assert matrix["passed"] is True
    assert matrix["integrity_passed"] is True
    assert matrix["method_acceptance"]["passed"] is False
    assert matrix["submission_claim_supported"] is False


def test_budget_matrix_uses_preregistered_method_thresholds():
    budgets = [0, 16, 32, 60, 120]
    strict_contract = {
        "efficacy_budgets": [60, 120],
        "contribution_budgets": [16, 32, 60, 120],
        "primary_policy": "cfcmt_fused_contrast_guard",
        "selected_source_prior": "selected_source_prior",
        "target_only_comparator": "causal_target_only_contrast_guard",
        "exact_mechanism_ablation": "cfcmt_fused_rigid_contrast_guard",
        "minimum_mean_relative_gain_vs_each_adaptation_comparator": 0.25,
        "minimum_positive_source_mechanism_city_groups": 2,
        "require_joint_non_degenerate_source_target_fusion": True,
        "minimum_efficacy_city_win_fraction": 2.0 / 3.0,
        "maximum_worst_city_relative_harm": 0.005,
        "maximum_top_gain_share": 0.9,
    }

    matrix = build_budget_matrix(
        [_fused_result(budget) for budget in budgets],
        expected_budgets=budgets,
        expected_primary_policy="cfcmt_fused_contrast_guard",
        method_acceptance_spec=strict_contract,
    )

    assert matrix["passed"] is True
    assert matrix["method_acceptance"]["contract"][
        "minimum_mean_relative_gain_vs_each_adaptation_comparator"
    ] == 0.25
    assert matrix["method_acceptance"]["mechanism_contribution"]["passed"] is False
    assert matrix["submission_claim_supported"] is False


def test_budget_matrix_rejects_primary_policy_spec_mismatch():
    matrix = build_budget_matrix(
        [_result(0), _result(8)],
        expected_budgets=[0, 8],
        expected_primary_policy="cfcmt_mechanism_contrast_guard",
    )

    assert matrix["passed"] is False
    assert sum(
        "primary policy does not match" in value for value in matrix["errors"]
    ) == 2


def test_budget_matrix_rejects_zero_shot_target_label_leakage():
    result = _result(0)
    result["model_diagnostics"]["city_a_net"]["source_domains"].append("a")
    matrix = build_budget_matrix([result], expected_budgets=[0])
    assert matrix["passed"] is False
    assert any("zero-shot target labels" in value for value in matrix["errors"])


def test_budget_matrix_rejects_protocol_hash_drift():
    first = _result(0)
    second = _result(8)
    second["setting"]["fold_protocol"] = "leave_one_network_out"
    matrix = build_budget_matrix([first, second], expected_budgets=[0, 8])
    assert matrix["passed"] is False
    assert any("protocol fingerprint drift" in value for value in matrix["errors"])


def test_budget_matrix_rejects_default_teleport_protocol():
    result = _result(0)
    result["setting"]["sumo_execution_protocol"] = {
        **SUMO_EXECUTION_PROTOCOL,
        "time_to_teleport_sec": 300,
    }
    matrix = build_budget_matrix([result], expected_budgets=[0])
    assert matrix["passed"] is False
    assert any("no-teleport protocol" in value for value in matrix["errors"])


def test_budget_matrix_rejects_budget_dependent_rule_rollout():
    first = _result(0)
    second = _result(8)
    for row in second["seed_results"]:
        if row["policy"] == "phase_pressure":
            row["metrics"][PRIMARY_METRIC] += 1.0
            break
    matrix = build_budget_matrix([first, second], expected_budgets=[0, 8])
    assert matrix["passed"] is False
    assert any("budget-invariant policy changed" in value for value in matrix["errors"])


def test_budget_matrix_rejects_missing_rollout_and_safety_event():
    result = _result(0)
    result["seed_results"].pop()
    result["seed_results"][0]["metrics"]["starting_teleports"] = 1
    matrix = build_budget_matrix([result], expected_budgets=[0])
    assert matrix["passed"] is False
    assert any("missing evaluation rollouts" in value for value in matrix["errors"])
    assert any("teleport events" in value for value in matrix["errors"])


def test_budget_matrix_rejects_incomplete_source_rule_selector_evidence():
    result = _result(0)
    result["source_rule_seed_results"].pop()

    matrix = build_budget_matrix([result], expected_budgets=[0])

    assert matrix["passed"] is False
    assert any(
        "missing source pressure-rule rollouts" in value
        for value in matrix["errors"]
    )


def test_budget_matrix_rejects_counterfactual_safety_count_mismatch():
    result = _result(0)
    safety = result["source_counterfactual_diagnostics"]["city_a_net"][
        "counterfactual_safety_audit"
    ]
    safety["retained_branches"] = 3

    matrix = build_budget_matrix([result], expected_budgets=[0])

    assert matrix["passed"] is False
    assert any(
        "counterfactual retained-row audit mismatch" in value
        for value in matrix["errors"]
    )


def test_budget_matrix_rejects_target_group_seed_role_leakage():
    result = _result(8)
    adaptation = result["model_diagnostics"]["city_a_net"]["target_adaptation"]
    leaked = "city_a_net:seed103:adaptation:leaked"
    adaptation["adaptation_group_ids"][0] = leaked
    adaptation["selected_group_ids"][0] = leaked
    adaptation["deployment_model_group_ids"][0] = leaked

    matrix = build_budget_matrix([result], expected_budgets=[8])

    assert matrix["passed"] is False
    assert any(
        "adaptation group uses calibration seed" in value
        for value in matrix["errors"]
    )


def test_budget_matrix_reports_but_accepts_unguarded_mpc_safety_event():
    result = _result(0)
    diagnostic_policy = "causal_core_advantage_contrast_mpc"
    result["setting"]["policies"].append(diagnostic_policy)
    for target, seed in product(SCENARIOS, SEEDS):
        result["seed_results"].append(
            {
                "target": target,
                "seed": seed,
                "policy": diagnostic_policy,
                "metrics": _metrics(9.1),
            }
        )
    result["seed_results"][-1]["metrics"]["collision_events"] = 1
    result["aggregate"][diagnostic_policy] = {PRIMARY_METRIC: 9.1}
    for target in result["targets"]:
        target["policy_metrics"][diagnostic_policy] = _metrics(9.1)

    matrix = build_budget_matrix([result], expected_budgets=[0])

    assert matrix["passed"] is True
    audit = matrix["individual_audits"][0]
    assert audit["deployable_safety_totals"]["collision_events"] == 0
    assert audit["unguarded_diagnostic_safety_totals"]["collision_events"] == 1
    assert any("unguarded MPC" in value for value in matrix["warnings"])


def test_budget_matrix_audits_native_collision_without_blanket_failure():
    result = _result(0)
    fixed = next(
        row for row in result["seed_results"] if row["policy"] == "fixed_program"
    )
    fixed["metrics"]["collision_events"] = 3
    fixed["metrics"]["collision_incidents"] = 1

    matrix = build_budget_matrix([result], expected_budgets=[0])

    assert matrix["passed"] is True
    assert any("raw benchmark junction collisions" in value for value in matrix["warnings"])


def test_budget_matrix_rejects_primary_collision_incident_excess():
    result = _result(0)
    primary = result["setting"]["primary_reference_policy"]
    row = next(
        row for row in result["seed_results"] if row["policy"] == primary
    )
    row["metrics"]["collision_events"] = 1
    row["metrics"]["collision_incidents"] = 1

    matrix = build_budget_matrix([result], expected_budgets=[0])

    assert matrix["passed"] is False
    assert any("collision-incident noninferiority" in value for value in matrix["errors"])


def test_budget_matrix_rejects_non_nested_target_groups():
    first = _result(8)
    second = _result(16)
    second = copy.deepcopy(second)
    adaptation = second["model_diagnostics"]["city_a_net"]["target_adaptation"]
    replacement = [f"replacement:{index}" for index in range(16)]
    adaptation["selected_group_ids"] = replacement
    adaptation["adaptation_group_ids"] = replacement[:8]
    adaptation["calibration_group_ids"] = replacement[8:]
    matrix = build_budget_matrix([first, second], expected_budgets=[8, 16])
    assert matrix["passed"] is False
    assert any("not nested" in value for value in matrix["errors"])


def test_budget_matrix_rejects_calibration_refit_into_deployment_model():
    result = _result(8)
    adaptation = result["model_diagnostics"]["city_a_net"]["target_adaptation"]
    adaptation["calibration_groups_refit_into_deployment_model"] = True
    adaptation["deployment_model_group_ids"] = adaptation["selected_group_ids"]
    adaptation["deployment_model_groups"] = len(adaptation["selected_group_ids"])

    matrix = build_budget_matrix([result], expected_budgets=[8])

    assert matrix["passed"] is False
    assert any(
        "calibration labels entered deployment fitting" in value
        for value in matrix["errors"]
    )


def test_budget_matrix_rejects_action_target_scale_protocol_drift():
    result = _result(8)
    result["setting"]["action_target_scale_protocol"] = "legacy-fixed-floor"

    matrix = build_budget_matrix([result], expected_budgets=[8])

    assert matrix["passed"] is False
    assert any("action target scale protocol" in value for value in matrix["errors"])


def test_budget_matrix_rejects_cfcmt_hierarchy_protocol_drift():
    result = _result(8)
    result["setting"]["cfcmt_hierarchy_protocol"] = "legacy-two-layer"

    matrix = build_budget_matrix([result], expected_budgets=[8])

    assert matrix["passed"] is False
    assert any("CFCMT hierarchy protocol" in value for value in matrix["errors"])


def test_budget_matrix_uses_equal_city_paired_ranking_not_raw_scale():
    result = _result(0)
    kept = ("selected_source_prior", "dense_contrast_guard")
    result["setting"]["policies"] = list(kept)
    result["setting"]["primary_reference_policy"] = "dense_contrast_guard"
    result["seed_results"] = [
        row for row in result["seed_results"] if row["policy"] in kept
    ]
    costs = {
        "city_a_net": {"selected_source_prior": 1000.0, "dense_contrast_guard": 990.0},
        "city_b_net": {"selected_source_prior": 10.0, "dense_contrast_guard": 11.0},
        "city_c_net": {"selected_source_prior": 10.0, "dense_contrast_guard": 11.0},
    }
    for row in result["seed_results"]:
        row["metrics"][PRIMARY_METRIC] = costs[row["target"]][row["policy"]]
    for target in result["targets"]:
        name = target["target"]
        target["policy_metrics"] = {
            policy: {
                **target["policy_metrics"][policy],
                PRIMARY_METRIC: costs[name][policy],
            }
            for policy in kept
        }
    result["aggregate"] = {
        policy: {
            PRIMARY_METRIC: sum(costs[target][policy] for target in SCENARIOS)
            / len(SCENARIOS)
        }
        for policy in kept
    }

    matrix = build_budget_matrix([result], expected_budgets=[0])

    assert matrix["passed"] is True
    row = matrix["summary"][0]
    assert row["best_absolute_policy"] == "dense_contrast_guard"
    assert row["best_policy"] == "selected_source_prior"
    assert row["primary_relative_to_selected_source_prior"] > 0.0


def test_budget_matrix_flags_single_city_improvement_concentration():
    result = _result(16)
    primary = "cfcmt_mechanism_contrast_regularized"
    prior = "selected_source_prior"
    costs = {
        "city_a_net": {primary: 5.0, prior: 10.0},
        "city_b_net": {primary: 10.0, prior: 10.0},
        "city_c_net": {primary: 10.0, prior: 10.0},
    }
    for row in result["seed_results"]:
        if row["policy"] in {primary, prior}:
            row["metrics"][PRIMARY_METRIC] = costs[row["target"]][row["policy"]]
    for target in result["targets"]:
        name = target["target"]
        for policy in (primary, prior):
            target["policy_metrics"][policy][PRIMARY_METRIC] = costs[name][policy]
    result["aggregate"][primary][PRIMARY_METRIC] = 25.0 / 3.0
    result["aggregate"][prior][PRIMARY_METRIC] = 10.0

    matrix = build_budget_matrix([result], expected_budgets=[16])

    assert matrix["passed"] is True
    summary = matrix["summary"][0]
    assert summary["primary_broadly_improving"] is False
    assert summary["primary_win_city_groups"] == 1
    assert summary["primary_top_gain_city"] == "a"
    assert summary["primary_top_gain_share"] == 1.0
    assert any("concentrated in a" in value for value in matrix["warnings"])
