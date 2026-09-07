import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    COUNTERFACTUAL_CACHE_VERSION,
    MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
    _aggregate_rows,
    _apply_local_graph_selector_v3,
    _collect_source_rule_rows_v3,
    _fit_target_models,
    _json_ready,
    _map_fresh_sumo_processes,
    _merge_collection_shards_v3,
    _paired_city_effects,
    _primary_reference_policy_v3,
    _rule_calibration_worker,
    _source_rule_cache_path_v3,
    _static_context_from_dataset,
    _sum_numeric,
    _strict_city_fold,
    _target_adaptation_split_v3,
    _target_adaptation_subset_v3,
    _validate_counterfactual_cache_dataset,
    select_target_closed_loop_policies_v3,
)
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset, save_mechanism_dataset
from cf_h2o.traffic_signal.mechanism_world_model import (
    MechanismDataset,
    MechanismFitConfig,
)
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport
from cf_h2o.traffic_signal.intervention_graph import (
    add_routing_conflicts,
    build_tls_intervention_graph,
    greedy_independent_interventions,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CFCMT_CORE_FAMILY_V3,
    CFCMT_TARGET_SPECIALIST_FAMILY_V3,
    COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
    ContrastGuardAudit,
    ContrastGuardConfig,
    ContrastProposal,
    PriorRegularizationConfig,
    ResidualExecutionTrustRegionConfig,
    STATE_SNAPSHOT_PROTOCOL_V3,
    STRICT_SAFETY_MONITORING_V3,
    UnsafeRolloutError,
    OUTPUT_NAMES_V3,
    OUTPUT_NAMES_V4,
    ONE_STEP_OUTPUT_NAMES_V4,
    POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
    _advance_with_actions,
    _finalize_safety_ledger,
    _least_covered_focal_tls_v3,
    _new_safety_ledger,
    _group_adjusted_scores,
    _relative_contrast_rule_gap,
    _contrast_proposal,
    _coordinate_contrast_proposals,
    _apply_residual_execution_trust_region,
    _calibrate_source_loo_uncertainty_v3,
    _fit_family_with_source_loo_v3,
    _select_hierarchical_proposal,
    calibrate_contrast_guard_v3,
    calibrate_prior_regularization_v3,
    has_fitted_target_specialist_v3,
    merge_counterfactual_datasets_v3,
    policy_consistent_mechanism_priors_v4,
    rollout_prefix_target_name,
    select_target_capacity_layers_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import (
    CONTEXT_NAMES_V2,
    FEATURE_NAMES_V2,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2_suite import paired_hierarchical_bootstrap
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import SUMO_EXECUTION_PROTOCOL


def test_fresh_sumo_workers_reject_inaccessible_spawn_directory(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.os.access",
        lambda *_args: False,
    )

    with pytest.raises(PermissionError, match="cannot enter the current directory"):
        _map_fresh_sumo_processes(object(), ({"job": 1},), workers=1)


class _SafetySimulation:
    def __init__(self, *, collisions=(), starting_teleports=0, ending_teleports=0):
        self._time = 0.0
        self._collisions = tuple(collisions)
        self._starting_teleports = int(starting_teleports)
        self._ending_teleports = int(ending_teleports)

    def getTime(self):
        return self._time

    def simulationStep(self):
        self._time += 1.0

    def getStartingTeleportNumber(self):
        return self._starting_teleports

    def getEndingTeleportNumber(self):
        return self._ending_teleports

    def getCollisions(self):
        return self._collisions


def _safety_api(*, collisions=(), starting_teleports=0, ending_teleports=0):
    simulation = _SafetySimulation(
        collisions=collisions,
        starting_teleports=starting_teleports,
        ending_teleports=ending_teleports,
    )
    return SimpleNamespace(
        simulation=simulation,
        simulationStep=simulation.simulationStep,
        vehicle=SimpleNamespace(getIDList=lambda: ()),
        trafficlight=SimpleNamespace(getIDList=lambda: ()),
    )


def test_counterfactual_collision_can_be_audited_without_becoming_a_label():
    collision = SimpleNamespace(
        collider="a",
        victim="b",
        type="junction",
        lane=":tls_0_0",
    )
    ledger = _new_safety_ledger()
    _advance_with_actions(
        sumo_api=_safety_api(collisions=(collision,)),
        executors={},
        actions={},
        control_interval_sec=1,
        collision_mode="audit",
        safety_ledger=ledger,
    )
    audit = _finalize_safety_ledger(ledger)
    assert audit["raw_collision_events"] == 1
    assert audit["unique_collision_incidents"] == 1
    assert audit["no_teleport_passed"] is True

    with pytest.raises(UnsafeRolloutError, match="junction collision"):
        _advance_with_actions(
            sumo_api=_safety_api(collisions=(collision,)),
            executors={},
            actions={},
            control_interval_sec=1,
            collision_mode="fail",
        )


def test_teleport_is_a_hard_failure_even_in_collision_audit_mode():
    with pytest.raises(UnsafeRolloutError, match="teleport"):
        _advance_with_actions(
            sumo_api=_safety_api(starting_teleports=1),
            executors={},
            actions={},
            control_interval_sec=1,
            collision_mode="audit",
        )


def test_policy_consistent_priors_use_control_and_rollout_horizons_separately():
    features = np.zeros((1, len(FEATURE_NAMES_V2)), dtype=float)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES_V2)}
    for name, value in {
        "total_q": 40.0,
        "total_veh": 50.0,
        "mean_speed": 4.0,
        "mean_occ": 0.35,
        "green_q": 24.0,
        "red_q": 16.0,
        "green_veh": 30.0,
        "green_down_occ": 0.20,
        "green_down_q": 3.0,
        "green_link_ratio": 0.5,
        "green_lane_ratio": 0.5,
        "red_pressure": 12.0,
        "clearance_fraction": 0.0,
        "lane_count_norm": 0.5,
    }.items():
        features[:, feature_index[name]] = value
    context = np.zeros((1, len(CONTEXT_NAMES_V2)), dtype=float)
    context[:, 0] = 1.0
    context[:, 1] = 0.5

    priors = policy_consistent_mechanism_priors_v4(
        features,
        context,
        control_interval_sec=10,
        counterfactual_horizon_intervals=6,
    )

    assert tuple(priors) == OUTPUT_NAMES_V4
    assert priors["one_step_total_queue"][0] > priors["next_total_queue"][0]
    assert priors["one_step_green_queue"][0] > priors["next_green_queue"][0]
    assert np.isfinite(np.asarray(list(priors.values()), dtype=float)).all()


def test_v3_audit_aggregation_recomputes_rates():
    result = _sum_numeric(
        [
            {"decisions": 10, "proposed_overrides": 4, "accepted_overrides": 2, "acceptance_rate": 0.5},
            {"decisions": 10, "proposed_overrides": 6, "accepted_overrides": 1, "acceptance_rate": 1 / 6},
        ]
    )
    assert result["decisions"] == 20
    assert result["proposal_rate"] == 0.5
    assert result["acceptance_rate"] == 0.3


def test_target_closed_loop_selector_compares_residual_to_selected_prior(
    monkeypatch,
    tmp_path,
):
    sumocfg = tmp_path / "target.sumocfg"
    sumocfg.write_text("<configuration/>\n", encoding="utf-8")
    model = SimpleNamespace(
        guards={"cfcmt_fused": ContrastGuardConfig(enabled=True)},
        regularizers={
            "cfcmt_fused": PriorRegularizationConfig(enabled=False)
        },
        family_models={"cfcmt_fused": object()},
        hierarchy_layers={},
        diagnostics={"cfcmt_hierarchy": {}},
    )
    observed_policies = []

    def fake_map(_worker, jobs, _workers):
        rows = []
        for job in jobs:
            policy = str(job["policy"])
            observed_policies.append(policy)
            cost = 9.0 if policy == "cfcmt_fused_contrast_guard" else 10.0
            rows.append(
                {
                    "target": str(job["scenario"]),
                    "policy": policy,
                    "seed": int(job["seed"]),
                    "metrics": {
                        "ok": True,
                        "mean_system_vehicles_per_controlled_lane": cost,
                        "p90_queue_per_lane": 5.0,
                        "throughput_ratio": 0.8,
                        "completion_ratio": 0.7,
                        "departure_service_ratio": 0.9,
                        "teleport_rate": 0.0,
                        "collision_rate": 0.0,
                    },
                }
            )
        return rows

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite._map_fresh_sumo_processes",
        fake_map,
    )
    result = select_target_closed_loop_policies_v3(
        {"target": model},
        env_root=tmp_path,
        policies=("cfcmt_fused_contrast_guard",),
        seeds=(17,),
        duration_sec=60.0,
        control_interval_sec=10,
        warmup_sec=10.0,
        workers=1,
        tripinfo_root=tmp_path / "tripinfo",
        scenario_sumocfgs={"target": sumocfg},
    )

    diagnostic = result["diagnostics"]["target"][
        "cfcmt_fused_contrast_guard"
    ]
    assert set(observed_policies) == {
        "cfcmt_fused_contrast_guard",
        "selected_source_prior",
    }
    assert diagnostic["selection_reference_policy"] == "selected_source_prior"
    assert diagnostic["accepted"] is True


def test_v3_aggregation_is_target_first():
    rows = []
    for target, values in (("small", (1.0, 3.0)), ("large", (9.0, 11.0))):
        for seed, value in enumerate(values):
            rows.append(
                {
                    "target": target,
                    "policy": "p",
                    "seed": seed,
                    "metrics": {
                        "ok": True,
                        "mean_queue": value,
                        "p90_queue": value,
                        "mean_queue_per_lane": value,
                        "p90_queue_per_lane": value,
                        "mean_queue_proxy": value,
                        "mean_queue_proxy_per_lane": value,
                        "halted_vehicle_hours": value,
                        "active_vehicle_hours": value,
                        "pending_vehicle_hours": value,
                        "system_vehicle_hours": value,
                        "mean_active_vehicles": value,
                        "mean_active_vehicles_per_controlled_lane": value,
                        "mean_system_vehicles": value,
                        "mean_system_vehicles_per_controlled_lane": value,
                        "active_vehicles_at_horizon": value,
                        "pending_vehicles_at_horizon": value,
                        "mean_tripinfo_duration": value,
                        "mean_tripinfo_waiting_time": value,
                        "mean_tripinfo_time_loss": value,
                        "throughput_ratio": value,
                        "departure_service_ratio": value,
                        "completion_ratio": value,
                        "starting_teleports": value,
                        "ending_teleports": value,
                        "collision_events": value,
                        "collision_event_steps": value,
                        "collision_incidents": value,
                        "emergency_stops": value,
                        "teleport_rate": value,
                        "collision_rate": value,
                        "collision_incident_rate": value,
                        "emergency_stop_rate": value,
                        "phase_execution_audit": {},
                    },
                }
            )
    targets, aggregate = _aggregate_rows(rows, policies=("p",))
    assert len(targets) == 2
    assert np.isclose(aggregate["p"]["mean_queue"], 6.0)


def test_v3_aggregation_is_city_first_when_subnet_counts_differ():
    rows = []
    for target, value in (("a1", 1.0), ("a2", 3.0), ("b1", 10.0)):
        rows.append(
            {
                "target": target,
                "policy": "p",
                "seed": 1,
                "metrics": {
                    "ok": True,
                    **{name: value for name in (
                        "mean_queue", "p90_queue", "mean_queue_per_lane",
                        "p90_queue_per_lane", "mean_queue_proxy",
                        "mean_queue_proxy_per_lane", "mean_tripinfo_duration",
                        "halted_vehicle_hours", "active_vehicle_hours",
                        "pending_vehicle_hours", "system_vehicle_hours",
                        "mean_active_vehicles", "mean_active_vehicles_per_controlled_lane",
                        "mean_system_vehicles", "mean_system_vehicles_per_controlled_lane",
                        "active_vehicles_at_horizon",
                        "pending_vehicles_at_horizon",
                        "mean_tripinfo_waiting_time", "mean_tripinfo_time_loss",
                        "throughput_ratio",
                        "departure_service_ratio", "completion_ratio",
                        "starting_teleports", "ending_teleports",
                        "collision_events", "emergency_stops",
                        "collision_event_steps", "collision_incidents",
                        "teleport_rate", "collision_rate", "collision_incident_rate",
                        "emergency_stop_rate",
                    )},
                },
            }
        )
    _, aggregate = _aggregate_rows(
        rows,
        policies=("p",),
        scenario_city_groups={"a1": "a", "a2": "a", "b1": "b"},
    )
    assert np.isclose(aggregate["p"]["mean_queue_per_lane"], 6.0)
    assert aggregate["p"]["city_group_count"] == 2


def test_paired_city_effects_normalize_before_city_aggregation():
    target_rows = [
        {
            "target": "large",
            "city_group": "a",
            "policy_metrics": {
                "method": {"cost": 90.0},
                "phase_pressure": {"cost": 100.0},
            },
        },
        {
            "target": "small1",
            "city_group": "b",
            "policy_metrics": {
                "method": {"cost": 1.1},
                "phase_pressure": {"cost": 1.0},
            },
        },
        {
            "target": "small2",
            "city_group": "b",
            "policy_metrics": {
                "method": {"cost": 2.2},
                "phase_pressure": {"cost": 2.0},
            },
        },
    ]
    result = _paired_city_effects(
        target_rows,
        policies=("method", "phase_pressure"),
        reference="phase_pressure",
        metric="cost",
    )
    assert np.isclose(result["method"]["city_relative_effect"]["a"], -0.1)
    assert np.isclose(result["method"]["city_relative_effect"]["b"], 0.1)
    assert np.isclose(result["method"]["mean_relative_effect"], 0.0)


def test_strict_city_fold_removes_all_same_city_subnets():
    groups = {"c1": "cologne", "c3": "cologne", "i1": "ingolstadt", "a": "atlanta"}
    source, heldout = _strict_city_fold(tuple(groups), "c1", groups)
    assert source == ["i1", "a"]
    assert heldout == ["c1", "c3"]


def test_bootstrap_reports_city_clusters_and_relative_effects():
    rows = []
    for target, city in (("c1", "cologne"), ("c3", "cologne"), ("i1", "ingolstadt")):
        for seed in (1, 2):
            for policy, value in (("method", 9.0), ("base", 10.0)):
                rows.append(
                    {
                        "target": target,
                        "policy": policy,
                        "seed": seed,
                        "metrics": {"ok": True, "mean_queue_per_lane": value},
                    }
                )
    result = paired_hierarchical_bootstrap(
        rows,
        reference="method",
        baseline="base",
        metric="mean_queue_per_lane",
        effect="relative",
        clusters={"c1": "cologne", "c3": "cologne", "i1": "ingolstadt"},
        samples=200,
    )
    assert result["network_count"] == 3
    assert result["cluster_count"] == 2
    assert np.isclose(result["mean_delta"], -0.1)
    assert result["win_clusters"] == 2


def test_mechanism_dataset_cache_round_trip(tmp_path):
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.asarray([[1.0], [2.0]]),
        context_names=("z",),
        context=np.asarray([[3.0], [4.0]]),
        priors={"y": np.asarray([5.0, 6.0])},
        targets={"y": np.asarray([7.0, 8.0])},
        domains=np.asarray(["a", "b"]),
        metadata={"scenario": "cache_test", "action_group_ids": ["a0", "b0"]},
    )
    path = tmp_path / "dataset.npz"
    save_mechanism_dataset(path, dataset)
    loaded = load_mechanism_dataset(path)
    assert loaded.feature_names == dataset.feature_names
    assert loaded.context_names == dataset.context_names
    assert np.array_equal(loaded.domains, dataset.domains)
    assert np.allclose(loaded.features, dataset.features)
    assert np.allclose(loaded.targets["y"], dataset.targets["y"])
    assert loaded.metadata == dataset.metadata


def test_group_adjusted_scores_maps_unsorted_groups_without_full_rescans():
    groups = ["g2", "g1", "g3", "g2", "g1", "g3"]
    references = [False, True, False, True, False, True]
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.arange(6, dtype=float).reshape(-1, 1),
        context_names=("z",),
        context=np.ones((6, 1), dtype=float),
        priors={"control_cost": np.zeros(6)},
        targets={"control_cost": np.zeros(6)},
        domains=np.asarray(["city"] * 6),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
        },
    )
    mean = np.asarray([8.0, 4.0, 9.0, 5.0, 7.0, 3.0])
    prediction = {
        "control_cost": {
            "mean": mean,
            "uncertainty": np.ones(6),
            "context_trust": np.full(6, 0.75),
        }
    }

    score, uncertainty, trust, reference_rows = _group_adjusted_scores(
        dataset,
        prediction,
        objective_mode="control_only",
    )

    assert np.array_equal(reference_rows, [3, 1, 5, 3, 1, 5])
    assert np.allclose(score, [3.0, 0.0, 6.0, 0.0, 3.0, 0.0])
    assert np.allclose(uncertainty, [2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    assert np.allclose(trust, 0.75)


def test_static_target_context_rejects_transition_varying_values():
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.asarray([[1.0], [2.0]]),
        context_names=("static",),
        context=np.asarray([[3.0], [3.0]]),
        priors={"y": np.zeros(2)},
        targets={"y": np.zeros(2)},
        domains=np.asarray(["a", "a"]),
        metadata={"action_group_ids": ["g", "g"]},
    )
    assert np.allclose(_static_context_from_dataset(dataset), [3.0])
    varying = MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=np.asarray([[3.0], [4.0]]),
        priors=dataset.priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata=dataset.metadata,
    )
    with pytest.raises(ValueError, match="not label-free static"):
        _static_context_from_dataset(varying)


def _counterfactual_shard(index: int, group: str) -> MechanismDataset:
    return MechanismDataset(
        feature_names=("x",),
        features=np.asarray([[float(index)]]),
        context_names=("z",),
        context=np.asarray([[1.0]]),
        priors={name: np.zeros(1) for name in OUTPUT_NAMES_V3},
        targets={name: np.ones(1) * index for name in OUTPUT_NAMES_V3},
        domains=np.asarray(["scenario"]),
        metadata={
            "scenario": "scenario",
            "seed": 17,
            "duration_sec": 120.0,
            "control_interval_sec": 10,
            "warmup_sec": 60.0,
            "max_focal_tls": 4,
            "counterfactual_horizon_sec": 20,
            "counterfactual_cost_scope": "system",
            "behavior_policy": "phase_pressure",
            "state_snapshot_protocol": dict(STATE_SNAPSHOT_PROTOCOL_V3),
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "behavior_safety_audit": {
                "raw_collision_events": 0,
                "collision_event_steps": 0,
                "unique_collision_incidents": 0,
                "starting_teleports": 0,
                "ending_teleports": 0,
                "collision_samples": [],
                "no_teleport_passed": True,
            },
            "counterfactual_safety_audit": {
                "protocol": COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
                "candidate_groups": 1,
                "retained_groups": 1,
                "censored_groups": 0,
                "candidate_branches": 1,
                "attempted_branches": 1,
                "retained_branches": 1,
                "censored_branches": 0,
                "unevaluated_censored_branches": 0,
                "observed_unsafe_branches": 0,
                "raw_collision_events_observed": 0,
                "collision_event_steps_observed": 0,
                "unique_collision_incidents_observed": 0,
                "censored_group_fraction": 0.0,
                "censored_group_ids": [],
                "collision_samples": [],
                "starting_teleports": 0,
                "ending_teleports": 0,
                "no_teleport_passed": True,
                "symmetric_group_censoring_passed": True,
            },
            "counterfactual_replay_audit": {
                "checks": 1,
                "max_abs_difference": 0.0,
                "passed": True,
            },
            "collection_shard_index": index,
            "collection_shard_count": 2,
            "eligible_collection_opportunities": 2,
            "behavior_trace_sha256": "a" * 64,
            "behavior_interval_count": 12,
            "captured_snapshot_count": 1,
            "action_group_ids": [group],
            "row_tls": [f"tls{index}"],
            "row_times": [float(index)],
            "candidate_states": [f"state{index}"],
            "counterfactual_branches": 1,
            "state_restores": 2,
            "phase_execution_audit": {},
            "focal_selection_protocol": STATE_SNAPSHOT_PROTOCOL_V3["focal_selection"],
            "focal_selection_count_by_tls": {"a": 1, "b": 1},
            "focal_selection_min_count": 1,
            "focal_selection_max_count": 1,
            "controllable_tls_ids": ["a", "b"],
            "covered_tls_ids": ["a" if index == 0 else "b"],
        },
    )


def test_counterfactual_collection_shards_merge_without_group_overlap():
    merged = _merge_collection_shards_v3(
        "scenario",
        17,
        [_counterfactual_shard(0, "g0"), _counterfactual_shard(1, "g1")],
    )
    assert merged.size == 2
    assert merged.metadata["collection_shard_indices"] == [0, 1]
    assert merged.metadata["counterfactual_branches"] == 2
    assert merged.metadata["tls_coverage_fraction"] == 1.0
    assert merged.metadata["counterfactual_replay_audit"]["passed"] is True
    assert merged.metadata["row_tls"] == ["tls0", "tls1"]
    assert merged.metadata["candidate_states"] == ["state0", "state1"]


def test_counterfactual_collection_shards_require_complete_partition():
    with pytest.raises(ValueError, match="incomplete counterfactual shard set"):
        _merge_collection_shards_v3("scenario", 17, [_counterfactual_shard(0, "g0")])


def test_least_covered_focal_selection_covers_large_static_network():
    tls_ids = tuple(f"tls_{index:03d}" for index in range(196))
    counts = {tls_id: 0 for tls_id in tls_ids}
    for opportunity in range(54):
        selected = _least_covered_focal_tls_v3(
            eligible_tls=tls_ids,
            controllable_tls=tls_ids,
            selection_counts=counts,
            opportunity_index=opportunity,
            max_focal_tls=4,
        )
        for tls_id in selected:
            counts[tls_id] += 1
    assert min(counts.values()) == 1
    assert max(counts.values()) == 2
    assert sum(counts.values()) == 54 * 4


def test_least_covered_focal_selection_catches_up_newly_feasible_tls():
    tls_ids = ("a", "b", "c", "d")
    counts = {tls_id: 0 for tls_id in tls_ids}
    eligible_by_opportunity = (("a", "b"), ("a", "b"), tls_ids, tls_ids)
    for opportunity, eligible in enumerate(eligible_by_opportunity):
        selected = _least_covered_focal_tls_v3(
            eligible_tls=eligible,
            controllable_tls=tls_ids,
            selection_counts=counts,
            opportunity_index=opportunity,
            max_focal_tls=2,
        )
        for tls_id in selected:
            counts[tls_id] += 1
    assert counts == {"a": 2, "b": 2, "c": 2, "d": 2}


def test_source_loo_uncertainty_calibration_only_inflates_by_worst_city():
    records = {
        "city_a": {
            "score": np.asarray([0.0, 0.0, 0.0]),
            "actual": np.asarray([100.0, 2.0, -2.0]),
            "uncertainty": np.asarray([0.0, 1.0, 1.0]),
            "reference_rows": np.asarray([0, 0, 0]),
        },
        "city_b": {
            "score": np.asarray([0.0, 0.0]),
            "actual": np.asarray([100.0, 4.0]),
            "uncertainty": np.asarray([0.0, 1.0]),
            "reference_rows": np.asarray([0, 0]),
        },
    }
    (scaled, min_trust), diagnostics = _calibrate_source_loo_uncertainty_v3(
        (records, 0.25), quantile=0.90
    )
    assert min_trust == 0.25
    assert diagnostics["scale"] == 4.0
    assert diagnostics["domain_scales"] == {"city_a": 2.0, "city_b": 4.0}
    assert diagnostics["shrinkage_allowed"] is False
    assert np.allclose(scaled["city_a"]["uncertainty"], [0.0, 4.0, 4.0])


def test_source_loo_thread_pool_is_numerically_equivalent_to_serial():
    features = []
    targets = []
    domains = []
    groups = []
    references = []
    for domain_index, domain in enumerate(("a", "b", "c", "d")):
        for group_index in range(6):
            for action in (0.0, 1.0):
                features.append([4.0 + group_index + action, action, float(domain_index)])
                targets.append(action * (0.2 * domain_index - 0.3 * group_index))
                domains.append(domain)
                groups.append(f"{domain}:g{group_index}")
                references.append(action == 0.0)
    values = np.asarray(targets, dtype=float)
    contrast = MechanismDataset(
        feature_names=("green_q", "delta_green_q", "domain_index"),
        features=np.asarray(features, dtype=float),
        context_names=("network_scale",),
        context=np.asarray([[float(ord(domain) - ord("a"))] for domain in domains]),
        priors={"interval_cost": np.zeros_like(values)},
        targets={"interval_cost": values},
        domains=np.asarray(domains),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "reference_rows": [index - (index % 2) for index in range(len(groups))],
            "reference_policy": "phase_pressure",
        },
    )
    serial_model, serial_fit, serial_loo, serial_parallel = (
        _fit_family_with_source_loo_v3(
            contrast,
            family="dense_advantage",
            cfcmt_config=SimpleNamespace(),
            target_domain=None,
            parallel_workers=1,
        )
    )
    parallel_model, parallel_fit, parallel_loo, parallel_diagnostics = (
        _fit_family_with_source_loo_v3(
            contrast,
            family="dense_advantage",
            cfcmt_config=SimpleNamespace(),
            target_domain=None,
            parallel_workers=5,
        )
    )
    assert serial_fit == parallel_fit
    assert serial_parallel["actual_workers"] == 1
    assert parallel_diagnostics["actual_workers"] == 5
    assert serial_loo.keys() == parallel_loo.keys()
    assert np.allclose(
        serial_model.predict(contrast)["control_cost"]["mean"],
        parallel_model.predict(contrast)["control_cost"]["mean"],
    )
    for domain in serial_loo:
        serial_prediction = serial_loo[domain][1]["control_cost"]
        parallel_prediction = parallel_loo[domain][1]["control_cost"]
        for key in ("mean", "uncertainty", "context_trust"):
            assert np.allclose(serial_prediction[key], parallel_prediction[key])

    reused_model, reused_fit, reused_loo, reused_parallel = (
        _fit_family_with_source_loo_v3(
            contrast,
            family="dense_advantage",
            cfcmt_config=SimpleNamespace(),
            target_domain=None,
            parallel_workers=5,
            reused_prediction_bundle=serial_loo,
        )
    )
    assert reused_fit == serial_fit
    assert reused_parallel["source_loo_fit_count"] == 0
    assert reused_parallel["reused_source_loo_fit_count"] == 4
    assert reused_parallel["reuse_protocol"] == "exact_matched_family_artifacts_v1"
    assert reused_loo.keys() == serial_loo.keys()
    assert np.allclose(
        reused_model.predict(contrast)["control_cost"]["mean"],
        serial_model.predict(contrast)["control_cost"]["mean"],
        rtol=0.0,
        atol=1e-12,
    )

    heldout = next(iter(serial_loo))
    valid, prediction = serial_loo[heldout]
    changed_target = np.asarray(valid.targets["interval_cost"], dtype=float).copy()
    changed_target[-1] += 1.0
    invalid_loo = dict(serial_loo)
    invalid_loo[heldout] = (
        replace(
            valid,
            targets={**valid.targets, "interval_cost": changed_target},
        ),
        prediction,
    )
    with pytest.raises(ValueError, match="dataset mismatch"):
        _fit_family_with_source_loo_v3(
            contrast,
            family="dense_advantage",
            cfcmt_config=SimpleNamespace(),
            target_domain=None,
            parallel_workers=5,
            reused_prediction_bundle=invalid_loo,
        )


def test_source_rule_cache_path_is_content_addressed(tmp_path):
    identity = {
        "version": "v1",
        "scenario": "grid",
        "seed": 17,
        "policy_spec": {"downstream_queue_weight": 0.45},
    }
    first = _source_rule_cache_path_v3(tmp_path, identity)
    second = _source_rule_cache_path_v3(tmp_path, dict(reversed(tuple(identity.items()))))
    changed = _source_rule_cache_path_v3(tmp_path, {**identity, "seed": 18})
    assert first == second
    assert first != changed
    assert first.parent == tmp_path


def test_source_rule_worker_uses_valid_cached_rollout(tmp_path, monkeypatch):
    identity = {"version": "v1", "scenario": "grid", "seed": 17}
    row = {
        "scenario": "grid",
        "policy": "phase_pressure",
        "pressure_spec": {},
        "seed": 17,
        "metrics": {
            "ok": True,
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "collision_events": 0,
            "collision_incidents": 0,
            "starting_teleports": 0,
            "ending_teleports": 0,
            "mean_system_vehicles_per_controlled_lane": 1.0,
        },
    }
    path = tmp_path / "source_rule.json"
    path.write_text(
        json.dumps({"identity": identity, "row": row}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.evaluate_policy_v3",
        lambda **kwargs: pytest.fail("valid source-rule cache should bypass SUMO"),
    )
    loaded = _rule_calibration_worker(
        {
            "scenario": "grid",
            "policy_key": "phase_pressure",
            "pressure_spec": {},
            "cache_identity": identity,
            "cache_path": str(path),
        }
    )
    assert loaded == row


def test_source_rule_cache_hits_avoid_fresh_processes_and_are_sorted(
    tmp_path, monkeypatch
):
    jobs = []
    for scenario, seed in (("z", 2), ("a", 3), ("a", 1)):
        identity = {"scenario": scenario, "seed": seed}
        row = {
            "scenario": scenario,
            "policy": "phase_pressure",
            "seed": seed,
            "metrics": {
                "ok": True,
                "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
                "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
                "collision_events": 0,
                "collision_incidents": 0,
                "starting_teleports": 0,
                "ending_teleports": 0,
            },
        }
        path = tmp_path / f"{scenario}_{seed}.json"
        path.write_text(
            json.dumps({"identity": identity, "row": row}), encoding="utf-8"
        )
        jobs.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "policy_key": "phase_pressure",
                    "pressure_spec": {},
                    "cache_identity": identity,
                "cache_path": str(path),
            }
        )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite._map_fresh_sumo_processes",
        lambda *args, **kwargs: pytest.fail(
            "complete source-rule cache should not start fresh processes"
        ),
    )
    rows = _collect_source_rule_rows_v3(jobs, workers=8)
    assert [(row["scenario"], row["seed"]) for row in rows] == [
        ("a", 1),
        ("a", 3),
        ("z", 2),
    ]


def test_source_rule_worker_audits_but_accepts_cached_collision(tmp_path):
    identity = {"version": "v1", "scenario": "grid", "seed": 17}
    row = {
        "scenario": "grid",
        "policy": "phase_pressure",
        "seed": 17,
        "metrics": {
            "ok": True,
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "collision_events": 1,
            "collision_incidents": 1,
            "starting_teleports": 0,
            "ending_teleports": 0,
        },
    }
    path = tmp_path / "unsafe_source_rule.json"
    path.write_text(
        json.dumps({"identity": identity, "row": row}), encoding="utf-8"
    )
    loaded = _rule_calibration_worker(
        {
            "scenario": "grid",
            "policy_key": "phase_pressure",
            "pressure_spec": {},
            "cache_identity": identity,
            "cache_path": str(path),
        }
    )
    assert loaded["metrics"]["collision_incidents"] == 1


def test_source_rule_worker_rejects_cached_teleport(tmp_path):
    identity = {"version": "v1", "scenario": "grid", "seed": 17}
    row = {
        "scenario": "grid",
        "policy": "phase_pressure",
        "seed": 17,
        "metrics": {
            "ok": True,
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "collision_events": 0,
            "collision_incidents": 0,
            "starting_teleports": 1,
            "ending_teleports": 0,
        },
    }
    path = tmp_path / "teleport_source_rule.json"
    path.write_text(
        json.dumps({"identity": identity, "row": row}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="teleport events"):
        _rule_calibration_worker(
            {
                "scenario": "grid",
                "policy_key": "phase_pressure",
                "pressure_spec": {},
                "cache_identity": identity,
                "cache_path": str(path),
            }
        )


def test_counterfactual_cache_validation_rejects_wrong_identity():
    identity = {
        "scenario": "scenario",
        "seed": 17,
        "collection_shard_index": 0,
        "collection_shard_count": 2,
    }
    dataset = _counterfactual_shard(0, "g0")
    dataset = MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata={**dict(dataset.metadata), "counterfactual_cache_identity": identity},
    )
    _validate_counterfactual_cache_dataset(dataset, identity, None)
    with pytest.raises(ValueError, match="identity mismatch"):
        _validate_counterfactual_cache_dataset(dataset, {**identity, "seed": 18}, None)


def test_current_counterfactual_cache_requires_policy_consistent_estimand():
    identity = {
        "version": COUNTERFACTUAL_CACHE_VERSION,
        "scenario": "scenario",
        "seed": 17,
        "control_interval_sec": 10,
        "counterfactual_horizon_intervals": 2,
        "collection_shard_index": 0,
        "collection_shard_count": 2,
    }
    source = _counterfactual_shard(0, "g0")
    metadata = {
        **dict(source.metadata),
        "counterfactual_cache_identity": identity,
        "counterfactual_estimand": POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
        "mechanism_output_names": list(OUTPUT_NAMES_V4),
        "local_mechanism_horizon_sec": 10,
        "rollout_value_horizon_sec": 20,
        "mechanism_estimands": {
            "one_step_local_state": {
                "output_names": list(ONE_STEP_OUTPUT_NAMES_V4),
                "horizon_sec": 10,
                "analytic_prior_horizon_sec": 10,
            }
        },
    }
    dataset = MechanismDataset(
        feature_names=source.feature_names,
        features=source.features,
        context_names=source.context_names,
        context=source.context,
        priors={name: np.zeros(source.size) for name in OUTPUT_NAMES_V4},
        targets={name: np.zeros(source.size) for name in OUTPUT_NAMES_V4},
        domains=source.domains,
        metadata=metadata,
    )

    _validate_counterfactual_cache_dataset(dataset, identity, None)
    mismatched = replace(
        dataset,
        metadata={**metadata, "local_mechanism_horizon_sec": 20},
    )
    with pytest.raises(ValueError, match="policy-consistent estimand mismatch"):
        _validate_counterfactual_cache_dataset(mismatched, identity, None)


def test_multihorizon_cache_requires_full_horizon_equivalence():
    horizons = (10, 20)
    full_target = rollout_prefix_target_name(20)
    prefix_names = tuple(rollout_prefix_target_name(value) for value in horizons)
    output_names = (*OUTPUT_NAMES_V4, *prefix_names)
    identity = {
        "version": MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION,
        "scenario": "scenario",
        "seed": 17,
        "control_interval_sec": 10,
        "counterfactual_horizon_intervals": 2,
        "rollout_prefix_horizons_sec": list(horizons),
        "collection_shard_index": 0,
        "collection_shard_count": 2,
    }
    source = _counterfactual_shard(0, "g0")
    metadata = {
        **dict(source.metadata),
        "counterfactual_cache_identity": identity,
        "counterfactual_estimand": POLICY_CONSISTENT_ESTIMAND_PROTOCOL_V4,
        "mechanism_output_names": list(output_names),
        "local_mechanism_horizon_sec": 10,
        "rollout_value_horizon_sec": 20,
        "rollout_prefix_horizons_sec": list(horizons),
        "mechanism_estimands": {
            "one_step_local_state": {
                "output_names": list(ONE_STEP_OUTPUT_NAMES_V4),
                "horizon_sec": 10,
                "analytic_prior_horizon_sec": 10,
            },
            "rollout_prefix_action_values": {
                "output_names": list(prefix_names),
                "horizons_sec": list(horizons),
                "aggregation": (
                    "arithmetic_mean_of_one_second_cost_samples_from_branch_start"
                ),
                "policy": "candidate_action_then_phase_pressure",
                "cost_mode": "system_vehicle_load",
            },
        },
    }
    targets = {name: np.zeros(source.size) for name in output_names}
    dataset = MechanismDataset(
        feature_names=source.feature_names,
        features=source.features,
        context_names=source.context_names,
        context=source.context,
        priors={name: np.zeros(source.size) for name in output_names},
        targets=targets,
        domains=source.domains,
        metadata=metadata,
    )

    _validate_counterfactual_cache_dataset(dataset, identity, None)
    broken = replace(
        dataset,
        targets={**targets, full_target: np.ones(source.size)},
    )
    with pytest.raises(ValueError, match="full-horizon prefix mismatch"):
        _validate_counterfactual_cache_dataset(broken, identity, None)


def test_counterfactual_cache_allows_a_valid_empty_trailing_shard():
    identity = {
        "scenario": "scenario",
        "seed": 17,
        "collection_shard_index": 1,
        "collection_shard_count": 2,
    }
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.zeros((0, 1)),
        context_names=("z",),
        context=np.zeros((0, 1)),
        priors={name: np.zeros(0) for name in OUTPUT_NAMES_V3},
        targets={name: np.zeros(0) for name in OUTPUT_NAMES_V3},
        domains=np.asarray([], dtype=str),
        metadata={
            "scenario": "scenario",
            "seed": 17,
            "counterfactual_cache_identity": identity,
            "state_snapshot_protocol": dict(STATE_SNAPSHOT_PROTOCOL_V3),
            "sumo_execution_protocol": dict(SUMO_EXECUTION_PROTOCOL),
            "strict_safety_monitoring": dict(STRICT_SAFETY_MONITORING_V3),
            "behavior_safety_audit": {
                "raw_collision_events": 0,
                "collision_event_steps": 0,
                "unique_collision_incidents": 0,
                "starting_teleports": 0,
                "ending_teleports": 0,
                "collision_samples": [],
                "no_teleport_passed": True,
            },
            "counterfactual_safety_audit": {
                "protocol": COUNTERFACTUAL_SAFETY_PROTOCOL_V3,
                "candidate_groups": 0,
                "retained_groups": 0,
                "censored_groups": 0,
                "candidate_branches": 0,
                "attempted_branches": 0,
                "retained_branches": 0,
                "censored_branches": 0,
                "unevaluated_censored_branches": 0,
                "observed_unsafe_branches": 0,
                "raw_collision_events_observed": 0,
                "collision_event_steps_observed": 0,
                "unique_collision_incidents_observed": 0,
                "censored_group_fraction": 0.0,
                "censored_group_ids": [],
                "collision_samples": [],
                "starting_teleports": 0,
                "ending_teleports": 0,
                "no_teleport_passed": True,
                "symmetric_group_censoring_passed": True,
            },
            "counterfactual_replay_audit": {
                "checks": 0,
                "absolute_tolerance": 1e-8,
                "max_abs_difference": 0.0,
                "not_applicable": True,
                "passed": True,
            },
            "counterfactual_branches": 0,
            "eligible_collection_opportunities": 1,
            "behavior_trace_sha256": "a" * 64,
            "behavior_interval_count": 12,
            "captured_snapshot_count": 0,
            "collection_shard_index": 1,
            "collection_shard_count": 2,
            "action_group_ids": [],
            "row_tls": [],
            "row_times": [],
            "candidate_states": [],
            "controllable_tls_ids": ["a"],
            "focal_selection_protocol": STATE_SNAPSHOT_PROTOCOL_V3["focal_selection"],
            "focal_selection_count_by_tls": {"a": 1},
            "focal_selection_min_count": 1,
            "focal_selection_max_count": 1,
        },
    )
    _validate_counterfactual_cache_dataset(dataset, identity, None)


def test_counterfactual_cache_allows_fully_censored_action_group_without_rows():
    identity = {
        "scenario": "scenario",
        "seed": 17,
        "collection_shard_index": 0,
        "collection_shard_count": 2,
    }
    source = _counterfactual_shard(0, "g0")
    metadata = {
        **dict(source.metadata),
        "counterfactual_cache_identity": identity,
        "counterfactual_branches": 0,
        "action_group_ids": [],
        "row_tls": [],
        "row_times": [],
        "candidate_states": [],
        "covered_tls_ids": [],
        "counterfactual_replay_audit": {
            "checks": 0,
            "absolute_tolerance": 1e-8,
            "max_abs_difference": 0.0,
            "not_applicable": True,
            "passed": True,
        },
        "counterfactual_safety_audit": {
            **dict(source.metadata["counterfactual_safety_audit"]),
            "candidate_groups": 1,
            "retained_groups": 0,
            "censored_groups": 1,
            "candidate_branches": 1,
            "attempted_branches": 1,
            "retained_branches": 0,
            "censored_branches": 1,
            "observed_unsafe_branches": 1,
            "raw_collision_events_observed": 1,
            "collision_event_steps_observed": 1,
            "unique_collision_incidents_observed": 1,
            "collision_incident_signatures": ["incident-1"],
            "censored_group_fraction": 1.0,
            "censored_group_ids": ["g0"],
        },
    }
    dataset = MechanismDataset(
        feature_names=source.feature_names,
        features=np.zeros((0, 1)),
        context_names=source.context_names,
        context=np.zeros((0, 1)),
        priors={name: np.zeros(0) for name in OUTPUT_NAMES_V3},
        targets={name: np.zeros(0) for name in OUTPUT_NAMES_V3},
        domains=np.asarray([], dtype=str),
        metadata=metadata,
    )

    _validate_counterfactual_cache_dataset(dataset, identity, None)


def test_counterfactual_cache_preserves_replay_check_before_full_censor():
    identity = {
        "scenario": "scenario",
        "seed": 17,
        "collection_shard_index": 0,
        "collection_shard_count": 2,
    }
    source = _counterfactual_shard(0, "g0")
    metadata = {
        **dict(source.metadata),
        "counterfactual_cache_identity": identity,
        "counterfactual_branches": 0,
        "action_group_ids": [],
        "row_tls": [],
        "row_times": [],
        "candidate_states": [],
        "covered_tls_ids": [],
        "counterfactual_replay_audit": {
            "checks": 1,
            "absolute_tolerance": 1e-8,
            "max_abs_difference": 0.0,
            "not_applicable": False,
            "passed": True,
        },
        "counterfactual_safety_audit": {
            **dict(source.metadata["counterfactual_safety_audit"]),
            "candidate_groups": 1,
            "retained_groups": 0,
            "censored_groups": 1,
            "candidate_branches": 1,
            "attempted_branches": 1,
            "retained_branches": 0,
            "censored_branches": 1,
            "observed_unsafe_branches": 1,
            "raw_collision_events_observed": 1,
            "collision_event_steps_observed": 1,
            "unique_collision_incidents_observed": 1,
            "collision_incident_signatures": ["incident-1"],
            "censored_group_fraction": 1.0,
            "censored_group_ids": ["g0"],
        },
    }
    dataset = MechanismDataset(
        feature_names=source.feature_names,
        features=np.zeros((0, 1)),
        context_names=source.context_names,
        context=np.zeros((0, 1)),
        priors={name: np.zeros(0) for name in OUTPUT_NAMES_V3},
        targets={name: np.zeros(0) for name in OUTPUT_NAMES_V3},
        domains=np.asarray([], dtype=str),
        metadata=metadata,
    )

    _validate_counterfactual_cache_dataset(dataset, identity, None)


def test_counterfactual_cache_rejects_missing_safety_protocol():
    identity = {
        "scenario": "scenario",
        "seed": 17,
        "collection_shard_index": 0,
        "collection_shard_count": 2,
    }
    source = _counterfactual_shard(0, "g0")
    metadata = dict(source.metadata)
    metadata.pop("strict_safety_monitoring")
    dataset = MechanismDataset(
        feature_names=source.feature_names,
        features=source.features,
        context_names=source.context_names,
        context=source.context,
        priors=source.priors,
        targets=source.targets,
        domains=source.domains,
        metadata={**metadata, "counterfactual_cache_identity": identity},
    )
    with pytest.raises(ValueError, match="safety monitoring protocol mismatch"):
        _validate_counterfactual_cache_dataset(dataset, identity, None)


def _grouped_dataset() -> MechanismDataset:
    groups = ["a", "a", "b", "b", "b", "c", "c"]
    rows = len(groups)
    return MechanismDataset(
        feature_names=("x",),
        features=np.arange(rows, dtype=float)[:, None],
        context_names=("z",),
        context=np.ones((rows, 1), dtype=float),
        priors={"y": np.zeros(rows)},
        targets={"y": np.arange(rows, dtype=float)},
        domains=np.asarray(["target"] * rows),
        metadata={
            "scenario": "target",
            "action_group_ids": groups,
            "row_times": list(range(rows)),
            "candidate_states": [f"s{row}" for row in range(rows)],
        },
    )


def test_target_adaptation_selects_complete_groups_deterministically():
    dataset = _grouped_dataset()
    first = _target_adaptation_subset_v3(dataset, group_budget=2, selection_seed=19)
    second = _target_adaptation_subset_v3(dataset, group_budget=2, selection_seed=19)
    selected = set(first.metadata["target_adaptation_selected_group_ids"])
    assert len(selected) == 2
    assert first.metadata["action_group_ids"] == second.metadata["action_group_ids"]
    assert set(first.metadata["action_group_ids"]) == selected
    assert first.metadata["row_times"] == first.features[:, 0].astype(int).tolist()
    for group in selected:
        assert first.metadata["action_group_ids"].count(group) == dataset.metadata[
            "action_group_ids"
        ].count(group)


def test_target_adaptation_zero_budget_and_budget_validation():
    dataset = _grouped_dataset()
    empty = _target_adaptation_subset_v3(dataset, group_budget=0, selection_seed=1)
    assert empty.size == 0
    assert empty.metadata["target_adaptation_groups_selected"] == 0
    all_rows = _target_adaptation_subset_v3(dataset, group_budget=99, selection_seed=1)
    assert all_rows.size == dataset.size
    assert all_rows.metadata["target_adaptation_groups_selected"] == 3
    with pytest.raises(ValueError, match="nonnegative"):
        _target_adaptation_subset_v3(dataset, group_budget=-1, selection_seed=1)


def test_target_adaptation_subset_recomputes_materialized_safety_audit():
    merged = _merge_collection_shards_v3(
        "scenario",
        17,
        [
            _counterfactual_shard(0, "scenario:seed2027:0:tls0"),
            _counterfactual_shard(1, "scenario:seed2027:1:tls1"),
        ],
    )

    subset = _target_adaptation_subset_v3(
        merged,
        group_budget=1,
        selection_seed=19,
    )
    audit = subset.metadata["counterfactual_safety_audit"]

    assert subset.size == 1
    assert audit["audit_scope"] == "materialized_retained_group_subset_v1"
    assert audit["candidate_groups"] == audit["retained_groups"] == 1
    assert audit["candidate_branches"] == audit["retained_branches"] == subset.size
    assert audit["censored_groups"] == audit["censored_branches"] == 0
    assert len(audit["parent_safety_audit_sha256"]) == 64
    assert audit["parent_safety_summary"]["retained_branches"] == merged.size
    assert merge_counterfactual_datasets_v3([subset]).size == subset.size


def test_target_adaptation_split_is_group_and_seed_disjoint():
    group_ids = [
        *(f"target:seed2027:{index}:tls" for index in range(4)),
        *(f"target:seed3037:{index}:tls" for index in range(4)),
        *(f"target:seed4047:{index}:tls" for index in range(4)),
    ]
    groups = [group for group in group_ids for _ in range(2)]
    rows = len(groups)
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.arange(rows, dtype=float)[:, None],
        context_names=("z",),
        context=np.ones((rows, 1)),
        priors={"y": np.zeros(rows)},
        targets={"y": np.arange(rows, dtype=float)},
        domains=np.asarray(["target"] * rows),
        metadata={
            "action_group_ids": groups,
            "row_tls": [group.rsplit(":", 1)[-1] for group in groups],
        },
    )
    split = _target_adaptation_split_v3(
        dataset,
        group_budget=10,
        selection_seed=7,
        calibration_seeds=(4047,),
    )
    adaptation = set(split["adaptation"].metadata["action_group_ids"])
    calibration = set(split["calibration"].metadata["action_group_ids"])
    selected = set(split["selected"].metadata["action_group_ids"])
    assert len(adaptation) == 8
    assert len(calibration) == 2
    assert adaptation.isdisjoint(calibration)
    assert selected == adaptation | calibration
    assert all(":seed4047:" not in group for group in adaptation)
    assert all(":seed4047:" in group for group in calibration)
    assert split["adaptation"].metadata["target_data_role"] == "model_adaptation"
    assert (
        split["calibration"].metadata["target_data_role"]
        == "policy_selection_calibration"
    )
    assert (
        split["selected"].metadata["target_data_role"]
        == "budget_union_audit_only"
    )


def test_target_adaptation_split_prioritizes_distinct_tls_coverage():
    group_ids = [
        f"target:seed{seed}:{tls_index}:tls_{tls_index}"
        for seed in (2027, 3037, 4047)
        for tls_index in range(8)
    ]
    groups = [group for group in group_ids for _ in range(2)]
    row_tls = [group.rsplit(":", 1)[-1] for group in groups]
    rows = len(groups)
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.arange(rows, dtype=float)[:, None],
        context_names=("z",),
        context=np.ones((rows, 1)),
        priors={"y": np.zeros(rows)},
        targets={"y": np.arange(rows, dtype=float)},
        domains=np.asarray(["target"] * rows),
        metadata={"action_group_ids": groups, "row_tls": row_tls},
    )

    split = _target_adaptation_split_v3(
        dataset,
        group_budget=8,
        selection_seed=7,
        calibration_seeds=(4047,),
        calibration_fraction=0.5,
    )

    assert len(set(split["adaptation"].metadata["row_tls"])) == 4
    assert len(set(split["calibration"].metadata["row_tls"])) == 4
    assert split["selected"].metadata["target_group_selection_protocol"] == (
        "coverage-first-per-tls-disjoint-seed-role-v1"
    )


def test_target_calibration_does_not_refit_the_deployment_model(monkeypatch):
    def dataset(domain: str, groups: list[str]) -> MechanismDataset:
        rows = len(groups)
        return MechanismDataset(
            feature_names=("x",),
            features=np.arange(rows, dtype=float)[:, None],
            context_names=("z",),
            context=np.ones((rows, 1), dtype=float),
            priors={"interval_cost": np.zeros(rows)},
            targets={"interval_cost": np.arange(rows, dtype=float)},
            domains=np.asarray([domain] * rows),
            metadata={
                "action_group_ids": list(groups),
                "target_adaptation_groups_selected": len(set(groups)),
                "target_adaptation_selected_group_ids": sorted(set(groups)),
                "target_role_groups": len(set(groups)),
            },
        )

    bank = {
        "target": dataset("target", ["all"]),
        "source_a": dataset("source_a", ["source_a"]),
        "source_b": dataset("source_b", ["source_b"]),
    }
    adaptation = dataset("target", ["adapt"])
    calibration = dataset("target", ["calibrate"])
    selected = dataset("target", ["adapt", "calibrate"])
    selected = MechanismDataset(
        feature_names=selected.feature_names,
        features=selected.features,
        context_names=selected.context_names,
        context=selected.context,
        priors=selected.priors,
        targets=selected.targets,
        domains=selected.domains,
        metadata={
            **dict(selected.metadata),
            "target_adaptation_groups_selected": 2,
            "target_adaptation_selected_group_ids": ["adapt", "calibrate"],
        },
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite._target_adaptation_split_v3",
        lambda *args, **kwargs: {
            "adaptation": adaptation,
            "calibration": calibration,
            "selected": selected,
        },
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.select_contextual_policy_from_domain_costs",
        lambda *args, **kwargs: ("prior", {"selected": "prior"}),
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.merge_counterfactual_datasets_v3",
        lambda parts: tuple(parts),
    )
    fit_calls = []

    def fake_fit(parts, **kwargs):
        fit_calls.append(parts)
        domains = sorted({str(value) for part in parts for value in part.domains})
        return SimpleNamespace(
            fit_tag=len(fit_calls),
            prior_spec="prior",
            target_support=None,
            guards={"causal_rigid_advantage": ContrastGuardConfig()},
                regularizers={
                    "causal_rigid_advantage": PriorRegularizationConfig()
                },
                family_models={"causal_rigid_advantage": object()},
                hierarchy_layers={
                    "guard": ("causal_core_fallback",),
                    "regularized": ("causal_core_fallback",),
                },
                diagnostics={
                    "source_domains": domains,
                    "cfcmt_hierarchy": {},
                    "guard_calibration": {
                    "causal_rigid_advantage": {"selected": None}
                },
            },
        )

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.fit_contrast_models_v3",
        fake_fit,
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.build_action_contrast_dataset",
        lambda data, **kwargs: data,
    )
    support = SimpleNamespace(diagnostics=lambda: {"groups": 1})
    monkeypatch.setattr(TargetActionSupport, "fit", lambda *args, **kwargs: support)
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite.calibrate_target_policy_selection_v3",
        lambda *args, **kwargs: {
            "eligible": True,
            "reason": "disjoint_target_group_calibration",
            "group_count": 1,
            "guards": {"causal_rigid_advantage": ContrastGuardConfig()},
            "regularizers": {
                "causal_rigid_advantage": PriorRegularizationConfig()
            },
            "diagnostics": {},
        },
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite._apply_local_graph_selector_v3",
        lambda models: {"selected": "causal_rigid_advantage"},
    )

    fitted = _fit_target_models(
        bank,
        "target",
        fit_config=MechanismFitConfig(),
        source_rule_costs={
            "source_a": {"prior": 1.0},
            "source_b": {"prior": 1.0},
        },
        source_rule_specs={"prior": object()},
        model_families=("causal_rigid_advantage",),
        target_group_budget=2,
        target_adaptation_selection_seed=7,
        target_calibration_seeds=(4047,),
        min_target_calibration_groups=1,
        target_calibration_fraction=0.5,
        target_guard_selection_mode="strict_worst_domain",
        source_selector_min_context_domains=2,
        scenario_city_groups={
            "target": "target_city",
            "source_a": "city_a",
            "source_b": "city_b",
        },
    )

    assert fitted.fit_tag == 1
    assert len(fit_calls) == 1
    target_parts = [
        part
        for part in fit_calls[0]
        if set(part.domains) == {"target_city"}
    ]
    assert len(target_parts) == 1
    assert target_parts[0].metadata["action_group_ids"] == ["adapt"]
    diagnostics = fitted.diagnostics["target_adaptation"]
    assert diagnostics["deployment_model_group_ids"] == ["adapt"]
    assert diagnostics["calibration_groups_refit_into_deployment_model"] is False


def test_v3_json_export_rejects_no_nonfinite_values():
    ready = _json_ready(
        {"nan": float("nan"), "inf": np.float64("inf"), "array": np.asarray([1.0, np.nan])}
    )
    assert ready == {"nan": None, "inf": None, "array": [1.0, None]}
    assert json.loads(json.dumps(ready, allow_nan=False)) == ready


def test_primary_reference_tracks_requested_method_family():
    assert _primary_reference_policy_v3(
        ("phase_pressure", "causal_ranker_contrast_regularized")
    ) == "causal_ranker_contrast_regularized"
    assert _primary_reference_policy_v3(("phase_pressure", "max_pressure")) is None
    assert _primary_reference_policy_v3(
        ("cfcmt_mechanism_contrast_regularized", "cfcmt_mechanism_contrast_guard"),
        declared="cfcmt_mechanism_contrast_guard",
    ) == "cfcmt_mechanism_contrast_guard"
    with pytest.raises(ValueError, match="absent from requested policies"):
        _primary_reference_policy_v3(
            ("cfcmt_mechanism_contrast_regularized",),
            declared="cfcmt_mechanism_contrast_guard",
        )


def test_residual_coordinator_matches_sparse_rollout_estimand():
    audit = ContrastGuardAudit()
    cooldowns = {"a": 0, "b": 0}
    proposals = {
        "a": ContrastProposal("a_prior", "a_model", True, True, 1.0),
        "b": ContrastProposal("b_prior", "b_model", True, True, 2.0),
    }
    first = _coordinate_contrast_proposals(
        proposals,
        cooldowns=cooldowns,
        cooldown_intervals=2,
        max_simultaneous_overrides=1,
        audit=audit,
    )
    assert first == {"a": "a_prior", "b": "b_model"}
    assert cooldowns == {"a": 0, "b": 2}

    second = _coordinate_contrast_proposals(
        proposals,
        cooldowns=cooldowns,
        cooldown_intervals=2,
        max_simultaneous_overrides=1,
        audit=audit,
    )
    assert second == {"a": "a_model", "b": "b_prior"}
    assert cooldowns == {"a": 2, "b": 1}
    assert audit.accepted_overrides == 2
    assert audit.rejected_coordination == 1
    assert audit.rejected_cooldown == 1


def test_contrast_proposal_uses_only_currently_feasible_action(monkeypatch):
    candidate = SimpleNamespace(state="only_feasible_phase")
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3._absolute_target_candidates",
        lambda *args, **kwargs: ((candidate,), object()),
    )

    class _ModelMustNotRun:
        def predict(self, dataset):
            raise AssertionError("single feasible action must not be scored")

    proposal = _contrast_proposal(
        model=_ModelMustNotRun(),
        family="family",
        models=SimpleNamespace(prediction_horizon_sec=60),
        state=object(),
        executor=object(),
        control_interval_sec=10,
        guarded=False,
        regularized=False,
    )

    assert proposal.prior_candidate is candidate
    assert proposal.selected_candidate is candidate
    assert proposal.learned_differs is False
    assert proposal.eligible is False


def test_hierarchical_proposal_preserves_core_and_prefers_stronger_refinement():
    core = ContrastProposal("prior", "core", True, True, 2.0)
    disabled_refinement = ContrastProposal(
        "prior", "refined", True, False, 0.0, rejection="disabled"
    )
    fallback = _select_hierarchical_proposal(disabled_refinement, core)
    assert fallback.selected_candidate == "core"
    assert fallback.selection_layer == "causal_core_fallback"

    refinement = ContrastProposal("prior", "refined", True, True, 3.0)
    selected = _select_hierarchical_proposal(refinement, core)
    assert selected.selected_candidate == "refined"
    assert selected.selection_layer == "mechanism_refinement"

    target_specialist = ContrastProposal("prior", "target", True, True, 4.0)
    target_selected = _select_hierarchical_proposal(
        refinement,
        core,
        target_specialist,
    )
    assert target_selected.selected_candidate == "target"
    assert target_selected.selection_layer == "target_specialist"


def test_hierarchical_proposal_respects_validated_layer_mask_and_conservative_ties():
    core = ContrastProposal("prior", "core", True, True, 2.0)
    refinement = ContrastProposal("prior", "refined", True, True, 2.0)
    target = ContrastProposal("prior", "target", True, True, 3.0)

    tied = _select_hierarchical_proposal(refinement, core)
    assert tied.selected_candidate == "core"
    assert tied.selection_layer == "causal_core_fallback"

    masked = _select_hierarchical_proposal(
        refinement,
        core,
        target,
        enabled_layers=("causal_core_fallback",),
    )
    assert masked.selected_candidate == "core"
    assert masked.selection_layer == "causal_core_fallback"

    prior = _select_hierarchical_proposal(
        refinement,
        core,
        target,
        enabled_layers=(),
    )
    assert prior.selected_candidate == "prior"
    assert prior.selection_layer == "pressure_prior_fallback"


def test_target_specialist_requires_a_trained_positive_weight_target_head():
    trained = SimpleNamespace(target_model=object(), target_weight=1.0)
    wrapped = SimpleNamespace(base_model=trained)
    assert has_fitted_target_specialist_v3(wrapped)
    assert not has_fitted_target_specialist_v3(
        SimpleNamespace(base_model=SimpleNamespace(target_model=None, target_weight=0.0))
    )
    assert CFCMT_TARGET_SPECIALIST_FAMILY_V3 == "causal_target_only"


def test_target_capacity_selector_replaces_source_core_after_target_fit():
    diagnostics = {
        CFCMT_TARGET_SPECIALIST_FAMILY_V3: {
            "guard": {
                "selected": {
                    "mean_actual_delta": -0.20,
                    "worst_domain_delta": 0.0,
                    "domain_mean_delta": {"g0": -0.20, "g1": -0.20},
                }
            },
            "regularizer": {"selected": None},
        },
        CFCMT_CORE_FAMILY_V3: {
            "guard": {
                "selected": {
                    "mean_actual_delta": -1.0,
                    "worst_domain_delta": 0.0,
                    "domain_mean_delta": {"g0": -1.0, "g1": -1.0},
                }
            },
            "regularizer": {"selected": None},
        },
        "cfcmt_mechanism": {
            "guard": {
                "selected": {
                    "mean_actual_delta": -0.205,
                    "worst_domain_delta": 0.0,
                    "domain_mean_delta": {"g0": -0.205, "g1": -0.205},
                }
            },
            "regularizer": {"selected": None},
        },
    }
    guards = {
        family: ContrastGuardConfig(enabled=True)
        for family in diagnostics
    }
    selected = select_target_capacity_layers_v3(
        target_specialist_fitted=True,
        guards=guards,
        regularizers={
            family: PriorRegularizationConfig() for family in diagnostics
        },
        calibration_diagnostics=diagnostics,
    )
    assert selected["anchor_family"] == CFCMT_TARGET_SPECIALIST_FAMILY_V3
    assert selected["layers"]["guard"] == ("target_specialist",)
    assert selected["layers"]["regularized"] == ()


def test_target_capacity_selector_requires_material_mechanism_gain():
    def family_diagnostics(mean_delta):
        return {
            "guard": {
                "selected": {
                    "mean_actual_delta": mean_delta,
                    "worst_domain_delta": 0.0,
                    "domain_mean_delta": {"g0": mean_delta, "g1": mean_delta},
                }
            },
            "regularizer": {"selected": None},
        }

    diagnostics = {
        CFCMT_TARGET_SPECIALIST_FAMILY_V3: family_diagnostics(-0.20),
        "cfcmt_mechanism": family_diagnostics(-0.25),
    }
    selected = select_target_capacity_layers_v3(
        target_specialist_fitted=True,
        guards={
            family: ContrastGuardConfig(enabled=True)
            for family in diagnostics
        },
        regularizers={
            family: PriorRegularizationConfig() for family in diagnostics
        },
        calibration_diagnostics=diagnostics,
    )
    assert selected["layers"]["guard"] == ("mechanism_refinement",)
    assert (
        selected["modes"]["guard"]["reason"]
        == "mechanism_materially_improves_over_calibrated_anchor"
    )


def test_target_capacity_selector_does_not_deploy_mechanism_without_anchor():
    diagnostics = {
        "cfcmt_mechanism": {
            "guard": {
                "selected": {
                    "mean_actual_delta": -1.0,
                    "worst_domain_delta": 0.0,
                    "domain_mean_delta": {"g0": -1.0},
                }
            },
            "regularizer": {"selected": None},
        }
    }
    selected = select_target_capacity_layers_v3(
        target_specialist_fitted=True,
        guards={"cfcmt_mechanism": ContrastGuardConfig(enabled=True)},
        regularizers={"cfcmt_mechanism": PriorRegularizationConfig()},
        calibration_diagnostics=diagnostics,
    )
    assert selected["layers"]["guard"] == ()
    assert selected["modes"]["guard"]["reason"] == "calibrated_anchor_unavailable"


def test_hierarchical_proposal_returns_pressure_prior_when_both_layers_reject():
    refinement = ContrastProposal(
        "prior", "refined", True, False, 0.0, rejection="confidence"
    )
    core = ContrastProposal(
        "prior", "core", True, False, 0.0, rejection="local_support"
    )
    selected = _select_hierarchical_proposal(refinement, core)
    assert selected.selected_candidate == "prior"
    assert selected.selection_layer == "pressure_prior_fallback"
    assert selected.rejection == "local_support"


def test_relative_rule_gap_and_terminal_guard_reject_short_horizon_clearance_loss():
    contrast = MechanismDataset(
        feature_names=("green_q",),
        features=np.asarray([[10.0], [5.0], [10.0], [9.0]]),
        context_names=("z",),
        context=np.ones((4, 1)),
        priors={"interval_cost": np.zeros(4)},
        targets={"interval_cost": np.asarray([0.0, -1.0, 0.0, -1.0])},
        domains=np.asarray(["d"] * 4),
        metadata={
            "action_group_ids": ["g1", "g1", "g2", "g2"],
            "is_reference": [True, False, True, False],
            "reference_rows": [0, 0, 2, 2],
            "reference_policy": "phase_pressure",
        },
    )
    assert np.allclose(_relative_contrast_rule_gap(contrast), [0.0, 0.5, 0.0, 0.1])
    record = {
        "score": np.asarray([0.0, -1.0, 0.0, -1.0]),
        "uncertainty": np.zeros(4),
        "trust": np.ones(4),
        "reference_rows": np.asarray([0, 0, 2, 2]),
        "actual": np.asarray([0.0, -1.0, 0.0, -1.0]),
        "actual_terminal": np.asarray([0.0, 3.0, 0.0, -1.0]),
        "groups": np.asarray(["g1", "g1", "g2", "g2"]),
        "rule_gap": np.asarray([0.0, 5.0, 0.0, 1.0]),
        "relative_rule_gap": np.asarray([0.0, 0.5, 0.0, 0.1]),
    }
    config, diagnostics = calibrate_contrast_guard_v3(
        contrast,
        family="unused",
        cfcmt_config=SimpleNamespace(),
        thresholds=(0.0,),
        max_relative_rule_gaps=(0.1, 0.5),
        noninferiority_tolerance=0.0,
        terminal_noninferiority_tolerance=0.0,
        records_bundle=({"d": record}, 0.0),
    )
    assert config.enabled
    assert config.max_relative_rule_gap == 0.1
    assert diagnostics["selected"]["mean_terminal_delta"] < 0.0


def test_prior_regularization_uses_dimensionless_rule_gap():
    record = {
        "score": np.asarray([0.0, -1.0]),
        "uncertainty": np.zeros(2),
        "trust": np.ones(2),
        "reference_rows": np.asarray([0, 0]),
        "actual": np.asarray([0.0, -0.5]),
        "groups": np.asarray(["g", "g"]),
        "rule_gap": np.asarray([0.0, 0.2]),
        "rule_gap_unit": "reference-normalized-pressure-gap",
    }
    config, diagnostics = calibrate_prior_regularization_v3(
        SimpleNamespace(),
        family="unused",
        cfcmt_config=SimpleNamespace(),
        blend_weights=(1.0,),
        risk_multipliers=(0.0,),
        trust_quantiles=(0.0,),
        records_bundle=({"a": record, "b": record}, 0.0),
    )

    assert config.enabled
    assert diagnostics["objective_units"] == {
        "rule_gap": "reference-normalized-pressure-gap",
        "model_score": "within-domain-median-action-range",
        "uncertainty": "within-domain-median-action-range",
    }


def test_intervention_graph_links_adjacent_tls_by_shared_lane():
    infos = {
        "a": SimpleNamespace(
            incoming_lanes=("a_in",),
            controlled_links=((('a_in', 'a_to_b', ''),),),
        ),
        "b": SimpleNamespace(
            incoming_lanes=("a_to_b", "b_in"),
            controlled_links=((('a_to_b', 'b_out', ''),),),
        ),
        "c": SimpleNamespace(
            incoming_lanes=("c_in",),
            controlled_links=((('c_in', 'c_out', ''),),),
        ),
    }
    graph = build_tls_intervention_graph(infos)
    assert graph.conflicts == {"a": ("b",), "b": ("a",), "c": ()}
    selected, rejected = greedy_independent_interventions(
        {"a": 3.0, "b": 2.0, "c": 1.0},
        graph.conflicts,
    )
    assert selected == ("a", "c")
    assert rejected == ("b",)


def test_routing_adjacency_extends_intervention_conflicts():
    graph = build_tls_intervention_graph(
        {
            "a": SimpleNamespace(incoming_lanes=("a",), controlled_links=()),
            "b": SimpleNamespace(incoming_lanes=("b",), controlled_links=()),
            "c": SimpleNamespace(incoming_lanes=("c",), controlled_links=()),
        }
    )
    augmented = add_routing_conflicts(graph, {"a": ("b",), "b": ("c",)})
    assert augmented.conflicts == {
        "a": ("b",),
        "b": ("a", "c"),
        "c": ("b",),
    }


def test_residual_coordinator_allows_spatially_separated_overrides():
    audit = ContrastGuardAudit()
    proposals = {
        key: ContrastProposal(f"{key}_prior", f"{key}_model", True, True, priority)
        for key, priority in (("a", 3.0), ("b", 2.0), ("c", 1.0))
    }
    actions = _coordinate_contrast_proposals(
        proposals,
        cooldowns={"a": 0, "b": 0, "c": 0},
        cooldown_intervals=2,
        max_simultaneous_overrides=None,
        audit=audit,
        conflicts={"a": ("b",), "b": ("a",), "c": ()},
    )
    assert actions == {"a": "a_model", "b": "b_prior", "c": "c_model"}
    assert audit.accepted_overrides == 2
    assert audit.rejected_coordination == 1


def test_residual_coordinator_direct_mode_accepts_every_eligible_override():
    audit = ContrastGuardAudit()
    proposals = {
        key: ContrastProposal(f"{key}_prior", f"{key}_model", True, True, priority)
        for key, priority in (("a", 3.0), ("b", 2.0), ("c", 1.0))
    }
    actions = _coordinate_contrast_proposals(
        proposals,
        cooldowns={"a": 0, "b": 0, "c": 0},
        cooldown_intervals=0,
        max_simultaneous_overrides=None,
        audit=audit,
        conflicts={"a": (), "b": (), "c": ()},
    )
    assert actions == {"a": "a_model", "b": "b_model", "c": "c_model"}
    assert audit.proposed_overrides == 3
    assert audit.accepted_overrides == 3
    assert audit.rejected_cooldown == 0
    assert audit.rejected_coordination == 0


def test_residual_execution_trust_region_is_disabled_by_default():
    proposals = {
        "a": ContrastProposal("prior", "model", True, True, 0.25),
    }
    states = {
        "a": SimpleNamespace(
            q_by_lane={"lane": 0.0},
            veh_by_lane={"lane": 0.0},
            speed_by_lane={"lane": 20.0},
        )
    }
    assert _apply_residual_execution_trust_region(
        proposals, states, ResidualExecutionTrustRegionConfig()
    ) == proposals


@pytest.mark.parametrize(
    ("config", "expected_rejection"),
    [
        (
            ResidualExecutionTrustRegionConfig(enabled=True, min_priority=0.5),
            "minimum_priority",
        ),
        (
            ResidualExecutionTrustRegionConfig(enabled=True, min_total_queue=5.0),
            "minimum_queue",
        ),
        (
            ResidualExecutionTrustRegionConfig(
                enabled=True, min_total_vehicles=2.0
            ),
            "minimum_vehicles",
        ),
        (
            ResidualExecutionTrustRegionConfig(enabled=True, max_mean_speed=5.0),
            "maximum_speed",
        ),
    ],
)
def test_residual_execution_trust_region_rejects_observable_state_risk(
    config, expected_rejection
):
    proposal = ContrastProposal("prior", "model", True, True, 0.25)
    states = {
        "a": SimpleNamespace(
            q_by_lane={"lane": 1.0},
            veh_by_lane={"lane": 1.0},
            speed_by_lane={"lane": 10.0},
        )
    }
    filtered = _apply_residual_execution_trust_region(
        {"a": proposal}, states, config
    )["a"]
    assert filtered.selected_candidate == "prior"
    assert not filtered.eligible
    assert filtered.rejection == expected_rejection


def test_residual_execution_trust_region_preserves_eligible_congested_action():
    proposal = ContrastProposal("prior", "model", True, True, 1.25)
    config = ResidualExecutionTrustRegionConfig(
        enabled=True,
        min_priority=0.5,
        min_total_queue=5.0,
        min_total_vehicles=2.0,
        max_mean_speed=5.0,
        max_simultaneous_overrides=1,
    )
    states = {
        "a": SimpleNamespace(
            q_by_lane={"left": 4.0, "right": 3.0},
            veh_by_lane={"left": 3.0, "right": 2.0},
            speed_by_lane={"left": 2.0, "right": 4.0},
        )
    }
    assert _apply_residual_execution_trust_region(
        {"a": proposal}, states, config
    )["a"] == proposal


def test_residual_execution_trust_region_enforces_network_global_cooldown():
    proposal = ContrastProposal("prior", "model", True, True, 1.25)
    config = ResidualExecutionTrustRegionConfig(
        enabled=True,
        max_simultaneous_overrides=1,
        global_cooldown_intervals=6,
    )
    states = {
        "a": SimpleNamespace(
            q_by_lane={"lane": 10.0},
            veh_by_lane={"lane": 5.0},
            speed_by_lane={"lane": 2.0},
        )
    }
    filtered = _apply_residual_execution_trust_region(
        {"a": proposal},
        states,
        config,
        global_cooldown_active=True,
    )
    assert not filtered["a"].eligible
    assert filtered["a"].rejection == "global_cooldown"

    audit = ContrastGuardAudit()
    actions = _coordinate_contrast_proposals(
        filtered,
        cooldowns={"a": 0},
        cooldown_intervals=0,
        max_simultaneous_overrides=1,
        audit=audit,
    )
    assert actions == {"a": "prior"}
    assert audit.rejected_global_cooldown == 1


def test_action_aware_global_cooldown_allows_stay_but_rejects_switch():
    prior = SimpleNamespace(state="prior")
    current = SimpleNamespace(state="current")
    alternate = SimpleNamespace(state="alternate")
    config = ResidualExecutionTrustRegionConfig(
        enabled=True,
        max_simultaneous_overrides=1,
        global_cooldown_intervals=6,
        allow_stay_during_global_cooldown=True,
    )
    states = {
        "stay": SimpleNamespace(
            q_by_lane={"lane": 10.0},
            veh_by_lane={"lane": 5.0},
            speed_by_lane={"lane": 2.0},
        ),
        "switch": SimpleNamespace(
            q_by_lane={"lane": 10.0},
            veh_by_lane={"lane": 5.0},
            speed_by_lane={"lane": 2.0},
        ),
    }
    filtered = _apply_residual_execution_trust_region(
        {
            "stay": ContrastProposal(prior, current, True, True, 1.0),
            "switch": ContrastProposal(prior, alternate, True, True, 1.0),
        },
        states,
        config,
        global_cooldown_active=True,
        current_green_states={"stay": "current", "switch": "current"},
    )
    assert filtered["stay"].eligible
    assert filtered["stay"].selected_candidate is current
    assert not filtered["switch"].eligible
    assert filtered["switch"].rejection == "global_cooldown"


def test_bounded_stay_aware_global_cooldown_consumes_declared_budget():
    prior = SimpleNamespace(state="prior")
    current = SimpleNamespace(state="current")
    proposal = ContrastProposal(prior, current, True, True, 1.0)
    config = ResidualExecutionTrustRegionConfig(
        enabled=True,
        max_simultaneous_overrides=1,
        global_cooldown_intervals=6,
        allow_stay_during_global_cooldown=True,
        max_stay_overrides_per_global_cooldown=1,
    )
    states = {
        "stay": SimpleNamespace(
            q_by_lane={"lane": 10.0},
            veh_by_lane={"lane": 5.0},
            speed_by_lane={"lane": 2.0},
        )
    }

    admitted = _apply_residual_execution_trust_region(
        {"stay": proposal},
        states,
        config,
        global_cooldown_active=True,
        current_green_states={"stay": "current"},
        stay_bypass_remaining=1,
    )
    exhausted = _apply_residual_execution_trust_region(
        {"stay": proposal},
        states,
        config,
        global_cooldown_active=True,
        current_green_states={"stay": "current"},
        stay_bypass_remaining=0,
    )

    assert admitted["stay"] == proposal
    assert not exhausted["stay"].eligible
    assert exhausted["stay"].rejection == "global_stay_budget"
    audit = ContrastGuardAudit()
    actions = _coordinate_contrast_proposals(
        exhausted,
        cooldowns={"stay": 0},
        cooldown_intervals=0,
        max_simultaneous_overrides=0,
        audit=audit,
        current_green_states={"stay": "current"},
    )
    assert actions == {"stay": prior}
    assert audit.rejected_global_stay_budget == 1


def test_residual_coordinator_audits_stay_and_switch_overrides():
    prior = SimpleNamespace(state="prior")
    current = SimpleNamespace(state="current")
    alternate = SimpleNamespace(state="alternate")
    audit = ContrastGuardAudit()
    actions = _coordinate_contrast_proposals(
        {
            "stay": ContrastProposal(prior, current, True, True, 2.0),
            "switch": ContrastProposal(prior, alternate, True, True, 1.0),
        },
        cooldowns={"stay": 0, "switch": 0},
        cooldown_intervals=0,
        max_simultaneous_overrides=None,
        audit=audit,
        current_green_states={"stay": "current", "switch": "current"},
    )
    assert actions == {"stay": current, "switch": alternate}
    assert audit.proposed_stay_overrides == 1
    assert audit.proposed_switch_overrides == 1
    assert audit.accepted_stay_overrides == 1
    assert audit.accepted_switch_overrides == 1


def test_residual_execution_trust_region_rejects_negative_global_cooldown():
    with pytest.raises(ValueError, match="invalid residual execution"):
        ResidualExecutionTrustRegionConfig(global_cooldown_intervals=-1)


def test_residual_execution_trust_region_rejects_negative_stay_budget():
    with pytest.raises(ValueError, match="invalid residual execution"):
        ResidualExecutionTrustRegionConfig(
            max_stay_overrides_per_global_cooldown=-1
        )


def test_residual_coordinator_audits_target_specialist_selection():
    audit = ContrastGuardAudit()
    proposal = ContrastProposal(
        "prior",
        "target",
        True,
        True,
        1.0,
        selection_layer="target_specialist",
    )
    actions = _coordinate_contrast_proposals(
        {"a": proposal},
        cooldowns={"a": 0},
        cooldown_intervals=0,
        max_simultaneous_overrides=None,
        audit=audit,
    )
    assert actions == {"a": "target"}
    assert audit.hierarchical_decisions == 1
    assert audit.selected_target_specialists == 1


def test_local_graph_selector_requires_material_source_stability():
    models = SimpleNamespace(
        family_models={
            CFCMT_CORE_FAMILY_V3: object(),
            "causal_balanced_advantage": object(),
        },
        guards={
            CFCMT_CORE_FAMILY_V3: ContrastGuardConfig(enabled=True),
            "causal_balanced_advantage": ContrastGuardConfig(enabled=True),
        },
        regularizers={
            CFCMT_CORE_FAMILY_V3: PriorRegularizationConfig(enabled=True),
            "causal_balanced_advantage": PriorRegularizationConfig(enabled=True),
        },
        diagnostics={
            "guard_calibration": {
                "causal_balanced_advantage": {
                    "selected": {
                        "mean_actual_delta": -0.0004,
                        "worst_domain_delta": 0.0,
                        "domain_mean_delta": {"a": -0.0004, "b": 0.0},
                    }
                }
            }
        },
    )
    selected = _apply_local_graph_selector_v3(models)
    assert selected["selected"] == CFCMT_CORE_FAMILY_V3
    assert not models.guards["causal_balanced_advantage"].enabled
    assert not models.regularizers["causal_balanced_advantage"].enabled


def test_dense_balanced_advantage_is_a_normalized_controlled_ablation():
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
        MODEL_FAMILIES_V3,
        NORMALIZED_ACTION_UNIT_FAMILIES_V3,
        DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
        BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
        GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10,
        ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11,
        SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12,
    )

    assert "dense_balanced_advantage" in MODEL_FAMILIES_V3
    assert "dense_balanced_advantage" in NORMALIZED_ACTION_UNIT_FAMILIES_V3
    assert GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10 in MODEL_FAMILIES_V3
    assert (
        GROUP_NORMALIZED_RIGID_ADVANTAGE_FAMILY_V10
        in NORMALIZED_ACTION_UNIT_FAMILIES_V3
    )
    assert ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11 in MODEL_FAMILIES_V3
    assert (
        ANTISYMMETRIC_PAIRWISE_ADVANTAGE_FAMILY_V11
        in NORMALIZED_ACTION_UNIT_FAMILIES_V3
    )
    assert SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12 in MODEL_FAMILIES_V3
    assert (
        SOURCE_GATED_PAIRWISE_ADVANTAGE_FAMILY_V12
        in NORMALIZED_ACTION_UNIT_FAMILIES_V3
    )


def test_cfcmt_fused_is_a_normalized_model_family():
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
        MODEL_FAMILIES_V3,
        NORMALIZED_ACTION_UNIT_FAMILIES_V3,
        ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4,
        ONE_STEP_PHYSICAL_FULL_FAMILY_V4,
        ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5,
        ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5,
        ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6,
        PHYSICAL_COMPONENT_FAMILIES_V3,
        ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
        DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
        BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
    )

    assert "cfcmt_fused" in MODEL_FAMILIES_V3
    assert "cfcmt_fused" in NORMALIZED_ACTION_UNIT_FAMILIES_V3
    assert "cfcmt_fused_rigid" in MODEL_FAMILIES_V3
    assert "cfcmt_fused_rigid" in NORMALIZED_ACTION_UNIT_FAMILIES_V3
    for family in (
        "cfcmt_physical_mechanism",
        "cfcmt_physical_fused",
        "cfcmt_physical_latent_mechanism",
        "cfcmt_physical_latent_fused",
        *PHYSICAL_COMPONENT_FAMILIES_V3,
        ONE_STEP_PHYSICAL_FULL_FAMILY_V4,
        *ONE_STEP_PHYSICAL_COMPONENT_FAMILIES_V4,
        ONE_STEP_RIGID_RESIDUAL_FULL_FAMILY_V5,
        *ONE_STEP_RIGID_RESIDUAL_COMPONENT_FAMILIES_V5,
        *ONE_STEP_SELECTIVE_RESIDUAL_FAMILIES_V6,
        ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
    ):
        assert family in MODEL_FAMILIES_V3
        assert family in NORMALIZED_ACTION_UNIT_FAMILIES_V3


@pytest.mark.parametrize(
    ("family", "expected_mechanism"),
    (
        ("cfcmt_physical_queue_only", ("queue_propagation",)),
        ("cfcmt_physical_red_only", ("red_accumulation",)),
        ("cfcmt_physical_spillback_only", ("spillback",)),
        ("cfcmt_physical_served_only", "served_movement"),
        ("cfcmt_physical_mobility_only", "mobility"),
        (
            "cfcmt_physical_mobility_queue",
            ("queue_propagation", "mobility"),
        ),
        ("cfcmt_physical_mobility_red", ("red_accumulation", "mobility")),
        (
            "cfcmt_physical_mobility_spillback",
            ("spillback", "mobility"),
        ),
        (
            "cfcmt_physical_mobility_served",
            ("served_movement", "mobility"),
        ),
    ),
)
def test_physical_single_mechanism_family_dispatches_exact_definition(
    monkeypatch, family, expected_mechanism
):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(self, definitions, *, target_domain, latent):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["target_domain"] = target_domain
            captured["latent"] = latent

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module, "PhysicalResidualCausalMechanismAdvantageModel", _Model
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        family,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    expected_names = (
        expected_mechanism
        if isinstance(expected_mechanism, tuple)
        else (expected_mechanism,)
    )
    assert captured == {
        "names": expected_names,
        "target_domain": "held_out_city",
        "latent": False,
        "size": 7,
    }


@pytest.mark.parametrize(
    ("family", "expected_names", "expected_targets"),
    (
        (
            "cfcmt_one_step_physical_queue_only",
            ("queue_propagation",),
            ("one_step_total_queue",),
        ),
        (
            "cfcmt_one_step_physical_mobility_only",
            ("mobility",),
            ("one_step_mean_speed",),
        ),
        (
            "cfcmt_one_step_physical_mechanism",
            (
                "queue_propagation",
                "served_movement",
                "red_accumulation",
                "spillback",
                "mobility",
                "terminal_clearance",
                "control_cost",
            ),
            (
                "one_step_total_queue",
                "one_step_green_queue",
                "one_step_red_queue",
                "one_step_downstream_occupancy",
                "one_step_mean_speed",
                "terminal_system_load",
                "interval_cost",
            ),
        ),
    ),
)
def test_one_step_physical_family_dispatches_policy_consistent_targets(
    monkeypatch,
    family,
    expected_names,
    expected_targets,
):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(self, definitions, *, target_domain, latent):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["targets"] = tuple(
                definition.target_name for definition in definitions
            )
            captured["target_domain"] = target_domain
            captured["latent"] = latent

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module,
        "PhysicalResidualCausalMechanismAdvantageModel",
        _Model,
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        family,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    assert captured == {
        "names": expected_names,
        "targets": expected_targets,
        "target_domain": "held_out_city",
        "latent": False,
        "size": 7,
    }


@pytest.mark.parametrize(
    ("family", "expected_names"),
    (
        (
            "cfcmt_one_step_rigid_residual_mobility_only",
            ("mobility",),
        ),
        (
            "cfcmt_one_step_rigid_residual_mobility_queue",
            ("queue_propagation", "mobility"),
        ),
        (
            "cfcmt_one_step_rigid_residual_mechanism",
            (
                "queue_propagation",
                "served_movement",
                "red_accumulation",
                "spillback",
                "mobility",
                "terminal_clearance",
                "control_cost",
            ),
        ),
    ),
)
def test_one_step_rigid_residual_family_dispatches_anchored_stack(
    monkeypatch,
    family,
    expected_names,
):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(
            self,
            definitions,
            *,
            target_domain,
            config,
            latent,
        ):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["target_domain"] = target_domain
            captured["source_stack_protocol"] = config.source_stack_protocol
            captured["latent"] = latent

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module,
        "PhysicalResidualCausalMechanismAdvantageModel",
        _Model,
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        family,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    assert captured == {
        "names": expected_names,
        "target_domain": "held_out_city",
        "source_stack_protocol": (
            "rigid_anchored_mechanism_residual_v1"
        ),
        "latent": False,
        "size": 7,
    }


@pytest.mark.parametrize(
    ("family", "expected_names", "expected_quantile"),
    (
        (
            "cfcmt_one_step_selective_residual_mobility_q50",
            ("mobility",),
            0.50,
        ),
        (
            "cfcmt_one_step_selective_residual_mobility_queue_q75",
            ("queue_propagation", "mobility"),
            0.75,
        ),
        (
            "cfcmt_one_step_selective_residual_mobility_queue_q90",
            ("queue_propagation", "mobility"),
            0.90,
        ),
    ),
)
def test_one_step_selective_residual_family_dispatches_source_only_lcb_gate(
    monkeypatch,
    family,
    expected_names,
    expected_quantile,
):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(self, definitions, *, target_domain, config, latent):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["target_domain"] = target_domain
            captured["source_stack_protocol"] = config.source_stack_protocol
            captured["source_expert_gate_protocol"] = (
                config.source_expert_gate_protocol
            )
            captured["error_quantile"] = config.source_expert_gate_error_quantile
            captured["latent"] = latent

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module,
        "PhysicalResidualCausalMechanismAdvantageModel",
        _Model,
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        family,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    assert captured == {
        "names": expected_names,
        "target_domain": "held_out_city",
        "source_stack_protocol": "rigid_anchored_mechanism_residual_v1",
        "source_expert_gate_protocol": "source_city_oof_ridge_lcb_v1",
        "error_quantile": expected_quantile,
        "latent": False,
        "size": 7,
    }


def test_rollout_value_residual_dispatches_terminal_policy_value_only(monkeypatch):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(self, definitions, *, target_domain, config):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["targets"] = tuple(
                definition.target_name for definition in definitions
            )
            captured["target_domain"] = target_domain
            captured["source_stack_protocol"] = config.source_stack_protocol

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module,
        "RolloutValueResidualCausalMechanismAdvantageModel",
        _Model,
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        module.ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V7,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    assert captured == {
        "names": (
            "queue_propagation",
            "served_movement",
            "red_accumulation",
            "spillback",
            "mobility",
            "terminal_clearance",
            "control_cost",
        ),
        "targets": (
            "next_total_queue",
            "next_green_queue",
            "next_red_queue",
            "next_downstream_occupancy",
            "next_mean_speed",
            "terminal_system_load",
            "interval_cost",
        ),
        "target_domain": "held_out_city",
        "source_stack_protocol": "rigid_anchored_mechanism_residual_v1",
        "size": 7,
    }


def test_direct_rollout_value_dispatches_zero_prior_invariant_terminal(monkeypatch):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(self, definitions, *, target_domain, config):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["target_domain"] = target_domain
            captured["source_stack_protocol"] = config.source_stack_protocol

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module,
        "DirectRolloutValueCausalMechanismAdvantageModel",
        _Model,
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        module.DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V8,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    assert captured == {
        "names": (
            "queue_propagation",
            "served_movement",
            "red_accumulation",
            "spillback",
            "mobility",
            "terminal_clearance",
            "control_cost",
        ),
        "target_domain": "held_out_city",
        "source_stack_protocol": "rigid_anchored_mechanism_residual_v1",
        "size": 7,
    }


def test_boosted_direct_rollout_value_dispatches_parent_restricted_backend(
    monkeypatch,
):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as module

    captured = {}

    class _Model:
        def __init__(self, definitions, *, target_domain, config):
            captured["names"] = tuple(definition.name for definition in definitions)
            captured["target_domain"] = target_domain
            captured["source_stack_protocol"] = config.source_stack_protocol

        def fit(self, dataset):
            captured["size"] = dataset.size
            return {"fit": "ok"}

    monkeypatch.setattr(
        module,
        "BoostedDirectRolloutValueCausalMechanismAdvantageModel",
        _Model,
    )
    model, diagnostics = module._fit_family_v3(
        _grouped_dataset(),
        module.BOOSTED_DIRECT_ROLLOUT_VALUE_RIGID_RESIDUAL_FAMILY_V9,
        MechanismFitConfig(),
        target_domain="held_out_city",
    )

    assert isinstance(model, _Model)
    assert diagnostics == {"fit": "ok"}
    assert captured == {
        "names": (
            "queue_propagation",
            "served_movement",
            "red_accumulation",
            "spillback",
            "mobility",
            "terminal_clearance",
            "control_cost",
        ),
        "target_domain": "held_out_city",
        "source_stack_protocol": "rigid_anchored_mechanism_residual_v1",
        "size": 7,
    }
