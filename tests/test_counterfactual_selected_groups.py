from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as collector
from cf_h2o.tests.test_traffic_signal_resco_cfcmt_v3_suite import _safety_api
from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor


@pytest.fixture
def collect(monkeypatch):
    candidates = tuple(SimpleNamespace(state=state) for state in ("Gr", "rG"))
    infos = {
        tls: SimpleNamespace(candidates=candidates, incoming_lanes=(), controlled_links=())
        for tls in ("a", "b", "c")
    }
    monkeypatch.setattr(collector, "_start_sumo", lambda *args: None)
    monkeypatch.setattr(collector, "_tls_phase_infos", lambda api: infos)
    monkeypatch.setattr(
        collector, "_scenario_context_v3",
        lambda **kwargs: np.zeros(len(collector.CONTEXT_NAMES_V3)),
    )
    monkeypatch.setattr(
        collector, "_read_graph_states_v3",
        lambda api, *args: {
            tls: SimpleNamespace(info=info, sim_time=api.simulation.getTime())
            for tls, info in infos.items()
        },
    )
    monkeypatch.setattr(
        collector, "build_safe_phase_executors",
        lambda api, infos: {
            tls: SafePhaseExecutor(
                sumo_api=api, tls_id=tls, initial_state="Gr",
                timings=[PhaseTiming(item.state, 0, 1, 1) for item in candidates],
            )
            for tls in infos
        },
    )
    monkeypatch.setattr(
        collector, "_rule_candidate",
        lambda state, *args: candidates[int(state.sim_time // 10) % 2],
    )

    def features(state, candidate, executor, **kwargs):
        result = np.zeros(len(collector.FEATURE_NAMES_V3))
        result[0] = state.sim_time + int(candidate.state == "rG")
        return result

    monkeypatch.setattr(collector, "candidate_features_v3", features)

    def reload(api, *args, begin_time):
        api.branch_phase = True
        api.simulation._time = begin_time

    monkeypatch.setattr(collector, "_reload_sumo_state", reload)

    def branch(**kwargs):
        value = float(kwargs["candidate"].state == "rG")
        aggregate = dict.fromkeys(
            ("total_q", "green_q", "red_q", "green_down_occ", "mean_speed"), value
        )
        return [value] * 10, aggregate, aggregate

    monkeypatch.setattr(collector, "_counterfactual_branch_outcome_v3", branch)

    def run(**kwargs):
        api = _safety_api()
        api.branch_phase = False
        api.behavior_writes = []
        api.saved_times = []
        api.close = lambda: None
        api.simulation.getMinExpectedNumber = lambda: 1
        api.simulation.saveState = lambda path: api.saved_times.append(api.simulation.getTime())
        api.trafficlight.setRedYellowGreenState = lambda tls, state: (
            api.behavior_writes.append((api.simulation.getTime(), tls, state))
            if not api.branch_phase else None
        )
        dataset = collector.collect_counterfactual_transitions_v3(
            sumo_api=api, sumocfg=Path("toy.sumocfg"), scenario="toy", seed=7,
            duration_sec=60, warmup_sec=0, control_interval_sec=10,
            counterfactual_horizon_intervals=1, max_focal_tls=1,
            counterfactual_cost_mode="halted_queue", **kwargs,
        )
        return dataset, api

    return run


@pytest.mark.parametrize("requested", [None, [], ["toy:seed7:99:a"], ["toy:seed7:1:b", "toy:seed7:4:b"]])
def test_requested_groups_only_reduce_branches_without_changing_behavior(collect, requested):
    baseline, original_api = collect()
    selected, selected_api = collect(selected_action_group_ids=requested)
    original_groups = baseline.metadata["action_group_ids"]
    mask = np.array([requested is None or group in requested for group in original_groups])
    assert selected.metadata["action_group_ids"] == list(np.array(original_groups)[mask])
    np.testing.assert_array_equal(selected.features, baseline.features[mask])
    for name in baseline.targets:
        np.testing.assert_array_equal(selected.targets[name], baseline.targets[name][mask])
    assert selected_api.behavior_writes == original_api.behavior_writes
    for key in ("behavior_trace_sha256", "behavior_interval_count", "phase_execution_audit",
                "eligible_collection_opportunities", "focal_selection_count_by_tls"):
        assert selected.metadata[key] == baseline.metadata[key]
    assert selected.metadata["selected_action_group_ids"] == (
        None if requested is None else sorted(requested)
    )
    assert selected.metadata["counterfactual_branches"] == int(mask.sum())
    assert len(selected_api.saved_times) == len(set(selected.metadata["action_group_ids"]))
    assert original_api.saved_times == [0, 10, 20, 30, 40, 50]


@pytest.mark.parametrize("shard", [0, 1])
def test_group_filter_preserves_original_shard_assignment(collect, shard):
    requested = ["toy:seed7:1:b", "toy:seed7:4:b"]
    baseline, original_api = collect(collection_shard_count=2, collection_shard_index=shard)
    selected, selected_api = collect(
        collection_shard_count=2, collection_shard_index=shard,
        selected_action_group_ids=requested,
    )
    assert selected.metadata["action_group_ids"] == [
        group for group in baseline.metadata["action_group_ids"] if group in requested
    ]
    assert selected_api.saved_times == ([40] if shard == 0 else [10])
    assert selected_api.behavior_writes == original_api.behavior_writes
    assert selected.metadata["eligible_collection_opportunities"] == 6
    assert selected.metadata["focal_selection_count_by_tls"] == {"a": 2, "b": 2, "c": 2}
