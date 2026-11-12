from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import collect_tsc_physical_response_trial as runner


def protocol():
    return {"protocol": runner.PROTOCOL, "response_horizon_sec": 10,
            "cost_horizon_sec": 450, "checkpoint_search_sec": 60, "max_new_branches": 16,
            "cases": [{"case_id": name, "city": city, "seed": 1,
                       "checkpoint_sec": 600, "checkpoint_search_sec": 60,
                       "tls_id": "tls", "sumocfg": "case.sumocfg"}
                      for name, city in (("source", "cologne"), ("target", "jinan"))]}


def test_fixed_case_shards_preserve_unique_groups_and_eight_action_budget():
    spec = protocol()
    runner.validate_spec(spec, [0, 1])
    for indices in ([], [0, 0], [-1], [2]):
        with pytest.raises(ValueError):
            runner.validate_spec(spec, indices)
    for key, value in (("response_horizon_sec", 20), ("cost_horizon_sec", 60), ("max_new_branches", 15)):
        altered = deepcopy(spec)
        altered[key] = value
        with pytest.raises(ValueError):
            runner.validate_spec(altered, [0])


def prefix_runtime(eligible_at=25800):
    state = SimpleNamespace(time=25200, actions=[])
    api = SimpleNamespace(simulation=SimpleNamespace(getTime=lambda: state.time))
    def advance(**kwargs):
        state.actions.append(kwargs["actions"])
        assert kwargs["control_interval_sec"] == 10
        assert kwargs["collision_mode"] == "audit"
        state.time += 10
    v3 = SimpleNamespace(
        _start_sumo=lambda api, path, seed: None,
        _tls_phase_infos=lambda api: {"tls": object()},
        build_signal_routing_graph=lambda api, infos: object(),
        _scenario_context_v3=lambda **kwargs: [],
        build_safe_phase_executors=lambda api, infos: {"tls": SimpleNamespace(
            feasible_states_now=lambda: list("ABCDEFGH") if state.time >= eligible_at else ["A"])},
        _new_safety_ledger=lambda: {},
        _read_graph_states_v3=lambda *args: {"tls": "state"},
        _rule_candidate=lambda *args: SimpleNamespace(state="PP"),
        _advance_with_actions=advance,
        _finalize_safety_ledger=lambda ledger: {"no_teleport_passed": True})
    return api, v3, state


def test_native_prefix_follows_declared_absolute_time_without_restoring():
    api, v3, state = prefix_runtime()
    case = {**protocol()["cases"][0], "checkpoint_sec": 25800}
    result = runner.native_prefix(api, v3, case)
    assert state.time == 25800 and state.actions == [{"tls": "PP"}] * 60
    assert result[-1]["no_teleport_passed"]


def test_checkpoint_search_uses_first_fixed_tls_eligible_epoch_then_reuses_it():
    api, v3, state = prefix_runtime(25820)
    case = {**protocol()["cases"][0], "checkpoint_sec": 25800}
    runner.native_prefix(api, v3, case)
    assert state.time == 25820 and len(state.actions) == 62
    api, v3, state = prefix_runtime(25820)
    runner.native_prefix(api, v3, {**case, "resolved_checkpoint_sec": 25820})
    assert state.time == 25820 and len(state.actions) == 62
    api, v3, state = prefix_runtime(25870)
    with pytest.raises(runner.IneligibleCase):
        runner.native_prefix(api, v3, case)
    assert state.time == 25860 and len(state.actions) == 66


def fake_branch(index):
    return {"start": {"time_sec": 600, "pending_ids": []},
            "candidate_states": list("ABCDEFGH"), "reference_index": 3,
            "candidate_index": index, "resolved_checkpoint_sec": 600,
            "preaction": {"feature_names": ["a", "b", "c"], "movement_ids": ["m1", "m2"],
                          "features": [[index, 1, 2], [index, 3, 4]]},
            "network_preaction": {"tls_ids": ["tls"]}, "focal_features": [index],
            "labels": [[index, index + 1], [index + 2, index + 3]],
            "label_names": ["stopline_entries_10s", "receiving_entries_10s"],
            "cost": 1 + index / 100, "costs": [1 + index / 100] * 450,
            "controlled_lane_count": 8, "checks": {"full_cost_horizon": True},
            "measurement": {"unresolved_count": 0}, "unobserved_first_receiving_count": 0,
            "prefix_safety": {}, "safety": {}}


def test_first_observed_receiving_vehicle_requires_review_not_zero_flow():
    movements = {"tls|in|out": {"outgoing_edge": "out"}}
    initial = {"old": {"road_id": "out", "route_index": 1},
               "tracked": {"road_id": "in", "route_index": 0}}
    safe = {"tracked": {"road_id": "out", "route_index": 1},
            "spawned": {"road_id": "out", "route_index": 0}}
    frames = [{"vehicles": initial}, {"vehicles": safe}]
    assert runner.unobserved_first_receiving_count(frames, movements) == 0
    suspect = {**safe, "missed": {"road_id": "out", "route_index": 1}}
    frames.extend([{"vehicles": suspect}, {"vehicles": suspect}])
    assert runner.unobserved_first_receiving_count(frames, movements) == 1


def install_case_mocks(monkeypatch, changed_order=False):
    def collect(api, v3, case, index, output, *args):
        index = 3 if index is None else index
        branch = fake_branch(index)
        if changed_order and index == 2:
            branch["preaction"]["movement_ids"].reverse()
        return branch
    monkeypatch.setattr(runner, "collect_branch", collect)
    monkeypatch.setitem(sys.modules, "cologne_action_branch_checks", SimpleNamespace(
        compare_physical=lambda a, b, tolerance: {"passed": a == b}))


def test_complete_group_writes_aligned_eight_action_tensor_and_original_costs(monkeypatch, tmp_path):
    install_case_mocks(monkeypatch)
    updates = []
    result = runner.collect_case(SimpleNamespace(close=lambda: None), None, protocol()["cases"][0],
        tmp_path / "source", None, None, None, lambda case, summary: updates.append(summary))
    data = json.loads((tmp_path / "source/group_data_server_only.json").read_text())
    assert result["status"] == "COMPLETE" and result["new_branches_attempted"] == 8
    assert len(updates) == 8 and data["reference_index"] == 3
    assert [row["candidate_index"] for row in updates] == [3, 0, 1, 2, 4, 5, 6, 7]
    assert np.asarray(data["X"]).shape == (8, 2, 3)
    assert np.asarray(data["Y"]).shape == (8, 2, 2)
    assert data["movement_ids"] == ["m1", "m2"]
    assert data["costs_450"] == [1 + i / 100 for i in range(8)]


def test_changed_movement_order_stops_group_and_never_labels_it_complete(monkeypatch, tmp_path):
    install_case_mocks(monkeypatch, changed_order=True)
    result = runner.collect_case(SimpleNamespace(close=lambda: None), None, protocol()["cases"][0],
        tmp_path / "source", None, None, None, lambda *args: None)
    assert result["status"] == "INVALID" and result["new_branches_attempted"] == 4
    assert result["branches"][-1]["checks"]["same_movement_roster"] is False
    assert not (tmp_path / "source/action_4").exists()
