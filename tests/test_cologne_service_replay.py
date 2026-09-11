from types import SimpleNamespace

from cf_h2o.eval.traffic_signal_cologne_service_replay import (
    CANDIDATE_FEATURES, ObservedRigidOriginator,
    StateConditionedSourceUtilityOriginator,
    TLS_ID, add_selected_demand, compare_replay_prefix, observe_lane,
)


def test_front_straight_vehicle_can_block_queued_green_turn_on_shared_lane():
    # This is the real mechanism under investigation: the front vehicle wants
    # link 2 (red) while the following vehicle wants link 3 (green).
    api = SimpleNamespace(
        lane=SimpleNamespace(
            getLastStepVehicleIDs=lambda lane: ["front", "turner"],
            getLastStepHaltingNumber=lambda lane: 2,
            getLastStepOccupancy=lambda lane: 50.0,
            getLastStepMeanSpeed=lambda lane: 0.0,
        ),
        vehicle=SimpleNamespace(
            getNextTLS=lambda vehicle: [(TLS_ID, 2 if vehicle == "front" else 3, 2.0, "r")],
            getLanePosition=lambda vehicle: 98.0 if vehicle == "front" else 90.0,
            getRoute=lambda vehicle: ["in", "straight"],
            getRouteIndex=lambda vehicle: 0,
            getSpeed=lambda vehicle: 0.0,
            getWaitingTime=lambda vehicle: 120.0,
        ),
    )
    row, ids = observe_lane(api, "in_1", {"front", "departed"}, incoming=True)
    lanes = {"in_1": row}
    add_selected_demand(lanes, "rrrGG")
    assert ids == {"front", "turner"}
    assert row["sampled_lane_leaves"] == 1
    assert row["head"]["id"] == "front"
    assert row["head_selected_signal"] == "r"
    assert row["selected_green_demand"] == 1


def test_replay_comparison_detects_changed_actions_despite_added_diagnostics():
    expected = {"metrics": {"accepted_intervention_trace": [
        {"time_sec": 25260.0, "selected_phase_state": "Gr", "target_originator": {"predicted_score": -0.3}},
        {"time_sec": 27000.0, "selected_phase_state": "rG"},
    ]}}
    actual = {"accepted_intervention_trace": [
        {"time_sec": 25260.0, "selected_phase_state": "Gr", "target_originator": {
            "predicted_score": -0.3, "new_diagnostic": 0,
        }},
    ]}
    assert compare_replay_prefix(expected, actual)["passed"]
    actual["accepted_intervention_trace"][0]["selected_phase_state"] = "rG"
    assert not compare_replay_prefix(expected, actual)["passed"]


def test_missing_original_override_is_not_an_exact_replay():
    expected = {"metrics": {"accepted_intervention_trace": [
        {"time_sec": 25260.0, "selected_phase_state": "Gr"},
    ]}}
    result = compare_replay_prefix(expected, {"accepted_intervention_trace": []})
    assert not result["passed"]
    assert "trace_length" in result["first_mismatches"]


def test_observer_returns_original_decision_without_modifying_candidate_features(monkeypatch):
    import numpy as np
    import cf_h2o.eval.traffic_signal_cologne_service_replay as module

    features = np.arange(2 * len(CANDIDATE_FEATURES), dtype=float).reshape(2, -1)
    before = features.copy()
    contrast = SimpleNamespace(
        feature_names=list(CANDIDATE_FEATURES), features=features,
        metadata={"candidate_states": ["Gr", "rG"]},
    )
    decision = SimpleNamespace(selected_index=1, reference_index=0, predicted_score=-0.5)
    monkeypatch.setattr(StateConditionedSourceUtilityOriginator, "select", lambda *a, **k: decision)
    monkeypatch.setattr(module, "_rigid_score", lambda *a: np.array([0.0, -0.5]))
    observer = object.__new__(ObservedRigidOriginator)
    observer.payload = {"rigid_model": object()}
    observer.current_service_sample = {"lanes": {}}
    observer._augment_contrast = lambda *a, **k: contrast
    assert observer.select(contrast, reference_index=0) is decision
    np.testing.assert_array_equal(features, before)
    assert observer.current_service_sample["selected_phase_state"] == "rG"
