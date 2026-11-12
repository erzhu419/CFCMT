from copy import deepcopy

import pytest

from scripts.cluster.tsc_joint_response_interface import (
    EXECUTOR_FIELDS, SINGLE_LANE_FIELDS, build_joint_response_interface,
)


def fixture():
    catalog = {
        "m0": dict(tls_id="tls", incoming_edge="in", outgoing_edge="out",
            incoming_lanes=["in_0"], outgoing_lanes=["out_0"], internal_lanes=[":a_0"], link_indices=[0]),
        "m1": dict(tls_id="tls", incoming_edge="in", outgoing_edge="side",
            incoming_lanes=["in_0"], outgoing_lanes=["side_0"], internal_lanes=[":b_0"], link_indices=[1]),
    }
    states = ["Gg", "gG", "Yy", "yY", "Gr", "rG", "rr", "gg"]
    data = dict(case_id="fixture", movement_ids=["m0", "m1"], reference_index=3,
        feature_names=list(EXECUTOR_FIELDS + SINGLE_LANE_FIELDS),
        X=[[[1.] * len(EXECUTOR_FIELDS) + [2., 1., .025, 4.5] for _ in range(2)] for _ in range(8)],
        Y="forbidden label payload", costs_450="forbidden cost payload",
        branches=[dict(candidate_index=i, preaction=dict(
            nominal_states={"tls": [states[i]] * 10}, nominal_request_accepted={"tls": True}),
            labels="forbidden branch labels", costs="forbidden cost curve") for i in [3, 0, 1, 2, 4, 5, 6, 7]])
    frame = dict(offset_sec=0, time_sec=600., vehicles={
        "near": dict(lane_id="in_0", position_m=160.),
        "far": dict(lane_id="in_0", position_m=20.),
        "inside": dict(lane_id=":a_0", position_m=2.)},
        receiving_lanes={"out_0": dict(vehicles=0, halted=0, mean_speed_mps=10.)})
    geometry = {lane: dict(length_m=200. if lane == "in_0" else 20., edge_id=lane[:-2])
                for lane in ("in_0", "out_0", "side_0", ":a_0", ":b_0")}
    speeds = {lane: 10. for lane in geometry}
    return data, frame, catalog, geometry, speeds


def test_shared_lane_supply_is_preserved_once_without_turn_allocation():
    result = build_joint_response_interface(*fixture())
    lane = next(row for row in result["lane_nodes"] if row["lane_id"] == "in_0")
    assert lane["vehicle_count"] == 2 and lane["near_stopline_vehicle_count"] == 1
    assert lane["distances_to_lane_end_m"] == [40., 180.]
    assert lane["shared_turn_lane"] and lane["candidate_outgoing_edges"] == ["out", "side"]
    assert lane["lane_measurements_source"] == "preaction_single_lane_row"
    assert lane["lane_measurements"] == dict(vehicles=2, halted=1, occupancy_raw=.025, mean_speed_mps=4.5)
    assert result["lane_to_movement"] == [["in_0", "m0"], ["in_0", "m1"]]
    assert all("vehicle_count" not in movement for movement in result["movement_nodes"])


def test_complete_joint_sequence_preserves_priority_and_other_movements():
    result = build_joint_response_interface(*fixture())
    views = result["candidate_views"]
    assert result["reference_index"] == 3 and views[3]["link_sequences"] == {"0": "y" * 10, "1": "Y" * 10}
    assert views[2]["link_sequences"] == {"0": "Y" * 10, "1": "y" * 10}
    assert views[0]["link_sequences"]["0"] == views[4]["link_sequences"]["0"]
    assert views[0]["link_sequences"] != views[4]["link_sequences"]


def test_encoder_ignores_labels_and_preserves_input_ownership():
    inputs = fixture(); before = deepcopy(inputs)
    result = build_joint_response_interface(*inputs)
    assert inputs == before
    data = inputs[0]; data["Y"] = [999999]; data["costs_450"] = [-999999]
    for branch in data["branches"]:
        branch["labels"] = object(); branch["costs"] = object()
    assert build_joint_response_interface(*inputs) == result
    result["movement_nodes"][0]["incoming_lanes"].append("mutated")
    result["candidate_views"][0]["nominal_states"][0] = "mutated"
    assert inputs[2] == before[2] and inputs[0]["branches"][1]["preaction"] == before[0]["branches"][1]["preaction"]


def test_future_frame_rejected_and_unknown_conflicts_not_invented():
    inputs = fixture(); result = build_joint_response_interface(*inputs)
    assert result["conflicts"]["status"] == "unavailable" and result["conflicts"]["foes"] is None
    relations = dict(foes=[[0, 1]], response=[[1, 0]], mapping_complete=False,
        indexing="mapped TLS link indices", source="static network")
    result = build_joint_response_interface(*inputs, conflict_relations=relations)
    assert result["conflicts"]["status"] == "partial"
    result["conflicts"]["foes"][0][0] = 1
    assert relations["foes"] == [[0, 1]]
    inputs[1]["offset_sec"] = 1
    with pytest.raises(ValueError, match="preaction"):
        build_joint_response_interface(*inputs)


def test_singleton_recovery_uses_t0_exact_rows_and_never_splits_multilane_aggregates():
    inputs = fixture()
    inputs[1]["receiving_lanes"]["in_0"] = dict(vehicles=2, halted=0, occupancy_raw=.03, mean_speed_mps=7.)
    result = build_joint_response_interface(*inputs)
    lane = next(n for n in result["lane_nodes"] if n["lane_id"] == "in_0")
    assert lane["lane_measurements_source"] == "t0_receiving_lane_frame"
    assert lane["lane_measurements"]["mean_speed_mps"] == 7.

    inputs = fixture(); data, frame = inputs[:2]
    frame["vehicles"] = {"inside": frame["vehicles"]["inside"]}
    for row in data["X"][data["reference_index"]]:
        row[-4:] = [0., 0., 0., 0.]
    result = build_joint_response_interface(*inputs)
    lane = next(n for n in result["lane_nodes"] if n["lane_id"] == "in_0")
    assert lane["lane_measurements"]["mean_speed_mps"] is None
    assert lane["lane_measurements"]["vehicles"] == 0

    inputs = fixture(); data, _, catalog, geometry, speeds = inputs
    for movement in catalog.values():
        movement["incoming_lanes"] = ["in_0", "in_1"]
    geometry["in_1"] = dict(length_m=200., edge_id="in"); speeds["in_1"] = 10.
    result = build_joint_response_interface(*inputs)
    assert all(n["lane_measurements"] is None for n in result["lane_nodes"] if "incoming" in n["roles"])

    inputs = fixture(); data = inputs[0]
    data["X"][data["reference_index"]][1][-1] += 1
    with pytest.raises(ValueError, match="singleton observations disagree"):
        build_joint_response_interface(*inputs)


def test_direct_focal_measurements_precede_receiving_and_singleton_rows_only_at_t0():
    inputs = fixture()
    data, frame = inputs[:2]
    direct = dict(vehicles=2, halted=0, occupancy_raw=.04, mean_speed_mps=8.)
    frame['focal_lane_measurements'] = {'in_0': direct}
    frame['receiving_lanes']['in_0'] = dict(vehicles=2, halted=1, occupancy_raw=.03, mean_speed_mps=7.)
    result = build_joint_response_interface(*inputs)
    lane = next(n for n in result['lane_nodes'] if n['lane_id'] == 'in_0')
    assert lane['lane_measurements_source'] == 'direct_t0_focal_lane'
    assert lane['lane_measurements'] == direct
    lane['lane_measurements']['mean_speed_mps'] = 999.
    assert frame['focal_lane_measurements']['in_0']['mean_speed_mps'] == 8.

    future = deepcopy(frame)
    future['offset_sec'] = 1
    future['time_sec'] += 1
    future['focal_lane_measurements']['in_0']['mean_speed_mps'] = 1000.
    with pytest.raises(ValueError, match='preaction'):
        build_joint_response_interface(data, future, *inputs[2:])
