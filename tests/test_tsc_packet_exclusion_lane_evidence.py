"""Posthoc evidence for fixed model-denying neighbours; no native execution."""
from copy import deepcopy

from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet, parallel_receivers
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_exclusion_mechanisms_20261110.lane_evidence import explain_lane


def lane_detail(position, neighbours):
    return dict(projected_lane_id='a0', projected_position_m=position, sides=dict(left=dict(
        target_lane_id='a1', potential_legal=True, allowed=False, neighbour_states=neighbours)))


def predicted(key, position):
    return dict(packet_id=key, lane_id='a1', position_m=position, speed_mps=7.,
                length_m=5., min_gap_m=2., resolved=True)


def test_rear_motion_overprediction_compares_same_actor_not_actual_nearest_and_keeps_speed_separate():
    topology = RoadTopology(parallel_receivers()[0])
    before, after = packet('ego', 'a0', 8, 2), packet('ego', 'a1', 14, 9)
    rear, next_rear = packet('rear', 'a1', 3, 1), packet('rear', 'a1', 6, 8)
    detail = lane_detail(15, [predicted('rear', 10)])
    frozen = deepcopy(detail)
    result = explain_lane(topology, {'ego': before, 'rear': rear}, {'ego': after, 'rear': next_rear},
        before, after, detail, 0, physical_proposal_mps=8., mean0_mps=7.)
    assert result['current_denial_reasons'] == ['rear_gap_denies_observed_lane_choice']
    evidence = result['denying_neighbours'][0]
    assert evidence['packet_id'] == 'rear' and evidence['model_gap_m'] == 0
    assert evidence['actual_terminal_gap']['gap_m'] == 3
    assert evidence['actual_step_displacement_m'] == 3
    assert evidence['next_speed_snapshot_mps'] == 8
    assert evidence['next_speed_times_dt_minus_displacement_m'] == 5
    assert detail == frozen and not result['future_used_to_change_mask']


def test_front_changes_out_of_model_target_and_same_road_displacement_is_not_extra_lane_length():
    topology = RoadTopology(parallel_receivers()[0])
    before, after = packet('ego', 'a0', 8, 2), packet('ego', 'a1', 13, 5)
    front, next_front = packet('front', 'a1', 11, 2), packet('front', 'a0', 15, 9)
    result = explain_lane(topology, {'ego': before, 'front': front}, {'ego': after, 'front': next_front},
        before, after, lane_detail(13, [predicted('front', 17)]), 0,
        physical_proposal_mps=5., mean0_mps=5.)
    evidence = result['denying_neighbours'][0]
    assert evidence['model_role'] == 'front'
    assert evidence['observed_same_road_lane_change']
    assert evidence['next_front_left_model_target_lane']
    assert not evidence['next_body_still_on_model_target_lane']
    assert evidence['actual_terminal_gap']['status'] == 'neighbour_front_on_different_lane'
    assert evidence['actual_step_displacement_m'] == 4


def test_all_front_and_rear_denials_survive_without_single_cause_assignment():
    topology = RoadTopology(parallel_receivers()[0])
    before, after = packet('ego', 'a0', 11, 3), packet('ego', 'a1', 18, 7)
    front, rear = packet('front', 'a1', 14, 1), packet('rear', 'a1', 6, 3)
    result = explain_lane(topology, {'ego': before, 'front': front, 'rear': rear},
        {'ego': after, 'front': packet('front', 'a1', 23, 9), 'rear': packet('rear', 'a1', 9, 3)},
        before, after, lane_detail(18, [predicted('front', 20), predicted('rear', 12)]), 0,
        physical_proposal_mps=7., mean0_mps=7.)
    assert set(result['current_denial_reasons']) == {
        'front_gap_denies_observed_lane_choice', 'rear_gap_denies_observed_lane_choice'}
    assert {row['packet_id'] for row in result['denying_neighbours']} == {'front', 'rear'}


def test_receiver_reach_is_separate_from_lateral_denial_and_connection_then_lc_path_is_collapsed():
    topology = RoadTopology(parallel_receivers()[0])
    before, after = packet('ego', 'a0', 19, 6), packet('ego', 'out1', 3, 12)
    detail = dict(projected_lane_id='I0', projected_position_m=5., sides={})
    result = explain_lane(topology, {'ego': before}, {'ego': after}, before, after, detail, 0,
        physical_proposal_mps=10., mean0_mps=6., observed_focal_path=['a0', 'I0', 'out0', 'out1'],
        candidate_connection=topology.connections[0])
    arrival = result['arrival_evidence']
    assert arrival['observed_receiver']['distance_m'] == 9
    assert arrival['actual_step_displacement_m'] == 12
    assert arrival['physical_distance_reaches_observed_receiver_entry']
    assert not arrival['mean_distance_reaches_observed_receiver_entry']
    assert 'model_mean_distance_short_of_observed_receiver_entry' in result['current_denial_reasons']
    assert 'no_adjacent_model_target_lane' in result['current_denial_reasons']
    assert not result['denying_neighbours']
