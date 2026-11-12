"""Current receiving bindings versus posthoc observed motion; synthetic only."""
from copy import deepcopy

import pytest

from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import corridor, parallel_receivers, packet
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_residual_truth_20261110.receiving_trace import explain_receiving


def test_moving_receiver_opens_room_without_equating_speed_to_step_displacement():
    static, unused = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    topology = RoadTopology(static)
    ego, lead = packet('ego', 'I0', 5, 2), packet('lead', 'out', 5, 3)
    current = {'ego': ego, 'lead': lead}
    nxt = {'ego': packet('ego', 'I0', 9, 6), 'lead': packet('lead', 'out', 8, 3)}
    connection = topology.connections_from('in')[0]
    result = explain_receiving(topology, ego, current, ('I0', 'out'), nxt,
        connection=connection, observed_ego_path=['I0'], observed_neighbour_paths={'lead': ['out']})
    assert result['current_bound_m'] == 3
    assert result['actual_ego_step_displacement_m'] == 4
    assert result['next_speed_snapshot_mps'] == 6
    assert result['next_speed_times_dt_minus_step_displacement_m'] == 2
    binding = result['binding_constraints'][0]
    assert binding['kind'] == 'body' and binding['current_static_gap_m'] == 5
    assert binding['posthoc_neighbour']['body_rear_projection_change_m'] == 3
    assert binding['posthoc_end_gap_m'] == 4
    assert binding['posthoc_end_gap_minus_min_gap_m'] == 2
    changed = deepcopy(nxt)
    changed['lead']['position_m'] = 40
    alternative = explain_receiving(topology, ego, current, ('I0', 'out'), changed,
                                    connection=connection, observed_ego_path=['I0'])
    assert alternative['current_bound_m'] == result['current_bound_m']
    for key in ('packet_id', 'bound_m', 'current_static_gap_m', 'segment_rear_m', 'segment_front_m'):
        assert alternative['binding_constraints'][0][key] == binding[key]


def test_same_road_neighbour_lane_change_is_not_a_whole_road_longitudinal_jump():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    ego, lead = packet('ego', 'out0', 10, 2), packet('lead', 'out0', 18, 3)
    nxt = {'ego': packet('ego', 'out0', 12, 2), 'lead': packet('lead', 'out1', 21, 3)}
    result = explain_receiving(topology, ego, {'ego': ego, 'lead': lead}, ('out0',), nxt,
        observed_ego_path=['out0'], observed_neighbour_paths={'lead': ['out0', 'out1']})
    assert result['current_bound_m'] == 1
    neighbour = result['binding_constraints'][0]['posthoc_neighbour']
    assert neighbour['observed_motion']['displacement_m'] == 3
    assert neighbour['same_road_lane_change'] and neighbour['front_left_model_path']
    assert not neighbour['next_body_still_on_model_path']
    assert result['binding_constraints'][0]['posthoc_end_gap_m'] is None


def test_connection_then_lane_change_does_not_add_the_receiving_lane_length_twice():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    ego = packet('ego', 'a0', 19, 4)
    nxt = {'ego': packet('ego', 'out1', 2, 9, trail=['I1', 'out1'])}
    connection = topology.connections_from('a0')[0]
    result = explain_receiving(topology, ego, {'ego': ego}, connection['path'], nxt,
        connection=connection, observed_ego_path=['a0', 'I0', 'out0', 'out1'])
    assert result['binding_constraints'] == [dict(kind='path_end', bound_m=109.)]
    assert result['actual_ego_step_displacement_m'] == pytest.approx(11.)
    assert result['ego_end_projection_on_model_path_m'] is None
    assert not result['future_observations_used_for_current_bound']
