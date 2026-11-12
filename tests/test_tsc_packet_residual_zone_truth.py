"""One-second zone brackets and real body-trail evidence only."""
from copy import deepcopy

from experiments.tsc_packet_residual_truth_20261110.zone_truth import observe_zone_truth
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    corridor, packet, lane, connection, road_graph)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def snapshot(time, observed):
    return dict(time_sec=time, packet=observed)


def test_zone_entry_front_exit_and_tail_clear_have_separate_frame_intervals():
    static, unused = corridor(incoming_length=10, internal_lengths=(2, 3), outgoing_length=100)
    topology = RoadTopology(static)
    movement = topology.connections_from('in')[0]
    snapshots = [snapshot(0, packet('p', 'in', 9, 30)),
                 snapshot(1, packet('p', 'I0', 1.5, 30, trail=['in', 'I0'])),
                 snapshot(2, packet('p', 'out', 3, 30, trail=['I1', 'out'])),
                 snapshot(3, packet('p', 'out', 8, 30))]
    result = observe_zone_truth(topology, movement, [11., 17.], snapshots)
    assert [frame['front_progress_m'] for frame in result['frames']] == [9., 11.5, 18., 23.]
    assert [frame['body_on_zone'] for frame in result['frames']] == [False, True, True, False]
    assert result['front_entry_interval_sec'] == [0, 1]
    assert result['front_exit_interval_sec'] == [1, 2]
    assert result['tail_clear_interval_sec'] == [2, 3]
    # Stored endpoint speeds do not interpolate any event inside these brackets.
    assert result['same_interval_event_groups'] == []


def test_passing_whole_zone_between_samples_keeps_all_subsecond_order_unresolved():
    static, unused = corridor(incoming_length=10, internal_lengths=(1,), outgoing_length=100)
    topology = RoadTopology(static)
    movement = topology.connections_from('in')[0]
    snapshots = [snapshot(7, packet('p', 'in', 9, 0)), snapshot(8, packet('p', 'out', 7, 9))]
    result = observe_zone_truth(topology, movement, [10., 11.], snapshots)
    assert result['front_entry_interval_sec'] == result['front_exit_interval_sec'] == result['tail_clear_interval_sec'] == [7, 8]
    assert result['same_interval_event_groups'] == [dict(interval_sec=[7, 8],
        events=['front_entry', 'front_exit', 'tail_clear'], within_interval_order='unresolved')]


def test_front_beyond_receiving_path_has_no_invented_progress_and_clear_requires_complete_body():
    static = road_graph([lane('in', 10), lane('I0', 5, internal=True), lane('out', 10), lane('after', 100)],
        [connection('in', 'out', 'I0', tls='J', link=0), connection('I0', 'out'), connection('out', 'after')])
    topology = RoadTopology(static)
    movement = topology.connections_from('in')[0]
    snapshots = [snapshot(0, packet('p', 'out', 8, 1)),
                 snapshot(1, packet('p', 'after', 1, 3, trail=['out', 'after'])),
                 snapshot(2, packet('p', 'after', 6, 5))]
    result = observe_zone_truth(topology, movement, [20., 24.], snapshots)
    assert [frame['front_progress_m'] for frame in result['frames']] == [23., None, None]
    assert [frame['body_on_zone'] for frame in result['frames']] == [True, True, False]
    assert result['front_exit_interval_sec'] is None and result['censoring']['front_exit'] == 'unresolved'
    assert result['tail_clear_interval_sec'] == [1, 2]
    incomplete = deepcopy(snapshots)
    incomplete[-1]['packet']['body_path_complete'] = False
    unresolved = observe_zone_truth(topology, movement, [20., 24.], incomplete)
    assert unresolved['frames'][-1]['body_on_zone'] is None
    assert unresolved['tail_clear_interval_sec'] is None and unresolved['censoring']['tail_clear'] == 'unresolved'
