"""Posthoc movement identity and sampled-clock clearance, without fitting."""
from experiments.tsc_packet_conflict_audit_20261110.truth import resolve_from_transitions
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import parallel_receivers, packet
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def frame(time, packets, paths=None, arrived=()):
    return dict(time_s=time, packets={p['packet_id']: p for p in packets}, arrived_ids=list(arrived),
                events=[dict(kind='lane_transition', vehicle_id=key, observed_lane_path=path)
                        for key, path in (paths or {}).items()])


def transitions(frames):
    return [(before, after, dict(time_sec=before['time_s'], applied_states={'J': 'Gg'}))
            for before, after in zip(frames, frames[1:])]


def query(key, time, pid, lane, sampled):
    return dict(query_id=key, time_sec=time, packet_id=pid, lane_id=lane,
                body_trail=[lane], sampled_connection_key=sampled)


def test_first_exit_after_lateral_resolves_other_physical_lane_on_same_road_and_separate_clear_times():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    sampled = topology.connections_from('a0')[0]
    actual = topology.connections_from('a1')[0]
    frames = [
        frame(0, [packet('p', 'a0', 19, 2)]),
        frame(1, [packet('p', 'I1', 1, 2, trail=['a1', 'I1'])], {'p': ['a0', 'a1', 'I1']}),
        frame(2, [packet('p', 'out1', 1, 8, trail=['I1', 'out1'])], {'p': ['I1', 'out1']}),
        frame(3, [packet('p', 'out1', 6, 5)]),
    ]
    truth = resolve_from_transitions(transitions(frames), topology,
        [query('q', 0, 'p', 'a0', sampled['key'])])[0]['observed_truth']
    assert truth['actual_connection_key'] == actual['key']
    assert truth['connection_verdict'] == 'different' and truth['road_target_verdict'] == 'same'
    assert truth['entry_interval_sec'] == [0., 1.]
    assert truth['front_clear_interval_sec'] == [1., 2.]
    assert truth['body_clear_interval_sec'] == [2., 3.]
    assert truth['query_plus1_snapshot']['occupied_internal_lanes'] == ['I1']
    assert truth['entry_left_censored'] is False and truth['body_clear_right_censored'] is False


def test_current_internal_uses_known_owner_and_recovers_past_entry_without_using_sampled_lane():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    c0, c1 = topology.connections_from('a0')[0], topology.connections_from('a1')[0]
    frames = [
        frame(0, [packet('left', 'I0', 1, 0, trail=['a0', 'I0']), packet('past', 'a1', 19, 1)]),
        frame(1, [packet('left', 'I0', 2, 1, trail=['a0', 'I0']),
                  packet('past', 'I1', 1, 2, trail=['a1', 'I1'])], {'past': ['a1', 'I1']}),
        frame(2, [packet('left', 'out0', 1, 7, trail=['I0', 'out0']),
                  packet('past', 'out1', 1, 8, trail=['I1', 'out1'])],
              {'left': ['I0', 'out0'], 'past': ['I1', 'out1']}),
        frame(3, [packet('left', 'out0', 6, 5), packet('past', 'out1', 6, 5)]),
    ]
    resolved = resolve_from_transitions(transitions(frames), topology, [
        query('left', 0, 'left', 'I0', c1['key']), query('past', 1, 'past', 'I1', c1['key'])])
    left, past = [result['observed_truth'] for result in resolved]
    assert left['actual_connection_key'] == c0['key'] and left['connection_verdict'] == 'different'
    assert left['resolution_source'] == 'current_internal_unique'
    assert left['entry_interval_sec'] is None and left['entry_left_censored']
    assert past['connection_verdict'] == 'same' and past['entry_interval_sec'] == [0., 1.]
    assert past['entry_left_censored'] is False
    assert past['front_clear_interval_sec'] == [1., 2.] and past['body_clear_interval_sec'] == [2., 3.]


def test_terminal_END_and_right_censor_do_not_guess_a_later_movement_or_premature_clearance():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    c0, c1 = topology.connections_from('a0')[0], topology.connections_from('a1')[0]
    frames = [
        frame(0, [packet('end', 'a0', 19, 1), packet('waiting', 'a1', 10),
                  packet('zero', 'I0', 0, 0, trail=['a0', 'I0'])]),
        frame(1, [packet('waiting', 'a1', 10), packet('zero', 'I0', 0, 0, trail=['a0', 'I0'])], arrived=['end']),
        frame(2, [packet('waiting', 'a1', 10), packet('zero', 'I0', 0, 0, trail=['a0', 'I0'])]),
    ]
    resolved = resolve_from_transitions(transitions(frames), topology, [
        query('end', 0, 'end', 'a0', c0['key']), query('waiting', 0, 'waiting', 'a1', c1['key']),
        query('zero', 0, 'zero', 'I0', c0['key'])])
    terminal, waiting, zero = [result['observed_truth'] for result in resolved]
    assert terminal['actual_road_target'] == 'END' and terminal['connection_verdict'] == 'different'
    assert terminal['road_target_verdict'] == 'different' and terminal['actual_connection_key'] is None
    assert terminal['resolution_source'] == 'observed_terminal_without_entry'
    assert terminal['terminal_or_disappearance'] == dict(kind='terminal_exit', interval_sec=[0., 1.])
    assert waiting['connection_verdict'] == 'unresolved' and waiting['entry_right_censored']
    assert waiting['entry_interval_sec'] is None
    assert zero['connection_verdict'] == 'same' and zero['front_clear_interval_sec'] is None
    assert zero['body_clear_interval_sec'] is None and zero['body_clear_right_censored']
