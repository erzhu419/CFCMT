"""Synthetic exclusion explanations; neither native simulation nor fitting."""
from copy import deepcopy
import json

from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    lane, road_graph, laws, packet, initial_observation, parallel_receivers, executor_seed,
)
from experiments.tsc_target_packet_training_20261109.lane_exclusion_detail import diagnose_lane


def frame(static, packets, time_s=0, executors=None, events=None):
    state = initial_observation(static, packets, executors=executors, time_s=time_s)
    return dict(time_s=time_s, packets=state['packets'], native_lanes=state['lane_sensors'],
                native_pending_ids=[], executors={tls: dict(snapshot=value)
                                                 for tls, value in state['executors'].items()}, unresolved=[],
                checks={'synthetic': True}, events=events or [])


def test_same_clock_front_gap_reason_and_actual_evidence_never_change_prediction():
    static = road_graph([lane('r0', 100, edge='r', index=0),
                         lane('r1', 100, edge='r', index=1)], [], cost_lanes=['r0', 'r1'])
    before = frame(static, [packet('ego', 'r0', 20, 5), packet('front', 'r1', 26, 0)])
    after = frame(static, [packet('ego', 'r1', 21, 1), packet('front', 'r1', 29, 3)], time_s=1,
                  events=[dict(kind='lane_transition', vehicle_id='ego',
                               observed_lane_path=['r0', 'r1'])])
    action = dict(time_sec=0, requests={}, applied_states={})
    reference = dict(seed=1, time_sec=0, packet_id='ego', y=0)
    original = deepcopy((before, after, action, reference))
    result = diagnose_lane(static, {}, laws(static), before, after, action, reference)
    candidate = result['candidates'][0]
    assert candidate['detail']['reasons'] == ['front_gap']
    assert not candidate['observed_direction_allowed']
    assert candidate['detail']['same_clock_gaps']['front_gap_m'] < 0
    assert result['actual']['same_clock_gaps']['front_gap_m'] == 3.
    assert result['actual']['lateral_steps'] == [dict(from_lane='r0', to_lane='r1')]
    assert not result['actual']['crossed_road']
    assert (before, after, action, reference) == original

    changed = deepcopy(after)
    changed['packets']['front']['position_m'] = 70.
    repeated = diagnose_lane(static, {}, laws(static), before, changed, action, reference)
    assert result['candidates'] == repeated['candidates']
    assert result['actual'] != repeated['actual']
    json.dumps(result, allow_nan=False)


def test_internal_to_receiving_lane_change_has_explicit_actual_path():
    static, target_laws = parallel_receivers()
    seeds = {'J': executor_seed(('gg',))}
    before = frame(static, [packet('ego', 'I0', 7, 5, trail=['a0', 'I0']),
                            packet('front', 'out1', 10, 0)], executors=seeds)
    after = frame(static, [packet('ego', 'out1', 6, 7, trail=['I1', 'out1']),
                           packet('front', 'out1', 13, 3)], time_s=1, executors=seeds,
                  events=[dict(kind='lane_transition', vehicle_id='ego',
                               observed_lane_path=['I0', 'out0', 'out1'])])
    result = diagnose_lane(static, seeds, target_laws, before, after,
        dict(time_sec=0, requests={}, applied_states={'J': 'gg'}),
        dict(seed=1, time_sec=0, packet_id='ego', y=0))
    assert len(result['candidates']) == 1
    candidate = result['candidates'][0]
    assert candidate['detail']['projected']['lane_id'] == 'out0'
    assert candidate['detail']['target_lane'] == 'out1'
    assert candidate['detail']['reasons'] == ['front_gap']
    assert result['actual']['crossed_road']
    assert result['actual']['lateral_steps'] == [dict(from_lane='out0', to_lane='out1')]
    json.dumps(result, allow_nan=False)
