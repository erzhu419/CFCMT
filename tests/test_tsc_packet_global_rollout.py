"""Exhaust small branch spaces and audit the sampled whole-frame transitions."""

from collections import defaultdict
from copy import deepcopy
from itertools import product
from math import exp, log, prod

import pytest

from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_conditional_law_20261110.synthetic import cases as background_cases
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_global_rollout_20261111.kernel import PacketKernel
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import cases
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES, FIXED_DIVISORS
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    corridor, executor_seed, initial_observation, packet,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.evaluate import StructuralAudit


def coefficients(*, left=1., right=2.):
    odds = (3/2, 1/3, 2., left, right)
    result = {name: [log(value)] + [0.] * (len(FEATURE_NAMES) - 1)
              for name, value in zip(PARAMETERS, odds, strict=True)}
    speed_feature = FEATURE_NAMES.index('speed_mps')
    result['brake'][speed_feature] = -.6
    result['lane_right'][speed_feature] = .8
    return result


def intervals(*boundaries):
    edges = sorted({0., *boundaries, 1.})
    return [((a+b)/2, b-a) for a,b in zip(edges, edges[1:])]


def speed_odds(theta, name, packet):
    i = FEATURE_NAMES.index('speed_mps')
    return exp(theta[name][0] + theta[name][i] * packet['speed_mps'] / FIXED_DIVISORS[i])


def lateral_intervals(theta, packet):
    """Stay is always legal. Partition the draw for every possible side mask."""
    left, right = (speed_odds(theta, name, packet) for name in ('lane_left', 'lane_right'))
    return intervals(left/(left+1), 1/(right+1), left/(left+1+right), (left+1)/(left+1+right))


def atom_key(actors):
    return tuple((key, tuple(sorted(value.items()))) for key,value in sorted(actors.items()))


def sampled_atom(kernel, old, row):
    actors = {}
    changes = row['lane_change_actions']
    for key, original in old.items():
        motion = kernel._last_receiving_resolution['results'][key]
        trace = motion['trace']
        entry = int(any(kernel.topology.lanes[lane]['internal'] for lane in trace[1:]) or any(
            kernel.topology.lanes[a]['edge_id'] != kernel.topology.lanes[b]['edge_id']
            for a,b in zip(trace, trace[1:])))
        survivor = kernel.state['packets'].get(key)
        speed = survivor['speed_mps'] if survivor is not None else None
        actors[key] = dict(survived=survivor is not None, physical_speed_mps=speed,
            started=int(speed >= .1) if survivor is not None and original['speed_mps'] < .1 else None,
            entry_event=entry, lane_change=changes[key] if survivor is not None else None,
            lane_id=survivor['lane_id'] if survivor is not None else None,
            position_m=survivor['position_m'] if survivor is not None else None)
    return atom_key(actors)


@pytest.mark.parametrize('name,signal,active_draws', [
    ('stopped_open', None, ('start',)),
    ('yielding_entry', 'g', ('yield',)),
    ('two_actor_lateral', None, ('lane',)),
    ('intent_and_survival', 'G', ('intent',)),
    ('receiving_intent', 'GG', ('intent','lane')),
])
@pytest.mark.parametrize('theta', [coefficients(), coefficients(left=2., right=1.)], ids=['old_lc','new_lc'])
def test_whole_frame_sampling_matches_every_atom_and_probability_when_j_is_all(name, signal, active_draws, theta):
    exact_kernel, focal = cases()[name]
    exact = enumerate_conditional(exact_kernel, focal, theta)
    old = exact_kernel.state['packets']
    assert set(exact['stochastic_participants']) == old.keys()
    expected = {atom_key(atom['actors']):atom['probability'] for atom in exact['atoms']}
    state = deepcopy(exact_kernel.state)
    # The existing fixture already drew an intent. Start from the prior here,
    # as the exact law integrates the current intent on these initial states.
    state['intents'] = {}
    seeds = {} if signal is None else {'J':executor_seed((signal,), clearance=tuple(exact_kernel.static['internal_lane_ids']))}
    choices = {}
    for key, packet in old.items():
        brake_odds = speed_odds(theta, 'brake', packet)
        choices[key] = {'brake':intervals(brake_odds/(1.+brake_odds)), 'start':intervals(.25),
            'yield':intervals(2/3), 'lane':lateral_intervals(theta, packet),
            'intent':intervals(.4 if name == 'intent_and_survival' else .3)}
    axes = [(key, draw) for key in old for draw in ('brake', *active_draws)]
    actual = defaultdict(float)
    for branch in product(*(choices[key][draw] for key,draw in axes)):
        selected = dict(zip(axes, (value for value,weight in branch), strict=True))
        kernel = PacketKernel(exact_kernel.static, state, seeds, exact_kernel.laws, theta, random_seed=23,
                              conflict_geometry=exact_kernel._spatial_geometry)
        kernel._step_uniforms = lambda key,now: {
            draw:selected.get((key,draw), .5) for draw in ('brake','start','yield','lane')}
        kernel.noise.intent_uniform = lambda key: selected.get((key,'intent'), .5)
        row = kernel.step()
        actual[sampled_atom(kernel, old, row)] += prod(weight for value,weight in branch)
    assert dict(actual) == pytest.approx(expected, abs=1e-12)


def test_indexed_prestate_features_equal_full_frame_features_with_foes_and_receivers():
    fixtures = background_cases()
    for name,signal in (('responsive_background','Gg'), ('receiving_with_background','GG')):
        oracle, _ = fixtures[name]
        seeds = {'J':executor_seed((signal,), clearance=tuple(oracle.static['internal_lane_ids']))}
        kernel = PacketKernel(oracle.static, oracle.state, seeds, oracle.laws, coefficients(), random_seed=23,
                              conflict_geometry=oracle._spatial_geometry)
        packets = kernel.state['packets']
        kernel._packet_order = {key:index for index,key in enumerate(packets)}
        kernel._connections = {key:kernel._connection(packet) for key,packet in packets.items()}
        full_frame = [kernel.model.features(kernel._context(packet, packets, *kernel._connections[key]))
                      for key,packet in packets.items()]
        assert kernel._features(packets).tolist() == [list(row) for row in full_frame]


def test_multistep_audit_covers_executor_requests_arrivals_and_exits():
    static, history = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=20)
    history['due_rate_per_sec'] = .7
    seeds = {'J':executor_seed(('G','g'), clearance=('I0',))}
    state = initial_observation(static, [packet('approach','in',18.,4.), packet('sink','out',19.,6.)],
                                pending=('queued',), executors=seeds)
    geometry = build_entity_geometry(RoadTopology(static), 2.)
    kernel = PacketKernel(static, state, seeds, history, coefficients(), random_seed=319,
                          conflict_geometry=geometry)
    audit = StructuralAudit(static, kernel.state, seeds)
    rows = []
    for step in range(12):
        before = deepcopy(kernel.state)
        requests = {'J':'g'} if step == 0 else None
        row = kernel.step(initial_requests=requests)
        audit.check(before, kernel.state, row, requests)
        rows.append(row)
    assert audit.steps == 12
    assert rows[0]['requests'] == {'J':'g'}
    assert rows[10]['requests']
    assert any(row['due_ids'] for row in rows)
    assert any('queued' in row['inserted_ids'] for row in rows)
    assert 'sink' in rows[0]['exited_ids']


@pytest.mark.parametrize('signal,mode,speed,can_enter', [
    ('r', 'all_red', .5, False),
    ('y', 'yellow', .5, False),
    ('Y', 'yellow', .5, False),
    ('G', 'green', .5, True),
    ('g', 'green', .5, True),
    ('Y', 'yellow', 3., True),
])
def test_stopline_entry_uses_current_signal_when_travel_reaches_lane_end(signal, mode, speed, can_enter):
    static, history = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=20)
    seeds = {'J': executor_seed((signal if mode == 'green' else 'G',), current=signal,
                               mode=mode, remaining=1., clearance=('I0',))}
    approach = packet('approach', 'in', 19.5, speed)
    approach['parameters']['max_speed_mps'] = speed
    state = initial_observation(static, [approach], executors=seeds, time_s=1)
    geometry = build_entity_geometry(RoadTopology(static), 2.)
    kernel = PacketKernel(static, state, seeds, history, coefficients(), random_seed=319,
                          conflict_geometry=geometry)
    kernel._step_uniforms = lambda key, now: dict(brake=.99, start=0., **{'yield': 0.}, lane=.5)

    row = kernel.step()

    assert row['applied_signals']['J'] == signal
    moved = kernel.state['packets']['approach']
    assert moved['lane_id'] == ('I0' if can_enter else 'in')
    if speed == .5:
        assert moved['position_m'] == (0. if can_enter else 20.)
    if signal == 'r':
        row = kernel.step()
        assert row['applied_signals']['J'] == 'G'
        assert kernel.state['packets']['approach']['lane_id'] == 'I0'


def test_red_allows_committed_internal_vehicle_to_clear_and_sink_vehicle_to_exit():
    static, history = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=20)
    seeds = {'J': executor_seed(current='r', mode='all_red', remaining=2., clearance=('I0',))}
    state = initial_observation(static, [packet('committed', 'I0', 9.5, 1., trail=('in', 'I0')),
                                        packet('sink', 'out', 19.5, 1.)], executors=seeds, time_s=1)
    geometry = build_entity_geometry(RoadTopology(static), 2.)
    kernel = PacketKernel(static, state, seeds, history, coefficients(), random_seed=319,
                          conflict_geometry=geometry)
    kernel._step_uniforms = lambda key, now: dict(brake=.99, start=0., **{'yield': 0.}, lane=.5)

    row = kernel.step()

    assert row['applied_signals']['J'] == 'r'
    assert kernel.state['packets']['committed']['lane_id'] == 'out'
    assert row['exited_ids'] == ['sink']
