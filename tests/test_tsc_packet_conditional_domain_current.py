"""Current-only binding preserves nuisance streams without response calls."""
from copy import deepcopy

from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.scenario_noise import ScenarioNoise
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    connection, corridor, lane, laws, packet, road_graph,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def frame(static, rows, pending=()):
    packets = {row['packet_id']: deepcopy(row) for row in rows}
    return dict(time_s=7, packets=packets,
        native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=list(pending), executors={}, unresolved=[], checks=dict(synthetic=True))


def shared_internal():
    static = road_graph([lane('a', 20), lane('b', 20), lane('I', 10, internal=True),
                         lane('out_a', 50), lane('out_b', 50)],
        [connection('a', 'out_a', 'I'), connection('b', 'out_b', 'I'),
         connection('I', 'out_a'), connection('I', 'out_b')])
    return static, laws(static, intents={'a': {'out_a': 1.}, 'b': {'out_b': 1.},
                                       'out_a': {'END': 1.}, 'out_b': {'END': 1.}})


def test_geometry_uses_current_whitelist_and_model_speed_formula_without_mutation():
    static = road_graph([lane('a', 100)], [])
    row = packet('ego', 'a', 20, 5)
    row['parameters'].update(max_speed_mps=12., speed_factor=.5)
    row['allowed_speed_mps'] = .01  # Native allowed speed is not the model formula.
    row['_movement_key'] = 'unapproved_route_audit'
    row['future_route'] = ['missing_lane']
    current = frame(static, [row])
    current['future_frame'] = {'packets': {'ego': {'speed_mps': -999}}}
    current['intents'] = {'ego': {'road_id': 'a', 'choice': 'unapproved'}}
    before = deepcopy(current)
    view = GeometryView(static, current, laws(static), {}, 23)
    assert view._allowed_speed(view.state['packets']['ego']) == 10.
    assert '_movement_key' not in view.state['packets']['ego']
    assert 'future_route' not in view.state['packets']['ego']
    assert view.state['intents'] == {} and current == before
    assert not hasattr(view, 'model') and not hasattr(view, '_connections')


def test_unique_internal_binding_ignores_a_raw_key_the_original_initializer_strips():
    static, history = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    row = packet('ego', 'I0', 5, 1, trail=['in', 'I0'])
    row['_movement_key'] = 'not_the_registered_current_binding'
    current = frame(static, [row])
    view = GeometryView(static, current, history, {'J': 'g'}, 23)
    expected = RoadTopology(static).internal_connections['I0'][0]['key']
    assert view.state['packets']['ego']['_movement_key'] == expected
    info = view.internal_bindings['ego']
    assert info['binding_candidate_count'] == 1 and not info['ambiguous']
    assert info['raw_movement_key_present_but_not_used']
    assert view.signals == {'J': 'g'}


def test_shared_internal_binding_uses_pending_registered_streams_and_fixed_seed():
    static, history = shared_internal()
    current = frame(static, [packet('ego', 'I', 5, 1)], pending=['aaa_pending'])
    first = GeometryView(static, current, history, {}, 12345)
    second = GeometryView(static, current, history, {}, 12345)
    expected_noise = ScenarioNoise(12345, {'ego', 'aaa_pending'}, 7)
    candidates = first.topology.internal_connections['I']
    expected = candidates[int(expected_noise.intent_uniform('ego')*len(candidates))]['key']
    assert first.state['packets']['ego']['_movement_key'] == expected
    assert first.internal_bindings == second.internal_bindings
    assert first.internal_bindings['ego']['ambiguous']
    assert set(first.noise._packets) == {'ego', 'aaa_pending'}
    assert current['packets']['ego']['body_trail'] == ['I']


def test_body_trail_disambiguation_uses_no_latent_draw_for_a_unique_candidate():
    static, history = shared_internal()
    current = frame(static, [packet('ego', 'I', 2, 1, trail=['a', 'I'])])
    view = GeometryView(static, current, history, {}, 23)
    selected = view.topology.by_key[view.state['packets']['ego']['_movement_key']]
    assert selected['from_lane'] == 'a'
    assert view.internal_bindings['ego']['static_candidate_count'] == 2
    assert view.internal_bindings['ego']['compatible_body_trail_count'] == 1
    unused_noise = ScenarioNoise(23, {'ego'}, 7)
    assert view.noise._packets['ego']['intent_rng'].getstate() == unused_noise._packets['ego']['intent_rng'].getstate()
