"""Conditional same-second receiving projections; no native records or fitting."""
from copy import deepcopy

from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import corridor, executor_seed, packet
from experiments.tsc_target_packet_training_20261109 import conditional_features as seed_context
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_spatial_conflict_20261110 import conditional_features as frozen
from experiments.tsc_packet_same_step_receiving_20261110.conditional_features import (
    _admission_detail, prepare_receiving_context,
)
from test_tsc_packet_spatial_forward_review import crossing_kernel


def context(packets):
    static, laws = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    laws['intent_probs']['in'] = {'out': .9, 'END': .1}
    seeds = {'J': executor_seed(('g',), clearance=('I0',))}
    packets = {item['packet_id']: deepcopy(item) for item in packets}
    frame = dict(time_s=0, packets=packets, native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(row['snapshot'])) for tls, row in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    kernel = seed_context.prepare_feature_kernel(static, frame, seeds, laws, {'J': 'g'}, 0)
    configure_geometry(kernel, build_entity_geometry(kernel.topology, 2.))
    prepare_receiving_context(kernel)
    return kernel


def query(kernel, packet_id, intent):
    focal = kernel.state['packets'][packet_id]
    previous = deepcopy(kernel.state['intents'].get(packet_id))
    old_connection = kernel._connections[packet_id]
    try:
        geometry = kernel.topology.lanes[focal['lane_id']]
        if not geometry['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=geometry['edge_id'], choice=intent)
        connection, compatible = kernel._connection(focal)
        kernel._connections[packet_id] = connection, compatible
        view = seed_context._context_view(kernel, focal, connection, compatible)
        return _admission_detail(kernel, focal, connection, compatible, view)
    finally:
        if previous is None:
            kernel.state['intents'].pop(packet_id, None)
        else:
            kernel.state['intents'][packet_id] = previous
        kernel._connections[packet_id] = old_connection


def test_resolved_moving_receiver_opens_space_without_observed_future_input():
    kernel = context([packet('ego', 'I0', 5, 5), packet('leader', 'out', 5, 6)])
    focal = kernel.state['packets']['ego']
    connection, compatible = kernel._connections['ego']
    view = seed_context._context_view(kernel, focal, connection, compatible)
    before = frozen._admission_detail(kernel, focal, connection, compatible, view)
    after = query(kernel, 'ego', 'out')
    assert before['path_bound_m'] == 3
    assert after['current_path_bound_m'] == 3
    assert after['upper'] > before['upper']
    assert after['path_bound_m'] > after['current_path_bound_m']
    assert after['receiving_release_m'] > 0
    assert after['upper'] <= after['physical_proposal_mps']


def test_receiver_constrained_by_its_own_front_body_cannot_release_free_proposal_room():
    kernel = context([packet('ego', 'I0', 5, 5), packet('leader', 'out', 5, 0),
                      packet('blocker', 'out', 10, 0)])
    after = query(kernel, 'ego', 'out')
    assert after['current_path_bound_m'] == 3
    assert after['upper'] <= 3
    assert after['receiving_release_m'] <= 1e-8


def test_candidate_order_does_not_reuse_another_ego_paths_resolution_or_touch_old_cache():
    packets = [packet('ego', 'in', 19, 5), packet('leader', 'out', 5, 6)]
    first, second = context(packets), context(packets)
    before_state = deepcopy(first.state)
    old_cache = deepcopy(first._conditional_zero_speeds)
    before_rng = {key: (value['intent_rng'].getstate(), value['decision_rng'].getstate())
                  for key, value in first.noise._packets.items()}
    out_first = query(first, 'ego', 'out')
    end_second = query(first, 'ego', 'END')
    end_first = query(second, 'ego', 'END')
    out_second = query(second, 'ego', 'out')
    assert out_first == out_second and end_first == end_second
    assert end_first['upper'] <= 1
    assert first.state == before_state
    assert all(first._conditional_zero_speeds[key] == value for key, value in old_cache.items())
    assert before_rng == {key: (value['intent_rng'].getstate(), value['decision_rng'].getstate())
                          for key, value in first.noise._packets.items()}


def test_receiver_tail_priority_uses_its_old_movement_when_current_road_intent_is_end():
    proto = crossing_kernel([packet('ego', 'b', 19, 5),
        packet('tail', 'out_a', 3, 0, trail=['Ia', 'out_a']), packet('pinner', 'out_a', 8, 0)])
    for lane, shape in {'a': '-50,0 -30,0', 'Ia': '-30,0 0,0', 'out_a': '0,0 100,0'}.items():
        proto.static['lanes'][lane]['shape'] = shape
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    frame = dict(time_s=0, packets=deepcopy(proto.state['packets']),
        native_lanes=deepcopy(proto.state['lane_sensors']), native_pending_ids=[],
        executors={tls: dict(snapshot=deepcopy(row)) for tls, row in proto.state['executors'].items()},
        unresolved=[], checks=dict(synthetic=True))
    kernel = seed_context.prepare_feature_kernel(proto.static, frame, seeds, proto.laws, {'J': 'Gg'}, 0)
    configure_geometry(kernel, build_entity_geometry(kernel.topology, 2.))
    prepare_receiving_context(kernel)
    assert kernel._connections['tail'] == (None, True)
    before_state = deepcopy(kernel.state)
    result = query(kernel, 'ego', 'out_b')
    assert result['active_conflict']
    assert result['resolved_packet_count'] >= 2
    assert kernel.state == before_state


def test_initial_end_focal_is_added_to_conditioned_local_tls_without_changing_fixed_index(monkeypatch):
    from experiments.tsc_packet_same_step_receiving_20261110 import motion
    proto = crossing_kernel([packet('primary', 'a', 19, 5), packet('ego', 'b', 19, 5)])
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    frame = dict(time_s=0, packets=deepcopy(proto.state['packets']),
        native_lanes=deepcopy(proto.state['lane_sensors']), native_pending_ids=[],
        executors={tls: dict(snapshot=deepcopy(row)) for tls, row in proto.state['executors'].items()},
        unresolved=[], checks=dict(synthetic=True))
    kernel = seed_context.prepare_feature_kernel(proto.static, frame, seeds, proto.laws, {'J': 'Gg'}, 0)
    configure_geometry(kernel, build_entity_geometry(kernel.topology, 2.))
    kernel.state['intents']['ego'] = dict(road_id='b', choice='END')
    kernel._connections['ego'] = (None, True)
    fixed = prepare_receiving_context(kernel)
    assert 'ego' not in fixed['tls_index']['J']
    old = deepcopy(kernel.state)
    seen = []
    original = motion.conflict_limit
    def inspect_local(k, focal, connection, compatible, speed, packets, **kwargs):
        seen.append('ego' in packets)
        return original(k, focal, connection, compatible, speed, packets, **kwargs)
    monkeypatch.setattr(motion, 'conflict_limit', inspect_local)
    try:
        kernel.state['intents']['ego'] = dict(road_id='b', choice='out_b')
        kernel._connections['ego'] = kernel._connection(kernel.state['packets']['ego'])
        resolver = motion.MotionResolver(fixed, kernel, fixed['zero_decisions'], fixed['permit_all'], focal_id='ego')
        resolver.spatial('primary', resolver.raw('primary'))
        assert seen == [True]
        assert 'ego' not in fixed['tls_index']['J']
    finally:
        kernel.state['intents']['ego'] = old['intents']['ego']
        kernel._connections['ego'] = (None, True)
    assert kernel.state == old
