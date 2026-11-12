"""Shared discrete physical formula, residual order and frozen path semantics."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_discrete_braking_20261110 import conditional_features as features
from experiments.tsc_packet_discrete_braking_20261110.kernel import PacketKernel
from experiments.tsc_packet_discrete_braking_20261110.mean import conditional_motion_mean, make_conditional_resolver
from experiments.tsc_packet_discrete_braking_20261110.motion import resolve_forward
from experiments.tsc_packet_discrete_braking_20261110.physics import ZERO, physical_proposal
from experiments.tsc_packet_same_step_receiving_20261110.motion import physical_proposal as continuous_proposal
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import corridor, executor_seed, packet
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def context(packets, signal='G'):
    static, laws = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    seeds = {'J': executor_seed((signal,), clearance=('I0',))}
    packets = {row['packet_id']: deepcopy(row) for row in packets}
    frame = dict(time_s=1, packets=packets, native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(seed['snapshot'])) for tls, seed in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    table = build_entity_geometry(RoadTopology(static), 2.)
    return features.prepare_feature_kernel(static, frame, seeds, laws, {}, 23, conflict_geometry=table)


def full_travel(kernel, key, residual):
    decisions = {pid: dict(motion_residual_mps=residual if pid == key else 0.,
        start_probability=1., yield_probability=1.) for pid in kernel.state['packets']}
    kernel._decision_uniforms = {pid: dict(start=0., **{'yield': 0.}) for pid in decisions}
    resolve_forward(kernel, kernel.state['packets'], decisions)
    return kernel._last_receiving_resolution['results'][key]['travel_m']


def test_new_following_formula_reaches_all_physical_global_and_cache_bindings():
    kernel = context([packet('ego', 'in', 5, 15), packet('leader', 'out', 12, 0)])
    p = kernel.state['packets']['ego']
    connection, compatible = kernel._connections['ego']
    packets = kernel.state['packets']
    assert continuous_proposal(kernel, p, ZERO, packets, connection, compatible) == 12.
    expected = 13.5
    assert physical_proposal(kernel, p, ZERO, packets, connection, compatible) == expected
    assert kernel._speed(p, ZERO, packets, connection, compatible) == expected
    ctx = kernel._mean_alignment_context
    assert ctx['zero_raw_cache']['ego'] == expected
    resolver = make_conditional_resolver(ctx, kernel, 'ego', connection, compatible)
    assert resolver.raw('ego') == resolver['ego'] == expected
    result = conditional_motion_mean(ctx, kernel, 'ego', connection, compatible, -2.)
    assert result['physical_ceiling_mps'] == expected
    assert result['mean_mps'] == expected-2.
    assert full_travel(kernel, 'ego', -2.) == result['mean_mps']


def test_active_stop_uses_discrete_inverse_but_yellow_predicate_stays_continuous():
    for signal, speed, expected in (('r', 5., 1.), ('Y', 5., 8.), ('y', 1., 1.)):
        kernel = context([packet('ego', 'in', 19, speed)], signal=signal)
        p = kernel.state['packets']['ego']
        connection, compatible = kernel._connections['ego']
        value = physical_proposal(kernel, p, ZERO, kernel.state['packets'], connection, compatible)
        assert value == expected
        if signal == 'r':
            assert continuous_proposal(kernel, p, ZERO, kernel.state['packets'], connection, compatible) < value
    # Keep the original <= branch at the exact continuous stopping threshold.
    kernel = context([packet('ego', 'in', 18, 4)], signal='Y')
    p = kernel.state['packets']['ego']
    connection, compatible = kernel._connections['ego']
    assert physical_proposal(kernel, p, ZERO, kernel.state['packets'], connection, compatible) == 2.


def test_nonzero_residual_precedes_body_projection_and_matches_closed_forward():
    kernel = context([packet('ego', 'I0', 5, 7), packet('leader', 'out', 8, 12),
                      packet('blocker', 'out', 13, 0)])
    p = kernel.state['packets']['ego']
    connection, compatible = kernel._connections['ego']
    ctx = kernel._mean_alignment_context
    assert physical_proposal(kernel, p, ZERO, kernel.state['packets'], connection, compatible) == 10.
    result = conditional_motion_mean(ctx, kernel, 'ego', connection, compatible, -2.)
    assert result['local_detail']['physical_proposal_mps'] == 8.
    assert result['mean_mps'] == pytest.approx(6.)
    assert full_travel(kernel, 'ego', -2.) == result['mean_mps']


def test_candidate_queries_restore_state_and_use_the_new_context_physical_proposal():
    kernel = context([packet('ego', 'in', 5, 15), packet('leader', 'out', 12, 0)])
    old_state, old_connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    old_rng = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
               for key, row in kernel.noise._packets.items()}
    out = features.candidate_features(kernel, 'ego', 'out')
    features.candidate_features(kernel, 'ego', 'END')
    assert out == features.candidate_features(kernel, 'ego', 'out')
    assert out['physical_ceiling_mps'] == out['mean0_mps'] == 13.5
    assert len(kernel.model.features(out['context'])) == 38
    assert kernel.state == old_state and kernel._connections == old_connections
    assert old_rng == {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
                       for key, row in kernel.noise._packets.items()}


def test_new_whole_step_preserves_the_processed_incompatible_connection_correction():
    from test_tsc_packet_mean_alignment_features import incompatible_kernel
    from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import coefficients
    proto = incompatible_kernel()
    initial = deepcopy(proto.state)
    initial['time_sec'] = 1.
    initial['intents']['ego'] = dict(road_id='a', choice='out_a')
    kernel = PacketKernel(proto.static, initial, {'J': executor_seed(('gg',))}, proto.laws,
        coefficients(start=True, yielding=True, lateral='stay'), random_seed=23,
        conflict_geometry=proto._spatial_geometry)
    assert kernel.step()['ledger_passed']
    result = kernel._last_receiving_resolution
    assert result['priority_order'].index('ego') < result['priority_order'].index('neighbour')
    assert not result['processed']['ego']['committed'] and not result['processed']['ego']['accepted']
    assert result['processed']['ego']['connection']['from_lane'] == 'a1'
    assert kernel.state['packets']['ego']['lane_id'] == 'a0'
    assert kernel.state['packets']['neighbour']['position_m'] > 12.
