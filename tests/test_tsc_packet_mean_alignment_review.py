"""Independent nonzero local responses and closed-state forward comparisons."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_mean_alignment_20261110.kernel import PacketKernel
from experiments.tsc_packet_mean_alignment_20261110.mean import conditional_motion_mean
from experiments.tsc_packet_mean_alignment_20261110.motion import resolve_forward
from experiments.tsc_packet_same_step_receiving_20261110.motion import build_context
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    coefficients, connection, corridor, executor_seed, initial_observation, lane,
    laws, packet, road_graph,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from test_tsc_packet_spatial_forward_review import crossing_kernel


def make_kernel(static, target_laws, packets, seeds):
    initial = initial_observation(static, packets, executors=seeds, time_s=1)
    kernel = PacketKernel(static, initial, seeds, target_laws, coefficients(), random_seed=0,
                        conflict_geometry=build_entity_geometry(RoadTopology(static), 2.))
    kernel._connections = {key: kernel._connection(item) for key, item in kernel.state['packets'].items()}
    kernel._decision_uniforms = {key: {'start': 0., 'yield': 0.} for key in kernel.state['packets']}
    return kernel


def exact_local_and_closed_forward(kernel, focal_id, residual):
    context = build_context(kernel)
    connection, compatible = kernel._connections[focal_id]
    local = conditional_motion_mean(context, kernel, focal_id, connection, compatible,
                                    residual_mps=residual)
    decisions = {key: dict(motion_residual_mps=residual if key == focal_id else 0.,
                           start_probability=1., yield_probability=1.)
                 for key in kernel.state['packets']}
    old = deepcopy(kernel.state['packets'])
    moved, exited, unused = resolve_forward(kernel, old, decisions)
    assert not exited
    assert local['mean_mps'] == pytest.approx(moved[focal_id]['speed_mps'])
    return local


def test_active_body_cap_is_applied_after_negative_residual_in_both_operators():
    static = road_graph([
        lane('in', 20), lane('I0', 10, internal=True), lane('out', 5),
        lane('I1', 5, internal=True), lane('after', 100),
    ], [connection('in', 'out', 'I0', tls='J', link=0), connection('I0', 'out'),
        connection('out', 'after', 'I1', tls='K', link=0), connection('I1', 'after')])
    target_laws = laws(static, intents={'in': {'out': 1.}, 'out': {'after': 1.}, 'after': {'END': 1.}})
    seeds = {'J': executor_seed(clearance=('I0',)),
             'K': executor_seed(current='r', mode='all_red', remaining=5., clearance=('I1',))}
    kernel = make_kernel(static, target_laws, [
        packet('ego', 'I0', 2, 7, trail=['in', 'I0']), packet('receiver', 'out', 5, 12),
    ], seeds)
    result = exact_local_and_closed_forward(kernel, 'ego', -2.)
    # P=10, actual downstream C=6: min(C,P-2)=6, whereas old U-2 gave4.
    assert result['physical_ceiling_mps'] == pytest.approx(10.)
    assert result['mean_mps'] == pytest.approx(6.)


def test_negative_residual_without_an_active_shared_cap_retains_nominal_motion():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    seeds = {'J': executor_seed(clearance=('I0',))}
    kernel = make_kernel(static, target_laws, [packet('ego', 'I0', 2, 7, trail=['in', 'I0'])], seeds)
    result = exact_local_and_closed_forward(kernel, 'ego', -2.)
    assert result['physical_ceiling_mps'] == pytest.approx(10.)
    assert result['mean_mps'] == pytest.approx(8.)


def test_negative_residual_can_avoid_time_overlap_and_increase_realized_travel():
    original = crossing_kernel([
        packet('primary', 'Ia', 10, 15), packet('ego', 'Ib', 10, 4),
    ], starts=True)
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    kernel = make_kernel(original.static, original.laws, original.state['packets'].values(), seeds)
    zero = exact_local_and_closed_forward(kernel, 'ego', 0.)
    delayed = exact_local_and_closed_forward(kernel, 'ego', -3.)
    # Zone starts3m ahead, and the priority foe clears at12/18=2/3s.
    # Ego7 reaches it too early; ego4 reaches it at3/4s and can enter.
    assert zero['physical_ceiling_mps'] == pytest.approx(7.)
    assert zero['mean_mps'] == pytest.approx(3., abs=1e-7)
    assert delayed['mean_mps'] == pytest.approx(4.)
    assert delayed['mean_mps'] > zero['mean_mps']
