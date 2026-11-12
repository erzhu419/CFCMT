"""Independent synthetic receiving tests; no native records or fitting."""
from copy import deepcopy

import pytest

from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    coefficients, connection, corridor, executor_seed, initial_observation,
    lane, laws, packet, road_graph,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109 import conditional_features as old_conditional
from experiments.tsc_packet_receiving_path_20261110 import conditional_features as conditional
from experiments.tsc_packet_receiving_path_20261110.kernel import PacketKernel
from experiments.tsc_packet_receiving_path_20261110.receiving import admissible_travel_m


def _frame(static, packets, seeds):
    packets = {p['packet_id']: deepcopy(p) for p in packets}
    return dict(time_s=0, packets=packets, native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(row['snapshot'])) for tls, row in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))


def test_receiver_front_overhang_limits_both_approach_and_internal_packet_travel():
    static, unused = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    topology = RoadTopology(static)
    receiver = packet('receiver', 'out', 3, 0, length=5, trail=['I0', 'out'])
    approach = packet('ego', 'in', 19, 2)
    # Receiver rear is at internal position8. Ego is1m before the entry,
    # and its own minimum gap is2m: 1+8-2=7m of front travel.
    assert admissible_travel_m(topology, approach, {'ego': approach, 'receiver': receiver},
                               ('in', 'I0', 'out')) == pytest.approx(7.)
    internal = packet('ego', 'I0', 5, 10)
    assert admissible_travel_m(topology, internal, {'ego': internal, 'receiver': receiver},
                               ('I0', 'out')) == pytest.approx(1.)
    # Partial occupancy is not packet mass: the front remains only in out.
    sensors = PacketObserver(static)._reconstruct({'receiver': receiver})
    assert sensors['I0']['N'] == 0 and sensors['I0']['O'] == pytest.approx(.2)


def test_head_beyond_complete_path_still_blocks_by_its_tail_in_helper_conditional_and_forward():
    static = road_graph(
        [lane('in', 20), lane('I0', 5, internal=True), lane('out', 10), lane('after', 100)],
        [connection('in', 'out', 'I0', tls='J', link=0), connection('I0', 'out'), connection('out', 'after')])
    target_laws = laws(static, intents={'in': {'out': 1.}, 'out': {'after': 1.}, 'after': {'END': 1.}})
    seeds = {'J': executor_seed(('G',), clearance=('I0',))}
    ego = packet('ego', 'I0', 2, 10, trail=['in', 'I0'])
    ahead = packet('ahead', 'after', 2, 0, length=8, trail=['out', 'after'])
    behind = packet('behind', 'in', 5, 0)
    packets = {p['packet_id']: p for p in (ego, ahead, behind)}
    topology = RoadTopology(static)
    # The front already left (I0,out); six metres of its body remain on
    # out, whose rear is4m from its start: 3+4-2=5m from ego.
    assert admissible_travel_m(topology, ego, packets, ('I0', 'out')) == pytest.approx(5.)
    assert admissible_travel_m(topology, ego, {'ego': ego, 'behind': behind}, ('I0', 'out')) == 13
    frame = _frame(static, list(packets.values()), seeds)
    feature_kernel = conditional.prepare_feature_kernel(static, frame, seeds, target_laws, {'J': 'G'}, 0)
    feature = conditional.candidate_features(feature_kernel, 'ego', 'out')
    assert feature['admission_detail']['path_bound_m'] == pytest.approx(5.)
    assert feature['upper'] == pytest.approx(5.)
    kernel = PacketKernel(static, initial_observation(static, list(packets.values()), executors=seeds),
                          seeds, target_laws, coefficients(start=0), random_seed=0)
    first = kernel.step()
    assert first['ledger_passed']
    assert kernel.state['packets']['ego']['lane_id'] == 'out'
    assert kernel.state['packets']['ego']['position_m'] == pytest.approx(2.)
    # Its front is now exactly2m before the other body's rear, not through it.
    second = kernel.step()
    assert second['ledger_passed']
    assert kernel.state['packets']['ego']['position_m'] == pytest.approx(2.)


def test_shared_admission_can_enter_internal_but_cannot_enter_when_long_tail_fills_the_path():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    seeds = {'J': executor_seed(('g',), clearance=('I0',))}
    packets = [packet('ego', 'in', 19, 2), packet('receiver', 'out', 3, 0, trail=['I0', 'out'])]
    frame = _frame(static, packets, seeds)
    old = old_conditional.prepare_feature_kernel(static, frame, seeds, target_laws, {'J': 'g'}, 0)
    assert old_conditional.candidate_features(old, 'ego', 'out')['yield_risk'] is False
    new = conditional.prepare_feature_kernel(static, frame, seeds, target_laws, {'J': 'g'}, 0)
    feature = conditional.candidate_features(new, 'ego', 'out')
    assert feature['yield_risk'] is True and feature['upper'] > 1
    kernel = PacketKernel(static, initial_observation(static, packets, executors=seeds), seeds, target_laws,
                          coefficients(start=0), random_seed=0)
    row = kernel.step()
    assert row['ledger_passed']
    assert kernel.state['packets']['ego']['lane_id'] == 'I0'
    assert kernel.state['packets']['ego']['position_m'] <= 6  # Rear at8 minus own minGap2.
    assert kernel.state['packets']['receiver']['position_m'] == 3

    short, short_laws = corridor(incoming_length=20, internal_lengths=(2, 2), outgoing_length=30)
    short_seeds = {'J': executor_seed(('g',), clearance=('I0', 'I1'))}
    blocked_packets = [packet('ego', 'in', 15, 1),
        packet('long', 'out', 1, 0, length=8, trail=['in', 'I0', 'I1', 'out'])]
    blocked_frame = _frame(short, blocked_packets, short_seeds)
    blocked_features = conditional.prepare_feature_kernel(short, blocked_frame, short_seeds, short_laws, {'J': 'g'}, 0)
    blocked = conditional.candidate_features(blocked_features, 'ego', 'out')
    assert blocked['admission_detail']['path_bound_m'] == 0
    assert blocked['yield_risk'] is False
    stopped = PacketKernel(short, initial_observation(short, blocked_packets, executors=short_seeds),
        short_seeds, short_laws, coefficients(start=0), random_seed=0)
    assert stopped.step()['ledger_passed']
    assert stopped.state['packets']['ego']['lane_id'] == 'in'
    assert stopped.state['packets']['ego']['position_m'] == 15
    # Native fractional occupancy can exceed one; the path bound used the
    # clipped physical body, rather than adding that accounting quantity.
    assert stopped.state['lane_sensors']['I1']['O'] > 1
