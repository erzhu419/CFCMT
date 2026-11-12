"""Fixed invented states for law checks; no native records or simulator."""
from copy import deepcopy

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_feature_kernel
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    corridor, executor_seed, lane, laws, packet, parallel_receivers, road_graph, connection,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def vehicle(key, lane_id, position, speed):
    row = packet(key, lane_id, position, speed)
    row['parameters'].update(decel_mps2=4.5, min_gap_m=2.5)
    return row


def make_kernel(static, rows, priors=None, signal=None):
    history = laws(static) if priors is None else priors
    seeds = {} if signal is None else {'J': executor_seed((signal,), clearance=tuple(static['internal_lane_ids']))}
    packets = {row['packet_id']: deepcopy(row) for row in rows}
    frame = dict(time_s=1, packets=packets, native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(seed['snapshot'])) for tls, seed in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    return prepare_feature_kernel(static, frame, seeds, history, {}, 23,
        conflict_geometry=build_entity_geometry(RoadTopology(static), 2.))


def cases():
    single = road_graph([lane('a', 100)], [])
    out = dict(stopped_open=(make_kernel(single, [vehicle('ego', 'a', 20, 0)]), 'ego'),
        blocked_start=(make_kernel(single, [vehicle('ego', 'a', 20, 0), vehicle('leader', 'a', 27.5, 0)]), 'ego'))
    graph, history = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    out['yielding_entry'] = make_kernel(graph, [vehicle('ego', 'in', 19, 5)], history, 'g'), 'ego'
    out['blocked_entry'] = make_kernel(graph, [vehicle('ego', 'in', 19, 5), vehicle('leader', 'I0', 7.4, 0)], history, 'g'), 'ego'
    lateral = road_graph([lane('a0', 100, edge='a', index=0), lane('a1', 100, edge='a', index=1),
                          lane('a2', 100, edge='a', index=2)], [])
    out['two_actor_lateral'] = make_kernel(lateral, [vehicle('focal', 'a0', 20, 5), vehicle('front', 'a1', 24, 5)]), 'focal'
    graph, history = corridor(incoming_length=10, internal_lengths=(10,), outgoing_length=50)
    history['intent_probs']['in'] = {'out': .4, 'END': .6}
    out['intent_and_survival'] = make_kernel(graph, [vehicle('ego', 'in', 9, 1)], history, 'G'), 'ego'
    graph, unused = parallel_receivers()
    graph = road_graph(list(graph['lanes'].items())+[lane('next', 100)],
        graph['connections']+[connection('out1', 'next')])
    history = laws(graph, intents={'a': {'out': 1.}, 'out': {'next': .3, 'END': .7}, 'next': {'END': 1.}})
    out['receiving_intent'] = make_kernel(graph, [vehicle('ego', 'I1', 7, 10)], history, 'GG'), 'ego'
    return out
