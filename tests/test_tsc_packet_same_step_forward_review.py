"""Independent shared-forward receiving regressions; synthetic records only."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_same_step_receiving_20261110.kernel import PacketKernel
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    coefficients, connection, corridor, executor_seed, initial_observation, lane,
    laws, packet, road_graph,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from test_tsc_packet_spatial_forward_review import crossing_kernel


def kernel_for(static, target_laws, packets, *, seeds=None):
    seeds = seeds or {'J': executor_seed(clearance=tuple(static['internal_lane_ids']))}
    state = initial_observation(static, packets, executors=seeds, time_s=1)
    table = build_entity_geometry(RoadTopology(static), 2.)
    return PacketKernel(static, state, seeds, target_laws, coefficients(),
                        random_seed=0, conflict_geometry=table)


def test_internal_follower_uses_room_released_by_later_ranked_receiving_vehicle():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    kernel = kernel_for(static, target_laws, [
        packet('ego', 'I0', 5, 5), packet('receiver', 'out', 5, 8),
    ])
    assert kernel.step()['ledger_passed']
    ego, receiver = (kernel.state['packets'][key] for key in ('ego', 'receiver'))
    # The original current receiver-tail constraint allowed only 3m. Resolving
    # the ordinary receiver first must release actual space to its internal follower.
    assert ego['speed_mps'] > 3
    assert receiver['position_m'] > 5
    if ego['lane_id'] == 'out':
        assert receiver['position_m']-5-ego['position_m'] >= 2


def test_downstream_queue_prevents_optimistic_release_from_receiver_free_proposal():
    static = road_graph([
        lane('in', 20), lane('I0', 10, internal=True), lane('out', 12),
        lane('I1', 5, internal=True), lane('after', 100),
    ], [connection('in', 'out', 'I0', tls='J', link=0), connection('I0', 'out'),
        connection('out', 'after', 'I1', tls='K', link=0), connection('I1', 'after')])
    target_laws = laws(static, intents={'in': {'out': 1.}, 'out': {'after': 1.}, 'after': {'END': 1.}})
    seeds = {'J': executor_seed(clearance=('I0',)),
             'K': executor_seed(current='r', mode='all_red', remaining=5., clearance=('I1',))}
    kernel = kernel_for(static, target_laws, [
        packet('ego', 'I0', 5, 5), packet('receiver', 'out', 5, 8),
        packet('blocker', 'out', 12, 8),
    ], seeds=seeds)
    assert kernel.step()['ledger_passed']
    ego, receiver, blocker = (kernel.state['packets'][key]
                              for key in ('ego', 'receiver', 'blocker'))
    assert blocker['position_m'] == 12 and blocker['speed_mps'] == 0
    assert receiver['position_m'] == 5
    assert ego['lane_id'] == 'I0' and ego['position_m'] == pytest.approx(8.)
    assert ego['speed_mps'] == pytest.approx(3.)


def test_front_outside_follower_path_still_releases_only_its_realized_body_tail():
    static = road_graph([
        lane('in', 20), lane('I0', 10, internal=True), lane('out', 10), lane('after', 100),
    ], [connection('in', 'out', 'I0', tls='J', link=0),
        connection('I0', 'out'), connection('out', 'after')])
    target_laws = laws(static, intents={'in': {'out': 1.}, 'out': {'after': 1.}, 'after': {'END': 1.}})
    kernel = kernel_for(static, target_laws, [
        packet('ego', 'I0', 5, 12),
        packet('receiver', 'after', 1, 8, trail=['out', 'after']),
    ])
    before = deepcopy(kernel.state['packets'])
    assert kernel.step()['ledger_passed']
    ego, receiver = (kernel.state['packets'][key] for key in ('ego', 'receiver'))
    assert before['receiver']['body_trail'] == ['out', 'after']
    assert receiver['body_trail'] == ['after']
    # The original out-lane tail occupied 6..10, limiting the ego to 9m.
    # The moved body has really cleared out; finite committed-path end remains 15m.
    assert ego['speed_mps'] > 9
    assert ego['lane_id'] == 'out' and ego['position_m'] <= 10


def test_shared_resolver_retains_directed_priority_and_processed_zone_reservation():
    original = crossing_kernel([
        packet('secondary', 'Ib', 10, 15), packet('primary', 'Ia', 10, 15),
    ], starts=True)
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    kernel = PacketKernel(original.static, deepcopy(original.state), seeds, original.laws,
                        coefficients(), random_seed=0,
                        conflict_geometry=build_entity_geometry(original.topology, 2.))
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['primary']['position_m'] > 22
    # The priority vehicle has cleared by the endpoint, while its actual
    # within-step crossing still blocks the overlapping secondary movement.
    assert kernel.state['packets']['secondary']['position_m'] == pytest.approx(13., abs=1e-7)


def test_free_proposal_changes_zone_timing_without_flipping_frozen_fifo_priority():
    from experiments.tsc_packet_same_step_receiving_20261110.conflict import conflict_limit

    original = crossing_kernel([
        packet('ego', 'Ib', 10, 15), packet('foe', 'Ia', 10, 15),
    ], starts=True)
    static = deepcopy(original.static)
    static['conflicts_by_tls']['J']['response'] = []
    seeds = {'J': executor_seed(('GG',), clearance=('Ia', 'Ib'))}
    state = initial_observation(static, original.state['packets'].values(), executors=seeds)
    kernel = PacketKernel(static, state, seeds, original.laws, coefficients(), random_seed=0,
                        conflict_geometry=build_entity_geometry(RoadTopology(static), 2.))
    packets = kernel.state['packets']
    kernel._connections = {key: kernel._connection(value) for key, value in packets.items()}
    connection, compatible = kernel._connections['ego']
    # With equal remaining distances, old FIFO forecasts3 vs10 give the foe
    # precedence. New ego proposal18 must change its region ETA, not that priority.
    result = conflict_limit(kernel, packets['ego'], connection, compatible, 18., packets,
                            speeds={'ego': 18., 'foe': 10.},
                            priority_speeds={'ego': 3., 'foe': 10.})
    assert result['active_conflict']
    assert result['limit_m'] == pytest.approx(3., abs=1e-7)
    region = result['details'][0]['active_zones'][0]
    assert region['ego_time']['start_sec'] == pytest.approx(3./18.)
