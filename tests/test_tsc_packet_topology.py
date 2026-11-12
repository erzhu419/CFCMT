"""Physical paths and directed conflicts are local graph joins, not routes."""
from copy import deepcopy

import pytest

from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def static_graph():
    lanes = {}
    for lane, edge, index, length, internal in (
        ('a1', 'A', 1, 100., False), ('a0', 'A', 0, 100., False),
        ('i0', ':I0', 0, 4., True), ('i1', ':I1', 0, 6., True),
        ('b0', 'B', 0, 80., False), ('b1', 'B', 1, 80., False),
        ('c0', 'C', 0, 90., False), ('j0', ':J0', 0, 7., True),
        ('d0', 'D', 0, 90., False), ('e0', 'E', 0, 90., False),
        ('k0', ':K0', 0, 8., True), ('f0', 'F', 0, 90., False)):
        lanes[lane] = dict(edge_id=edge, index=index, length_m=length,
                           internal=internal, speed_limit_mps=11.11)

    def connection(origin, target, via=None, tls=None, link=None, direction='s'):
        return dict(from_lane=origin, to_lane=target, via=via, tls_id=tls,
                    link_index=link, direction=direction)

    # Three physical movements share one link and the two internal lanes.
    # Continuation rows are interspersed: XML order must remain visible.
    rows = [connection('a0', 'b0', 'i0', 'X', 3, 'r'),
            connection('i0', 'b0', 'i1'),
            connection('a1', 'b1', 'i0', 'X', 3, 'r'),
            connection('i1', 'b0'),
            connection('a0', 'b1', 'i0', 'X', 3, 'r'),
            connection('i1', 'b1'),
            connection('c0', 'd0', 'j0', 'X', 4),
            connection('e0', 'f0', 'k0', 'Y', 4),
            connection('j0', 'd0'), connection('k0', 'f0')]
    successors, predecessors = {lane: [] for lane in lanes}, {lane: [] for lane in lanes}
    for row in rows:
        following = row['via'] or row['to_lane']
        if following not in successors[row['from_lane']]:
            successors[row['from_lane']].append(following)
            predecessors[following].append(row['from_lane'])
    return dict(lanes=lanes, connections=rows, successors=successors, predecessors=predecessors,
        conflicts_by_tls={
            'X': dict(foes=[[3, 4], [4, 3]], response=[[3, 4]], full_context_complete=True),
            'Y': dict(foes=[], response=[], full_context_complete=True)})


def test_complete_multilane_path_progress_and_xml_order():
    topology = RoadTopology(static_graph())
    first = topology.connections[0]
    assert [row['key'] for row in topology.connections] == ['conn_0', 'conn_2', 'conn_4', 'conn_6', 'conn_7']
    assert first['path'] == ('a0', 'i0', 'i1', 'b0')
    assert (first['direction'], first['from_edge'], first['to_edge']) == ('r', 'A', 'B')
    assert topology.path_progress(first, 'i1', 2.) == 106.
    assert topology.internal_progress['i1'][first['key']] == 104.
    assert topology.path_distance(first['key'], 'a0', 98., 'b0', 3.) == 15.
    assert topology.path_distance(first, 'b0', 3., 'a0', 98.) == -15.
    assert [row['to_lane'] for row in topology.compatible('a0', 'B')] == ['b0', 'b1']
    assert topology.compatible('a0', 'D') == ()


def test_shared_internal_and_signal_link_keep_all_physical_movements():
    topology = RoadTopology(static_graph())
    assert [row['key'] for row in topology.internal_connections['i0']] == ['conn_0', 'conn_2', 'conn_4']
    assert {row['to_lane'] for row in topology.internal_connections['i0']} == {'b0', 'b1'}
    assert topology.metadata()['shared_internal_lanes'] == 2
    assert topology.metadata()['shared_internal_different_receivers'] == 2
    assert topology.metadata()['multiple_movements_per_link'] == 1
    assert topology.pp_lane_by_link == {'X': {3: 'a0', 4: 'c0'}, 'Y': {4: 'e0'}}
    # A getter export states which physical group[0] lane the original PP uses.
    static = static_graph()
    static['pp_lane_by_link'] = {'X': {'3': 'a1', '4': 'c0'}, 'Y': {'4': 'e0'}}
    assert RoadTopology(static).pp_lane_by_link['X'][3] == 'a1'


def test_response_direction_and_cross_tls_indices_are_not_merged():
    topology = RoadTopology(static_graph())
    a, shared, unused, conflicting, other_tls = topology.connections
    assert topology.foes(a, conflicting) and topology.foes(conflicting, a)
    assert topology.yields_to(a, conflicting) and topology.yields_to(shared, conflicting)
    assert not topology.yields_to(conflicting, a)
    assert not topology.foes(a, other_tls)
    assert not topology.yields_to(a, other_tls)


def test_lateral_neighbors_follow_lane_indices_and_exclude_internal_lanes():
    topology = RoadTopology(static_graph())
    assert topology.road_lanes('A') == ('a0', 'a1')
    assert topology.adjacent('a0', 1) == 'a1'
    assert topology.adjacent('a1', -1) == 'a0'
    assert topology.adjacent('a0', 0) == 'a0'
    assert topology.adjacent('a1', 1) is None
    assert topology.road_lanes(':I0') == ()
    assert topology.adjacent('i0', 1) is None


def test_a_whole_connection_does_not_invent_an_internal_branch():
    static = deepcopy(static_graph())
    static['successors']['i0'].append('b0')  # A second path to the same recorded receiver.
    with pytest.raises(ValueError, match='2 complete internal paths'):
        RoadTopology(static)
