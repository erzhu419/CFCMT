"""Real-shape width corridors, sampled length units, and short merge zones."""
import pytest

from experiments.tsc_packet_spatial_conflict_20261110.geometry import (
    build_conflict_geometry, pair_geometry, _segment_capsule_interval)
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import lane, connection, road_graph
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def topology(paths, lengths=None):
    geometries, connections = {}, []
    lengths = lengths or {}
    for index, chain in enumerate(paths):
        source, internal, target = f'a{index}', f'I{index}', f'out{index}'
        for key, points, is_internal in ((source, chain[0], False), (internal, chain[1], True), (target, chain[2], False)):
            default = sum(((b[0]-a[0])**2+(b[1]-a[1])**2)**.5 for a, b in zip(points, points[1:]))
            unused, geometry = lane(key, lengths.get(key, default), internal=is_internal)
            geometry.update(width_m=3.2, shape=' '.join(f'{x},{y}' for x, y in points))
            geometries[key] = geometry
        connections.extend([connection(source, target, internal, tls='J', link=index), connection(internal, target)])
    static = road_graph(list(geometries.items()), connections, foes=((0, 1), (1, 0)))
    result = RoadTopology(static)
    return result, result.connections_from('a0')[0], result.connections_from('a1')[0]


def parallel_paths(separation):
    return [([(-10, y), (0, y)], [(0, y), (10, y)], [(10, y), (410, y)])
            for y in (0., separation)]


def test_parallel_real_width_corridor_edge_contact_is_retained_without_centerline_intersection():
    net, first, second = topology(parallel_paths(3.2))
    table = build_conflict_geometry(net)
    pair = pair_geometry(table, first, second)
    assert pair['status'] == 'resolved' and pair['zones']
    assert pair['zones'][0]['ego_interval_m'] == pytest.approx([10., 23.2])
    assert pair['zones'][0]['foe_interval_m'] == pytest.approx([10., 23.2])
    swapped = pair_geometry(table, second, first)
    assert swapped['zones'][0]['ego_interval_m'] == pair['zones'][0]['foe_interval_m']


def test_receiver_only_uses_unmerged_lane_contacts_and_reverse_query_swaps_piece_ownership():
    net, first, second = topology(parallel_paths(3.2))
    table = build_conflict_geometry(net)
    pair = pair_geometry(table, first, second)
    assert pair['zones'][0]['ego_interval_m'][0] == 10.
    assert len(pair['contact_pieces']) == 4
    assert all(len(piece['lane_pairs']) == 1 for piece in pair['contact_pieces'])
    receiver_only = [piece for piece in pair['contact_pieces'] if piece['lane_pairs'][0][1] == 'out1']
    assert len(receiver_only) == 2
    assert min(piece['ego_interval_m'][0] for piece in receiver_only) == pytest.approx(20.)
    assert min(piece['foe_interval_m'][0] for piece in receiver_only) == pytest.approx(20.)
    reverse = pair_geometry(table, second, first)
    for original, swapped in zip(pair['contact_pieces'], reverse['contact_pieces']):
        assert swapped['ego_interval_m'] == original['foe_interval_m']
        assert swapped['foe_interval_m'] == original['ego_interval_m']
        a, b = original['lane_pairs'][0]
        assert swapped['lane_pairs'] == [(b, a)]
    assert min(piece['ego_interval_m'][0] for piece in receiver_only) == pytest.approx(20.)


def test_endpoint_tangency_is_a_point_zone_and_progress_uses_stored_shape_scale():
    assert _segment_capsule_interval((0, -5), (0, 0), (-5, 3.2), (0, 3.2), 3.2) == pytest.approx([5., 5.])
    paths = [([(-10, -5), (-5, -5)], [(-5, -5), (0, -5), (0, 0)], [(0, 0), (400, 0)]),
             ([(-10, 3.2), (-5, 3.2)], [(-5, 3.2), (0, 3.2)], [(0, 3.2), (400, 3.2)])]
    net, first, second = topology(paths, dict(a0=100., I0=20., a1=200., I1=10.))
    pair = pair_geometry(build_conflict_geometry(net), first, second)
    assert pair['status'] == 'resolved'
    assert len(pair['zones']) == 1
    # The internal endpoints touch at120/210; the adjacent receiving prefixes
    # also touch over their first real3.2m and are deliberately retained.
    assert pair['zones'][0]['ego_interval_m'] == pytest.approx([120., 123.2])
    assert pair['zones'][0]['foe_interval_m'] == pytest.approx([210., 213.2])
    reverse = pair_geometry(build_conflict_geometry(net), second, first)
    assert reverse['zones'][0]['ego_interval_m'] == pytest.approx([210., 213.2])
    assert reverse['zones'][0]['foe_interval_m'] == pytest.approx([120., 123.2])


def test_asymmetric_crossing_uses_both_lane_half_widths_and_each_direction_progress():
    paths = [([(-25, 0), (-15, 0)], [(-15, 0), (15, 0)], [(15, 0), (415, 0)]),
             ([(5, -20), (5, -10)], [(5, -10), (5, 40)], [(5, 40), (5, 440)])]
    net, first, second = topology(paths)
    table = build_conflict_geometry(net)
    pair = pair_geometry(table, first, second)
    assert pair['zones'][0]['ego_interval_m'] == pytest.approx([26.8, 33.2])
    assert pair['zones'][0]['foe_interval_m'] == pytest.approx([16.8, 23.2])
    reverse = pair_geometry(table, second, first)
    assert reverse['zones'][0]['ego_interval_m'] == pytest.approx([16.8, 23.2])
    assert reverse['zones'][0]['foe_interval_m'] == pytest.approx([26.8, 33.2])


def test_shared_receiving_zone_is_only_one_real_lane_width_not_400m_following_suffix():
    lanes = [lane('a', 10), lane('b', 10), lane('Ia', 5, internal=True), lane('Ib', 5, internal=True), lane('out', 400)]
    static = road_graph(lanes, [connection('a', 'out', 'Ia', tls='J', link=0),
        connection('b', 'out', 'Ib', tls='J', link=1), connection('Ia', 'out'), connection('Ib', 'out')],
        foes=((0, 1), (1, 0)))
    shapes = {'a': '0,-15 0,-5', 'b': '-15,0 -5,0', 'Ia': '0,-5 0,0', 'Ib': '-5,0 0,0', 'out': '0,0 400,0'}
    for key, geometry in static['lanes'].items():
        geometry.update(width_m=3.2, shape=shapes[key])
    net = RoadTopology(static)
    first, second = net.connections_from('a')[0], net.connections_from('b')[0]
    pair = pair_geometry(build_conflict_geometry(net), first, second)
    assert pair['status'] == 'resolved'
    assert pair['first_receiving_prefix_m'] == 3.2
    assert max(zone['ego_interval_m'][1] for zone in pair['zones']) == pytest.approx(18.2)
    assert max(zone['foe_interval_m'][1] for zone in pair['zones']) == pytest.approx(18.2)


@pytest.mark.parametrize('missing', ['width_m', 'shape'])
def test_unknown_real_width_or_shape_keeps_pair_unresolved(missing):
    net, first, second = topology(parallel_paths(0.))
    net.lanes['I0'][missing] = None
    pair = pair_geometry(build_conflict_geometry(net), first, second)
    assert pair['status'] == 'unresolved' and pair['zones'] == []


def test_valid_disjoint_corridors_can_explicitly_have_no_zone():
    net, first, second = topology(parallel_paths(20.))
    pair = pair_geometry(build_conflict_geometry(net), first, second)
    assert pair['status'] == 'no_conflict_zone' and pair['zones'] == []
