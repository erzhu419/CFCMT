"""Explicit entity width changes only the static swept-path inflation."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_entity_footprint_20261110.geometry import (
    build_entity_geometry, pair_geometry, geometry_metadata)
from experiments.tsc_packet_spatial_conflict_20261110.geometry import build_conflict_geometry
from test_tsc_packet_spatial_geometry import topology, parallel_paths


def test_two_metre_entities_on_adjacent_3_2_metre_lanes_do_not_conflict_or_change_lane_prefixes():
    net, first, second = topology(parallel_paths(3.2))
    original = deepcopy(net.lanes)
    lane = pair_geometry(build_conflict_geometry(net), first, second)
    table = build_entity_geometry(net, 2.0)
    entity = pair_geometry(table, first, second)
    assert lane['status'] == 'resolved' and entity['status'] == 'no_conflict_zone'
    assert entity['zones'] == [] and entity['contact_pieces'] == []
    assert entity['first_receiving_prefix_m'] == entity['second_receiving_prefix_m'] == 3.2
    assert net.lanes == original
    metadata = geometry_metadata(table)
    assert metadata['entity_width_m'] == 2.0
    assert metadata['coverage']['minimum_width_m'] == metadata['coverage']['maximum_width_m'] == 3.2


def test_true_crossing_retains_entity_zones_and_ordered_progress():
    paths = [([(-25, 0), (-15, 0)], [(-15, 0), (15, 0)], [(15, 0), (415, 0)]),
             ([(5, -20), (5, -10)], [(5, -10), (5, 40)], [(5, 40), (5, 440)])]
    net, first, second = topology(paths)
    table = build_entity_geometry(net, 2.0)
    pair = pair_geometry(table, first, second)
    assert pair['status'] == 'resolved' and pair['contact_pieces']
    assert pair['zones'][0]['ego_interval_m'] == pytest.approx([28., 32.])
    assert pair['zones'][0]['foe_interval_m'] == pytest.approx([18., 22.])
    reverse = pair_geometry(table, second, first)
    assert reverse['zones'][0]['ego_interval_m'] == pytest.approx([18., 22.])
    assert reverse['contact_pieces'][0]['lane_pairs'] == [(b, a) for a, b in pair['contact_pieces'][0]['lane_pairs']]


def test_equal_explicit_entity_and_lane_width_reproduce_all_original_geometry_without_tuning():
    net, first, second = topology(parallel_paths(3.2))
    old = build_conflict_geometry(net)
    same = build_entity_geometry(net, 3.2)
    assert same['status_counts'] == old['status_counts']
    for key, pair in old['pairs'].items():
        assert same['pairs'][key]['zones'] == pair['zones']
        assert same['pairs'][key]['contact_pieces'] == pair['contact_pieces']
    assert pair_geometry(build_entity_geometry(net, 2.0), first, second)['status'] == 'no_conflict_zone'


def test_entity_boundary_tangency_is_still_retained_at_the_explicit_entity_width():
    net, first, second = topology(parallel_paths(2.0))
    pair = pair_geometry(build_entity_geometry(net, 2.0), first, second)
    assert pair['status'] == 'resolved' and pair['zones']


def test_required_vehicle_width_never_backfills_unknown_lane_width():
    net, first, second = topology(parallel_paths(3.2))
    with pytest.raises(ValueError, match='explicit vehicle width'):
        build_entity_geometry(net, None)
    net.lanes['I0']['width_m'] = None
    pair = pair_geometry(build_entity_geometry(net, 2.0), first, second)
    assert pair['status'] == 'unresolved' and pair['zones'] == []
    assert net.lanes['I0']['width_m'] is None
