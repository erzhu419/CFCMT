"""Entity-width swept-path abstraction on the unchanged static lane paths.

The caller supplies the explicit vehicle width. Lane geometry, stored-distance
mapping and receiver-prefix length retain their original lane-width semantics.
This is a centerline swept-path abstraction, not exact rigid-body collision.
"""
from collections import Counter, defaultdict
from math import isfinite

from experiments.tsc_packet_spatial_conflict_20261110.geometry import (
    EPS_M, _movement_segments, _pair, _pair_key, pair_geometry, geometry_metadata,
)


def build_entity_geometry(topology, vehicle_width_m):
    """Use a required, observed-type vehicle width only for capsule inflation."""
    if vehicle_width_m is None:
        raise ValueError('An explicit vehicle width is required; no default is supplied')
    width = float(vehicle_width_m)
    if not isfinite(width) or width <= 0:
        raise ValueError('The explicit vehicle width must be finite and positive')
    by_tls = defaultdict(list)
    paths = {}
    for connection in topology.connections:
        if connection['tls_id'] is not None:
            by_tls[connection['tls_id']].append(connection)
        # Prefix clipping and path validity use the original physical lane data.
        path = _movement_segments(topology, connection)
        paths[connection['key']] = dict(path, segments=[dict(segment, width=width)
                                                      for segment in path['segments']])
    pairs, counts, unresolved = {}, Counter(), Counter()
    for connections in by_tls.values():
        connections = sorted(connections, key=lambda connection: connection['key'])
        for index, first in enumerate(connections):
            for second in connections[index+1:]:
                if not (topology.foes(first, second) or topology.foes(second, first)):
                    continue
                pair = _pair(first, second, paths)
                if pair['status'] == 'resolved':
                    pair['reason'] = 'entity_width_swept_path_overlap'
                elif pair['status'] == 'no_conflict_zone':
                    pair['reason'] = 'validated_entity_swept_paths_disjoint'
                pairs[_pair_key(first['key'], second['key'])] = pair
                counts[pair['status']] += 1
                if pair['status'] == 'unresolved':
                    unresolved.update({reason.split(':', 1)[0] for reason in pair['reason'].split(';')})
    lane_widths = [float(lane['width_m']) for lane in topology.lanes.values()
                   if lane.get('width_m') is not None and isfinite(float(lane['width_m'])) and float(lane['width_m']) > 0]
    shape_present = lambda lane: isinstance(lane.get('shape'), str) and bool(lane['shape'].strip())
    return dict(pairs=pairs, status_counts=dict(counts), pair_count=len(pairs),
                unresolved_reason_counts=dict(unresolved), entity_width_m=width,
                coverage=dict(lanes=len(topology.lanes), lanes_with_real_width=len(lane_widths),
                              lanes_with_shape_text=sum(shape_present(lane) for lane in topology.lanes.values()),
                              internal_lanes=sum(lane['internal'] for lane in topology.lanes.values()),
                              internal_lanes_with_shape_text=sum(lane['internal'] and shape_present(lane) for lane in topology.lanes.values()),
                              full_movement_paths=len(paths),
                              resolved_movement_paths=sum(path['status'] == 'resolved' for path in paths.values()),
                              minimum_width_m=min(lane_widths) if lane_widths else None,
                              maximum_width_m=max(lane_widths) if lane_widths else None),
                method='Entity-width centerline swept-path abstraction: capsule radius=(vehicle_width_a+vehicle_width_b)/2; not exact rigid-body collision.',
                entity_width_source='Required caller-supplied explicit vehicle-type width; no default or outcome tuning.',
                width_source='Unchanged topology.lanes width_m for path validity and receiver-prefix clipping.',
                receiving_prefix_rule='Unchanged: one actual receiving-lane width, not vehicle width.',
                path_lateral_offset_m=0., numeric_tolerance_m=EPS_M, module_source=__file__)
