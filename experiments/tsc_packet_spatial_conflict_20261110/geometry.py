"""Static lane-width corridor conflict zones in stored path-distance units.

No observed future, fitted coefficient or invented conflict point is used.
For each pair of finite shape segments, project the intersection with the
other segment's capsule of radius (width_a+width_b)/2. This includes tangency
and near/merge interactions without requiring a centerline intersection.
"""
from collections import Counter, defaultdict
from math import hypot, isfinite, sqrt


EPS_M = 1e-9  # Numeric geometry tolerance, independent of outcomes.


def _dot(first, second):
    return first[0]*second[0]+first[1]*second[1]


def _sub(first, second):
    return first[0]-second[0], first[1]-second[1]


def _clip_linear(interval, offset, slope, lower, upper):
    if abs(slope) < 1e-14:
        return interval if lower-EPS_M <= offset <= upper+EPS_M else None
    first, second = (lower-offset)/slope, (upper-offset)/slope
    lo, hi = max(interval[0], min(first, second)), min(interval[1], max(first, second))
    if lo > hi+EPS_M:
        return None
    if lo > hi:
        lo = hi = .5*(lo+hi)
    return lo, hi


def _disk_interval(start, direction, length, center, radius):
    delta = _sub(start, center)
    along = _dot(delta, direction)
    perpendicular_sq = max(_dot(delta, delta)-along*along, 0.)
    discriminant = radius*radius-perpendicular_sq
    if discriminant < -(2*radius*EPS_M+EPS_M*EPS_M):
        return None
    half = sqrt(max(discriminant, 0.))
    lo, hi = max(0., -along-half), min(length, -along+half)
    if lo > hi+EPS_M:
        return None
    if lo > hi:
        lo = hi = min(max(.5*(lo+hi), 0.), length)
    return lo, hi


def _segment_capsule_interval(a0, a1, b0, b1, radius):
    """Distances along finite A whose centers are within radius of finite B."""
    da, db = _sub(a1, a0), _sub(b1, b0)
    la, lb = hypot(*da), hypot(*db)
    if la <= EPS_M or lb <= EPS_M:
        raise ValueError('Geometry segments must have positive geometric length')
    ua, ub = (da[0]/la, da[1]/la), (db[0]/lb, db[1]/lb)
    intervals = [value for value in (
        _disk_interval(a0, ua, la, b0, radius),
        _disk_interval(a0, ua, la, b1, radius)) if value is not None]
    delta = _sub(a0, b0)
    middle = _clip_linear((0., la), _dot(delta, ub), _dot(ua, ub), 0., lb)
    if middle is not None:
        normal = (-ub[1], ub[0])
        middle = _clip_linear(middle, _dot(delta, normal), _dot(ua, normal), -radius, radius)
        if middle is not None:
            intervals.append(middle)
    # A capsule is convex, so its intersection with a finite line is connected.
    if not intervals:
        return None
    return min(value[0] for value in intervals), max(value[1] for value in intervals)


def _lane_geometry(topology, lane_id):
    lane = topology.lanes[lane_id]
    width = lane.get('width_m')
    if width is None or not isfinite(float(width)) or float(width) <= 0:
        return None, 'missing_or_invalid_width:'+lane_id
    shape = lane.get('shape')
    if not isinstance(shape, str):
        return None, 'missing_shape:'+lane_id
    try:
        points = [tuple(map(float, value.split(','))) for value in shape.split()]
    except ValueError:
        return None, 'invalid_shape:'+lane_id
    if len(points) < 2 or any(len(point) != 2 or not all(map(isfinite, point)) for point in points):
        return None, 'invalid_shape:'+lane_id
    clean = [points[0]]
    for point in points[1:]:
        if hypot(*_sub(point, clean[-1])) > EPS_M:
            clean.append(point)
    arc = sum(hypot(*_sub(b, a)) for a, b in zip(clean, clean[1:]))
    stored = float(lane['length_m'])
    if arc <= EPS_M or not isfinite(stored) or stored <= 0:
        return None, 'degenerate_shape_or_length:'+lane_id
    return dict(points=clean, width=float(width), stored_length=stored, arc_length=arc,
                stored_per_shape_m=stored/arc), None


def _movement_segments(topology, connection):
    path = connection['path']
    if len(path) < 3 or not any(topology.lanes[lane]['internal'] for lane in path[1:-1]):
        return dict(status='unresolved', reason='no_internal_shape_path', segments=[])
    lanes = {}
    for lane_id in path:
        geometry, error = _lane_geometry(topology, lane_id)
        if error:
            return dict(status='unresolved', reason=error, segments=[])
        lanes[lane_id] = geometry
    for first, second in zip(path, path[1:]):
        a, b = lanes[first], lanes[second]
        gap = hypot(*_sub(a['points'][-1], b['points'][0]))
        # Existing endpoint corridors must meet; missing geometry is not bridged.
        if gap > .5*(a['width']+b['width'])+EPS_M:
            return dict(status='unresolved', reason='disconnected_path_shapes:'+first+'->'+second, segments=[])
    segments, offset = [], 0.
    for index, lane_id in enumerate(path):
        lane = lanes[lane_id]
        if index == 0:
            offset += lane['stored_length']
            continue
        receiving = index == len(path)-1
        limit = min(lane['stored_length'], lane['width']) if receiving else lane['stored_length']
        shape_arc = 0.
        for first, second in zip(lane['points'], lane['points'][1:]):
            length = hypot(*_sub(second, first))
            stored0 = shape_arc*lane['stored_per_shape_m']
            if stored0 >= limit-EPS_M:
                break
            stored1 = (shape_arc+length)*lane['stored_per_shape_m']
            clipped1 = min(stored1, limit)
            fraction = (clipped1-stored0)/(stored1-stored0)
            endpoint = (first[0]+fraction*(second[0]-first[0]), first[1]+fraction*(second[1]-first[1]))
            segments.append(dict(lane_id=lane_id, receiving=receiving, p0=first, p1=endpoint,
                                 progress0=offset+stored0, progress1=offset+clipped1, width=lane['width']))
            shape_arc += length
        offset += lane['stored_length']
    return dict(status='resolved', segments=segments, source_length_m=lanes[path[0]]['stored_length'],
                receiving_prefix_m=min(lanes[path[-1]]['stored_length'], lanes[path[-1]]['width']),
                lane_scales={lane: value['stored_per_shape_m'] for lane, value in lanes.items()})


def _mapped_interval(segment, interval):
    length = hypot(*_sub(segment['p1'], segment['p0']))
    factor = (segment['progress1']-segment['progress0'])/length
    return [segment['progress0']+interval[0]*factor, segment['progress0']+interval[1]*factor]


def _touches(first, second):
    return all(max(first[name][0], second[name][0]) <= min(first[name][1], second[name][1])+EPS_M
               for name in ('ego_interval_m', 'foe_interval_m'))


def _merge_zones(zones):
    merged = []
    for zone in zones:
        pending = zone
        changed = True
        while changed:
            changed = False
            remaining = []
            for existing in merged:
                if _touches(pending, existing):
                    for name in ('ego_interval_m', 'foe_interval_m'):
                        pending[name] = [min(pending[name][0], existing[name][0]),
                                         max(pending[name][1], existing[name][1])]
                    pending['lane_pairs'] = sorted(set(pending['lane_pairs']) | set(existing['lane_pairs']))
                    changed = True
                else:
                    remaining.append(existing)
            merged = remaining
        merged.append(pending)
    return sorted(merged, key=lambda zone: (zone['ego_interval_m'], zone['foe_interval_m']))


def _pair(first, second, paths):
    a, b = paths[first['key']], paths[second['key']]
    result = dict(first_connection_key=first['key'], second_connection_key=second['key'],
                  coordinate='stored connection.path progress from ordinary source-road start',
                  zones=[], contact_pieces=[])
    if a['status'] != 'resolved' or b['status'] != 'resolved':
        result.update(status='unresolved', reason=';'.join(value['reason'] for value in (a, b)
                                                         if value['status'] != 'resolved'))
        return result
    zones = []
    for sa in a['segments']:
        for sb in b['segments']:
            radius = .5*(sa['width']+sb['width'])
            along_a = _segment_capsule_interval(sa['p0'], sa['p1'], sb['p0'], sb['p1'], radius)
            if along_a is None:
                continue
            along_b = _segment_capsule_interval(sb['p0'], sb['p1'], sa['p0'], sa['p1'], radius)
            if along_b is None:
                result.update(status='unresolved', reason='asymmetric_numeric_geometry')
                return result
            zones.append(dict(ego_interval_m=_mapped_interval(sa, along_a),
                              foe_interval_m=_mapped_interval(sb, along_b),
                              lane_pairs=[(sa['lane_id'], sb['lane_id'])]))
    # Keep each physical lane-pair contact separately before merging: a packet
    # already wholly in a receiver has no unknown past-internal occupancy.
    pieces = [dict(ego_interval_m=list(zone['ego_interval_m']),
                   foe_interval_m=list(zone['foe_interval_m']), lane_pairs=list(zone['lane_pairs']))
              for zone in zones]
    result.update(status='resolved' if zones else 'no_conflict_zone', zones=_merge_zones(zones),
                  contact_pieces=pieces,
                  reason='real_width_capsule_overlap' if zones else 'validated_real_corridors_disjoint',
                  first_source_length_m=a['source_length_m'], second_source_length_m=b['source_length_m'],
                  first_receiving_prefix_m=a['receiving_prefix_m'], second_receiving_prefix_m=b['receiving_prefix_m'])
    return result


def _pair_key(first, second):
    return '|'.join(sorted((first, second)))


def build_conflict_geometry(topology):
    """Precompute declared foe pairs; missing real geometry keeps old constraints."""
    by_tls = defaultdict(list)
    for connection in topology.connections:
        if connection['tls_id'] is not None:
            by_tls[connection['tls_id']].append(connection)
    paths = {connection['key']: _movement_segments(topology, connection) for connection in topology.connections}
    pairs, counts, unresolved = {}, Counter(), Counter()
    for connections in by_tls.values():
        connections = sorted(connections, key=lambda connection: connection['key'])
        for index, first in enumerate(connections):
            for second in connections[index+1:]:
                if not (topology.foes(first, second) or topology.foes(second, first)):
                    continue
                pair = _pair(first, second, paths)
                pairs[_pair_key(first['key'], second['key'])] = pair
                counts[pair['status']] += 1
                if pair['status'] == 'unresolved':
                    unresolved.update({reason.split(':', 1)[0] for reason in pair['reason'].split(';')})
    widths = [float(lane['width_m']) for lane in topology.lanes.values()
              if lane.get('width_m') is not None and isfinite(float(lane['width_m'])) and float(lane['width_m']) > 0]
    shape_present = lambda lane: isinstance(lane.get('shape'), str) and bool(lane['shape'].strip())
    return dict(pairs=pairs, status_counts=dict(counts), pair_count=len(pairs),
                unresolved_reason_counts=dict(unresolved),
                coverage=dict(lanes=len(topology.lanes), lanes_with_real_width=len(widths),
                              lanes_with_shape_text=sum(shape_present(lane) for lane in topology.lanes.values()),
                              internal_lanes=sum(lane['internal'] for lane in topology.lanes.values()),
                              internal_lanes_with_shape_text=sum(lane['internal'] and shape_present(lane) for lane in topology.lanes.values()),
                              full_movement_paths=len(paths),
                              resolved_movement_paths=sum(path['status'] == 'resolved' for path in paths.values()),
                              minimum_width_m=min(widths) if widths else None, maximum_width_m=max(widths) if widths else None),
                method='Conservative analytic segment proximity envelope: capsule radius=(width_a+width_b)/2; stored/shape arc scaling.',
                width_source='Explicit topology.lanes width_m; this module never supplies a width default.',
                receiving_prefix_rule='One actual receiving-lane width; downstream following remains a body/path constraint.',
                numeric_tolerance_m=EPS_M, module_source=__file__)


def geometry_metadata(table):
    """Small coverage/provenance summary without serializing the geometry table."""
    return {key: value for key, value in table.items() if key != 'pairs'}


def pair_geometry(table, first, second):
    """Return zones ordered to the requested ego/foe connection keys."""
    first = first['key'] if isinstance(first, dict) else first
    second = second['key'] if isinstance(second, dict) else second
    value = table['pairs'].get(_pair_key(first, second))
    if value is None:
        return dict(status='unresolved', reason='pair_not_precomputed', zones=[])
    if value['first_connection_key'] == first:
        return value
    result = dict(value, first_connection_key=first, second_connection_key=second)
    for field in ('zones', 'contact_pieces'):
        result[field] = [dict(zone, ego_interval_m=list(zone['foe_interval_m']),
                             foe_interval_m=list(zone['ego_interval_m']),
                             lane_pairs=[(b, a) for a, b in zone['lane_pairs']]) for zone in value[field]]
    for name in ('source_length_m', 'receiving_prefix_m'):
        if 'first_'+name in value:
            result['first_'+name], result['second_'+name] = value['second_'+name], value['first_'+name]
    return result
