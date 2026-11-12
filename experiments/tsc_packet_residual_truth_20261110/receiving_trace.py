"""Read-only explanation of a frozen receiving bound and subsequent observations."""
from math import isclose

from experiments.tsc_packet_receiving_path_20261110.receiving import (
    _body_intervals, admissible_travel_m,
)


def _snapshot(packet):
    if packet is None:
        return None
    return {name: packet[name] for name in
            ('packet_id', 'lane_id', 'position_m', 'speed_mps', 'body_trail')}


def _longitudinal_displacement(topology, before, after, path, source):
    """Collapse same-road lateral transitions instead of adding a whole lane."""
    if after is None:
        return dict(status='unresolved', reason='not_in_next_observation', displacement_m=None)
    if not path or path[0] != before['lane_id'] or path[-1] != after['lane_id']:
        return dict(status='unresolved', reason='front_path_endpoints_do_not_match', displacement_m=None)
    groups, lateral = [], False
    for lane in path:
        geometry = topology.lanes[lane]
        if (groups and not geometry['internal'] and not topology.lanes[groups[-1][-1]]['internal']
                and geometry['edge_id'] == topology.lanes[groups[-1][-1]]['edge_id']):
            lateral |= lane != groups[-1][-1]
            groups[-1].append(lane)
        else:
            groups.append([lane])
    distance = -before['position_m']+after['position_m']
    distance += sum(topology.lanes[group[-1]]['length_m'] for group in groups[:-1])
    return dict(status='resolved', source=source, observed_front_path=list(path),
                ordinary_same_road_lateral_transition=bool(lateral),
                distance_kind='native_longitudinal_coordinate_change' if lateral else 'observed_path_arc_change',
                displacement_m=float(distance))


def _observed_motion(topology, before, after, supplied=None):
    if after is None:
        return dict(status='unresolved', reason='not_in_next_observation', displacement_m=None)
    if supplied:
        return _longitudinal_displacement(topology, before, after, supplied, 'provided_observed_front_path')
    old_lane, next_lane = before['lane_id'], after['lane_id']
    a, b = topology.lanes[old_lane], topology.lanes[next_lane]
    if old_lane == next_lane or (not a['internal'] and not b['internal'] and a['edge_id'] == b['edge_id']):
        return _longitudinal_displacement(topology, before, after, [old_lane] if old_lane == next_lane
                                          else [old_lane, next_lane], 'native_same_road_positions')
    trail = after['body_trail']
    if old_lane in trail:
        return _longitudinal_displacement(topology, before, after, trail[trail.index(old_lane):],
                                          'next_observed_body_trail')
    owners = topology.internal_connections[old_lane] if a['internal'] else topology.connections_from(old_lane)
    paths = set()
    for connection in owners:
        path = connection['path']
        if next_lane in path and path.index(next_lane) > path.index(old_lane):
            paths.add(tuple(path[path.index(old_lane):path.index(next_lane)+1]))
    if len(paths) == 1:
        return _longitudinal_displacement(topology, before, after, next(iter(paths)),
                                          'unique_static_connection_between_observed_front_lanes')
    return dict(status='unresolved', reason='observed_transition_path_not_unique_or_not_on_connection',
                displacement_m=None, possible_path_count=len(paths))


def explain_receiving(topology, ego, packets, path, next_packets, *, connection=None,
                      observed_ego_path=None, observed_neighbour_paths=None):
    """Explain current bindings, then append posthoc t+1 evidence without intervention.

    The current bound and binding set are finalized before next_packets is read.
    Snapshot speeds and native path displacement are distinct reported quantities.
    """
    path = tuple(path)
    current_bound = admissible_travel_m(topology, ego, packets, path)
    offsets, end = {}, -ego['position_m']
    for lane in path:
        offsets[lane] = end
        end += topology.lanes[lane]['length_m']
    constraints = [dict(kind='path_end', bound_m=max(end, 0.))]
    for packet_id, other in packets.items():
        if packet_id == ego['packet_id']:
            continue
        for lane, rear, front in _body_intervals(topology, other):
            if lane not in offsets or front <= rear or offsets[lane]+front <= 1e-12:
                continue
            static_gap = offsets[lane]+rear
            constraints.append(dict(kind='body', packet_id=packet_id, lane_id=lane,
                segment_rear_m=rear, segment_front_m=front,
                current_static_gap_m=static_gap, bound_m=max(static_gap-ego['parameters']['min_gap_m'], 0.)))
    bindings = [dict(row) for row in constraints if isclose(row['bound_m'], current_bound, rel_tol=0, abs_tol=1e-8)]
    if not bindings:
        raise ValueError('The explanation must reproduce a binding constraint from the frozen helper')

    # Everything below is observation-only evidence; none changes current_bound.
    next_ego = next_packets.get(ego['packet_id'])
    ego_motion = _observed_motion(topology, ego, next_ego, observed_ego_path)
    reference = (topology.path_progress(connection, ego['lane_id'], ego['position_m'])
                 if connection is not None else None)

    def project(lane, position):
        if lane not in offsets:
            return None
        if connection is not None:
            return topology.path_progress(connection, lane, position)-reference
        return offsets[lane]+position

    def on_path_segments(packet):
        if packet is None:
            return []
        return [dict(lane_id=lane, rear_relative_m=project(lane, rear), front_relative_m=project(lane, front))
                for lane, rear, front in _body_intervals(topology, packet)
                if lane in offsets and front > rear]

    ego_end = None if next_ego is None else project(next_ego['lane_id'], next_ego['position_m'])
    neighbours = {}
    observed_neighbour_paths = observed_neighbour_paths or {}
    for binding in bindings:
        if binding['kind'] != 'body':
            continue
        packet_id = binding['packet_id']
        if packet_id not in neighbours:
            before, after = packets[packet_id], next_packets.get(packet_id)
            segments = on_path_segments(after)
            motion = _observed_motion(topology, before, after, observed_neighbour_paths.get(packet_id))
            front_end = None if after is None else project(after['lane_id'], after['position_m'])
            prior_body = on_path_segments(before)
            rear_before = min((row['rear_relative_m'] for row in prior_body), default=None)
            rear_after = min((row['rear_relative_m'] for row in segments), default=None)
            a = topology.lanes[before['lane_id']]
            b = None if after is None else topology.lanes[after['lane_id']]
            same_road_change = (after is not None and before['lane_id'] != after['lane_id']
                               and not a['internal'] and not b['internal'] and a['edge_id'] == b['edge_id'])
            neighbours[packet_id] = dict(current=_snapshot(before), next=_snapshot(after), observed_motion=motion,
                current_front_relative_m=project(before['lane_id'], before['position_m']),
                next_front_relative_m=front_end, current_body_rear_on_path_m=rear_before,
                next_body_rear_on_path_m=rear_after,
                body_rear_projection_change_m=None if rear_before is None or rear_after is None else rear_after-rear_before,
                front_left_model_path=None if after is None else after['lane_id'] not in offsets,
                same_road_lane_change=bool(same_road_change), next_body_still_on_model_path=bool(segments),
                next_body_segments_on_model_path=segments,
                relation_status='not_in_next_observation' if after is None else
                    'left_path_with_remaining_body' if front_end is None and segments else
                    'left_model_path' if front_end is None else 'front_still_on_model_path')
        neighbour = neighbours[packet_id]
        segments = neighbour['next_body_segments_on_model_path']
        ahead = [] if ego_end is None else [segment for segment in segments
                                           if segment['front_relative_m'] > ego_end+1e-12]
        gap = min((segment['rear_relative_m']-ego_end for segment in ahead), default=None)
        binding.update(posthoc_neighbour=neighbour, posthoc_end_gap_m=gap,
            posthoc_end_gap_status='ego_left_model_path_or_missing' if ego_end is None else
                'binding_body_no_longer_ahead_on_model_path' if not ahead else 'projected_on_same_model_path',
            posthoc_end_gap_minus_min_gap_m=None if gap is None else gap-ego['parameters']['min_gap_m'])
    displacement = ego_motion.get('displacement_m')
    next_speed = None if next_ego is None else next_ego['speed_mps']
    return dict(current_bound_m=current_bound, current_path=list(path), current_ego=_snapshot(ego),
        current_min_gap_m=ego['parameters']['min_gap_m'], binding_constraints=bindings,
        posthoc_next_ego=_snapshot(next_ego), observed_ego_motion=ego_motion,
        actual_ego_step_displacement_m=displacement, ego_end_projection_on_model_path_m=ego_end,
        next_speed_snapshot_mps=next_speed,
        next_speed_times_dt_minus_step_displacement_m=None if next_speed is None or displacement is None else next_speed-displacement,
        timestep_sec=1., future_observations_used_for_current_bound=False,
        scope='Bindings use current packets only. Next-frame speed, path displacement and gaps are separate posthoc evidence; no penetration or simulator integration rule is inferred.')
