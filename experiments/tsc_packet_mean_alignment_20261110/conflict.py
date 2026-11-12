"""Frozen conflict timing with only the processed-movement insertion corrected."""
from experiments.tsc_packet_spatial_conflict_20261110.conflict import (
    _progress, _whole_interval, _old_precedes, _occupancy_interval, _overlap,
)
from experiments.tsc_packet_spatial_conflict_20261110.geometry import pair_geometry


from experiments.tsc_packet_receiving_path_20261110 import conditional_features as receiving_features
from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals


def _movements(kernel, ego_connection, packets, speeds, processed):
    """Current committed bodies, green approaches, and already accepted in-step paths."""
    result = {}
    topology = kernel.topology
    for key, packet in packets.items():
        current, legal = kernel._connections[key]
        possibilities = []
        front_internal = topology.lanes[packet['lane_id']]['internal']
        if current is not None and legal:
            possibilities.append((current, front_internal, False))
        for lane, rear, front in _body_intervals(topology, packet):
            if front > rear and topology.lanes[lane]['internal']:
                possibilities.extend((c, True, False) for c in topology.internal_connections[lane])
        # A body can have fully entered the receiver yet still occupy the
        # width-sized merge prefix. Its known receiver fragment is sufficient;
        # previous internal fragments lie behind its rear and are never invented.
        if not front_internal:
            geometry = topology.lanes[packet['lane_id']]
            width = geometry.get('width_m')
            if width is not None and packet['position_m']-packet['parameters']['length_m'] <= width:
                possibilities.extend((c, True, True) for c in
                    kernel._spatial_receiving_connections.get(packet['lane_id'], ()))
        for connection, committed, receiver_only in possibilities:
            if not topology.foes(ego_connection, connection):
                continue
            record_key = key, connection['key']
            previous = result.get(record_key)
            if previous is None or (committed and not previous['committed']):
                result[record_key] = dict(packet=packet, connection=connection,
                    committed=bool(committed), receiver_only=bool(receiver_only), accepted=False)
    for key, motion in (processed or {}).items():
        connection = motion['connection']
        if connection is None or not topology.foes(ego_connection, connection):
            continue
        record_key = key, connection['key']
        # A merely processed incompatible ordinary intent has not traversed
        # this borrowed adjacent-lane path. It cannot create a new reservation.
        # Existing legal approaches and physical tails still receive the
        # processed speed below, even when neither entry flag is set.
        if record_key not in result and not (motion['committed'] or motion['accepted']):
            continue
        if record_key not in result:
            result[record_key] = dict(packet=motion['packet'], connection=connection,
                committed=bool(motion['committed']), receiver_only=False, accepted=bool(motion['accepted']))
        result[record_key]['accepted'] = bool(motion['accepted'])
        result[record_key]['speed'] = motion['speed']
    for record in result.values():
        key = record['packet']['packet_id']
        if key in (processed or {}):
            record['speed'] = processed[key]['speed']
            record['processed'] = True
        elif 'speed' not in record:
            record['speed'] = speeds[key] if speeds is not None else receiving_features._zero_speed(kernel, record['packet'])
    return list(result.values())


def conflict_limit(kernel, packet, connection, compatible, speed, packets,
                   *, speeds=None, processed=None, priority_index=None, priority_speeds=None):
    """Bound front travel before active crossing/merge regions, including internally.

    ``packets`` are pre-step positions. Processed motions keep those starting
    positions and their actual travelled distance, retaining within-step
    reservations even when a processed front/tail has cleared by the poststate.
    """
    path = kernel._path(packet, connection, compatible)
    remaining = sum(kernel.topology.lanes[lane]['length_m'] for lane in path)-packet['position_m']
    limit = max(float(remaining), 0.)
    base = dict(limit_m=limit, active_conflict=False, entry_blocked=False,
        unresolved_pairs=0, resolved_pairs=0, no_zone_pairs=0, details=[])
    if connection is None or not compatible or speed <= 0:
        return base
    topology = kernel.topology
    front = _progress(kernel, packet, connection)
    ego_internal = topology.lanes[packet['lane_id']]['internal']
    entry_distance = topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
    whole = _whole_interval(kernel, packet, connection, speed)
    seen_zones = set()
    for movement in _movements(kernel, connection, packets, speeds, processed):
        foe, other, other_speed = movement['packet'], movement['connection'], movement['speed']
        if foe['packet_id'] == packet['packet_id']:
            continue
        other_signal = kernel._signal(other)
        ranking_ego = speed if priority_speeds is None else priority_speeds[packet['packet_id']]
        ranking_foe = other_speed if priority_speeds is None else priority_speeds[foe['packet_id']]
        precedes = _old_precedes(kernel, packet, foe, connection, other, ranking_ego, ranking_foe, priority_index)
        if not movement['committed'] and not movement['accepted']:
            if other_signal not in ('G', 'g') or not kernel._gives_way(
                    connection, kernel._signal(connection), other, other_signal):
                continue
        geometry = pair_geometry(kernel._spatial_geometry, connection, other)
        status = geometry['status']
        if status == 'no_conflict_zone':
            base['no_zone_pairs'] += 1
            continue
        other_front = _progress(kernel, foe, other)
        detail = dict(packet_id=foe['packet_id'], connection_key=other['key'], committed=movement['committed'],
            receiver_only=movement['receiver_only'], accepted_in_step=movement['accepted'],
            speed_mps=other_speed, geometry_status=status, active_zones=[])
        if status == 'unresolved':
            reserved = _whole_interval(kernel, foe, other, other_speed)
            overlap = max(whole[0], reserved[0]) < min(whole[1], reserved[1])
            if overlap:
                fallback = 0. if ego_internal else max(entry_distance-1e-9, 0.)
                limit = min(limit, fallback)
                base['active_conflict'] = True
                base['unresolved_pairs'] += 1
                detail.update(fallback_limit_m=fallback, whole_interval=list(reserved))
                base['details'].append(detail)
            continue
        base['resolved_pairs'] += 1
        zones = geometry['contact_pieces'] if movement['receiver_only'] else geometry['zones']
        for zone in zones:
            if movement['receiver_only'] and not any(foe_lane == foe['lane_id']
                    for unused, foe_lane in zone['lane_pairs']):
                continue
            ego_zone, foe_zone = zone['ego_interval_m'], zone['foe_interval_m']
            # Deduplicate equivalent receiver-prefix fragments from multiple
            # possible previous physical connections of the same known body.
            foe_origin = sum(topology.lanes[lane]['length_m'] for lane in other['path'][:-1])
            signature = (foe['packet_id'], tuple(round(x, 9) for x in ego_zone),
                foe['lane_id'] if movement['receiver_only'] else other['key'],
                tuple(round(x-foe_origin, 9) for x in foe_zone) if movement['receiver_only'] else tuple(foe_zone))
            if signature in seen_zones:
                continue
            seen_zones.add(signature)
            own_interval = _occupancy_interval(front, packet['parameters']['length_m'], speed, ego_zone)
            foe_interval = _occupancy_interval(other_front, foe['parameters']['length_m'], other_speed, foe_zone,
                                                processed=movement.get('processed', False))
            if not _overlap(own_interval, foe_interval):
                continue
            # An internal front already beyond the region must be able to
            # clear its remaining tail. A vehicle already in the region owns
            # it before a merely approaching body; ties retain existing order.
            if ego_internal and front > ego_zone[1]+1e-12:
                continue
            ego_inside = own_interval['occupied_now']
            foe_inside = foe_interval['occupied_now']
            if ego_internal:
                if foe_inside and other_front > foe_zone[1]+1e-12:
                    # A tail still occupies the region even after its front
                    # passed it. Right of way cannot erase an existing body.
                    bound = max(ego_zone[0]-front-1e-9, 0.)
                    limit = min(limit, bound)
                    base['active_conflict'] = True
                    detail['active_zones'].append(dict(ego_interval_m=ego_zone, foe_interval_m=foe_zone,
                        ego_time=own_interval, foe_time=foe_interval, limit_m=bound))
                    continue
                if ego_inside and not foe_inside:
                    continue
                if ego_inside and not movement['committed']:
                    continue
                if movement['committed'] and ego_inside and foe_inside and not precedes:
                    continue
                if movement['committed'] and not ego_inside and not foe_inside and not precedes:
                    continue
            bound = max(ego_zone[0]-front-1e-9, 0.)
            limit = min(limit, bound)
            base['active_conflict'] = True
            detail['active_zones'].append(dict(ego_interval_m=ego_zone, foe_interval_m=foe_zone,
                ego_time=own_interval, foe_time=foe_interval, limit_m=bound))
        if detail['active_zones']:
            base['details'].append(detail)
    base['limit_m'] = float(limit)
    base['entry_blocked'] = bool(not ego_internal and limit < entry_distance)
    return base
