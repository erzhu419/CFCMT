"""Current-state spatial conflict limits shared by admission and forward motion."""
from collections import defaultdict

from experiments.tsc_packet_receiving_path_20261110 import conditional_features as receiving_features
from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals
from .geometry import pair_geometry


def configure_geometry(kernel, table):
    """Attach one read-only static table, reusable for a batch of current states."""
    kernel._spatial_geometry = table
    incoming = defaultdict(list)
    for connection in kernel.topology.connections:
        incoming[connection['to_lane']].append(connection)
    kernel._spatial_receiving_connections = incoming


def _progress(kernel, packet, connection):
    return kernel.topology.path_progress(connection, packet['lane_id'], packet['position_m'])


def _occupancy_interval(front, length, speed, zone, *, processed=False):
    """Constant current-step motion; a stopped occupied zone has no finite exit."""
    start, end = zone
    rear = front-length
    if rear > end+1e-12:
        return None
    occupied = front >= start-1e-12 and rear <= end+1e-12
    if speed <= 0:
        return dict(start_sec=0., end_sec=None, occupied_now=True) if occupied else None
    enters, leaves = max((start-front)/speed, 0.), max((end+length-front)/speed, 0.)
    if processed:
        # A completed in-step motion ends at t=1. Do not extrapolate a vehicle
        # which stopped before this region through it after the known endpoint.
        if enters > 1.:
            return None
        if leaves > 1.:
            leaves = None
    return dict(start_sec=enters, end_sec=leaves, occupied_now=bool(occupied))


def _overlap(first, second):
    if first is None or second is None:
        return False
    end_a, end_b = first['end_sec'], second['end_sec']
    lo = max(first['start_sec'], second['start_sec'])
    return (end_a is None or lo < end_a) and (end_b is None or lo < end_b)


def _whole_interval(kernel, packet, connection, speed):
    front = _progress(kernel, packet, connection)
    origin = kernel.topology.lanes[connection['path'][0]]['length_m']
    end = sum(kernel.topology.lanes[lane]['length_m'] for lane in connection['path'][:-1])
    return (max((origin-front)/max(speed, .1), 0.),
            max((end+packet['parameters']['length_m']-front)/max(speed, .1), 0.))


def _old_precedes(kernel, ego, foe, connection, other, speed, other_speed, priority_index):
    if priority_index is not None:
        return priority_index[foe['packet_id']] < priority_index[ego['packet_id']]
    a_internal = kernel.topology.lanes[ego['lane_id']]['internal']
    b_internal = kernel.topology.lanes[foe['lane_id']]['internal']
    if a_internal != b_internal:
        return b_internal
    if ego['lane_id'] == foe['lane_id'] and ego['position_m'] != foe['position_m']:
        return foe['position_m'] > ego['position_m']
    first_signal, second_signal = kernel._signal(connection), kernel._signal(other)
    if kernel._gives_way(connection, first_signal, other, second_signal):
        return True
    if kernel._gives_way(other, second_signal, connection, first_signal):
        return False
    order = {key: index for index, key in enumerate(kernel.state['packets'])}
    def rank(packet, c, v):
        internal = kernel.topology.lanes[packet['lane_id']]['internal']
        signal = kernel._signal(c)
        tier = 0 if internal else 1 if signal == 'G' else 2
        distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
        return tier, distance/max(v, .1), order[packet['packet_id']]
    return rank(foe, other, other_speed) < rank(ego, connection, speed)


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
                   *, speeds=None, processed=None, priority_index=None):
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
        precedes = _old_precedes(kernel, packet, foe, connection, other, speed, other_speed, priority_index)
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
