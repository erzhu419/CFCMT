"""Finite longitudinal room along the current committed lane path."""


def _body_intervals(topology, packet):
    """Split the physical body over its observed/predicted accounting trail."""
    lane, position = packet['lane_id'], packet['position_m']
    trail = packet['body_trail']
    if not trail or trail[-1] != lane:
        raise ValueError('A body trail must end on the current front lane')
    length = packet['parameters']['length_m']
    intervals = [(lane, max(position-length, 0.), position)]
    remaining = max(length-position, 0.)
    for prior in reversed(trail[:-1]):
        if remaining <= 0:
            break
        lane_length = topology.lanes[prior]['length_m']
        covered = min(remaining, lane_length)
        intervals.append((prior, lane_length-covered, lane_length))
        remaining -= covered
    return intervals


def admissible_travel_m(topology, packet, packets, path):
    """Stop the front before an occupied body or the finite end of ``path``.

    The path begins on this packet's current front lane and includes its
    internal lanes and receiving lane. Other fronts may already have left that
    path while their tails still occupy it. Current or jointly updated packets
    are supplied by the caller; this helper reads no future observation.
    """
    if not path or path[0] != packet['lane_id']:
        raise ValueError('The admissible path must begin on the current front lane')
    offsets, end = {}, -packet['position_m']
    for lane in path:
        offsets[lane] = end
        end += topology.lanes[lane]['length_m']
    bound = max(end, 0.)
    gap = packet['parameters']['min_gap_m']
    for other in packets.values():
        if other['packet_id'] == packet['packet_id']:
            continue
        for lane, rear, front in _body_intervals(topology, other):
            if lane not in offsets or front <= rear:
                continue
            origin = offsets[lane]
            if origin+front > 1e-12:
                bound = min(bound, max(origin+rear-gap, 0.))
    return float(bound)
