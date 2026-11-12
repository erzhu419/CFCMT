"""Observed local-zone events on a specified path, without subsecond interpolation."""
from collections import defaultdict

from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals


def _frame(topology, connection, offsets, zone, snapshot):
    packet = snapshot['packet']
    result = dict(time_sec=snapshot['time_sec'], packet_present=packet is not None,
                  front_progress_m=None, front_relation='unknown', body_on_zone=None,
                  mapped_body_intervals_m=[], off_path_body_segments=[])
    if packet is None:
        return result
    lane = packet['lane_id']
    result.update(lane_id=lane, position_m=packet['position_m'], speed_mps=packet['speed_mps'])
    if lane in offsets:
        progress = topology.path_progress(connection, lane, packet['position_m'])
        result['front_progress_m'] = progress
        result['front_relation'] = 'before' if progress < zone[0] else 'after' if progress > zone[1] else 'inside'
    else:
        result['front_relation'] = 'off_path'
    trail = packet.get('body_trail')
    if not trail or trail[-1] != lane:
        return result
    for occupied_lane, rear, front in _body_intervals(topology, packet):
        if front <= rear:
            continue
        if occupied_lane in offsets:
            result['mapped_body_intervals_m'].append(
                [offsets[occupied_lane]+rear, offsets[occupied_lane]+front])
        else:
            result['off_path_body_segments'].append(dict(lane_id=occupied_lane, rear_m=rear, front_m=front))
    occupied = any(max(rear, zone[0]) <= min(front, zone[1])
                   for rear, front in result['mapped_body_intervals_m'])
    if occupied:
        result['body_on_zone'] = True
    elif packet.get('body_path_complete') is True:
        result['body_on_zone'] = False
    return result


def observe_zone_truth(topology, connection, zone_interval_m, snapshots):
    """Return first observed front entry/exit and tail clearance intervals.

    ``snapshots`` contains consecutive ``{time_sec, packet}`` observations.
    Packet may be None; that never proves clearance. Progress is defined only
    on the exact specified connection path. A complete body trail can prove
    physical departure even after the front leaves that path. All event times
    are (previous observed frame, current observed frame]; no interpolation or
    ordering inside a one-second interval is asserted.
    """
    if isinstance(connection, str):
        connection = topology.by_key[connection]
    zone = list(map(float, zone_interval_m))
    if len(zone) != 2 or zone[1] < zone[0]:
        raise ValueError('Zone must contain ordered start/end path distances')
    snapshots = list(snapshots)
    if not snapshots:
        raise ValueError('At least one observed snapshot is required')
    if any(after['time_sec']-before['time_sec'] != 1
           for before, after in zip(snapshots, snapshots[1:])):
        raise ValueError('Zone truth requires consecutive one-second snapshots')
    offsets, distance = {}, 0.
    for lane in connection['path']:
        offsets[lane] = distance
        distance += topology.lanes[lane]['length_m']
    frames = [_frame(topology, connection, offsets, zone, snapshot) for snapshot in snapshots]
    events = dict(front_entry=None, front_exit=None, tail_clear=None)
    for before, after in zip(frames, frames[1:]):
        interval = [before['time_sec'], after['time_sec']]
        old, new = before['front_progress_m'], after['front_progress_m']
        if old is not None and new is not None:
            if events['front_entry'] is None and old < zone[0] <= new:
                events['front_entry'] = interval
            if events['front_exit'] is None and old <= zone[1] < new:
                events['front_exit'] = interval
        # The second clause covers traversing the complete zone between two
        # samples; both entry and rear-clear remain in this same coarse interval.
        whole_zone_passed = (events['front_entry'] == interval and new is not None and new > zone[1])
        if (events['tail_clear'] is None and after['body_on_zone'] is False and
                (before['body_on_zone'] is True or whole_zone_passed)):
            events['tail_clear'] = interval
    first, last = frames[0], frames[-1]
    all_front_known = all(frame['front_progress_m'] is not None for frame in frames)
    first_progress, last_progress = first['front_progress_m'], last['front_progress_m']
    censoring = {}
    for event, threshold, strict in (('front_entry', zone[0], False), ('front_exit', zone[1], True)):
        if events[event] is not None:
            censoring[event] = None
        elif first_progress is not None and (first_progress > threshold if strict else first_progress >= threshold):
            censoring[event] = 'left'
        elif all_front_known and (last_progress <= threshold if strict else last_progress < threshold):
            censoring[event] = 'right'
        else:
            censoring[event] = 'unresolved'
    if events['tail_clear'] is not None:
        censoring['tail_clear'] = None
    elif first['body_on_zone'] is False and first_progress is not None and first_progress > zone[1]:
        censoring['tail_clear'] = 'left'
    elif last['body_on_zone'] is True or (all_front_known and last_progress < zone[0]):
        censoring['tail_clear'] = 'right'
    else:
        censoring['tail_clear'] = 'unresolved'
    by_interval = defaultdict(list)
    for event, interval in events.items():
        if interval is not None:
            by_interval[tuple(interval)].append(event)
    return dict(connection_key=connection['key'], zone_interval_m=zone, frames=frames,
                front_entry_interval_sec=events['front_entry'], front_exit_interval_sec=events['front_exit'],
                tail_clear_interval_sec=events['tail_clear'], censoring=censoring,
                same_interval_event_groups=[dict(interval_sec=list(interval), events=names,
                                                within_interval_order='unresolved')
                                            for interval, names in by_interval.items() if len(names) > 1],
                timing_convention='(previous_frame_time,current_frame_time]; no subsecond interpolation.',
                off_path_rule='No front-progress extrapolation; body clearance requires explicit complete body-trail evidence.')
