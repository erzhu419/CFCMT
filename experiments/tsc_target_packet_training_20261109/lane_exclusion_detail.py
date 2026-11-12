"""Explain fixed lane-mask exclusions from current state; no fit or rollout."""
from copy import deepcopy
import math

from .conditional_features import (
    prepare_feature_kernel, candidate_features, _project_front,
)


def _packet(packet):
    if packet is None:
        return None
    return dict(packet_id=packet['packet_id'], lane_id=packet['lane_id'],
                position_m=packet['position_m'], speed_mps=packet['speed_mps'],
                length_m=packet['parameters']['length_m'],
                min_gap_m=packet['parameters']['min_gap_m'])


def _gaps(packet, target, packets, lane_length):
    """Use the helper's same-clock front/rear definitions and thresholds."""
    other = [p for key, p in packets.items()
             if key != packet['packet_id'] and p['lane_id'] == target]
    x = packet['position_m']
    ahead = [p for p in other if p['position_m'] >= x]
    behind = [p for p in other if p['position_m'] < x]
    front = min(ahead, key=lambda p: p['position_m']-p['parameters']['length_m']) if ahead else None
    rear = max(behind, key=lambda p: p['position_m']) if behind else None
    front_gap = front['position_m']-front['parameters']['length_m']-x if front else lane_length-x
    rear_gap = x-packet['parameters']['length_m']-rear['position_m'] if rear else x
    rear_required = rear['parameters']['min_gap_m'] if rear else 0.
    return dict(front_vehicle=_packet(front), rear_vehicle=_packet(rear),
                front_gap_m=front_gap, rear_gap_m=rear_gap,
                required_front_gap_m=packet['parameters']['min_gap_m'],
                required_rear_gap_m=rear_required,
                front_blocked=bool(front is not None and front_gap < packet['parameters']['min_gap_m']),
                rear_blocked=bool(rear is not None and rear_gap < rear_required))


def _finite(value):
    return value if math.isfinite(value) else None


def _direction_detail(kernel, packet, connection, compatible, upper, intent, direction):
    projected = _project_front(kernel, packet, connection, compatible, upper)
    result = dict(projected=_packet(projected), target_lane=None, reasons=[], same_clock_gaps=None)
    if projected is None:
        result['reasons'].append('projected_terminal')
        return result
    geometry = kernel.topology.lanes[projected['lane_id']]
    if geometry['internal']:
        result['reasons'].append('projected_internal')
        return result
    target = kernel.topology.adjacent(projected['lane_id'], 1 if direction == 'left' else -1)
    result['target_lane'] = target
    if target is None:
        result['reasons'].append('no_adjacent_lane')
        return result
    target_geometry = kernel.topology.lanes[target]
    result['target_lane_length_m'] = target_geometry['length_m']
    if projected['position_m'] > target_geometry['length_m']:
        result['reasons'].append('beyond_target_lane_length')
        return result
    old_edge = kernel.topology.lanes[packet['lane_id']]['edge_id']
    result['same_road'] = old_edge == geometry['edge_id']
    if result['same_road']:
        result['goal_distance_before'] = _finite(kernel._road_goal_distance(packet['lane_id'], intent))
        result['goal_distance_after'] = _finite(kernel._road_goal_distance(target, intent))
        result['lateral_legal'] = bool(kernel._lateral_legal(packet, target))
        if not result['lateral_legal']:
            result['reasons'].append('lateral_goal_away')
    gaps = _gaps(projected, target, kernel._conditional_next_packets, target_geometry['length_m'])
    result['same_clock_gaps'] = gaps
    if gaps['front_blocked']:
        result['reasons'].append('front_gap')
    if gaps['rear_blocked']:
        result['reasons'].append('rear_gap')
    return result


def diagnose_lane(static, executor_seeds, laws, frame, current, action, reference):
    """Trace a frozen left/right label; ``current`` is poststep evidence only.

    All candidate predictions are finalized before poststep fields are read.
    Actual positions and the observed path appear only in ``actual``.
    """
    seed, time_s = int(reference['seed']), int(reference['time_sec'])
    pid, label = reference['packet_id'], int(reference['y'])
    if label not in (0, 2):
        raise ValueError('Lane exclusion detail requires a left0 or right2 event')
    if frame['time_s'] != time_s or action['time_sec'] != time_s:
        raise ValueError('The frozen reference must match the current frame/action clock')
    direction = 'left' if label == 0 else 'right'
    feature_seed = 76116999+seed*100000+time_s
    kernel = prepare_feature_kernel(static, frame, executor_seeds, laws, action['requests'], feature_seed)
    if kernel.signals != action['applied_states']:
        raise ValueError('Recorded current actions must reproduce the current applied signals')
    packet = kernel.state['packets'][pid]
    geometry = kernel.topology.lanes[packet['lane_id']]
    priors = ({kernel._connections[pid][0]['to_edge']: 1.} if geometry['internal']
              else laws['intent_probs'][geometry['edge_id']])
    candidates = []
    for intent, prior in priors.items():
        features = candidate_features(kernel, pid, intent)
        previous_intent = deepcopy(kernel.state['intents'].get(pid))
        had_intent = pid in kernel.state['intents']
        previous_connection = kernel._connections[pid]
        try:
            if not geometry['internal']:
                kernel.state['intents'][pid] = dict(road_id=geometry['edge_id'], choice=intent)
            connection, compatible = kernel._connection(packet)
            kernel._connections[pid] = connection, compatible
            detail = _direction_detail(kernel, packet, connection, compatible,
                                       features['upper'], intent, direction)
            allowed = bool(features['lane_mask'][direction])
            if allowed != (not detail['reasons']):
                raise ValueError('Diagnostic trace disagrees with the frozen lane mask')
            candidates.append(dict(intent=intent, prior=prior, upper_mps=features['upper'],
                compatible=compatible, lane_mask=features['lane_mask'],
                lane_risk=features['lane_mask']['left'] or features['lane_mask']['right'],
                observed_direction_allowed=allowed, detail=detail,
                connection=None if connection is None else {key: connection[key] for key in
                    ('key', 'from_lane', 'to_lane', 'path', 'tls_id', 'link_index')}))
        finally:
            if had_intent:
                kernel.state['intents'][pid] = previous_intent
            else:
                kernel.state['intents'].pop(pid, None)
            kernel._connections[pid] = previous_connection

    # Poststep truth is used only below, after every prediction is complete.
    if current['time_s'] != time_s+1:
        raise ValueError('Poststep diagnostic evidence must be exactly one second later')
    observed = current['packets'].get(pid)
    path = next((event['observed_lane_path'] for event in current['events']
                 if event.get('vehicle_id') == pid and event['kind'] == 'lane_transition'),
                [packet['lane_id']] if observed is not None else [])
    lateral_steps = [dict(from_lane=a, to_lane=b) for a, b in zip(path, path[1:])
                     if not kernel.topology.lanes[a]['internal']
                     and kernel.topology.lanes[a]['edge_id'] == kernel.topology.lanes[b]['edge_id']
                     and kernel.topology.lanes[a]['index'] != kernel.topology.lanes[b]['index']]
    actual = dict(packet=_packet(observed), observed_lane_path=path, lateral_steps=lateral_steps,
                  crossed_road=any(kernel.topology.lanes[a]['edge_id'] != kernel.topology.lanes[b]['edge_id']
                                   for a, b in zip(path, path[1:])), same_clock_gaps=None)
    if observed is not None and not kernel.topology.lanes[observed['lane_id']]['internal']:
        actual['same_clock_gaps'] = _gaps(observed, observed['lane_id'], current['packets'],
                                            kernel.topology.lanes[observed['lane_id']]['length_m'])
    return dict(reference=dict(reference), observed_direction=direction, feature_seed=feature_seed,
                current_packet=_packet(packet), candidates=candidates, actual=actual,
                new_native_starts=0, new_fits=0)
