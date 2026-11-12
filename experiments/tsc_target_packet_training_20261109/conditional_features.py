"""Current-state conditional features; no fitted model or future state access."""
from collections import defaultdict
from copy import deepcopy

from experiments.tsc_target_packet_kernel_20261109.kernel import PacketKernel
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation


def prepare_feature_kernel(static, frame, executor_seeds, laws, requests, random_seed):
    """Freeze other packets' prior intents once, then condition on observed a_t.

    Road-level priors retain lane-incompatible intentions: changing lanes can
    still realize them. No individual observed future movement conditions them.
    """
    initial = initialize_observation(frame, frame['time_s'])
    kernel = PacketKernel(static, initial, executor_seeds, laws, random_seed=random_seed)
    for tls, request in sorted(requests.items()):
        kernel.executors[tls].request(request)
    packets = kernel.state['packets']
    kernel._connections = {key: kernel._connection(packet) for key, packet in packets.items()}
    lane_packets, tls_packets = defaultdict(set), defaultdict(set)
    for key, packet in packets.items():
        lane_packets[packet['lane_id']].add(key)
        connection, compatible = kernel._connections[key]
        if connection is not None and connection['tls_id'] is not None:
            tls_packets[connection['tls_id']].add(key)
    kernel._conditional_lane_packets = lane_packets
    kernel._conditional_tls_packets = tls_packets
    kernel._conditional_zero_speeds = {}
    # The lateral mask belongs after longitudinal propagation. Predict every
    # neighbour at that same clock from current state only; never combine the
    # projected focal position with an unadvanced neighbouring vehicle.
    predicted = {}
    for key, packet in packets.items():
        connection, compatible = kernel._connections[key]
        view = _context_view(kernel, packet, connection, compatible)
        upper, unused = _admission_upper(kernel, packet, connection, compatible, view)
        projected = _project_front(kernel, packet, connection, compatible, upper)
        if projected is not None:
            predicted[key] = dict(projected, speed_mps=upper)
    kernel._conditional_next_packets = predicted
    return kernel


def _lane_view(kernel, lanes):
    keys = set()
    for lane in lanes:
        keys.update(kernel._conditional_lane_packets[lane])
    return {key: kernel.state['packets'][key] for key in keys}


def _context_view(kernel, packet, connection, compatible):
    lanes = set(kernel._path(packet, connection, compatible))
    for lane in list(lanes):
        geometry = kernel.topology.lanes[lane]
        if not geometry['internal']:
            lanes.update(kernel.topology.road_lanes(geometry['edge_id']))
    view = _lane_view(kernel, lanes)
    if connection is not None and connection['tls_id'] is not None:
        for key in kernel._conditional_tls_packets[connection['tls_id']]:
            view[key] = kernel.state['packets'][key]
    view[packet['packet_id']] = packet
    return view


def _zero_speed(kernel, packet):
    key = packet['packet_id']
    if key not in kernel._conditional_zero_speeds:
        connection, compatible = kernel._connections[key]
        view = _lane_view(kernel, kernel._path(packet, connection, compatible))
        kernel._conditional_zero_speeds[key] = kernel._speed(
            packet, {'motion_residual_mps': 0.}, view, connection, compatible)
    return kernel._conditional_zero_speeds[key]


def _admission_upper(kernel, packet, connection, compatible, view):
    base = kernel._speed(packet, {'motion_residual_mps': 0.}, view, connection, compatible)
    ordinary = not kernel.topology.lanes[packet['lane_id']]['internal']
    distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
    admissible = True
    if connection is not None and compatible and ordinary and base > 0 and base >= distance:
        interval = kernel._reservation(packet, connection, base)
        blockers = []
        signal = kernel._signal(connection)
        for key, other in view.items():
            if key == packet['packet_id']:
                continue
            candidate, legal = kernel._connections[key]
            if candidate is None or not legal or not kernel.topology.foes(connection, candidate):
                continue
            internal = kernel.topology.lanes[other['lane_id']]['internal']
            other_signal = kernel._signal(candidate)
            if internal or (other_signal in ('G', 'g') and kernel._gives_way(connection, signal, candidate, other_signal)):
                speed = _zero_speed(kernel, other)
                if internal or speed > 0:
                    blockers.append((candidate, kernel._reservation(other, candidate, speed)))
        receiving = kernel._receiving_space(connection, view)
        admissible = (receiving >= packet['parameters']['length_m']+packet['parameters']['min_gap_m']
                      and not kernel._conflict(connection, interval, blockers))
        if not admissible:
            base = min(base, max(distance-1e-9, 0.))
    return base, admissible


def _project_front(kernel, packet, connection, compatible, distance):
    projected = dict(packet)
    path = kernel._path(packet, connection, compatible)
    traversed = [packet['lane_id']]
    for index, lane in enumerate(path):
        remaining = kernel.topology.lanes[lane]['length_m']-projected['position_m']
        if distance < remaining-1e-12:
            projected['position_m'] += distance
            return projected
        distance -= max(remaining, 0.)
        projected['position_m'] += max(remaining, 0.)
        if connection is None:
            return None  # The candidate END would leave this ordinary road.
        if index+1 == len(path):
            return projected
        projected['lane_id'], projected['position_m'] = path[index+1], 0.
        traversed.append(path[index+1])
        if distance <= 1e-12:
            return projected
    return projected


def _lane_mask(kernel, packet, connection, compatible, upper):
    mask = {'left': False, 'stay': True, 'right': False}
    projected = _project_front(kernel, packet, connection, compatible, upper)
    if projected is None or kernel.topology.lanes[projected['lane_id']]['internal']:
        return mask
    old_edge = kernel.topology.lanes[packet['lane_id']]['edge_id']
    projected_edge = kernel.topology.lanes[projected['lane_id']]['edge_id']
    neighbours = kernel._conditional_next_packets
    for name, delta in (('left', 1), ('right', -1)):
        target = kernel.topology.adjacent(projected['lane_id'], delta)
        if target is None or projected['position_m'] > kernel.topology.lanes[target]['length_m']:
            continue
        if old_edge == projected_edge and not kernel._lateral_legal(packet, target):
            continue
        front, rear, unused = kernel._side_gaps(projected, target, neighbours)
        others = [other for key, other in neighbours.items() if key != packet['packet_id'] and other['lane_id'] == target]
        ahead = [other for other in others if other['position_m'] >= projected['position_m']]
        behind = [other for other in others if other['position_m'] < projected['position_m']]
        needed = max(behind, key=lambda p: p['position_m'])['parameters']['min_gap_m'] if behind else 0.
        mask[name] = not ((ahead and front < packet['parameters']['min_gap_m']) or (behind and rear < needed))
    return mask


def candidate_features(kernel, packet_id, intent):
    """Condition one packet on k, then restore its intent and connection.

    The upper/risk operators use the same pre-state and zero-residual physical
    proposals. The full free-rollout joint resolver may impose further limits.
    This is not the expectation of that full stochastic transition.
    """
    packet = kernel.state['packets'][packet_id]
    previous = deepcopy(kernel.state['intents'].get(packet_id))
    had_intent = packet_id in kernel.state['intents']
    old_connection = kernel._connections[packet_id]
    try:
        edge = kernel.topology.lanes[packet['lane_id']]['edge_id']
        if not kernel.topology.lanes[packet['lane_id']]['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=edge, choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        view = _context_view(kernel, packet, connection, compatible)
        context = kernel._context(packet, view, connection, compatible)
        upper, admissible = _admission_upper(kernel, packet, connection, compatible, view)
        distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
        ordinary = not kernel.topology.lanes[packet['lane_id']]['internal']
        yielding = (ordinary and connection is not None and compatible
                    and kernel._signal(connection) == 'g' and upper > 0 and upper >= distance and admissible)
        mask = _lane_mask(kernel, packet, connection, compatible, upper)
        return dict(context=context, upper=upper, a=packet['parameters']['accel_mps2'],
            b=packet['parameters']['decel_mps2'], motion_risk=True, start_risk=upper >= .1,
            yield_risk=bool(yielding), lane_mask=mask)
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = previous
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = old_connection
