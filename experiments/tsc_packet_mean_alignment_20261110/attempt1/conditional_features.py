"""Full 38-feature, zero-parameter local working likelihood readout.

Longitudinal neighbours use the same query's accepted one-second motion.
Other lateral actions stay fixed; receiving-road legality is the union of
positive-prior next-road intents. No next observation enters this module.
"""
from collections import defaultdict
from copy import deepcopy

from experiments.tsc_packet_same_step_receiving_20261110.kernel import PacketKernel
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation
from experiments.tsc_target_packet_training_20261109.conditional_features import _context_view
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_same_step_receiving_20261110.motion import build_context, physical_proposal, ZERO
from .mean import make_conditional_resolver


def prepare_receiving_context(kernel):
    """Freeze current-prior configurations; cache no focal accepted states."""
    context = build_context(kernel)
    # Endpoint ranges are only an inexpensive superset for lateral blockers.
    # They use current physical proposals, never observed future positions.
    reaches = defaultdict(list)
    for key, packet in context['packets'].items():
        connection, compatible = context['connections'][key]
        path = kernel._path(packet, connection, compatible)
        keys = set().union(*(context['front_index'].get(lane, set()) for lane in path))
        view = {other: context['packets'][other] for other in keys}
        proposal = physical_proposal(kernel, packet, ZERO, view, connection, compatible)
        context['zero_raw_cache'][key] = proposal
        offset = -packet['position_m']
        for index, lane in enumerate(path):
            length = kernel.topology.lanes[lane]['length_m']
            if proposal >= offset and not kernel.topology.lanes[lane]['internal']:
                low = packet['position_m'] if index == 0 else 0.
                high = min(length, max(proposal-offset, low))
                reaches[lane].append((key, low, high))
            offset += length
    context['lane_endpoint_ranges'] = reaches
    kernel._mean_alignment_context = context
    return context


def prepare_feature_kernel(static, frame, executor_seeds, laws, requests, random_seed, *, conflict_geometry):
    """Use the canonical context function with the frozen prior draw order."""
    initial = initialize_observation(frame, frame['time_s'])
    kernel = PacketKernel(static, initial, executor_seeds, laws, random_seed=random_seed,
                          conflict_geometry=conflict_geometry)
    for tls, request in sorted(requests.items()):
        kernel.executors[tls].request(request)
    packets = kernel.state['packets']
    kernel._connections = {key: kernel._connection(packet) for key, packet in packets.items()}
    lane_packets, tls_packets = defaultdict(set), defaultdict(set)
    for key, packet in packets.items():
        lane_packets[packet['lane_id']].add(key)
        connection, unused = kernel._connections[key]
        if connection is not None and connection['tls_id'] is not None:
            tls_packets[connection['tls_id']].add(key)
    kernel._conditional_lane_packets = lane_packets
    kernel._conditional_tls_packets = tls_packets
    configure_geometry(kernel, conflict_geometry)
    prepare_receiving_context(kernel)
    return kernel


def _admission_result(kernel, packet, connection, compatible, resolver):
    result = resolver.resolve(packet['packet_id'])
    spatial = result['spatial_summary']
    geometry = kernel.topology.lanes[packet['lane_id']]
    distance = geometry['length_m']-packet['position_m']
    ordinary = not geometry['internal']
    bound, spatial_bound = result['same_step_path_bound_m'], result['spatial_limit_m']
    entry_blocked = bool(ordinary and min(bound, spatial_bound) < distance)
    detail = dict(upper=float(result['travel_m']), mean0_mps=float(result['travel_m']),
        physical_ceiling_mps=float(resolver.raw(packet['packet_id'])),
        admissible=not entry_blocked if ordinary else True, entry_blocked=entry_blocked,
        path_bound_m=float(bound), current_path_bound_m=float(result['current_path_bound_m']),
        raw_upper_mps=float(result['raw_upper_mps']), spatial_limit_m=float(spatial_bound),
        receiving_release_m=float(result['receiving_release_m']),
        active_conflict=bool(spatial['active_conflict']),
        receiving_binding_summary=result['receiving_binding_summary'],
        resolved_packet_count=len(resolver.results))
    return detail, result


def _potential_legal(kernel, projected, target, choice):
    return kernel._road_goal_distance(target, choice) <= kernel._road_goal_distance(projected['lane_id'], choice)


def _lane_mask(kernel, packet, intent, result, resolver):
    mask = {'left': False, 'stay': True, 'right': False}
    summary = dict(clock='accepted longitudinal t+1; other lateral actions stay',
        receiving_road_potential=False, projected_lane_id=None, projected_position_m=None, sides={})
    if result['ended']:
        summary['focal_ended'] = True
        return mask, summary
    projected = result['projected_packet']
    summary.update(projected_lane_id=projected['lane_id'], projected_position_m=projected['position_m'])
    geometry = kernel.topology.lanes[projected['lane_id']]
    if geometry['internal']:
        return mask, summary
    old_edge = kernel.topology.lanes[packet['lane_id']]['edge_id']
    cross_road = old_edge != geometry['edge_id']
    summary['receiving_road_potential'] = cross_road
    if cross_road:
        priors = kernel.laws['intent_probs'].get(geometry['edge_id'], {'END': 1.})
        choices = [choice for choice, prior in priors.items() if prior > 0]
    else:
        choices = [intent]
    context = kernel._mean_alignment_context
    # Resolve the union in a deterministic order before reading either side.
    needed, side_keys = set(), {}
    for name, delta in (('left', 1), ('right', -1)):
        target = kernel.topology.adjacent(projected['lane_id'], delta)
        side = dict(target_lane_id=target, potential_legal=False, resolved_neighbour_ids=[])
        summary['sides'][name] = side
        if target is None or projected['position_m'] > kernel.topology.lanes[target]['length_m']:
            continue
        legal = any(_potential_legal(kernel, projected, target, choice) for choice in choices)
        side['potential_legal'] = legal
        if not legal:
            continue
        keys = set()
        x = projected['position_m']
        for key, low, high in context['lane_endpoint_ranges'].get(target, ()):
            if key == packet['packet_id']:
                continue
            other = context['packets'][key]
            lower = x-packet['parameters']['length_m']-other['parameters']['min_gap_m']
            upper = x+other['parameters']['length_m']+packet['parameters']['min_gap_m']
            if high >= lower and low <= upper:
                keys.add(key)
        needed.update(keys)
        side_keys[name] = (target, keys)
    for key in sorted(needed, key=lambda key: context['packet_order'][key]):
        resolver.resolve(key)
    for name, (target, keys) in side_keys.items():
        side = summary['sides'][name]
        others = {key: resolver.results[key]['projected_packet'] for key in keys
                  if key in resolver.results and not resolver.results[key]['ended']
                  and resolver.results[key]['projected_packet']['lane_id'] == target}
        # A dependency cycle preserves its old body in longitudinal resolution.
        # For this pre-LC mask it likewise retains that unresolved front.
        for key in keys:
            if key not in resolver.results and context['packets'][key]['lane_id'] == target:
                others[key] = context['packets'][key]
        ahead = [other for other in others.values() if other['position_m'] >= projected['position_m']]
        behind = [other for other in others.values() if other['position_m'] < projected['position_m']]
        front, rear, unused = kernel._side_gaps(projected, target, others)
        needed_gap = max(behind, key=lambda row: row['position_m'])['parameters']['min_gap_m'] if behind else 0.
        mask[name] = not ((ahead and front < packet['parameters']['min_gap_m'])
                          or (behind and rear < needed_gap))
        side.update(allowed=bool(mask[name]), front_gap_m=float(front), rear_gap_m=float(rear),
            rear_required_gap_m=float(needed_gap), resolved_neighbour_ids=sorted(others),
            neighbour_states=[dict(packet_id=key, lane_id=other['lane_id'], position_m=other['position_m'],
                speed_mps=other['speed_mps'], length_m=other['parameters']['length_m'],
                min_gap_m=other['parameters']['min_gap_m'], resolved=key in resolver.results)
                for key, other in sorted(others.items())])
    return mask, summary


def candidate_features(kernel, packet_id, intent):
    """Build a theta=0 working component; restore all focal intent state."""
    packet = kernel.state['packets'][packet_id]
    old_intent = deepcopy(kernel.state['intents'].get(packet_id))
    had_intent = packet_id in kernel.state['intents']
    old_connection = kernel._connections[packet_id]
    try:
        geometry = kernel.topology.lanes[packet['lane_id']]
        if not geometry['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=geometry['edge_id'], choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        view = _context_view(kernel, packet, connection, compatible)
        features = kernel._context(packet, view, connection, compatible)
        resolver = make_conditional_resolver(kernel._mean_alignment_context, kernel,
                                             packet_id, connection, compatible)
        admission, result = _admission_result(kernel, packet, connection, compatible, resolver)
        mean = admission['mean0_mps']
        distance = geometry['length_m']-packet['position_m']
        yielding = (not geometry['internal'] and connection is not None and compatible
            and kernel._signal(connection) == 'g' and mean > 0 and mean >= distance and admission['admissible'])
        mask, lane_summary = _lane_mask(kernel, packet, intent, result, resolver)
        return dict(context=features, upper=mean, mean0_mps=mean,
            physical_ceiling_mps=admission['physical_ceiling_mps'],
            a=packet['parameters']['accel_mps2'], b=packet['parameters']['decel_mps2'],
            motion_risk=True, start_risk=mean >= .1, yield_risk=bool(yielding),
            lane_mask=mask, admissible=admission['admissible'],
            admission_detail=admission, lane_detail=lane_summary,
            connection_key=None if connection is None else connection['key'])
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = old_intent
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = old_connection
