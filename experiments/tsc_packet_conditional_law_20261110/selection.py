"""Current-only geometric stochastic set for a conditional local law.

The endpoint/landing envelopes are conservative supersets. Background actors
remain in the complete physical frame; membership only decides which latent
actions are random. No native future, learned score or posthoc denier is read.
"""


def _paths(kernel, packet):
    topology = kernel.topology
    lane = packet['lane_id']
    geometry = topology.lanes[lane]
    if geometry['internal']:
        connection = topology.by_key[packet['_movement_key']]
        return (connection['path'][connection['path'].index(lane):],)
    prior = kernel.laws['intent_probs'].get(geometry['edge_id'])
    if prior is None:
        if any(topology.connections_from(k) for k in topology.road_lanes(geometry['edge_id'])):
            raise ValueError('A reachable road lacks its frozen intent support')
        prior = {'END': 1.}
    paths = []
    for goal, probability in prior.items():
        if probability <= 0:
            continue
        matches = topology.compatible(lane, goal)
        if goal == 'END':
            path = (lane,)
        elif matches:
            path = matches[0]['path']
        elif any(topology.compatible(k, goal) for k in topology.road_lanes(geometry['edge_id'])):
            path = (lane,)  # An incompatible current lane cannot enter yet.
        else:
            raise ValueError('A positive-prior goal has no static road connection')
        if path not in paths:
            paths.append(path)
    if not paths:
        raise ValueError('An actor has no positive current intent support')
    return tuple(paths)


def _envelope(kernel, packet):
    topology = kernel.topology
    upper = max(0., min(kernel._allowed_speed(packet),
                       packet['speed_mps']+packet['parameters']['accel_mps2']))
    fronts = {}
    for path in _paths(kernel, packet):
        limit = min(upper, max(sum(topology.lanes[lane]['length_m'] for lane in path)
                               - packet['position_m'], 0.))
        offset = -packet['position_m']
        for lane in path:
            geometry = topology.lanes[lane]
            length = geometry['length_m']
            if not geometry['internal'] and limit >= max(offset, 0.):
                low, high = max(-offset, 0.), min(length, limit-offset)
                if lane in fronts:
                    low, high = min(low, fronts[lane][0]), max(high, fronts[lane][1])
                fronts[lane] = (float(low), float(high))
            offset += length
    landings = {}
    for lane, (low, high) in fronts.items():
        for target in (lane, topology.adjacent(lane, 1), topology.adjacent(lane, -1)):
            if target is None or low > topology.lanes[target]['length_m']:
                continue
            lo, hi = low, min(high, topology.lanes[target]['length_m'])
            if target in landings:
                lo, hi = min(lo, landings[target][0]), max(hi, landings[target][1])
            landings[target] = (float(lo), float(hi))
    return dict(kinematic_upper_m=upper, longitudinal_fronts=fronts, potential_landing_fronts=landings)


def select_stochastic(kernel, focal_id, *, max_stochastic=3):
    """Choose every direct potential landing interactor, without truncation.

    Both actors' ordinary endpoint intervals expand by one lateral lane. Thus
    an actor initially two lane indices away is included if it can enter the
    focal's landing lane. Intersections use actual lengths and asymmetric
    minimum gaps. The rule is not recursive stochastic closure: outside J is
    explicitly conditioned to stay, with its physical response still resolved.
    """
    packets = kernel.state['packets']
    ids = list(packets)
    result = dict(status=None, focal_id=focal_id, participants=[], required_participants=[],
                  max_stochastic=max_stochastic, bounds={}, reasons={})
    try:
        result['bounds'] = {key: _envelope(kernel, packet) for key, packet in packets.items()}
    except ValueError as error:
        result.update(status='OBSTRUCTION_GEOMETRIC_INTENT_DOMAIN', reason=str(error))
        return result
    focal = packets[focal_id]
    required = {focal_id}
    result['reasons'][focal_id] = ['focal']
    focal_ranges = result['bounds'][focal_id]['potential_landing_fronts']
    for key, packet in packets.items():
        if key == focal_id:
            continue
        matching = []
        for lane, (low, high) in result['bounds'][key]['potential_landing_fronts'].items():
            if lane not in focal_ranges:
                continue
            focal_low, focal_high = focal_ranges[lane]
            if (high >= focal_low-focal['parameters']['length_m']-packet['parameters']['min_gap_m']
                    and low <= focal_high+packet['parameters']['length_m']+focal['parameters']['min_gap_m']):
                matching.append(lane)
        if matching:
            required.add(key)
            result['reasons'][key] = matching
    result['required_participants'] = [key for key in ids if key in required]
    result['required_count'] = len(required)
    if len(required) > max_stochastic:
        result.update(status='OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP',
                      reason='The full geometry-selected set exceeds the declared enumeration limit')
        return result
    result.update(status='SELECTED_GEOMETRIC_ENVELOPE', participants=list(result['required_participants']), reason=None)
    return result
