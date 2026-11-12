"""Current-only decomposition of the frozen zero-residual physical proposal.

Receiving/body and spatial reservation constraints are separate from this P.
"""
from math import sqrt


def _safe_speed(decel, tau, gap, leader_speed, leader_decel):
    return max(0., -decel*tau + sqrt((decel*tau)**2 + 2*decel*max(gap, 0.)
                                   + decel/leader_decel*leader_speed**2))


def proposal_terms(kernel, packet, connection, compatible, packets):
    """Reproduce physical_proposal(..., residual=0) from current inputs only.

    ``packets`` is the same current front-neighbour view supplied to the shared
    proposal. Inactive terms use None, so the readout has no infinite scalars.
    Binding uses exact equality and retains all ties; no threshold is added.
    """
    parameters = packet['parameters']
    speed = packet['speed_mps']
    accel, decel, tau = (parameters[name] for name in ('accel_mps2', 'decel_mps2', 'tau_s'))
    allowed = kernel._allowed_speed(packet)
    path = kernel._path(packet, connection, compatible)
    remaining = max(sum(kernel.topology.lanes[lane]['length_m'] for lane in path)-packet['position_m'], 0.)
    gap, leader = kernel._leader(packet, packets, path)
    leader_safe = None if leader is None else _safe_speed(decel, tau, gap,
        leader['speed_mps'], leader['parameters']['decel_mps2'])
    terms = dict(allowed_speed=allowed, current_speed_plus_acceleration=speed+accel,
                 leader_safe=leader_safe, finite_path=remaining, signal_stop=None)
    signal = kernel._signal(connection, compatible)
    ordinary = not kernel.topology.lanes[packet['lane_id']]['internal']
    distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
    blocked = not compatible or signal == 'r'
    yellow_can_stop = None
    if signal in ('y', 'Y') and ordinary:
        yellow_can_stop = speed**2/(2*decel) <= distance
        blocked = yellow_can_stop
    active = bool(ordinary and connection is not None and blocked)
    stop_components = None
    if active:
        stop_components = dict(allowed_speed=allowed, current_speed_plus_acceleration=speed+accel,
            stationary_stop_safe=_safe_speed(decel, tau, max(distance, 0.), 0., decel),
            stopline_distance=max(distance, 0.))
        terms['signal_stop'] = max(min(stop_components.values()), 0.)
    proposal = max(min(value for value in terms.values() if value is not None), 0.)
    if not ordinary:
        branch = 'internal_no_signal_stop'
    elif connection is None:
        branch = 'no_connection'
    elif not compatible:
        branch = 'incompatible_connection'
    elif signal == 'r':
        branch = 'red'
    elif yellow_can_stop is not None:
        branch = 'yellow_can_stop' if yellow_can_stop else 'yellow_cannot_stop'
    else:
        branch = 'signal_allows_motion'
    return dict(residual_mps=0., physical_proposal_mps=proposal, terms_mps=terms,
        binding_terms=[name for name, value in terms.items() if value == proposal],
        path_lane_ids=list(path), finite_path_remaining_m=remaining,
        leader=dict(packet_id=None if leader is None else leader['packet_id'], net_gap_m=gap,
            speed_mps=None if leader is None else leader['speed_mps'],
            parameters=None if leader is None else dict(leader['parameters']), safe_speed_mps=leader_safe),
        signal_stop=dict(signal=signal, ordinary=ordinary, compatible=bool(compatible),
            active=active, branch=branch, distance_to_lane_end_m=distance,
            stopping_distance_m=speed**2/(2*decel), components_mps=stop_components,
            term_mps=terms['signal_stop'], binding_terms=[] if not active else
            [name for name, value in stop_components.items() if value == terms['signal_stop']]),
        current_body_cap_applied=False, receiving_constraint='separate same-step body resolution')
