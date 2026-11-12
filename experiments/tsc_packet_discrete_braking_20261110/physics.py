"""One-second physical proposals with discrete following and stop braking.

Only those two safe-speed inverses change. The yellow stopping predicate,
other upper bounds and residual-before-projection order remain frozen.
"""
from experiments.tsc_packet_exclusion_mechanisms_20261110.braking_terms import discrete_safe_speed


ZERO = {'motion_residual_mps': 0.}


def bounded_speed_proposal(packet, residual_mps, *, net_gap_m=None,
                           leader_speed_mps=0., leader_decel_mps2=None,
                           admissible_travel_m=None):
    speed = packet['speed_mps']
    params = packet['parameters']
    accel, decel, tau = (params[name] for name in ('accel_mps2', 'decel_mps2', 'tau_s'))
    upper = min(packet['allowed_speed_mps'], speed+accel)
    if net_gap_m is not None:
        lead_decel = decel if leader_decel_mps2 is None else leader_decel_mps2
        upper = min(upper, discrete_safe_speed(net_gap_m, leader_speed_mps, decel, lead_decel, tau))
    if admissible_travel_m is not None:
        upper = min(upper, max(admissible_travel_m, 0.))
    upper = max(upper, 0.)
    proposal = upper+max(-decel, min(residual_mps, accel))
    return min(upper, max(0., proposal))


def _proposal(kernel, packet, decision, packets, connection, compatible, body_limit):
    """Frozen min terms; body_limit supplies only the old priority forecast."""
    current = dict(packet, allowed_speed_mps=kernel._allowed_speed(packet))
    path = kernel._path(packet, connection, compatible)
    end = max(sum(kernel.topology.lanes[lane]['length_m'] for lane in path)-packet['position_m'], 0.)
    gap, leader = kernel._leader(packet, packets, path)
    kwargs = dict(net_gap_m=gap, admissible_travel_m=end if body_limit is None else min(end, body_limit))
    if leader is not None:
        kwargs.update(leader_speed_mps=leader['speed_mps'], leader_decel_mps2=leader['parameters']['decel_mps2'])
    signal = kernel._signal(connection, compatible)
    ordinary = not kernel.topology.lanes[packet['lane_id']]['internal']
    distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
    blocked = not compatible or signal == 'r'
    if signal in ('y', 'Y') and ordinary:
        blocked = packet['speed_mps']**2/(2*packet['parameters']['decel_mps2']) <= distance
    stop = float('inf')
    if ordinary and connection is not None and blocked:
        stop = bounded_speed_proposal(current, decision['motion_residual_mps'],
                                      net_gap_m=max(distance, 0.), admissible_travel_m=distance)
    return min(stop, bounded_speed_proposal(current, decision['motion_residual_mps'], **kwargs))


def physical_proposal(kernel, packet, decision, packets, connection, compatible):
    """Nominal one-second proposal before same-step body/spatial resolution."""
    return _proposal(kernel, packet, decision, packets, connection, compatible, None)
