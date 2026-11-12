"""Exact focal response of the local shared resolver, before any optimizer.

Only the focal residual varies. Other residuals stay zero and start/yield gates
remain potential permissions. This is not a full-global-forward expectation;
local dependencies and a complete globally ordered step can differ. The
response can be nonsmooth and nonmonotone when reservation timing changes.
"""
from math import isfinite

from experiments.tsc_packet_same_step_receiving_20261110.motion import (
    MotionResolver, ZERO, physical_proposal,
)


def make_conditional_resolver(context, kernel, packet_id, connection, compatible, residual_mps=0.):
    """Create a fresh local query; no accepted-motion memo crosses focal queries."""
    residual = float(residual_mps)
    if not isfinite(residual):
        raise ValueError('The focal residual must be finite')
    if kernel._connections[packet_id] != (connection, compatible):
        raise ValueError('Install the focal current intent/connection before querying its mean')
    if residual == 0.:
        decisions = context['zero_decisions']
    else:
        decisions = dict(context['zero_decisions'])
        decisions[packet_id] = {'motion_residual_mps': residual}
    return MotionResolver(context, kernel, decisions, context['permit_all'], focal_id=packet_id)


def conditional_motion_mean(context, kernel, packet_id, connection, compatible, residual_mps=0.):
    """Return mu(r) by residual-before-projection, without shifting or capping mu(0)."""
    resolver = make_conditional_resolver(context, kernel, packet_id, connection, compatible, residual_mps)
    result = dict(resolver.resolve(packet_id))
    packet = context['packets'][packet_id]
    parameters = packet['parameters']
    # P0, rather than the already projected mu(0), is a physical proposal bound.
    physical_ceiling = physical_proposal(kernel, packet, ZERO, resolver.view(packet_id), connection, compatible)
    result.pop('projected_packet')
    result.pop('trace')
    result.pop('ended')
    result['resolved_packet_count'] = len(resolver.results)
    residual = float(residual_mps)
    return dict(mean_mps=result['travel_m'], requested_residual_mps=residual,
                clipped_residual_mps=max(-parameters['decel_mps2'], min(residual, parameters['accel_mps2'])),
                physical_ceiling_mps=physical_ceiling, local_detail=result)
