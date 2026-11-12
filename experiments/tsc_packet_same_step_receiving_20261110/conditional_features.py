"""Conditional receiving readout on fixed current-state, zero-residual envelopes.

Other road intents remain the frame's frozen prior configuration. Each focal
condition resolves its own required receiving dependencies with a fresh local
memo. This is a feasible-motion upper-envelope approximation, not the
expectation of the full stochastic joint transition.
"""
from .motion import build_context, resolve_conditional


def prepare_receiving_context(kernel):
    """Prepare invariant current-state indices/proposals before focal overrides."""
    context = build_context(kernel)
    kernel._same_step_receiving_context = context
    return context


def _admission_detail(kernel, packet, connection, compatible, view):
    """Use model-resolved same-second bodies; never an observed future frame."""
    result = resolve_conditional(kernel._same_step_receiving_context, kernel,
                                 packet['packet_id'], connection, compatible)
    spatial = result['spatial_summary']
    upper, path_bound = float(result['travel_m']), float(result['same_step_path_bound_m'])
    spatial_bound = float(result['spatial_limit_m'])
    raw = float(result['raw_upper_mps'])
    geometry = kernel.topology.lanes[packet['lane_id']]
    distance = geometry['length_m']-packet['position_m']
    ordinary = not geometry['internal']
    entry_blocked = bool(ordinary and min(path_bound, spatial_bound) < distance)
    return dict(upper=upper, admissible=not entry_blocked if ordinary else True,
        path_bound_m=path_bound, raw_upper_mps=raw,
        current_path_bound_m=float(result['current_path_bound_m']),
        physical_proposal_mps=float(result['physical_proposal_mps']),
        receiving_release_m=float(result['receiving_release_m']),
        conflict=bool(spatial['active_conflict']), active_conflict=bool(spatial['active_conflict']),
        entry_blocked=entry_blocked, spatial_limit_m=spatial_bound,
        can_reach_entry_before_conflict=bool(connection is not None and compatible and ordinary
                                            and raw > 0 and raw >= distance),
        resolved_pairs=int(spatial['resolved_pairs']), no_zone_pairs=int(spatial['no_zone_pairs']),
        unresolved_pairs=int(spatial['unresolved_pairs']),
        receiving_binding_summary=result['receiving_binding_summary'],
        resolved_packet_count=int(result['resolved_packet_count']),
        spatial_details=spatial['details'],
        conditioning='Fixed other intents; zero residual and potential start/yield; locally resolved receiving dependencies.')
