"""Frozen shared-beta longitudinal readout; no lateral or future-state queries."""
from copy import deepcopy

import numpy as np

from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_packet_discrete_braking_20261110.physics import ZERO, physical_proposal


class SharedDecisions(dict):
    """Query-local canonical scores for every requested dependency, lazily."""
    def __init__(self, kernel, context, focal_id, beta):
        super().__init__()
        self.kernel, self.context, self.focal_id = kernel, context, focal_id
        self.beta = np.asarray(beta, dtype=float)
        self.zero = not np.any(self.beta)
        self.rows = {}

    def __missing__(self, key):
        kernel, packets = self.kernel, self.context['packets']
        packet = packets[key]
        connection, compatible = (kernel._connections[key] if key == self.focal_id
                                   else self.context['connections'][key])
        # Exactly the full-current-frame feature definition of canonical_features.
        score = 0. if self.zero else float(self.beta @ np.asarray(kernel.model.features(
            kernel._context(packet, packets, *kernel._connections[key])), dtype=float))
        parameters = packet['parameters']
        residual = float(np.clip(score, -parameters['decel_mps2'], parameters['accel_mps2']))
        base = float(physical_proposal(kernel, packet, ZERO, self.resolver.view(key), connection, compatible))
        self.rows[key] = dict(score=score, residual_mps=residual, base_proposal_mps=base)
        self[key] = dict(motion_residual_mps=residual)
        return self[key]


def _diagnostics(decisions, resolver, focal_id):
    rows = decisions.rows
    focal = rows[focal_id]
    packets = decisions.context['packets']
    return dict(queried_packets=len(rows), resolved_packets=len(resolver.results),
        forecast_packets=len(resolver.forecast_cache),
        background_queried_packets=sum(key != focal_id for key in rows),
        negative_scores=sum(row['score'] < 0 for row in rows.values()),
        positive_scores=sum(row['score'] > 0 for row in rows.values()),
        negative_background_scores=sum(key != focal_id and row['score'] < 0 for key, row in rows.items()),
        lower_clip=sum(row['score'] < -packets[key]['parameters']['decel_mps2'] for key, row in rows.items()),
        upper_clip=sum(row['score'] > packets[key]['parameters']['accel_mps2'] for key, row in rows.items()),
        zero_base_proposal=sum(row['base_proposal_mps'] == 0. for row in rows.values()),
        zero_speed_floor=sum(row['residual_mps'] < 0 and row['base_proposal_mps'] > 0
            and row['base_proposal_mps']+row['residual_mps'] <= 0 for row in rows.values()),
        focal_score=focal['score'], focal_clipped_residual_mps=focal['residual_mps'])


def _arm(kernel, context, packet_id, beta, connection, compatible):
    packet = context['packets'][packet_id]
    decisions = SharedDecisions(kernel, context, packet_id, beta)
    resolver = MotionResolver(context, kernel, decisions, context['permit_all'], focal_id=packet_id)
    decisions.resolver = resolver
    result = resolver.resolve(packet_id)
    geometry = kernel.topology.lanes[packet['lane_id']]
    ordinary = not geometry['internal']
    distance = geometry['length_m']-packet['position_m']
    entry_blocked = bool(ordinary and min(result['same_step_path_bound_m'], result['spatial_limit_m']) < distance)
    admissible = not entry_blocked if ordinary else True
    mean = float(result['travel_m'])
    yielding = bool(ordinary and connection is not None and compatible
        and kernel._signal(connection) == 'g' and mean > 0 and mean >= distance and admissible)
    answer = dict(mean_mps=mean, physical_ceiling_mps=decisions.rows[packet_id]['base_proposal_mps'],
        residual_physical_proposal_mps=float(result['physical_proposal_mps']),
        start_risk=bool(mean >= .1), yield_risk=yielding, admissible=admissible,
        entry_blocked=entry_blocked, connection_key=None if connection is None else connection['key'],
        path_bound_m=float(result['same_step_path_bound_m']), spatial_limit_m=float(result['spatial_limit_m']),
        active_conflict=bool(result['spatial_summary']['active_conflict']),
        diagnostics=_diagnostics(decisions, resolver, packet_id))
    return answer, resolver


def query_pair(kernel, packet_id, intent, beta):
    """Compare beta0 and frozen beta at one original focal/prior component.

    The original frame's background connections and current-only indices are
    retained. Only the focal connection is conditioned, as in the saved audit.
    Fresh resolver/decision caches prevent accepted motion crossing queries.
    """
    packet = kernel.state['packets'][packet_id]
    had_intent = packet_id in kernel.state['intents']
    old_intent = deepcopy(kernel.state['intents'].get(packet_id))
    old_connection = kernel._connections[packet_id]
    frozen = kernel._mean_alignment_context
    context = dict(frozen, zero_raw_cache={}, zero_forecast_cache={})
    try:
        geometry = kernel.topology.lanes[packet['lane_id']]
        if not geometry['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=geometry['edge_id'], choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        baseline, before = _arm(kernel, context, packet_id, np.zeros_like(beta), connection, compatible)
        candidate, after = _arm(kernel, context, packet_id, beta, connection, compatible)
        both = before.results.keys() & after.results.keys()
        changes = {key: after.results[key]['travel_m']-before.results[key]['travel_m'] for key in both}
        comparison = dict(resolved_both=len(both),
            baseline_only_resolved=len(before.results.keys()-after.results.keys()),
            candidate_only_resolved=len(after.results.keys()-before.results.keys()),
            changed_accepted_responses=sum(abs(value) > 1e-8 for value in changes.values()),
            changed_background_accepted_responses=sum(key != packet_id and abs(value) > 1e-8 for key, value in changes.items()),
            max_abs_accepted_motion_change_mps=max(map(abs, changes.values()), default=0.),
            mean_signed_accepted_motion_change_mps=float(np.mean(list(changes.values()))) if changes else 0.)
        return dict(baseline=baseline, candidate=candidate, comparison=comparison)
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = old_intent
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = old_connection
