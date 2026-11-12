"""Shared discrete-braking binding; receiving and conflict rules stay frozen."""
from experiments.tsc_packet_mean_alignment_20261110.motion import MotionResolver as FrozenMotionResolver
from experiments.tsc_packet_same_step_receiving_20261110.motion import (
    build_context, _body_constraint, _priority_order,
)
from .physics import ZERO, _proposal, physical_proposal


class MotionResolver(FrozenMotionResolver):
    def raw(self, key):
        cache = self.raw_cache
        if self.decisions is self.context['zero_decisions'] and key != self.focal_id:
            cache = self.context['zero_raw_cache']
        if key not in cache:
            connection, compatible = self.connection(key)
            value = physical_proposal(self.kernel, self.old[key], self.decisions[key], self.view(key), connection, compatible)
            cache[key] = value if self.permissions[key]['start'] else 0.
        return cache[key]

    def __getitem__(self, key):
        """Same prestate/body priority forecast, using the new braking inverse."""
        cache = self.forecast_cache
        if self.decisions is self.context['zero_decisions'] and key != self.focal_id:
            cache = self.context['zero_forecast_cache']
        if key not in cache:
            packet = self.old[key]
            connection, compatible = self.connection(key)
            bound, unused = _body_constraint(self.context, self.kernel, packet, connection, compatible,
                                               self.context['body_intervals'], ())
            value = _proposal(self.kernel, packet, self.decisions[key], self.view(key), connection, compatible, bound)
            cache[key] = value if self.permissions[key]['start'] else 0.
        return cache[key]


def resolve_forward(kernel, old, decisions):
    context = build_context(kernel)
    permissions = {key: dict(start=(packet['speed_mps'] >= .1 or
                                    kernel._decision_uniforms[key]['start'] < decisions[key]['start_probability']),
                             **{'yield': kernel._decision_uniforms[key]['yield'] < decisions[key]['yield_probability']})
                   for key, packet in old.items()}
    resolver = MotionResolver(context, kernel, decisions, permissions)
    ordered = _priority_order(resolver)
    resolver.priority_index = {key: index for index, key in enumerate(ordered)}
    for key in ordered:
        resolver.resolve(key)
    exited, events = [], []
    for key in resolver.order:
        packet, result = old[key], resolver.results[key]
        if result['ended']:
            exited.append(key)
            lane = result['projected_packet']['lane_id']
            events.append(dict(kind='terminal_exit', vehicle_id=key, from_lane=lane,
                               nonsink=bool(kernel.topology.connections_from(lane))))
            kernel.state['intents'].pop(key, None)
        elif len(result['trace']) > 1:
            events.append(dict(kind='lane_transition', vehicle_id=key,
                               observed_lane_path=result['trace'], transition_type='connection'))
        kernel.diagnostics['spatial_conflict_limited_steps'] += int(result['spatial_summary']['active_conflict'])
        kernel.diagnostics['unresolved_spatial_pairs'] += result['spatial_summary']['unresolved_pairs']
        kernel.diagnostics['stronger_than_nominal_braking_steps'] += int(
            result['travel_m'] < packet['speed_mps']-packet['parameters']['decel_mps2'])
    kernel.diagnostics['receiving_dependency_cycle_edges'] += len(resolver.cycle_edges)
    kernel._last_receiving_resolution = dict(priority_order=ordered, accepted_order=resolver.order,
                                           processed=resolver.processed, results=resolver.results)
    return resolver.working, exited, events
