"""Local resolver binding for the corrected processed-movement insertion.

The inherited motion, receiving, memo and precedence rules remain frozen.
"""
from experiments.tsc_packet_same_step_receiving_20261110.motion import (
    MotionResolver as FrozenMotionResolver, build_context, _priority_order,
)
from .conflict import conflict_limit


class MotionResolver(FrozenMotionResolver):
    def spatial(self, key, raw):
        packet = self.old[key]
        connection, compatible = self.connection(key)
        tls = None if connection is None else connection['tls_id']
        keys = set(self.context['tls_index'].get(tls, ()))
        # A focal END prior had no TLS entry in the frozen current index. Its
        # conditioned turn must still participate in recursively queried foes.
        if self.focal_id is not None:
            focal_connection, unused = self.connection(self.focal_id)
            if focal_connection is not None and focal_connection['tls_id'] == tls:
                keys.add(self.focal_id)
        current = {other: self.old[other] for other in keys}
        return conflict_limit(self.kernel, packet, connection, compatible, raw, current,
                              speeds=self, processed=self.processed, priority_index=self.priority_index,
                              priority_speeds=self)


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
