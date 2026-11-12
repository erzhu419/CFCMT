"""Local same-step body dependencies shared by conditional and forward motion."""
from collections import defaultdict
from copy import deepcopy
from functools import cmp_to_key

from experiments.tsc_target_packet_protocol_20261109.packet_contract import bounded_speed_proposal
from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals
from experiments.tsc_packet_spatial_conflict_20261110.conflict import _old_precedes
from .conflict import conflict_limit


ZERO = {'motion_residual_mps': 0.}
PERMIT = {'start': True, 'yield': True}


def _must_stop_at_signal(packet, signal, distance):
    return signal == 'r' or (signal in ('y', 'Y') and
        packet['speed_mps']**2/(2*packet['parameters']['decel_mps2']) <= distance)


def _proposal(kernel, packet, decision, packets, connection, compatible, body_limit):
    """Original prestate leader-safe formula; body_limit only supplies old control."""
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
    stop = float('inf')
    if ordinary and connection is not None and (not compatible or _must_stop_at_signal(packet, signal, distance)):
        stop = bounded_speed_proposal(current, decision['motion_residual_mps'],
                                      net_gap_m=max(distance, 0.), admissible_travel_m=distance)
    return min(stop, bounded_speed_proposal(current, decision['motion_residual_mps'], **kwargs))


def physical_proposal(kernel, packet, decision, packets, connection, compatible):
    """Nominal one-second speed before joint body resolution; no static body cap."""
    return _proposal(kernel, packet, decision, packets, connection, compatible, None)


def build_context(kernel):
    """Cache current-only indices; no query-dependent accepted motion is cached."""
    if not getattr(kernel, '_spatial_geometry', {}).get('entity_width_m'):
        raise ValueError('The fixed explicit entity-width geometry is required')
    packets = kernel.state['packets']
    front, body, tls = defaultdict(set), defaultdict(set), defaultdict(set)
    body_intervals = {}
    for key, packet in packets.items():
        front[packet['lane_id']].add(key)
        connection, compatible = kernel._connections[key]
        if connection is not None and connection['tls_id'] is not None:
            tls[connection['tls_id']].add(key)
        intervals = body_intervals[key] = _body_intervals(kernel.topology, packet)
        for lane, rear, head in intervals:
            if head <= rear:
                continue
            body[lane].add(key)
            if kernel.topology.lanes[lane]['internal']:
                for previous in kernel.topology.internal_connections[lane]:
                    if previous['tls_id'] is not None:
                        tls[previous['tls_id']].add(key)
        width = kernel.topology.lanes[packet['lane_id']].get('width_m')
        if width is not None and packet['position_m']-packet['parameters']['length_m'] <= width:
            for previous in kernel._spatial_receiving_connections.get(packet['lane_id'], ()):
                if previous['tls_id'] is not None:
                    tls[previous['tls_id']].add(key)
    return dict(packets=packets, connections=dict(kernel._connections), front_index=front, body_index=body,
                body_intervals=body_intervals,
                tls_index=tls, zero_decisions={key: ZERO for key in packets},
                permit_all={key: PERMIT for key in packets}, zero_raw_cache={}, zero_forecast_cache={},
                packet_order={key: index for index, key in enumerate(packets)})


def _body_constraint(context, kernel, packet, connection, compatible, intervals, changed):
    path = kernel._path(packet, connection, compatible)
    offsets, end = {}, -packet['position_m']
    keys = set(changed)
    for lane in path:
        offsets[lane] = end
        end += kernel.topology.lanes[lane]['length_m']
        keys.update(context['body_index'].get(lane, ()))
    bound, nearest = max(end, 0.), []
    packet_id = packet['packet_id']
    min_gap = packet['parameters']['min_gap_m']
    for key in keys:
        if key == packet_id:
            continue
        for lane, rear, front in intervals.get(key, ()):
            if lane not in offsets or front <= rear or offsets[lane]+front <= 1e-12:
                continue
            limit = max(offsets[lane]+rear-min_gap, 0.)
            if limit < bound-1e-12:
                bound, nearest = limit, [key]
            elif abs(limit-bound) <= 1e-12:
                nearest.append(key)
    return float(bound), list(dict.fromkeys(nearest))


def _project(kernel, packet, connection, compatible, travel):
    """Pure finite-path walker: never samples a future road intent or RNG."""
    moved = deepcopy(packet)
    path = kernel._path(packet, connection, compatible)
    remaining, trace, ended = max(travel, 0.), [packet['lane_id']], False
    for index, lane in enumerate(path):
        if remaining <= 1e-12:
            break
        rest = kernel.topology.lanes[lane]['length_m']-moved['position_m']
        if remaining < rest-1e-12:
            moved['position_m'] += remaining
            remaining = 0.
            break
        remaining -= max(rest, 0.)
        moved['position_m'] += max(rest, 0.)
        if connection is None and compatible:
            ended = True
            break
        if index+1 == len(path):
            break
        if (index == 0 and not kernel.topology.lanes[lane]['internal'] and
                _must_stop_at_signal(packet, kernel._signal(connection, compatible), rest)):
            # Reaching the stop line does not grant entry into the junction.
            break
        moved['lane_id'], moved['position_m'] = path[index+1], 0.
        trace.append(path[index+1])
        if kernel.topology.lanes[moved['lane_id']]['internal']:
            moved['_movement_key'] = connection['key']
        else:
            moved.pop('_movement_key', None)
        if remaining <= 1e-12:
            break
    actual = max(travel-remaining, 0.)
    if not ended:
        moved.update(speed_mps=actual, acceleration_mps2=actual-packet['speed_mps'],
                     allowed_speed_mps=kernel._allowed_speed(moved))
        kernel._trim_body(moved, list(packet['body_trail'])+trace[1:])
    return moved, trace, ended, actual


class MotionResolver:
    def __init__(self, context, kernel, decisions, permissions, *, focal_id=None, priority_index=None):
        self.context, self.kernel, self.old = context, kernel, context['packets']
        self.decisions, self.permissions = decisions, permissions
        self.focal_id, self.priority_index = focal_id, priority_index
        self.working, self.processed = dict(self.old), {}
        self.working_intervals = dict(context['body_intervals'])
        self.results, self.visiting, self.raw_cache, self.forecast_cache = {}, set(), {}, {}
        self.cycle_edges, self.order = [], []

    def connection(self, key):
        return self.kernel._connections[key] if key == self.focal_id else self.context['connections'][key]

    def view(self, key):
        packet = self.old[key]
        connection, compatible = self.connection(key)
        keys = set()
        for lane in self.kernel._path(packet, connection, compatible):
            keys.update(self.context['front_index'].get(lane, ()))
        return {other: self.old[other] for other in keys}

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
        """Frozen prestate forecasts for unprocessed conflict reservations."""
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

    def precedes(self, foe, ego, foe_connection_key):
        if self.priority_index is not None:
            return self.priority_index[foe] < self.priority_index[ego]
        connection, unused = self.connection(ego)
        # A receiver's tail can still reserve its past movement while its
        # current front intent is END or belongs to the next junction.
        other = self.kernel.topology.by_key[foe_connection_key]
        return _old_precedes(self.kernel, self.old[ego], self.old[foe], connection, other,
                             self[ego], self[foe], None)

    def resolve(self, key):
        if key in self.results:
            return self.results[key]
        if key in self.visiting:
            self.cycle_edges.append(key)
            return None
        self.visiting.add(key)
        packet = self.old[key]
        connection, compatible = self.connection(key)
        proposal = self.raw(key)
        static_bound, unused = _body_constraint(self.context, self.kernel, packet, connection, compatible,
                                                 self.context['body_intervals'], ())
        # Resolve only bodies which can bind this step, front-to-back even when
        # an ordinary receiver occurred later in the old internal-first order.
        attempted = set()
        while True:
            bound, nearest = _body_constraint(self.context, self.kernel, packet, connection, compatible,
                                               self.working_intervals, self.processed)
            if bound >= proposal or any(other in self.processed for other in nearest):
                break
            candidates = [other for other in nearest if other not in attempted]
            if not candidates:
                break
            for other in sorted(candidates, key=lambda pid: self.context['packet_order'][pid]):
                attempted.add(other)
                self.resolve(other)
        # A dependency's own higher-priority conflicts must resolve before its
        # motion is counted as released space. The priority map itself is fixed.
        attempted_foes = set()
        while True:
            spatial = self.spatial(key, proposal)
            foes = [detail['packet_id'] for detail in spatial['details']
                    if detail['packet_id'] not in self.processed and detail['packet_id'] not in attempted_foes
                    and self.precedes(detail['packet_id'], key, detail['connection_key'])]
            if not foes:
                break
            for other in sorted(set(foes), key=lambda pid: self.context['packet_order'][pid]):
                attempted_foes.add(other)
                self.resolve(other)
        bound, nearest = _body_constraint(self.context, self.kernel, packet, connection, compatible,
                                           self.working_intervals, self.processed)
        spatial = self.spatial(key, proposal)
        travel = min(proposal, bound, spatial['limit_m'])
        ordinary = not self.kernel.topology.lanes[packet['lane_id']]['internal']
        distance = self.kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
        if (connection is not None and compatible and ordinary and travel >= distance and
                self.kernel._signal(connection, compatible) == 'g' and not self.permissions[key]['yield']):
            travel = min(travel, max(distance-1e-9, 0.))
        moved, trace, ended, actual = _project(self.kernel, packet, connection, compatible, travel)
        if ended:
            self.working.pop(key, None)
            self.working_intervals.pop(key, None)
        else:
            self.working[key] = moved
            self.working_intervals[key] = _body_intervals(self.kernel.topology, moved)
        internal = self.kernel.topology.lanes[packet['lane_id']]['internal']
        self.processed[key] = dict(packet=packet, connection=connection, speed=actual,
                                   committed=bool(internal), accepted=bool(internal or len(trace) > 1))
        result = dict(travel_m=actual, physical_proposal_mps=proposal, raw_upper_mps=min(proposal, bound),
                      current_path_bound_m=static_bound, same_step_path_bound_m=bound,
                      receiving_release_m=bound-static_bound, spatial_limit_m=spatial['limit_m'], spatial_summary=spatial,
                      receiving_binding_summary=dict(packet_ids=nearest, cycle_encountered=bool(self.cycle_edges)),
                      projected_packet=moved, trace=trace, ended=ended)
        self.results[key] = result
        self.order.append(key)
        self.visiting.remove(key)
        return result


def _priority_order(resolver):
    kernel, old = resolver.kernel, resolver.old
    def order(key):
        packet = old[key]
        connection, compatible = resolver.connection(key)
        signal = kernel._signal(connection, compatible)
        rank = 0 if kernel.topology.lanes[packet['lane_id']]['internal'] else 1 if signal == 'G' else 2
        distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
        return rank, distance/max(resolver[key], .1), resolver.context['packet_order'][key]
    def compare(first, second):
        a, b = old[first], old[second]
        ai, bi = kernel.topology.lanes[a['lane_id']]['internal'], kernel.topology.lanes[b['lane_id']]['internal']
        if ai != bi:
            return -1 if ai else 1
        if a['lane_id'] == b['lane_id'] and a['position_m'] != b['position_m']:
            return -1 if a['position_m'] > b['position_m'] else 1
        ac, al = resolver.connection(first)
        bc, bl = resolver.connection(second)
        if ac is not None and bc is not None and al and bl:
            if kernel._gives_way(ac, kernel._signal(ac), bc, kernel._signal(bc)):
                return 1
            if kernel._gives_way(bc, kernel._signal(bc), ac, kernel._signal(ac)):
                return -1
        return -1 if order(first) < order(second) else 1 if order(first) > order(second) else 0
    return sorted(old, key=cmp_to_key(compare))


def resolve_conditional(context, kernel, packet_id, connection, compatible):
    if kernel._connections[packet_id] != (connection, compatible):
        raise ValueError('The focal current intent/connection must be installed before conditional resolution')
    resolver = MotionResolver(context, kernel, context['zero_decisions'], context['permit_all'], focal_id=packet_id)
    result = dict(resolver.resolve(packet_id))
    result.pop('projected_packet')
    result.pop('trace')
    result.pop('ended')
    result['resolved_packet_count'] = len(resolver.results)
    return result


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
