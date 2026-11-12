"""Posthoc movement/clearance truth from existing one-second native records."""
from collections import defaultdict
from copy import deepcopy

from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals
from experiments.tsc_target_packet_training_20261109.audit_exclusions import match_observed_entry
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions


def _current_internal_connection(topology, packet):
    owners = list(topology.internal_connections[packet['lane_id']])
    compatible = [connection for connection in owners
                  if all(lane in connection['path'] for lane in packet['body_trail'])]
    owners = compatible or owners
    return owners[0] if len(owners) == 1 else None


def _body_inside(topology, packet, connection):
    internal = set(connection['path'][1:-1])
    return any(lane in internal and front > rear
               for lane, rear, front in _body_intervals(topology, packet))


def _snapshot(topology, packet, connection):
    if packet is None:
        return None
    occupied_internal = sorted({lane for lane, rear, front in _body_intervals(topology, packet)
                                if topology.lanes[lane]['internal'] and front > rear})
    return dict(lane_id=packet['lane_id'], position_m=packet['position_m'],
                speed_mps=packet['speed_mps'], body_trail=list(packet['body_trail']),
                occupied_internal_lanes=occupied_internal,
                front_is_internal=topology.lanes[packet['lane_id']]['internal'],
                front_on_movement_internal=None if connection is None else packet['lane_id'] in connection['path'][1:-1],
                body_on_movement_internal=None if connection is None else _body_inside(topology, packet, connection))


def _set_connection(state, connection, source):
    state['connection'] = connection
    state['resolution_source'] = source
    state['awaiting_entry'] = False


def _update_clearance(state, topology, packet, interval):
    connection = state['connection']
    if connection is None or packet is None:
        return
    if packet['lane_id'] not in connection['path'][1:-1] and state['front_clear_interval_sec'] is None:
        state['front_clear_interval_sec'] = interval
    if (state['front_clear_interval_sec'] is not None and
            not _body_inside(topology, packet, connection) and state['body_clear_interval_sec'] is None):
        state['body_clear_interval_sec'] = interval
        state['closed'] = True


def resolve_from_transitions(transitions, topology, queries):
    """Resolve query movements and sampled-clock clearances without intervention.

    Query time is the pre-step observation. Every observed transition time is
    an interval ``(previous_time, current_time]``, never an exact subsecond time.
    An approaching query stops at its first road exit, arrival or disappearance.
    """
    query_ids = [query['query_id'] for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError('Every blocker query requires a unique query_id')
    scheduled = defaultdict(list)
    for query in queries:
        scheduled[float(query['time_sec'])].append(query)
    wanted = {query['packet_id'] for query in queries}
    states, active, last_entry = {}, defaultdict(list), {}
    observation_end = None
    for previous, current, action in transitions:
        before, after = float(previous['time_s']), float(current['time_s'])
        observation_end = after
        interval = [before, after]
        for query in scheduled.get(before, ()):
            packet = previous['packets'].get(query['packet_id'])
            if packet is None or packet['lane_id'] != query['lane_id']:
                raise ValueError('Blocker query must match its original native pre-step packet')
            internal = topology.lanes[packet['lane_id']]['internal']
            state = dict(query=query, connection=None, awaiting_entry=not internal, closed=False,
                         original_edge=topology.lanes[packet['lane_id']]['edge_id'],
                         resolution_source='awaiting_first_road_exit' if not internal else 'ambiguous_current_internal',
                         entry_interval_sec=None, entry_left_censored=internal,
                         front_clear_interval_sec=None, body_clear_interval_sec=None,
                         terminal_or_disappearance=None, actual_road_target=None, query_packet=deepcopy(packet),
                         query_plus1_packet=None, last_observed_time_sec=before)
            if internal:
                connection = _current_internal_connection(topology, packet)
                if connection is not None:
                    _set_connection(state, connection, 'current_internal_unique')
                    past = last_entry.get(query['packet_id'])
                    if past is not None and past['connection']['key'] == connection['key']:
                        state['entry_interval_sec'] = past['interval']
                        state['entry_left_censored'] = False
            states[query['query_id']] = state
            active[query['packet_id']].append(state)

        paths = {event['vehicle_id']: event.get('observed_lane_path')
                 for event in current['events']
                 if event['kind'] == 'lane_transition' and event['vehicle_id'] in wanted}
        matches = {}
        for packet_id, path in paths.items():
            match = match_observed_entry(topology, path, action['applied_states'])
            matches[packet_id] = match
            if match['event_kind'] == 'stopline_entry' and len(match['matches']) == 1:
                last_entry[packet_id] = dict(connection=match['matches'][0], interval=interval)

        for packet_id, packet_states in list(active.items()):
            packet = current['packets'].get(packet_id)
            for state in packet_states:
                if before == float(state['query']['time_sec']):
                    state['query_plus1_packet'] = deepcopy(packet)
                if state['closed']:
                    continue
                state['last_observed_time_sec'] = after
                if packet is None:
                    kind = 'terminal_exit' if packet_id in current['arrived_ids'] else 'unaccounted_disappearance'
                    state['terminal_or_disappearance'] = dict(kind=kind, interval_sec=interval)
                    if state['awaiting_entry']:
                        state['resolution_source'] = 'observed_terminal_without_entry' if kind == 'terminal_exit' else kind+'_before_entry'
                        if kind == 'terminal_exit':
                            state['actual_road_target'] = 'END'
                    state['closed'] = True
                    continue
                if state['awaiting_entry']:
                    match = matches.get(packet_id)
                    if match is not None and match['event_kind'] == 'stopline_entry':
                        eligible = [connection for connection in match['matches']
                                    if connection['from_edge'] == state['original_edge']]
                        if len(eligible) == 1:
                            _set_connection(state, eligible[0], 'observed_first_road_exit')
                            state['entry_interval_sec'] = interval
                        else:
                            state['resolution_source'] = 'ambiguous_first_road_exit' if eligible else 'unmatched_first_road_exit'
                            state['awaiting_entry'] = False
                            state['closed'] = True
                    elif (topology.lanes[packet['lane_id']]['internal'] or
                          topology.lanes[packet['lane_id']]['edge_id'] != state['original_edge']):
                        state['resolution_source'] = 'unmatched_first_road_exit'
                        state['awaiting_entry'] = False
                        state['closed'] = True
                _update_clearance(state, topology, packet, interval)
            active[packet_id] = [state for state in packet_states if not state['closed']]
            if not active[packet_id]:
                del active[packet_id]

    if set(states) != set(query_ids):
        raise ValueError('All blocker query clocks must occur in the frozen native stream')
    results = []
    for query in queries:
        state = states[query['query_id']]
        connection = state['connection']
        sampled = topology.by_key[query['sampled_connection_key']]
        actual_target = connection['to_edge'] if connection is not None else state['actual_road_target']
        verdict = ('different' if actual_target == 'END' else 'unresolved') if connection is None else 'same' if connection['key'] == sampled['key'] else 'different'
        road_verdict = 'unresolved' if actual_target is None else 'same' if actual_target == sampled['to_edge'] else 'different'
        truth = dict(connection_verdict=verdict, road_target_verdict=road_verdict,
                     actual_road_target=actual_target,
                     actual_connection_key=None if connection is None else connection['key'],
                     actual_connection=connection, resolution_source=state['resolution_source'],
                     entry_interval_sec=state['entry_interval_sec'], entry_left_censored=state['entry_left_censored'],
                     front_clear_interval_sec=state['front_clear_interval_sec'],
                     body_clear_interval_sec=state['body_clear_interval_sec'],
                     entry_right_censored=not state['closed'] and state['entry_interval_sec'] is None and connection is None,
                     front_clear_right_censored=not state['closed'] and state['front_clear_interval_sec'] is None,
                     body_clear_right_censored=not state['closed'] and state['body_clear_interval_sec'] is None,
                     terminal_or_disappearance=state['terminal_or_disappearance'],
                     observation_end_sec=observation_end, last_observed_time_sec=state['last_observed_time_sec'],
                     query_snapshot=_snapshot(topology, state['query_packet'], connection),
                     query_plus1_snapshot=_snapshot(topology, state['query_plus1_packet'], connection),
                     timing_convention='Intervals are (previous_frame_time,current_frame_time]; snapshots are observed frame clocks.',
                     use='Posthoc diagnostic truth only; not an admission or prediction input.')
        results.append(dict(query_id=query['query_id'], observed_truth=truth))
    return results


def resolve_blocker_truth(directory, topology, queries):
    return resolve_from_transitions(iter_transitions(directory), topology, queries)


def attach_observed_truth(records, directory, topology):
    """Return copied records with truth attached only to overlapping blockers."""
    copied, queries = deepcopy(records), []
    for index, record in enumerate(copied):
        reference = record['reference']
        packet = record['ego_packet']
        record['ego_truth_query_id'] = f'ego:{index}'
        queries.append(dict(query_id=record['ego_truth_query_id'], time_sec=reference['time_sec'],
                            packet_id=reference['packet_id'], lane_id=packet['lane_id'],
                            body_trail=packet['body_trail'], sampled_connection_key=record['ego_connection']['key']))
        for blocker in record['blockers']:
            if not blocker['is_overlapping']:
                continue
            packet = blocker['current_packet']
            queries.append(dict(query_id=blocker['query_id'], time_sec=record['reference']['time_sec'],
                                packet_id=blocker['packet_id'], lane_id=packet['lane_id'],
                                body_trail=packet['body_trail'], sampled_connection_key=blocker['connection']['key']))
    truth_by_id = {result['query_id']: result['observed_truth']
                   for result in resolve_blocker_truth(directory, topology, queries)}
    for record in copied:
        record['ego_observed_truth'] = truth_by_id[record.pop('ego_truth_query_id')]
        for blocker in record['blockers']:
            if blocker['is_overlapping']:
                blocker['observed_truth'] = truth_by_id[blocker['query_id']]
    return copied
