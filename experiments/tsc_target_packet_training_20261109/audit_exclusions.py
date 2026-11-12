"""Zero-fit audit of exact cached positive exclusions; large evidence stays server-side."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from . import conditional_features as conditional_implementation
from .training_data import iter_transitions, _front_path, _candidates, RESERVOIR_SEED
from .conditional_features import (
    prepare_feature_kernel, candidate_features, _context_view, _zero_speed)


def reference_key(row):
    return int(row['seed']), float(row['time_sec']), row['packet_id']


def select_references(reservoirs, retained):
    """Exact OLD-reservoir minus NEW-retained keys; labels never rescue rows."""
    selected, counts = {}, dict(excluded_yield_positive=0, retained_yield_positive=0, excluded_lane_positive=0)
    for seed, rows in reservoirs.items():
        selected[int(seed)] = dict(yield_entry=[], lane_change=[])
    for head in ('yield_entry', 'lane_change'):
        original = {reference_key(row): row for seed in reservoirs.values() for row in seed[head]}
        kept = {reference_key(row): row for row in retained[head]}
        if not set(kept) <= set(original) or any(kept[key]['y'] != original[key]['y'] for key in kept):
            raise ValueError('Retained references must be unchanged members of the full frozen reservoir')
        for key, row in original.items():
            positive = row['y'] != 1 if head == 'lane_change' else row['y'] > 0
            if not positive:
                continue
            partition = 'retained' if key in kept else 'excluded'
            if head == 'lane_change' and partition == 'retained':
                continue
            selected[key[0]][head].append(dict(row, partition=partition))
            if head == 'yield_entry':
                counts[partition+'_yield_positive'] += 1
            else:
                counts['excluded_lane_positive'] += 1
    for seed in selected.values():
        for rows in seed.values():
            rows.sort(key=lambda row: (row['time_sec'], row['packet_id']))
    return dict(by_seed=selected, counts=counts)


def _signal(connection, applied):
    if connection['tls_id'] is None:
        return 'uncontrolled'
    state = applied.get(connection['tls_id'])
    index = connection['link_index']
    return state[index] if state is not None and index < len(state) else 'unknown_signal'


def match_observed_entry(topology, path, applied):
    """Match the actual entering ordinary lane, including an earlier lateral segment."""
    if not path:
        return dict(event_kind='unknown_path', actual_signal='unknown', matches=[])
    for index, (origin, following) in enumerate(zip(path, path[1:])):
        first, second = topology.lanes[origin], topology.lanes[following]
        if first['internal'] or first['edge_id'] == second['edge_id']:
            continue
        observed = path[index:]
        matches = []
        for connection in topology.connections_from(origin):
            physical = connection['path']
            common = min(len(observed), len(physical))
            if tuple(observed[:common]) == tuple(physical[:common]):
                matches.append(connection)
        if len(matches) == 1:
            return dict(event_kind='stopline_entry', actual_signal=_signal(matches[0], applied), matches=matches)
        return dict(event_kind='stopline_entry', actual_signal='ambiguous' if matches else 'unknown',
                    possible_signals=sorted({_signal(c, applied) for c in matches}), matches=matches)
    lateral = any(topology.lanes[a]['edge_id'] == topology.lanes[b]['edge_id'] and
                  not topology.lanes[a]['internal'] and a != b for a, b in zip(path, path[1:]))
    return dict(event_kind='ordinary_lane_change_only' if lateral else 'no_stopline_entry',
                actual_signal='no_stopline_entry', matches=[])


def _yield_detail(kernel, reference, actual_connection):
    """Explain frozen-configuration g-risk, not native physical impossibility."""
    key = reference['packet_id']
    packet = kernel.state['packets'][key]
    priors = _candidates(kernel, key)
    details = []
    for intent, prior in priors.items():
        candidate = candidate_features(kernel, key, intent)
        old_intent = deepcopy(kernel.state['intents'].get(key))
        old_connection = kernel._connections[key]
        had_intent = key in kernel.state['intents']
        try:
            edge = kernel.topology.lanes[packet['lane_id']]['edge_id']
            kernel.state['intents'][key] = dict(road_id=edge, choice=intent)
            connection, compatible = kernel._connection(packet)
            kernel._connections[key] = connection, compatible
            view = _context_view(kernel, packet, connection, compatible)
            raw = kernel._speed(packet, {'motion_residual_mps': 0.}, view, connection, compatible)
            distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
            signal = kernel._signal(connection, compatible)
            receiving = kernel._receiving_space(connection, view)
            needed = packet['parameters']['length_m']+packet['parameters']['min_gap_m']
            conflict = None
            blockers = []
            if connection is not None and compatible and raw > 0 and raw >= distance:
                interval = kernel._reservation(packet, connection, raw)
                for other_key, other in view.items():
                    if other_key == key:
                        continue
                    foe, legal = kernel._connections[other_key]
                    if foe is None or not legal or not kernel.topology.foes(connection, foe):
                        continue
                    internal = kernel.topology.lanes[other['lane_id']]['internal']
                    foe_signal = kernel._signal(foe)
                    if internal or (foe_signal in ('G', 'g') and kernel._gives_way(connection, signal, foe, foe_signal)):
                        speed = _zero_speed(kernel, other)
                        if internal or speed > 0:
                            blockers.append((foe, kernel._reservation(other, foe, speed)))
                conflict = kernel._conflict(connection, interval, blockers)
            reasons = []
            if connection is None:
                reasons.append('END_no_junction_entry')
            elif not compatible:
                reasons.append('current_lane_incompatible_with_intent')
            elif signal != 'g':
                reasons.append('not_g_signal_'+signal)
            elif raw <= 0 or raw < distance:
                reasons.append('raw_speed_ceiling_not_reaching_stopline')
            else:
                if receiving < needed:
                    reasons.append('current_receiving_space_blocked')
                if conflict:
                    reasons.append('frozen_other_intents_conflict_reservation')
                if not reasons:
                    reasons.append('risk_true' if candidate['yield_risk'] else 'postadmission_not_reaching')
            details.append(dict(intent=intent, prior=prior,
                matches_actual_intent=intent == actual_connection['to_edge'],
                chosen_connection=connection, same_physical_connection=(connection is not None and connection['key'] == actual_connection['key']),
                compatible=compatible, signal=signal, raw_speed_ceiling_mps=raw,
                stopline_distance_m=distance, postadmission_upper_mps=candidate['upper'],
                receiving_free_space_m=receiving, receiving_required_m=needed,
                conflict_under_frozen_other_intents=conflict, blocker_count=len(blockers),
                yield_risk=bool(candidate['yield_risk']), reasons=reasons))
        finally:
            if had_intent:
                kernel.state['intents'][key] = old_intent
            else:
                kernel.state['intents'].pop(key, None)
            kernel._connections[key] = old_connection
    return dict(current_packet={name: packet[name] for name in
            ('lane_id', 'position_m', 'speed_mps', 'acceleration_mps2', 'parameters')}, candidates=details,
        interpretation='Risk0 under the frozen XML-first connection and one current prior draw for other intents; not proof of native physical impossibility.')


def _audit_seed(task):
    seed, directory, static, laws, selected = task
    from .lane_exclusion_detail import diagnose_lane
    print(json.dumps(dict(stage='exclusion_audit_seed_begin', seed=seed)), flush=True)
    topology = RoadTopology(static)
    executor_seeds = json.loads((Path(directory)/'executor_seed_server_only.json').read_text())
    by_time = defaultdict(lambda: dict(yield_entry=[], lane_change=[]))
    for head, references in selected.items():
        for reference in references:
            by_time[reference['time_sec']][head].append(reference)
    signals = {partition: Counter() for partition in ('excluded', 'retained')}
    event_counts, matched_reasons, other_reasons = Counter(), Counter(), Counter()
    examples, lane_details, seen = {}, [], Counter()
    connection_differences, excluded_rebuild_supported = 0, 0
    for frame, current, action in iter_transitions(directory):
        time_s = frame['time_s']
        if time_s % 300 == 0:
            print(json.dumps(dict(stage='exclusion_audit_progress', seed=seed, time_sec=time_s)), flush=True)
        if time_s not in by_time:
            continue
        kernel = None
        for reference in by_time[time_s]['yield_entry']:
            key, partition = reference['packet_id'], reference['partition']
            path = _front_path(frame, current, key)
            match = match_observed_entry(topology, path, action['applied_states'])
            signal = match['actual_signal']
            signals[partition][signal] += 1
            seen[partition+'_yield'] += 1
            event_counts[partition+':'+match['event_kind']] += 1
            example = dict(reference=reference, observed_front_path=path, actual_entry=match)
            if partition == 'excluded' and signal == 'g':
                if kernel is None:
                    feature_seed = RESERVOIR_SEED+int(seed)*100000+int(time_s)
                    kernel = prepare_feature_kernel(static, frame, executor_seeds, laws, action['requests'], feature_seed)
                detail = _yield_detail(kernel, reference, match['matches'][0])
                example['frozen_configuration_detail'] = detail
                example['actual_next_packet'] = ({name: current['packets'][key][name]
                    for name in ('lane_id', 'position_m', 'speed_mps')} if key in current['packets'] else None)
                excluded_rebuild_supported += int(any(c['yield_risk'] for c in detail['candidates']))
                for candidate in detail['candidates']:
                    target = matched_reasons if candidate['matches_actual_intent'] else other_reasons
                    target.update(candidate['reasons'])
                    if candidate['matches_actual_intent'] and not candidate['same_physical_connection']:
                        connection_differences += 1
            examples.setdefault((partition, signal), example)
        for reference in by_time[time_s]['lane_change']:
            lane_details.append(diagnose_lane(static, executor_seeds, laws, frame, current, action, reference))
            seen['excluded_lane'] += 1
    expected = dict(excluded_yield=sum(r['partition'] == 'excluded' for r in selected['yield_entry']),
                    retained_yield=sum(r['partition'] == 'retained' for r in selected['yield_entry']),
                    excluded_lane=len(selected['lane_change']))
    if any(seen[name] != count for name, count in expected.items()):
        raise ValueError('Every selected exclusion/comparison reference must be found in its original stream')
    return dict(seed=seed, counts=dict(seen), excluded_positive_by_observed_signal=dict(signals['excluded']),
        retained_positive_by_observed_signal=dict(signals['retained']), event_counts=dict(event_counts),
        actual_g_frozen_configuration_risk0=dict(matched_intent_reason_counts=dict(matched_reasons),
            other_intent_reason_counts=dict(other_reasons), matched_intent_physical_connection_differs=connection_differences,
            excluded_rows_rebuilt_with_any_yield_risk=excluded_rebuild_supported),
        examples=list(examples.values()), lane_details=lane_details)


def run(cache_base, corrected_base, output, workers=4):
    cache_base, corrected_base, output = Path(cache_base), Path(corrected_base), Path(output)
    if not 1 <= workers <= 4:
        raise ValueError('Audit permits one to four seed workers')
    output.mkdir(parents=True, exist_ok=False)
    reservoirs = json.loads((cache_base/'training_reservoir_server_only.json').read_text())
    retained = json.loads((corrected_base/'training_references_server_only.json').read_text())
    selection = select_references(reservoirs, retained)
    if (selection['counts']['excluded_yield_positive'], selection['counts']['excluded_lane_positive']) != (2560, 9):
        raise ValueError('Exact cached positive exclusion counts must reproduce yield2560/lane9')
    static = json.loads((cache_base/'static_graph_server_only.json').read_text())
    laws = json.loads((cache_base/'target_laws_server_only.json').read_text())
    tasks = [(seed, cache_base/'training'/f'seed{seed}', static, laws, references)
             for seed, references in sorted(selection['by_seed'].items())]
    if workers == 1:
        results = [_audit_seed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(_audit_seed, tasks))
    examples = [example for result in results for example in result.pop('examples')]
    lanes = [detail for result in results for detail in result.pop('lane_details')]
    if len(lanes) != 9:
        raise ValueError('All nine exact positive lane exclusions require detailed evidence')
    for name, value in (('yield_examples_server_only.json', examples), ('lane_details_server_only.json', lanes)):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    totals = {}
    for field in ('excluded_positive_by_observed_signal', 'retained_positive_by_observed_signal'):
        counts = Counter()
        for result in results:
            counts.update(result[field])
        totals[field] = dict(counts)
    summary = dict(status='COMPLETE_EXCLUSION_AUDIT', new_native_starts=0, new_fits=0,
        changes_to_frozen_training=False, selection_counts=selection['counts'], totals=totals, by_seed=results,
        conditional_features_source=str(Path(conditional_implementation.__file__).resolve()),
        detailed_lane_cases=len(lanes), server_only_yield_examples=len(examples),
        interpretation='Exact cached exclusions and retained-positive comparison. g-risk reasons concern the frozen XML-first/local-prior configuration, not native physical impossibility.')
    (output/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--corrected-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    print(json.dumps(run(args.cache_base, args.corrected_base, args.output, args.workers)), flush=True)
