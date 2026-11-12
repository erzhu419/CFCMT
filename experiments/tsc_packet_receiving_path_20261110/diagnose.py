"""Frozen zero-fit paired receiving-path permission diagnostic."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import multiprocessing
from pathlib import Path

from experiments.tsc_target_packet_training_20261109 import conditional_features as baseline
from experiments.tsc_target_packet_training_20261109 import audit_exclusions as audit
from experiments.tsc_target_packet_training_20261109.training_data import (
    iter_transitions, _candidates, _front_path, RESERVOIR_SEED)
from . import conditional_features as candidate


def paired_admissions(kernel, packet_id, intent):
    """Compare only admission operators on identical current packets/intents."""
    packet = kernel.state['packets'][packet_id]
    previous = deepcopy(kernel.state['intents'].get(packet_id))
    had_intent = packet_id in kernel.state['intents']
    previous_connection = kernel._connections[packet_id]
    try:
        geometry = kernel.topology.lanes[packet['lane_id']]
        if not geometry['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=geometry['edge_id'], choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        view = baseline._context_view(kernel, packet, connection, compatible)
        old_upper, old_admissible = baseline._admission_upper(kernel, packet, connection, compatible, view)
        new = candidate._admission_detail(kernel, packet, connection, compatible, view)
        distance = geometry['length_m']-packet['position_m']
        yielding = (not geometry['internal'] and connection is not None and compatible
                    and kernel._signal(connection) == 'g')
        old_risk = bool(yielding and old_upper > 0 and old_upper >= distance and old_admissible)
        new_risk = bool(yielding and new['upper'] > 0 and new['upper'] >= distance and new['admissible'])
        return dict(intent=intent, connection=connection, compatible=compatible,
            stopline_distance_m=distance,
            baseline=dict(upper=old_upper, admissible=bool(old_admissible), yield_risk=old_risk),
            candidate=dict(new, yield_risk=new_risk))
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = previous
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = previous_connection


def update_permission(stats, label, priors, paired):
    """Record structural permission mass, never an entry prediction score."""
    row = stats[str(int(label))]
    old = [entry['baseline']['yield_risk'] for entry in paired]
    new = [entry['candidate']['yield_risk'] for entry in paired]
    old_mass = sum(p*allowed for p, allowed in zip(priors, old))
    new_mass = sum(p*allowed for p, allowed in zip(priors, new))
    old_any, new_any = any(p > 0 and allowed for p, allowed in zip(priors, old)), any(
        p > 0 and allowed for p, allowed in zip(priors, new))
    row.update(rows=row['rows']+1,
        baseline_permission_mass_sum=row['baseline_permission_mass_sum']+old_mass,
        candidate_permission_mass_sum=row['candidate_permission_mass_sum']+new_mass,
        baseline_any_count=row['baseline_any_count']+int(old_any),
        candidate_any_count=row['candidate_any_count']+int(new_any),
        any_expansion_count=row['any_expansion_count']+int(new_any and not old_any),
        any_contraction_count=row['any_contraction_count']+int(old_any and not new_any))
    return old_any, new_any


def receiving_only_focus(detail):
    """The declared posthoc same-connection, receiving-only subgroup."""
    return next((entry for entry in detail['candidates']
        if entry['matches_actual_intent'] and entry['same_physical_connection'] and
        entry['reasons'] == ['current_receiving_space_blocked']), None)


def _remaining_reason(entry):
    detail = entry['candidate']
    if detail['yield_risk']:
        return 'matched_intent_permission_recovered'
    if detail['path_bound_m'] < entry['stopline_distance_m']:
        return 'finite_path_bound_before_entry'
    if not detail['can_reach_entry_before_conflict']:
        return 'raw_speed_ceiling_before_entry'
    if detail['conflict']:
        return 'conflict_reservation'
    return 'postadmission_not_reaching_entry'


def _empty_permission():
    return {str(label): dict(rows=0, baseline_permission_mass_sum=0., candidate_permission_mass_sum=0.,
        baseline_any_count=0, candidate_any_count=0, any_expansion_count=0, any_contraction_count=0)
        for label in (0, 1)}


def _seed(task):
    seed, directory, static, laws, references, output = task
    print(json.dumps(dict(stage='receiving_pair_seed_begin', seed=seed)), flush=True)
    executor_seeds = json.loads((Path(directory)/'executor_seed_server_only.json').read_text())
    by_time = defaultdict(list)
    for reference in references:
        by_time[reference['time_sec']].append(reference)
    stats, actual_signals, focus_reasons = _empty_permission(), Counter(), Counter()
    focus = dict(rows=0, matched_intent_permission_recovered=0, matched_intent_permission_remaining=0,
                 any_prior_permission_recovered=0)
    frames, samples = 0, 0
    trace_path = Path(output)/f'seed{seed}_pairs_server_only.jsonl.gz'
    with gzip.open(trace_path, 'xt', compresslevel=1) as trace:
        for frame, current, action in iter_transitions(directory):
            time_s = frame['time_s']
            if time_s % 300 == 0:
                print(json.dumps(dict(stage='receiving_pair_progress', seed=seed, time_sec=time_s)), flush=True)
            feature_seed = RESERVOIR_SEED+int(seed)*100000+int(time_s)
            kernel = baseline.prepare_feature_kernel(static, frame, executor_seeds, laws,
                                                      action['requests'], feature_seed)
            # Both operators share this exact actuator/current-state instance.
            if kernel.signals != action['applied_states']:
                raise ValueError('Current requests do not reproduce the fixed observed signals')
            frames += 1
            for reference in by_time.get(time_s, ()):
                key = reference['packet_id']
                priors = _candidates(kernel, key)
                paired = [paired_admissions(kernel, key, intent) for intent in priors]
                old_any, new_any = update_permission(stats, reference['y'], list(priors.values()), paired)
                record = dict(reference=reference, feature_seed=feature_seed, prior=priors, paired=paired)
                # All admission outputs are fixed before posthoc path/outcome use.
                if reference['y'] == 1:
                    path = _front_path(frame, current, key)
                    match = audit.match_observed_entry(kernel.topology, path, action['applied_states'])
                    actual_signals[match['actual_signal']] += 1
                    record['posthoc_actual_entry'] = dict(observed_front_path=path, match=match)
                    if not old_any and match['actual_signal'] == 'g':
                        detail = audit._yield_detail(kernel, reference, match['matches'][0])
                        selected = receiving_only_focus(detail)
                        if selected is not None:
                            entry = next(item for item in paired if item['intent'] == selected['intent'])
                            recovered = entry['candidate']['yield_risk']
                            focus['rows'] += 1
                            focus['matched_intent_permission_recovered'] += int(recovered)
                            focus['matched_intent_permission_remaining'] += int(not recovered)
                            focus['any_prior_permission_recovered'] += int(new_any)
                            reason = _remaining_reason(entry)
                            focus_reasons[reason] += 1
                            record['posthoc_focus'] = dict(baseline_matched_intent=selected,
                                remaining_or_recovered_reason=reason, matched_pair=entry)
                trace.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                samples += 1
            if kernel.signals != action['applied_states']:
                raise ValueError('A paired admission call changed the shared current signal/request state')
    if samples != len(references):
        raise ValueError('Every frozen yield reservoir reference must be paired once')
    return dict(seed=seed, frames=frames, samples=samples, permission_by_label=stats,
        positive_actual_signal_counts=dict(actual_signals), focus=focus,
        focus_recovered_or_remaining_reasons=dict(focus_reasons),
        paired_current_requests_and_signals_equal=True)


def summarize(results):
    permission = _empty_permission()
    focus, reasons, signals = Counter(), Counter(), Counter()
    for result in results:
        for label, values in result['permission_by_label'].items():
            for key, value in values.items():
                permission[label][key] += value
        focus.update(result['focus'])
        reasons.update(result['focus_recovered_or_remaining_reasons'])
        signals.update(result['positive_actual_signal_counts'])
    for values in permission.values():
        count = values['rows']
        values['baseline_permission_mass_mean'] = values['baseline_permission_mass_sum']/count if count else None
        values['candidate_permission_mass_mean'] = values['candidate_permission_mass_sum']/count if count else None
        values['permission_mass_mean_delta'] = (values['candidate_permission_mass_sum']-values['baseline_permission_mass_sum'])/count if count else None
    positive, negative = permission['1'], permission['0']
    reproduction = dict(samples=positive['rows']+negative['rows'], positive_total=positive['rows'],
        zero_total=negative['rows'], excluded_positives=positive['rows']-positive['baseline_any_count'],
        kept_positives=positive['baseline_any_count'], baseline_any_total=positive['baseline_any_count']+negative['baseline_any_count'])
    passed = reproduction == dict(samples=40000, positive_total=9371, zero_total=30629,
        excluded_positives=2560, kept_positives=6811, baseline_any_total=13808)
    return dict(status='COMPLETE_PAIRED_PERMISSION_DIAGNOSIS' if passed else 'HOLD_BASELINE_REPRODUCTION',
        new_native_starts=0, new_fits=0, free_rollouts=0, baseline_reproduction=reproduction,
        permission_by_label=permission, focus=dict(focus), focus_recovered_or_remaining_reasons=dict(reasons),
        positive_actual_signal_counts=dict(signals), by_seed=results,
        baseline_conditional_features_source=str(Path(baseline.__file__).resolve()),
        candidate_conditional_features_source=str(Path(candidate.__file__).resolve()),
        baseline_audit_source=str(Path(audit.__file__).resolve()),
        interpretation='Prior-weighted permission is structural admissibility, not an entry probability or control benefit. Actual paths/outcomes only define posthoc subgroups. Both arms share each current request, packet state, prior draw and actuator instance.')


def run(cache_base, corrected_base, output, workers=4):
    cache_base, corrected_base, output = Path(cache_base), Path(corrected_base), Path(output)
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four seed workers')
    output.mkdir(parents=True, exist_ok=False)
    # Source law/reservoir ownership is OLD; CLOCK proves the fixed training roster.
    protocol = json.loads((corrected_base/'protocol.json').read_text())
    expected = protocol['cohorts']['training_seeds']
    reservoirs = json.loads((cache_base/'training_reservoir_server_only.json').read_text())
    if set(map(int, reservoirs)) != set(expected):
        raise ValueError('Use the unchanged four training reservoirs only')
    static = json.loads((cache_base/'static_graph_server_only.json').read_text())
    laws = json.loads((cache_base/'target_laws_server_only.json').read_text())
    tasks = [(seed, cache_base/'training'/f'seed{seed}', static, laws,
              reservoirs[str(seed)]['yield_entry'], output) for seed in sorted(expected)]
    if workers == 1:
        results = [_seed(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(_seed, tasks))
    result = summarize(results)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--corrected-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.corrected_base, args.output, args.workers)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'COMPLETE_PAIRED_PERMISSION_DIAGNOSIS' else 2)
