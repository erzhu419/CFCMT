"""Paired current-state readout of a shared same-second receiving operator."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import multiprocessing
from pathlib import Path
import traceback

from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110 import conditional_features as baseline
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import (
    load_focus, load_widths, save, seed_context, iter_transitions, _candidates,
    RESERVOIR_SEED, RoadTopology, _empty_permission, update_permission,
)
from . import conditional_features as candidate


SEEDS = [76116901, 76116902, 76116903, 76116904]
TOL = 1e-8


def load_probes(entity):
    saved, bad = defaultdict(dict), 0
    for seed in SEEDS:
        path = entity/'output'/f'seed{seed}_internal_probes_server_only.json'
        for row in json.loads(path.read_text()):
            ref = int(row['time_sec']), row['packet_id']
            if ref in saved[seed]:
                raise ValueError('Frozen internal references must be unique')
            is_bad = row['paired']['candidate']['upper']+TOL < row['actual_next_speed_mps']
            saved[seed][ref] = dict(row, previously_bad=bool(is_bad))
            bad += is_bad
    if sum(map(len, saved.values())) != 253 or bad != 3:
        raise ValueError('All253 fixed entity probes and its exact3 residuals are required')
    return dict(saved)


def paired_admission(kernel, packet_id, intent):
    packet = kernel.state['packets'][packet_id]
    previous = deepcopy(kernel.state['intents'].get(packet_id))
    had_intent = packet_id in kernel.state['intents']
    old_connection = kernel._connections[packet_id]
    try:
        geometry = kernel.topology.lanes[packet['lane_id']]
        if not geometry['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=geometry['edge_id'], choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        view = seed_context._context_view(kernel, packet, connection, compatible)
        before = baseline._admission_detail(kernel, packet, connection, compatible, view)
        after = candidate._admission_detail(kernel, packet, connection, compatible, view)
        distance = geometry['length_m']-packet['position_m']
        yielding = (not geometry['internal'] and connection is not None and compatible
                    and kernel._signal(connection) == 'g')
        for detail in (before, after):
            detail['yield_risk'] = bool(yielding and detail['upper'] > 0 and
                detail['upper'] >= distance and detail['admissible'])
        return dict(intent=intent, connection_key=None if connection is None else connection['key'],
                    baseline=before, candidate=after)
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = previous
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = old_connection


def verify_old_row(old, ref, prior, pairs):
    if old['reference'] != ref or old['prior'] != prior:
        raise ValueError('The original reference order and current intent prior must reproduce')
    previous = {row['intent']: row for row in old['paired']}
    for row in pairs:
        saved = previous.pop(row['intent'])
        if row['connection_key'] != saved['connection_key']:
            raise ValueError('Every frozen physical connection must reproduce')
        for key in ('upper', 'path_bound_m', 'raw_upper_mps', 'spatial_limit_m'):
            if abs(row['baseline'][key]-saved['candidate'][key]) > TOL:
                raise ValueError(f'Frozen entity baseline {key} changed')
        for key in ('admissible', 'yield_risk', 'active_conflict'):
            if row['baseline'][key] != saved['candidate'][key]:
                raise ValueError(f'Frozen entity baseline {key} changed')
    if previous:
        raise ValueError('Every prior intent must be retained')


def inspect_seed(task):
    seed, directory, static, laws, refs, focus, probes, width, entity, output = task
    print(json.dumps(dict(stage='same_step_begin', seed=seed)), flush=True)
    table = build_entity_geometry(RoadTopology(static), width)
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    by_time, probe_times = defaultdict(list), defaultdict(list)
    for ref in refs:
        by_time[ref['time_sec']].append(ref)
    if set(probes) != {(t+1, pid) for (t, pid), row in focus.items() if row['clean']}:
        raise ValueError('The fixed253 cohort cannot change')
    for t, pid in probes:
        probe_times[t].append(pid)
    stats, focus_counts, counts, uses = _empty_permission(), Counter(), Counter(), Counter()
    probe_rows, old_bad, new_bad, samples = [], [], [], 0
    with gzip.open(entity/'output'/f'seed{seed}_pairs_server_only.jsonl.gz', 'rt') as previous, \
            gzip.open(output/f'seed{seed}_pairs_server_only.jsonl.gz', 'xt', compresslevel=1) as stream:
        for frame, current, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t % 300 == 0:
                print(json.dumps(dict(stage='same_step_progress', seed=seed, time_sec=t)), flush=True)
            if t not in by_time and t not in probe_times:
                continue
            kernel = seed_context.prepare_feature_kernel(static, frame, executors, laws,
                action['requests'], RESERVOIR_SEED+seed*100000+t)
            configure_geometry(kernel, table)
            if kernel.signals != action['applied_states']:
                raise ValueError('Both arms must retain observed current actuator states')
            candidate.prepare_receiving_context(kernel)
            for ref in by_time[t]:
                pid = ref['packet_id']
                prior = _candidates(kernel, pid)
                pairs = [paired_admission(kernel, pid, intent) for intent in prior]
                verify_old_row(json.loads(next(previous)), ref, prior, pairs)
                update_permission(stats, ref['y'], list(prior.values()), pairs)
                record = dict(reference=ref, prior=prior, paired=pairs)
                selected = focus.get((t, pid))
                if selected is not None:
                    pair = next(row for row in pairs if row['intent'] == selected['intent'])
                    if pair['connection_key'] != selected['connection_key'] or not pair['baseline']['yield_risk']:
                        raise ValueError('The original focus permission must reproduce')
                    focus_counts['all_rows'] += 1
                    focus_counts['baseline_all_allowed'] += 1
                    focus_counts['candidate_all_allowed'] += int(pair['candidate']['yield_risk'])
                    if selected['clean']:
                        focus_counts['clean_rows'] += 1
                        focus_counts['baseline_clean_allowed'] += 1
                        focus_counts['candidate_clean_allowed'] += int(pair['candidate']['yield_risk'])
                    record['posthoc_focus'] = selected
                for pair in pairs:
                    for name in ('resolved_pairs', 'no_zone_pairs', 'unresolved_pairs'):
                        uses[name+'_visits'] += pair['candidate'].get(name, 0)
                    detail = pair['candidate']
                    uses['locally_resolved_packets'] += detail['resolved_packet_count']
                    uses['dependency_cycle_queries'] += int(detail['receiving_binding_summary']['cycle_encountered'])
                    uses['receiving_bound_expansion_queries'] += int(detail['receiving_release_m'] > TOL)
                    uses['receiving_bound_contraction_queries'] += int(detail['receiving_release_m'] < -TOL)
                stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                samples += 1
            for pid in probe_times[t]:
                saved = probes[(t, pid)]
                packet, observed = frame['packets'].get(pid), current['packets'].get(pid)
                if (packet is None or observed is None or packet['lane_id'] != saved['current_lane'] or
                        abs(packet['position_m']-saved['current_position_m']) > TOL):
                    raise ValueError('Every native internal state must reproduce')
                connection, unused = kernel._connections[pid]
                pair = paired_admission(kernel, pid, connection['to_edge'])
                before, after, speed = pair['baseline']['upper'], pair['candidate']['upper'], observed['speed_mps']
                if (abs(before-saved['paired']['candidate']['upper']) > TOL or
                        abs(speed-saved['actual_next_speed_mps']) > TOL):
                    raise ValueError('The exact entity baseline ceiling and native speed must reproduce')
                was_bad, is_bad = before+TOL < speed, after+TOL < speed
                counts.update(evaluated=1, baseline_ceiling_below_observed_speed=int(was_bad),
                    candidate_ceiling_below_observed_speed=int(is_bad),
                    candidate_limit_raised_vs_entity=int(after > before+TOL),
                    candidate_limit_reduced_vs_entity=int(after < before-TOL),
                    old_bad_fixed=int(was_bad and not is_bad), old_bad_remaining=int(was_bad and is_bad),
                    new_bad_previously_ok=int(is_bad and not was_bad),
                    candidate_zero_while_actual_moving=int(after < .1 <= speed))
                brief = dict(seed=seed, time_sec=t, packet_id=pid, baseline_upper_mps=before,
                    candidate_upper_mps=after, actual_next_speed_mps=speed,
                    previously_bad=bool(was_bad), candidate_bad=bool(is_bad))
                probe_rows.append(dict(brief, current_lane=packet['lane_id'],
                    current_position_m=packet['position_m'], paired=pair))
                if was_bad:
                    old_bad.append(brief)
                elif is_bad:
                    new_bad.append(brief)
        if next(previous, None) is not None:
            raise ValueError('Every old reference must be paired exactly once')
    if samples != len(refs) or len(probe_rows) != len(probes):
        raise ValueError('No original yield sample or internal probe may be dropped')
    counts['planned'] = len(probes)
    save(output/f'seed{seed}_internal_probes_server_only.json', probe_rows)
    return dict(seed=seed, samples=samples, baseline_rows_reproduced=samples,
        permission_by_label=stats, focus=dict(focus_counts), internal_probes=dict(counts),
        old_bad_case_outcomes=old_bad, new_bad_case_refs=new_bad, geometry_uses=dict(uses))


def summarize(results):
    permission, focus, probes, uses = _empty_permission(), Counter(), Counter(), Counter()
    old_bad, new_bad = [], []
    for row in results:
        for label, values in row['permission_by_label'].items():
            for name, value in values.items():
                permission[label][name] += value
        focus.update(row['focus']); probes.update(row['internal_probes']); uses.update(row['geometry_uses'])
        old_bad.extend(row.pop('old_bad_case_outcomes')); new_bad.extend(row.pop('new_bad_case_refs'))
    for values in permission.values():
        values['baseline_permission_mass_mean'] = values['baseline_permission_mass_sum']/values['rows']
        values['candidate_permission_mass_mean'] = values['candidate_permission_mass_sum']/values['rows']
    reproduced = [permission['1']['rows'], permission['0']['rows'], permission['1']['baseline_any_count'],
        permission['0']['baseline_any_count'], focus['all_rows'], focus['baseline_all_allowed'],
        focus['clean_rows'], focus['baseline_clean_allowed'], probes['planned'], probes['evaluated'],
        probes['baseline_ceiling_below_observed_speed'], len(old_bad)]
    passed = reproduced == [9371, 30629, 9368, 29458, 374, 374, 253, 253, 253, 253, 3, 3]
    return dict(status='COMPLETE_SAME_STEP_RECEIVING_DIAGNOSIS' if passed else 'HOLD_BASELINE_REPRODUCTION',
        baseline_reproduced=passed, baseline_reproduction_values=reproduced,
        baseline_rows_reproduced=sum(row['baseline_rows_reproduced'] for row in results),
        permission_by_label=permission, focus=dict(focus), internal_probes=dict(probes),
        old_bad_case_outcomes=old_bad, new_bad_case_refs=new_bad, geometry_uses=dict(uses), by_seed=results,
        new_native_starts=0, new_fits=0, new_nuisance_estimations=0, free_rollouts=0,
        interpretation='Conditional readout fixes other intent draws and zero residual motion. Shared receiving and reservation operators do not make it the expectation of the full stochastic joint transition; permission is not prediction accuracy or control benefit.')


def run(cache, entity, conflict, preflight, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    try:
        width, proof = verify_entity_preflight(preflight)
        focus, saved = load_focus(conflict), load_probes(entity)
        static, width_source = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        reservoirs = json.loads((cache/'training_reservoir_server_only.json').read_text())
        tasks = [(seed, cache/'training'/f'seed{seed}', static, laws, reservoirs[str(seed)]['yield_entry'],
            focus[seed], saved[seed], width, entity, output) for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(inspect_seed, tasks))
        result = summarize(results)
        result.update(entity_width_m=width, lane_width_source=width_source,
            centerline_observed=proof['centerline_observed'], candidate_source=candidate.__file__,
            baseline_source=baseline.__file__, seed_context_source=seed_context.__file__)
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_SAME_STEP_READOUT_ERROR', error=f'{type(error).__name__}: {error}',
            new_native_starts=0, new_fits=0, free_rollouts=0)
    save(output/'result.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--entity-base', type=Path, required=True)
    parser.add_argument('--conflict-base', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.entity_base, args.conflict_base, args.preflight, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_SAME_STEP_RECEIVING_DIAGNOSIS' else 2)
