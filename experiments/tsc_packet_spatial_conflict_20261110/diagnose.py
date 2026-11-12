"""Zero-fit spatial-conflict comparison on frozen native training records."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import multiprocessing
from pathlib import Path
import traceback

from experiments.tsc_target_packet_training_20261109 import conditional_features as seed_context
from experiments.tsc_target_packet_training_20261109.training_data import (
    iter_transitions, _candidates, RESERVOIR_SEED)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_receiving_path_20261110 import conditional_features as baseline
from experiments.tsc_packet_receiving_path_20261110.diagnose import _empty_permission, update_permission
from . import conditional_features as candidate
from .conflict import configure_geometry
from .geometry import build_conflict_geometry


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2)+'\n')


def load_widths(static):
    """Resolve XML/default lane widths with the frozen, read-only SUMO parser."""
    import sumolib.net
    copied = deepcopy(static)
    net = sumolib.net.readNet(static['netfile'], withInternal=True)
    resolved = Counter()
    for key, lane in copied['lanes'].items():
        if lane.get('width_m') is None:
            lane['width_m'] = float(net.getLane(key).getWidth())
            resolved['from_frozen_sumolib_parser'] += 1
        else:
            resolved['already_exported'] += 1
    return copied, dict(counts=dict(resolved), parser_source=sumolib.net.__file__,
        netfile=static['netfile'], widths_m=sorted({lane['width_m'] for lane in copied['lanes'].values()}))


def load_focus(conflict_base):
    result, counts = defaultdict(dict), Counter()
    for path in sorted((Path(conflict_base)/'output').glob('seed*_conflict_details_server_only.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                ref = row['reference']
                overlaps = [b for b in row['blockers'] if b['is_overlapping']]
                clean = (all(b['internal'] and b['observed_truth']['connection_verdict'] == 'same' for b in overlaps)
                    and row['poststep_observed_ego']['front_internal']
                    and any(b['observed_truth']['query_plus1_snapshot'] is not None
                            and b['observed_truth']['query_plus1_snapshot']['occupied_internal_lanes']
                            for b in overlaps))
                key = (int(ref['time_sec']), ref['packet_id'])
                result[int(ref['seed'])][key] = dict(connection_key=row['ego_connection']['key'],
                    intent=row['ego_connection']['to_edge'], clean=bool(clean))
                counts['all'] += 1
                counts['clean'] += int(clean)
    if counts != Counter(all=374, clean=253):
        raise ValueError('The prior 374/253 conflict cohorts must reproduce without replacement')
    return dict(result)


def paired_admission(kernel, packet_id, intent):
    packet = kernel.state['packets'][packet_id]
    old_intent = deepcopy(kernel.state['intents'].get(packet_id))
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
            kernel.state['intents'][packet_id] = old_intent
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = old_connection


def _seed(task):
    seed, directory, static, laws, references, focus_refs, output = task
    print(json.dumps(dict(stage='spatial_pair_begin', seed=seed)), flush=True)
    table = build_conflict_geometry(RoadTopology(static))
    executor_seeds = json.loads((directory/'executor_seed_server_only.json').read_text())
    by_time, probe_by_time = defaultdict(list), defaultdict(list)
    for ref in references:
        by_time[ref['time_sec']].append(ref)
    for (t, pid), focus in focus_refs.items():
        if focus['clean']:
            probe_by_time[t+1].append((pid, focus))
    stats, focus_counts, probes = _empty_permission(), Counter(), Counter()
    geometry_uses, samples = Counter(), 0
    probe_rows = []
    with gzip.open(output/f'seed{seed}_pairs_server_only.jsonl.gz', 'xt', compresslevel=1) as trace:
        for frame, current, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t % 300 == 0:
                print(json.dumps(dict(stage='spatial_pair_progress', seed=seed, time_sec=t)), flush=True)
            if t not in by_time and t not in probe_by_time:
                continue
            kernel = seed_context.prepare_feature_kernel(static, frame, executor_seeds, laws,
                action['requests'], RESERVOIR_SEED+seed*100000+t)
            configure_geometry(kernel, table)
            if kernel.signals != action['applied_states']:
                raise ValueError('Both arms must use the recorded current signals')
            for ref in by_time[t]:
                pid = ref['packet_id']
                prior = _candidates(kernel, pid)
                pairs = [paired_admission(kernel, pid, intent) for intent in prior]
                update_permission(stats, ref['y'], list(prior.values()), pairs)
                record = dict(reference=ref, prior=prior, paired=pairs)
                selected = focus_refs.get((t, pid))
                if selected is not None:
                    pair = next(row for row in pairs if row['intent'] == selected['intent'])
                    if pair['connection_key'] != selected['connection_key'] or pair['baseline']['yield_risk']:
                        raise ValueError('The frozen conflict focus must keep its original connection and denial')
                    focus_counts['all_rows'] += 1
                    focus_counts['all_recovered'] += int(pair['candidate']['yield_risk'])
                    if selected['clean']:
                        focus_counts['clean_rows'] += 1
                        focus_counts['clean_recovered'] += int(pair['candidate']['yield_risk'])
                    record['posthoc_focus'] = selected
                for pair in pairs:
                    detail = pair['candidate']
                    for key in ('resolved_pairs', 'no_zone_pairs', 'unresolved_pairs'):
                        geometry_uses[key+'_visits'] += detail.get(key, 0)
                trace.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                samples += 1
            for pid, selected in probe_by_time[t]:
                packet = frame['packets'].get(pid)
                if packet is None or not kernel.topology.lanes[packet['lane_id']]['internal']:
                    raise ValueError('The declared internal probe must be currently observed internal')
                connection, compatible = kernel._connections[pid]
                pair = paired_admission(kernel, pid, connection['to_edge'])
                observed = current['packets'].get(pid)
                probes['evaluated'] += 1
                probes['candidate_limit_reduced'] += int(pair['candidate']['upper'] < pair['baseline']['upper']-1e-8)
                row = dict(seed=seed, time_sec=t, packet_id=pid, current_lane=packet['lane_id'],
                           current_position_m=packet['position_m'], paired=pair)
                if observed is not None:
                    speed = observed['speed_mps']
                    row['actual_next_speed_mps'] = speed
                    probes['next_speed_labels'] += 1
                    probes['baseline_ceiling_below_observed_speed'] += int(pair['baseline']['upper']+1e-8 < speed)
                    probes['candidate_ceiling_below_observed_speed'] += int(pair['candidate']['upper']+1e-8 < speed)
                    probes['candidate_zero_while_actual_moving'] += int(pair['candidate']['upper'] < .1 <= speed)
                probe_rows.append(row)
    if samples != len(references):
        raise ValueError('All original yield references must be compared exactly once')
    planned = sum(len(rows) for rows in probe_by_time.values())
    probes['planned'] = planned
    probes['not_evaluable_no_following_transition'] = planned-probes['evaluated']
    save(output/f'seed{seed}_internal_probes_server_only.json', probe_rows)
    return dict(seed=seed, samples=samples, permission_by_label=stats, focus=dict(focus_counts),
        internal_probes=dict(probes), geometry_uses=dict(geometry_uses),
        geometry_metadata={key: value for key, value in table.items() if key != 'pairs'})


def summarize(results):
    permission, focus, probes, uses = _empty_permission(), Counter(), Counter(), Counter()
    for result in results:
        for label, values in result['permission_by_label'].items():
            for key, value in values.items():
                permission[label][key] += value
        focus.update(result['focus'])
        probes.update(result['internal_probes'])
        uses.update(result['geometry_uses'])
    for values in permission.values():
        n = values['rows']
        values['baseline_permission_mass_mean'] = values['baseline_permission_mass_sum']/n
        values['candidate_permission_mass_mean'] = values['candidate_permission_mass_sum']/n
    expected = (permission['1']['rows'], permission['0']['rows'],
                permission['1']['baseline_any_count'], permission['0']['baseline_any_count'],
                focus['all_rows'], focus['clean_rows'], probes['planned'])
    reproduced = expected == (9371, 30629, 8447, 7745, 374, 253, 253)
    return dict(status='COMPLETE_SPATIAL_CONFLICT_DIAGNOSIS' if reproduced else 'HOLD_BASELINE_REPRODUCTION',
        baseline_reproduction_values=list(expected), baseline_reproduced=reproduced,
        new_native_starts=0, new_fits=0, new_nuisance_estimations=0, free_rollouts=0,
        permission_by_label=permission, focus=dict(focus), internal_probes=dict(probes),
        geometry_uses=dict(uses), by_seed=results,
        interpretation='Structural permission and zero-residual motion ceilings on existing training records; not calibrated entry accuracy, conflict-point crossing validation or control benefit.')


def run(cache_base, conflict_base, output, workers=4):
    output.mkdir(parents=True, exist_ok=False)
    try:
        focus = load_focus(conflict_base)
        reservoirs = json.loads((cache_base/'training_reservoir_server_only.json').read_text())
        original_static = json.loads((cache_base/'static_graph_server_only.json').read_text())
        static, width_source = load_widths(original_static)
        laws = json.loads((cache_base/'target_laws_server_only.json').read_text())
        seeds = [76116901, 76116902, 76116903, 76116904]
        tasks = [(seed, cache_base/'training'/f'seed{seed}', static, laws,
                  reservoirs[str(seed)]['yield_entry'], focus[seed], output) for seed in seeds]
        if workers == 1:
            results = [_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                results = list(pool.map(_seed, tasks))
        result = summarize(results)
        result.update(width_source=width_source, baseline_source=baseline.__file__,
            candidate_source=candidate.__file__, seed_context_source=seed_context.__file__)
    except Exception as error:
        result = dict(status='HOLD_SPATIAL_DIAGNOSTIC_ERROR', error=f'{type(error).__name__}: {error}',
                      new_native_starts=0, new_fits=0, free_rollouts=0)
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    save(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--conflict-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.conflict_base, args.output, args.workers)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'COMPLETE_SPATIAL_CONFLICT_DIAGNOSIS' else 2)
