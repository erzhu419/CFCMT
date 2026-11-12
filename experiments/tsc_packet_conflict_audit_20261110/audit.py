"""Read-only blocker/reservation audit of the frozen same-connection cohort."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import math
import multiprocessing
from pathlib import Path

from experiments.tsc_target_packet_training_20261109 import conditional_features as baseline
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, _front_path, RESERVOIR_SEED
from experiments.tsc_packet_receiving_path_20261110 import conditional_features as receiving
from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals


def _packet(packet):
    return {name: deepcopy(packet[name]) for name in
            ('packet_id', 'lane_id', 'position_m', 'speed_mps', 'body_trail', 'parameters')}


def interval_overlap(first, second):
    """The original strict reservation predicate; touching endpoints do not overlap."""
    return max(0., min(first[1], second[1])-max(first[0], second[0]))


def select_focus(paired_base):
    selected, remainder = defaultdict(list), Counter()
    for path in sorted((Path(paired_base)/'output').glob('seed*_pairs_server_only.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['reference']['y'] != 1 or any(
                        p['candidate']['yield_risk'] and row['prior'][p['intent']] > 0 for p in row['paired']):
                    continue
                actual = row['posthoc_actual_entry']['match']['matches']
                if len(actual) != 1:
                    raise ValueError('The frozen remaining positive must have one actual connection')
                actual = actual[0]
                matched = next(p for p in row['paired'] if p['intent'] == actual['to_edge'])
                same = matched['connection']['key'] == actual['key']
                reason = 'conflict' if matched['candidate']['conflict'] else 'raw_reach'
                remainder[reason+(':same_connection' if same else ':different_connection')] += 1
                if same and matched['candidate']['conflict']:
                    selected[int(row['reference']['seed'])].append(dict(
                        reference=row['reference'], feature_seed=row['feature_seed'],
                        saved_pair=matched, saved_actual_entry=row['posthoc_actual_entry']))
    expected = {'conflict:same_connection': 374, 'conflict:different_connection': 547,
                'raw_reach:same_connection': 2, 'raw_reach:different_connection': 1}
    if dict(remainder) != expected:
        raise ValueError(f'Frozen remaining cohort does not reproduce: {dict(remainder)}')
    for rows in selected.values():
        rows.sort(key=lambda row: (row['reference']['time_sec'], row['reference']['packet_id']))
    return dict(selected), dict(remainder)


def trace_current(kernel, packet_id, intent):
    """Reproduce one conditioned current-state admission and every eligible reservation."""
    packet = kernel.state['packets'][packet_id]
    previous = deepcopy(kernel.state['intents'].get(packet_id))
    had_intent = packet_id in kernel.state['intents']
    previous_connection = kernel._connections[packet_id]
    try:
        edge = kernel.topology.lanes[packet['lane_id']]['edge_id']
        kernel.state['intents'][packet_id] = dict(road_id=edge, choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        view = baseline._context_view(kernel, packet, connection, compatible)
        admission = receiving._admission_detail(kernel, packet, connection, compatible, view)
        if connection is None or not compatible or not admission['can_reach_entry_before_conflict']:
            raise ValueError('Selected conflict must reach its compatible connection before conflict gating')
        signal = kernel._signal(connection)
        interval = kernel._reservation(packet, connection, admission['raw_upper_mps'])
        blockers = []
        for other_key, other in view.items():
            if other_key == packet_id:
                continue
            foe, legal = kernel._connections[other_key]
            if foe is None or not legal or not kernel.topology.foes(connection, foe):
                continue
            internal = kernel.topology.lanes[other['lane_id']]['internal']
            other_signal = kernel._signal(foe)
            gives_way = kernel._gives_way(connection, signal, foe, other_signal)
            if not (internal or (other_signal in ('G', 'g') and gives_way)):
                continue
            speed = receiving._zero_speed(kernel, other)
            if not internal and speed <= 0:
                continue
            reserved = kernel._reservation(other, foe, speed)
            overlap = interval_overlap(interval, reserved)
            blockers.append(dict(packet_id=other_key, current_packet=_packet(other),
                connection=deepcopy(foe), assigned_intent=deepcopy(kernel.state['intents'].get(other_key)),
                internal=bool(internal), signal=other_signal,
                inclusion_reason='committed_internal' if internal else 'green_approach_with_priority',
                raw_zero_speed_mps=float(kernel._conditional_zero_speeds[other_key]),
                capped_zero_speed_mps=float(speed), interval=list(reserved), overlap_s=overlap,
                is_overlapping=bool(overlap > 0),
                ego_yields_to_foe=kernel.topology.yields_to(connection, foe),
                foe_yields_to_ego=kernel.topology.yields_to(foe, connection),
                ego_gives_way=bool(gives_way),
                foe_gives_way=bool(kernel._gives_way(foe, other_signal, connection, signal))))
        reconstructed = any(row['is_overlapping'] for row in blockers)
        if bool(admission['conflict']) != reconstructed:
            raise ValueError('Explicit reservation trace does not reproduce the current conflict predicate')
        return dict(ego_packet=_packet(packet), ego_connection=deepcopy(connection), ego_intent=intent,
            ego_signal=signal, ego_interval=list(interval), admission=admission, blockers=blockers)
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = previous
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = previous_connection


def _observed_ego(topology, before, after, packet_id):
    packet = after['packets'].get(packet_id)
    physical = [] if packet is None else [lane for lane, rear, front in _body_intervals(topology, packet)
        if front > rear and topology.lanes[lane]['internal']]
    return dict(time_sec=after['time_s'], packet=None if packet is None else _packet(packet),
        observed_front_path=_front_path(before, after, packet_id),
        front_internal=False if packet is None else topology.lanes[packet['lane_id']]['internal'],
        occupied_internal_lanes=physical)


def _reproduces(saved, current):
    for key in ('upper', 'path_bound_m', 'raw_upper_mps'):
        if not math.isclose(saved[key], current[key], rel_tol=0, abs_tol=1e-10):
            return False
    return all(bool(saved[key]) == bool(current[key]) for key in
               ('admissible', 'conflict', 'can_reach_entry_before_conflict'))


def _seed(task):
    from .truth import attach_observed_truth
    seed, directory, static, laws, selected, output = task
    print(json.dumps(dict(stage='conflict_audit_seed_begin', seed=seed)), flush=True)
    executor_seeds = json.loads((Path(directory)/'executor_seed_server_only.json').read_text())
    by_time = defaultdict(list)
    for row in selected:
        by_time[row['reference']['time_sec']].append(row)
    records, topology = [], None
    for before, after, action in iter_transitions(directory):
        t = before['time_s']
        if t % 300 == 0:
            print(json.dumps(dict(stage='conflict_audit_progress', seed=seed, time_sec=t)), flush=True)
        if t not in by_time:
            continue
        feature_seed = RESERVOIR_SEED+seed*100000+int(t)
        kernel = baseline.prepare_feature_kernel(static, before, executor_seeds, laws, action['requests'], feature_seed)
        topology = kernel.topology
        if kernel.signals != action['applied_states']:
            raise ValueError('Current actuator signals do not reproduce the fixed observed interval')
        for selected_row in by_time[t]:
            ref, saved = selected_row['reference'], selected_row['saved_pair']
            record = trace_current(kernel, ref['packet_id'], saved['intent'])
            if (record['ego_connection']['key'] != saved['connection']['key'] or
                    not _reproduces(saved['candidate'], record['admission']) or
                    not record['admission']['conflict']):
                raise ValueError('The saved receiving-corrected conflict does not reproduce')
            record.update(reference=ref, feature_seed=feature_seed,
                          poststep_observed_ego=_observed_ego(topology, before, after, ref['packet_id']))
            for index, blocker in enumerate(record['blockers']):
                blocker['query_id'] = f'{seed}:{len(records)}:{index}'
            records.append(record)
    if len(records) != len(selected):
        raise ValueError('Every selected frozen reference must be found exactly once')
    if topology is None:
        from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
        topology = RoadTopology(static)
    records = attach_observed_truth(records, directory, topology)
    path = Path(output)/f'seed{seed}_conflict_details_server_only.jsonl.gz'
    with gzip.open(path, 'xt', compresslevel=1) as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
    result = summarize(records)
    result.update(seed=seed, detail_records=len(records))
    return result


def summarize(records):
    reservations, case_types, case_truths, case_roads = Counter(), Counter(), Counter(), Counter()
    truth_counts, road_counts, sources, concurrency = Counter(), Counter(), Counter(), Counter()
    examples = {}
    for row in records:
        overlaps = [blocker for blocker in row['blockers'] if blocker['is_overlapping']]
        if not overlaps:
            raise ValueError('Each selected conflict needs a genuinely overlapping reservation')
        reservations['eligible'] += len(row['blockers'])
        reservations['overlapping'] += len(overlaps)
        types, verdicts, road_verdicts = set(), set(), set()
        any_body_internal_next = False
        for blocker in overlaps:
            kind = 'committed_internal' if blocker['internal'] else 'latent_approach'
            reservations[kind] += 1
            types.add(kind)
            truth = blocker['observed_truth']
            verdict = truth['connection_verdict']
            truth_counts[verdict] += 1
            road = truth.get('road_target_verdict', 'unresolved')
            road_counts[road] += 1
            sources[truth.get('resolution_source', 'unspecified')] += 1
            verdicts.add(verdict)
            road_verdicts.add(road)
            plus1 = truth.get('query_plus1_snapshot')
            if plus1 is not None and row['poststep_observed_ego']['front_internal'] and plus1.get('occupied_internal_lanes'):
                reservations['ego_front_internal_with_blocker_body_still_internal_next_frame'] += 1
                any_body_internal_next = True
        type_group = next(iter(types)) if len(types) == 1 else 'mixed_internal_and_approach'
        truth_group = 'all_same' if verdicts == {'same'} else 'all_different' if verdicts == {'different'} else (
            'all_unresolved' if verdicts == {'unresolved'} else 'mixed_truth_verdicts')
        road_group = 'all_same' if road_verdicts == {'same'} else 'all_different' if road_verdicts == {'different'} else (
            'all_unresolved' if road_verdicts == {'unresolved'} else 'mixed_truth_verdicts')
        case_types[type_group] += 1
        case_truths[truth_group] += 1
        case_roads[road_group] += 1
        concurrency['ego_front_internal_with_any_blocker_body_internal_next_frame'] += int(any_body_internal_next)
        examples.setdefault(type_group+':'+truth_group, row)
    return dict(cases=len(records), reservation_counts=dict(reservations), case_blocker_types=dict(case_types),
                case_physical_truth_groups=dict(case_truths), case_road_target_truth_groups=dict(case_roads),
                observed_next_frame_concurrency=dict(concurrency), truth_resolution_sources=dict(sources),
                blocker_physical_connection_truth=dict(truth_counts), blocker_road_target_truth=dict(road_counts),
                examples=list(examples.values()))


def run(cache_base, paired_base, output, workers=4):
    cache_base, paired_base, output = Path(cache_base), Path(paired_base), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='SELECTING_FROZEN_CONFLICTS', new_fits=0, new_native_starts=0, free_rollouts=0)
    (output/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        selected, remainder = select_focus(paired_base)
        expected = [76116901, 76116902, 76116903, 76116904]
        if sorted(selected) != expected:
            raise ValueError('The fixed four training seeds must own the selected cohort')
        static = json.loads((cache_base/'static_graph_server_only.json').read_text())
        laws = json.loads((cache_base/'target_laws_server_only.json').read_text())
        tasks = [(seed, cache_base/'training'/f'seed{seed}', static, laws, selected[seed], output) for seed in expected]
        receipt.update(status='AUDITING_CURRENT_BLOCKERS', selected_cases=374, frozen_remaining_categories=remainder)
        (output/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if workers == 1:
            results = [_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                results = list(pool.map(_seed, tasks))
        counters = {name: Counter() for name in ('reservation_counts', 'case_blocker_types',
                    'case_physical_truth_groups', 'case_road_target_truth_groups', 'observed_next_frame_concurrency',
                    'truth_resolution_sources', 'blocker_physical_connection_truth', 'blocker_road_target_truth')}
        examples = []
        for result in results:
            examples.extend(result.pop('examples'))
            for name in counters:
                counters[name].update(result[name])
        (output/'examples_server_only.json').write_text(json.dumps(examples, indent=2, allow_nan=False)+'\n')
        receipt.update(status='COMPLETE_FROZEN_CONFLICT_AUDIT', reproduction_mismatches=0,
            **{name: dict(values) for name, values in counters.items()}, by_seed=results,
            baseline_features_source=str(Path(baseline.__file__).resolve()),
            receiving_features_source=str(Path(receiving.__file__).resolve()),
            interpretation='Observed paths and timing are posthoc diagnostic truth only. Connection differences are evidence, not causal recovery; overlapping full-junction reservations do not locate actual physical conflict-point crossings. Any later local-conflict correction must constrain both initial entry and committed internal progression; this audit changes neither.')
    except Exception as error:
        import traceback
        receipt.update(status='HOLD_CONFLICT_AUDIT', error=f'{type(error).__name__}: {error}')
        (output/'error.txt').write_text(traceback.format_exc())
    (output/'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--paired-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.paired_base, args.output, args.workers)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'COMPLETE_FROZEN_CONFLICT_AUDIT' else 2)
