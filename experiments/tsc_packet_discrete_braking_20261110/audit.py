"""Rebuild only the discrete candidate on the existing frozen training refs.

The continuous comparator is read from its saved output, never recomputed.
No simulator, sampler, history estimator, optimizer or future feature is used.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
from math import isfinite
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.training_data import HEADS, RESERVOIR_SEED, iter_transitions, _candidates
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight, SEEDS
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save
from experiments.tsc_packet_mean_alignment_20261110.audit import (
    reference_key, zero_log_probability, new_head_statistics, add_statistics,
    build_component_row, _pack_full, summarize,
)
from .conditional_features import prepare_feature_kernel, candidate_features


def _seed(task):
    seed, directory, static, laws, reservoir, old_sampling, width, original_lane, fixed_cases, output = task
    began = time.monotonic()
    print(json.dumps(dict(stage='discrete_feature_begin', seed=seed)), flush=True)
    table = build_entity_geometry(RoadTopology(static), width)
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    selected = defaultdict(lambda: defaultdict(list))
    for head in HEADS:
        for reference in reservoir[head]:
            selected[reference['time_sec']][reference['packet_id']].append((head, reference))
    rows = {head: [] for head in HEADS}
    stats = {head: new_head_statistics() for head in HEADS}
    sampling = deepcopy(old_sampling)
    for stat in sampling.values():
        stat.update(retained=0, allzero_excluded=0, allzero_excluded_positive_labels=0)
    counts, original_rows = Counter(), []
    original_lane = {reference_key(reference): reference for reference in original_lane}
    fixed = {reference_key(case["reference"]): case for case in fixed_cases}
    fixed_rows = []
    with gzip.open(output/f'seed{seed}_references_server_only.jsonl.gz', 'xt', compresslevel=1) as refs:
        for frame, unused_next, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t % 300 == 0:
                print(json.dumps(dict(stage='discrete_feature_progress', seed=seed, time_sec=t)), flush=True)
            if t not in selected:
                continue
            kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'],
                RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
            if kernel.signals != action['applied_states']:
                raise ValueError('Current executor/request must reproduce the recorded current interval signal')
            for pid, head_refs in selected[t].items():
                priors = _candidates(kernel, pid)
                candidates = []
                for intent in priors:
                    try:
                        candidates.append(candidate_features(kernel, pid, intent))
                    except Exception as error:
                        save(output/f'seed{seed}_failed_query_context_server_only.json', dict(
                            seed=seed, time_sec=t, packet_id=pid, candidate_intent=intent,
                            focal_current_packet=frame['packets'][pid],
                            lane_neighbour=getattr(error, 'lane_neighbour_context', None),
                            error=f'{type(error).__name__}: {error}'))
                        raise
                feature_vectors = [kernel.model.features(candidate['context']) for candidate in candidates]
                counts['focal_state_queries'] += 1
                counts['intent_queries'] += len(candidates)
                counts['receiving_road_potential_queries'] += sum(row['lane_detail']['receiving_road_potential'] for row in candidates)
                counts['dependency_cycle_queries'] += sum(row['admission_detail']['receiving_binding_summary']['cycle_encountered'] for row in candidates)
                fixed_case = fixed.pop((seed, t, pid), None)
                if fixed_case is not None:
                    fixed_rows.append(dict(case_id=fixed_case['case_id'], head=fixed_case['head'],
                        reference=fixed_case['reference'], current_packet=frame['packets'][pid],
                        components=[dict(intent=intent, prior=priors[intent], **candidate)
                                    for intent, candidate in zip(priors, candidates)]))
                for head, reference in head_refs:
                    row = build_component_row(head, reference, priors, candidates, feature_vectors)
                    retained = any(row['risk'])
                    logp = zero_log_probability(head, row)
                    add_statistics(stats[head], row, logp, retained)
                    sampling[head]['retained' if retained else 'allzero_excluded'] += 1
                    if not retained:
                        sampling[head]['allzero_excluded_positive_labels'] += int(reference['y'] != 1 if head == 'lane_change' else reference['y'] > 0)
                    rows[head].append(row)
                    detail = dict(head=head, reference=reference, candidate_intents=list(priors), prior=list(priors.values()),
                        risk=row['risk'], retained=bool(retained), label_supported_at_zero=isfinite(logp),
                        zero_log_probability=logp if isfinite(logp) else None)
                    detail.update(mean0_mps=[c['mean0_mps'] for c in candidates],
                                  physical_ceiling_mps=[c['physical_ceiling_mps'] for c in candidates])
                    if head == 'lane_change':
                        detail['mask'] = row['mask']
                        detail['receiving_road_potential'] = [candidate['lane_detail']['receiving_road_potential'] for candidate in candidates]
                    refs.write(json.dumps(detail, allow_nan=False, separators=(',', ':'))+'\n')
                    if head == 'lane_change' and reference_key(reference) in original_lane:
                        expected = original_lane.pop(reference_key(reference))
                        if expected['y'] != reference['y']:
                            raise ValueError('The original excluded lane label must remain unchanged')
                        original_rows.append(dict(detail, components=[dict(intent=intent,
                            mean0_mps=candidate['mean0_mps'], physical_ceiling_mps=candidate['physical_ceiling_mps'],
                            lane_detail=candidate['lane_detail'])
                            for intent, candidate in zip(priors, candidates)]))
    if original_lane or fixed:
        raise ValueError('Every original lane-positive and preselected fixed15 reference must be consumed')
    for head in HEADS:
        if len(rows[head]) != len(reservoir[head]):
            raise ValueError('Every frozen reservoir reference must have one retained/excluded disposition')
    arrays = {f'{head}_{key}': value for head in HEADS for key, value in _pack_full(rows[head], head).items()}
    np.savez_compressed(output/f'seed{seed}_full_features_server_only.npz', **arrays)
    save(output/f'seed{seed}_original_lane9_server_only.json', original_rows)
    save(output/f'seed{seed}_fixed15_server_only.json', fixed_rows)
    result = dict(seed=seed, heads=stats, sampling=sampling, counts=dict(counts),
        original_lane_cases=[{key: row[key] for key in ('reference', 'retained', 'risk', 'mask', 'label_supported_at_zero')}
                             for row in original_rows], elapsed_sec=time.monotonic()-began)
    save(output/f'seed{seed}_summary.json', result)
    print(json.dumps(dict(stage='discrete_feature_complete', seed=seed, records={head: len(rows[head]) for head in HEADS},
                          elapsed_sec=result['elapsed_sec'])), flush=True)
    return result


def run(cache, exclusion, selection, preflight, output, workers=4):
    output.mkdir(parents=True, exist_ok=False)
    try:
        width, proof = verify_entity_preflight(preflight)
        static, width_source = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        reservoirs = json.loads((cache/'training_reservoir_server_only.json').read_text())
        sampling = json.loads((cache/'training_sampling_result.json').read_text())
        if sorted(map(int, reservoirs)) != SEEDS or sorted(map(int, sampling)) != SEEDS:
            raise ValueError('Only the four complete frozen training seeds are allowed')
        if any(len(reservoirs[str(seed)][head]) != 10000 for seed in SEEDS for head in HEADS):
            raise ValueError('All160000 original references are required, without refill')
        original = [row['reference'] for row in json.loads((exclusion/'output'/'lane_details_server_only.json').read_text())]
        fixed = json.loads(selection.read_text())['cases']
        if len(original) != 9 or len({reference_key(row) for row in original}) != 9:
            raise ValueError('The exact original9 excluded lane-positive references are required')
        if len(fixed) != 15 or len({reference_key(row['reference']) for row in fixed}) != 15:
            raise ValueError('All15 preselected mechanism references are required')
        tasks = [(seed, cache/'training'/f'seed{seed}', static, laws, reservoirs[str(seed)], sampling[str(seed)], width,
            [reference for reference in original if int(reference['seed']) == seed],
            [case for case in fixed if int(case['reference']['seed']) == seed], output) for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(_seed, tasks))
        result = summarize(results)
        result['status'] = 'COMPLETE_ZERO_PARAMETER_DISCRETE_BRAKING_AUDIT'
        if len(result['original_lane9']) != 9:
            raise ValueError('Every original9 lane reference must remain in the readout')
        result.update(entity_width_m=width, lane_width_source=width_source, centerline_observed=proof['centerline_observed'],
            training_seeds=SEEDS, feature_names=list(FEATURE_NAMES), feature_source=str(Path(__file__).with_name('conditional_features.py')),
            sampling={str(row['seed']): row['sampling'] for row in results},
            formula='Discrete dt1 following and active stationary-stop safe speed. Yellow branch predicate remains frozen.',
            baseline_recomputed=False, optimizer_calls=0)
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_ZERO_PARAMETER_DISCRETE_BRAKING_AUDIT', error=f'{type(error).__name__}: {error}',
                      new_native_starts=0, new_fits=0, free_rollouts=0)
    save(output/'result.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('cache-base', 'exclusion-base', 'selection', 'preflight', 'output'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.exclusion_base, args.selection, args.preflight, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_ZERO_PARAMETER_DISCRETE_BRAKING_AUDIT' else 2)
