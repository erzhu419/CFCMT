"""Rebuild every frozen training reference and evaluate theta=0 only.

Arrays and exact reference details remain on the server. No sampler, history
estimator, optimizer, gradient, simulator or observed future feature is called.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
from math import isfinite, log, pi
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.training_data import (
    HEADS, RESERVOIR_SEED, iter_transitions, _candidates, _pack,
)
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight, SEEDS
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save
from .conditional_features import prepare_feature_kernel, candidate_features


TOL = 1e-8


def reference_key(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def zero_log_probability(head, row):
    """Untruncated Gaussian sigma1 and actual theta0 discrete mixture support."""
    prior = np.asarray(row['prior'], dtype=float)
    risk = np.asarray(row['risk'], dtype=bool)
    if head == 'motion':
        component = -.5*(float(row['y'])-np.asarray(row['mean0_mps']))**2-.5*log(2*pi)
    elif head == 'lane_change':
        mask = np.asarray(row['mask'], dtype=bool).copy()
        mask[~risk | (prior == 0)] = (False, True, False)
        allowed = mask[:, int(row['y'])]
        component = np.where(allowed, -np.log(mask.sum(axis=1)), -np.inf)
    else:
        component = np.where(risk, -log(2.), 0. if row['y'] == 0 else -np.inf)
    with np.errstate(divide='ignore'):
        return float(logsumexp(np.log(prior)+component))


def new_head_statistics():
    return dict(reservoir_records=0, retained=0, excluded=0, excluded_positive_labels=0,
        full_unsupported_labels=0, retained_unsupported_labels=0, excluded_unsupported_labels=0,
        full_finite_nll_sum=0., retained_finite_nll_sum=0.,
        observed_above_max_mean0_count=0, observed_above_max_physical_ceiling_count=0)


def add_statistics(stat, row, log_probability, retained):
    stat['reservoir_records'] += 1
    stat['retained' if retained else 'excluded'] += 1
    positive = row['y'] != 1 if row.get('head') == 'lane_change' else row['y'] > 0
    stat['excluded_positive_labels'] += int(not retained and positive)
    unsupported = not isfinite(log_probability)
    stat['full_unsupported_labels'] += int(unsupported)
    stat['retained_unsupported_labels'] += int(retained and unsupported)
    stat['excluded_unsupported_labels'] += int(not retained and unsupported)
    if not unsupported:
        stat['full_finite_nll_sum'] -= log_probability
        if retained:
            stat['retained_finite_nll_sum'] -= log_probability
    if row.get('head') == 'motion':
        supported = [index for index, prior in enumerate(row['prior']) if prior > 0]
        stat['observed_above_max_mean0_count'] += int(row['y'] > max(row['mean0_mps'][index] for index in supported)+TOL)
        stat['observed_above_max_physical_ceiling_count'] += int(row['y'] > max(row['physical_ceiling_mps'][index] for index in supported)+TOL)


def finish_statistics(stat):
    result = dict(stat)
    for name, count, unsupported, total in (
            ('full', stat['reservoir_records'], stat['full_unsupported_labels'], stat['full_finite_nll_sum']),
            ('retained', stat['retained'], stat['retained_unsupported_labels'], stat['retained_finite_nll_sum'])):
        result[name+'_mean_nll'] = total/count if count and not unsupported else None
        result[name+'_objective_status'] = ('ZERO_LABEL_LIKELIHOOD' if unsupported else
                                            'FINITE' if count else 'EMPTY')
    return result


def load_old_yield(path):
    """Keep only prior/mean/risk from saved pair rows, not large blocker traces."""
    result = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = reference_key(row['reference'])
            if key in result:
                raise ValueError('Frozen yield references must be unique')
            result[key] = dict(reference=row['reference'], prior=row['prior'],
                candidates=[dict(intent=pair['intent'], connection_key=pair['connection_key'],
                    mean0_mps=pair['candidate']['upper'], yield_risk=pair['candidate']['yield_risk'])
                    for pair in row['paired']])
    return result


def verify_old_yield(saved, reference, intents, prior, candidates):
    key = reference_key(reference)
    old = saved.pop(key)
    if old['reference'] != reference or old['prior'] != prior or [row['intent'] for row in old['candidates']] != intents:
        raise ValueError('Frozen exact yield reference, prior values and intent order must reproduce')
    for before, after in zip(old['candidates'], candidates):
        if (before['connection_key'] != after['connection_key']
                or abs(before['mean0_mps']-after['mean0_mps']) > TOL
                or before['yield_risk'] != after['yield_risk']):
            raise ValueError('Shared r0 mean/physical connection/yield risk changed during full-feature wiring')


def build_component_row(head, reference, priors, candidates, feature_vectors):
    risk_key = {'motion': 'motion_risk', 'start': 'start_risk', 'yield_entry': 'yield_risk'}
    if head == 'lane_change':
        risk = [row['lane_mask']['left'] or row['lane_mask']['right'] for row in candidates]
    else:
        risk = [row[risk_key[head]] for row in candidates]
    row = dict(head=head, X=feature_vectors, prior=list(priors.values()), risk=risk, y=reference['y'])
    if head == 'motion':
        row.update({name: [candidate[name] for candidate in candidates]
                    for name in ('upper', 'a', 'b', 'mean0_mps', 'physical_ceiling_mps')})
    if head == 'lane_change':
        row['mask'] = [[candidate['lane_mask'][name] for name in ('left', 'stay', 'right')]
                       for candidate in candidates]
    return row


def _pack_full(rows, head):
    arrays = _pack(rows, head)
    arrays['retained'] = np.asarray([any(row['risk']) for row in rows], dtype=bool)
    arrays['zero_log_probability'] = np.asarray([zero_log_probability(head, row) for row in rows])
    if head == 'motion':
        arrays['mean0_mps'] = arrays.pop('upper')
        arrays['physical_ceiling_mps'] = np.zeros_like(arrays['mean0_mps'])
        for index, row in enumerate(rows):
            arrays['physical_ceiling_mps'][index, :len(row['prior'])] = row['physical_ceiling_mps']
    return arrays


def _seed(task):
    seed, directory, static, laws, reservoir, old_sampling, width, old_yield_file, original_lane, output = task
    began = time.monotonic()
    print(json.dumps(dict(stage='mean_feature_begin', seed=seed)), flush=True)
    table = build_entity_geometry(RoadTopology(static), width)
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    saved_yield = load_old_yield(old_yield_file) if old_yield_file is not None else {}
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
    with gzip.open(output/f'seed{seed}_references_server_only.jsonl.gz', 'xt', compresslevel=1) as refs:
        for frame, unused_next, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t % 300 == 0:
                print(json.dumps(dict(stage='mean_feature_progress', seed=seed, time_sec=t)), flush=True)
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
                for head, reference in head_refs:
                    if head == 'yield_entry' and old_yield_file is not None:
                        verify_old_yield(saved_yield, reference, list(priors), priors, candidates)
                        counts['old_shared_yield_rows_reproduced'] += 1
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
                    if head == 'motion':
                        detail.update(mean0_mps=row['mean0_mps'], physical_ceiling_mps=row['physical_ceiling_mps'])
                    if head == 'lane_change':
                        detail['mask'] = row['mask']
                        detail['receiving_road_potential'] = [candidate['lane_detail']['receiving_road_potential'] for candidate in candidates]
                    refs.write(json.dumps(detail, allow_nan=False, separators=(',', ':'))+'\n')
                    if head == 'lane_change' and reference_key(reference) in original_lane:
                        expected = original_lane.pop(reference_key(reference))
                        if expected['y'] != reference['y']:
                            raise ValueError('The original excluded lane label must remain unchanged')
                        original_rows.append(dict(detail, components=[dict(intent=intent,
                            mean0_mps=candidate['mean0_mps'], lane_detail=candidate['lane_detail'])
                            for intent, candidate in zip(priors, candidates)]))
    if saved_yield or original_lane:
        raise ValueError('Every saved shared-yield row and original lane-positive reference must be consumed')
    for head in HEADS:
        if len(rows[head]) != len(reservoir[head]):
            raise ValueError('Every frozen reservoir reference must have one retained/excluded disposition')
    arrays = {f'{head}_{key}': value for head in HEADS for key, value in _pack_full(rows[head], head).items()}
    np.savez_compressed(output/f'seed{seed}_full_features_server_only.npz', **arrays)
    save(output/f'seed{seed}_original_lane9_server_only.json', original_rows)
    result = dict(seed=seed, heads=stats, sampling=sampling, counts=dict(counts),
        original_lane_cases=[{key: row[key] for key in ('reference', 'retained', 'risk', 'mask', 'label_supported_at_zero')}
                             for row in original_rows], elapsed_sec=time.monotonic()-began)
    save(output/f'seed{seed}_summary.json', result)
    print(json.dumps(dict(stage='mean_feature_complete', seed=seed, records={head: len(rows[head]) for head in HEADS},
                          elapsed_sec=result['elapsed_sec'])), flush=True)
    return result


def summarize(results):
    totals = {head: new_head_statistics() for head in HEADS}
    counts = Counter()
    original = []
    for result in results:
        counts.update(result['counts'])
        original.extend(result['original_lane_cases'])
        for head in HEADS:
            for key, value in result['heads'][head].items():
                totals[head][key] += value
    heads = {head: finish_statistics(stat) for head, stat in totals.items()}
    objective = {}
    for group in ('full', 'retained'):
        parts = {head: stat[group+'_mean_nll'] for head, stat in heads.items()}
        objective[group] = dict(head_mean_nll=parts,
            sum_four_head_mean_nll=sum(parts.values()) if all(value is not None for value in parts.values()) else None,
            unsupported_label_count=sum(stat[group+'_unsupported_labels'] for stat in heads.values()))
    return dict(status='COMPLETE_ZERO_PARAMETER_MEAN_SUPPORT_AUDIT', heads=heads, zero_objective=objective,
        counts=dict(counts), original_lane9=original, by_seed=results,
        feature_count=len(FEATURE_NAMES), coefficient_count=5*len(FEATURE_NAMES), motion_sigma_mps=1.,
        new_native_starts=0, new_fits=0, new_nuisance_estimations=0, free_rollouts=0,
        laws_reestimated=False, reservoirs_resampled=False, future_used_as_feature=False,
        optimizer_compatible=False, gradients_available=False,
        objective_definition='Theta0 only: sum four head mean mixture NLLs; zero L2. Gaussian motion is untruncated sigma1.',
        mean_definition='Fresh focal local MotionResolver with fixed prior intents, zero other residuals and potential start/yield. It is not full globally ordered forward expectation.',
        lane_mask_definition='Same candidate local accepted longitudinal t+1, other lateral actions stay. Receiving-road legality is the union of positive-prior next-road intents.',
        limitations='Theta0 risks and lane masks do not define the general-theta joint objective. Mu0 is a configuration mean, not an all-residual upper; only physical free P is a universal model speed ceiling. Mask exclusions and discrete zero likelihoods include the original full reservoir; audit completion is not permission to fit or a support pass.')


def run(cache, shared, exclusion, preflight, output, workers=4):
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
        original = json.loads((exclusion/'output'/'lane_details_server_only.json').read_text())
        original = [row['reference'] for row in original]
        if len(original) != 9 or len({reference_key(row) for row in original}) != 9:
            raise ValueError('The exact original9 excluded lane-positive references are required')
        tasks = [(seed, cache/'training'/f'seed{seed}', static, laws, reservoirs[str(seed)], sampling[str(seed)], width,
            shared/'output'/f'seed{seed}_pairs_server_only.jsonl.gz',
            [reference for reference in original if int(reference['seed']) == seed], output) for seed in SEEDS]
        if workers == 1:
            results = [_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                results = list(pool.map(_seed, tasks))
        result = summarize(results)
        if result['counts']['old_shared_yield_rows_reproduced'] != 40000 or len(result['original_lane9']) != 9:
            raise ValueError('All40000 shared r0 rows and original9 lane refs must reproduce')
        result.update(entity_width_m=width, lane_width_source=width_source, centerline_observed=proof['centerline_observed'],
            training_seeds=SEEDS, feature_names=list(FEATURE_NAMES), feature_source=str(Path(__file__).with_name('conditional_features.py')),
            sampling={str(row['seed']): row['sampling'] for row in results})
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_ZERO_PARAMETER_MEAN_SUPPORT_AUDIT', error=f'{type(error).__name__}: {error}',
                      new_native_starts=0, new_fits=0, free_rollouts=0)
    save(output/'result.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('cache-base', 'same-step-base', 'exclusion-base', 'preflight', 'output'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.same_step_base, args.exclusion_base, args.preflight, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_ZERO_PARAMETER_MEAN_SUPPORT_AUDIT' else 2)
