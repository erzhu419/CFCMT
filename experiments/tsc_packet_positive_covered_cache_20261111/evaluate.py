"""Build the registered exact/mixture LC caches and read two fixed points."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110 import cache as exact
from experiments.tsc_packet_conditional_cache_20261110.verify import physical_evaluation_forbidden
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_lane_change_holdout_20261111.evaluate import exact_budget
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw
from experiments.tsc_packet_mixture_confirmation_20261111.sampler import MixtureSampler
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import collect_receipts, primary_response_budget
from experiments.tsc_packet_real_particle_pilot_20261110.register import save
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES

from .prepare import prepare
from .readout import aggregate_records, read_case


_kernels = None


def set_worker_kernels(kernels):
    global _kernels
    _kernels = kernels


def load_reused_pool(task):
    receipt = task['proposed_reused_cache']
    with gzip.open(receipt['particle_path'], 'rt') as stream:
        data = json.load(stream)
    proposal = data['proposal']
    if (len(data['particles']) != 256 or data['focal_id'] != task['reference']['packet_id']
            or data['stochastic_participants'] != task['required_participants']
            or data['frame_actor_count'] != task['frame_actor_count']
            or data['eligibility'] != task['eligibility']
            or data['parameter_names'] != list(PARAMETERS) or data['feature_names'] != list(FEATURE_NAMES)
            or proposal['design_weights'] != {'0': .5, '1': .5}
            or proposal['seeds'] != {'0': receipt['seed_zero'], '1': receipt['seed_fitted']}):
        raise ValueError('A reused mixture pool differs from its frozen identity or proposal')
    return ParticleLaw(data)


def inspect_case(task, candidate, output):
    directory = output/task['stem']
    directory.mkdir(exist_ok=False)
    save(directory/'started.json', task)
    began = time.monotonic()
    counts = dict(physical_calls=0, canonical_feature_evaluations=0, leaf_visits=0)
    sampler = None
    row = {k: task[k] for k in ('case_index', 'reference', 'seed', 'y', 'raw_weight',
                               'objective_weight', 'proposed_method')}
    row['reused_cache'] = task['proposed_reused_cache'] is not None
    try:
        kernel = _kernels[(task['seed'], task['reference']['time_sec'])]
        if task['proposed_method'] == 'exact':
            with exact_budget(task, counts, directory/'progress.json'):
                law = exact.build_cache(kernel, task['reference']['packet_id'], max_stochastic=10)
            if (counts['physical_calls'] != law.stats['physical_calls']
                    or counts['leaf_visits'] != law.stats['leaf_visits']
                    or counts['canonical_feature_evaluations'] != len(law.data['goal_configurations'])*task['required_actor_count']):
                raise ArithmeticError('Exact construction counters differ from the frozen law')
            cache_name = 'exact_cache_server_only.json.gz'
        else:
            if row['reused_cache']:
                with physical_evaluation_forbidden():
                    pool = load_reused_pool(task)
            else:
                def progress(stats):
                    save(directory/'progress.json', stats)
                sampler = MixtureSampler(kernel, task['reference']['packet_id'], candidate,
                    max_stochastic=10, physical_limit=task['physical_limit'],
                    feature_limit=task['feature_limit'], progress=progress)
                with primary_response_budget(sampler, 256) as accounting:
                    pool = sampler.draw_mixture(256, task['seed_zero'], task['seed_fitted'])
                counts.update(physical_calls=sampler.stats['physical_calls'],
                              canonical_feature_evaluations=sampler.stats['canonical_feature_evaluations'])
                if accounting['resolver_constructions'] != counts['physical_calls']:
                    raise ArithmeticError('Sampled particle and resolver counters differ')
            law = MixtureParticleLaw(pool.data, candidate)
            cache_name = 'mixture_cache_server_only.json.gz'
        if law.data['stochastic_participants'] != task['required_participants']:
            raise ValueError('The cache changed the complete registered stochastic set')
        if row['reused_cache']:
            row['cache_path'] = task['proposed_reused_cache']['particle_path']
        else:
            with gzip.open(directory/cache_name, 'wt', compresslevel=1) as stream:
                json.dump(law.data, stream, allow_nan=False)
            row['cache_path'] = str(directory/cache_name)
        with physical_evaluation_forbidden():
            row['readouts'], gradients = read_case(law, candidate, task['y'])
        save(directory/'gradients_server_only.json', {view: {point: value.tolist() if value is not None else None
            for point, value in points.items()} for view, points in gradients.items()})
        row['status'] = 'COMPLETE_POSITIVE_COVERED_CACHE_CASE'
    except Exception as failure:
        if sampler is not None:
            counts.update(physical_calls=sampler.stats['physical_calls'],
                          canonical_feature_evaluations=sampler.stats['canonical_feature_evaluations'])
        row.update(status='HOLD_POSITIVE_COVERED_CACHE_CASE', error=f'{type(failure).__name__}: {failure}')
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
    row.update(**counts, elapsed_sec=time.monotonic()-began)
    save(directory/'summary.json', row)
    print(json.dumps({k: row[k] for k in ('status', 'case_index', 'proposed_method', 'reused_cache',
        'physical_calls', 'canonical_feature_evaluations', 'leaf_visits', 'elapsed_sec')}), flush=True)
    return row


def run(base, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    completed, tasks, registration = [], [], {}
    error, stopped = None, False
    try:
        proof = read_json(base/'target_packet_mixture_confirmation/synthetic_result.json')
        if proof['status'] != 'PASS_SYNTHETIC_MIXTURE_GATE' or proof['full_forward_calls'] != 854:
            raise ValueError('The existing exact/mixture equivalence gate is required')
        candidate = frozen_candidate(base)
        tasks, kernels, registration = prepare(base, output)
        print(json.dumps(dict(stage='positive_covered_prepared', **registration)), flush=True)
        ordered = iter(sorted(tasks, key=lambda task: (
            -task['physical_limit']*task['frame_actor_count'], task['case_index'])))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                                 initializer=set_worker_kernels, initargs=(kernels,)) as executor:
            pending = {executor.submit(inspect_case, next(ordered), candidate, output) for _ in range(workers)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    completed.append(row)
                    stopped |= row['status'] != 'COMPLETE_POSITIVE_COVERED_CACHE_CASE'
                if not stopped:
                    for _ in range(workers-len(pending)):
                        task = next(ordered, None)
                        if task is not None:
                            pending.add(executor.submit(inspect_case, task, candidate, output))
    except Exception as failure:
        error = f'{type(failure).__name__}: {failure}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    completed, unknown = collect_receipts(output, completed)
    physical = sum(row['physical_calls'] for row in completed)+sum(row['physical_calls_lower_bound'] for row in unknown)
    features = sum(row['canonical_feature_evaluations'] for row in completed)+sum(
        row['canonical_feature_evaluations_lower_bound'] for row in unknown)
    leaves = sum(row['leaf_visits'] for row in completed)
    if physical > 17510 or features > 24738 or leaves > 65436:
        error = 'Frozen construction budget exceeded'
    complete = (error is None and not unknown and len(completed) == 424
                and all(row['status'] == 'COMPLETE_POSITIVE_COVERED_CACHE_CASE' for row in completed))
    aggregate = None
    if complete:
        try:
            gradients = {}
            for task in tasks:
                data = read_json(output/task['stem']/'gradients_server_only.json')
                gradients[task['case_index']] = {view: {point: np.asarray(value, dtype=float) if value is not None else None
                    for point, value in points.items()} for view, points in data.items()}
            aggregate = aggregate_records(completed, gradients, expected_references=424)
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
            (output/'error_server_only.txt').write_text(traceback.format_exc())
            complete = False
    primary_support = bool(complete and aggregate is not None and all(
        point['all_references_supported']
        and point['weighted_nll_contribution']['state'] == 'finite'
        and point['weighted_gradient']['defined']
        for point in aggregate['views']['primary']['points'].values()))
    result = dict(status='COMPLETE_POSITIVE_COVERED_TWO_POINT_CACHE' if complete else 'HOLD_POSITIVE_COVERED_CACHE_TECHNICAL',
        error=error, registration=registration, aggregate=aggregate, completed_case_attempts=len(completed),
        primary_two_point_support_passed=primary_support,
        scientific_status='PASS_PRIMARY_TWO_POINT_SUPPORT' if primary_support else 'HOLD_PRIMARY_TWO_POINT_SUPPORT',
        failed_cases=[{k:row[k] for k in ('case_index', 'reference', 'error')} for row in completed if 'error' in row],
        incomplete_started_cases=unknown, stopped_after_failure=stopped or error is not None,
        undispatched_case_indices=[task['case_index'] for task in tasks
            if task['case_index'] not in {row['case_index'] for row in completed+unknown}],
        consumed_counters_exact=not unknown, full_forward_calls_lower_bound=physical,
        canonical_feature_evaluations_lower_bound=features, exact_leaf_visits_lower_bound=leaves,
        new_full_forward_calls=physical if not unknown else None,
        new_canonical_feature_evaluations=features if not unknown else None,
        exact_leaf_visits=leaves if not unknown else None,
        new_native_starts=0, new_nuisance_estimations=0, new_fits=0, optimizer_calls=0,
        calibration_seeds_used=False, original_fit_hold_preserved=True, model_admitted=False,
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('status', 'error', 'completed_case_attempts',
        'new_full_forward_calls', 'new_canonical_feature_evaluations', 'exact_leaf_visits', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', default=4, choices=(1, 2, 3, 4), type=int)
    args = parser.parse_args()
    result = run(args.base, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_POSITIVE_COVERED_TWO_POINT_CACHE' else 2)
