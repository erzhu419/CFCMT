"""Bounded fixed-parameter mixture confirmation on the original fifteen states."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import (
    collect_receipts, counters_from_sampler, pool_eligibility, primary_response_budget,
)
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, save
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology

from .readout import MixtureParticleLaw, compare_readouts, evaluate_pool
from .sampler import MixtureSampler


PARTICLES = 256
CASE_COUNT = 15
ACTOR_COUNT = 67
_kernels = None


def read_json(path):
    return json.loads(path.read_text())


def frozen_candidate(base):
    root = base / 'target_packet_particle_first_fit' / 'fit'
    fit = read_json(root / 'summary.json')
    saved = read_json(root / 'coefficients_server_only.json')
    candidate = np.asarray(saved['coefficients'], dtype=float)
    if (fit['status'] != 'HOLD_FIRST_FIT_OPTIMIZER_TERMINATION'
            or fit['new_fits'] != 1 or fit['optimizer_invocations'] != 1
            or fit['optimizer']['nit'] != 200 or saved['parameter_names'] != list(PARAMETERS)
            or saved['feature_names'] != list(FEATURE_NAMES)
            or candidate.shape != (len(PARAMETERS), len(FEATURE_NAMES))
            or not np.isfinite(candidate).all()):
        raise ValueError('The original frozen first-fit point is required')
    return candidate


def register_cases(base, output, counters):
    """Initialize every original current frame once and verify bindings before dispatch."""
    root = base / 'target_packet_real_particle_pilot'
    preflight = root / 'preflight'
    registered = read_json(preflight / 'summary.json')
    prior = read_json(root / 'evaluation/summary.json')
    exact = read_json(base / 'target_packet_conditional_panel/evaluation/summary.json')
    cases = registered['cases']
    if (registered['status'] != 'PASS_FROZEN_REAL_PARTICLE_COHORT'
            or prior['status'] != 'PASS_BOUNDED_REAL_PARTICLE_PILOT'
            or [r['case_index'] for r in cases] != list(range(CASE_COUNT))
            or [r['case_index'] for r in prior['cases']] != list(range(CASE_COUNT))
            or sum(r['required_actor_count'] for r in cases) != ACTOR_COUNT):
        raise ValueError('The complete frozen fifteen-case pilot is required')
    cache = base / 'target_packet_training'
    static, _ = load_widths(read_json(cache / 'static_graph_server_only.json'))
    laws = read_json(cache / 'target_laws_server_only.json')
    width, _ = verify_entity_preflight(base / 'target_packet_entity_footprint/entity_input_preflight.json')
    geometry = build_entity_geometry(RoadTopology(static), width)
    executors = {seed: read_json(cache / 'training' / f'seed{seed}/executor_seed_server_only.json')
                 for seed in {r['reference']['seed'] for r in cases}}
    exact_by_ref = {reference_key(r['reference']): r for r in exact['cases']}
    tasks, kernels = [], {}
    for row, old in zip(cases, prior['cases']):
        index = row['case_index']
        ref = row['reference']
        seed, t, focal = reference_key(ref)
        if reference_key(old['reference']) != (seed, t, focal):
            raise ValueError(f'Original reference mismatch in case {index}')
        current = read_json(preflight / row['current_cache'])
        saved = read_json(preflight / row['selection_cache'])
        counters['current_frame_initializations'] += 1
        kernel = prepare_zero_kernel(static, current['frame'], executors[seed], laws,
                                     current['action'], RESERVOIR_SEED + seed*100000 + t, geometry)
        selected = select_stochastic(kernel, focal, max_stochastic=10)
        if (json.loads(json.dumps(selected)) != saved['selection']
                or selected['required_participants'] != row['required_participants']
                or len(kernel.state['packets']) != row['frame_actor_count']
                or any(kernel.state['packets'][key]['_movement_key'] != item['movement_key']
                       for key, item in saved['internal_bindings'].items())):
            raise ValueError(f'Original geometry or committed movement changed in case {index}')
        observations = [{k: obs[k] for k in ('head', 'y', 'manifest_ordinal')}
                        for obs in old['readouts']['1024']['0']['observed_heads']]
        eligibility = pool_eligibility(kernel, focal)
        if not observations or any(not eligibility[obs['head']] for obs in observations):
            raise ValueError(f'Original observations or selector changed in case {index}')
        native_lc = next((int(o['y']) for o in observations if o['head'] == 'lane_change'), None)
        native_speed = None
        if index < 9:
            truth = exact_by_ref[(seed, t, focal)]
            native_speed = truth['native_speed_mps']
            if truth['observed_lc_label'] != native_lc or not np.isfinite(native_speed):
                raise ValueError(f'Saved exact native label mismatch in case {index}')
        stem = f'case{index:02d}_seed{seed}_t{t}'
        old_pool = root / 'evaluation' / stem / 'particles_server_only.json.gz'
        if not old_pool.is_file():
            raise FileNotFoundError(old_pool)
        kernels[index] = kernel
        tasks.append(dict(case_index=index, reference=ref, stem=stem,
            required_actor_count=row['required_actor_count'],
            required_participants=row['required_participants'],
            committed_internal_bindings=len(saved['internal_bindings']),
            old_pool=str(old_pool), observations=observations, native_lc_label=native_lc,
            native_speed_mps=native_speed, seed_zero=20261113+100000*index,
            seed_fitted=20261114+100000*index,
            physical_limit=PARTICLES, feature_limit=PARTICLES*row['required_actor_count']))
    if Counter(o['head'] for r in tasks for o in r['observations']) != {
            'lane_change': 10, 'motion': 1, 'yield_entry': 2, 'start': 2}:
        raise ValueError('Original fifteen observed-head references changed')
    save(output / 'registered.json', dict(status='PASS_ORIGINAL_FIFTEEN_REGISTRATION',
        cases=tasks, **counters, full_forward_ceiling=CASE_COUNT*PARTICLES,
        feature_ceiling=ACTOR_COUNT*PARTICLES, registered_at_utc=datetime.now(timezone.utc).isoformat()))
    return tasks, kernels


def set_worker_kernels(kernels):
    global _kernels
    _kernels = kernels


def inspect_case(task, anchor, output):
    index, ref = task['case_index'], task['reference']
    directory = output / task['stem']
    directory.mkdir(exist_ok=False)
    save(directory / 'started.json', task)
    began = time.monotonic()
    sampler = None
    accounting = dict(resolver_constructions=0, forbidden_feature_or_extra_resolver_attempts=0)
    row = dict(case_index=index, reference=ref, stop_new_cases=False)
    try:
        labels = dict(native_lc_label=task['native_lc_label'], native_speed_mps=task['native_speed_mps'])
        with gzip.open(task['old_pool'], 'rt') as stream:
            data = json.load(stream)
        if (len(data['particles']) != 1024 or data['focal_id'] != ref['packet_id']
                or data['stochastic_participants'] != task['required_participants']):
            raise ValueError('Saved q0 pool does not match the frozen reference')
        with physical_forbidden():
            row['old_q0'] = evaluate_pool(ParticleLaw(data), anchor, task['observations'],
                                          sizes=(256, 1024), **labels)
        del data

        def progress(counts):
            save(directory / 'progress.json', counts)
            print(json.dumps(dict(stage='mixture_progress', case_index=index,
                physical_calls=counts['physical_calls'],
                canonical_feature_evaluations=counts['canonical_feature_evaluations'])), flush=True)

        sampler = MixtureSampler(_kernels[index], ref['packet_id'], anchor, progress=progress,
                                 physical_limit=task['physical_limit'], feature_limit=task['feature_limit'])
        with primary_response_budget(sampler, PARTICLES) as accounting:
            pool = sampler.draw_mixture(PARTICLES, task['seed_zero'], task['seed_fitted'])
        if accounting['resolver_constructions'] != sampler.stats['physical_calls']:
            raise ArithmeticError('Recorded full-forward calls do not match resolver constructions')
        with gzip.open(directory / 'particles_server_only.json.gz', 'wt', compresslevel=1) as stream:
            json.dump(pool.data, stream, allow_nan=False)
        with physical_forbidden():
            mixture = MixtureParticleLaw(pool.data, anchor)
            row['mixture'] = evaluate_pool(mixture, anchor, task['observations'],
                                            sizes=(128, 256), **labels)
            row['comparison'] = compare_readouts(row['mixture'], row['old_q0'])
        row['status'] = 'COMPLETE_MIXTURE_CASE'
    except Exception as error:
        (directory / 'error_server_only.txt').write_text(traceback.format_exc())
        row.update(status='HOLD_MIXTURE_CASE', stop_new_cases=True,
                   error=f'{type(error).__name__}: {error}',
                   current_goal_tuple=getattr(error, 'current_goal_tuple', None))
    row.update(counters_from_sampler(sampler), **accounting, elapsed_sec=time.monotonic()-began)
    save(directory / 'summary.json', row)
    print(json.dumps({k:row[k] for k in ('status', 'case_index', 'physical_calls',
        'canonical_feature_evaluations', 'elapsed_sec')}), flush=True)
    return row


def run(base, gate, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    counters = dict(current_frame_initializations=0)
    completed, tasks, error = [], [], None
    stopped = False
    try:
        proof = read_json(gate)
        if (proof['status'] != 'PASS_SYNTHETIC_MIXTURE_GATE'
                or proof['full_forward_calls'] != proof['planned_full_forward_calls']
                or not 0 < proof['full_forward_calls'] <= 1024):
            raise ValueError('The fixed-mixture synthetic gate must pass first')
        candidate = frozen_candidate(base)
        tasks, kernels = register_cases(base, output, counters)
        print(json.dumps(dict(stage='original_fifteen_registered', **counters)), flush=True)
        ordered = iter(sorted(tasks, key=lambda r: (-r['required_actor_count'], r['case_index'])))
        # Fork shares the already registered kernels. Samplers make their own physical views.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                                 initializer=set_worker_kernels, initargs=(kernels,)) as executor:
            pending = {executor.submit(inspect_case, next(ordered), candidate, output) for _ in range(workers)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    completed.append(row)
                    stopped |= row['stop_new_cases']
                if not stopped:
                    for _ in range(workers-len(pending)):
                        task = next(ordered, None)
                        if task is not None:
                            pending.add(executor.submit(inspect_case, task, candidate, output))
    except Exception as failure:
        error = f'{type(failure).__name__}: {failure}'
        (output / 'error_server_only.txt').write_text(traceback.format_exc())
    completed, unknown = collect_receipts(output, completed)
    physical = sum(r['physical_calls'] for r in completed) + sum(r['physical_calls_lower_bound'] for r in unknown)
    features = sum(r['canonical_feature_evaluations'] for r in completed) + sum(
        r['canonical_feature_evaluations_lower_bound'] for r in unknown)
    if physical > CASE_COUNT*PARTICLES or features > ACTOR_COUNT*PARTICLES:
        error = 'The recorded run exceeded the frozen physical or feature ceiling'
    success = (error is None and not unknown and len(completed) == CASE_COUNT
               and all(r['status'] == 'COMPLETE_MIXTURE_CASE' for r in completed)
               and physical == CASE_COUNT*PARTICLES)
    result = dict(status='COMPLETE_FIXED_MIXTURE_CONFIRMATION' if success else 'HOLD_FIXED_MIXTURE_CONFIRMATION',
        error=error, cases=completed, incomplete_started_cases=unknown,
        completed_case_attempts=len(completed), stopped_after_failure=stopped or error is not None,
        undispatched_case_indices=[r['case_index'] for r in tasks
            if r['case_index'] not in {x['case_index'] for x in completed+unknown}],
        consumed_counters_exact=not unknown, full_forward_calls_lower_bound=physical,
        canonical_feature_evaluations_lower_bound=features,
        new_real_full_forward_calls=physical if not unknown else None,
        new_real_canonical_feature_evaluations=features if not unknown else None,
        **counters, feature_stage_full_forward_calls=0, parameter_readout_reuse_full_forward_calls=0,
        new_real_exact_reference_calls=0, new_native_starts=0, new_fits=0, optimizer_calls=0,
        new_nuisance_estimations=0, source_fits=0, free_rollouts=0, calibration_seeds_used=False,
        first_fit_hold_preserved=True, model_admitted=False, elapsed_sec=time.monotonic()-began,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'summary.json', result)
    print(json.dumps({k:result[k] for k in ('status', 'error', 'completed_case_attempts',
        'new_real_full_forward_calls', 'new_real_canonical_feature_evaluations', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.base, args.gate, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_FIXED_MIXTURE_CONFIRMATION' else 2)
