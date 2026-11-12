"""One exact frozen-model comparison on the archived 904 matched pairs."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from experiments.tsc_packet_conditional_cache_20261110 import cache as exact
from experiments.tsc_packet_conditional_cache_20261110.verify import physical_evaluation_forbidden
from experiments.tsc_packet_joint_law_definition_20261110.law import OutsidePrototype
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import collect_receipts
from experiments.tsc_packet_real_particle_pilot_20261110.register import save
from experiments.tsc_target_packet_kernel_20261109.local_model import FrozenLocalModel

from .readout import aggregate_pairs, read_case
from .register import register_pairs


_kernels = None


def set_worker_kernels(kernels):
    global _kernels
    _kernels = kernels


@contextmanager
def exact_budget(task, counts, progress):
    """Charge the three production call sites before starting each operation."""
    def counted(original, name, ceiling):
        def call(*args, **kwargs):
            if counts[name] >= ceiling:
                raise ArithmeticError(f'{name} exceeded the frozen case budget of {ceiling}')
            counts[name] += 1
            if name == 'physical_calls' and counts[name] % 16 == 0:
                save(progress, counts)
            return original(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for owner, member, counter, ceiling in (
            (exact.MotionResolver, '__init__', 'physical_calls', task['physical_limit']),
            (FrozenLocalModel, 'features', 'canonical_feature_evaluations', task['feature_limit']),
            (exact, '_focal_outcome', 'leaf_visits', task['leaf_limit']),
        ):
            stack.enter_context(patch.object(owner, member, counted(getattr(owner, member), counter, ceiling)))
        yield


def inspect_case(task, anchor, output):
    directory = output / task['stem']
    directory.mkdir(exist_ok=False)
    save(directory / 'started.json', task)
    began = time.monotonic()
    counts = dict(physical_calls=0, canonical_feature_evaluations=0, leaf_visits=0)
    row = {k:task[k] for k in ('case_index', 'pair_index', 'role', 'direction', 'reference', 'time_sec')}
    try:
        kernel = _kernels[task['time_sec']]
        with exact_budget(task, counts, directory/'progress.json'):
            cache = exact.build_cache(kernel, task['reference']['packet_id'], max_stochastic=4)
        if (counts['physical_calls'] != cache.stats['physical_calls']
                or counts['leaf_visits'] != cache.stats['leaf_visits']
                or counts['canonical_feature_evaluations'] != len(cache.data['goal_configurations'])*task['required_actor_count']
                or cache.data['stochastic_participants'] != task['required_participants']):
            raise ArithmeticError('Exact cache construction disagrees with the registered accounting')
        with gzip.open(directory/'cache_server_only.json.gz', 'wt', compresslevel=1) as stream:
            json.dump(cache.data, stream, allow_nan=False)
        with physical_evaluation_forbidden():
            row['readouts'] = read_case(cache, anchor, direction=task['direction'], actual_label=task['y'])
        row['status'] = 'COMPLETE_EXACT_CASE'
    except OutsidePrototype as error:
        row.update(status='UNDEFINED_EXACT_BRANCH', error=f'{type(error).__name__}: {error}')
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
    except Exception as error:
        row.update(status='HOLD_TECHNICAL_CASE', error=f'{type(error).__name__}: {error}')
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
    row.update(**counts, elapsed_sec=time.monotonic()-began)
    save(directory/'summary.json', row)
    print(json.dumps({k:row[k] for k in ('status', 'case_index', 'physical_calls', 'leaf_visits', 'elapsed_sec')}),
          flush=True)
    return row


def pair_readouts(rows):
    pairs = {}
    for row in rows:
        i = row['pair_index']
        pair = pairs.setdefault(i, dict(pair_index=i, direction=row['direction']))
        pair[row['role']] = row
    return list(pairs.values())


def run(base, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    registered, completed, budget = [], [], {}
    error = None
    technical_stop = False
    try:
        proof = read_json(base/'target_packet_mixture_confirmation/synthetic_result.json')
        if proof['status'] != 'PASS_SYNTHETIC_MIXTURE_GATE' or proof['full_forward_calls'] != 854:
            raise ValueError('The existing exact/mixture equivalence gate is required')
        candidate = frozen_candidate(base)
        registered, kernels, budget = register_pairs(base, output)
        if (budget['status'] != 'PASS_FROZEN_904_PAIR_REGISTRATION'
                or len(registered) != 124 or budget['full_forward_ceiling'] > 2444
                or budget['leaf_ceiling'] > 31008):
            raise ValueError('The complete 904 exact budget must be registered before responses')
        print(json.dumps(budget), flush=True)
        tasks = iter(sorted(registered, key=lambda r:(-r['physical_limit']*r['frame_actor_count'], r['case_index'])))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                                 initializer=set_worker_kernels, initargs=(kernels,)) as executor:
            pending = {executor.submit(inspect_case, next(tasks), candidate, output) for _ in range(workers)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    completed.append(row)
                    technical_stop |= row['status'] == 'HOLD_TECHNICAL_CASE'
                if not technical_stop:
                    for _ in range(workers-len(pending)):
                        task = next(tasks, None)
                        if task is not None:
                            pending.add(executor.submit(inspect_case, task, candidate, output))
    except Exception as failure:
        error = f'{type(failure).__name__}: {failure}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    completed, unknown = collect_receipts(output, completed)
    physical = sum(r['physical_calls'] for r in completed) + sum(r['physical_calls_lower_bound'] for r in unknown)
    features = sum(r['canonical_feature_evaluations'] for r in completed) + sum(
        r['canonical_feature_evaluations_lower_bound'] for r in unknown)
    leaves = sum(r['leaf_visits'] for r in completed)
    success = (error is None and not unknown and len(completed) == 124
               and all(r['status'] == 'COMPLETE_EXACT_CASE' for r in completed))
    groups = None
    if success:
        try:
            groups = aggregate_pairs(pair_readouts(completed))
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
            (output/'error_server_only.txt').write_text(traceback.format_exc())
            success = False
    failures = [{k:r[k] for k in ('case_index', 'pair_index', 'role', 'status', 'error')}
                for r in completed if r['status'] != 'COMPLETE_EXACT_CASE']
    result = dict(status='COMPLETE_904_EXACT_PAIRED_HOLDOUT' if success else 'HOLD_904_EXACT_PAIRED_HOLDOUT',
        error=error, registered=budget, groups=groups, completed_case_attempts=len(completed),
        defined_cases=len(completed)-len(failures), failures=failures,
        incomplete_started_cases=unknown, stopped_after_technical_failure=technical_stop or error is not None,
        undispatched_case_indices=[r['case_index'] for r in registered
            if r['case_index'] not in {x['case_index'] for x in completed+unknown}],
        consumed_counters_exact=not unknown, full_forward_calls_lower_bound=physical,
        canonical_feature_evaluations_lower_bound=features, leaf_visits_lower_bound=leaves,
        new_full_forward_calls=physical if not unknown else None,
        new_canonical_feature_evaluations=features if not unknown else None,
        leaf_visits=leaves if not unknown else None, new_native_starts=0, new_fits=0,
        optimizer_calls=0, new_nuisance_estimations=0, source_fits=0, free_rollouts=0,
        calibration_seeds_used=False, first_fit_hold_preserved=True, model_admitted=False,
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('status', 'error', 'defined_cases', 'new_full_forward_calls',
        'new_canonical_feature_evaluations', 'leaf_visits', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', default=4, choices=(1, 2, 3, 4), type=int)
    args = parser.parse_args()
    result = run(args.base, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_904_EXACT_PAIRED_HOLDOUT' else 2)
