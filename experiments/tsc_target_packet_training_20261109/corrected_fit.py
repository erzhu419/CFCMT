"""Rebuild frozen cached references and attempt one clock-corrected joint fit."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from .fit_model import HEAD_NAMES, objective, fit_joint
from .training_data import build_from_cached


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')


def check_zero_support(heads, record_references):
    """Inspect the original likelihood domain; never remove or rescue a row."""
    counts, failures = {}, []
    for head in HEAD_NAMES:
        row = heads[head]
        prior, risk, y = np.asarray(row['prior']), np.asarray(row['risk'], bool), np.asarray(row['y'])
        if len(record_references[head]) != len(y):
            raise ValueError('Every rebuilt row must retain its frozen reference')
        if head == 'motion':
            supported = np.ones(len(y), bool)  # Finite Gaussian, including upper0.
        elif head in ('start', 'yield_entry'):
            supported = (y == 0) | np.any((prior > 0) & risk, axis=1)
        else:
            mask = np.array(row['mask'], dtype=bool, copy=True)
            mask[~risk | (prior == 0)] = (False, True, False)
            selected = np.take_along_axis(mask,
                np.broadcast_to(y.astype(int)[:, None, None], (*prior.shape, 1)), axis=2)[:, :, 0]
            supported = np.any((prior > 0) & selected, axis=1)
        bad = np.flatnonzero(~supported)
        counts[head] = dict(rows=len(y), unsupported_rows=len(bad))
        if len(bad):
            index = int(bad[0])
            example = dict(record_references[head][index], head=head, array_row=index,
                           prior=prior[index].tolist(), risk=risk[index].tolist())
            if head == 'lane_change':
                example['mask'] = np.asarray(row['mask'])[index].tolist()
                example['effective_mask'] = mask[index].tolist()
            failures.append(example)
    result = dict(status='HOLD_ZERO_SUPPORT' if failures else 'PASS_ZERO_SUPPORT',
        unsupported_total=sum(row['unsupported_rows'] for row in counts.values()),
        by_head=counts, first_failures=failures)
    if not failures:
        try:
            result['initial_loss'] = objective(np.zeros(190), heads)[0]
        except Exception as error:
            result.update(status='HOLD_ZERO_SUPPORT', error=f'{type(error).__name__}: {error}')
    return result


def run(base, cache_base, workers=4):
    """Use old laws/reservoirs unchanged; all outputs belong to a fresh base."""
    base, cache_base = Path(base), Path(cache_base)
    if base.resolve() == cache_base.resolve():
        raise ValueError('Corrected outputs must not replace the failed cache base')
    outputs = ('fit_result.json', 'model.json', 'support_check_result.json',
        'training_arrays_server_only.npz', 'training_references_server_only.json',
        'training_sampling_result.json')
    if any((base/name).exists() for name in outputs):
        raise ValueError('Corrected output artifacts already exist; no retry or overwrite')
    protocol = json.loads((base/'protocol.json').read_text())
    if protocol != json.loads((cache_base/'protocol.json').read_text()):
        raise ValueError('The original frozen protocol must remain unchanged')
    amendment = json.loads((base/'clock_fix_protocol.json').read_text())
    training = json.loads((cache_base/'training/result.json').read_text())
    expected = protocol['cohorts']['training_seeds']
    if training['status'] != 'COMPLETE' or [row['seed'] for row in training['seeds']] != expected:
        raise ValueError('All original fixed training trajectories must be complete')
    old_fit = json.loads((cache_base/'fit_result.json').read_text())
    old_total = old_fit['total_fitting_operations']
    old_optimizer_calls = old_fit.get('optimization', {}).get('optimizer_calls', 0)
    base.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='REBUILDING_CACHED_FEATURES', training_seeds=expected,
        cache_base=str(cache_base), clock_fix_protocol=amendment, feature_workers=workers,
        new_native_starts=0, new_nuisance_estimation_attempts=0, new_joint_optimization_attempts=0,
        new_total_fitting_operations=0, cumulative_fitting_operations=old_total,
        optimizer_calls=0, cumulative_optimizer_calls=old_optimizer_calls,
        source_data_used=False, calibration_data_used=False,
        laws_reestimated=False, reservoir_resampled=False)
    save(base/'fit_result.json', receipt)
    begin = perf_counter()
    try:
        laws = json.loads((cache_base/'target_laws_server_only.json').read_text())
        reservoir = json.loads((cache_base/'training_reservoir_server_only.json').read_text())
        sampling = json.loads((cache_base/'training_sampling_result.json').read_text())
        history = json.loads((cache_base/'training_history_result.json').read_text())
        static = json.loads((cache_base/'static_graph_server_only.json').read_text())
        seed_dirs = {seed: cache_base/'training'/f'seed{seed}' for seed in expected}
        executor_seeds = json.loads((seed_dirs[expected[0]]/'executor_seed_server_only.json').read_text())
        data = build_from_cached(seed_dirs, protocol, static, executor_seeds,
                                 laws, reservoir, sampling, history, workers=workers)
        save(base/'training_sampling_result.json', data['sampling'])
        save(base/'training_references_server_only.json', data['record_references'])
        np.savez_compressed(base/'training_arrays_server_only.npz',
            **{head+'__'+key: value for head, arrays in data['heads'].items() for key, value in arrays.items()})
        support = check_zero_support(data['heads'], data['record_references'])
        save(base/'support_check_result.json', support)
        receipt.update(sampling=data['sampling'], support=support)
        if support['status'] != 'PASS_ZERO_SUPPORT':
            receipt.update(status='HOLD_ZERO_SUPPORT', wall_seconds=perf_counter()-begin)
            save(base/'support_failure_summary.json', support)
            save(base/'fit_result.json', receipt)
            return receipt
        receipt.update(status='FITTING', new_joint_optimization_attempts=1,
            new_total_fitting_operations=1, cumulative_fitting_operations=old_total+1)
        save(base/'fit_result.json', receipt)
        coefficients, optimization = fit_joint(data['heads'])
        receipt.update(optimization=optimization, optimizer_calls=optimization['optimizer_calls'],
            cumulative_optimizer_calls=old_optimizer_calls+optimization['optimizer_calls'],
            wall_seconds=perf_counter()-begin)
        if coefficients is None or not optimization['success']:
            receipt['status'] = 'HOLD_FIT'
        else:
            save(base/'model.json', dict(coefficients=coefficients, target_laws=data['laws'],
                training_seeds=expected, source_data_used=False, calibration_data_used=False,
                family=protocol['model']['family'], real_data_fitting_operations=old_total+1,
                cached_laws_source=str(cache_base/'target_laws_server_only.json')))
            receipt['status'] = 'PASS_FIT'
        save(base/'fit_result.json', receipt)
    except Exception as error:
        receipt.update(status='HOLD_CACHED_REBUILD', error=f'{type(error).__name__}: {error}',
                       wall_seconds=perf_counter()-begin)
        (base/'fit_error.txt').write_text(traceback.format_exc())
        save(base/'fit_result.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    result = run(args.base, args.cache_base, args.workers)
    print(json.dumps(dict(status=result['status'], new_joint_optimization_attempts=result['new_joint_optimization_attempts'],
        cumulative_fitting_operations=result['cumulative_fitting_operations'], optimizer_calls=result['optimizer_calls'])), flush=True)
    raise SystemExit(0 if result['status'] == 'PASS_FIT' else 2)
