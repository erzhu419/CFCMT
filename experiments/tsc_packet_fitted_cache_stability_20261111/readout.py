"""Fixed-coefficient cache diagnostics; no kernels, samplers or new particles."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from math import isfinite, isnan
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_packet_particle_first_fit_20261110.model import weight_maxima


HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
DIMENSION = 5*len(FEATURE_NAMES)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def extended(value):
    if value is None or isnan(float(value)):
        return dict(state='undefined', value=None)
    value = float(value)
    if isfinite(value):
        return dict(state='finite', value=value)
    return dict(state='positive_infinity' if value > 0 else 'negative_infinity', value=None)


def paired_difference(candidate, zero):
    c, z = candidate['state'], zero['state']
    if 'undefined' in (c, z) or (c == z and c != 'finite'):
        return extended(None)
    if c == z == 'finite':
        return extended(candidate['value']-zero['value'])
    if c == 'positive_infinity' or z == 'negative_infinity':
        return extended(float('inf'))
    return extended(-float('inf'))


def strict_sum(values):
    states = {value['state'] for value in values}
    if not values or 'undefined' in states or {'positive_infinity', 'negative_infinity'} <= states:
        return extended(None)
    if 'positive_infinity' in states:
        return extended(float('inf'))
    if 'negative_infinity' in states:
        return extended(-float('inf'))
    return extended(sum(value['value'] for value in values))


def strict_mean(values):
    result = strict_sum(values)
    return extended(result['value']/len(values)) if result['state'] == 'finite' else result


def slice_pool(payload, start, stop):
    return ParticleLaw(dict(payload, particles=payload['particles'][start:stop]))


def missing_readout(failure):
    return dict(defined=False, supported=False, failure=failure, loss=extended(None),
        nll=None, zero_signal=None, gradient_norm_l2=None, gradient_dimension=None,
        global_ess=None, survivor_ess=None, posterior_ess=None,
        max_normalized_weight=None, max_survivor_normalized_weight=None,
        max_posterior_normalized_weight=None)


def read_query(pool, theta, head, y):
    try:
        raw = pool.marginal(theta, head, y)
        gradient = raw.get('gradient')
        gradient = None if gradient is None else np.asarray(gradient, dtype=float).reshape(-1)
        finite_g = gradient is not None and gradient.size == DIMENSION and np.isfinite(gradient).all()
        loss = extended(raw.get('nll'))
        failure = ('NO_SURVIVOR_SAMPLES' if not raw['defined'] else
            'ZERO_NUMERATOR' if not raw.get('supported', False) else
            'NONFINITE_NLL' if loss['state'] != 'finite' else
            'MISSING_GRADIENT' if gradient is None else
            'GRADIENT_DIMENSION_MISMATCH' if gradient.size != DIMENSION else
            'NONFINITE_GRADIENT' if not finite_g else None)
        norm = float(np.linalg.norm(gradient)) if finite_g else None
        survivor_max, posterior_max = None, None
        if raw['defined'] and raw.get('supported', False) and loss['state'] == 'finite':
            survivor_max, posterior_max = weight_maxima(pool, theta, head, y)
        row = dict(defined=bool(raw['defined']), supported=bool(raw.get('supported', False)),
            failure=failure, loss=loss, nll=loss['value'],
            gradient_dimension=None if gradient is None else int(gradient.size),
            gradient_norm_l2=norm, zero_signal=norm == 0. if norm is not None else None,
            max_survivor_normalized_weight=survivor_max,
            max_posterior_normalized_weight=posterior_max,
            **{key:float(raw[key]) if raw.get(key) is not None and isfinite(raw[key]) else None
               for key in ('global_ess', 'survivor_ess', 'posterior_ess', 'max_normalized_weight')})
        if failure is None and any(row[key] is None for key in
                ('global_ess', 'survivor_ess', 'posterior_ess', 'max_normalized_weight',
                 'max_posterior_normalized_weight')):
            row['failure'] = 'NONFINITE_WEIGHT_DIAGNOSTIC'
        return row, gradient if finite_g and row['failure'] is None else None
    except Exception as error:
        return missing_readout(f'{type(error).__name__}: {error}'), None


def receipt_check(readout, expected_nll, tol=1e-10):
    finite = readout['loss']['state'] == 'finite' and isfinite(expected_nll)
    difference = abs(readout['loss']['value']-expected_nll) if finite else None
    return dict(matched=bool(finite and difference <= tol), expected_nll=float(expected_nll),
                absolute_error=difference, tolerance=tol)


def gradient_comparison(full, blocks):
    full_row, full_g = full
    usable = full_g is not None and all(g is not None for row, g in blocks)
    result = dict(all_gradients_usable=usable, full_zero_signal=full_row['zero_signal'],
        zero_signal_blocks=[i for i, (row, g) in enumerate(blocks) if row['zero_signal'] is True],
        mean_gradient_shift_l2=None, normalized_mean_gradient_shift=None,
        gradient_block_rms_dispersion_l2=None, normalized_gradient_block_rms_dispersion=None,
        cosine_to_full=[None]*len(blocks))
    if usable:
        gradients = np.asarray([g for row, g in blocks])
        mean = gradients.mean(axis=0)
        norm = float(np.linalg.norm(full_g))
        shift = float(np.linalg.norm(mean-full_g))
        dispersion = float(np.sqrt(np.mean(np.sum((gradients-mean)**2, axis=1))))
        result.update(mean_gradient_shift_l2=shift, normalized_mean_gradient_shift=shift/(1.+norm),
            gradient_block_rms_dispersion_l2=dispersion,
            normalized_gradient_block_rms_dispersion=dispersion/(1.+norm),
            cosine_to_full=[float(np.dot(g, full_g)/(np.linalg.norm(g)*norm))
                if norm > 0 and np.linalg.norm(g) > 0 else None for row, g in blocks])
    return result


def _view(record, point, view):
    values = record.get('points', {}).get(point, {})
    if view.startswith('block_'):
        index = int(view.split('_')[1])
        blocks = values.get('blocks_64', [])
        return blocks[index] if index < len(blocks) else missing_readout('MISSING_BLOCK')
    return values.get(view, missing_readout('MISSING_READOUT'))


def aggregate_records(records, view='full_256', expected_counts=None):
    expected_counts = dict.fromkeys(HEADS, 32) if expected_counts is None else dict(expected_counts)
    counts = Counter(record['head'] for record in records)
    complete = dict(counts) == expected_counts
    points, changes, labels = {}, {}, {}
    for point in ('zero', 'candidate'):
        means = {head:strict_mean([_view(row, point, view)['loss'] for row in records if row['head'] == head])
                 if counts[head] == n else extended(None) for head, n in expected_counts.items()}
        points[point] = dict(head_means=means, head_mean_sum=strict_sum(list(means.values())),
            failure_count=sum(_view(row, point, view)['failure'] is not None for row in records))
    for head, n in expected_counts.items():
        rows = [row for row in records if row['head'] == head]
        changes[head] = strict_mean([paired_difference(_view(row, 'candidate', view)['loss'],
            _view(row, 'zero', view)['loss']) for row in rows]) if len(rows) == n else extended(None)
        choices = ('zero', 'positive') if head == 'motion' else ('0', '1', '2') if head == 'lane_change' else ('0', '1')
        labels[head] = {}
        for label in choices:
            group = [row for row in rows if (('zero' if row['y'] == 0 else 'positive')
                     if head == 'motion' else str(int(row['y']))) == label]
            zero = strict_mean([_view(row, 'zero', view)['loss'] for row in group]) if len(rows) == n else extended(None)
            candidate = strict_mean([_view(row, 'candidate', view)['loss'] for row in group]) if len(rows) == n else extended(None)
            labels[head][label] = dict(n=len(group), zero=zero, candidate=candidate,
                                     paired_change=paired_difference(candidate, zero))
    total_change = strict_sum(list(changes.values())) if complete else extended(None)
    if not complete:
        for values in points.values():
            values['head_mean_sum'] = extended(None)
    return dict(reference_count=len(records), head_counts=dict(counts), expected_counts=expected_counts,
        complete_reference_counts=complete, view=view, points=points,
        paired_change=dict(head_means=changes, head_mean_sum=total_change), by_label=labels)


def _reference_id(row):
    return int(row['case_index']), row['head'], int(row['manifest_ordinal'])


def _load_payload(path, size):
    with gzip.open(path, 'rt') as stream:
        payload = json.load(stream)
    if len(payload['particles']) != size:
        raise ValueError(f'Frozen pool must contain exactly{size} particles')
    return payload


def primary_readout(first_root, candidate, fit, records):
    particle_root = first_root/'particles'
    receipt = json.loads((particle_root/'summary.json').read_text())
    if receipt['status'] != 'PASS_FROZEN_PARTICLE_TRAINING_POOLS':
        raise ValueError('The completed512-reference support receipt is required')
    selected = [(case, obs) for case in receipt['cases'] for obs in case['observations']
                if int(case['reference']['seed']) == 76116904]
    if len(selected) != 128 or Counter(obs['head'] for case, obs in selected) != Counter(dict.fromkeys(HEADS, 32)):
        raise ValueError('All128 original development references,32 per head, are required')
    expected = {point:{_reference_id(row):row['nll'] for row in fit[key]['development']['references']}
                for point, key in (('zero', 'zero'), ('candidate', 'final'))}
    identities = {_reference_id(dict(obs, case_index=case['case_index'])) for case, obs in selected}
    if len(identities) != 128 or any(set(values) != identities for values in expected.values()):
        raise ValueError('Development head/ordinal identities differ from the frozen fit receipt')
    grouped = defaultdict(list)
    for case, obs in selected:
        grouped[case['case_index']].append((case, obs))
    for index, linked in sorted(grouped.items()):
        case = linked[0][0]
        try:
            path = Path(case['particle_path'])
            payload = _load_payload(path if path.is_absolute() else particle_root/path, 256)
            full = slice_pool(payload, 0, 256)
            blocks = [slice_pool(payload, i*64, (i+1)*64) for i in range(4)]
            error = None
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        for case, observation in linked:
            row = dict(reference=case['reference'], case_index=index, head=observation['head'],
                y=observation['y'], manifest_ordinal=observation['manifest_ordinal'], points={}, receipt_checks={})
            records.append(row)
            for point, theta in (('zero', np.zeros_like(candidate)), ('candidate', candidate)):
                if error is None:
                    full_read = read_query(full, theta, row['head'], row['y'])
                    block_reads = [read_query(pool, theta, row['head'], row['y']) for pool in blocks]
                else:
                    full_read = missing_readout(error), None
                    block_reads = [(missing_readout(error), None) for i in range(4)]
                row['points'][point] = dict(full_256=full_read[0], blocks_64=[item[0] for item in block_reads],
                    gradient_comparison=gradient_comparison(full_read, block_reads))
                row['receipt_checks'][point] = receipt_check(full_read[0], expected[point][_reference_id(row)])
            row['paired_delta'] = dict(full_256=paired_difference(row['points']['candidate']['full_256']['loss'],
                row['points']['zero']['full_256']['loss']), blocks_64=[paired_difference(
                    row['points']['candidate']['blocks_64'][i]['loss'], row['points']['zero']['blocks_64'][i]['loss']) for i in range(4)])
        if error is None:
            del full, blocks, payload


def auxiliary_readout(pilot_root, candidate, records):
    receipt = json.loads((pilot_root/'summary.json').read_text())
    cases = sorted(receipt['cases'], key=lambda row:row['case_index'])
    if receipt['status'] != 'PASS_BOUNDED_REAL_PARTICLE_PILOT' or [row['case_index'] for row in cases] != list(range(15)):
        raise ValueError('The complete original ordered15-case pilot is required')
    expected_counts = Counter()
    for case in cases:
        index, ref = case['case_index'], case['reference']
        observations = case['readouts']['1024']['0']['observed_heads']
        if not observations:
            raise ValueError('Each fixed pilot case must retain its original observed heads')
        expected_counts.update(obs['head'] for obs in observations)
        try:
            directory = pilot_root/f"case{index:02d}_seed{int(ref['seed'])}_t{int(ref['time_sec'])}"
            payload = _load_payload(directory/'particles_server_only.json.gz', 1024)
            pools = {'256':slice_pool(payload, 0, 256), '1024':slice_pool(payload, 0, 1024)}
            error = None
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        for obs in observations:
            row = dict(reference=ref, case_index=index, head=obs['head'], y=obs['y'],
                manifest_ordinal=obs.get('manifest_ordinal'), points={})
            records.append(row)
            for point, theta in (('zero', np.zeros_like(candidate)), ('candidate', candidate)):
                row['points'][point] = {n:read_query(pool, theta, row['head'], row['y'])[0]
                                       for n, pool in pools.items()} if error is None else {
                                           n:missing_readout(error) for n in ('256', '1024')}
            row['paired_delta'] = {n:paired_difference(row['points']['candidate'][n]['loss'],
                row['points']['zero'][n]['loss']) for n in ('256', '1024')}
            row['size_sensitivity'] = {point:paired_difference(row['points'][point]['1024']['loss'],
                row['points'][point]['256']['loss']) for point in ('zero', 'candidate')}
            row['paired_change_sensitivity'] = paired_difference(row['paired_delta']['1024'], row['paired_delta']['256'])
        if error is None:
            del pools, payload
    return dict(expected_counts)


def _failures(records, sizes):
    failures = []
    for row in records:
        for point in ('zero', 'candidate'):
            for size in sizes:
                value = _view(row, point, size)
                if value['failure'] is not None:
                    failures.append(dict(case_index=row['case_index'], reference=row['reference'], head=row['head'],
                        y=row['y'], point=point, view=size, failure=value['failure'], loss=value['loss']))
    return failures


def sensitivity_summary(records):
    values = [row['paired_change_sensitivity'] for row in records]
    finite = bool(values) and all(value['state'] == 'finite' for value in values)
    ordered = sorted(records, key=lambda row:(row['paired_change_sensitivity']['state'] == 'finite',
        -abs(row['paired_change_sensitivity']['value']) if row['paired_change_sensitivity']['state'] == 'finite' else 0.))
    return dict(query_count=len(records), state_counts=dict(Counter(value['state'] for value in values)),
        mean_signed_paired_change_sensitivity=strict_mean(values),
        maximum_absolute_paired_change_sensitivity=max(abs(value['value']) for value in values) if finite else None,
        finite_sign_reversals=[dict(reference=row['reference'], head=row['head'], y=row['y'],
            paired_change_256=row['paired_delta']['256'], paired_change_1024=row['paired_delta']['1024'])
            for row in records if all(row['paired_delta'][n]['state'] == 'finite' for n in ('256', '1024'))
            and row['paired_delta']['256']['value']*row['paired_delta']['1024']['value'] < 0.],
        worst_five_retaining_nonfinite_first=[dict(reference=row['reference'], case_index=row['case_index'],
            head=row['head'], y=row['y'], paired_delta=row['paired_delta'],
            size_sensitivity=row['size_sensitivity'], paired_change_sensitivity=row['paired_change_sensitivity'])
            for row in ordered[:5]])


def run(first_fit_root, pilot_root, output):
    first_fit_root, pilot_root, output = map(Path, (first_fit_root, pilot_root, output))
    output.mkdir(parents=True, exist_ok=False)
    began, primary, auxiliary, expected_aux = time.monotonic(), [], [], {}
    failure, fit = None, {}
    try:
        with physical_forbidden():
            fit = json.loads((first_fit_root/'fit/summary.json').read_text())
            parameters = json.loads((first_fit_root/'fit/coefficients_server_only.json').read_text())
            candidate = np.asarray(parameters['coefficients'], dtype=float)
            if (fit['status'] != 'HOLD_FIRST_FIT_OPTIMIZER_TERMINATION' or fit['new_fits'] != 1
                    or fit['optimizer_invocations'] != 1 or fit['optimizer']['nit'] != 200
                    or parameters['parameter_names'] != list(PARAMETERS)
                    or parameters['feature_names'] != list(FEATURE_NAMES)
                    or candidate.shape != (5, len(FEATURE_NAMES)) or not np.isfinite(candidate).all()):
                raise ValueError('Only the exact finite returned200-iteration first-fit candidate is allowed')
            primary_readout(first_fit_root, candidate, fit, primary)
            expected_aux = auxiliary_readout(pilot_root, candidate, auxiliary)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    save(output/'records_server_only.json', dict(primary=primary, auxiliary=auxiliary))
    primary_scores = {view:aggregate_records(primary, view) for view in
                      ('full_256', 'block_0', 'block_1', 'block_2', 'block_3')}
    auxiliary_scores = {n:aggregate_records(auxiliary, n, expected_aux) for n in ('256', '1024')}
    mismatch = [dict(reference=row['reference'], head=row['head'], manifest_ordinal=row['manifest_ordinal'],
        point=point, **check) for row in primary for point, check in row['receipt_checks'].items() if not check['matched']]
    primary_failures = _failures(primary, ('full_256', 'block_0', 'block_1', 'block_2', 'block_3'))
    auxiliary_failures = _failures(auxiliary, ('256', '1024'))
    gradient_summary = {}
    for point in ('zero', 'candidate'):
        rows = [row['points'][point]['gradient_comparison'] for row in primary]
        complete = len(rows) == 128 and all(row['all_gradients_usable'] for row in rows)
        gradient_summary[point] = dict(all128_gradient_comparisons_usable=complete,
            unavailable_comparisons=sum(not row['all_gradients_usable'] for row in rows),
            full_zero_signal=sum(row['full_zero_signal'] is True for row in rows),
            maximum_normalized_mean_shift=max(row['normalized_mean_gradient_shift'] for row in rows) if complete else None,
            maximum_normalized_block_rms_dispersion=max(row['normalized_gradient_block_rms_dispersion'] for row in rows) if complete else None)
    status = ('HOLD_CACHE_READOUT_TECHNICAL' if failure else 'HOLD_CACHED_READOUT_MISMATCH' if mismatch else
              'COMPLETE_CACHE_DIAGNOSTIC_WITH_SAMPLING_FAILURES' if primary_failures or auxiliary_failures else
              'COMPLETE_FIXED_CANDIDATE_CACHE_DIAGNOSTIC')
    result = dict(status=status, error=failure, primary_reference_count=len(primary),
        auxiliary_reference_count=len(auxiliary), auxiliary_case_count=len({row['case_index'] for row in auxiliary}),
        primary_scores=primary_scores, auxiliary_scores=auxiliary_scores,
        auxiliary_size_sensitivity=sensitivity_summary(auxiliary),
        original_fit_status=fit.get('status'), original_optimizer=fit.get('optimizer'),
        primary_full256_receipt_mismatches=mismatch, primary_failures=primary_failures,
        auxiliary_failures=auxiliary_failures, gradient_summary=gradient_summary,
        detailed_records_file='records_server_only.json',
        definitions=dict(paired_change='Candidate minus zero marginal NLL; negative is improvement.',
            primary='All128 shared-history internal-development references, not independent validation.',
            auxiliary='Original15 selected training diagnostics; one also entered the coefficient training sample.',
            overlap='seed76116903/time1362/jinan_3x4_real_1178; retain it and do not call it held out.',
            block_reference='Four64 blocks are pieces of the same256 pool and share labels/current states.',
            particle_reference='Full256 and full1024 are finite-SNIS readouts, not exact truth.',
            infinity='All rows retained. Infinity minus itself is undefined; no finite-subset averages.',
            gradient_scale='1 + L2 norm of the containing full256 gradient over all190 coordinates.'),
        new_full_forward_calls=0, new_canonical_feature_evaluations=0, new_exact_reference_calls=0,
        new_native_starts=0, new_fits=0, optimizer_calls=0, new_nuisance_estimations=0,
        source_fits=0, free_rollouts=0, calibration_seeds_used=False,
        first_fit_hold_preserved=True, candidate_selected_or_scaled=False, model_admitted=False,
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'summary.json', result)
    print(json.dumps({key:result[key] for key in ('status', 'primary_reference_count', 'auxiliary_case_count',
          'new_full_forward_calls', 'new_fits', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-fit-root', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True, help='Completed pilot evaluation directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.first_fit_root, args.pilot_root, args.output)
    raise SystemExit(2 if result['status'].startswith('HOLD_') else 0)
