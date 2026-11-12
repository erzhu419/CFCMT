"""Exploratory cache-only block readout; this is not a preregistered gate."""
import argparse
import gzip
import json
from math import isfinite
from pathlib import Path

import numpy as np

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_conditional_cache_20261110.verify import parameter_probes
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


DIMENSION = 5*len(FEATURE_NAMES)


def _number(value):
    return float(value) if value is not None and isfinite(value) else None


def _read(pool, theta, head, y):
    row = pool.marginal(theta, head, y)
    raw = row.get('gradient')
    gradient = None if raw is None else np.asarray(raw, dtype=float).reshape(-1)
    failure = ('NO_SURVIVOR_SAMPLES' if not row['defined'] else
               'ZERO_NUMERATOR' if not row.get('supported', False) else
               'NONFINITE_NLL' if _number(row.get('nll')) is None else
               'MISSING_GRADIENT' if gradient is None else
               'GRADIENT_DIMENSION_MISMATCH' if gradient.size != DIMENSION else
               'NONFINITE_GRADIENT' if not np.isfinite(gradient).all() else None)
    if failure is None and any(_number(row.get(key)) is None for key in
                               ('global_ess','survivor_ess','posterior_ess','max_normalized_weight')):
        failure = 'NONFINITE_WEIGHT_DIAGNOSTIC'
    norm = float(np.linalg.norm(gradient)) if gradient is not None and np.isfinite(gradient).all() else None
    compact = dict(defined=bool(row['defined']), supported=bool(row.get('supported', False)),
        failure=failure, nll=_number(row.get('nll')),
        nll_infinite=bool(row.get('nll') is not None and not isfinite(row['nll'])),
        gradient_dimension=None if gradient is None else int(gradient.size), gradient_norm_l2=norm,
        zero_signal=norm == 0. if norm is not None else None,
        **{key:_number(row.get(key)) for key in ('global_ess','survivor_ess','posterior_ess','max_normalized_weight')})
    return compact, gradient if failure is None else None


def compare_query(full_pool, block_pools, theta, observation):
    """All four blocks contribute, or comparison statistics remain undefined."""
    head, y = observation['head'], observation['y']
    full, full_g = _read(full_pool, theta, head, y)
    block_rows = [_read(pool, theta, head, y) for pool in block_pools]
    blocks = [row for row, unused in block_rows]
    usable = full_g is not None and all(g is not None for unused,g in block_rows)
    result = dict(head=head, y=y, manifest_ordinal=observation.get('manifest_ordinal'),
        full_1024=full, blocks_256=blocks, all_five_readouts_usable=usable,
        cosine_to_full=[None]*4, gradient_mean_norm_l2=None,
        gradient_mean_shift_l2=None, normalized_gradient_mean_shift=None,
        gradient_block_rms_dispersion_l2=None, normalized_gradient_block_rms_dispersion=None,
        mean_block_nll_minus_full=None, mean_block_nll_infinite=any(row['nll_infinite'] for row in blocks),
        full_zero_signal=full['zero_signal'], zero_signal_blocks=[i for i,row in enumerate(blocks) if row['zero_signal']])
    if not usable:
        return result
    gradients = np.asarray([g for unused,g in block_rows])
    mean = gradients.mean(axis=0)
    full_norm = full['gradient_norm_l2']
    shift = float(np.linalg.norm(mean-full_g))
    dispersion = float(np.sqrt(np.mean(np.sum((gradients-mean)**2,axis=1))))
    result.update(gradient_mean_norm_l2=float(np.linalg.norm(mean)),
        gradient_mean_shift_l2=shift, normalized_gradient_mean_shift=shift/(1.+full_norm),
        gradient_block_rms_dispersion_l2=dispersion,
        normalized_gradient_block_rms_dispersion=dispersion/(1.+full_norm),
        mean_block_nll_minus_full=float(np.mean([row['nll'] for row in blocks])-full['nll']),
        cosine_to_full=[float(np.dot(g,full_g)/(row['gradient_norm_l2']*full_norm))
                        if row['gradient_norm_l2'] > 0. and full_norm > 0. else None
                        for row,g in block_rows])
    return result


def run(evaluation, output):
    evaluation, output = Path(evaluation), Path(output)
    # A primary final receipt is the barrier: never postprocess an active run.
    primary = json.loads((evaluation/'summary.json').read_text())
    cases, records = [], []
    with physical_forbidden():
        for receipt in sorted(primary['cases'],key=lambda row:row['case_index']):
            ref, index = receipt['reference'], receipt['case_index']
            directory = evaluation/f"case{index:02d}_seed{int(ref['seed'])}_t{int(ref['time_sec'])}"
            path = directory/'particles_server_only.json.gz'
            case = dict(reference=ref,case_index=index,primary_status=receipt['status'],points={})
            cases.append(case)
            if not (directory/'summary.json').exists():
                case['status'] = 'NO_CASE_SUMMARY'
                continue
            receipt = json.loads((directory/'summary.json').read_text())
            if not path.exists():
                case['status'] = 'NO_COMPLETE_PARTICLE_POOL'
                continue
            with gzip.open(path,'rt') as stream:
                payload = json.load(stream)
            if len(payload['particles']) != 1024:
                case['status'] = 'INCOMPLETE_PARTICLE_POOL'
                continue
            full = ParticleLaw(payload)
            blocks = [ParticleLaw(dict(payload,particles=payload['particles'][i*256:(i+1)*256])) for i in range(4)]
            observed = receipt.get('readouts',{}).get('1024',{})
            missing = []
            for point,theta in enumerate(parameter_probes()):
                if str(point) not in observed:
                    missing.append(point)
                    continue
                rows = [compare_query(full,blocks,theta,obs) for obs in observed[str(point)]['observed_heads']]
                case['points'][str(point)] = rows
                records.extend(rows)
            case['missing_primary_points'] = missing
            case['status'] = ('READOUTS_RETAINED_WITH_FAILURES' if missing or
                any(not row['all_five_readouts_usable'] for rows in case['points'].values() for row in rows)
                else 'COMPLETE_EXPLORATORY_BLOCK_READOUT')
            del full, blocks, payload
    expected_cases = primary.get('requested_states',15)
    complete = len(cases)==expected_cases and all(
        case['status']=='COMPLETE_EXPLORATORY_BLOCK_READOUT' for case in cases)
    failures = [row for row in records if not row['all_five_readouts_usable']]
    aggregate = dict(case_receipts=len(cases),expected_primary_case_receipts=expected_cases,
        query_point_records=len(records),
        failed_query_point_records=len(failures),
        failed_full_readouts=sum(row['full_1024']['failure'] is not None for row in records),
        failed_block_readouts=sum(block['failure'] is not None for row in records for block in row['blocks_256']),
        full_zero_signal_records=sum(row['full_zero_signal'] is True for row in records),
        block_zero_signal_readouts=sum(len(row['zero_signal_blocks']) for row in records),
        all_records_available_and_usable=complete,
        maximum_normalized_gradient_mean_shift=None,
        maximum_normalized_gradient_block_rms_dispersion=None)
    if complete and records:
        aggregate.update(maximum_normalized_gradient_mean_shift=max(row['normalized_gradient_mean_shift'] for row in records),
            maximum_normalized_gradient_block_rms_dispersion=max(row['normalized_gradient_block_rms_dispersion'] for row in records))
    result = dict(status='COMPLETE_EXPLORATORY_BLOCK_READOUT',
        exploratory_secondary_not_preregistered_gate=True,primary_status=primary['status'],
        primary_result_unchanged=True,cases=cases,aggregate=aggregate,
        not_completed_primary_references=primary.get('not_completed_references',[]),
        definitions=dict(full_reference='Same1024-particle finite-SNIS estimate; not exact truth.',
            blocks='Four disjoint consecutive256-particle blocks, same source features/factors; each shares observations with the full1024 estimate.',
            cohort='Original9 and added6 are selected training diagnostic states, not independent validation; no significance inference from the block comparisons.',
            normalized_scale='1 + L2 norm of the full1024 gradient over all190 coordinates.',
            dispersion='sqrt(mean_i ||G_block_i - mean_block_gradient||_2^2).',
            zero_signal='Exactly zero computed L2 norm; no fitted near-zero threshold.',
            failure_statistics='No finite-subset replacement of failed full/block comparisons.'),
        new_particle_full_forward_calls=0,new_exact_full_forward_calls=0,new_fits=0,
        new_native_starts=0,parameter_probes=2,gradient_dimension=DIMENSION,
        scope='Posthoc sampling-noise and reweighting description only; no new threshold, coefficient choice or training admission.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],**aggregate)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.evaluation,args.output)
