"""Fixed cached-prefix factor accounting; no simulation or optimization."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from math import exp, isfinite, log
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import extended, strict_mean
from experiments.tsc_packet_focal_latent_decomposition_20261111.readout import (
    external_receipt_check, select_primary_references,
)
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


FAMILIES = ('brake_residual', 'start_permission', 'yield_permission', 'prefocal_lane_change')
ROLES = ('focal', 'other')
VIEWS = ('zero', 'full')+tuple('only_'+g for g in FAMILIES)+tuple('without_'+g for g in FAMILIES)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _maximum_error(a, b):
    return float(np.max(np.abs(a-b))) if len(a) else 0.


def factor_family_ratios(pool, candidate):
    """Group exactly the factors already recorded in the current prefix."""
    theta = np.asarray(candidate, dtype=float)
    if theta.shape != (5, len(FEATURE_NAMES)) or not np.isfinite(theta).all():
        raise ValueError('The unchanged finite five-by38 candidate is required')
    nf = len(pool.data['factors'])
    logs = np.empty(nf)
    names, roles = [], []
    for i, factor in enumerate(pool.data['factors']):
        phi = pool.features[factor['feature_id']]
        if factor['kind'] == 'bernoulli':
            block = int(factor['block'])
            if block not in (0, 1, 2):
                raise ValueError('Only the three recorded Bernoulli families are defined')
            family = FAMILIES[block]
            z = float(theta[block] @ phi)
            logs[i] = factor['bit']*z-np.logaddexp(0., z)+log(2.)
        elif factor['kind'] == 'categorical':
            family = FAMILIES[3]
            z = np.array([theta[3] @ phi, 0., theta[4] @ phi])
            allowed = np.asarray(factor['allowed'], dtype=int)
            logs[i] = z[factor['choice']]-logsumexp(z[allowed])+log(len(allowed))
        else:
            raise ValueError('Unknown recorded factor kind: '+str(factor['kind']))
        names.append(family)
        roles.append('focal' if factor['actor'] == pool.focal_id else 'other')
    if not np.isfinite(logs).all():
        raise ValueError('Nonfinite factor density ratios are retained as a failed readout')
    family_values, role_values, counts = {}, {}, {}
    incidences = np.asarray(pool.incidence.sum(axis=0)).ravel()
    for family in FAMILIES:
        mask = np.asarray([name == family for name in names], dtype=bool)
        family_values[family] = np.asarray(pool.incidence @ np.where(mask, logs, 0.)).ravel()
        role_values[family] = {}
        counts[family] = dict(stored_factor_columns=int(mask.sum()),
            active_factor_columns=int(np.count_nonzero(incidences[mask])),
            factor_incidences=int(incidences[mask].sum()), roles={})
        for role in ROLES:
            selected = mask & np.asarray([name == role for name in roles], dtype=bool)
            role_values[family][role] = np.asarray(pool.incidence @ np.where(selected, logs, 0.)).ravel()
            counts[family]['roles'][role] = dict(stored_factor_columns=int(selected.sum()),
                active_factor_columns=int(np.count_nonzero(incidences[selected])),
                factor_incidences=int(incidences[selected].sum()))
    total = sum(family_values.values(), np.zeros(len(pool.rows)))
    expected = pool._evaluate(theta)[0]
    total_error = _maximum_error(total, expected)
    role_errors = {g:_maximum_error(family_values[g],
        role_values[g]['focal']+role_values[g]['other']) for g in FAMILIES}
    checks = dict(maximum_full_log_ratio_error=total_error,
        maximum_family_role_errors=role_errors, tolerance=1e-10,
        passed=bool(total_error <= 1e-10 and all(v <= 1e-10 for v in role_errors.values())))
    return dict(family_log_ratios=family_values, role_log_ratios=role_values,
                reconstruction_checks=checks, factor_counts=counts)


def diagnostic_weight_arrays(family_log_ratios):
    if set(family_log_ratios) != set(FAMILIES):
        raise ValueError('All four families, including empty ones, must remain')
    total = sum(family_log_ratios.values(), np.zeros_like(family_log_ratios[FAMILIES[0]]))
    return dict(zero=np.zeros_like(total), full=total,
        **{'only_'+g:family_log_ratios[g] for g in FAMILIES},
        **{'without_'+g:total-family_log_ratios[g] for g in FAMILIES})


def _mask_matrix(pool):
    result = np.zeros((len(pool.rows), 3), dtype=bool)
    for i in np.flatnonzero(pool.survived):
        result[i, pool.data['focal_groups'][pool.focal_groups[i]]['allowed']] = True
    return result


def _ess(log_weights):
    if not len(log_weights):
        return 0.
    weights = np.exp(log_weights-logsumexp(log_weights))
    return float(1./np.sum(weights**2))


def weight_readout(pool, logw, native_label, zero_focal_table):
    logw = np.asarray(logw, dtype=float)
    if logw.shape != (len(pool.rows),) or not np.isfinite(logw).all():
        raise ValueError('Each diagnostic must retain all finite recorded prefix weights')
    survived = pool.survived
    mask = _mask_matrix(pool)[survived]
    group_ids = pool.focal_groups[survived]
    lglobal = float(logsumexp(logw))
    lsurvivor = float(logsumexp(logw[survived]))
    defined = bool(np.any(survived) and isfinite(lsurvivor))
    probabilities, masses = [], []
    log_probabilities = []
    for label in range(3):
        p = zero_focal_table[group_ids, label]
        positive = p > 0.
        lp = (float(logsumexp(logw[survived][positive]+np.log(p[positive]))-lsurvivor)
              if defined and np.any(positive) else -float('inf') if defined else None)
        lm = (float(logsumexp(logw[survived][mask[:, label]])-lsurvivor)
              if defined and np.any(mask[:, label]) else -float('inf') if defined else None)
        log_probabilities.append(extended(lp))
        probabilities.append(exp(lp) if lp is not None else None)
        masses.append(exp(lm) if lm is not None else None)
    native_nll = (extended(-log_probabilities[native_label]['value'])
        if log_probabilities[native_label]['state'] == 'finite' else
        extended(float('inf')) if log_probabilities[native_label]['state'] == 'negative_infinity'
        else extended(None))
    total = sum(probabilities) if defined else None
    normalization = dict(class_sum_within_tolerance=bool(defined and abs(total-1.) <= 1e-12),
        probabilities_within_mask_mass=bool(defined and all(
            -1e-12 <= p <= m+1e-12 and -1e-12 <= m <= 1.+1e-12
            for p,m in zip(probabilities, masses))))
    return dict(defined=defined, supported=native_nll['state'] == 'finite',
        probabilities=probabilities, log_probabilities=log_probabilities,
        native_probability=probabilities[native_label], native_nll=native_nll,
        mask_availability=masses, native_mask_availability=masses[native_label],
        global_ess=_ess(logw), survivor_ess=_ess(logw[survived]),
        max_normalized_weight=float(np.exp(np.max(logw)-lglobal)),
        max_survivor_normalized_weight=float(np.exp(np.max(logw[survived])-lsurvivor)) if defined else None,
        survival_probability=exp(lsurvivor-lglobal) if defined else 0.,
        survivor_samples=int(survived.sum()), normalization_checks=normalization)


def role_accounting(pool, role_logs, weight_arrays, native_label):
    """Conditional means of additive log ratios, not causal percentages."""
    survived = pool.survived
    allowed = _mask_matrix(pool)[:, native_label]
    populations = dict(all_survived=survived, native_allowed=survived & allowed,
                       native_disallowed=survived & ~allowed)
    result = {}
    for point in ('zero', 'full'):
        logw = weight_arrays[point]
        lsurvivor = float(logsumexp(logw[survived]))
        result[point] = {}
        for population, selected in populations.items():
            valid = bool(np.any(selected) and isfinite(lsurvivor))
            lz = float(logsumexp(logw[selected])) if valid else None
            weights = np.exp(logw[selected]-lz) if valid else None
            families = {}
            for family in FAMILIES:
                family_roles = {}
                for role in ROLES:
                    values = role_logs[family][role][selected]
                    family_roles[role] = dict(mean_log_ratio=float(weights @ values) if valid else None,
                        sampled_log_ratio_min=float(values.min()) if valid else None,
                        sampled_log_ratio_max=float(values.max()) if valid else None)
                families[family] = family_roles
            result[point][population] = dict(particle_count=int(selected.sum()),
                survivor_conditioned_mass=exp(lz-lsurvivor) if valid else 0. if isfinite(lsurvivor) else None,
                families=families)
    return result


def decompose_pool(pool, candidate, native_label, expected=None):
    if native_label not in (0, 1, 2) or not pool.data['eligibility']['lane_change']:
        raise ValueError('A frozen lane-change reference with its original label is required')
    grouped = factor_family_ratios(pool, candidate)
    if not grouped['reconstruction_checks']['passed']:
        raise ArithmeticError('Family/role log ratios do not reconstruct the original fitted prefix')
    arrays = diagnostic_weight_arrays(grouped['family_log_ratios'])
    zero_table = pool._evaluate(np.zeros_like(candidate))[2]
    views = {name:weight_readout(pool, arrays[name], native_label, zero_table) for name in VIEWS}
    endpoints = {}
    if expected is not None:
        for view, combo, point in (('zero', '00', 'zero'), ('full', 'f0', 'candidate')):
            check = external_receipt_check(views[view]['native_nll'],
                _extended_number(expected['combinations'][combo]['native_nll']))
            old_mass = expected['availability'][point]
            mass = views[view]['native_mask_availability']
            difference = abs(mass-old_mass) if mass is not None and old_mass is not None else None
            check.update(mask_availability_error=difference,
                mask_availability_matched=bool(difference is not None and difference <= 1e-10))
            endpoints[view] = check
    return dict(views=views, family_checks=grouped['reconstruction_checks'],
        family_factor_counts=grouped['factor_counts'], endpoint_checks=endpoints,
        role_contributions=role_accounting(pool, grouped['role_log_ratios'], arrays, native_label),
        particle_count=len(pool.rows))


def _extended_number(value):
    return (value['value'] if value['state'] == 'finite' else
            float('inf') if value['state'] == 'positive_infinity' else
            -float('inf') if value['state'] == 'negative_infinity' else None)


def _key(row):
    ref = row.get('reference', row)
    return (row['panel'], int(ref['seed']), int(row['case_index']),
            int(row['manifest_ordinal']), int(row['particle_size']))


def _same_reference(a, b):
    return all(a[k] == b[k] for k in ('seed', 'time_sec', 'packet_id'))


def select_references(particle_receipt, fit_receipt, pilot_receipt, decomposition_records):
    previous = decomposition_records['records']
    expected_counts = Counter({('train',256):96, ('development',256):32,
                               ('auxiliary',256):10, ('auxiliary',1024):10})
    if (len(previous) != 148 or len({_key(row) for row in previous}) != 148
            or Counter((r['panel'], r['particle_size']) for r in previous) != expected_counts):
        raise ValueError('All148 complete previous size-readout identities must remain')
    old = {_key(row):row for row in previous}
    primary = select_primary_references(particle_receipt, fit_receipt)
    jobs = [dict(row, particle_size=256) for row in primary]
    cases = sorted(pilot_receipt['cases'], key=lambda row:row['case_index'])
    if (pilot_receipt['status'] != 'PASS_BOUNDED_REAL_PARTICLE_PILOT'
            or [r['case_index'] for r in cases] != list(range(15))):
        raise ValueError('The original complete15-case pilot is required')
    auxiliary = []
    for case in cases:
        for obs in case['readouts']['1024']['0']['observed_heads']:
            if obs['head'] == 'lane_change':
                auxiliary.append(dict(reference=case['reference'], panel='auxiliary',
                    case_index=case['case_index'], head='lane_change',
                    manifest_ordinal=obs['manifest_ordinal'], y=int(obs['y'])))
    if len(auxiliary) != 10:
        raise ValueError('All10 original auxiliary lane-change references must remain')
    jobs += [dict(row, particle_size=n) for row in auxiliary for n in (256,1024)]
    if len(jobs) != 148 or {_key(row) for row in jobs} != set(old):
        raise ValueError('Current and preceding panel identities differ')
    for row in jobs:
        previous_row = old[_key(row)]
        if (previous_row['head'] != 'lane_change' or previous_row['y'] != row['y']
                or not _same_reference(previous_row['reference'], row['reference'])):
            raise ValueError('A frozen reference, label or ordinal changed')
        row['expected'] = previous_row
    train = {(r['reference']['seed'],r['reference']['time_sec'],r['reference']['packet_id'])
             for r in primary if r['panel'] == 'train'}
    overlap = train & {(r['reference']['seed'],r['reference']['time_sec'],r['reference']['packet_id']) for r in auxiliary}
    if overlap != {(76116903,1362,'jinan_3x4_real_1178')}:
        raise ValueError('The one predeclared training/diagnostic overlap must remain visible')
    return jobs


def _empty_view():
    return dict(defined=False, supported=False, native_nll=extended(None),
        probabilities=[None]*3, mask_availability=[None]*3, native_mask_availability=None,
        global_ess=None, survivor_ess=None, max_normalized_weight=None,
        max_survivor_normalized_weight=None, normalization_checks={})


def summarize_records(records, expected_count):
    complete = len(records) == expected_count
    views = {}
    for name in VIEWS:
        rows = [r['views'][name] for r in records]
        scalar = lambda key: strict_mean([extended(v[key]) for v in rows]) if complete else extended(None)
        views[name] = dict(native_nll_mean=strict_mean([r['native_nll'] for r in rows]) if complete else extended(None),
            native_mask_mass_mean=scalar('native_mask_availability'),
            mask_mass_means=[strict_mean([extended(r['mask_availability'][j]) for r in rows])
                             if complete else extended(None) for j in range(3)],
            global_ess_mean=scalar('global_ess'), survivor_ess_mean=scalar('survivor_ess'),
            maximum_global_weight=(max(r['max_normalized_weight'] for r in rows)
                if complete and rows and all(r['max_normalized_weight'] is not None for r in rows) else None))
    labels = {}
    for label in (0,1,2,'positive'):
        group = [r for r in records if (r['y'] != 1 if label == 'positive' else r['y'] == label)]
        labels[str(label)] = dict(n=len(group), views={name:dict(
            native_nll_mean=strict_mean([r['views'][name]['native_nll'] for r in group])
                if complete else extended(None),
            native_mask_mass_mean=strict_mean([extended(r['views'][name]['native_mask_availability']) for r in group])
                if complete else extended(None)) for name in VIEWS})
    return dict(reference_count=len(records), expected_count=expected_count,
        complete_reference_count=complete, views=views, by_label=labels)


def _read_payload(path, size):
    with gzip.open(path, 'rt') as stream:
        payload = json.load(stream)
    if len(payload['particles']) != size:
        raise ValueError('The saved particle count changed')
    return payload


def run(first_fit_root, pilot_root, decomposition_root, output):
    first_fit_root, pilot_root, decomposition_root, output = map(Path,
        (first_fit_root,pilot_root,decomposition_root,output))
    output.mkdir(parents=True,exist_ok=False)
    began, records, failure, fit = time.monotonic(), [], None, {}
    jobs = []
    try:
        with physical_forbidden():
            fit = json.loads((first_fit_root/'fit/summary.json').read_text())
            coefficients = json.loads((first_fit_root/'fit/coefficients_server_only.json').read_text())
            candidate = np.asarray(coefficients['coefficients'],dtype=float)
            if (fit['status'] != 'HOLD_FIRST_FIT_OPTIMIZER_TERMINATION' or fit['new_fits'] != 1
                    or fit['optimizer_invocations'] != 1 or fit['optimizer']['nit'] != 200
                    or coefficients['parameter_names'] != list(PARAMETERS)
                    or coefficients['feature_names'] != list(FEATURE_NAMES)
                    or candidate.shape != (5,len(FEATURE_NAMES)) or not np.isfinite(candidate).all()):
                raise ValueError('Only the unchanged returned first-fit candidate is permitted')
            preceding = json.loads((decomposition_root/'summary.json').read_text())
            if preceding['status'] != 'COMPLETE_FIXED_FOCAL_LATENT_DECOMPOSITION':
                raise ValueError('The completed four-grid decomposition is required')
            jobs = select_references(json.loads((first_fit_root/'particles/summary.json').read_text()),fit,
                json.loads((pilot_root/'summary.json').read_text()),
                json.loads((decomposition_root/'records_server_only.json').read_text()))
            for ref in jobs:
                row = {k:v for k,v in ref.items() if k not in ('expected','expected_nll','particle_path')}
                try:
                    if ref['panel'] == 'auxiliary':
                        r = ref['reference']
                        path = pilot_root/f"case{ref['case_index']:02d}_seed{int(r['seed'])}_t{int(r['time_sec'])}"/'particles_server_only.json.gz'
                        payload = _read_payload(path,1024)
                    else:
                        path = Path(ref['particle_path'])
                        payload = _read_payload(path if path.is_absolute() else first_fit_root/'particles'/path,256)
                    pool = ParticleLaw(dict(payload,particles=payload['particles'][:ref['particle_size']]))
                    row.update(decompose_pool(pool,candidate,ref['y'],ref['expected']))
                    del pool,payload
                except Exception as error:
                    row.update(error=f'{type(error).__name__}: {error}',views={v:_empty_view() for v in VIEWS},
                        family_checks={'passed':False},endpoint_checks={},role_contributions={},family_factor_counts={})
                records.append(row)
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    save(output/'records_server_only.json',dict(records=records))
    groups={}
    for panel,size,n in (('train',256,96),('development',256,32),('auxiliary',256,10),('auxiliary',1024,10)):
        groups[panel+'_'+str(size)]=summarize_records([r for r in records if r['panel']==panel and r['particle_size']==size],n)
    failed=[]
    for row in records:
        bad = (not row['family_checks'].get('passed',False)
            or any(not v['matched'] or not v['mask_availability_matched'] for v in row['endpoint_checks'].values())
            or any(not all(v['normalization_checks'].values()) for v in row['views'].values()))
        if row.get('error') or bad:
            failed.append(dict(reference=row['reference'],panel=row['panel'],particle_size=row['particle_size'],
                               error=row.get('error'),family_checks=row['family_checks'],endpoint_checks=row['endpoint_checks']))
    aux=[]
    for row in records:
        if row['panel']=='auxiliary':
            compact={k:v for k,v in row.items() if k!='role_contributions'}
            if row['case_index'] in (4,5,8):
                compact['role_contributions']=row['role_contributions']
            aux.append(compact)
    status=('HOLD_PREFIX_FAMILY_TECHNICAL' if failure else 'HOLD_PREFIX_FAMILY_MISMATCH'
            if failed or len(records)!=148 else 'COMPLETE_FIXED_PREFIX_FAMILY_DECOMPOSITION')
    result=dict(status=status,error=failure,original_fit_status=fit.get('status'),
        training_references=sum(r['panel']=='train' for r in jobs),
        development_references=sum(r['panel']=='development' for r in jobs),
        auxiliary_references=sum(r['panel']=='auxiliary' and r['particle_size']==256 for r in jobs),
        panel_references=len(jobs)-sum(r['panel']=='auxiliary' and r['particle_size']==1024 for r in jobs),
        size_readouts=len(records),failed_or_mismatched_readouts=failed,
        group_summaries=groups,auxiliary_details=aux,detailed_records_file='records_server_only.json',
        definitions=dict(families=list(FAMILIES),views=list(VIEWS),focal_table='Unchanged zero focal class probabilities for all ten diagnostic weight arrays.',
            roles='actor==focal_id versus other selected actors; no actor is removed or simulated again.',
            factor_counts='Stored table columns, active columns in this prefix, and actual incidence totals are separate.',
            role_means='Additive fitted-vs-zero log-ratio means, conditional on all survived/native allowed/native disallowed prefixes, under the named zero/full normalized weights.',
            factor_readouts='Ten explanatory arrays, not fitted candidates or policies; family interactions and survival normalization preclude unique causal percentages.',
            cohorts='138 panel references include1178 in both training and auxiliary panels; four trajectories7200 seconds entered priors; not independent validation.',
            retained_issues='Focal class suppression, the adverse motion diagnostic and first-fit iteration-limit HOLD remain unresolved.'),
        new_full_forward_calls=0,new_canonical_feature_evaluations=0,new_exact_reference_calls=0,
        new_native_starts=0,new_fits=0,optimizer_calls=0,new_nuisance_estimations=0,source_fits=0,
        free_rollouts=0,calibration_seeds_used=False,parameters_changed=False,new_candidate_selected=False,
        first_fit_hold_preserved=True,model_admitted=False,elapsed_sec=time.monotonic()-began,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('status','panel_references','size_readouts','new_full_forward_calls','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('first-fit-root','pilot-root','decomposition-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=run(args.first_fit_root,args.pilot_root,args.decomposition_root,args.output)
    raise SystemExit(2 if result['status'].startswith('HOLD_') else 0)
