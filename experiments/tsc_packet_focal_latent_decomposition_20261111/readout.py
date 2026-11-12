"""Cross frozen latent weights and focal class tables using existing caches."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from math import exp, isfinite
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import (
    extended, paired_difference, strict_mean, strict_sum,
)
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


COMBINATIONS = ('00', '0f', 'f0', 'ff')
TRAIN_SEEDS = (76116901, 76116902, 76116903)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def external_receipt_check(loss, expected, tol=1e-10):
    target = extended(expected)
    error = (abs(loss['value']-target['value'])
             if loss['state'] == target['state'] == 'finite' else None)
    matched = ((error is not None and error <= tol) or
               (loss['state'] == target['state'] and loss['state'] != 'finite'))
    return dict(matched=bool(matched), expected=target, absolute_error=error, tolerance=tol)


def _negative(value):
    return paired_difference(extended(0.), value)


def nll_decomposition(combinations):
    v = {key:combinations[key]['native_nll'] for key in COMBINATIONS}
    focal_first = paired_difference(v['0f'], v['00'])
    latent_second = paired_difference(v['ff'], v['0f'])
    latent_first = paired_difference(v['f0'], v['00'])
    focal_second = paired_difference(v['ff'], v['f0'])
    total = paired_difference(v['ff'], v['00'])
    paths = dict(
        focal_then_latent=dict(focal=focal_first, latent=latent_second,
            sum=strict_sum([focal_first, latent_second])),
        latent_then_focal=dict(latent=latent_first, focal=focal_second,
            sum=strict_sum([latent_first, focal_second])))
    return dict(total=total, **paths,
        interaction=strict_sum([v['ff'], _negative(v['f0']), _negative(v['0f']), v['00']]))


def _score_readout(pool, candidate, logw, native_label):
    survived = pool.survived
    ids = pool.focal_groups[survived]
    features = pool.features[[pool.data['focal_groups'][i]['feature_id'] for i in ids]]
    allowed = np.asarray([native_label in pool.data['focal_groups'][i]['allowed'] for i in ids], dtype=bool)
    left_allowed = np.asarray([0 in pool.data['focal_groups'][i]['allowed'] for i in ids], dtype=bool)
    right_allowed = np.asarray([2 in pool.data['focal_groups'][i]['allowed'] for i in ids], dtype=bool)
    local_logw = logw[survived]
    summary, details = {}, {}
    for population, mask in (('all_survived', np.ones(len(ids), dtype=bool)),
                              ('native_allowed', allowed), ('left_allowed', left_allowed),
                              ('right_allowed', right_allowed)):
        valid = bool(np.any(mask) and np.isfinite(logsumexp(local_logw[mask])))
        mass = (exp(float(logsumexp(local_logw[mask])-logsumexp(local_logw)))
                if valid else 0. if len(ids) else None)
        summary[population] = dict(mass=mass, particle_count=int(mask.sum()))
        details[population] = dict(mass=mass, particle_count=int(mask.sum()))
        for direction, block in (('left', 3), ('right', 4)):
            if not valid:
                summary[population][direction] = dict(weighted_mean_logit=None,
                    sampled_logit_min=None, sampled_logit_max=None, top_five_contributions=[])
                details[population][direction] = dict(weighted_mean_contribution=None,
                    weighted_mean_absolute_contribution=None, weighted_mean_phi=None)
                continue
            w = np.exp(local_logw[mask]-logsumexp(local_logw[mask]))
            phi = features[mask]
            contribution = phi*candidate[block]
            mean_phi = w @ phi
            signed = w @ contribution
            absolute = w @ np.abs(contribution)
            logits = contribution.sum(axis=1)
            top = np.argsort(-np.abs(signed), kind='stable')[:5]
            summary[population][direction] = dict(weighted_mean_logit=float(w @ logits),
                sampled_logit_min=float(logits.min()), sampled_logit_max=float(logits.max()),
                top_five_contributions=[dict(feature=FEATURE_NAMES[j], feature_index=int(j),
                    weighted_mean_contribution=float(signed[j]),
                    weighted_mean_absolute_contribution=float(absolute[j]),
                    weighted_mean_phi=float(mean_phi[j])) for j in top])
            details[population][direction] = dict(weighted_mean_contribution=signed.tolist(),
                weighted_mean_absolute_contribution=absolute.tolist(), weighted_mean_phi=mean_phi.tolist())
    return summary, details


def decompose_pool(pool, candidate, native_label):
    """Only cross the two cached numeric parts; do not change global LC blocks."""
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (5, len(FEATURE_NAMES)) or not np.isfinite(candidate).all():
        raise ValueError('The unchanged finite5-by38 candidate is required')
    if native_label not in (0, 1, 2):
        raise ValueError('Native lane-change label must be left0/stay1/right2')
    if not pool.data['eligibility']['lane_change']:
        raise ValueError('The frozen reference must be lane-change eligible')
    zero = np.zeros_like(candidate)
    numeric = {}
    for point, theta in (('zero', zero), ('candidate', candidate)):
        logw, unused_scores, focal_prob, unused_focal_scores = pool._evaluate(theta)
        numeric[point] = (logw, focal_prob)
    survived, group_ids = pool.survived, pool.focal_groups[pool.survived]
    allowed = np.asarray([native_label in pool.data['focal_groups'][i]['allowed'] for i in group_ids], dtype=bool)
    combinations, availability, scores, feature_details = {}, {}, {}, {}
    for point in ('zero', 'candidate'):
        logw = numeric[point][0]
        valid = bool(np.any(survived) and np.isfinite(logsumexp(logw[survived])))
        availability[point] = (exp(float(logsumexp(logw[survived][allowed])-logsumexp(logw[survived])))
                               if valid and np.any(allowed) else 0. if valid else None)
        scores[point], feature_details[point] = _score_readout(pool, candidate, logw, native_label)
    for key, latent, focal in (('00', 'zero', 'zero'), ('0f', 'zero', 'candidate'),
                               ('f0', 'candidate', 'zero'), ('ff', 'candidate', 'candidate')):
        logw, table = numeric[latent][0][survived], numeric[focal][1]
        defined = bool(len(logw) and np.isfinite(logsumexp(logw)))
        probabilities, log_probabilities = [], []
        for label in range(3):
            p = table[group_ids, label]
            positive = p > 0.
            log_p = (float(logsumexp(logw[positive]+np.log(p[positive]))-logsumexp(logw))
                     if defined and np.any(positive) else -float('inf') if defined else None)
            log_probabilities.append(extended(log_p))
            probabilities.append(exp(log_p) if log_p is not None else None)
        native_log = log_probabilities[native_label]
        native_nll = _negative(native_log)
        total = sum(probabilities) if defined else None
        mask_mass = availability[latent]
        combinations[key] = dict(latent_point=latent, focal_point=focal, defined=defined,
            probabilities=probabilities, log_probabilities=log_probabilities,
            native_probability=probabilities[native_label], native_nll=native_nll,
            probability_sum=total, normalization_error=abs(total-1.) if defined else None,
            native_mask_availability=mask_mass,
            probability_given_native_allowed=(probabilities[native_label]/mask_mass
                if defined and mask_mass is not None and mask_mass > 0. else None))
    normalization = {key:dict(class_sum_within_tolerance=row['normalization_error'] is not None
            and row['normalization_error'] <= 1e-12,
        probability_within_mask_mass=(row['defined'] and row['native_mask_availability'] is not None
            and -1e-12 <= row['native_probability'] <= row['native_mask_availability']+1e-12
            and -1e-12 <= row['native_mask_availability'] <= 1.+1e-12))
        for key, row in combinations.items()}
    endpoints = {}
    for key, theta in (('00', zero), ('ff', candidate)):
        expected = pool.marginal(theta, 'lane_change', native_label)['nll']
        endpoints[key] = external_receipt_check(combinations[key]['native_nll'], expected)
    return dict(combinations=combinations, decomposition=nll_decomposition(combinations),
        availability=availability, score_summary=scores, feature_details=feature_details,
        normalization_checks=normalization, endpoint_checks=endpoints,
        particle_count=len(pool.rows), survivor_samples=int(survived.sum()))


def _identity(row):
    reference = row.get('reference', row)
    return int(reference['seed']), int(row['case_index']), int(row['manifest_ordinal'])


def _same_reference(a, b):
    return all(a[k] == b[k] for k in ('seed', 'time_sec', 'packet_id'))


def select_primary_references(particle_receipt, fit_receipt):
    if particle_receipt['status'] != 'PASS_FROZEN_PARTICLE_TRAINING_POOLS':
        raise ValueError('The complete frozen512-reference particle receipt is required')
    expected = {}
    for panel in ('train', 'development'):
        for point, key in (('zero', 'zero'), ('candidate', 'final')):
            rows = [r for r in fit_receipt[key][panel]['references'] if r['head'] == 'lane_change']
            if len(rows) != (96 if panel == 'train' else 32) or len({_identity(r) for r in rows}) != len(rows):
                raise ValueError('Original lane-change fit identities must remain complete and unique')
            expected[panel, point] = {_identity(r):r for r in rows}
    result = []
    for case in particle_receipt['cases']:
        ref = case['reference']
        panel = 'train' if int(ref['seed']) in TRAIN_SEEDS else 'development' if int(ref['seed']) == 76116904 else None
        if panel is None:
            raise ValueError('Only the original four training trajectories are allowed')
        for obs in case['observations']:
            if obs['head'] != 'lane_change':
                continue
            row = dict(reference=ref, panel=panel, case_index=case['case_index'], head='lane_change',
                manifest_ordinal=obs['manifest_ordinal'], y=int(obs['y']),
                particle_path=case['particle_path'], expected_nll={})
            for point in ('zero', 'candidate'):
                old = expected[panel, point].get(_identity(row))
                if (old is None or int(old['y']) != row['y'] or not _same_reference(old, ref)
                        or not isfinite(old['nll']) or obs.get('split', panel) != panel):
                    raise ValueError('A selected lane-change identity, label or old NLL changed')
                row['expected_nll'][point] = old['nll']
            result.append(row)
    if (len(result) != 128 or len({_identity(row) for row in result}) != 128
            or Counter((int(row['reference']['seed'])) for row in result)
                != Counter({seed:32 for seed in (*TRAIN_SEEDS, 76116904)})):
        raise ValueError('All128 original lane-change references must remain,32 per seed')
    return sorted(result, key=lambda row:(row['panel'], row['case_index'], row['manifest_ordinal']))


def select_auxiliary_references(pilot_receipt, stability_records):
    cases = sorted(pilot_receipt['cases'], key=lambda row:row['case_index'])
    if (pilot_receipt['status'] != 'PASS_BOUNDED_REAL_PARTICLE_PILOT'
            or [row['case_index'] for row in cases] != list(range(15))):
        raise ValueError('The complete original15-case pilot is required')
    old_rows = [row for row in stability_records['auxiliary'] if row['head'] == 'lane_change']
    if len(old_rows) != 10 or len({_identity(row) for row in old_rows}) != 10:
        raise ValueError('All10 auxiliary lane-change stability references are required')
    old = {_identity(row):row for row in old_rows}
    result = []
    for case in cases:
        for obs in case['readouts']['1024']['0']['observed_heads']:
            if obs['head'] != 'lane_change':
                continue
            row = dict(reference=case['reference'], panel='auxiliary', case_index=case['case_index'],
                head='lane_change', manifest_ordinal=obs['manifest_ordinal'], y=int(obs['y']))
            prior = old.get(_identity(row))
            if prior is None or prior['y'] != row['y'] or not _same_reference(prior['reference'], row['reference']):
                raise ValueError('An auxiliary identity or label changed')
            row['expected_nll'] = {n:{point:prior['points'][point][n]['nll']
                for point in ('zero', 'candidate')} for n in ('256', '1024')}
            result.append(row)
    if len(result) != 10 or {_identity(row) for row in result} != set(old):
        raise ValueError('Every original auxiliary lane-change reference must remain')
    return result


def _missing_result(error):
    combos = {key:dict(probabilities=[None]*3, native_nll=extended(None), native_probability=None,
        native_mask_availability=None, normalization_error=None, defined=False) for key in COMBINATIONS}
    return dict(error=error, combinations=combos, decomposition=nll_decomposition(combos),
        availability=dict(zero=None, candidate=None), score_summary={}, feature_details={},
        normalization_checks={}, endpoint_checks={}, particle_count=None, survivor_samples=None)


def summarize_records(records, expected_count):
    complete = len(records) == expected_count
    means = {key:strict_mean([r['combinations'][key]['native_nll'] for r in records])
             if complete else extended(None) for key in COMBINATIONS}
    decompositions = nll_decomposition({key:dict(native_nll=value) for key, value in means.items()})
    labels = {}
    for label in (0, 1, 2, 'positive'):
        group = [r for r in records if (r['y'] != 1 if label == 'positive' else r['y'] == label)]
        values = {key:strict_mean([r['combinations'][key]['native_nll'] for r in group])
                  if complete else extended(None) for key in COMBINATIONS}
        labels[str(label)] = dict(n=len(group), combination_native_nll_means=values,
            decomposition=nll_decomposition({key:dict(native_nll=value) for key, value in values.items()}))
    return dict(reference_count=len(records), expected_count=expected_count,
        complete_reference_count=complete, combination_native_nll_means=means,
        decomposition_means=decompositions, by_label=labels)


def _read_payload(path, size):
    with gzip.open(path, 'rt') as stream:
        payload = json.load(stream)
    if len(payload['particles']) != size:
        raise ValueError(f'The frozen pool must contain exactly{size} particles')
    return payload


def compact_auxiliary_record(record):
    """Keep all four probabilities and direction-conditional score evidence."""
    compact = {key:value for key,value in record.items() if key not in ('feature_details', 'score_summary')}
    compact['score_summary'] = {}
    for latent, populations in record['score_summary'].items():
        rows = {}
        for population, direction in (('left_allowed', 'left'), ('right_allowed', 'right')):
            item = populations[population]
            rows[population] = {key:item[key] for key in ('mass', 'particle_count', direction)}
        item = populations['all_survived']
        rows['all_survived'] = {key:item[key] for key in ('mass', 'particle_count')}
        for direction in ('left', 'right'):
            rows['all_survived'][direction] = {key:value for key,value in item[direction].items()
                if key != 'top_five_contributions'}
        rows['native_allowed'] = {key:populations['native_allowed'][key] for key in ('mass', 'particle_count')}
        compact['score_summary'][latent] = rows
    return compact


def run(first_fit_root, pilot_root, stability_root, output):
    first_fit_root, pilot_root, stability_root, output = map(Path,
        (first_fit_root, pilot_root, stability_root, output))
    output.mkdir(parents=True, exist_ok=False)
    began, records, failure = time.monotonic(), [], None
    primary, auxiliary, fit = [], [], {}
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
                raise ValueError('Only the exact finite first-fit candidate is allowed')
            primary = select_primary_references(json.loads((first_fit_root/'particles/summary.json').read_text()), fit)
            stability = json.loads((stability_root/'summary.json').read_text())
            if stability['status'] != 'COMPLETE_FIXED_CANDIDATE_CACHE_DIAGNOSTIC':
                raise ValueError('The preceding complete fixed-candidate stability readout is required')
            auxiliary = select_auxiliary_references(json.loads((pilot_root/'summary.json').read_text()),
                json.loads((stability_root/'records_server_only.json').read_text()))
            jobs = [(row, 256) for row in primary]+[(row, n) for row in auxiliary for n in (256, 1024)]
            for ref, size in jobs:
                row = {key:value for key,value in ref.items() if key not in ('particle_path', 'expected_nll')}
                row['particle_size'] = size
                try:
                    if ref['panel'] == 'auxiliary':
                        r = ref['reference']
                        path = pilot_root/f"case{ref['case_index']:02d}_seed{int(r['seed'])}_t{int(r['time_sec'])}"/'particles_server_only.json.gz'
                        payload = _read_payload(path, 1024)
                    else:
                        path = Path(ref['particle_path'])
                        payload = _read_payload(path if path.is_absolute() else first_fit_root/'particles'/path, 256)
                    pool = ParticleLaw(dict(payload, particles=payload['particles'][:size]))
                    row.update(decompose_pool(pool, candidate, ref['y']))
                    del pool, payload
                except Exception as error:
                    row.update(_missing_result(f'{type(error).__name__}: {error}'))
                expected = ref['expected_nll'][str(size)] if ref['panel'] == 'auxiliary' else ref['expected_nll']
                row['receipt_checks'] = {key:external_receipt_check(row['combinations'][key]['native_nll'], expected[point])
                    for key, point in (('00', 'zero'), ('ff', 'candidate'))}
                records.append(row)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    save(output/'records_server_only.json', dict(records=records))
    groups = {}
    for panel, size, n in (('train', 256, 96), ('development', 256, 32),
                            ('auxiliary', 256, 10), ('auxiliary', 1024, 10)):
        group = [r for r in records if r['panel'] == panel and r['particle_size'] == size]
        groups[panel+'_'+str(size)] = summarize_records(group, n)
    checks = []
    for row in records:
        bad = ([key for key, check in row['receipt_checks'].items() if not check['matched']]
            +[key for key, check in row['endpoint_checks'].items() if not check['matched']]
            +[key for key, check in row['normalization_checks'].items() if not all(check.values())])
        if row.get('error') or bad:
            checks.append(dict(reference=row['reference'], panel=row['panel'], particle_size=row['particle_size'],
                error=row.get('error'), failed_checks=bad))
    compact_aux = [compact_auxiliary_record(r) for r in records if r['panel'] == 'auxiliary']
    status = ('HOLD_FOCAL_LATENT_DECOMPOSITION_TECHNICAL' if failure else
        'HOLD_FOCAL_LATENT_DECOMPOSITION_MISMATCH' if checks or len(records) != 148 else
        'COMPLETE_FIXED_FOCAL_LATENT_DECOMPOSITION')
    result = dict(status=status, error=failure, original_fit_status=fit.get('status'),
        training_references=len([r for r in primary if r['panel'] == 'train']),
        development_references=len([r for r in primary if r['panel'] == 'development']),
        auxiliary_references=len(auxiliary), panel_references=len(primary)+len(auxiliary),
        size_readouts=len(records), failed_or_mismatched_readouts=checks,
        group_summaries=groups, auxiliary_details=compact_aux,
        detailed_records_file='records_server_only.json',
        definitions=dict(combination='First character selects latent-prefix weights; second selects focal class table.0=zero,f=fixed fitted candidate.',
            class_labels='0=left lane change,1=stay,2=right lane change; not route turns.',
            weights='Normalize each latent point over the same survived prefix particles before crossing the focal table.',
            availability='Survivor-conditioned latent mass whose stored focal mask permits the native class.',
            all_survived_scores='Candidate score relative to stay, averaged over every survived cached prefix under the stated latent weights, including masks forbidding that class.',
            native_allowed_scores='Candidate score/contributions conditional on the stored mask permitting the native class; denominator is the corresponding latent mask mass.',
            direction_allowed_scores='left_allowed and right_allowed condition on the respective direction mask, including for stay-labelled examples; only these conditional scores are executable-direction evidence.',
            feature_contributions='Full38 signed/absolute contribution means remain server-side; top five ranked by absolute signed weighted mean are descriptive only.',
            paths='Two telescoping paths and their interaction; order dependent, not a unique causal percentage.',
            cohorts='Four trajectories7200s entered priors; no independent validation.1178 appears in training and diagnostics and is retained.',
            motion_failure='The adverse single auxiliary motion case is unresolved; this lane-change diagnostic does not remove it.'),
        new_full_forward_calls=0, new_canonical_feature_evaluations=0, new_exact_reference_calls=0,
        new_native_starts=0, new_fits=0, optimizer_calls=0, new_nuisance_estimations=0,
        source_fits=0, free_rollouts=0, calibration_seeds_used=False,
        parameters_changed=False, new_candidate_selected=False, first_fit_hold_preserved=True, model_admitted=False,
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'summary.json', result)
    print(json.dumps({key:result[key] for key in ('status', 'panel_references', 'size_readouts',
        'new_full_forward_calls', 'new_fits', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('first-fit-root', 'pilot-root', 'stability-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.first_fit_root, args.pilot_root, args.stability_root, args.output)
    raise SystemExit(2 if result['status'].startswith('HOLD_') else 0)
