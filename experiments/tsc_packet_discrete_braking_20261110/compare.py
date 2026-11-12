"""Read-only exact-reference comparison of two completed theta0 audits."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.special import logsumexp


HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
SEEDS = (76116901, 76116902, 76116903, 76116904)
TOL = 1e-8


def _key(row):
    reference = row['reference']
    return row['head'], int(reference['seed']), float(reference['time_sec']), reference['packet_id']


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _load_references(path):
    groups = {head: [] for head in HEADS}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            groups[row['head']].append(row)
    return groups


def _align(old_rows, new_rows):
    """Return new indices in exact old-reference order; do not join on labels."""
    positions = {_key(row): index for index, row in enumerate(new_rows)}
    if len(positions) != len(new_rows) or len({_key(row) for row in old_rows}) != len(old_rows):
        raise ValueError('Duplicate head/seed/time/packet references cannot be paired')
    if set(positions) != {_key(row) for row in old_rows}:
        raise ValueError('The two complete original-reference sets differ')
    order = [positions[_key(row)] for row in old_rows]
    for before, index in zip(old_rows, order):
        after = new_rows[index]
        if before['reference']['y'] != after['reference']['y']:
            raise ValueError('A frozen label changed for an exact reference')
        if before['candidate_intents'] != after['candidate_intents'] or before['prior'] != after['prior']:
            raise ValueError('Frozen intent ordering or prior mass changed')
    return np.asarray(order, dtype=int)


def _arrays(archive, head, order=None):
    fields = ['prior', 'risk', 'y', 'retained', 'zero_log_probability']
    if head == 'motion':
        fields += ['mean0_mps', 'physical_ceiling_mps']
    elif head == 'lane_change':
        fields += ['mask']
    result = {field: archive[head+'_'+field] for field in fields}
    return result if order is None else {field: value[order] for field, value in result.items()}


def _log_probability(head, data):
    prior, risk, y = data['prior'], data['risk'], data['y']
    if head == 'motion':
        components = -.5*(y[:, None]-data['mean0_mps'])**2-.5*np.log(2*np.pi)
    elif head == 'lane_change':
        mask = data['mask'].copy()
        mask[~risk | (prior == 0)] = (False, True, False)
        allowed = np.take_along_axis(mask, np.broadcast_to(y.astype(int)[:, None, None],
            (*prior.shape, 1)), axis=2)[:, :, 0]
        components = np.where(allowed, -np.log(mask.sum(axis=2)), -np.inf)
    else:
        components = np.where(risk, -np.log(2.), np.where(y[:, None] == 0, 0., -np.inf))
    with np.errstate(divide='ignore'):
        return logsumexp(np.log(prior)+components, axis=1)


def _validate(head, data, rows):
    if len(data['y']) != len(rows) or not np.array_equal(data['y'], [row['reference']['y'] for row in rows]):
        raise ValueError('Saved arrays must match the per-head exact reference order and labels')
    if not np.array_equal(data['retained'], data['risk'].any(axis=1)):
        raise ValueError('Saved retained flags must reflect the current risk filter')
    computed = _log_probability(head, data)
    if not np.allclose(computed, data['zero_log_probability'], rtol=0, atol=TOL):
        raise ValueError('Saved theta0 log likelihood differs from its actual prior/risk/mask/mean')
    for index, row in enumerate(rows):
        k = len(row['prior'])
        if (not np.array_equal(data['prior'][index, :k], row['prior'])
                or not np.array_equal(data['risk'][index, :k], row['risk'])
                or bool(data['retained'][index]) != row['retained']
                or bool(np.isfinite(computed[index])) != row['label_supported_at_zero']):
            raise ValueError('Saved references and arrays disagree about prior/risk/retention/support')
        if head == 'lane_change' and not np.array_equal(data['mask'][index, :k], row['mask']):
            raise ValueError('Saved lateral class masks differ from their exact reference records')
        saved = row['zero_log_probability']
        if saved is not None and abs(saved-computed[index]) > TOL:
            raise ValueError('Saved reference log likelihood changed')
    return computed


def _summary(data, logp):
    supported, retained = np.isfinite(logp), data['retained']
    result = dict(records=len(logp), retained=int(retained.sum()), excluded=int((~retained).sum()),
        full_unsupported=int((~supported).sum()), retained_unsupported=int((retained & ~supported).sum()),
        excluded_unsupported=int((~retained & ~supported).sum()),
        full_finite_nll_sum=float(-logp[supported].sum()),
        retained_finite_nll_sum=float(-logp[retained & supported].sum()))
    for group, selected in (('full', np.ones(len(logp), dtype=bool)), ('retained', retained)):
        values = logp[selected]
        result[group+'_mean_nll'] = float(-values.mean()) if len(values) and np.isfinite(values).all() else None
    return result


def _transitions(old_support, new_support, selected):
    return dict(records=int(selected.sum()),
        gained=int((selected & ~old_support & new_support).sum()),
        lost=int((selected & old_support & ~new_support).sum()),
        supported_both=int((selected & old_support & new_support).sum()),
        unsupported_both=int((selected & ~old_support & ~new_support).sum()),
        new_unsupported_total=int((selected & ~new_support).sum()))


def _permission_transition(old, new, selected):
    baseline = (old['risk'] & (old['prior'] > 0)).any(axis=1)
    candidate = (new['risk'] & (new['prior'] > 0)).any(axis=1)
    return dict(records=int(selected.sum()), baseline_any=int((selected & baseline).sum()),
        candidate_any=int((selected & candidate).sum()),
        gained=int((selected & ~baseline & candidate).sum()),
        lost=int((selected & baseline & ~candidate).sum()),
        both_permitted=int((selected & baseline & candidate).sum()),
        both_not_permitted=int((selected & ~baseline & ~candidate).sum()))


def _head_comparison(head, old, new, old_logp, new_logp):
    old_support, new_support = np.isfinite(old_logp), np.isfinite(new_logp)
    result = dict(baseline=_summary(old, old_logp), candidate=_summary(new, new_logp),
        support_transition=_transitions(old_support, new_support, np.ones(len(old_logp), dtype=bool)),
        retention_transition=dict(gained=int((~old['retained'] & new['retained']).sum()),
                                 lost=int((old['retained'] & ~new['retained']).sum())))
    if head != 'motion':
        positive = old['y'] != 1 if head == 'lane_change' else old['y'] == 1
        result['risk_permission_transition_by_label'] = {str(int(label)): _permission_transition(old, new, old['y'] == label)
            for label in np.unique(old['y'])}
        result['support_by_label'] = {str(int(label)): _transitions(old_support, new_support, old['y'] == label)
                                     for label in np.unique(old['y'])}
        result['positive_labels'] = _transitions(old_support, new_support, positive)
        result['negative_labels'] = _transitions(old_support, new_support, ~positive)
    return result


def _feature_changes(old, new, order, prior, names):
    if old.shape != new.shape or old.shape != (*prior.shape, len(names)):
        raise ValueError('The frozen38 feature-array dimensions changed')
    new = new[order]
    difference = np.abs(new-old)
    changed = (new != old) & (prior > 0)[:, :, None]
    return [dict(name=name, changed_components=int(changed[:, :, index].sum()),
                 maximum_absolute_change=float(difference[:, :, index][changed[:, :, index]].max()))
            for index, name in enumerate(names) if changed[:, :, index].any()]


def _sum_counts(values):
    keys = dict.fromkeys(key for value in values for key in value)
    return {key: sum(value.get(key, 0) for value in values) for key in keys}


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, minimum=None, maximum=None, mean=None, p50=None, p90=None, p95=None, p99=None)
    return dict(count=len(values), minimum=float(values.min()), maximum=float(values.max()), mean=float(values.mean()),
                **{'p'+str(q): float(np.quantile(values, q/100)) for q in (50, 90, 95, 99)})


def _motion_vectors(data):
    positive = data['prior'] > 0
    max_mean = np.where(positive, data['mean0_mps'], -np.inf).max(axis=1)
    max_physical = np.where(positive, data['physical_ceiling_mps'], -np.inf).max(axis=1)
    return dict(max_mean0_mps=max_mean, max_physical_ceiling_mps=max_physical,
                mean_deficit_mps=data['y']-max_mean, physical_deficit_mps=data['y']-max_physical)


def _motion_summary(old, new):
    baseline, candidate = _motion_vectors(old), _motion_vectors(new)
    fixed = baseline['physical_deficit_mps'] > TOL
    report = {}
    for name, values in (('baseline', baseline), ('candidate', candidate)):
        report[name] = dict(observed_above_max_mean0=int((values['mean_deficit_mps'] > TOL).sum()),
            observed_above_max_physical_ceiling=int((values['physical_deficit_mps'] > TOL).sum()),
            all_physical_signed_deficits=_distribution(values['physical_deficit_mps']),
            all_mean_signed_deficits=_distribution(values['mean_deficit_mps']),
            positive_physical_deficits=_distribution(values['physical_deficit_mps'][values['physical_deficit_mps'] > TOL]))
    report['fixed_baseline_physical_deficit_subset'] = dict(count=int(fixed.sum()),
        baseline_physical_signed_deficits=_distribution(baseline['physical_deficit_mps'][fixed]),
        candidate_physical_signed_deficits=_distribution(candidate['physical_deficit_mps'][fixed]),
        candidate_mean_signed_deficits=_distribution(candidate['mean_deficit_mps'][fixed]),
        physical_recovered=int((fixed & (candidate['physical_deficit_mps'] <= TOL)).sum()),
        physical_still_above=int((fixed & (candidate['physical_deficit_mps'] > TOL)).sum()),
        candidate_mean_still_above=int((fixed & (candidate['mean_deficit_mps'] > TOL)).sum()),
        newly_above_physical_outside_fixed_subset=int((~fixed & (candidate['physical_deficit_mps'] > TOL)).sum()),
        newly_above_mean_outside_fixed_subset=int((~fixed & (candidate['mean_deficit_mps'] > TOL)).sum()))
    return report, baseline, candidate, fixed


def _row_readout(row):
    keys = ('reference', 'candidate_intents', 'prior', 'risk', 'retained', 'label_supported_at_zero',
            'zero_log_probability', 'mask', 'mean0_mps', 'physical_ceiling_mps')
    return {key: row[key] for key in keys if key in row}


def _merge_counts(values):
    totals = Counter()
    for value in values:
        totals.update({key: item for key, item in value.items() if isinstance(item, (int, float)) and item is not None
                       and not key.endswith('_mean_nll')})
    count = totals['records']
    totals['full_mean_nll'] = totals['full_finite_nll_sum']/count if count and not totals['full_unsupported'] else None
    count = totals['retained']
    totals['retained_mean_nll'] = totals['retained_finite_nll_sum']/count if count and not totals['retained_unsupported'] else None
    return dict(totals)


def run(baseline, candidate, output, mechanism_baseline):
    baseline, candidate, output, mechanism_baseline = map(Path, (baseline, candidate, output, mechanism_baseline))
    fixed_cases = json.loads(mechanism_baseline.read_text())['cases']
    fixed_keys = {_key(dict(head=row['head'], reference=row['reference'])): row for row in fixed_cases}
    if len(fixed_keys) != 15 or Counter(row['head'] for row in fixed_cases) != Counter(lane_change=9, yield_entry=3, motion=3):
        raise ValueError('The original fixed15 must be exactly9 lane+3 yield+3 registered motion cases')
    feature_names = json.loads((baseline/'result.json').read_text())['feature_names']
    if len(feature_names) != 38 or json.loads((candidate/'result.json').read_text())['feature_names'] != feature_names:
        raise ValueError('The two audit outputs must declare the same38 feature definitions')
    output.parent.mkdir(parents=True, exist_ok=True)
    by_seed, newly_unsupported, fixed_changes, all_motion, baseline_unsupported = [], [], [], [], []
    old_heads, new_heads, transition_heads = {head: [] for head in HEADS}, {head: [] for head in HEADS}, {head: [] for head in HEADS}
    with gzip.open(output.with_name(output.stem+'_new_unsupported_server_only.jsonl.gz'), 'xt', compresslevel=1) as new_zero, \
         gzip.open(output.with_name(output.stem+'_old578_server_only.jsonl.gz'), 'xt', compresslevel=1) as old578:
        for seed in SEEDS:
            before_rows = _load_references(baseline/f'seed{seed}_references_server_only.jsonl.gz')
            after_rows = _load_references(candidate/f'seed{seed}_references_server_only.jsonl.gz')
            seed_result = dict(seed=seed, heads={})
            with np.load(baseline/f'seed{seed}_full_features_server_only.npz') as before_arrays, \
                 np.load(candidate/f'seed{seed}_full_features_server_only.npz') as after_arrays:
                for head in HEADS:
                    rows = before_rows[head]
                    if len(rows) != 10000 or len(after_rows[head]) != 10000:
                        raise ValueError('All10000 original references per head/seed are required')
                    order = _align(rows, after_rows[head])
                    aligned = [after_rows[head][index] for index in order]
                    old, new = _arrays(before_arrays, head), _arrays(after_arrays, head, order)
                    old_logp, new_logp = _validate(head, old, rows), _validate(head, new, aligned)
                    if old['risk'].shape != new['risk'].shape:
                        raise ValueError('Frozen candidate risk-array dimensions changed')
                    if not np.array_equal(old['y'], new['y']) or not np.array_equal(old['prior'], new['prior']):
                        raise ValueError('Label/prior arrays must remain frozen after exact-reference alignment')
                    compared = _head_comparison(head, old, new, old_logp, new_logp)
                    compared['feature_changes'] = _feature_changes(before_arrays[head+'_X'], after_arrays[head+'_X'],
                        order, old['prior'], feature_names)
                    seed_result['heads'][head] = compared
                    old_heads[head].append(compared['baseline']); new_heads[head].append(compared['candidate'])
                    transition_heads[head].append(compared)
                    old_supported, new_supported = np.isfinite(old_logp), np.isfinite(new_logp)
                    for index, (before, after) in enumerate(zip(rows, aligned)):
                        key = _key(before)
                        if not old_supported[index]:
                            baseline_unsupported.append(dict(head=head, reference=before['reference']))
                        if old_supported[index] and not new_supported[index]:
                            record = dict(head=head, baseline=_row_readout(before), candidate=_row_readout(after))
                            new_zero.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                            if len(newly_unsupported) < 8:
                                newly_unsupported.append(record)
                        if key in fixed_keys:
                            registered = fixed_keys.pop(key)
                            if registered['reference']['y'] != before['reference']['y']:
                                raise ValueError('A registered fixed-case label changed')
                            baseline_row = _row_readout(before)
                            # The prior mechanism trace supplies P/mu for the
                            # original discrete cases, absent in old arrays.
                            components = registered['components']
                            baseline_row['mean0_mps'] = [component['mean0_mps'] for component in components]
                            baseline_row['physical_ceiling_mps'] = [component['P_mps'] for component in components]
                            if [component['intent'] for component in components] != before['candidate_intents']:
                                raise ValueError('Fixed mechanism components must preserve original intent order')
                            if 'mean0_mps' not in after or 'physical_ceiling_mps' not in after:
                                raise ValueError('New fixed-case records must save each head\'s actual mu0 and physical P')
                            fixed_changes.append(dict(case_id=registered['case_id'], head=head,
                                baseline=baseline_row, candidate=_row_readout(after)))
                    if head == 'motion':
                        metrics, old_values, new_values, fixed = _motion_summary(old, new)
                        seed_result['motion'] = metrics
                        all_motion.append((old, new))
                        for index in np.flatnonzero(fixed):
                            old578.write(json.dumps(dict(reference=rows[index]['reference'],
                                baseline={key: float(value[index]) for key, value in old_values.items()},
                                candidate={key: float(value[index]) for key, value in new_values.items()}),
                                allow_nan=False, separators=(',', ':'))+'\n')
            by_seed.append(seed_result)
            print(json.dumps(dict(stage='discrete_comparison_seed_complete', seed=seed)), flush=True)
    if fixed_keys or len(baseline_unsupported) != 12 or Counter(row['head'] for row in baseline_unsupported) != Counter(lane_change=9, yield_entry=3):
        raise ValueError('All original15 cases and exact12 discrete unsupported labels must reproduce')
    unsupported_keys = {_key(row) for row in baseline_unsupported}
    fixed_discrete_keys = {_key(dict(head=row['head'], reference=row['baseline']['reference'])) for row in fixed_changes if row['head'] != 'motion'}
    if unsupported_keys != fixed_discrete_keys:
        raise ValueError('The fixed12 must be precisely the original full-reservoir discrete zero labels')
    merged_motion = [{key: np.concatenate([pair[index][key] for pair in all_motion])
                      for key in all_motion[0][index]} for index in (0, 1)]
    motion, unused, unused_new, fixed = _motion_summary(*merged_motion)
    if int(fixed.sum()) != 578:
        raise ValueError('The original578 physical-deficit references must reproduce before comparison')
    heads = {}
    for head in HEADS:
        head_result = dict(baseline=_merge_counts(old_heads[head]), candidate=_merge_counts(new_heads[head]),
            support_transition=_sum_counts([row['support_transition'] for row in transition_heads[head]]),
            retention_transition=_sum_counts([row['retention_transition'] for row in transition_heads[head]]))
        if head != 'motion':
            for group in ('positive_labels', 'negative_labels'):
                head_result[group] = _sum_counts([row[group] for row in transition_heads[head]])
            labels = sorted(set().union(*(row['support_by_label'] for row in transition_heads[head])))
            head_result['support_by_label'] = {label: _sum_counts([row['support_by_label'].get(label, {})
                for row in transition_heads[head]]) for label in labels}
            head_result['risk_permission_transition_by_label'] = {label: _sum_counts([
                row['risk_permission_transition_by_label'].get(label, {}) for row in transition_heads[head]]) for label in labels}
        changed_columns = {}
        for row in transition_heads[head]:
            for column in row['feature_changes']:
                item = changed_columns.setdefault(column['name'], dict(name=column['name'], changed_components=0,
                                                                       maximum_absolute_change=0.))
                item['changed_components'] += column['changed_components']
                item['maximum_absolute_change'] = max(item['maximum_absolute_change'], column['maximum_absolute_change'])
        head_result['feature_changes'] = list(changed_columns.values())
        heads[head] = head_result
    objective = {}
    for version in ('baseline', 'candidate'):
        objective[version] = {}
        for group in ('full', 'retained'):
            values = [heads[head][version][group+'_mean_nll'] for head in HEADS]
            objective[version][group+'_sum_four_mean_nll'] = sum(values) if all(value is not None for value in values) else None
    result = dict(status='COMPLETE_PAIRED_DISCRETE_BRAKING_COMPARISON', exact_reference_pairs=160000,
        heads=heads, by_seed=by_seed, zero_objective=objective, motion=motion,
        original12_discrete_unsupported_changes=[row for row in fixed_changes if row['head'] != 'motion'],
        original9_lane_are_subset_of_original12=True,
        original3_registered_motion_changes=[row for row in fixed_changes if row['head'] == 'motion'],
        newly_unsupported_examples=newly_unsupported,
        newly_unsupported_full_records=output.stem+'_new_unsupported_server_only.jsonl.gz',
        fixed_old578_full_records=output.stem+'_old578_server_only.jsonl.gz',
        new_native_starts=0, new_fits=0, optimizer_calls=0, new_nuisance_estimations=0,
        retained_objective_definition='Retained NLL uses each version\'s own current risk-selected rows; cohort sizes can change, so its difference is not a paired same-cohort loss gain.',
        risk_permission_definition='Any positive-prior candidate with current risk true, stratified by native label; distinct from positive likelihood for that label. Lane risk means some lateral alternative exists, not that the observed side is allowed.',
        scope='Saved theta0 full-reservoir readouts, exact reference/label/prior joins. Normal sigma1 motion has finite likelihood beyond means or P; physical deficits are distinct from discrete zero likelihood. The fixed old578 set is unchanged. This is not a fitted model or a control result.')
    _write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('baseline-output', 'candidate-output', 'mechanism-baseline', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.baseline_output, args.candidate_output, args.output, args.mechanism_baseline)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name('comparison_error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_PAIRED_DISCRETE_BRAKING_COMPARISON', error=f'{type(error).__name__}: {error}',
                      new_native_starts=0, new_fits=0)
        _write(args.output, result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    raise SystemExit(0 if result['status'] == 'COMPLETE_PAIRED_DISCRETE_BRAKING_COMPARISON' else 2)
