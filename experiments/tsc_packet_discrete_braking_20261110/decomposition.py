"""Small saved-array decomposition after the complete exact-reference compare."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare import HEADS, SEEDS, TOL, _arrays, _motion_vectors


def _statistics(old, new, selected):
    a, b = old['zero_log_probability'], new['zero_log_probability']
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    common = selected & finite_a & finite_b
    rho_a, rho_b = (old['prior']*old['risk']).sum(axis=1), (new['prior']*new['risk']).sum(axis=1)
    return dict(records=int(selected.sum()), baseline_unsupported=int((selected & ~finite_a).sum()),
        candidate_unsupported=int((selected & ~finite_b).sum()),
        support_gained=int((selected & ~finite_a & finite_b).sum()),
        support_lost=int((selected & finite_a & ~finite_b).sum()),
        unsupported_both=int((selected & ~finite_a & ~finite_b).sum()),
        finite_both_records=int(common.sum()), finite_both_delta_nll_sum=float((a[common]-b[common]).sum()),
        baseline_rho_sum=float(rho_a[selected].sum()), candidate_rho_sum=float(rho_b[selected].sum()),
        rho_delta_sum=float((rho_b[selected]-rho_a[selected]).sum()))


def _groups(head, old, new):
    selected = {}
    if head == 'motion':
        fixed = _motion_vectors(old)['physical_deficit_mps'] > TOL
        selected = dict(all=np.ones(len(fixed), bool), old578=fixed, other39422=~fixed)
    elif head in ('start', 'yield_entry'):
        finite = np.isfinite(old['zero_log_probability']) & np.isfinite(new['zero_log_probability'])
        for label in (0, 1):
            rows = old['y'] == label
            selected['y'+str(label)+'_all'] = rows
            selected['y'+str(label)+'_finite_both'] = rows & finite
    else:
        risk_a = (old['risk'] & (old['prior'] > 0)).any(axis=1)
        risk_b = (new['risk'] & (new['prior'] > 0)).any(axis=1)
        changes = dict(risk_gained=~risk_a & risk_b, risk_lost=risk_a & ~risk_b,
                       risk_stable_present=risk_a & risk_b, risk_stable_absent=~risk_a & ~risk_b)
        for label, rows in (('stay1', old['y'] == 1), ('lane_change', old['y'] != 1)):
            selected[label+'_all'] = rows
            selected.update({label+'_'+name: rows & change for name, change in changes.items()})
    return {name: _statistics(old, new, rows) for name, rows in selected.items()}


def _finish(groups, motion=False):
    result = {}
    for name, original in groups.items():
        row = dict(original)
        count = row['records']
        total = row.pop('finite_both_delta_nll_sum')
        all_finite = count > 0 and row['finite_both_records'] == count
        row['delta_nll_sum'] = total if all_finite else None
        row['delta_nll_mean'] = total/count if all_finite else None
        a, b, delta = (row.pop(key) for key in ('baseline_rho_sum', 'candidate_rho_sum', 'rho_delta_sum'))
        if not motion:
            row.update(baseline_rho_mean=a/count if count else None,
                candidate_rho_mean=b/count if count else None, rho_delta_mean=delta/count if count else None)
        result[name] = row
    return result


def run(baseline, candidate, output):
    totals, by_seed = {head: {} for head in HEADS}, []
    for seed in SEEDS:
        groups = {}
        with np.load(baseline/f'seed{seed}_full_features_server_only.npz') as a, \
             np.load(candidate/f'seed{seed}_full_features_server_only.npz') as b:
            for head in HEADS:
                old, new = _arrays(a, head), _arrays(b, head)
                if not np.array_equal(old['y'], new['y']) or not np.array_equal(old['prior'], new['prior']):
                    raise ValueError('This post-comparison readout requires the established identical row order')
                groups[head] = _groups(head, old, new)
                for name, values in groups[head].items():
                    accumulated = totals[head].setdefault(name, {key: 0 for key in values})
                    for key, value in values.items():
                        accumulated[key] += value
        by_seed.append(dict(seed=seed, heads={head: _finish(groups[head], head == 'motion') for head in HEADS}))
    result = dict(status='COMPLETE_SAVED_ARRAY_LOSS_DECOMPOSITION',
        heads={head: _finish(groups, head == 'motion') for head, groups in totals.items()}, by_seed=by_seed,
        delta_definition='Candidate NLL minus baseline NLL; positive means worse. Each displayed NLL group must be finite on both versions for every selected row.',
        rho_definition='Sum of fixed positive/nonnegative prior masses on current-risk components; group means include all labels in that named group.',
        fixed_subset='Old physical-ceiling deficit>1e-8; old578 is never reselected using candidate values.',
        new_native_starts=0, new_fits=0, optimizer_calls=0,
        scope='Existing same-order audited arrays only. Finite-both groups are explicitly separate from full groups containing unsupported labels; no replacement objective or model is fitted.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline-output', 'candidate-output', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.baseline_output, args.candidate_output, args.output)
    print(json.dumps(dict(status=result['status'])), flush=True)
