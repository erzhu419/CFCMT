#!/usr/bin/env python3
"""Describe the closed response trial; no fitting, simulation or policy selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from context_fields import read_context

ARMS = ('target_only', 'source_target', 'source_placebo')


def ranges(values):
    return dict(zip(('min', 'median', 'max'), map(float, np.quantile(values, [0, .5, 1]))))


def subset_summary(rows, mask, city):
    chosen = [r for r, selected in zip(rows, mask) if selected]
    if not chosen:
        return {'cells': 0, 'mass': 0., 'groups': 0, 'seeds': 0}
    truth = np.array([r['truth'] for r in chosen])
    result = dict(cells=len(chosen), mass=len(chosen)/len(rows),
        groups=len({r['case_id'] for r in chosen}), seeds=len({r['seed'] for r in chosen}),
        mean_true_contrast=float(truth.mean()), mean_abs_true_contrast=float(np.abs(truth).mean()),
        nonzero_true_fraction=float(np.mean(truth != 0)))
    if city == 'jinan':
        errors = {a: np.array([r['predictions'][a] - r['truth'] for r in chosen]) for a in ARMS}
        result['arms'] = {a: dict(mae=float(np.abs(e).mean()), signed_error=float(e.mean()),
            mean_predicted_contrast=float((e+truth).mean()),
            mean_abs_predicted_contrast=float(np.abs(e+truth).mean())) for a, e in errors.items()}
        result['source_minus_comparator_contribution'] = {
            a: float((np.abs(errors['source_target'])-np.abs(errors[a])).sum()/len(rows))
            for a in ('target_only', 'source_placebo')}
    return result


def partition_summary(rows_by_city, field):
    categories = sorted({r[field] for rows in rows_by_city.values() for r in rows})
    return {category: {city: subset_summary(rows, [r[field] == category for r in rows], city)
        for city, rows in rows_by_city.items()} for category in categories}


def sign_name(number):
    return 'less' if number < -1e-9 else 'more' if number > 1e-9 else 'same'


def run(root):
    start = time.monotonic()
    spec = json.loads((root/'protocol.json').read_text())
    old = json.loads((root/'response_fit/result.json').read_text())
    assert old['status'] == 'HOLD_RESPONSE' and old['new_fits'] == 30
    oof = json.loads((root/'response_fit/oof_server_only.json').read_text())['records']
    preds = {(r['case_id'], r['arm']): np.array(r['predicted_counts']) for r in oof}
    names = spec['feature_names']; col = {name: i for i, name in enumerate(names)}
    rows_by_city = {'cologne': [], 'jinan': []}
    raw_moments = {a: [] for a in ARMS}
    input_rows = {'cologne': [], 'jinan': []}
    context_rows = {'cologne': [], 'jinan': []}
    per_case = []; bytes_read = 0
    for case in spec['cases']:
        directory = root/'collection'/f"seed_{case['seed']}"/case['case_id']
        data = json.loads((directory/'group_data_server_only.json').read_text())
        assert json.loads((directory/'result.json').read_text())['status'] == 'COMPLETE'
        X, Y = np.array(data['X']), np.array(data['Y']); ref = data['reference_index']
        context = read_context(data, directory); bytes_read += context['obs_bytes_read']
        nonref = np.arange(8) != ref; movements = X.shape[1]
        truth = (Y-Y[ref])[nonref, :, 0]
        input_rows[case['city']].append(X.reshape(-1, len(names)))
        cx = {key: float(np.mean(value)) for key, value in context.items() if key != 'obs_bytes_read'}
        cx['near_supply_fraction'] = float(np.sum(context['near_stopline_count'][ref]) /
            np.sum(X[ref,:,col['incoming_vehicle_count']])) if np.sum(X[ref,:,col['incoming_vehicle_count']]) else None
        context_rows[case['city']].append(cx)
        if case['city'] == 'jinan':
            for arm in ARMS:
                raw = preds[(case['case_id'], arm)][:,:,0]-Y[:,:,0]
                delta = raw[nonref]-raw[ref]
                raw_moments[arm].append(dict(raw_mae=float(np.abs(raw).mean()), raw_mse=float(np.square(raw).mean()),
                    candidate_mse=float(np.square(raw[nonref]).mean()), pp_mse=float(np.square(raw[ref]).mean()),
                    cross_product=float((raw[nonref]*raw[ref]).mean()), contrast_mse=float(np.square(delta).mean()),
                    contrast_mae=float(np.abs(delta).mean())))
            per_case.append({'case_id':case['case_id'], 'seed':case['seed'],
                'entering_mae':{a:raw_moments[a][-1]['contrast_mae'] for a in ARMS}})
        for action in np.flatnonzero(nonref):
            for movement in range(movements):
                f, p = X[action,movement], X[ref,movement]
                G = f[col['nominal_major_green_sec']] - p[col['nominal_major_green_sec']]
                permissive = sum(f[col[k]]-p[col[k]] for k in ('nominal_minor_green_sec','nominal_stop_go_sec','nominal_off_priority_sec'))
                row = dict(case_id=case['case_id'], seed=case['seed'], truth=float(Y[action,movement,0]-Y[ref,movement,0]),
                    signal_change='G_'+sign_name(G)+'__permissive_'+sign_name(permissive),
                    supply=('empty' if f[col['incoming_vehicle_count']] == 0 else
                            'vehicles_no_halted' if f[col['incoming_halted_count']] == 0 else 'halted_present'),
                    lane_allocation='dedicated' if np.isclose(f[col['movement_lane_equivalent']], f[col['incoming_lane_count']]) else 'shared',
                    clearance='empty' if f[col['clearance_vehicle_count']] == 0 else 'occupied',
                    schedule='mismatch' if context['schedule_mismatch'][action,movement] or context['schedule_mismatch'][ref,movement] else 'matched',
                    truth_sign=sign_name(Y[action,movement,0]-Y[ref,movement,0]))
                if case['city'] == 'jinan':
                    row['predictions'] = {a: float(preds[(case['case_id'],a)][action,movement,0]-preds[(case['case_id'],a)][ref,movement,0]) for a in ARMS}
                rows_by_city[case['city']].append(row)
    assert len(rows_by_city['cologne']) == 896 and len(rows_by_city['jinan']) == 1680
    overall = {c: subset_summary(rows, [True]*len(rows), c) for c,rows in rows_by_city.items()}
    partitions = {field:partition_summary(rows_by_city,field) for field in
                  ('signal_change','supply','lane_allocation','clearance','schedule')}
    for partition in partitions.values():
        for arm in ('target_only','source_placebo'):
            total = sum(cell['jinan'].get('source_minus_comparator_contribution',{}).get(arm,0.) for cell in partition.values())
            np.testing.assert_allclose(total, overall['jinan']['source_minus_comparator_contribution'][arm], atol=1e-12, rtol=0)
    for arm in ARMS:
        np.testing.assert_allclose(overall['jinan']['arms'][arm]['mae'],old['mean_output_contrast_mae'][arm]['stopline_entries_10s'],atol=1e-12,rtol=0)
    moments = {a:{k:float(np.mean([r[k] for r in raw_moments[a]])) for k in raw_moments[a][0]} for a in ARMS}
    for m in moments.values():
        np.testing.assert_allclose(m['contrast_mse'],m['candidate_mse']+m['pp_mse']-2*m['cross_product'],atol=1e-12,rtol=0)
    source, target = (np.concatenate(input_rows[c]) for c in ('cologne','jinan'))
    support = {name:{'source':ranges(source[:,i]),'target':ranges(target[:,i]),
                    'target_outside_source_range_fraction':float(np.mean((target[:,i]<source[:,i].min()-1e-9)|(target[:,i]>source[:,i].max()+1e-9)))} for i,name in enumerate(names)}
    context_summary = {city:{key:ranges([r[key] for r in values if r[key] is not None]) for key in values[0]} for city,values in context_rows.items()}
    return dict(protocol='closed-response-conditional-audit-v1',status='DESCRIPTIVE_ONLY',new_fits=0,new_simulations=0,
        original_gate=old['status'], weighting='Each city equal groups, seven nonPP actions and focal movements; two groups per seed. Target conditional contribution sums use all1680 cells as denominator.',
        scope='Post hoc conditional descriptions; no causal attribution, policy selection, new trained candidate or gate revision.',
        overall=overall, partitions=partitions, truth_conditioned_diagnostic=partition_summary(rows_by_city,'truth_sign'),
        raw_count_and_contrast_error_moments=moments, feature_marginal_support=support,
        context_group_ranges=context_summary, per_case_entering_error=per_case,
        observations_bytes_read_on_server=bytes_read, elapsed_sec=time.monotonic()-start,
        checks={'original_entering_mae_reproduced':True,'all_partition_contributions_add_up':True,'error_moment_identity':True})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=run(args.root)
    (args.output/'result.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','new_fits','new_simulations','checks','elapsed_sec')}),flush=True)
