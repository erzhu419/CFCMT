"""Zero-fit audit of horizon ranks and subsequent native PP divergence."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

HORIZONS = (10, 30, 60, 450)


def rank(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    return ((end-counts+end-1)/2.)[inverse]


def correlation(a, b):
    a, b = rank(a), rank(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare_sign(a, b):
    return dict(both_nonzero=bool(a and b),
                reversal=bool(a*b < 0), same_sign=bool(a*b > 0))


def run(base):
    data = json.loads((base/'training_dataset_server_only.json').read_text())
    result = json.loads((base/'nested_fit/result.json').read_text())
    source = {record['group_id']: record for record in result['records'] if record['arm'] == 'S'}
    rows, action_rows = [], []
    for group in data['groups']:
        if group['city'] != 'jinan':
            continue
        identity = f"jinan/seed{group['seed']}/anchor{group['anchor']}"
        record = source[identity]
        case = group['case']
        branches = sorted(group['branches'], key=lambda row: row['candidate_index'])
        futures = [json.loads((base/'measurement'/case['case_id']/branch['raw_directory']/
                               'future_execution_server_only.json').read_text()) for branch in branches]
        pp, selected = group['pp_index'], record['selected_action']
        lane_count = futures[pp]['controlled_lane_count']
        integral = {h: [future['integrals'][str(h)] for future in futures] for h in HORIZONS}
        delta = {h: [value-integral[h][pp] for value in integral[h]] for h in HORIZONS}
        selected_delta = {str(h): delta[h][selected] for h in HORIZONS}
        selected_segments = dict(first_10=selected_delta['10'],
            next_50=selected_delta['60']-selected_delta['10'],
            last_390=selected_delta['450']-selected_delta['60'])
        phases = dict(pp_request_differs=branches[pp]['inputs']['request_differs'],
                      selected_request_differs=branches[selected]['inputs']['request_differs'])
        choices = {}
        for horizon in HORIZONS:
            scores = integral[horizon]
            best = pp if scores[pp] == min(scores) else scores.index(min(scores))
            choices[str(horizon)] = dict(action=best, actual450_delta=delta[450][best]/(450*lane_count))
        for index, branch in enumerate(branches):
            if index == pp:
                continue
            post = futures[index]['execution_trace'][1:]
            pp_post = futures[pp]['execution_trace'][1:]
            differing_tls = [sum(step['requests'][tls] != reference['requests'][tls]
                                for tls in step['requests']) for step, reference in zip(post, pp_post)]
            focal_diff = sum(step['requests'][case['tls_id']] != reference['requests'][case['tls_id']]
                             for step, reference in zip(post, pp_post))
            action_rows.append(dict(group_id=identity, action=index,
                delta_integrals={str(h): delta[h][index] for h in HORIZONS},
                reversal10_450=compare_sign(delta[10][index], delta[450][index]),
                reversal60_450=compare_sign(delta[60][index], delta[450][index]),
                post10_pp_epochs_different_any_tls=sum(count > 0 for count in differing_tls),
                post10_pp_epochs_different_focal=focal_diff,
                post10_pp_different_tls_epoch_sum=sum(differing_tls)))
        rows.append(dict(group_id=identity, seed=group['seed'], anchor=group['anchor'], tls=case['tls_id'],
            lane_count=lane_count, pp=pp, selected=selected, **phases,
            selected_delta_integrals=selected_delta, selected_segments=selected_segments,
            selected_delta_cost450=delta[450][selected]/(450*lane_count),
            spearman_with450={str(h): correlation(integral[h], integral[450]) for h in (10, 30, 60)},
            label_min_choices=choices))
    selected = [row for row in rows if row['selected'] != row['pp']]
    def correlations(h):
        values = [row['spearman_with450'][str(h)] for row in rows]
        present = [value for value in values if value is not None]
        return dict(states=len(present), mean=mean(present), positive=sum(value > 0 for value in present),
                    negative=sum(value < 0 for value in present), constant_states=sum(value is None for value in values))
    output = dict(status='DESCRIPTIVE_ZERO_FIT', new_simulations=0, new_fits=0, states=len(rows),
        horizons_sec=list(HORIZONS), spearman={str(h): correlations(h) for h in (10, 30, 60)},
        nonpp_action_pairs=len(action_rows),
        reversals={label: dict(nonzero_pairs=sum(row[label]['both_nonzero'] for row in action_rows),
                    opposite_sign=sum(row[label]['reversal'] for row in action_rows))
                   for label in ('reversal10_450', 'reversal60_450')},
        selected_overrides=dict(count=len(selected),
            mean_delta_integrals={str(h): mean(row['selected_delta_integrals'][str(h)] for row in rows) for h in HORIZONS},
            mean_delta_cost_contributions={name: mean(row['selected_segments'][name]/(450*row['lane_count'])
                       for row in rows) for name in ('first_10', 'next_50', 'last_390')},
            holds=sum(not row['selected_request_differs'] for row in selected),
            switches=sum(row['selected_request_differs'] for row in selected),
            selected_sign_reversal10_450=sum(row['selected_delta_integrals']['10']*row['selected_delta_integrals']['450'] < 0
                                           for row in selected)),
        post10_pp_divergence=dict(
            mean_epochs_any_tls=mean(row['post10_pp_epochs_different_any_tls'] for row in action_rows),
            mean_epochs_focal=mean(row['post10_pp_epochs_different_focal'] for row in action_rows),
            mean_tls_epoch_count=mean(row['post10_pp_different_tls_epoch_sum'] for row in action_rows)),
        label_min_450_outcome={str(h):mean(row['label_min_choices'][str(h)]['actual450_delta'] for row in rows)
                               for h in HORIZONS},
        records=rows, action_records=action_rows,
        interpretation='Retrospective horizon/continuation description; sign reversal is not evidence of stochastic label variance. Early-horizon minima are not learned policies.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('records', 'action_records')}))
