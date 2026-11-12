"""Partition the frozen endpoint count loss by whether a phase choice was possible.

At h=10 the physical indicator is constant within the stored context. Thus its
T/S decoded bits are exactly recoverable from each block-context prediction
counter; no histogram refit or new prediction rule is needed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from arrival_policy import ArrivalPolicy, fit_hours
from model_gate import load_training, release_id, rows
from residual_model import feature_context, label_category, update_history
from loss_context import context_label


def run(root, frozen, full_result, output):
    full = json.loads(full_result.read_text())
    old = json.loads((frozen/'development/result.json').read_text())
    cities = json.loads((frozen/'cohort.json').read_text())['cities']
    old_sources = {}
    for city, spec in cities.items():
        labels = load_training(root, 'donor', city, spec['donor'])
        old_sources[city] = fit_hours(labels), fit_hours(labels, permute=True)
    result = {}
    for city, donor in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
        cells = full['loss_context']['by_city'][city]['all_observable_risk']['tables']['block_context']
        lookup = {}
        for key, cell in cells.items():
            n = cell['counts'].get('flight_snapshots', 0)
            if not n:
                continue
            values = [cell['counts'][arm+'_predicted10'] for arm in ('T', 'S')]
            if any(value not in (0, n) for value in values):
                raise ValueError('The frozen endpoint decoder is not constant within its complete context')
            lookup[key] = tuple(int(value == n) for value in values)
        totals = defaultdict(lambda: defaultdict(Counter))
        seed_results = []
        original = {r['seed']: r for r in old['cities'][city]['by_seed']}
        for block, seeds in enumerate(cities[city]['adapt']):
            target = fit_hours(load_training(root, 'adapt', city, seeds))
            policy = ArrivalPolicy(target, *old_sources[donor])
            for seed in cities[city]['dev'][2*block:2*block+2]:
                directory = root/'dev'/city/str(seed)
                outcomes = {release_id(r): r for r in rows(directory/'labels.jsonl.gz')}
                history, counters = {}, defaultdict(Counter)
                for row in rows(directory/'decisions.jsonl.gz'):
                    observable = bool(row['flights'] and not row['spillback'] and not row['unmatched'])
                    common = observable and bool(policy.forecast(row))
                    enabled = dict(original_common_scored=common, all_observable_risk=observable)
                    per_edge = defaultdict(lambda: [0, 0, 0])
                    if observable:
                        for flight in row['flights']:
                            context, _ = feature_context(flight, row['t'], history)
                            t, s = lookup[str(block)+'::'+context_label(context)]
                            y = label_category(flight, outcomes[release_id(flight)], row['t']) != 0
                            counts = per_edge[flight['edge']]
                            counts[0] += t; counts[1] += s; counts[2] += y
                    update_history(row, history)
                    mode = 'locked' if len(row['states']) == 1 else 'unlocked'
                    for scope, active in enabled.items():
                        if not active:
                            continue
                        n = original[seed]['scopes'][scope]['counts']['scored_edge_epochs']
                        weight = 1/(6*n)
                        for t, s, y in per_edge.values():
                            lt, ls = abs(t-y), abs(s-y)
                            counters[scope]['N'] += 1
                            counters[scope]['T_error'] += lt
                            counters[scope]['S_error'] += ls
                            cell = totals[scope][mode]
                            cell['edge_epochs'] += 1
                            cell['T_error'] += lt; cell['S_error'] += ls
                            cell['weighted_T_mae_contribution'] += weight*lt
                            cell['weighted_S_mae_contribution'] += weight*ls
                            cell['weighted_mae_delta_contribution'] += weight*(ls-lt)
                            cell['improved_edge_epochs'] += ls < lt
                            cell['worsened_edge_epochs'] += ls > lt
                for scope, count in counters.items():
                    expected = original[seed]['scopes'][scope]
                    if (count['N'] != expected['counts']['scored_edge_epochs']
                            or any(abs(count[arm+'_error']/count['N']-expected['mae'][arm])>1e-12 for arm in ('T', 'S'))):
                        raise ValueError('Compressed frozen endpoint predictions did not reproduce the original result')
                seed_results.append(dict(seed=seed, block=block, reproduced=True))
        result[city] = dict(scopes={scope: {mode: dict(c) for mode,c in modes.items()} for scope,modes in totals.items()},
                            seed_checks=seed_results)
    output.write_text(json.dumps(dict(status='COMPLETE_EXACT_LOCK_PARTITION', by_city=result,
         definition='Weighted contributions use1/(6*N_seed_all_scored); locked plus unlocked reconstruct city MAE, not conditional-subgroup MAE.'), indent=2)+'\n')
    print('COMPLETE_EXACT_LOCK_PARTITION', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','frozen','full-result','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.root,a.frozen,a.full_result,a.output)
