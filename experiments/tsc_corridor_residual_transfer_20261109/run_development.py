"""Fit one frozen calibration candidate and evaluate the existing development panel."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / 'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0, str(path))
from arrival_policy import ArrivalPolicy, fit_hours as old_fit
from model_gate import hour, load_training, verify_protocol
from residual_model import ResidualPredictor, fit_hours, fit_source, PSEUDOCOUNT, PERMUTATION_SEED
from evaluate import evaluate_hour, summarize


def run(root, output):
    plan = json.loads((HERE / 'protocol.json').read_text())
    cities = json.loads((HERE / 'cohort.json').read_text())['cities']
    expected = json.loads((HERE / 'expected_baselines.json').read_text())
    verify_protocol(cities)
    if (PSEUDOCOUNT != plan['physical_pseudocount'] or PERMUTATION_SEED != plan['permutation_seed']):
        raise ValueError('Model constants do not match the frozen protocol')
    output.mkdir(parents=True, exist_ok=True)
    sources, old_sources, metadata = {}, {}, dict(source={}, target={})
    for city in cities:
        seeds = cities[city]['donor']
        sources[city] = fit_source(root, 'donor', city, seeds)
        old_labels = load_training(root, 'donor', city, seeds)
        old_sources[city] = old_fit(old_labels), old_fit(old_labels, permute=True)
        metadata['source'][city] = dict(real=sources[city][0].metadata, placebo=sources[city][1].metadata)
        print(f'Fitted source/{city}: {sources[city][0].metadata["snapshot_examples"]} flight snapshots', flush=True)
    summaries = {}
    checks = []
    with gzip.open(output / 'request_disagreements_server_only.jsonl.gz', 'wt') as stream:
        for city, donor_city in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
            panel = []
            metadata['target'][city] = []
            for block, seeds in enumerate(cities[city]['adapt']):
                target = fit_hours(root, 'adapt', city, seeds)
                target_old = old_fit(load_training(root, 'adapt', city, seeds))
                old_policy = ArrivalPolicy(target_old, *old_sources[donor_city])
                metadata['target'][city].append(dict(block=block, **target.metadata))
                print(f'Fitted target/{city}/block{block}: {target.metadata["snapshot_examples"]} snapshots', flush=True)
                for seed in cities[city]['dev'][2*block:2*block+2]:
                    predictor = ResidualPredictor(target, *sources[donor_city])
                    result = evaluate_hour(city, seed, block, hour(root, 'dev', city, seed), predictor, old_policy, stream)
                    actual = result['scopes']['original_common_scored']
                    baseline = expected[f'{city}/{seed}']
                    if (actual['counts']['scored_edge_epochs'] != baseline['edge_epochs']
                            or abs(actual['mae']['analytic'] - baseline['analytic_count_mae']) > 1e-12
                            or abs(actual['mae']['history'] - baseline['history_count_mae']) > 1e-12):
                        raise ValueError(f'Original baselines or denominators not reproduced for {city}/{seed}')
                    result['prediction_support'] = dict(predictor.counts)
                    panel.append(result)
                    checks.append(dict(city=city, seed=seed, baseline_reproduced=True))
                    print(f'Evaluated {city}/{seed}: {json.dumps(actual["mae"])}', flush=True)
            summaries[city] = summarize(panel)
    passed = all(s['scopes']['original_common_scored']['passed'] for s in summaries.values())
    receipt = dict(status='DEVELOPMENT_SIGNAL_ONLY' if passed else 'HOLD_RESIDUAL_DEVELOPMENT',
                   original_study_status='HOLD_DEVELOPMENT', new_simulations=0, action_branches=0,
                   independent_confirmation=0, reused_development_seeds=True, cities=summaries,
                   training=metadata, baseline_checks=checks, model_protocol=plan,
                   prediction_unit='Next10s straight detector count MAE per edge-epoch, then equal complete development seeds',
                   source_attribution='S must improve on T, physically-stratified P and history in both cities; no control benefit measured.')
    (output / 'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(receipt['status'], flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
