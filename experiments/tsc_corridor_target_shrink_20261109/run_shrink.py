"""Fit one target-constrained source weight, then screen on reused PP hours."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for sibling in ('tsc_corridor_arrival_transfer_20261107',
                'tsc_corridor_arrival_residual_20261109',
                'tsc_corridor_residual_transfer_20261109',
                'tsc_corridor_count_fusion_20261109',
                'tsc_corridor_service_alignment_20261109',
                'tsc_corridor_expected_service_20261109'):
    sys.path.insert(0, str(HERE.parent/sibling))
sys.path.insert(0, str(HERE))
from arrival_policy import ArrivalPolicy, fit_hours as old_fit
from model_gate import hour, load_training, verify_protocol
from residual_model import fit_source
from shrink_model import fit_block, ShrinkPredictor
from evaluate_shrink import evaluate_hour, summarize


def run(root, cohort_file, previous_result, output):
    protocol = json.loads((HERE/'protocol.json').read_text())
    cohort = json.loads(cohort_file.read_text())['cities']
    expected = json.loads(previous_result.read_text())
    verify_protocol(cohort)
    sources, old_sources, blocks, training = {}, {}, {}, {'source': {}, 'target': {}}
    for city, spec in cohort.items():
        sources[city] = fit_source(root, 'donor', city, spec['donor'])
        labels = load_training(root, 'donor', city, spec['donor'])
        old_sources[city] = old_fit(labels), old_fit(labels, permute=True)
        training['source'][city] = dict(real=sources[city][0].metadata,
                                        placebo=sources[city][1].metadata)
        if training['source'][city] != expected['training']['source'][city]:
            raise ValueError('Original source training metadata changed')
    for city, donor in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
        training['target'][city] = []
        blocks[city] = []
        for block, seeds in enumerate(cohort[city]['adapt']):
            target, calibration = fit_block(root, city, seeds, *sources[donor])
            metadata = dict(block=block, **target.metadata)
            if metadata != expected['training']['target'][city][block]:
                raise ValueError('Original target training metadata changed')
            old_target = old_fit(load_training(root, 'adapt', city, seeds))
            policy = ArrivalPolicy(old_target, *old_sources[donor])
            blocks[city].append((target, calibration, policy))
            training['target'][city].append(dict(metadata=metadata, calibration=calibration))
            print(f'Frozen {city}/block{block}: {calibration["weights"]}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    frozen = dict(status='ALL_TARGET_WEIGHTS_FROZEN', development_hours_read=0,
                  original_training_reproduced=True, protocol=protocol, training=training)
    (output/'training_result.json').write_text(json.dumps(frozen, indent=2, allow_nan=False)+'\n')
    print('All training artifacts written before development replay', flush=True)
    cities = {}
    for city, donor in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
        prior = {row['seed']: row for row in expected['cities'][city]['by_seed']}
        panel = []
        for block, (target, calibration, policy) in enumerate(blocks[city]):
            for seed in cohort[city]['dev'][2*block:2*block+2]:
                predictor = ShrinkPredictor(target, *sources[donor], calibration['weights'])
                old_predictor = ShrinkPredictor(target, *sources[donor], {'S': 1.0, 'P': 1.0})
                result = evaluate_hour(city, seed, block, hour(root, 'dev', city, seed),
                                       predictor, old_predictor, policy, prior[seed])
                result['weights'] = calibration['weights']
                panel.append(result)
                print(f'Screened {city}/{seed}; original expected service reproduced', flush=True)
        cities[city] = summarize(panel)
    prediction = all(city['prediction_pass'] for city in cities.values())
    ranking = prediction and all(city['ranking_pass'] for city in cities.values())
    result = dict(status=('DEVELOPMENT_RANKING_SIGNAL_ONLY' if ranking else
                          'DEVELOPMENT_PREDICTION_ONLY' if prediction else
                          'HOLD_TARGET_SHRINK_DEVELOPMENT'),
                  protocol=protocol, training=training, cities=cities,
                  new_simulations=0, newly_collected_labels=0, action_branches=0,
                  independent_confirmation=0, reused_development=True,
                  training_frozen_before_development=True,
                  training_seed_hours=28, development_seed_hours=12)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with gzip.open(output/'aggregate_result.json.gz', 'wt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cohort-file', 'previous-result', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.cohort_file, args.previous_result, args.output)
