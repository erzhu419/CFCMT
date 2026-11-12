"""Evaluate a frozen selective-source candidate on six new Hangzhou PP hours."""
import argparse
import gzip
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import sys

HERE = Path(__file__).resolve().parent
for sibling in ('tsc_corridor_arrival_transfer_20261107',
                'tsc_corridor_residual_transfer_20261109',
                'tsc_corridor_count_fusion_20261109',
                'tsc_corridor_service_alignment_20261109',
                'tsc_corridor_expected_service_20261109',
                'tsc_corridor_target_shrink_20261109'):
    sys.path.insert(0, str(HERE.parent/sibling))
from arrival_policy import ArrivalPolicy, fit_hours as old_fit
from model_gate import hour, load_training, verify_protocol
from residual_model import fit_source, fit_hours
from shrink_model import ShrinkPredictor
from evaluate_shrink import evaluate_hour, summarize

T95_DF5 = 2.570581835636314


def inference(panel):
    """Two required contrasts; uncertainty uses six paired seed-hour losses."""
    contrasts = {}
    for other in ('T', 'P'):
        pairs = [dict(seed=row['seed'], block=row['block'],
                      difference=row['scopes']['all_observable_risk']['proper_brier']['S']-
                                 row['scopes']['all_observable_risk']['proper_brier'][other])
                 for row in panel]
        values = [row['difference'] for row in pairs]
        if len(values) != 6:
            raise ValueError('Confirmation requires the original six scheduled seed hours')
        delta = mean(values)
        half = T95_DF5*stdev(values)/sqrt(6)
        contrasts['S_vs_'+other] = dict(mean_difference=delta,
            approximate_95pct_interval=[delta-half, delta+half], paired_seed_differences=pairs,
            by_block=[dict(block=block, mean_difference=mean(
                row['difference'] for row in pairs if row['block'] == block)) for block in range(3)])
    directional = all(row['mean_difference'] < 0 for row in contrasts.values())
    supported = all(row['approximate_95pct_interval'][1] < 0 for row in contrasts.values())
    contrary = any(row['approximate_95pct_interval'][0] > 0 for row in contrasts.values())
    return dict(status='PROBABILITY_INCREMENT_SUPPORTED' if supported else
                'DIRECTION_REPLICATED' if directional else 'NOT_REPLICATED' if contrary else 'INCONCLUSIVE',
                contrasts=contrasts, uncertainty_unit='paired seed-hour; n=6, df=5; small-sample t approximation')


def run(training_root, confirmation_root, cohort_file, frozen_training, output):
    protocol = json.loads((HERE/'protocol.json').read_text())
    cohort = json.loads(cohort_file.read_text())['cities']
    frozen = json.loads(frozen_training.read_text())
    verify_protocol(cohort)
    if frozen['status'] != 'ALL_TARGET_WEIGHTS_FROZEN' or frozen['development_hours_read'] != 0:
        raise ValueError('Source weights were not frozen before development')
    source, placebo = fit_source(training_root, 'donor', 'jinan', cohort['jinan']['donor'])
    if dict(real=source.metadata, placebo=placebo.metadata) != frozen['training']['source']['jinan']:
        raise ValueError('Frozen source histogram metadata changed')
    labels = load_training(training_root, 'donor', 'jinan', cohort['jinan']['donor'])
    old_sources = old_fit(labels), old_fit(labels, permute=True)
    models, receipts = [], []
    for block, seeds in enumerate(cohort['hangzhou']['adapt']):
        expected = frozen['training']['target']['hangzhou'][block]
        weights = expected['calibration']['weights']
        if weights != {'S': protocol['weights_S'][block], 'P': protocol['weights_P'][block]}:
            raise ValueError('Previously frozen source weight changed')
        target = fit_hours(training_root, 'adapt', 'hangzhou', seeds)
        if dict(block=block, **target.metadata) != expected['metadata']:
            raise ValueError('Frozen target histogram metadata changed')
        policy = ArrivalPolicy(old_fit(load_training(training_root, 'adapt', 'hangzhou', seeds)), *old_sources)
        models.append((target, weights, policy))
        receipts.append(dict(block=block, metadata=expected['metadata'], weights=weights))
    output.mkdir(parents=True, exist_ok=True)
    (output/'model_restoration.json').write_text(json.dumps(dict(
        status='FROZEN_HISTOGRAMS_RESTORED', new_calibration_fits=0,
        confirmation_label_hours_read=0, blocks=receipts), indent=2)+'\n')
    panel = []
    for block, seeds in enumerate(protocol['seeds_by_block']):
        target, weights, policy = models[block]
        for seed in seeds:
            prediction = ShrinkPredictor(target, source, placebo, weights)
            reference = ShrinkPredictor(target, source, placebo, {'S': 1., 'P': 1.})
            record = evaluate_hour('hangzhou', seed, block,
                hour(confirmation_root, 'confirmation', 'hangzhou', seed),
                prediction, reference, policy, None)
            panel.append(record)
            print(f'Confirmed hangzhou/{seed}; fresh hour admitted', flush=True)
    result = dict(protocol=protocol, inference=inference(panel), candidate_summary=summarize(panel),
        old_two_city_status='HOLD_TARGET_SHRINK_DEVELOPMENT',
        new_calibration_fits=0, new_pp_seed_hours=6, action_branches=0,
        learned_controller_closed_loop_runs=0, confirmation_labels_used_for_training=False,
        jinan_source_contribution='structurally zero; no new Jinan simulations')
    result['status'] = result['inference']['status']
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with gzip.open(output/'aggregate_result.json.gz', 'wt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-root', 'confirmation-root', 'cohort-file', 'frozen-training', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.training_root, args.confirmation_root, args.cohort_file, args.frozen_training, args.output)
