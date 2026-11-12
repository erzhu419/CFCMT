"""Diagnose frozen service probabilities using existing server-side PP records."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
for sibling in ('tsc_corridor_arrival_transfer_20261107',
                'tsc_corridor_residual_transfer_20261109',
                'tsc_corridor_count_fusion_20261109',
                'tsc_corridor_service_alignment_20261109',
                'tsc_corridor_expected_service_20261109'):
    sys.path.insert(0, str(HERE.parent/sibling))
sys.path.insert(0, str(HERE))
from arrival_policy import ArrivalPolicy, fit_hours as old_fit
from model_gate import hour, load_training, verify_protocol
from residual_model import fit_hours, fit_source
from replay_calibration import ReplayPredictor, evaluate_hour
from calibration_metrics import ProbabilityCalibration, DecompositionAccumulator


def calibration_summary(records):
    """Pool descriptive bins, but weight nonempty seed-level losses equally."""
    collector = ProbabilityCalibration()
    for record in records:
        collector.merge(record['raw_sums'])
    pooled = collector.result()
    out = {}
    for arm, columns in pooled['arms'].items():
        out[arm] = {}
        for query, column in columns.items():
            available = [r['arms'][arm][query] for r in records
                         if r['arms'][arm][query]['n']]
            item = {k: v for k, v in column.items() if k not in ('bins', 'raw_sums')}
            item['nonempty_seed_count'] = len(available)
            item['equal_seed'] = {key: mean(r[key] for r in available) if available else None
                                  for key in ('p_mean', 'y_mean', 'bias', 'brier')}
            if query in ('detector_10', 'service_13'):
                item['pooled_descriptive_bins'] = column['bins']
            out[arm][query] = item
    return out


def summarize(panel):
    summaries = dict(calibration={}, decompositions={})
    for scope in panel[0]['calibration']:
        summaries['calibration'][scope] = {
            group: calibration_summary([r['calibration'][scope][group] for r in panel])
            for group in panel[0]['calibration'][scope]}
        summaries['decompositions'][scope] = {}
        for contrast in panel[0]['decompositions'][scope]:
            collector = DecompositionAccumulator()
            for record in panel:
                collector.merge(record['decompositions'][scope][contrast])
            summaries['decompositions'][scope][contrast] = collector.result()
    return summaries


def seed_summary(record):
    calibration = {scope: {group: {arm: {query: {
        key: value for key, value in columns[query].items() if key != 'bins'}
        for query in ('detector_10', 'service_13')}
        for arm, columns in result['arms'].items()}
        for group, result in groups.items()}
        for scope, groups in record['calibration'].items()}
    decompositions = {scope: {contrast: dict(decisions=value['decisions'], totals=value['totals'],
        by_label_sign={sign: {key: item for key, item in group.items() if key != 'contexts'}
                       for sign, group in value['by_label_sign'].items()})
        for contrast, value in contrasts.items()}
        for scope, contrasts in record['decompositions'].items()}
    return dict(seed=record['seed'], block=record['block'], counts=record['counts'],
                expected_reproduction=record['expected_reproduction'],
                calibration=calibration, decompositions=decompositions)


def run(root, cohort_file, frozen_result, output):
    plan = json.loads((HERE/'protocol.json').read_text())
    cohort = json.loads(cohort_file.read_text())['cities']
    expected = json.loads(frozen_result.read_text())
    verify_protocol(cohort)
    sources, old_sources = {}, {}
    for city, spec in cohort.items():
        sources[city] = fit_source(root, 'donor', city, spec['donor'])
        labels = load_training(root, 'donor', city, spec['donor'])
        old_sources[city] = old_fit(labels), old_fit(labels, permute=True)
        metadata = dict(real=sources[city][0].metadata, placebo=sources[city][1].metadata)
        if metadata != expected['training']['source'][city]:
            raise ValueError('Frozen source counts changed')
        print(f'Restored unchanged source/{city}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    cities, full = {}, {}
    with gzip.open(output/'flight_calibration_server_only.jsonl.gz', 'wt') as flights, \
            gzip.open(output/'decision_decomposition_server_only.jsonl.gz', 'wt') as decisions:
        for city, donor in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
            panel = []
            previous = {r['seed']: r for r in expected['cities'][city]['by_seed']}
            for block, seeds in enumerate(cohort[city]['adapt']):
                target = fit_hours(root, 'adapt', city, seeds)
                if dict(block=block, **target.metadata) != expected['training']['target'][city][block]:
                    raise ValueError('Frozen target counts changed')
                policy = ArrivalPolicy(old_fit(load_training(root, 'adapt', city, seeds)),
                                       *old_sources[donor])
                for seed in cohort[city]['dev'][2*block:2*block+2]:
                    predictor = ReplayPredictor(target, *sources[donor])
                    record = evaluate_hour(city, seed, block, hour(root, 'dev', city, seed),
                                           predictor, policy, previous[seed], flights, decisions)
                    panel.append(record)
                    print(f'Diagnosed {city}/{seed}; previous probabilities/actions/scores reproduced', flush=True)
            full[city] = panel
            cities[city] = summarize(panel)
            cities[city]['by_seed'] = [seed_summary(record) for record in panel]
    # Full per-seed bins and traces stay on the server. Only aggregate diagnostics are pulled.
    (output/'full_result_server_only.json').write_text(json.dumps(full, allow_nan=False)+'\n')
    result = dict(status='COMPLETE_EXISTING_RECORD_DIAGNOSTIC', previous_status=expected['status'],
                  new_simulations=0, new_models=0, new_labels=0, independent_confirmation=0,
                  frozen_seed_hours_reproduced=12, cities=cities, protocol=plan)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cohort-file', 'frozen-result', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.cohort_file, a.frozen_result, a.output)
