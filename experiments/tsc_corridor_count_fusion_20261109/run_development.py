"""Evaluate a single count-fusion repair on the unchanged existing PP panel."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent/'tsc_corridor_residual_transfer_20261109',
             HERE.parent/'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0, str(path))
from arrival_policy import ArrivalPolicy, fit_hours as old_fit
from model_gate import hour, load_training, verify_protocol
from residual_model import fit_hours, fit_source, PSEUDOCOUNT, PERMUTATION_SEED
from evidence_model import EvidencePredictor
from evaluate import ARMS, PAIRS, SCOPES, evaluate_hour, summarize

CONTROLS = ('T', 'history', 'analytic', 'GP')
CONTROL_PAIRS = tuple(f'{a}_vs_{b}' for a, b in PAIRS if a in CONTROLS and b in CONTROLS)


def verify_controls(actual, expected):
    """S/P may change; the target and physical controls must remain frozen."""
    label = f"{actual['city']}/{actual['seed']}"
    for scope in SCOPES:
        a, e = actual['scopes'][scope], expected['scopes'][scope]
        if a['counts']['scored_edge_epochs'] != e['counts']['scored_edge_epochs']:
            raise ValueError(f'Frozen edge denominator changed: {label}/{scope}')
        if any(abs(a['mae'][arm]-e['mae'][arm]) > 1e-12 for arm in CONTROLS if arm != 'GP'):
            raise ValueError(f'Frozen target/physical MAE changed: {label}/{scope}')
        for field in ('action_differences', 'action_differences_unlocked'):
            if any(a[field][pair] != e[field][pair] for pair in CONTROL_PAIRS):
                raise ValueError(f'Frozen control request differences changed: {label}/{scope}/{field}')
    return dict(city=actual['city'], seed=actual['seed'], all_scopes_reproduced=True,
                controls=list(CONTROLS), control_pairs=list(CONTROL_PAIRS))


def verify_training(actual, expected, label):
    if actual != expected:
        raise ValueError(f'Training metadata differs from the frozen source/local budget: {label}')


def sparse_contexts(target, source, predictor, block):
    """Inspect the two already-identified low-source-support Jinan contexts."""
    results = []
    for queued in (False, True):
        context = (True, 11, '<=0.5', queued)
        ct, ch = target.counts[context], source.counts[context]
        nt, nh = float(ct.sum()), float(ch.sum())
        local = float(target.cdf(context, 11.0, True, [10])[0])
        old_source = float(source.cdf(context, 11.0, True, [10])[0])
        pooled = float(predictor.pooled_source.cdf(context, 11.0, True, [10])[0])
        placebo = float(predictor.pooled_placebo.cdf(context, 11.0, True, [10])[0])
        results.append(dict(block=block, current_straight=True, history_eta_bin=11,
            speed_ratio='<=0.5', queued_at_entry=queued, target_n=int(nt), source_n=int(nh),
            source_empirical_arrival10=float(ch[1:].sum()/nh), target_cdf10=local,
            old_source_cdf10=old_source, old_equal_mixture_cdf10=.5*(local+old_source),
            new_count_fusion_cdf10=pooled, new_placebo_cdf10=placebo,
            effective_source_weight=nh/(PSEUDOCOUNT+nt+nh),
            effective_physical_weight=PSEUDOCOUNT/(PSEUDOCOUNT+nt+nh),
            target_decoded10=local >= .5, old_source_mixture_decoded10=.5*(local+old_source) >= .5,
            new_source_decoded10=pooled >= .5))
    return results


def run(root, frozen_dir, output):
    plan = json.loads((HERE/'protocol.json').read_text())
    cities = json.loads((frozen_dir/'cohort.json').read_text())['cities']
    expected = json.loads((frozen_dir/'development/result.json').read_text())
    verify_protocol(cities)
    if PSEUDOCOUNT != plan['physical_pseudocount'] or PERMUTATION_SEED != plan['permutation_seed']:
        raise ValueError('The model constants differ from the fixed candidate protocol')
    output.mkdir(parents=True, exist_ok=True)
    sources, old_sources = {}, {}
    metadata = dict(source={}, target={})
    for city in cities:
        seeds = cities[city]['donor']
        sources[city] = fit_source(root, 'donor', city, seeds)
        labels = load_training(root, 'donor', city, seeds)
        old_sources[city] = old_fit(labels), old_fit(labels, permute=True)
        metadata['source'][city] = dict(real=sources[city][0].metadata, placebo=sources[city][1].metadata)
        verify_training(metadata['source'][city], expected['training']['source'][city], 'source/'+city)
        print(f'Restored unchanged source/{city}: {sources[city][0].metadata["snapshot_examples"]} snapshots', flush=True)
    summaries, checks, sparse = {}, [], []
    with gzip.open(output/'request_disagreements_server_only.jsonl.gz', 'wt') as stream:
        for city, donor in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
            panel = []
            metadata['target'][city] = []
            old_seeds = {r['seed']: r for r in expected['cities'][city]['by_seed']}
            for block, seeds in enumerate(cities[city]['adapt']):
                target = fit_hours(root, 'adapt', city, seeds)
                target_meta = dict(block=block, **target.metadata)
                verify_training(target_meta, expected['training']['target'][city][block], f'target/{city}/{block}')
                metadata['target'][city].append(target_meta)
                policy = ArrivalPolicy(old_fit(load_training(root, 'adapt', city, seeds)), *old_sources[donor])
                if city == 'jinan':
                    sparse.extend(sparse_contexts(target, sources[donor][0], EvidencePredictor(target, *sources[donor]), block))
                for seed in cities[city]['dev'][2*block:2*block+2]:
                    predictor = EvidencePredictor(target, *sources[donor])
                    record = evaluate_hour(city, seed, block, hour(root, 'dev', city, seed), predictor, policy, stream)
                    checks.append(verify_controls(record, old_seeds[seed]))
                    record['prediction_support'] = dict(predictor.counts)
                    panel.append(record)
                    print(f'Evaluated {city}/{seed}: {json.dumps(record["scopes"]["original_common_scored"]["mae"])}', flush=True)
            summaries[city] = summarize(panel)
    passed = all(s['scopes']['original_common_scored']['passed'] for s in summaries.values())
    comparisons = {city: {scope: dict(old_equal_mixture_mae=expected['cities'][city]['scopes'][scope]['equal_seed_mae'],
                                    new_count_fusion_mae=result['scopes'][scope]['equal_seed_mae'])
                          for scope in SCOPES} for city, result in summaries.items()}
    receipt = dict(status='DEVELOPMENT_SIGNAL_ONLY' if passed else 'HOLD_COUNT_FUSION_DEVELOPMENT',
        previous_model_status=expected['status'], new_simulations=0, action_branches=0,
        independent_confirmation=0, newly_collected_labels=0, reused_development_seeds=True,
        cities=summaries, training=metadata, control_reproduction_checks=checks,
        comparison_to_previous=comparisons, jinan_sparse_contexts=sparse, model_protocol=plan,
        prediction_unit='Next10s straight detector count MAE per edge-epoch, then equal complete development seeds',
        source_weight='Per-context source count / (20 + target count + source count); not fixed 0.5.',
        source_attribution='Both cities must satisfy unchanged S<T/P/history forecast and >=2-seed S/T and S/P request gates; no control benefit measured.')
    (output/'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(receipt['status'], flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'frozen-dir', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.frozen_dir, args.output)
