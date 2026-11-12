"""Compare existing source/adapt/dev arrival CDFs within frozen causal states."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for sibling in ('tsc_corridor_arrival_transfer_20261107',
                'tsc_corridor_arrival_residual_20261109',
                'tsc_corridor_residual_transfer_20261109',
                'tsc_corridor_service_alignment_20261109'):
    sys.path.insert(0, str(HERE.parent/sibling))
sys.path.insert(0, str(HERE))
from model_gate import hour, verify_protocol
from residual_model import rows, release_id, feature_context, label_category, update_history
from state_features import recover_state_features
from stratification_stats import StratificationStats


def process_hour(directory, city, part, seed, block, panel, stats, expected):
    records = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(r): r for r in records}
    if len(outcomes) != len(records) or any(r['status'] == 'unknown' for r in records):
        raise ValueError('Stored hour has unresolved detector outcomes')
    stats.register_hour(panel, seed, block)
    history, previous, timestamps = {}, {}, {}
    counts, features = Counter(), Counter()
    for row in rows(directory/'decisions.jsonl.gz'):
        t, tls = row['t'], row['tls']
        timestamps.setdefault(tls, []).append(t)
        unlocked = len(row['states']) > 1
        counts['decisions'] += 1
        counts['unlocked_decisions'] += unlocked
        counts['locked_decisions'] += not unlocked
        counts['inflight_decisions'] += bool(row['flights'])
        counts['spillback_decisions'] += row['spillback']
        counts['unmatched_decisions'] += row['unmatched']
        usable = not row['spillback'] and not row['unmatched'] and t+10 <= 3600
        if usable:
            counts['observable_flights'] += len(row['flights'])
            selected = []
            for flight in row['flights']:
                context, _ = feature_context(flight, t, history)
                if context in stats.contexts:
                    selected.append((flight, context))
            if selected:
                state = recover_state_features(row, previous.get(tls))
                for flight, context in selected:
                    stratum = state['by_edge'][flight['edge']]['stratum']
                    # All conditioning variables are fixed before reading this outcome.
                    category = label_category(flight, outcomes[release_id(flight)], t)
                    stats.update(panel, seed, block, context, stratum, category, unlocked)
                    counts['selected_context_flights'] += 1
                    features['identified' if state['identified'] else 'unknown'] += 1
                    if unlocked:
                        features['unlocked_flights'] += 1
                        if not state['identified']:
                            features['unlocked_unknown'] += 1
                            features['reason:'+state['reason']] += 1
        update_history(row, history)
        previous[tls] = row['states'][row['pp']]
    if (len(timestamps) != (16 if city == 'hangzhou' else 12) or
            any(times != list(range(0, 3600, 10)) for times in timestamps.values())):
        raise ValueError('Stored hour no longer has the frozen complete TLS roster')
    if panel == 'dev':
        if any(counts[k] != v for k, v in expected['counts'].items()):
            raise ValueError('Frozen development roster changed')
        if counts['observable_flights'] != expected['scopes']['all_observable_risk']['counts']['scored_flights']:
            raise ValueError('Frozen development flight risk set changed')
    elif counts['observable_flights'] != expected:
        raise ValueError('Frozen donor/adaptation snapshot budget changed')
    print(f'Observed {part}/{city}/{seed}; selected={counts["selected_context_flights"]}; '
          f'unlocked_unknown={features["unlocked_unknown"]}', flush=True)
    return dict(city=city, part=part, panel=panel, seed=seed, block=block,
                frozen_roster_reproduced=True, counts=dict(counts), state_recovery=dict(features))


def run(root, cohort_file, frozen_result, output):
    plan = json.loads((HERE/'protocol.json').read_text())
    cohort = json.loads(cohort_file.read_text())['cities']
    previous = json.loads(frozen_result.read_text())
    verify_protocol(cohort)
    collectors = {city: StratificationStats(plan['frozen_contexts']) for city in cohort}
    receipts = []
    for city, other in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
        source_meta = previous['training']['source'][city]['real']
        for seed in cohort[city]['donor']:
            receipts.append(process_hour(hour(root, 'donor', city, seed), city, 'donor', seed,
                None, 'source', collectors[other], source_meta['seed_examples'][str(seed)]))
        old_dev = {r['seed']: r for r in previous['cities'][city]['by_seed']}
        for block, seeds in enumerate(cohort[city]['adapt']):
            target_meta = previous['training']['target'][city][block]
            for seed in seeds:
                receipts.append(process_hour(hour(root, 'adapt', city, seed), city, 'adapt', seed,
                    block, 'local_adapt', collectors[city], target_meta['seed_examples'][str(seed)]))
            for seed in cohort[city]['dev'][2*block:2*block+2]:
                receipts.append(process_hour(hour(root, 'dev', city, seed), city, 'dev', seed,
                    block, 'dev', collectors[city], old_dev[seed]))
    result = dict(status='COMPLETE_EXISTING_STATE_STRATIFICATION', protocol=plan,
                  previous_development_status=previous['status'],
                  new_simulations=0, new_models=0, new_labels=0, new_candidates=0,
                  reused_development=True, hours=receipts,
                  targets={city: collector.result() for city, collector in collectors.items()})
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with gzip.open(output/'aggregate_result.json.gz', 'wt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'cohort-file', 'frozen-result', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.cohort_file, a.frozen_result, a.output)
