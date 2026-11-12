"""Split frozen no-arrival labels without changing the previous state cohort."""
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
                'tsc_corridor_service_alignment_20261109',
                'tsc_corridor_state_stratification_20261109'):
    sys.path.insert(0, str(HERE.parent/sibling))
sys.path.insert(0, str(HERE))
from model_gate import hour, verify_protocol
from residual_model import rows, release_id, feature_context, update_history
from state_features import recover_state_features
from outcome_categories import classify_outcome
from outcome_stats import OutcomeStats


def process_hour(directory, receipt, stats):
    labels = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(item['status'] == 'unknown' for item in labels):
        raise ValueError('Stored hour has unresolved detector outcomes')
    panel, seed, block = (receipt[key] for key in ('panel', 'seed', 'block'))
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
        if not row['spillback'] and not row['unmatched'] and t+10 <= 3600:
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
                    # The old context and joint state are fixed before outcome classification.
                    category, name = classify_outcome(flight, outcomes[release_id(flight)], t)
                    stats.update(panel, seed, block, context, stratum, category, name, unlocked)
                    counts['selected_context_flights'] += 1
                    features['identified' if state['identified'] else 'unknown'] += 1
                    if unlocked:
                        features['unlocked_flights'] += 1
                        if not state['identified']:
                            features['unlocked_unknown'] += 1
                            features['reason:'+state['reason']] += 1
        update_history(row, history)
        previous[tls] = row['states'][row['pp']]
    if (len(timestamps) != (16 if receipt['city'] == 'hangzhou' else 12) or
            any(times != list(range(0, 3600, 10)) for times in timestamps.values())):
        raise ValueError('Stored hour has changed its complete TLS roster')
    if dict(counts) != receipt['counts'] or dict(features) != receipt['state_recovery']:
        raise ValueError('Previous flight cohort or state recovery changed')
    print(f'Decomposed {receipt["part"]}/{receipt["city"]}/{seed}; '
          f'previous cohort reproduced; selected={counts["selected_context_flights"]}', flush=True)
    return dict(receipt, outcome_cohort_reproduced=True)


def run(root, cohort_file, previous_result, output):
    plan = json.loads((HERE/'protocol.json').read_text())
    cohort = json.loads(cohort_file.read_text())['cities']
    previous = json.loads(previous_result.read_text())
    verify_protocol(cohort)
    if plan['frozen_contexts'] != previous['protocol']['frozen_contexts']:
        raise ValueError('Frozen context set differs from the previous diagnostic')
    expected_hours = {(city, part, seed) for city, spec in cohort.items()
                      for part, seeds in (('donor', spec['donor']),
                                          ('adapt', sum(spec['adapt'], [])), ('dev', spec['dev']))
                      for seed in seeds}
    if {(r['city'], r['part'], r['seed']) for r in previous['hours']} != expected_hours:
        raise ValueError('Previous diagnostic does not cover the fixed forty hours')
    collectors = {city: OutcomeStats(plan['frozen_contexts']) for city in cohort}
    receipts = []
    for receipt in previous['hours']:
        city, part, seed = (receipt[key] for key in ('city', 'part', 'seed'))
        target = ('jinan' if city == 'hangzhou' else 'hangzhou') if part == 'donor' else city
        receipts.append(process_hour(hour(root, part, city, seed), receipt, collectors[target]))
    result = dict(status='COMPLETE_EXISTING_OUTCOME_DECOMPOSITION', protocol=plan,
                  previous_development_status=previous['previous_development_status'],
                  new_simulations=0, new_models=0, new_labels=0, new_candidates=0,
                  reused_development=True, hours=receipts,
                  targets={city: stats.result(previous['targets'][city])
                           for city, stats in collectors.items()})
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with gzip.open(output/'aggregate_result.json.gz', 'wt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'cohort-file', 'previous-result', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.cohort_file, a.previous_result, a.output)
