"""Diagnose recorded PP plans while preserving frozen magnitude boundaries."""
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
                'tsc_corridor_state_stratification_20261109',
                'tsc_corridor_outcome_decomposition_20261109',
                'tsc_corridor_late_motion_20261109',
                'tsc_corridor_predecision_slowing_20261109',
                'tsc_corridor_slowing_magnitude_20261109'):
    sys.path.insert(0, str(HERE.parent/sibling))
sys.path.insert(0, str(HERE))
from model_gate import hour, verify_protocol
from residual_model import rows, release_id, feature_context, update_history
from state_features import recover_state_features
from outcome_categories import classify_outcome
from motion_categories import classify_late_motion
from slowing_features import causal_slowing
from magnitude_features import normalized_magnitude
from plan_features import pp_plan_feature
from plan_stats import PlanStats


def process_hour(directory, receipt, stats):
    labels = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(item['status'] == 'unknown' for item in labels):
        raise ValueError('Stored hour has unresolved detector outcomes')
    decisions = list(rows(directory/'decisions.jsonl.gz'))
    indexed = {(row['tls'], row['t']): row for row in decisions}
    panel, seed, block = (receipt[key] for key in ('panel', 'seed', 'block'))
    stats.register_hour(panel, seed, block)
    history, previous, timestamps = {}, {}, {}
    counts, features, motion_counts, slowing_counts, reasons = (Counter() for _ in range(5))
    for row in decisions:
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
                    # All three predecision features precede outcome or future lookup.
                    slowing = causal_slowing(flight, t, indexed.get((tls, t-20)),
                                            indexed.get((tls, t-10)))
                    magnitude = normalized_magnitude(flight, slowing)
                    plan = pp_plan_feature(row, flight)
                    slowing_counts[slowing['group']] += 1
                    reasons[slowing['reason']] += 1
                    outcome = outcomes[release_id(flight)]
                    category, name = classify_outcome(flight, outcome, t)
                    progress = None
                    if name == 'straight_after10':
                        motion = classify_late_motion(flight, outcome, t, indexed.get((tls, t+10)))
                        name, progress = motion['motion_class'], motion['progress_m']
                        motion_counts[name] += 1
                    stats.update(panel, seed, block, context, stratum, category, name, unlocked,
                                 progress_m=progress, slowing=slowing, magnitude=magnitude, plan=plan)
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
    if (dict(counts) != receipt['counts'] or dict(features) != receipt['state_recovery'] or
            dict(motion_counts) != receipt['late_motion_counts'] or
            dict(slowing_counts) != receipt['slowing_counts'] or
            dict(reasons) != receipt['slowing_reasons']):
        raise ValueError('Previous cohort, state, motion labels or binary slowing changed')
    print(f'PP plan {receipt["part"]}/{receipt["city"]}/{seed}; reproduced', flush=True)
    return dict(receipt, pp_plan_cohort_reproduced=True)


def run(root, cohort_file, previous_result, output):
    plan = json.loads((HERE/'protocol.json').read_text())
    cohort = json.loads(cohort_file.read_text())['cities']
    previous = json.loads(previous_result.read_text())
    verify_protocol(cohort)
    if plan['frozen_contexts'] != previous['protocol']['frozen_contexts']:
        raise ValueError('Frozen contexts differ from the previous diagnostic')
    expected_hours = {(city, part, seed) for city, spec in cohort.items()
                      for part, seeds in (('donor', spec['donor']),
                                          ('adapt', sum(spec['adapt'], [])), ('dev', spec['dev']))
                      for seed in seeds}
    key = lambda receipt: (receipt['city'], receipt['part'], receipt['seed'])
    if ({key(receipt) for receipt in previous['hours']} != expected_hours or
            len(previous['hours']) != len(expected_hours)):
        raise ValueError('Previous diagnostic does not cover the fixed forty hours')
    training = [receipt for receipt in previous['hours'] if receipt['part'] in ('donor', 'adapt')]
    development = [receipt for receipt in previous['hours'] if receipt['part'] == 'dev']
    if len(training) != 28 or len(development) != 12:
        raise ValueError('Expected twenty-eight training and twelve development hours')
    collectors = {city: PlanStats(plan['frozen_contexts']) for city in cohort}
    replayed = {}
    output.mkdir(parents=True, exist_ok=True)
    for receipt in training:
        city, part, seed = key(receipt)
        target = ('jinan' if city == 'hangzhou' else 'hangzhou') if part == 'donor' else city
        replayed[key(receipt)] = process_hour(hour(root, part, city, seed), receipt, collectors[target])
    artifacts = {city: stats.freeze_training(previous['targets'][city]['training'])
                 for city, stats in collectors.items()}
    frozen = dict(status='FROZEN_PREVIOUS_TRAINING_WITH_PP_PLAN', protocol=plan,
                  training_seed_hours=28, development_hours_read=0, targets=artifacts)
    (output/'training_result.json').write_text(json.dumps(frozen, indent=2, allow_nan=False)+'\n')
    print('Both targets training frozen and written; beginning development replay', flush=True)
    for receipt in development:
        city, part, seed = key(receipt)
        replayed[key(receipt)] = process_hour(hour(root, part, city, seed), receipt, collectors[city])
    result = dict(status='COMPLETE_EXISTING_PP_PLAN_DIAGNOSTIC', protocol=plan,
                  previous_development_status=previous['previous_development_status'],
                  new_simulations=0, new_models=0, new_labels=0, new_candidates=0,
                  reused_development=True, training_frozen_before_development=True,
                  training_seed_hours=28, development_seed_hours=12,
                  hours=[replayed[key(receipt)] for receipt in previous['hours']],
                  targets={city: stats.result(previous['targets'][city])
                           for city, stats in collectors.items()})
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with gzip.open(output/'aggregate_result.json.gz', 'wt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cohort-file', 'previous-result', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.cohort_file, a.previous_result, a.output)
