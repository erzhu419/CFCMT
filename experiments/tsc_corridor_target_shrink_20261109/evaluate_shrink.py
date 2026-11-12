"""One frozen source-shrink candidate on the existing PP development hours."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for sibling in ('tsc_corridor_expected_service_20261109', 'tsc_corridor_service_alignment_20261109',
                'tsc_corridor_residual_transfer_20261109', 'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0, str(HERE.parent/sibling))
from model_gate import release_id, rows
from evaluate import ARMS, PAIRS, SCOPES
from evaluate_expected import CONTROLLERS, CONTRASTS
from evaluate_alignment import choose_aligned, label_service_scores
from service_window import extend_window

UNCHANGED = (0, 3, 4)


def _panel():
    return dict(counts=Counter(), endpoint_error=np.zeros(5), brier=np.zeros(5),
        h10_bias=np.zeros(5), h10_brier=np.zeros(5), fallback=Counter(),
        differences={f'{a}_vs_{b}': 0 for a, b in PAIRS},
        unlocked={f'{a}_vs_{b}': 0 for a, b in PAIRS},
        delta=Counter({key: 0.0 for key in CONTRASTS}),
        positive=Counter({key: 0.0 for key in CONTRASTS}),
        negative=Counter({key: 0.0 for key in CONTRASTS}),
        signs=Counter({key+'_'+sign: 0 for key in CONTRASTS for sign in ('positive', 'negative', 'zero')}))


def _finish(panel, decisions, unlocked):
    edges, flights = panel['counts']['scored_edge_epochs'], panel['counts']['scored_flights']
    return dict(counts=dict(panel['counts']),
        mae={arm: float(panel['endpoint_error'][i]/edges) if edges else None for i, arm in enumerate(ARMS)},
        proper_brier={arm: float(panel['brier'][i]/(10*flights)) if flights else None for i, arm in enumerate(ARMS)},
        h10_bias={arm: float(panel['h10_bias'][i]/flights) if flights else None for i, arm in enumerate(ARMS)},
        h10_brier={arm: float(panel['h10_brier'][i]/flights) if flights else None for i, arm in enumerate(ARMS)},
        action_differences=panel['differences'], action_differences_unlocked=panel['unlocked'],
        action_difference_rates={key: value/decisions for key, value in panel['differences'].items()},
        action_difference_unlocked_rates={key: value/unlocked if unlocked else None for key, value in panel['unlocked'].items()},
        fallback=dict(panel['fallback']), label_score_difference_sums=dict(panel['delta']),
        label_score_differences={key: value/decisions for key, value in panel['delta'].items()},
        label_score_positive_sums=dict(panel['positive']), label_score_negative_sums=dict(panel['negative']),
        label_score_difference_signs=dict(panel['signs']))


def _close(left, right, tolerance=1e-12):
    return left == right if left is None or right is None else abs(left-right) <= tolerance


def verify_old(actual, expected, label):
    if any(actual['counts'].get(key, 0) != value for key, value in expected['counts'].items()):
        raise ValueError(f'Old expected-service scoring denominator changed: {label}')
    if any(not _close(actual['mae'][arm], expected['mae'][arm]) for arm in ARMS):
        raise ValueError(f'Old expected-service endpoint MAE changed: {label}')
    if any(actual[field] != expected[field] for field in ('action_differences', 'action_differences_unlocked')):
        raise ValueError(f'Old expected-service request pairs changed: {label}')
    if any(not _close(actual['label_score_difference_sums'][key], expected['label_score_difference_sums'][key], 1e-9)
           for key in CONTRASTS):
        raise ValueError(f'Old expected-service PP-label score changed: {label}')
    return dict(endpoint_mae_and_denominators=True, request_pairs_all_and_unlocked=True, label_contrasts=True)


def _truth(row, outcome, flight):
    if (outcome['end_s'] <= row['t'] or flight['age_s'] != row['t']-outcome['entry_s'] or
            (outcome['status'] == 'censored' and outcome['end_s'] < row['t']+10)):
        raise ValueError('Future detector label differs from the fixed current flight')
    return ((np.arange(1, 11) >= outcome['end_s']-row['t']) & (outcome['status'] == 'straight')).astype(float)


def evaluate_hour(city, seed, block, directory, predictor, old_predictor, old_policy, previous_result):
    fresh = previous_result is None
    receipt = json.loads((directory/'result.json').read_text()) if fresh else None
    if fresh and (receipt['status'] != 'PASS_HOUR' or receipt['city'] != city or receipt['seed'] != seed or
                  receipt['elapsed_sec'] != 3600 or receipt['horizon_sec'] != 3600 or receipt['decision_interval_sec'] != 10):
        raise ValueError('Fresh confirmation collector receipt is incomplete or differs from the scheduled hour')
    labels = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(item['status'] == 'unknown' for item in labels):
        raise ValueError('Unresolved or duplicate detector outcomes')
    counts, previous_requests, reconstruction, timestamps = Counter(), {}, Counter(), {}
    panels = {scope: dict(new=_panel(), old=_panel(), changes=Counter({arm: 0 for arm in CONTROLLERS})) for scope in SCOPES}
    for row in rows(directory/'decisions.jsonl.gz'):
        t, tls, gp = row['t'], row['tls'], row['gp']
        if fresh:
            timestamps.setdefault(tls, []).append(t)
        unlocked = len(row['states']) > 1
        counts['decisions'] += 1
        counts['unlocked_decisions'] += unlocked
        counts['locked_decisions'] += not unlocked
        counts['inflight_decisions'] += bool(row['flights'])
        counts['spillback_decisions'] += row['spillback']
        counts['unmatched_decisions'] += row['unmatched']
        forecasts = dict(new=predictor.forecast(row), old=old_predictor.forecast(row))
        raw = dict(new=predictor.raw_detector_cdfs, old=old_predictor.raw_detector_cdfs)
        observable = bool(row['flights'] and not row['spillback'] and not row['unmatched'] and t+10 <= 3600)
        common = observable and bool(old_policy.forecast(row))
        enabled = dict(original_common_scored=common, all_observable_risk=observable)
        if observable:
            edges = {flight['edge'] for flight in row['flights']}
            if any(set(value) != edges for value in forecasts.values()) or any(len(value) != len(row['flights']) for value in raw.values()):
                raise ValueError('Source shrink changed the observable risk set')
            for edge in edges:
                new, old = forecasts['new'][edge], forecasts['old'][edge]
                if new.shape != (5, 14) or old.shape != (5, 14) or not np.array_equal(new[list(UNCHANGED)], old[list(UNCHANGED)]):
                    raise ValueError('Pure-target or physical service forecast changed')
            for flight, (new_flight, new), (old_flight, old) in zip(row['flights'], raw['new'], raw['old']):
                if (release_id(flight) != release_id(new_flight) or release_id(flight) != release_id(old_flight) or
                        new.shape != (5, 10) or old.shape != (5, 10) or
                        not np.array_equal(new[list(UNCHANGED)], old[list(UNCHANGED)])):
                    raise ValueError('Pure-target or physical raw CDF changed')
        aligned = extend_window(row, previous_requests.get(tls), horizon=13) if observable and unlocked else None
        if aligned is not None:
            reconstruction['unlocked_reconstructed'] += 1
        elif observable:
            reconstruction['locked_structural_zero'] += 1
        fallback = {arm: gp for arm in CONTROLLERS}
        actions = {name: choose_aligned(aligned, value) if aligned is not None else fallback for name, value in forecasts.items()}
        requests = {scope: {name: value if active else fallback for name, value in actions.items()}
                    for scope, active in enabled.items()}
        if any(requests[scope]['new'][arm] != requests[scope]['old'][arm]
               for scope in SCOPES for arm in ('T', 'history', 'analytic', 'GP')):
            raise ValueError('Unchanged controller request changed')
        # Both request sets, in both scopes, are fixed before current future labels are consulted.
        truths, actual10 = [], Counter()
        if observable:
            if aligned is not None:
                label_scores, actual10 = label_service_scores(aligned, row['flights'], outcomes)
            else:
                label_scores = np.zeros(1)
            for flight in row['flights']:
                truth = _truth(row, outcomes[release_id(flight)], flight)
                truths.append(truth)
                if aligned is None:
                    actual10[flight['edge']] += int(truth[-1])
        for scope, both in panels.items():
            for arm in CONTROLLERS:
                both['changes'][arm] += requests[scope]['new'][arm] != requests[scope]['old'][arm]
            for name in ('new', 'old'):
                panel, chosen = both[name], requests[scope][name]
                for a, b in PAIRS:
                    key = f'{a}_vs_{b}'
                    panel['differences'][key] += chosen[a] != chosen[b]
                    panel['unlocked'][key] += (chosen[a] != chosen[b]) and unlocked
                for other in ('T', 'P'):
                    panel['counts']['eligible_S_vs_'+other+'_disagreement_epochs'] += 60 <= t <= 3150 and chosen['S'] != chosen[other]
                if enabled[scope]:
                    panel['counts']['scored_decisions'] += 1
                    panel['counts']['scored_unlocked_decisions'] += unlocked
                    panel['counts']['scored_flights'] += len(row['flights'])
                    panel['counts']['scored_observed_arrivals'] += sum(actual10.values())
                    panel['counts']['scored_query_epochs'] += 10*len(row['flights'])
                    for edge, values in forecasts[name].items():
                        panel['counts']['scored_edge_epochs'] += 1
                        panel['endpoint_error'] += np.abs(values[:, -1]-actual10[edge])
                    for (flight, values), truth in zip(raw[name], truths):
                        error = values-truth
                        panel['brier'] += np.square(error).sum(axis=1)
                        panel['h10_bias'] += error[:, -1]
                        panel['h10_brier'] += np.square(error[:, -1])
                else:
                    reason = ('spillback' if row['spillback'] else 'unmatched' if row['unmatched'] else
                              'no_eligible_flight' if not row['flights'] else 'incomplete_window' if t+10 > 3600 else
                              'outside_original_common_support')
                    panel['fallback'][reason] += 1
                for key in CONTRASTS:
                    delta = float(label_scores[chosen['S']]-label_scores[chosen[key[5:]]]) if enabled[scope] else 0.0
                    panel['delta'][key] += delta
                    panel['positive'][key] += max(delta, 0.0)
                    panel['negative'][key] += min(delta, 0.0)
                    panel['signs'][key+('_positive' if delta > 0 else '_negative' if delta < 0 else '_zero')] += 1
        previous_requests[tls] = row['states'][row['pp']]
    admission = None
    if fresh:
        if len(timestamps) != 16 or any(times != list(range(0, 3600, 10)) for times in timestamps.values()):
            raise ValueError('Fresh confirmation TLS roster is incomplete or contains duplicate decision times')
        collected = receipt['counts']
        expected_counts = dict(decisions=collected['tls_epochs'], unlocked_decisions=collected['controllable_epochs'],
            locked_decisions=collected['tls_epochs']-collected['controllable_epochs'],
            inflight_decisions=collected['flight_epochs'], spillback_decisions=collected['spillback_epochs'],
            unmatched_decisions=collected['unmatched_epochs'])
        if dict(counts) != expected_counts:
            raise ValueError('Fresh confirmation decision counts differ from the collector receipt')
        admission = dict(collector_receipt=True, complete_tls_roster=True, decision_counts=True,
                         tls_count=len(timestamps), decision_epochs=counts['decisions'])
    elif dict(counts) != previous_result['counts']:
        raise ValueError('Original expected-service decision counts changed')
    result, reproduction = {}, None if fresh else {}
    for scope, both in panels.items():
        new, old = (_finish(both[name], counts['decisions'], counts['unlocked_decisions']) for name in ('new', 'old'))
        if not fresh:
            reproduction[scope] = verify_old(old, previous_result['scopes'][scope], f'{city}/{seed}/{scope}')
        new.update(old=old, scoring_changes=dict(both['changes']))
        result[scope] = new
    record = dict(city=city, seed=seed, block=block, weights=dict(predictor.weights), counts=dict(counts),
                  reconstruction=dict(reconstruction), old_reproduction=reproduction, scopes=result)
    if fresh:
        record['new_panel_admission'] = admission
    return record


def _average(records, field):
    return {arm: mean(record[field][arm] for record in records)
            if all(record[field][arm] is not None for record in records) else None for arm in ARMS}


def summarize(panel):
    if len(panel) != 6 or any(sum(record['block'] == block for record in panel) != 2 for block in range(3)):
        raise ValueError('Expected six development hours, two per fixed adaptation block')
    result = dict(by_seed=panel, scopes={})
    for scope in SCOPES:
        records = [record['scopes'][scope] for record in panel]
        losses = _average(records, 'proper_brier')
        labels = {key: mean(record['label_score_differences'][key] for record in records) for key in CONTRASTS}
        disagreement_seeds = {key: [record['seed'] for record in panel if record['scopes'][scope]['action_differences'][key]]
                              for key in ('S_vs_T', 'S_vs_P')}
        prediction = losses['S'] is not None and all(losses[other] is not None and losses['S'] < losses[other]
                                                   for other in ('T', 'P', 'history'))
        distinct = all(len(seeds) >= 2 for seeds in disagreement_seeds.values())
        ranking = prediction and distinct and all(labels[key] > 0 for key in ('S_vs_T', 'S_vs_P'))
        summary = dict(equal_seed_proper_brier=losses, equal_seed_h10_bias=_average(records, 'h10_bias'),
            equal_seed_h10_brier=_average(records, 'h10_brier'), equal_seed_mae=_average(records, 'mae'),
            old_equal_seed_proper_brier=_average([record['old'] for record in records], 'proper_brier'),
            label_score_differences=labels, disagreement_seeds=disagreement_seeds,
            prediction_pass=prediction, ranking_pass=ranking, distinct_requests=distinct,
            empty_probability_seeds=[record['seed'] for record in panel if record['scopes'][scope]['proper_brier']['S'] is None],
            action_differences={key: sum(record['action_differences'][key] for record in records) for key in records[0]['action_differences']},
            action_differences_unlocked={key: sum(record['action_differences_unlocked'][key] for record in records) for key in records[0]['action_differences_unlocked']},
            by_block=[dict(block=block, proper_brier=_average([record['scopes'][scope] for record in panel if record['block'] == block], 'proper_brier'),
                label_score_differences={key: mean(record['scopes'][scope]['label_score_differences'][key]
                    for record in panel if record['block'] == block) for key in CONTRASTS},
                weights=[dict(seed=record['seed'], **record['weights']) for record in panel if record['block'] == block]) for block in range(3)],
            paired_differences={other: [dict(seed=record['seed'], block=record['block'],
                proper_brier=record['scopes'][scope]['proper_brier']['S']-record['scopes'][scope]['proper_brier'][other]
                    if record['scopes'][scope]['proper_brier']['S'] is not None else None,
                label_score=record['scopes'][scope]['label_score_differences']['S_vs_'+other]) for record in panel] for other in ('T', 'P')})
        result['scopes'][scope] = summary
    primary = result['scopes']['all_observable_risk']
    result.update(prediction_pass=primary['prediction_pass'], ranking_pass=primary['ranking_pass'],
        status='CONTROL_RANKING_DEVELOPMENT_SIGNAL_ONLY' if primary['ranking_pass'] else
               'PREDICTION_ONLY' if primary['prediction_pass'] else 'HOLD')
    return result
