"""Compare expected and hard service scores on the existing frozen PP records."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE.parent/'tsc_corridor_service_alignment_20261109',
             HERE.parent/'tsc_corridor_residual_transfer_20261109',
             HERE.parent/'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0, str(path))
from model_gate import release_id, rows
from evaluate import ARMS, PAIRS, SCOPES, summarize as original_summarize
from evaluate_alignment import choose_aligned, label_service_scores
from service_window import extend_window

HORIZON = 13
CONTROLLERS = (*ARMS, 'GP')
CONTRASTS = tuple('S_vs_'+other for other in ('T', 'P', 'history', 'GP'))
OLD_CONTRASTS = ('S_vs_T', 'S_vs_P', 'S_vs_GP')
PHYSICAL_PAIRS = ('history_vs_analytic', 'history_vs_GP', 'analytic_vs_GP')


def _panel():
    pairs = {f'{a}_vs_{b}': 0 for a, b in PAIRS}
    return dict(counts=Counter(), error=Counter({arm: 0.0 for arm in ARMS}),
                differences=pairs.copy(), unlocked=pairs.copy(),
                old_differences=pairs.copy(), old_unlocked=pairs.copy(),
                changes=Counter({arm: 0 for arm in ARMS}), fallback=Counter(),
                label_delta=Counter({key: 0.0 for key in CONTRASTS}),
                old_label_delta=Counter({key: 0.0 for key in OLD_CONTRASTS}),
                label_positive=Counter({key: 0.0 for key in CONTRASTS}),
                label_negative=Counter({key: 0.0 for key in CONTRASTS}),
                label_signs=Counter({key+'_'+sign: 0 for key in CONTRASTS
                                     for sign in ('positive', 'negative', 'zero')}),
                regrets={decoder: {arm: dict(strict_score_gain_sum=0.0, positive_epochs=0)
                                    for arm in CONTROLLERS} for decoder in ('soft', 'hard')})


def _finish(panel, decisions, unlocked):
    n = panel['counts']['scored_edge_epochs']
    result = dict(counts=dict(panel['counts']),
                  mae={arm: panel['error'][arm]/n if n else None for arm in ARMS},
                  action_differences=panel['differences'],
                  action_differences_unlocked=panel['unlocked'],
                  action_difference_rates={key: value/decisions for key, value in panel['differences'].items()},
                  action_difference_unlocked_rates={key: value/unlocked if unlocked else None
                                                    for key, value in panel['unlocked'].items()},
                  scoring_changes=dict(panel['changes']), fallback=dict(panel['fallback']),
                  label_score_differences={key: value/decisions for key, value in panel['label_delta'].items()},
                  label_score_difference_sums=dict(panel['label_delta']),
                  label_score_positive_sums=dict(panel['label_positive']),
                  label_score_negative_sums=dict(panel['label_negative']),
                  label_score_difference_signs=dict(panel['label_signs']),
                  label_score_all_decision_means={key: dict(
                      net=panel['label_delta'][key]/decisions,
                      positive=panel['label_positive'][key]/decisions,
                      negative=panel['label_negative'][key]/decisions,
                      **{sign+'_frequency': panel['label_signs'][key+'_'+sign]/decisions
                         for sign in ('positive', 'negative', 'zero')}) for key in CONTRASTS})
    for decoder, arms in panel['regrets'].items():
        result['oracle_regret_'+decoder] = {
            arm: dict(values, mean_score_gain_all_decisions=values['strict_score_gain_sum']/decisions)
            for arm, values in arms.items()}
    return result


def verify_old(panel, previous, scope, label):
    """The hard view must reproduce the prior 13-second evaluation exactly."""
    expected = previous['scopes'][scope]
    n = panel['counts']['scored_edge_epochs']
    mae = {arm: panel['error'][arm]/n if n else None for arm in ARMS}
    if n != expected['counts']['scored_edge_epochs'] or any(
            mae[arm] != expected['mae'][arm] and
            (mae[arm] is None or expected['mae'][arm] is None or
             abs(mae[arm]-expected['mae'][arm]) > 1e-12) for arm in ARMS):
        raise ValueError(f'Frozen predictions or denominators changed: {label}/{scope}')
    if (panel['old_differences'] != expected['action_differences'] or
            panel['old_unlocked'] != expected['action_differences_unlocked']):
        raise ValueError(f'Old hard thirteen-second requests changed: {label}/{scope}')
    if any(abs(panel['old_label_delta'][key]-expected['label_score_difference_sums'][key]) > 1e-9
           for key in OLD_CONTRASTS):
        raise ValueError(f'Old hard PP-label score differences changed: {label}/{scope}')
    if any(panel['differences'][key] != expected['action_differences'][key] or
           panel['unlocked'][key] != expected['action_differences_unlocked'][key]
           for key in PHYSICAL_PAIRS):
        raise ValueError(f'Physical or reference requests changed: {label}/{scope}')
    return dict(five_arm_mae_and_denominator=True, all_request_pairs=True,
                unlocked_request_pairs=True, PP_label_score_deltas=True,
                physical_reference_pairs=True)


def evaluate_hour(city, seed, block, directory, predictor, old_policy,
                  previous_result, disagreement_stream=None):
    """Freeze both request sets before consulting any future label for the row.

    Label scores and oracle gaps are nominal PP-label proxies, not realized
    counterfactual service, waiting time, or control cost.
    """
    labels = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(item['status'] == 'unknown' for item in labels):
        raise ValueError('Unresolved or duplicate detector outcomes')
    previous_requests, counts, reconstruction = {}, Counter(), Counter()
    panels = {scope: _panel() for scope in SCOPES}
    for row in rows(directory/'decisions.jsonl.gz'):
        t, tls, gp = row['t'], row['tls'], row['gp']
        unlocked = len(row['states']) > 1
        counts['decisions'] += 1
        counts['unlocked_decisions'] += unlocked
        counts['locked_decisions'] += not unlocked
        counts['inflight_decisions'] += bool(row['flights'])
        counts['spillback_decisions'] += row['spillback']
        counts['unmatched_decisions'] += row['unmatched']
        forecasts = predictor.forecast(row)
        hard = predictor.hard_forecasts
        observable = bool(row['flights'] and not row['spillback'] and
                          not row['unmatched'] and t+10 <= 3600)
        common = observable and bool(old_policy.forecast(row))
        enabled = dict(original_common_scored=common, all_observable_risk=observable)
        if observable:
            if set(forecasts) != {flight['edge'] for flight in row['flights']} or set(hard) != set(forecasts):
                raise ValueError('Expected service changed the observable edge risk set')
            for edge, values in forecasts.items():
                if values.shape != (len(ARMS), HORIZON+1) or hard[edge].shape != values.shape:
                    raise ValueError('Expected five arms, thirteen service columns and detector10')
                if not np.array_equal(values[:, -1], hard[edge][:, -1]) or not np.array_equal(values[3:], hard[edge][3:]):
                    raise ValueError('Expected service changed the hard detector endpoint or physical forecasts')
        aligned = None
        if observable and unlocked:
            aligned = extend_window(row, previous_requests.get(tls), horizon=HORIZON)
            reconstruction['unlocked_reconstructed'] += 1
        elif observable:
            reconstruction['locked_structural_zero'] += 1
        fallback = {arm: gp for arm in CONTROLLERS}
        soft_actions = choose_aligned(aligned, forecasts) if aligned is not None else fallback
        hard_actions = choose_aligned(aligned, hard) if aligned is not None else fallback
        soft_by_scope = {scope: soft_actions if active else fallback for scope, active in enabled.items()}
        hard_by_scope = {scope: hard_actions if active else fallback for scope, active in enabled.items()}
        # Both complete request sets are fixed before a current future outcome is read.
        if observable:
            if aligned is not None:
                label_scores, actual10 = label_service_scores(aligned, row['flights'], outcomes)
            else:
                actual10 = Counter()
                for flight in row['flights']:
                    outcome = outcomes[release_id(flight)]
                    if (outcome['end_s'] <= t or flight['age_s'] != t-outcome['entry_s'] or
                            (outcome['status'] == 'censored' and outcome['end_s'] < t+10)):
                        raise ValueError('Incomplete or mismatched current risk-set label')
                    actual10[flight['edge']] += int(outcome['status'] == 'straight' and outcome['end_s'] <= t+10)
                label_scores = np.zeros(1)
        for scope, panel in panels.items():
            soft, old = soft_by_scope[scope], hard_by_scope[scope]
            for a, b in PAIRS:
                key = f'{a}_vs_{b}'
                panel['differences'][key] += soft[a] != soft[b]
                panel['unlocked'][key] += (soft[a] != soft[b]) and unlocked
                panel['old_differences'][key] += old[a] != old[b]
                panel['old_unlocked'][key] += (old[a] != old[b]) and unlocked
            for arm in ARMS:
                panel['changes'][arm] += soft[arm] != old[arm]
            for other in ('T', 'P'):
                panel['counts']['eligible_S_vs_'+other+'_disagreement_epochs'] += 60 <= t <= 3150 and soft['S'] != soft[other]
            if enabled[scope]:
                panel['counts']['scored_decisions'] += 1
                panel['counts']['scored_unlocked_decisions'] += unlocked
                panel['counts']['scored_flights'] += len(row['flights'])
                panel['counts']['scored_observed_arrivals'] += sum(actual10.values())
                for edge, values in forecasts.items():
                    panel['counts']['scored_edge_epochs'] += 1
                    for index, arm in enumerate(ARMS):
                        panel['error'][arm] += abs(float(values[index, -1])-actual10[edge])
                for key in OLD_CONTRASTS:
                    panel['old_label_delta'][key] += float(label_scores[old['S']]-label_scores[old[key[5:]]])
                best = float(np.max(label_scores))
                for decoder, actions in (('soft', soft), ('hard', old)):
                    for arm in CONTROLLERS:
                        gap = best-float(label_scores[actions[arm]])
                        panel['regrets'][decoder][arm]['strict_score_gain_sum'] += gap
                        panel['regrets'][decoder][arm]['positive_epochs'] += gap > 0
            else:
                reason = ('spillback' if row['spillback'] else 'unmatched' if row['unmatched'] else
                          'no_eligible_flight' if not row['flights'] else 'incomplete_window' if t+10 > 3600 else
                          'outside_original_common_support')
                panel['fallback'][reason] += 1
            # Inactive and locked decisions contribute structural zeros to the all-TLS estimand.
            for key in CONTRASTS:
                delta = float(label_scores[soft['S']]-label_scores[soft[key[5:]]]) if enabled[scope] else 0.0
                panel['label_delta'][key] += delta
                panel['label_positive'][key] += max(delta, 0.0)
                panel['label_negative'][key] += min(delta, 0.0)
                panel['label_signs'][key+('_positive' if delta > 0 else '_negative' if delta < 0 else '_zero')] += 1
            if disagreement_stream is not None and len({soft[arm] for arm in ('T', 'S', 'P')}) > 1:
                disagreement_stream.write(json.dumps(dict(city=city, seed=seed, block=block, scope=scope,
                    t=t, tls=tls, states=row['states'], actions=soft, hard_actions=old), separators=(',', ':'))+'\n')
        previous_requests[tls] = row['states'][row['pp']]
    if dict(counts) != previous_result['counts']:
        raise ValueError(f'Frozen decision roster or counts changed: {city}/{seed}')
    reproduced = {scope: verify_old(panel, previous_result, scope, f'{city}/{seed}')
                  for scope, panel in panels.items()}
    return dict(city=city, seed=seed, block=block, counts=dict(counts), reconstruction=dict(reconstruction),
                old_reproduction=reproduced,
                scopes={scope: _finish(panel, counts['decisions'], counts['unlocked_decisions'])
                        for scope, panel in panels.items()})


def summarize(panel):
    """Retain the original forecast/action gates and equal weights for six seeds."""
    result = original_summarize(panel)
    for scope, summary in result['scopes'].items():
        records = [record['scopes'][scope] for record in panel]
        summary['label_score_differences'] = {
            key: mean(record['label_score_differences'][key] for record in records) for key in CONTRASTS}
        summary['label_score_all_decision_means'] = {key: {
            metric: mean(record['label_score_all_decision_means'][key][metric] for record in records)
            for metric in ('net', 'positive', 'negative', 'positive_frequency', 'negative_frequency', 'zero_frequency')}
            for key in CONTRASTS}
        summary['scoring_changes'] = {arm: sum(record['scoring_changes'][arm] for record in records) for arm in ARMS}
        summary['label_score_raw_totals'] = {key: dict(
            net=sum(record['label_score_difference_sums'][key] for record in records),
            positive=sum(record['label_score_positive_sums'][key] for record in records),
            negative=sum(record['label_score_negative_sums'][key] for record in records),
            **{sign+'_epochs': sum(record['label_score_difference_signs'][key+'_'+sign] for record in records)
               for sign in ('positive', 'negative', 'zero')}) for key in CONTRASTS}
        summary['eligible_source_disagreements'] = {
            other: sum(record['counts'].get('eligible_S_vs_'+other+'_disagreement_epochs', 0) for record in records)
            for other in ('T', 'P')}
        for decoder in ('soft', 'hard'):
            field = 'oracle_regret_'+decoder
            summary[field] = {arm: dict(
                strict_score_gain_sum=sum(record[field][arm]['strict_score_gain_sum'] for record in records),
                positive_epochs=sum(record[field][arm]['positive_epochs'] for record in records),
                mean_score_gain_all_decisions=mean(record[field][arm]['mean_score_gain_all_decisions'] for record in records))
                for arm in CONTROLLERS}
        for block in summary['by_block']:
            block['label_score_differences'] = {
                key: mean(record['scopes'][scope]['label_score_differences'][key]
                          for record in panel if record['block'] == block['block']) for key in CONTRASTS}
    return result
