"""Compare one aligned service score on frozen PP observations, without simulation."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent/'tsc_corridor_residual_transfer_20261109',
             HERE.parent/'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0, str(path))
from model_gate import release_id, rows
from evaluate import ARMS, PAIRS, SCOPES, choose_requests, summarize as original_summarize
from service_window import extend_window

HORIZON = 13


def old_forecasts(forecasts):
    """Endpoint is the last column; column10 now means service second11."""
    out = {}
    for edge, values in forecasts.items():
        if values.shape != (len(ARMS), HORIZON+1):
            raise ValueError('Aligned forecasts require five arms, thirteen service columns and detector10')
        out[edge] = np.column_stack((values[:, :10], values[:, -1]))
    return out


def choose_aligned(row, forecasts):
    if not forecasts:
        return {arm: row['gp'] for arm in (*ARMS, 'GP')}
    scores = np.broadcast_to(row['base'], (len(ARMS), len(row['base']))).astype(float).copy()
    for edge, values in forecasts.items():
        for request, mask in enumerate(row['protected'][edge]):
            for bit in range(HORIZON):
                if int(mask) & (1 << bit):
                    scores[:, request] += values[:, bit]
    chosen = np.argmax(np.round(scores, 12), axis=1)
    return {**{arm: int(index) for arm, index in zip(ARMS, chosen)}, 'GP': row['gp']}


def label_service_scores(row, flights, outcomes):
    """PP detector labels plus a nominal free-flow tail; never realized action cost."""
    truth = {}
    actual10 = Counter()
    t = row['t']
    for flight in flights:
        outcome = outcomes[release_id(flight)]
        if outcome['end_s'] <= t or flight['age_s'] != t-outcome['entry_s']:
            raise ValueError('Future label does not match its fixed current flight')
        if outcome['status'] == 'censored' and outcome['end_s'] < t+10:
            raise ValueError('Incomplete detector10 label window')
        edge = flight['edge']
        truth.setdefault(edge, np.zeros(HORIZON))
        if outcome['status'] == 'straight' and outcome['end_s'] <= t+10:
            actual10[edge] += 1
            ready = outcome['end_s']-t+flight['tail_s']
            truth[edge] += np.arange(1, HORIZON+1) >= ready
    scores = np.asarray(row['base'], dtype=float).copy()
    for edge, values in truth.items():
        for request, mask in enumerate(row['protected'][edge]):
            scores[request] += sum(values[bit] for bit in range(HORIZON) if int(mask) & (1 << bit))
    return scores, actual10


def _panel():
    return dict(counts=Counter(), error=Counter({arm: 0.0 for arm in ARMS}),
                old_differences={f'{a}_vs_{b}': 0 for a,b in PAIRS},
                old_unlocked={f'{a}_vs_{b}': 0 for a,b in PAIRS},
                differences={f'{a}_vs_{b}': 0 for a,b in PAIRS},
                unlocked={f'{a}_vs_{b}': 0 for a,b in PAIRS},
                changes=Counter({arm: 0 for arm in ARMS}),
                label_delta=Counter({'S_vs_T': 0.0, 'S_vs_P': 0.0, 'S_vs_GP': 0.0}),
                label_signs=Counter(), label_positive=Counter(), label_negative=Counter(), fallback=Counter())


def _finish(panel, decisions, unlocked):
    n = panel['counts']['scored_edge_epochs']
    return dict(counts=dict(panel['counts']), mae={arm: panel['error'][arm]/n for arm in ARMS},
                action_differences=panel['differences'], action_differences_unlocked=panel['unlocked'],
                action_difference_rates={k:v/decisions for k,v in panel['differences'].items()},
                action_difference_unlocked_rates={k:v/unlocked if unlocked else None for k,v in panel['unlocked'].items()},
                alignment_changes=dict(panel['changes']), fallback=dict(panel['fallback']),
                label_score_differences={k:v/decisions for k,v in panel['label_delta'].items()},
                label_score_difference_sums=dict(panel['label_delta']), label_score_positive_sums=dict(panel['label_positive']),
                label_score_negative_sums=dict(panel['label_negative']), label_score_difference_signs=dict(panel['label_signs']))


def verify_old(panel, previous, scope, label):
    n = panel['counts']['scored_edge_epochs']
    e = previous['scopes'][scope]
    if n != e['counts']['scored_edge_epochs'] or any(abs(panel['error'][a]/n-e['mae'][a])>1e-12 for a in ARMS):
        raise ValueError(f'Frozen predictions or denominators changed: {label}/{scope}')
    if (panel['old_differences'] != e['action_differences']
            or panel['old_unlocked'] != e['action_differences_unlocked']):
        raise ValueError(f'Old ten-second requests changed: {label}/{scope}')
    return dict(five_arm_mae_and_denominator=True, all_request_pairs=True, unlocked_request_pairs=True)


def evaluate_hour(city, seed, block, directory, predictor, old_policy, previous_result, disagreement_stream=None):
    labels = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(x['status']=='unknown' for x in labels):
        raise ValueError('Unresolved or duplicate detector outcomes')
    previous_requests, counts, reconstruction = {}, Counter(), Counter()
    panels = {scope: _panel() for scope in SCOPES}
    for row in rows(directory/'decisions.jsonl.gz'):
        t, tls, gp = row['t'], row['tls'], row['gp']
        unlocked = len(row['states']) > 1
        counts['decisions'] += 1; counts['unlocked_decisions'] += unlocked
        counts['locked_decisions'] += not unlocked; counts['inflight_decisions'] += bool(row['flights'])
        counts['spillback_decisions'] += row['spillback']; counts['unmatched_decisions'] += row['unmatched']
        forecasts = predictor.forecast(row)
        old = old_forecasts(forecasts)
        observable = bool(row['flights'] and not row['spillback'] and not row['unmatched'] and t+10<=3600)
        common = observable and bool(old_policy.forecast(row))
        enabled = dict(original_common_scored=common, all_observable_risk=observable)
        if observable and set(forecasts) != {f['edge'] for f in row['flights']}:
            raise ValueError('Aligned forecast changed its observable edge risk set')
        aligned = None
        if observable and unlocked:
            aligned = extend_window(row, previous_requests.get(tls), horizon=HORIZON)
            reconstruction['unlocked_reconstructed'] += 1
        elif observable:
            reconstruction['locked_structural_zero'] += 1
        new_actions = choose_aligned(aligned, forecasts) if aligned is not None else {a:gp for a in (*ARMS,'GP')}
        old_actions = {scope: choose_requests(row, old if active else None) for scope,active in enabled.items()}
        new_actions_by_scope = {scope:new_actions if active else {a:gp for a in (*ARMS,'GP')}
                                for scope,active in enabled.items()}
        # Choices are frozen before any outcome for this row is consulted.
        if observable:
            if aligned is not None:
                label_scores, actual10 = label_service_scores(aligned,row['flights'],outcomes)
            else:
                # Single feasible request: relative label scores are structurally zero.
                actual10=Counter()
                for f in row['flights']:
                    o=outcomes[release_id(f)]
                    if o['end_s']<=t or (o['status']=='censored' and o['end_s']<t+10):
                        raise ValueError('Incomplete current risk-set label')
                    actual10[f['edge']]+=int(o['status']=='straight' and o['end_s']<=t+10)
                label_scores=np.zeros(1)
        for scope,panel in panels.items():
            oa, na = old_actions[scope], new_actions_by_scope[scope]
            for a,b in PAIRS:
                key=f'{a}_vs_{b}'
                panel['old_differences'][key]+=oa[a]!=oa[b]
                panel['old_unlocked'][key]+=(oa[a]!=oa[b]) and unlocked
                panel['differences'][key]+=na[a]!=na[b]
                panel['unlocked'][key]+=(na[a]!=na[b]) and unlocked
            for arm in ARMS:
                panel['changes'][arm]+=oa[arm]!=na[arm]
            for other in ('T','P'):
                panel['counts']['eligible_S_vs_'+other+'_disagreement_epochs'] += 60<=t<=3150 and na['S']!=na[other]
            if enabled[scope]:
                panel['counts']['scored_decisions']+=1
                panel['counts']['scored_unlocked_decisions']+=unlocked
                panel['counts']['scored_flights']+=len(row['flights'])
                panel['counts']['scored_observed_arrivals']+=sum(actual10.values())
                for edge, values in forecasts.items():
                    panel['counts']['scored_edge_epochs']+=1
                    for i,arm in enumerate(ARMS):
                        panel['error'][arm]+=abs(float(values[i,-1])-actual10[edge])
                for other in ('T','P','GP'):
                    key='S_vs_'+other
                    delta=float(label_scores[na['S']]-label_scores[na[other]])
                    panel['label_delta'][key]+=delta
                    panel['label_positive'][key]+=max(delta,0.0)
                    panel['label_negative'][key]+=min(delta,0.0)
                    panel['label_signs'][key+('_positive' if delta>0 else '_negative' if delta<0 else '_zero')]+=1
            else:
                reason='spillback' if row['spillback'] else 'unmatched' if row['unmatched'] else 'no_eligible_flight' if not row['flights'] else 'incomplete_window' if t+10>3600 else 'outside_original_common_support'
                panel['fallback'][reason]+=1
            if disagreement_stream is not None and len({na[a] for a in ('T','S','P')})>1:
                disagreement_stream.write(json.dumps(dict(city=city,seed=seed,block=block,scope=scope,t=t,tls=tls,
                    states=row['states'],actions=na,old_actions=oa),separators=(',',':'))+'\n')
        previous_requests[tls]=row['states'][row['pp']]
    if dict(counts)!=previous_result['counts']:
        raise ValueError(f'Frozen decision roster or counts changed: {city}/{seed}')
    reproduced={scope:verify_old(p,previous_result,scope,f'{city}/{seed}') for scope,p in panels.items()}
    return dict(city=city,seed=seed,block=block,counts=dict(counts),reconstruction=dict(reconstruction),
                old_reproduction=reproduced,
                scopes={scope:_finish(p,counts['decisions'],counts['unlocked_decisions']) for scope,p in panels.items()})


def summarize(panel):
    result=original_summarize(panel)
    for scope, summary in result['scopes'].items():
        summary['label_score_differences']={key:mean(r['scopes'][scope]['label_score_differences'][key] for r in panel)
                                           for key in ('S_vs_T','S_vs_P','S_vs_GP')}
        summary['alignment_changes']={arm:sum(r['scopes'][scope]['alignment_changes'][arm] for r in panel) for arm in ARMS}
        summary['label_score_raw_totals']={key:dict(
            net=sum(r['scopes'][scope]['label_score_difference_sums'].get(key,0) for r in panel),
            positive=sum(r['scopes'][scope]['label_score_positive_sums'].get(key,0) for r in panel),
            negative=sum(r['scopes'][scope]['label_score_negative_sums'].get(key,0) for r in panel),
            positive_epochs=sum(r['scopes'][scope]['label_score_difference_signs'].get(key+'_positive',0) for r in panel),
            negative_epochs=sum(r['scopes'][scope]['label_score_difference_signs'].get(key+'_negative',0) for r in panel))
            for key in ('S_vs_T','S_vs_P','S_vs_GP')}
        summary['eligible_source_disagreements']={other:sum(r['scopes'][scope]['counts'].get('eligible_S_vs_'+other+'_disagreement_epochs',0) for r in panel) for other in ('T','P')}
    return result
