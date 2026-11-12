"""Read-only per-flight calibration replay of the frozen expected-service policy."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'tsc_corridor_expected_service_20261109'))
from expected_predictor import ExpectedServicePredictor
from residual_model import PSEUDOCOUNT, feature_context
import evaluate_expected as e
from calibration_metrics import (
    DecompositionAccumulator, ProbabilityCalibration, decision_decomposition,
)

GROUPS = ('all_observable', 'unlocked', 'S_vs_T', 'S_vs_P',
          'S_vs_T_w_positive', 'S_vs_T_w_negative',
          'S_vs_P_w_positive', 'S_vs_P_w_negative')
CONTRASTS = ('S_vs_T', 'S_vs_P')


def _empirical(counts, bounds):
    if counts is None or not counts.sum():
        return None
    cumulative = np.r_[0., np.cumsum(counts[1:])]
    values = cumulative[np.clip(np.floor(bounds).astype(int), 0, 10)]/counts.sum()
    return np.where(np.asarray(bounds) > 0, values, 0.)


class ReplayPredictor(ExpectedServicePredictor):
    """Capture the same raw CDF call before endpoint decoding or history update."""
    def forecast(self, row):
        self.captures = []
        return super().forecast(row)

    def flight_cdfs(self, flight, time_s, bounds):
        context, eta = feature_context(flight, time_s, self.history)
        raw = super().flight_cdfs(flight, time_s, bounds)
        frozen = raw.copy()
        target = self.target.counts.get(context)
        source = self.source.counts.get(context)
        nt = float(target.sum()) if target is not None else 0.
        ns = float(source.sum()) if source is not None else 0.
        self.captures.append(dict(flight=flight, raw=frozen, context=context, eta=eta,
            bounds=np.asarray(bounds), n_target=nt, n_source=ns,
            source_weight=ns/(PSEUDOCOUNT+nt+ns) if source is not None else 0.,
            target_empirical=_empirical(target, bounds),
            source_empirical=_empirical(source, bounds)))
        return raw


def _truth(flight, outcome, time_s, bounds):
    if (outcome['end_s'] <= time_s or flight['age_s'] != time_s-outcome['entry_s'] or
            (outcome['status'] == 'censored' and outcome['end_s'] < time_s+10)):
        raise ValueError('Incomplete or mismatched current risk-set label')
    lag = outcome['end_s']-time_s
    arrived = outcome['status'] == 'straight' and lag <= 10
    return ((np.asarray(bounds) > 0) & arrived & (np.asarray(bounds) >= lag)).astype(float)


def _write(stream, record):
    if stream is not None:
        stream.write(json.dumps(record, default=lambda value: value.tolist(),
                                separators=(',', ':'))+'\n')


def _calibrate(calibration, decompositions, aligned, flights, captures, actions,
               label_scores, unlocked):
    for capture in captures:
        calibration['all_observable'].update(capture['raw'][:3], capture['truth'])
        if unlocked:
            calibration['unlocked'].update(capture['raw'][:3], capture['truth'])
    details = {}
    for contrast in CONTRASTS:
        a, b = contrast.split('_vs_')
        if not unlocked or actions[a] == actions[b]:
            continue
        detail = decision_decomposition(aligned, flights, captures, actions, label_scores,
                                        contrast=(a, b))
        details[contrast] = detail
        decompositions[contrast].update(detail)
        for capture in captures:
            raw, truth = capture['raw'][:3], capture['truth']
            calibration[contrast].update(raw, truth)
            masks = aligned['protected'][capture['flight']['edge']]
            signed = np.asarray([int(bool(int(masks[actions[a]]) & (1 << h)))-
                                 int(bool(int(masks[actions[b]]) & (1 << h)))
                                 for h in range(e.HORIZON)])
            for sign, mask in (('positive', signed > 0), ('negative', signed < 0)):
                calibration[contrast+'_w_'+sign].update(raw, truth, mask=np.r_[mask, False])
    return details


def evaluate_hour(city, seed, block, directory, predictor, old_policy, previous_result,
                  flight_stream=None, decision_stream=None):
    """One raw-record pass; freeze all requests before looking up each row's labels."""
    directory = Path(directory)
    labels = list(e.rows(directory/'labels.jsonl.gz'))
    outcomes = {e.release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(item['status'] == 'unknown' for item in labels):
        raise ValueError('Unresolved or duplicate detector outcomes')
    previous_requests, counts, reconstruction = {}, Counter(), Counter()
    panels = {scope: e._panel() for scope in e.SCOPES}
    calibration = {scope: {group: ProbabilityCalibration() for group in GROUPS} for scope in e.SCOPES}
    decompositions = {scope: {key: DecompositionAccumulator() for key in CONTRASTS} for scope in e.SCOPES}
    for row in e.rows(directory/'decisions.jsonl.gz'):
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
            if len(predictor.captures) != len(row['flights']):
                raise ValueError('Per-flight capture differs from the observable risk set')
            for edge, values in forecasts.items():
                if values.shape != (len(e.ARMS), e.HORIZON+1) or hard[edge].shape != values.shape:
                    raise ValueError('Expected five arms, thirteen service columns and detector10')
                if not np.array_equal(values[:, -1], hard[edge][:, -1]) or not np.array_equal(values[3:], hard[edge][3:]):
                    raise ValueError('Expected service changed detector endpoints or physical forecasts')
        aligned = None
        if observable and unlocked:
            aligned = e.extend_window(row, previous_requests.get(tls), horizon=e.HORIZON)
            reconstruction['unlocked_reconstructed'] += 1
        elif observable:
            reconstruction['locked_structural_zero'] += 1
        fallback = {arm: gp for arm in e.CONTROLLERS}
        soft_actions = e.choose_aligned(aligned, forecasts) if aligned is not None else fallback
        hard_actions = e.choose_aligned(aligned, hard) if aligned is not None else fallback
        soft_by_scope = {scope: soft_actions if active else fallback for scope, active in enabled.items()}
        hard_by_scope = {scope: hard_actions if active else fallback for scope, active in enabled.items()}
        # All scope/decoder actions above are fixed before accessing any row outcome.
        if observable:
            for capture in predictor.captures:
                flight = capture['flight']
                capture['truth'] = _truth(flight, outcomes[e.release_id(flight)], t, capture['bounds'])
                _write(flight_stream, dict(city=city, seed=seed, block=block, t=t, tls=tls,
                    enabled=enabled, unlocked=unlocked, **capture))
            if aligned is not None:
                label_scores, actual10 = e.label_service_scores(aligned, row['flights'], outcomes)
            else:
                actual10 = Counter()
                for capture in predictor.captures:
                    actual10[capture['flight']['edge']] += int(capture['truth'][-1])
                label_scores = np.zeros(1)
        details_by_scope = {}
        for scope, panel in panels.items():
            soft, old = soft_by_scope[scope], hard_by_scope[scope]
            for a, b in e.PAIRS:
                key = f'{a}_vs_{b}'
                panel['differences'][key] += soft[a] != soft[b]
                panel['unlocked'][key] += (soft[a] != soft[b]) and unlocked
                panel['old_differences'][key] += old[a] != old[b]
                panel['old_unlocked'][key] += (old[a] != old[b]) and unlocked
            for arm in e.ARMS:
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
                    for index, arm in enumerate(e.ARMS):
                        panel['error'][arm] += abs(float(values[index, -1])-actual10[edge])
                for key in e.OLD_CONTRASTS:
                    panel['old_label_delta'][key] += float(label_scores[old['S']]-label_scores[old[key[5:]]])
                best = float(np.max(label_scores))
                for decoder, actions in (('soft', soft), ('hard', old)):
                    for arm in e.CONTROLLERS:
                        gap = best-float(label_scores[actions[arm]])
                        panel['regrets'][decoder][arm]['strict_score_gain_sum'] += gap
                        panel['regrets'][decoder][arm]['positive_epochs'] += gap > 0
                details_by_scope[scope] = _calibrate(calibration[scope], decompositions[scope],
                    aligned, row['flights'], predictor.captures, soft, label_scores, unlocked)
            else:
                reason = ('spillback' if row['spillback'] else 'unmatched' if row['unmatched'] else
                          'no_eligible_flight' if not row['flights'] else 'incomplete_window' if t+10 > 3600 else
                          'outside_original_common_support')
                panel['fallback'][reason] += 1
            for key in e.CONTRASTS:
                delta = float(label_scores[soft['S']]-label_scores[soft[key[5:]]]) if enabled[scope] else 0.0
                panel['label_delta'][key] += delta
                panel['label_positive'][key] += max(delta, 0.0)
                panel['label_negative'][key] += min(delta, 0.0)
                panel['label_signs'][key+('_positive' if delta > 0 else '_negative' if delta < 0 else '_zero')] += 1
        _write(decision_stream, dict(city=city, seed=seed, block=block, t=t, tls=tls,
            enabled=enabled, unlocked=unlocked, actions=soft_by_scope, hard_actions=hard_by_scope,
            label_scores=label_scores if observable else None, decompositions=details_by_scope))
        previous_requests[tls] = row['states'][row['pp']]
    scopes = {scope: e._finish(panel, counts['decisions'], counts['unlocked_decisions'])
              for scope, panel in panels.items()}
    if dict(counts) != previous_result['counts']:
        raise ValueError(f'Frozen decision roster or counts changed: {city}/{seed}')
    for scope in e.SCOPES:
        if scopes[scope] != previous_result['scopes'][scope]:
            raise ValueError(f'Frozen expected-service metrics changed: {city}/{seed}/{scope}')
    return dict(city=city, seed=seed, block=block, counts=dict(counts), reconstruction=dict(reconstruction),
        expected_reproduction={scope: True for scope in e.SCOPES}, scopes=scopes,
        calibration={scope: {group: values.result() for group, values in groups.items()}
                     for scope, groups in calibration.items()},
        decompositions={scope: {key: values.result() for key, values in contrasts.items()}
                        for scope, contrasts in decompositions.items()})
