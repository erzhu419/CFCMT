"""Frozen requests, raw proper scores and unchanged expected-service replay."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from experiments.tsc_corridor_target_shrink_20261109 import evaluate_shrink as e


def write_rows(path, records):
    with gzip.open(path, 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')


def hour(tmp_path, monkeypatch):
    def row(t, pp, flights, locked=False):
        return dict(t=t, tls='signal', states=['G'] if locked else ['Gr', 'rG'], pp=pp, gp=0,
                    base=[0] if locked else [0, 0], protected={'road': [0] if locked else [0, 0]},
                    flights=flights, spillback=False, unmatched=False)
    a = dict(edge='road', vid='a', entry_s=0, age_s=20, tail_s=2.7)
    b = dict(edge='road', vid='b', entry_s=25, age_s=5, tail_s=2.7)
    data = [row(0, 1, []), row(10, 0, []), row(20, 1, [a]), row(30, 0, [b], True)]
    write_rows(tmp_path/'decisions.jsonl.gz', data)
    write_rows(tmp_path/'labels.jsonl.gz', [dict(edge='road', vid='a', entry_s=0, end_s=30, status='straight'),
                                           dict(edge='road', vid='b', entry_s=25, end_s=40, status='straight')])
    def extend(row, previous_request, horizon):
        assert previous_request == 'Gr' and horizon == 13
        return dict(row, base=[.3, 0], protected={'road': [0, 1 << 12]})
    monkeypatch.setattr(e, 'extend_window', extend)
    order = []
    choose, label, truth = e.choose_aligned, e.label_service_scores, e._truth
    def tracked_choose(*args):
        order.append('choose')
        return choose(*args)
    def tracked_label(*args):
        assert order == ['choose', 'choose']
        order.append('label')
        return label(*args)
    def tracked_truth(*args):
        assert order[:3] == ['choose', 'choose', 'label']
        return truth(*args)
    monkeypatch.setattr(e, 'choose_aligned', tracked_choose)
    monkeypatch.setattr(e, 'label_service_scores', tracked_label)
    monkeypatch.setattr(e, '_truth', tracked_truth)
    class Predictor:
        def __init__(self, old):
            self.old, self.calls = old, 0
            self.weights = dict(S=1.0, P=1.0) if old else dict(S=.5, P=.8)
        def forecast(self, row):
            self.calls += 1
            self.raw_detector_cdfs = []
            if not row['flights']:
                return {}
            probabilities = np.array([.6, .2, .1, 0, 0] if self.old else [.6, .4, .2, 0, 0])
            values = np.zeros((5, 14)); values[:, 12] = probabilities; values[:, -1] = probabilities >= .5
            self.raw_detector_cdfs = [(flight, np.repeat(probabilities[:, None], 10, axis=1)) for flight in row['flights']]
            return {'road': values}
    class OldPolicy:
        def forecast(self, row):
            return {'road': True} if row['t'] == 20 else None
    old_actions = dict(T=1, S=0, P=0, history=0, analytic=0, GP=0)
    pairs = {f'{a}_vs_{b}': int(old_actions[a] != old_actions[b]) for a, b in e.PAIRS}
    previous = dict(counts=dict(decisions=4, unlocked_decisions=3, locked_decisions=1,
        inflight_decisions=2, spillback_decisions=0, unmatched_decisions=0), scopes={})
    for scope in e.SCOPES:
        n = 1 if scope == e.SCOPES[0] else 2
        previous['scopes'][scope] = dict(counts=dict(scored_decisions=n, scored_unlocked_decisions=1,
            scored_flights=n, scored_observed_arrivals=n, scored_edge_epochs=n,
            eligible_S_vs_T_disagreement_epochs=0, eligible_S_vs_P_disagreement_epochs=0),
            mae=dict(T=0, S=1, P=1, history=1, analytic=1), action_differences=pairs.copy(),
            action_differences_unlocked=pairs.copy(),
            label_score_difference_sums={key: -.7 if key == 'S_vs_T' else 0 for key in e.CONTRASTS})
    predictor, old = Predictor(False), Predictor(True)
    result = e.evaluate_hour('jinan', 1, 0, tmp_path, predictor, old, OldPolicy(), previous)
    assert predictor.calls == old.calls == len(data)
    return result, previous


def test_requests_are_fixed_before_labels_and_inactive_or_locked_decisions_contribute_zero(tmp_path, monkeypatch):
    record, previous = hour(tmp_path, monkeypatch)
    assert record['weights'] == dict(S=.5, P=.8)
    assert record['scopes'][e.SCOPES[0]]['counts']['scored_flights'] == 1
    assert record['scopes'][e.SCOPES[1]]['counts']['scored_flights'] == 2
    for scope in e.SCOPES:
        result = record['scopes'][scope]
        assert all(record['old_reproduction'][scope].values())
        assert result['label_score_difference_sums']['S_vs_P'] == pytest.approx(.7)
        assert result['label_score_differences']['S_vs_P'] == pytest.approx(.7/4)
        assert result['label_score_difference_signs']['S_vs_P_zero'] == 3
        assert result['scoring_changes'] == dict(T=0, S=1, P=0, history=0, analytic=0, GP=0)
        assert result['old']['action_differences'] == previous['scopes'][scope]['action_differences']


def test_old_reproduction_rejects_mae_requests_or_label_sum_drift(tmp_path, monkeypatch):
    record, previous = hour(tmp_path, monkeypatch)
    actual, expected = record['scopes'][e.SCOPES[0]]['old'], previous['scopes'][e.SCOPES[0]]
    for field in ('mae', 'action_differences', 'label_score_difference_sums'):
        changed = deepcopy(actual)
        key = 'S' if field == 'mae' else 'S_vs_T'
        changed[field][key] += 1
        with pytest.raises(ValueError, match='Old expected-service'):
            e.verify_old(changed, expected, 'seed')


def test_raw_cdf_proper_score_is_separate_from_hard_endpoint_edge_mae(tmp_path, monkeypatch):
    record, previous = hour(tmp_path, monkeypatch)
    for scope in e.SCOPES:
        result = record['scopes'][scope]
        assert result['proper_brier']['S'] == pytest.approx((9*.4**2+.6**2)/10)
        assert result['proper_brier']['T'] == pytest.approx((9*.6**2+.4**2)/10)
        assert result['proper_brier']['S'] < result['proper_brier']['T']
        assert result['mae']['S'] > result['mae']['T']
        assert result['h10_bias']['S'] == pytest.approx(-.6)
        assert result['h10_brier']['S'] == pytest.approx(.36)
        assert result['old']['proper_brier']['S'] == pytest.approx(.1)
        assert result['counts']['scored_query_epochs'] == 10*result['counts']['scored_flights']


def test_summary_uses_equal_hours_and_locked_primary_prediction_and_ranking_gates():
    records = []
    for seed in range(6):
        panel = e._panel(); panel['counts'].update(scored_flights=1 if seed == 0 else 100, scored_edge_epochs=1)
        result = e._finish(panel, 10, 10)
        result.update(proper_brier=dict(T=.2, S=.1 if seed == 0 else .01, P=.3, history=.4, analytic=.5),
                      h10_bias={arm: 0.0 for arm in e.ARMS}, h10_brier={arm: .1 for arm in e.ARMS})
        result['old'] = deepcopy(result)
        for key in ('S_vs_T', 'S_vs_P'):
            result['label_score_differences'][key] = .01
            result['action_differences'][key] = int(seed < 2)
        records.append(dict(seed=seed, block=seed//2, weights=dict(S=0 if seed < 2 else .5, P=0),
                            scopes={scope: deepcopy(result) for scope in e.SCOPES}))
    summary = e.summarize(records)
    primary = summary['scopes']['all_observable_risk']
    assert primary['equal_seed_proper_brier']['S'] == pytest.approx((.1+5*.01)/6)
    assert summary['prediction_pass'] and summary['ranking_pass']
    assert summary['status'] == 'CONTROL_RANKING_DEVELOPMENT_SIGNAL_ONLY'
    assert primary['by_block'][0]['weights'][0]['S'] == 0
    for record in records:
        record['scopes']['all_observable_risk']['label_score_differences']['S_vs_P'] = 0
    assert e.summarize(records)['status'] == 'PREDICTION_ONLY'
    records[0]['scopes']['all_observable_risk']['proper_brier'] = {arm: None for arm in e.ARMS}
    summary = e.summarize(records)
    assert summary['status'] == 'HOLD'
    assert summary['scopes']['all_observable_risk']['empty_probability_seeds'] == [0]
