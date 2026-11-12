"""Expected service choices, frozen hard reproduction and label-only diagnostics."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from experiments.tsc_corridor_expected_service_20261109 import evaluate_expected as e


def write_rows(path, records):
    with gzip.open(path, 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')


def test_fractional_service_can_change_choice_without_changing_endpoint_or_physical_arms():
    row = dict(base=[.3, 0], protected={'road': [0, 1 << 12]}, gp=0)
    hard = np.zeros((5, 14)); hard[:, -1] = 100
    soft = hard.copy(); soft[1, 12] = .4
    before = e.choose_aligned(row, {'road': hard})
    after = e.choose_aligned(row, {'road': soft})
    assert before == dict(T=0, S=0, P=0, history=0, analytic=0, GP=0)
    assert after == dict(T=0, S=1, P=0, history=0, analytic=0, GP=0)


def frozen_hour(tmp_path, monkeypatch):
    def row(t, pp, flights, states=None):
        return dict(t=t, tls='signal', states=states or ['Gr', 'rG'], pp=pp, gp=0,
                    base=[0, 0] if states is None else [0],
                    protected={'road': [0, 0] if states is None else [0]},
                    flights=flights, spillback=False, unmatched=False)
    a = dict(edge='road', vid='a', entry_s=0, age_s=20, tail_s=2.7)
    b = dict(edge='road', vid='b', entry_s=25, age_s=5, tail_s=2.7)
    data = [row(0, 1, []), row(10, 0, []), row(20, 1, [a]), row(30, 0, [b], ['G'])]
    write_rows(tmp_path/'decisions.jsonl.gz', data)
    write_rows(tmp_path/'labels.jsonl.gz', [
        dict(edge='road', vid='a', entry_s=0, end_s=30, status='straight'),
        dict(edge='road', vid='b', entry_s=25, end_s=40, status='straight')])
    order = []
    def extend(row, previous_request, horizon):
        assert previous_request == 'Gr'  # The preceding fallback PP request defines current green.
        assert horizon == 13
        return dict(row, base=[.3, 0], protected={'road': [0, 1 << 12]})
    monkeypatch.setattr(e, 'extend_window', extend)
    choose = e.choose_aligned
    def tracked_choose(*args):
        order.append('choose')
        return choose(*args)
    monkeypatch.setattr(e, 'choose_aligned', tracked_choose)
    label = e.label_service_scores
    def tracked_label(*args):
        assert order == ['choose', 'choose']
        order.append('label')
        return label(*args)
    monkeypatch.setattr(e, 'label_service_scores', tracked_label)
    class Predictor:
        calls = 0
        def forecast(self, row):
            self.calls += 1
            self.hard_forecasts = {}
            if not row['flights']:
                return {}
            hard = np.zeros((5, 14)); hard[:, -1] = 1
            self.hard_forecasts = {'road': hard}
            soft = hard.copy(); soft[1, 12] = .4
            return {'road': soft}
    class OldPolicy:
        def forecast(self, row):
            return {'road': True} if row['t'] == 20 else None
    pairs = {f'{a}_vs_{b}': 0 for a, b in e.PAIRS}
    previous = dict(counts=dict(decisions=4, unlocked_decisions=3, locked_decisions=1,
                               inflight_decisions=2, spillback_decisions=0, unmatched_decisions=0),
        scopes={scope: dict(counts={'scored_edge_epochs': 1 if scope == e.SCOPES[0] else 2},
                            mae={arm: 0.0 for arm in e.ARMS}, action_differences=pairs.copy(),
                            action_differences_unlocked=pairs.copy(),
                            label_score_difference_sums={key: 0.0 for key in e.OLD_CONTRASTS})
                for scope in e.SCOPES})
    predictor = Predictor()
    record = e.evaluate_hour('jinan', 1, 0, tmp_path, predictor, OldPolicy(), previous)
    assert predictor.calls == len(data)
    return record


def test_hour_freezes_both_choices_then_labels_and_keeps_two_scope_hard_reproduction(tmp_path, monkeypatch):
    record = frozen_hour(tmp_path, monkeypatch)
    assert record['reconstruction'] == dict(unlocked_reconstructed=1, locked_structural_zero=1)
    for scope in e.SCOPES:
        assert all(record['old_reproduction'][scope].values())
        result = record['scopes'][scope]
        assert result['scoring_changes'] == dict(T=0, S=1, P=0, history=0, analytic=0)
        assert result['action_differences']['S_vs_T'] == 1
        assert result['action_differences_unlocked']['S_vs_T'] == 1
        assert result['counts']['eligible_S_vs_T_disagreement_epochs'] == 0
        assert result['mae'] == {arm: 0.0 for arm in e.ARMS}
    assert record['scopes'][e.SCOPES[0]]['counts']['scored_edge_epochs'] == 1
    assert record['scopes'][e.SCOPES[1]]['counts']['scored_edge_epochs'] == 2


def test_label_contrasts_and_oracle_use_same_pp_scores_and_all_decision_denominator(tmp_path, monkeypatch):
    record = frozen_hour(tmp_path, monkeypatch)
    for scope in e.SCOPES:
        result = record['scopes'][scope]
        for key in e.CONTRASTS:
            assert result['label_score_difference_sums'][key] == pytest.approx(.7)
            assert result['label_score_differences'][key] == pytest.approx(.7/4)
            assert result['label_score_positive_sums'][key] == pytest.approx(.7)
            assert result['label_score_negative_sums'][key] == 0
            assert result['label_score_difference_signs'][key+'_positive'] == 1
            assert result['label_score_difference_signs'][key+'_zero'] == 3
            assert result['label_score_all_decision_means'][key]['positive'] == pytest.approx(.7/4)
            assert result['label_score_all_decision_means'][key]['positive_frequency'] == .25
        assert result['oracle_regret_soft']['S']['strict_score_gain_sum'] == 0
        assert result['oracle_regret_hard']['S']['strict_score_gain_sum'] == pytest.approx(.7)
        assert result['oracle_regret_hard']['S']['positive_epochs'] == 1
        assert result['oracle_regret_soft']['T']['mean_score_gain_all_decisions'] == pytest.approx(.7/4)


def test_old_reproduction_rejects_label_and_physical_request_drift():
    panel = e._panel()
    panel['counts']['scored_edge_epochs'] = 1
    previous = dict(scopes={'scope': dict(counts={'scored_edge_epochs': 1},
        mae={arm: 0.0 for arm in e.ARMS}, action_differences=panel['old_differences'].copy(),
        action_differences_unlocked=panel['old_unlocked'].copy(),
        label_score_difference_sums={key: 0.0 for key in e.OLD_CONTRASTS})})
    assert all(e.verify_old(panel, previous, 'scope', 'seed').values())
    changed = deepcopy(panel); changed['old_differences']['S_vs_GP'] = 1
    with pytest.raises(ValueError, match='hard thirteen-second requests'):
        e.verify_old(changed, previous, 'scope', 'seed')
    changed = deepcopy(panel); changed['old_label_delta']['S_vs_T'] = 1
    with pytest.raises(ValueError, match='PP-label score differences'):
        e.verify_old(changed, previous, 'scope', 'seed')
    changed = deepcopy(panel); changed['differences']['history_vs_GP'] = 1
    with pytest.raises(ValueError, match='Physical or reference requests'):
        e.verify_old(changed, previous, 'scope', 'seed')


def test_summary_equal_weights_seeds_and_does_not_add_label_or_oracle_gate():
    records = []
    for seed in range(6):
        decisions = 10 if seed == 0 else 100
        panel = e._panel(); panel['counts']['scored_edge_epochs'] = 1
        panel['error'].update(T=2, S=1, P=3, history=4, analytic=5)
        if seed < 2:
            for key in ('S_vs_T', 'S_vs_P'):
                panel['differences'][key] = panel['unlocked'][key] = 1
        for key in e.CONTRASTS:
            panel['label_delta'][key] = panel['label_negative'][key] = -1
            panel['label_signs'][key+'_negative'] = 1
        panel['regrets']['soft']['S'].update(strict_score_gain_sum=10, positive_epochs=1)
        result = e._finish(panel, decisions, decisions)
        records.append(dict(seed=seed, block=seed//2, counts=dict(decisions=decisions, unlocked_decisions=decisions),
                            scopes={scope: deepcopy(result) for scope in e.SCOPES}))
    summary = e.summarize(records)
    for scope in e.SCOPES:
        result = summary['scopes'][scope]
        assert result['better_forecast'] and result['distinct_requests'] and result['passed']
        assert result['label_score_differences']['S_vs_history'] == pytest.approx((-.1-5*.01)/6)
        assert result['label_score_raw_totals']['S_vs_history']['negative'] == -6
        assert result['label_score_all_decision_means']['S_vs_history']['negative'] == pytest.approx((-.1-5*.01)/6)
        assert result['label_score_all_decision_means']['S_vs_history']['negative_frequency'] == pytest.approx((.1+5*.01)/6)
        assert result['oracle_regret_soft']['S']['mean_score_gain_all_decisions'] == pytest.approx((1+5*.1)/6)
        assert result['disagreement_seeds']['S_vs_T'] == [0, 1]
