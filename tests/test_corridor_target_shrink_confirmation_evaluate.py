"""Fresh Hangzhou admission without claiming historical development replay."""
import json

import numpy as np
import pytest

from experiments.tsc_corridor_target_shrink_20261109 import evaluate_shrink as e


class Predictor:
    def __init__(self, old=False, changed_target=False):
        self.old, self.changed_target = old, changed_target
        self.weights = dict(S=1, P=1) if old else dict(S=.1, P=.07)

    def forecast(self, row):
        self.raw_detector_cdfs = []
        if not row['flights']:
            return {}
        probabilities = np.array([.2, .6, .8, 0, 0] if self.old else [.2, .3, .4, 0, 0])
        if self.changed_target:
            probabilities[0] = .25
        values = np.zeros((5, 14)); values[:, :13] = probabilities[:, None]
        values[:, -1] = probabilities >= .5
        self.raw_detector_cdfs = [(flight, np.repeat(probabilities[:, None], 10, axis=1)) for flight in row['flights']]
        return {'road': values}


class Policy:
    def forecast(self, row):
        return {'road': True} if row['flights'] else None


def setup_hour(tmp_path, monkeypatch):
    decisions = [dict(t=t, tls=f'tls{tls}', states=['G'], pp=0, gp=0, base=[0],
        protected={'road': [0]}, flights=[], spillback=False, unmatched=False)
        for t in range(0, 3600, 10) for tls in range(16)]
    decisions[32]['flights'] = [dict(edge='road', vid='a', entry_s=0, age_s=20, tail_s=2.7)]
    labels = [dict(edge='road', vid='a', entry_s=0, end_s=30, status='straight')]
    receipt = dict(status='PASS_HOUR', city='hangzhou', seed=76113701, elapsed_sec=3600,
        horizon_sec=3600, decision_interval_sec=10,
        counts=dict(tls_epochs=5760, controllable_epochs=0, flight_epochs=1, spillback_epochs=0, unmatched_epochs=0))
    (tmp_path/'result.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(e, 'rows', lambda path: iter(labels if path.name == 'labels.jsonl.gz' else decisions))
    return decisions, receipt


def evaluate(tmp_path, changed_target=False):
    return e.evaluate_hour('hangzhou', 76113701, 0, tmp_path,
                           Predictor(changed_target=changed_target), Predictor(old=True), Policy(), None)


def test_complete_fresh_roster_is_admitted_without_fake_old_reproduction(tmp_path, monkeypatch):
    setup_hour(tmp_path, monkeypatch)
    result = evaluate(tmp_path)
    assert result['old_reproduction'] is None
    assert result['new_panel_admission'] == dict(collector_receipt=True, complete_tls_roster=True,
        decision_counts=True, tls_count=16, decision_epochs=5760)
    assert result['counts']['decisions'] == result['counts']['locked_decisions'] == 5760
    for scope in e.SCOPES:
        panel = result['scopes'][scope]
        assert panel['counts']['scored_flights'] == 1
        assert panel['proper_brier']['S'] == pytest.approx(.13)
        assert panel['old']['proper_brier']['S'] == pytest.approx(.34)
        assert panel['label_score_difference_sums']['S_vs_T'] == 0


def test_fresh_roster_clock_and_collector_count_drift_are_rejected(tmp_path, monkeypatch):
    decisions, receipt = setup_hour(tmp_path, monkeypatch)
    decisions[-1]['t'] = 3580
    with pytest.raises(ValueError, match='TLS roster'):
        evaluate(tmp_path)
    decisions[-1]['t'] = 3590
    receipt['counts']['tls_epochs'] -= 1
    (tmp_path/'result.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='decision counts differ'):
        evaluate(tmp_path)
    receipt['counts']['tls_epochs'] += 1
    receipt['elapsed_sec'] = 3599
    (tmp_path/'result.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='collector receipt'):
        evaluate(tmp_path)


def test_fresh_panel_still_enforces_perrow_unchanged_target_and_physical_predictions(tmp_path, monkeypatch):
    setup_hour(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Pure-target or physical service forecast changed'):
        evaluate(tmp_path, changed_target=True)
