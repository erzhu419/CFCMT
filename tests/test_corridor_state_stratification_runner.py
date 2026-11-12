"""Preserve causal history and previous requests outside the selected cohort."""
from pathlib import Path

from experiments.tsc_corridor_state_stratification_20261109 import run_stratification as r


def test_excluded_row_still_updates_history_and_previous_pp(monkeypatch):
    context = (True, 1, '(0.5,0.9]', False)
    first = dict(edge='road', vid='car', entry_s=0, age_s=0,
                 remaining_m=100, on_straight_lane=True)
    second = dict(first, age_s=10, remaining_m=90)
    decisions = [dict(t=t, tls=str(tls), states=['Gr', 'rG'], pp=1, gp=0,
        flights=[first if t == 0 else second] if tls == 0 and t in (0, 10) else [],
        spillback=t == 0 and tls == 0, unmatched=False, protected={})
        for t in range(0, 3600, 10) for tls in range(12)]
    labels = [dict(edge='road', vid='car', entry_s=0, end_s=12, status='straight')]
    monkeypatch.setattr(r, 'rows', lambda path: iter(labels if path.name.startswith('labels') else decisions))
    order = []
    def feature(flight, t, history):
        assert t == 10 and history[('road', 'car', 0)] == (0, 100, True)
        order.append('context')
        return context, .8
    def state(row, previous):
        assert previous == 'rG' and order == ['context']
        order.append('state')
        return dict(identified=True, by_edge={'road': {'stratum': ['protected', 'zero']}})
    category = r.label_category
    def checked_category(*args):
        assert order == ['context', 'state']
        return category(*args)
    monkeypatch.setattr(r, 'feature_context', feature)
    monkeypatch.setattr(r, 'recover_state_features', state)
    monkeypatch.setattr(r, 'label_category', checked_category)
    stats = r.StratificationStats([context])
    receipt = r.process_hour(Path('/unused'), 'jinan', 'donor', 1, None, 'source', stats, 1)
    assert receipt['frozen_roster_reproduced']
    assert receipt['counts']['selected_context_flights'] == 1
    assert next(iter(stats.counts.values())).tolist() == [0, 0, 1]+[0]*8
