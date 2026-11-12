"""Keep the old risk set when zero labels have different eventual outcomes."""
from pathlib import Path

from experiments.tsc_corridor_outcome_decomposition_20261109 import run_decomposition as new
from experiments.tsc_corridor_state_stratification_20261109 import run_stratification as old


def test_replay_preserves_old_cohort_and_splits_zero_labels(monkeypatch):
    context = (True, 1, '(0.5,0.9]', False)
    flights = [dict(edge='road', vid=vid, entry_s=0, age_s=0,
                    remaining_m=100, on_straight_lane=True) for vid in ('late', 'other')]
    decisions = [dict(t=t, tls=str(tls), states=['Gr', 'rG'], pp=1, gp=0,
        flights=[dict(f, age_s=t, remaining_m=100-t) for f in flights]
                if tls == 0 and t in (0, 10) else [],
        spillback=t == 0 and tls == 0, unmatched=False, protected={})
        for t in range(0, 3600, 10) for tls in range(12)]
    labels = [dict(edge='road', vid='late', entry_s=0, end_s=25, status='straight'),
              dict(edge='road', vid='other', entry_s=0, end_s=12, status='other')]
    def feature(flight, t, history):
        assert history['road', flight['vid'], 0] == (0, 100, True)
        return context, .8
    def state(row, previous):
        assert row['t'] == 10 and previous == 'rG'
        return dict(identified=True, by_edge={'road': {'stratum': ['protected', 'zero']}})
    for module in (old, new):
        monkeypatch.setattr(module, 'rows', lambda path: iter(labels if path.name.startswith('labels') else decisions))
        monkeypatch.setattr(module, 'feature_context', feature)
        monkeypatch.setattr(module, 'recover_state_features', state)
    previous_stats = old.StratificationStats([context])
    receipt = old.process_hour(Path('/unused'), 'jinan', 'donor', 1, None, 'source', previous_stats, 2)
    stats = new.OutcomeStats([context])
    assert new.process_hour(Path('/unused'), receipt, stats)['outcome_cohort_reproduced']
    result = stats.result(previous_stats.result())
    rates = result['by_hour'][0]['outcomes']['rates']
    assert rates['straight_after10'] == rates['other_by10'] == .5
    assert rates['straight_by10'] == 0
