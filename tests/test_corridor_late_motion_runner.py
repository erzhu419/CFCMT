"""Keep future blocked rows when splitting the original late-flight cohort."""
from pathlib import Path

from experiments.tsc_corridor_state_stratification_20261109 import run_stratification as state
from experiments.tsc_corridor_outcome_decomposition_20261109 import run_decomposition as outcome
from experiments.tsc_corridor_late_motion_20261109 import run_motion as motion


def test_future_spillback_does_not_hide_a_moving_late_vehicle(monkeypatch):
    context = (True, 1, '(0.5,0.9]', False)
    flights = [dict(edge='road', vid=vid, entry_s=0, age_s=0,
                    remaining_m=100, on_straight_lane=True) for vid in ('stopped', 'moving')]
    decisions = []
    for t in range(0, 3600, 10):
        for tls in range(12):
            observed = []
            if tls == 0 and t in (0, 10):
                observed = [dict(f, age_s=t, remaining_m=100-t) for f in flights]
            elif tls == 0 and t == 20:
                observed = [dict(flights[1], age_s=20, remaining_m=88)]
            decisions.append(dict(t=t, tls=str(tls), states=['Gr', 'rG'], pp=1, gp=0,
                flights=observed, spillback=t in (0, 20) and tls == 0,
                unmatched=t == 20 and tls == 0, protected={}))
    labels = [dict(edge='road', vid=f['vid'], entry_s=0, end_s=35, status='straight')
              for f in flights]
    def feature(flight, t, history):
        assert t == 10 and history['road', flight['vid'], 0] == (0, 100, True)
        return context, .8
    def recovery(row, previous):
        assert previous == 'rG'
        return dict(identified=True, by_edge={'road': {'stratum': ['protected', 'zero']}})
    for module in (state, outcome, motion):
        monkeypatch.setattr(module, 'rows', lambda path: iter(labels if path.name.startswith('labels') else decisions))
        monkeypatch.setattr(module, 'feature_context', feature)
        monkeypatch.setattr(module, 'recover_state_features', recovery)
    base = state.StratificationStats([context])
    receipt = state.process_hour(Path('/unused'), 'jinan', 'donor', 1, None, 'source', base, 2)
    old = outcome.OutcomeStats([context])
    receipt = outcome.process_hour(Path('/unused'), receipt, old)
    previous = old.result(base.result())
    stats = motion.MotionStats([context])
    receipt = motion.process_hour(Path('/unused'), receipt, stats)
    assert receipt['late_motion_counts'] == dict(stopped_at10=1, moving_straight_at10=1)
    result = stats.result(previous)
    row = result['by_hour'][0]['motions']
    assert row['rates']['stopped_at10'] == row['rates']['moving_straight_at10'] == .5
    assert row['progress']['conditional_mean_progress_m'] == 2
    assert row['progress']['conditional_mean_realized_speed_mps'] == .2
