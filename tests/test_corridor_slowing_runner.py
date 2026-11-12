"""Past blocked rows remain observable history; future labels cannot set trend."""
from pathlib import Path

from experiments.tsc_corridor_state_stratification_20261109 import run_stratification as state
from experiments.tsc_corridor_outcome_decomposition_20261109 import run_decomposition as outcome
from experiments.tsc_corridor_late_motion_20261109 import run_motion as motion
from experiments.tsc_corridor_predecision_slowing_20261109 import run_slowing as slowing


def test_past_twenty_seconds_determine_trend_before_outcomes(monkeypatch):
    context = (True, 1, '(0.5,0.9]', False)
    distances = {'braking': (141, 61, 1), 'steady': (121, 61, 1)}
    decisions = []
    for t in range(0, 3600, 10):
        for tls in range(12):
            flights = [dict(edge='road', vid=vid, entry_s=0, age_s=t,
                remaining_m=positions[t//10], on_straight_lane=True, queued_at_entry=False,
                speed_limit_mps=10, analytic_eta_s=positions[t//10]/10)
                for vid, positions in distances.items()] if tls == 0 and t in (0, 10, 20) else []
            decisions.append(dict(t=t, tls=str(tls), states=['Gr', 'rG'], pp=1, gp=0,
                flights=flights, spillback=t in (0, 10, 30) and tls == 0,
                unmatched=False, protected={}))
    labels = [dict(edge='road', vid='braking', entry_s=0, end_s=35, status='straight'),
              dict(edge='road', vid='steady', entry_s=0, end_s=25, status='straight')]
    def recovery(row, previous):
        assert row['t'] == 20 and previous == 'rG'
        return dict(identified=True, by_edge={'road': {'stratum': ['protected', 'zero']}})
    for module in (state, outcome, motion, slowing):
        monkeypatch.setattr(module, 'rows', lambda path: iter(labels if path.name.startswith('labels') else decisions))
        monkeypatch.setattr(module, 'recover_state_features', recovery)
    base = state.StratificationStats([context])
    receipt = state.process_hour(Path('/unused'), 'jinan', 'donor', 1, None, 'source', base, 2)
    events = outcome.OutcomeStats([context])
    receipt = outcome.process_hour(Path('/unused'), receipt, events)
    previous7 = events.result(base.result())
    moves = motion.MotionStats([context])
    receipt = motion.process_hour(Path('/unused'), receipt, moves)
    previous9 = moves.result(previous7)
    observed = []
    causal, classify = slowing.causal_slowing, slowing.classify_outcome
    def record_feature(*args):
        feature = causal(*args)
        observed.append((args[0]['vid'], feature))
        return feature
    def check_label(flight, *args):
        assert observed[-1][0] == flight['vid']
        return classify(flight, *args)
    monkeypatch.setattr(slowing, 'causal_slowing', record_feature)
    monkeypatch.setattr(slowing, 'classify_outcome', check_label)
    stats = slowing.SlowingStats([context])
    receipt = slowing.process_hour(Path('/unused'), receipt, stats)
    assert receipt['slowing_counts'] == dict(slowing=1, not_slowing=1)
    assert [row['delta'] for _, row in observed] == [-2, 0]
    assert stats.result(previous9)['reproduction']['previous_motion_panel']
