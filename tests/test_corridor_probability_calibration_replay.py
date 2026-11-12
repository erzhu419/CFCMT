"""Frozen predictions, label ordering and single-pass calibration replay."""
from collections import Counter
from copy import deepcopy
import gzip
from io import StringIO
import json

import numpy as np
import pytest

from experiments.tsc_corridor_probability_calibration_20261109 import replay_calibration as r
from residual_model import HistogramHead


def flight(*, vid='a', remaining=12., entry=0, t=20):
    return dict(edge='road', vid=vid, entry_s=entry, age_s=t-entry, tail_s=2.7,
                on_straight_lane=True, analytic_eta_s=remaining/10., remaining_m=remaining,
                speed_limit_mps=10., queued_at_entry=False)


def row(t=20, flights=None, *, blocked=False, locked=False):
    return dict(t=t, tls='signal', states=['G'] if locked else ['Gr', 'rG'], pp=0, gp=0,
                flights=[flight(t=t)] if flights is None else flights, spillback=blocked,
                unmatched=False, base=[0.] if locked else [0., 0.],
                protected={'road': [0] if locked else [0, 0]})


def heads(*, supported=True):
    context = (True, 11, '<=0.5', False)
    target = HistogramHead({context: [500, 0, 500]+[0]*8})
    source = HistogramHead({context: [470, 0, 530]+[0]*8} if supported else {})
    placebo = HistogramHead({context: [1000]+[0]*10} if supported else {})
    return target, source, placebo


def test_capture_keeps_raw_endpoint_pre_update_context_and_effective_source_weight():
    predictor = r.ReplayPredictor(*heads())
    predictor.forecast(row(10, [flight(remaining=22, t=10)], blocked=True))
    assert predictor.captures == []
    forecast = predictor.forecast(row())
    capture, = predictor.captures
    assert capture['context'] == (True, 11, '<=0.5', False)
    assert capture['eta'] == 12
    assert capture['n_target'] == capture['n_source'] == 1000
    assert capture['source_weight'] == pytest.approx(1000/2020)
    assert capture['raw'][0, -1] == pytest.approx(500/1020)
    assert forecast['road'][0, -1] == 0
    assert capture['target_empirical'][-1] == .5
    assert capture['source_empirical'][-1] == .53
    np.testing.assert_allclose(capture['raw'][1]-capture['raw'][0],
        capture['source_weight']*(capture['source_empirical']-capture['raw'][0]), atol=1e-16)
    assert predictor.history[('road', 'a', 0)] == (20, 12, True)
    predictor.forecast(row(30, [flight(t=30)], blocked=True))
    assert predictor.captures == [] and predictor.hard_forecasts == {}


def test_source_missing_has_zero_weight_and_no_empirical_source():
    predictor = r.ReplayPredictor(*heads(supported=False))
    predictor.forecast(row(10, [flight(remaining=22, t=10)], blocked=True))
    predictor.forecast(row())
    capture, = predictor.captures
    assert capture['source_weight'] == capture['n_source'] == 0
    assert capture['source_empirical'] is None
    np.testing.assert_array_equal(capture['raw'][0], capture['raw'][1])


def _write_rows(path, records):
    with gzip.open(path, 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')


def frozen_fixture(tmp_path, monkeypatch):
    data = [row(0, []), row(10, [flight(remaining=22, t=10)], blocked=True), row(),
            row(30, [flight(vid='b', entry=25, t=30)], locked=True),
            row(40, [flight(t=40)], blocked=True),
            row(50, [flight(vid='c', entry=45, t=50)])]
    labels = [dict(edge='road', vid=vid, entry_s=entry, end_s=end, status=status)
              for vid, entry, end, status in [('a', 0, 30, 'straight'),
                  ('b', 25, 40, 'other'), ('c', 45, 80, 'straight')]]
    _write_rows(tmp_path/'decisions.jsonl.gz', data)
    _write_rows(tmp_path/'labels.jsonl.gz', labels)
    def extend(current, previous, horizon):
        assert horizon == 13 and previous == 'Gr'
        return dict(current, base=[.5, 0.], protected={'road': [0, 1 << 12]})
    monkeypatch.setattr(r.e, 'extend_window', extend)
    monkeypatch.setattr(r.e, 'verify_old', lambda *args: dict(fixture_hard=True))
    class OldPolicy:
        def forecast(self, current):
            return {'road': True} if current['t'] == 20 else None
    counts = Counter()
    for current in data:
        for key, value in dict(decisions=1, unlocked_decisions=len(current['states']) > 1,
            locked_decisions=len(current['states']) == 1, inflight_decisions=bool(current['flights']),
            spillback_decisions=current['spillback'], unmatched_decisions=current['unmatched']).items():
            counts[key] += value
    previous = r.e.evaluate_hour('jinan', 1, 0, tmp_path, r.ExpectedServicePredictor(*heads()),
                                 OldPolicy(), dict(counts=dict(counts)))
    return previous, OldPolicy


def test_single_pass_reproduces_all_metrics_and_uses_raw_perflight_truth_after_choices(tmp_path, monkeypatch):
    previous, OldPolicy = frozen_fixture(tmp_path, monkeypatch)
    reads, choices = [], Counter()
    original_rows, original_choose, original_truth = r.e.rows, r.e.choose_aligned, r._truth
    def tracked_rows(path):
        reads.append(path.name)
        return original_rows(path)
    def choose(current, forecasts):
        choices[current['t']] += 1
        return original_choose(current, forecasts)
    def truth(current, outcome, time_s, bounds):
        assert choices[time_s] == (0 if time_s == 30 else 2)
        return original_truth(current, outcome, time_s, bounds)
    monkeypatch.setattr(r.e, 'rows', tracked_rows)
    monkeypatch.setattr(r.e, 'choose_aligned', choose)
    monkeypatch.setattr(r, '_truth', truth)
    flights, decisions = StringIO(), StringIO()
    result = r.evaluate_hour('jinan', 1, 0, tmp_path, r.ReplayPredictor(*heads()), OldPolicy(),
                             previous, flights, decisions)
    assert reads == ['labels.jsonl.gz', 'decisions.jsonl.gz']
    assert result['scopes'] == previous['scopes']
    assert result['counts'] == previous['counts']
    assert all(result['expected_reproduction'].values())
    details = [json.loads(line) for line in flights.getvalue().splitlines()]
    assert len(details) == 3 and len(decisions.getvalue().splitlines()) == 6
    assert details[0]['truth'] == [0.]*12+[1., 1.]
    assert details[0]['raw'][0][-1] == pytest.approx(500/1020)
    assert details[1]['truth'] == details[2]['truth'] == [0.]*14
    for scope, n, unlocked in [('original_common_scored', 1, 1), ('all_observable_risk', 3, 2)]:
        calibration = result['calibration'][scope]
        assert calibration['all_observable']['arms']['T']['detector_10']['n'] == n
        assert calibration['unlocked']['arms']['T']['detector_10']['n'] == unlocked
        assert calibration['S_vs_T']['arms']['T']['detector_10']['n'] == 1
        assert calibration['S_vs_T_w_positive']['arms']['T']['service_13']['n'] == 1
        assert calibration['S_vs_T_w_positive']['arms']['T']['detector_10']['n'] == 0
        assert calibration['S_vs_T_w_negative']['arms']['T']['service_13']['n'] == 0
        attribution = result['decompositions'][scope]['S_vs_T']
        assert attribution['decisions'] == attribution['by_label_sign']['positive']['decisions'] == 1
        assert attribution['totals']['label_margin'] == .5


def test_replay_rejects_changed_frozen_result(tmp_path, monkeypatch):
    previous, OldPolicy = frozen_fixture(tmp_path, monkeypatch)
    changed = deepcopy(previous)
    changed['scopes']['all_observable_risk']['label_score_positive_sums']['S_vs_T'] += 1
    with pytest.raises(ValueError, match='Frozen expected-service metrics changed'):
        r.evaluate_hour('jinan', 1, 0, tmp_path, r.ReplayPredictor(*heads()), OldPolicy(), changed)
