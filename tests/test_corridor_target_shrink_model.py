"""Whole-hour calibration weights and unchanged detector/service semantics."""
import gzip
import json

import numpy as np
import pytest

from experiments.tsc_corridor_target_shrink_20261109 import shrink_model as r
from residual_model import HistogramHead


def heads():
    context = (True, 2, 'missing', False)
    return tuple(HistogramHead({context: values}) for values in (
        [700, 0, 300]+[0]*8, [300, 0, 700]+[0]*8, [900, 0, 100]+[0]*8))


def flight(vid='car', remaining=12., time=20):
    return dict(edge='road', vid=vid, entry_s=0, age_s=time, tail_s=2.7,
        on_straight_lane=True, analytic_eta_s=remaining/10, remaining_m=remaining,
        speed_limit_mps=10., queued_at_entry=False)


def test_closed_form_minimizes_brier_and_keeps_both_boundaries_and_zero_support():
    target = np.array([.2, .4, .6])
    difference, truth = np.array([.4, .2, -.2]), np.array([1., 0., 1.])
    a, b, c = np.mean(difference**2), np.mean(difference*(truth-target)), np.mean((truth-target)**2)
    result = r._quadratic_fit(a, b, c)
    assert result['weight'] == pytest.approx(2/3)
    losses = [np.mean((target+weight*difference-truth)**2) for weight in np.linspace(0, 1, 1001)]
    assert result['loss_at_fitted'] <= min(losses)+1e-12
    assert r._quadratic_fit(1., -1., 1.)['weight'] == 0
    assert r._quadratic_fit(1., 2., 4.)['weight'] == 1
    assert r._quadratic_fit(0., 0., .25)['weight'] == 0


def test_fit_block_weights_hours_equally_and_refits_original_full_target(monkeypatch):
    calls = []
    def fit(root, part, city, seeds):
        calls.append((part, list(seeds)))
        return HistogramHead(metadata=dict(city=city, seeds=list(seeds)))
    def evaluate(root, city, seed, target, source, placebo):
        assert target.metadata['seeds'] == [2 if seed == 1 else 1]
        n, positive = (10, 1.) if seed == 1 else (1000, 0.)
        moments = dict(A_sum=float(n), B_sum=n*positive, C_sum=n*positive,
                       A=1., B=positive, C=positive)
        return dict(held_seed=seed, hour_weight=.5, samples=n//10, queries=n,
                    arms={arm: moments.copy() for arm in ('S', 'P')})
    monkeypatch.setattr(r, 'fit_hours', fit)
    monkeypatch.setattr(r, '_evaluate_hour', evaluate)
    full, calibration = r.fit_block('/unused', 'jinan', [2, 1], HistogramHead(), HistogramHead())
    assert calibration['weights'] == dict(S=.5, P=.5)
    assert calls == [('adapt', [1]), ('adapt', [2]), ('adapt', [1, 2])]
    assert full.metadata == calibration['full_target_metadata'] == dict(city='jinan', seeds=[1, 2])
    for arm in ('S', 'P'):
        assert calibration['arms'][arm]['A'] == 1 and calibration['arms'][arm]['B'] == .5
        assert calibration['arms'][arm]['loss_at_fitted'] == .25
        assert np.mean([fold['arms'][arm]['loss_at_fitted'] for fold in calibration['folds']]) == .25


def test_zero_and_one_are_exact_original_cdfs_and_convex_interiors_remain_valid():
    target, source, placebo = heads()
    current, bounds = flight(), np.arange(1., 11.)
    original = r.EvidencePredictor(target, source, placebo).flight_cdfs(current, 20, bounds)
    zero = r.ShrinkPredictor(target, source, placebo, dict(S=0, P=0)).flight_cdfs(current, 20, bounds)
    one = r.ShrinkPredictor(target, source, placebo, dict(S=1, P=1)).flight_cdfs(current, 20, bounds)
    np.testing.assert_array_equal(one, original)
    for index in (1, 2):
        np.testing.assert_array_equal(zero[index], original[0])
    interior = r.ShrinkPredictor(target, source, placebo, dict(S=.5, P=.25)).flight_cdfs(current, 20, bounds)
    np.testing.assert_array_equal(interior[[0, 3, 4]], original[[0, 3, 4]])
    assert np.all((interior >= 0) & (interior <= 1)) and np.all(np.diff(interior, axis=1) >= 0)


def test_forecast_caches_unthresholded_detector_cdfs_and_updates_history_once(monkeypatch):
    predictor = r.ShrinkPredictor(*heads(), dict(S=.5, P=.25))
    current = flight()
    row = dict(t=20, flights=[current], spillback=False, unmatched=False)
    old_update, calls = r.ExpectedServicePredictor.forecast.__globals__['update_history'], []
    def update(current_row, history):
        calls.append(current_row['t'])
        old_update(current_row, history)
    monkeypatch.setitem(r.ExpectedServicePredictor.forecast.__globals__, 'update_history', update)
    output = predictor.forecast(row)
    captured_flight, raw = predictor.raw_detector_cdfs[0]
    assert captured_flight is current and raw.shape == (5, 10)
    assert output['road'].shape == predictor.hard_forecasts['road'].shape == (5, 14)
    assert 0 < raw[1, -1] < .5 and output['road'][1, -1] == 0
    np.testing.assert_array_equal(output['road'][:, -1], predictor.hard_forecasts['road'][:, -1])
    assert predictor.counts['source_supported_flights'] == 1
    predictor.forecast(dict(t=30, flights=[flight(remaining=9., time=30)], spillback=True, unmatched=False))
    assert calls == [20, 30] and predictor.raw_detector_cdfs == [] and predictor.hard_forecasts == {}
    assert predictor.history[('road', 'car', 0)] == (30, 9., True)


def test_whole_hour_folds_predict_all_flights_before_labels_without_held_hour_fit(tmp_path, monkeypatch):
    def write(path, records):
        with gzip.open(path, 'wt') as stream:
            for record in records:
                stream.write(json.dumps(record)+'\n')
    for seed in (1, 2):
        directory = tmp_path/'adapt'/'jinan'/str(seed)
        directory.mkdir(parents=True)
        (directory/'result.json').write_text(json.dumps(dict(status='PASS_HOUR', city='jinan', seed=seed, horizon_sec=30)))
        decisions = [dict(t=t, flights=[flight(f'{seed}:{vid}', 22. if t == 0 else 12., t) for vid in ('a', 'b')],
                          spillback=t == 0, unmatched=False) for t in (0, 10)]
        labels = [dict(edge='road', vid=f'{seed}:{vid}', entry_s=0, end_s=(20 if seed == 1 else 11)
                       if vid == 'a' else 25, status='straight' if vid == 'a' else 'other') for vid in ('a', 'b')]
        write(directory/'decisions.jsonl.gz', decisions)
        write(directory/'labels.jsonl.gz', labels)
    events = []
    original_predict, original_label = r.EvidencePredictor.flight_cdfs, r.label_category
    def predict(predictor, current, time_s, bounds):
        held = int(current['vid'].split(':')[0])
        assert predictor.target.metadata['seeds'] == [2 if held == 1 else 1]
        events.append(('predict', held, current['vid']))
        return original_predict(predictor, current, time_s, bounds)
    def label(current, outcome, time_s):
        held = int(current['vid'].split(':')[0])
        assert {vid for kind, seed, vid in events if kind == 'predict' and seed == held} == {f'{held}:a', f'{held}:b'}
        events.append(('label', held, current['vid']))
        return original_label(current, outcome, time_s)
    monkeypatch.setattr(r.EvidencePredictor, 'flight_cdfs', predict)
    monkeypatch.setattr(r, 'label_category', label)
    full, calibration = r.fit_block(tmp_path, 'jinan', [1, 2], HistogramHead(), HistogramHead())
    expected = r.fit_hours(tmp_path, 'adapt', 'jinan', [1, 2])
    assert full.metadata == expected.metadata == calibration['full_target_metadata']
    for context in expected.counts:
        np.testing.assert_array_equal(full.counts[context], expected.counts[context])
    assert [(fold['training_seed'], fold['held_seed'], fold['samples'], fold['queries'])
            for fold in calibration['folds']] == [(2, 1, 2, 20), (1, 2, 2, 20)]
    assert [kind for kind, _, _ in events] == ['predict', 'predict', 'label', 'label']*2
