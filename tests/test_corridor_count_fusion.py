"""Count evidence must not treat a sparse source prior as negative data."""
import json

import numpy as np
import pytest

from experiments.tsc_corridor_count_fusion_20261109 import evidence_model as e


def flight(*, eta=1.2, straight=True, queued=False, vid='car'):
    return dict(edge='road', vid=vid, entry_s=0, age_s=20,
                on_straight_lane=straight, analytic_eta_s=eta, remaining_m=10*eta,
                speed_limit_mps=10, queued_at_entry=queued, tail_s=2.7)


def row(t=20, flights=None, spillback=False):
    return dict(t=t, tls='signal', flights=flights if flights is not None else [flight()],
                spillback=spillback, unmatched=False)


def categories(*, absent=0, **events):
    counts = [0]*11
    counts[0] = absent
    for second, count in events.items():
        counts[int(second[1:])] = count
    return counts


def test_three_positive_source_records_raise_local_probability_in_the_observed_bad_context():
    f = flight(eta=8, queued=True)
    history = {('road', 'car', 0): (10, 120, True)}
    context = (True, 11, '<=0.5', True)
    target = e.HistogramHead({context: categories(absent=33, s10=107)})
    source = e.HistogramHead({context: categories(s10=3)})
    predictor = e.EvidencePredictor(target, source, source)
    predictor.history = history
    raw = predictor.flight_cdfs(f, 20, [10])
    assert raw[0, 0] == pytest.approx(107/160)
    assert raw[1, 0] == pytest.approx(110/163)
    assert raw[1, 0] > raw[0, 0] > .5
    old_average = .5*(raw[0, 0] + 3/23)
    assert old_average < .5
    # This source's positive evidence no longer flips a local arrival to non-arrival.
    result = predictor.forecast(row(flights=[f]))['road']
    assert result[0, 10] == result[1, 10] == result[2, 10] == 1


def test_joint_head_has_one_physical_prior_and_preserves_competing_mass():
    f = flight(eta=3.2)
    context, eta = e.feature_context(f, 20, {})
    target = e.HistogramHead({context: categories(absent=10, s2=5)})
    source = e.HistogramHead({context: categories(absent=20, s4=7)})
    joint = e.combine_evidence(target, source)
    bounds = [-1, 0, 1.9, 2, 3.1, 3.2, 4, 10]
    result = joint.cdf(context, eta, True, bounds)
    np.testing.assert_allclose(result, [0, 0, 0, 5/62, 5/62, 25/62, 32/62, 32/62])
    assert np.all(np.diff(result) >= 0)
    assert result[-1] < 1
    old_average = .5*(target.cdf(context, eta, True, bounds)+source.cdf(context, eta, True, bounds))
    assert not np.allclose(result, old_average)


def test_no_source_exactly_preserves_all_target_queries_and_inputs():
    f = flight(eta=3.2)
    context, _ = e.feature_context(f, 20, {})
    counts = categories(absent=10, s2=5)
    target, empty = e.HistogramHead({context: counts}), e.HistogramHead()
    saved = target.counts[context].copy()
    predictor = e.EvidencePredictor(target, empty, empty)
    raw = predictor.flight_cdfs(f, 20, [-1, 0, 1.2, 2, 3.2, 10])
    np.testing.assert_array_equal(raw[0], raw[1])
    np.testing.assert_array_equal(raw[0], raw[2])
    np.testing.assert_array_equal(target.counts[context], saved)


def test_target_is_independent_of_source_support_and_values():
    f = flight(eta=8)
    context, _ = e.feature_context(f, 20, {})
    target = e.HistogramHead({context: categories(absent=40, s4=40)})
    empty = e.HistogramHead()
    source = e.HistogramHead({context: categories(s1=1000)})
    original = e.EvidencePredictor(target, empty, empty).forecast(row(flights=[f]))['road']
    changed = e.EvidencePredictor(target, source, source).forecast(row(flights=[f]))['road']
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[3:], changed[3:])


def test_source_can_supply_evidence_when_target_context_is_absent():
    f = flight(eta=12)
    context, _ = e.feature_context(f, 20, {})
    empty = e.HistogramHead()
    source = e.HistogramHead({context: categories(s8=40)})
    raw = e.EvidencePredictor(empty, source, source).flight_cdfs(f, 20, [10])
    assert raw[0, 0] == 0
    assert raw[1, 0] == raw[2, 0] == pytest.approx(40/60)


def test_real_and_permuted_heads_use_the_same_source_budget():
    a = (True, 2, '(0.5,0.9]', False)
    b = (True, 11, 'missing', True)
    target = e.HistogramHead({a: categories(absent=5, s2=10)})
    real = e.HistogramHead({a: categories(s2=20), b: categories(absent=7, s10=3)})
    permuted = e.HistogramHead({a: categories(absent=10, s2=10), b: categories(absent=2, s10=8)})
    predictor = e.EvidencePredictor(target, real, permuted)
    for context in (a, b):
        assert predictor.pooled_source.counts[context].sum() == predictor.pooled_placebo.counts[context].sum()
    assert predictor.pooled_source.metadata['source_snapshot_examples'] == 30
    assert predictor.pooled_placebo.metadata['source_snapshot_examples'] == 30
    json.dumps(predictor.pooled_source.metadata)
    np.testing.assert_array_equal(real.counts[a], categories(s2=20))


def test_original_continuous_tail_decoder_and_excluded_row_history_are_inherited():
    empty = e.HistogramHead()
    predictor = e.EvidencePredictor(empty, empty, empty)
    result = predictor.forecast(row())['road']
    # 1.2 detector ETA + 2.7 tail -> first credited line second4, not5.
    assert np.all(result[:, :3] == 0)
    assert np.all(result[:, 3:] == 1)
    predictor = e.EvidencePredictor(empty, empty, empty)
    assert predictor.forecast(row(10, [flight(eta=16)], spillback=True)) == {}
    f = flight(eta=8)
    context, eta = e.feature_context(f, 20, predictor.history)
    assert eta == 10  # the excluded t=10 snapshot remained in the past state
    assert context[2] == '(0.5,0.9]'
