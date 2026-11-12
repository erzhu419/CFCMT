"""Physical anchoring, complete-window labels and a strict residual permutation."""
from collections import Counter, defaultdict
import gzip
import json
import random

import numpy as np
import pytest

from experiments.tsc_corridor_residual_transfer_20261109 import residual_model as m


def flight(*, eta=1.2, straight=True, queued=False, vid="car", entry=0):
    return dict(edge="road", vid=vid, entry_s=entry, age_s=20-entry,
                on_straight_lane=straight, analytic_eta_s=eta, remaining_m=10*eta,
                speed_limit_mps=10, queued_at_entry=queued, tail_s=2.7)


def row(t=20, flights=None, **values):
    return dict(t=t, tls="signal", flights=flights if flights is not None else [flight()],
                spillback=values.get("spillback", False), unmatched=False)


def outcome(f, status="straight", end=25):
    return dict(edge=f["edge"], vid=f["vid"], entry_s=f["entry_s"], status=status, end_s=end)


def test_continuous_physical_anchor_and_fractional_tail_are_not_rounded():
    empty = m.HistogramHead()
    predictor = m.ResidualPredictor(empty, empty, empty)
    prediction = predictor.forecast(row())["road"]
    # ETA 1.2 + tail 2.7 = 3.9, so the first credited line second is four, not five.
    assert np.all(prediction[:, :3] == 0)
    assert np.all(prediction[:, 3:] == 1)
    np.testing.assert_array_equal(empty.cdf((True, 2, "missing", False), 1.2, True, [0, 1.19, 1.2]), [0, 0, 1])


def test_conditional_histogram_has_competing_mass_and_monotone_real_queries():
    f = flight(eta=3.2)
    context, eta = m.feature_context(f, 20, {})
    counts = [20, 0, 10, 0, 10, 0, 0, 0, 0, 0, 0]
    head = m.HistogramHead({context: counts})
    bounds = [-1, 0, 1.9, 2, 3.1, 3.2, 4, 10]
    cdf = head.cdf(context, eta, True, bounds)
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[0] == cdf[1] == 0
    assert cdf[3] == pytest.approx(10/60)
    assert cdf[-1] == pytest.approx(40/60)  # no-arrival mass stays outside the window


@pytest.mark.parametrize("status,end,expected", [
    ("straight", 21, 1), ("straight", 30, 10), ("straight", 31, 0),
    ("other", 23, 0), ("terminal", 23, 0), ("censored", 30, 0), ("censored", 3600, 0),
])
def test_label_keeps_competing_and_complete_censored_windows(status, end, expected):
    f = flight()
    assert m.label_category(f, outcome(f, status, end), 20) == expected


def test_incomplete_censored_window_is_not_labeled_zero():
    f = flight()
    with pytest.raises(ValueError, match="censored"):
        m.label_category(f, outcome(f, "censored", 29), 20)


def test_history_only_uses_the_same_release_ten_seconds_earlier():
    f = flight(eta=8)
    history = {m.release_id(f): (10, 130, True)}
    context, eta = m.feature_context(f, 20, history)
    assert eta == 16  # 50m / 10s historical advance gives 5m/s
    assert context == (True, 11, "<=0.5", False)
    history[m.release_id(f)] = (30, 130, True)
    assert m.feature_context(f, 20, history)[1] == 8
    history = {("road", "car", -10): (10, 130, True)}
    assert m.feature_context(f, 20, history)[1] == 8


def test_missing_source_support_preserves_target_and_fixed_median_decoder():
    f = flight(eta=8)
    context, _ = m.feature_context(f, 20, {})
    counts = [0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0]
    target, empty = m.HistogramHead({context: counts}), m.HistogramHead()
    predictor = m.ResidualPredictor(target, empty, empty)
    prediction = predictor.forecast(row(flights=[f]))["road"]
    np.testing.assert_array_equal(prediction[0], prediction[1])
    np.testing.assert_array_equal(prediction[0], prediction[2])
    assert set(np.unique(prediction[:3])) <= {0, 1}
    assert prediction[0, 6] == 1  # stop-line bound 7-2.7 reaches empirical time 4
    assert prediction[3, 6] == 0  # the original physical anchor is still ETA 8


def test_source_changes_do_not_change_pure_target_and_blocked_rows_update_history():
    f = flight(eta=8)
    context, _ = m.feature_context(f, 20, {})
    target, empty = m.HistogramHead(), m.HistogramHead()
    counts = [40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    source = m.HistogramHead({context: counts})
    p0, p1 = m.ResidualPredictor(target, empty, empty), m.ResidualPredictor(target, source, source)
    np.testing.assert_array_equal(p0.forecast(row(flights=[f]))["road"][0], p1.forecast(row(flights=[f]))["road"][0])
    p = m.ResidualPredictor(empty, empty, empty)
    assert p.forecast(row(10, [flight(eta=16)], spillback=True)) == {}
    assert m.feature_context(f, 20, p.history)[1] == 10  # the excluded old snapshot is retained


def test_permutation_preserves_each_physical_stratum_not_only_global_counts():
    low = (True, 2, "(0.5,0.9]", False)
    high = (True, 2, "(0.9,1.1]", True)
    other = (True, 11, "missing", False)
    groups = {
        ("road_a", 2, True): [(low, 1)]*25 + [(high, 0)]*25,
        ("road_b", 11, True): [(other, 0)]*20,
    }
    counts = m._new_counts()
    changed = m._permute_groups(groups, random.Random(m.PERMUTATION_SEED), counts)
    assert changed > 0
    assert sum(counts[low]) == sum(counts[high]) == 25
    assert counts[low][1] + counts[high][1] == 25
    assert counts[other] == [20] + [0]*10
    assert sum(counts[low][1:]) < 25


def write_gzip(path, records):
    with gzip.open(path, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def test_fit_source_uses_original_split_and_never_drops_nonstraight(tmp_path):
    directory = tmp_path / "donor" / "hangzhou" / "123"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps(dict(status="PASS_HOUR", city="hangzhou", seed=123, horizon_sec=3600)))
    a, b = flight(eta=8, vid="a"), flight(eta=8, straight=False, vid="b")
    write_gzip(directory / "decisions.jsonl.gz", [row(flights=[a, b])])
    write_gzip(directory / "labels.jsonl.gz", [outcome(a, "straight", 25), outcome(b, "other", 25)])
    real, placebo = m.fit_source(tmp_path, "donor", "hangzhou", [123])
    assert real.metadata["snapshot_examples"] == 2
    assert real.metadata["positive_arrival_examples"] == real.metadata["zero_label_examples"] == 1
    assert set(real.counts) == set(placebo.counts)
    assert len(real.counts) == 2
    for key in real.counts:
        np.testing.assert_array_equal(real.counts[key], placebo.counts[key])
