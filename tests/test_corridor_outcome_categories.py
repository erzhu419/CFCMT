"""Boundary timing and original detector-label accounting remain unchanged."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_outcome_decomposition_20261109.outcome_categories import classify_outcome


def example(status='straight', lag=10):
    flight = dict(edge='road', vid='car', entry_s=5, age_s=15)
    outcome = dict(edge='road', vid='car', entry_s=5, status=status, end_s=20+lag)
    return flight, outcome


def test_straight_at_ten_is_positive_and_later_detection_is_category_zero():
    assert classify_outcome(*example(lag=10), 20) == (10, 'straight_by10')
    assert classify_outcome(*example(lag=11), 20) == (0, 'straight_after10')
    assert classify_outcome(*example(lag=1), 20) == (1, 'straight_by10')


def test_other_and_terminal_after_ten_are_not_in_window_competing_events():
    for status in ('other', 'terminal'):
        assert classify_outcome(*example(status, 10), 20) == (0, status+'_by10')
        assert classify_outcome(*example(status, 11), 20) == (0, status+'_after10')


def test_complete_censoring_retains_observed_no_arrival_label_without_mutation():
    for lag in (10, 30):
        flight, outcome = example('censored', lag)
        saved = deepcopy((flight, outcome))
        assert classify_outcome(flight, outcome, 20) == (0, 'censored')
        assert (flight, outcome) == saved


def test_identity_age_and_future_endpoint_must_match_the_current_risk_set():
    for key, value in (('vid', 'different'), ('edge', 'other_road'), ('entry_s', 6), ('end_s', 20)):
        flight, outcome = example()
        outcome[key] = value
        with pytest.raises(ValueError, match='current in-flight release'):
            classify_outcome(flight, outcome, 20)
    flight, outcome = example()
    flight['age_s'] = 14
    with pytest.raises(ValueError, match='Label age'):
        classify_outcome(flight, outcome, 20)


def test_unknown_incomplete_censor_and_fractional_positive_keep_old_rejections():
    for status, lag, message in [('unknown', 10, 'Unresolved arrival label'),
                                 ('censored', 9, 'window is censored'),
                                 ('straight', 9.5, 'integer first-observed')]:
        with pytest.raises(ValueError, match=message):
            classify_outcome(*example(status, lag), 20)
