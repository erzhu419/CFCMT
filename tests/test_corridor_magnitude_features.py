"""Dimensionless causal magnitude and unchanged non-slowing/unknown groups."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_slowing_magnitude_20261109.magnitude_features import normalized_magnitude


def feature(group='slowing', old=8., recent=6., reason='available'):
    return dict(group=group, reason=reason, old_progress_rate=old,
                recent_progress_rate=recent, delta=recent-old if old is not None else None)


def test_magnitude_uses_actual_limit_and_retains_original_rates_without_mutation():
    flight, slowing = dict(speed_limit_mps=10.), feature()
    saved = deepcopy((flight, slowing))
    result = normalized_magnitude(flight, slowing)
    assert result == dict(value=.2, group='slowing', reason='available',
        old_progress_rate=8., recent_progress_rate=6., delta=-2., speed_limit_mps=10.)
    assert (flight, slowing) == saved


def test_proportional_speed_units_produce_the_same_dimensionless_magnitude():
    original = normalized_magnitude(dict(speed_limit_mps=10.), feature())
    scaled = normalized_magnitude(dict(speed_limit_mps=20.), feature(old=16., recent=12.))
    assert original['value'] == scaled['value'] == .2
    assert scaled['old_progress_rate'] == 16. and scaled['delta'] == -4.


def test_not_slowing_and_missing_history_are_retained_without_magnitude():
    for slowing in (feature('not_slowing', old=6., recent=6.),
                    feature('unknown', old=None, recent=6., reason='missing_past20_row')):
        result = normalized_magnitude(dict(speed_limit_mps=10.), slowing)
        assert result['value'] is None
        assert all(result[key] == value for key, value in slowing.items())


def test_slowness_is_never_normalized_with_zero_negative_or_nonfinite_limit():
    for limit in (0., -10., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='actual finite positive speed limit'):
            normalized_magnitude(dict(speed_limit_mps=limit), feature())
