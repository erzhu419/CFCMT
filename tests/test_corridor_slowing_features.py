"""Two past intervals, exact identities, missing histories and the fixed cut."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_predecision_slowing_20261109.slowing_features import causal_slowing


def example(old=8., recent=6.):
    flight = dict(edge='road', vid='car', entry_s=5, age_s=25,
                  on_straight_lane=True, remaining_m=20.)
    past10 = dict(t=20, tls='signal', flights=[dict(flight, remaining_m=20+10*recent)],
                  spillback=True, unmatched=True)
    past20 = dict(t=10, tls='signal', flights=[dict(flight, remaining_m=20+10*(recent+old))])
    return flight, past20, past10


def test_causal_net_progress_detects_slowing_without_mutating_or_filtering_rows():
    flight, past20, past10 = example()
    saved = deepcopy((flight, past20, past10))
    result = causal_slowing(flight, 30, past20, past10)
    assert result == dict(group='slowing', reason='available', old_progress_rate=8.,
        recent_progress_rate=6., delta=-2., current_remaining=20.,
        history20_available=True, history10_available=True)
    assert (flight, past20, past10) == saved


def test_fixed_zero_threshold_keeps_ties_and_increases_not_slowing():
    for old, recent in [(6., 6.), (5., 6.)]:
        flight, past20, past10 = example(old, recent)
        result = causal_slowing(flight, 30, past20, past10)
        assert result['group'] == 'not_slowing' and result['delta'] == recent-old


def test_missing_or_not_yet_entered_history_never_becomes_zero_rate():
    flight, past20, past10 = example()
    result = causal_slowing(flight, 30, None, past10)
    assert result['group'] == 'unknown' and result['reason'] == 'missing_past20_row'
    assert result['old_progress_rate'] is None and result['recent_progress_rate'] == 6.
    assert not result['history20_available'] and result['history10_available']
    flight['entry_s'] = 15
    past10['flights'][0]['entry_s'] = 15
    result = causal_slowing(flight, 30, past20, past10)
    assert result['group'] == 'unknown' and result['reason'] == 'not_yet_entered_at20'
    assert result['old_progress_rate'] is None


def test_full_release_identity_and_all_three_straight_endpoints_are_required():
    for key, value in [('edge', 'other_road'), ('vid', 'different'), ('entry_s', 6)]:
        flight, past20, past10 = example()
        past20['flights'][0][key] = value
        result = causal_slowing(flight, 30, past20, past10)
        assert result['group'] == 'unknown' and result['reason'] == 'not_listed_moving_at20'
    for endpoint in ('current', 'past20', 'past10'):
        flight, past20, past10 = example()
        selected = flight if endpoint == 'current' else (past20 if endpoint == 'past20' else past10)['flights'][0]
        selected['on_straight_lane'] = False
        result = causal_slowing(flight, 30, past20, past10)
        assert result['group'] == 'unknown' and result['reason'] == 'nonstraight_'+endpoint
        assert result['delta'] is None


def test_future_rows_are_rejected_and_nonconsecutive_past_is_unknown():
    flight, past20, past10 = example()
    with pytest.raises(ValueError, match='future row'):
        causal_slowing(flight, 30, None, dict(past10, t=40))
    result = causal_slowing(flight, 30, dict(past20, t=11), past10)
    assert result['group'] == 'unknown' and result['reason'] == 'past20_wrong_time'
    assert result['old_progress_rate'] is None
