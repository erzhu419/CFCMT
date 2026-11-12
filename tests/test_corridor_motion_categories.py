"""Snapshot motion, exact release matching and comparable endpoint displacement."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_late_motion_20261109.motion_categories import classify_late_motion


def example():
    flight = dict(edge='road', vid='car', entry_s=5, age_s=15,
                  on_straight_lane=True, remaining_m=100.)
    outcome = dict(edge='road', vid='car', entry_s=5, end_s=35, status='straight')
    future = dict(t=30, tls='signal', flights=[dict(flight, age_s=25, remaining_m=70.)],
                  spillback=False, unmatched=False)
    return flight, outcome, future


def test_moving_straight_progress_uses_comparable_endpoints_and_ten_seconds():
    flight, outcome, future = example()
    saved = deepcopy((flight, outcome, future))
    result = classify_late_motion(flight, outcome, 20, future)
    assert result == dict(motion_class='moving_straight_at10', progress_m=30.,
        realized_10s_mean_speed=3., initial_remaining_m=100., future_remaining_m=70.)
    assert (flight, outcome, future) == saved


def test_missing_late_release_means_instantaneous_stop_without_invented_progress():
    flight, outcome, future = example()
    future['flights'] = []
    result = classify_late_motion(flight, outcome, 20, future)
    assert result['motion_class'] == 'stopped_at10'
    assert result['progress_m'] is result['realized_10s_mean_speed'] is result['future_remaining_m'] is None


def test_other_lane_moving_and_unaligned_initial_lane_keep_progress_undefined():
    flight, outcome, future = example()
    future['flights'][0]['on_straight_lane'] = False
    result = classify_late_motion(flight, outcome, 20, future)
    assert result['motion_class'] == 'moving_other_lane_at10'
    assert result['progress_m'] is result['realized_10s_mean_speed'] is None
    flight['on_straight_lane'] = False
    future['flights'][0]['on_straight_lane'] = True
    result = classify_late_motion(flight, outcome, 20, future)
    assert result['motion_class'] == 'moving_straight_at10' and result['progress_m'] is None


def test_full_release_identity_and_future_flags_do_not_filter_snapshot():
    flight, outcome, future = example()
    future.update(spillback=True, unmatched=True)
    assert classify_late_motion(flight, outcome, 20, future)['motion_class'] == 'moving_straight_at10'
    for key, value in [('entry_s', 6), ('edge', 'other_road'), ('vid', 'different')]:
        changed = deepcopy(future)
        changed['flights'][0][key] = value
        assert classify_late_motion(flight, outcome, 20, changed)['motion_class'] == 'stopped_at10'


def test_exact_time_boundary_and_original_label_checks_are_required():
    flight, outcome, future = example()
    with pytest.raises(ValueError, match='after t\\+10'):
        classify_late_motion(flight, dict(outcome, end_s=30), 20, future)
    with pytest.raises(ValueError, match='after t\\+10'):
        classify_late_motion(flight, dict(outcome, status='other'), 20, future)
    with pytest.raises(ValueError, match='row at t\\+10'):
        classify_late_motion(flight, outcome, 20, dict(future, t=40))
    with pytest.raises(ValueError, match='current in-flight release'):
        classify_late_motion(flight, dict(outcome, vid='different'), 20, future)
