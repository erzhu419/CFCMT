"""Nine-to-seven closure and correctly weighted late-vehicle progress."""
from copy import deepcopy
import json

import pytest

from experiments.tsc_corridor_late_motion_20261109.motion_stats import LATE_MOTIONS, MOTIONS, MotionStats
from experiments.tsc_corridor_outcome_decomposition_20261109.outcome_stats import OutcomeStats
from experiments.tsc_corridor_state_stratification_20261109.stratification_stats import StratificationStats

CONTEXT = (True, 3, '<=0.5', False)
A, B = ('protected', 'zero'), ('not_protected', 'positive')


def recorders():
    return MotionStats([CONTEXT]), OutcomeStats([CONTEXT]), StratificationStats([CONTEXT])


def add(stats, panel, seed, block, name, state=A, n=1, progress=None):
    outcome = 'straight_after10' if name in LATE_MOTIONS else name
    category = 2 if name == 'straight_by10' else 0
    for _ in range(n):
        stats[0].update(panel, seed, block, CONTEXT, state, category, name, True, progress)
        stats[1].update(panel, seed, block, CONTEXT, state, category, outcome, True)
        stats[2].update(panel, seed, block, CONTEXT, state, category, True)


def finish(stats):
    previous = stats[1].result(stats[2].result())
    return stats[0].result(previous)


def test_nine_classes_reproduce_all_previous_seven_outcomes_and_gap_identities():
    stats = recorders()
    for name in MOTIONS:
        add(stats, 'source', 1, None, name, progress=20 if name == 'moving_straight_at10' else None)
        for block in range(3):
            add(stats, 'local_adapt', 10+block, block, name)
    for block in range(3):
        add(stats, 'dev', 20+block, block, 'straight_by10', n=9)
    result = finish(stats)
    assert all(result['motion_reproduction'].values())
    for scope in ('all_observable', 'unlocked'):
        row = result['comparisons'][scope]['by_block'][0]['contexts'][0]
        full = row['groups']['source']['motions']['full']
        assert full['counts'] == {name: 1 for name in MOTIONS}
        assert sum(full['rates'].values()) == pytest.approx(1)
        assert row['groups']['source']['outcomes']['full']['counts']['straight_after10'] == 3
        pair = row['pairs']['source_vs_dev']
        late = pair['late_motion_decomposition']['standardized_within']
        assert late['straight_after10_gap'] == pytest.approx(1/3)
        assert late['three_motion_gap_sum'] == pytest.approx(1/3)
        zero = pair['motion_zero_category_decomposition']['standardized_within']
        assert zero['straight_by10_gap'] == pytest.approx(-8/9)
        assert zero['opposite_zero_gap_sum'] == pytest.approx(-8/9)
        equal = result['comparisons'][scope]['equal_block_summary'][0]['pairs']['source_vs_dev']
        assert equal['motion_gaps']['standardized_within'] == pytest.approx(pair['motion_gaps']['standardized_within'])
    json.dumps(result, allow_nan=False)


def test_standardized_progress_weights_moments_before_conditional_division():
    stats = recorders()
    add(stats, 'source', 1, None, 'moving_straight_at10', progress=10)
    add(stats, 'source', 1, None, 'straight_by10')
    add(stats, 'source', 1, None, 'moving_straight_at10', state=B, progress=30)
    add(stats, 'source', 1, None, 'straight_by10', state=B, n=3)
    for panel, seed in (('local_adapt', 2), ('dev', 3)):
        add(stats, panel, seed, 0, 'straight_by10')
        add(stats, panel, seed, 0, 'straight_by10', state=B)
    result = finish(stats)
    group = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]['groups']['source']['motions']
    raw, standard = group['full']['progress'], group['standardized']['progress']
    assert raw['progress_n'] == 2
    assert raw['progress_sum_m'] == 40
    assert raw['conditional_mean_progress_m'] == pytest.approx(20)
    assert standard['moving_straight_probability'] == pytest.approx(.375)
    assert standard['comparable_progress_probability'] == pytest.approx(.375)
    assert standard['progress_m_per_100_eligible'] == pytest.approx(625)
    assert standard['conditional_mean_progress_m'] == pytest.approx(6.25/.375)
    assert standard['conditional_mean_realized_speed_mps'] == pytest.approx((6.25/.375)/10)
    assert standard['conditional_mean_progress_m'] != 20  # Mean of state-conditional means is wrong here.


def test_missing_comparable_progress_and_missing_support_remain_explicit():
    stats = recorders()
    add(stats, 'source', 1, None, 'moving_straight_at10', n=2)
    add(stats, 'source', 1, None, 'moving_straight_at10', progress=12)
    add(stats, 'local_adapt', 2, 0, 'stopped_at10')
    add(stats, 'dev', 3, 0, 'moving_other_lane_at10')
    for item in stats:
        item.register_hour('dev', 99, 2)
    result = finish(stats)
    source = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]['groups']['source']['motions']['standardized']
    assert source['rates']['moving_straight_at10'] == 1
    assert source['progress']['comparable_given_moving'] == pytest.approx(1/3)
    assert source['progress']['conditional_mean_progress_m'] == pytest.approx(12)
    absent = next(row for row in result['comparisons']['unlocked']['by_dev_seed'] if row['seed'] == 99)
    mass = absent['contexts'][0]['groups']['dev']['motions']['standardized']
    assert mass['rates'] is None
    assert mass['progress']['progress_m_per_100_eligible'] is None
    assert mass['progress']['conditional_mean_progress_m'] is None


def test_previous_seven_outcome_drift_is_rejected_before_motion_reporting():
    stats = recorders()
    for panel, seed, block in (('source', 1, None), ('local_adapt', 2, 0), ('dev', 3, 0)):
        add(stats, panel, seed, block, 'stopped_at10')
    previous = stats[1].result(stats[2].result())
    changed = deepcopy(previous)
    changed['by_hour'][0]['outcomes']['counts']['straight_after10'] += 1
    with pytest.raises(ValueError, match='Previous stratification value changed'):
        stats[0].result(changed)


def test_progress_is_never_assigned_to_stopped_or_other_lane_motion():
    stats = MotionStats([CONTEXT])
    for name in ('stopped_at10', 'moving_other_lane_at10'):
        with pytest.raises(ValueError, match='comparable moving-straight'):
            stats.update('source', 1, None, CONTEXT, A, 0, name, True, progress_m=5)
