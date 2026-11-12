"""Fixed-cohort trend composition and unchanged nine-outcome evidence."""
from copy import deepcopy
import json

import pytest

from experiments.tsc_corridor_predecision_slowing_20261109.slowing_stats import SlowingStats
from experiments.tsc_corridor_late_motion_20261109.motion_stats import LATE_MOTIONS, MotionStats
from experiments.tsc_corridor_outcome_decomposition_20261109.outcome_stats import OutcomeStats
from experiments.tsc_corridor_state_stratification_20261109.stratification_stats import StratificationStats

CONTEXT = (True, 3, '<=0.5', False)
A, B = ('protected', 'zero'), ('not_protected', 'positive')


def recorders():
    return SlowingStats([CONTEXT]), MotionStats([CONTEXT]), OutcomeStats([CONTEXT]), StratificationStats([CONTEXT])


def add(stats, panel, seed, block, trend, name, state=A, n=1, reason='available'):
    feature = dict(group=trend, reason=reason, old_progress_rate=5 if trend != 'unknown' else None,
                   recent_progress_rate=3 if trend == 'slowing' else 5 if trend == 'not_slowing' else None,
                   delta=-2 if trend == 'slowing' else 0 if trend == 'not_slowing' else None)
    outcome = 'straight_after10' if name in LATE_MOTIONS else name
    category = 2 if name == 'straight_by10' else 0
    progress = 10 if name == 'moving_straight_at10' else None
    for _ in range(n):
        stats[0].update(panel, seed, block, CONTEXT, state, category, name, True, progress, feature)
        stats[1].update(panel, seed, block, CONTEXT, state, category, name, True, progress)
        stats[2].update(panel, seed, block, CONTEXT, state, category, outcome, True)
        stats[3].update(panel, seed, block, CONTEXT, state, category, True)


def previous(stats):
    return stats[1].result(stats[2].result(stats[3].result()))


def test_fixed_joint_cohort_separates_trend_composition_and_retains_feature_moments():
    stats = recorders()
    add(stats, 'source', 1, None, 'slowing', 'stopped_at10', n=9)
    add(stats, 'source', 1, None, 'not_slowing', 'moving_straight_at10')
    for panel, seed in (('local_adapt', 2), ('dev', 3)):
        add(stats, panel, seed, 0, 'slowing', 'stopped_at10')
        add(stats, panel, seed, 0, 'not_slowing', 'moving_straight_at10', n=9)
    result = stats[0].result(previous(stats))
    assert result['reproduction']['previous_motion_panel']
    row = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    pair = row['pairs']['source_vs_dev']
    assert pair['original_common'] == pytest.approx(.8)
    assert pair['cohort_change'] == 0
    assert pair['joint_state_only_standardized'] == pytest.approx(.8)
    assert pair['joint_standardized'] == 0
    assert pair['trend_composition_fixed_joint_cohort'] == pytest.approx(.8)
    slowed = row['original_common_states'][0]['panels']['source']['trends']['slowing']
    assert slowed['mixture_fraction'] == .9
    assert slowed['stopped_at10_rate'] == 1
    assert slowed['progress_features']['stopped_at10']['delta'] == dict(n=9, sum=-18, mean=-2)
    source_hour = next(item for item in result['by_hour'] if item['panel'] == 'source')
    assert next(item for item in source_hour['strata'] if item['trend'] == 'slowing')['mixture_fraction'] == .9
    assert 'groups' not in result  # The old nested CDF tables are not copied into the compact report.
    json.dumps(result, allow_nan=False)


def test_unknown_cohort_loss_is_not_attributed_to_known_trend_composition():
    stats = recorders()
    add(stats, 'source', 1, None, 'slowing', 'stopped_at10', n=5)
    add(stats, 'source', 1, None, 'not_slowing', 'moving_straight_at10', n=5)
    add(stats, 'source', 1, None, 'unknown', 'stopped_at10', n=10, reason='not_listed_moving_at20')
    for panel, seed in (('local_adapt', 2), ('dev', 3)):
        add(stats, panel, seed, 0, 'slowing', 'stopped_at10')
        add(stats, panel, seed, 0, 'not_slowing', 'moving_straight_at10')
        add(stats, panel, seed, 0, 'unknown', 'moving_straight_at10', n=8, reason='not_yet_entered_at20')
    for item in stats:
        item.register_hour('dev', 99, 2)
    result = stats[0].result(previous(stats))
    row = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    source = row['groups']['source']
    assert source['joint_coverage_original_common'] == .5
    assert source['unknown']['counts']['stopped_at10'] == 10
    assert source['unknown']['reason_counts'] == dict(not_listed_moving_at20=10)
    pair = row['pairs']['source_vs_dev']
    assert pair['cohort_change'] == pytest.approx(-.65)
    assert pair['trend_composition_fixed_joint_cohort'] == 0
    assert pair['joint_standardized'] == 0
    missing = next(item for item in result['comparisons']['unlocked']['by_dev_seed'] if item['seed'] == 99)
    assert missing['contexts'][0]['groups']['dev']['joint_standardized_rates'] is None
    assert missing['contexts'][0]['pairs']['source_vs_dev']['joint_standardized'] is None


def test_state_only_standardization_prevents_state_mix_from_becoming_trend_explanation():
    stats = recorders()
    add(stats, 'source', 1, None, 'slowing', 'stopped_at10', n=9)
    add(stats, 'source', 1, None, 'slowing', 'moving_straight_at10', state=B)
    for panel, seed in (('local_adapt', 2), ('dev', 3)):
        add(stats, panel, seed, 0, 'slowing', 'stopped_at10')
        add(stats, panel, seed, 0, 'slowing', 'moving_straight_at10', state=B, n=9)
    row = stats[0].result(previous(stats))['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    pair = row['pairs']['source_vs_dev']
    assert pair['original_common'] == pytest.approx(.8)
    assert pair['joint_state_only_standardized'] == 0
    assert pair['joint_standardized'] == 0
    assert pair['trend_composition_fixed_joint_cohort'] == 0


def test_changed_previous_motion_result_is_rejected_before_compact_tables():
    stats = recorders()
    for panel, seed, block in (('source', 1, None), ('local_adapt', 2, 0), ('dev', 3, 0)):
        add(stats, panel, seed, block, 'unknown', 'stopped_at10', reason='missing_past20_row')
    old = previous(stats)
    changed = deepcopy(old)
    changed['by_hour'][0]['motions']['counts']['stopped_at10'] += 1
    with pytest.raises(ValueError, match='Previous motion value changed'):
        stats[0].result(changed)
