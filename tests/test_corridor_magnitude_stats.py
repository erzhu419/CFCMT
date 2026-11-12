"""Training-only balanced cuts and fixed-C2 magnitude risk attribution."""
from copy import deepcopy
import json

import pytest

from experiments.tsc_corridor_slowing_magnitude_20261109.magnitude_stats import MagnitudeStats

CONTEXT = (True, 3, '<=0.5', False)
STATE = ('protected', 'zero')


def add(stats, panel, seed, block, value, name='stopped_at10', n=1, trend='slowing'):
    delta = -value if value is not None else -1 if trend == 'slowing' else 0
    slowing = dict(group=trend, reason='available', old_progress_rate=5,
                   recent_progress_rate=5+delta, delta=delta)
    magnitude = dict(value=value, group=trend, reason='available' if value is not None else 'missing_magnitude')
    for _ in range(n):
        stats.update(panel, seed, block, CONTEXT, STATE, 0, name, True,
                     progress_m=10 if name == 'moving_straight_at10' else None,
                     slowing=slowing, magnitude=magnitude)


def training(source, local, copies=1, name='stopped_at10'):
    stats = MagnitudeStats([CONTEXT])
    for value in source:
        add(stats, 'source', 1, None, value, name, n=copies)
    for value in local:
        add(stats, 'local_adapt', 2, 0, value, name)
    return stats


def test_balanced_empirical_quartiles_are_invariant_to_source_sample_replication_and_labels():
    one = training([1, 2], [3, 4]).freeze_training()
    repeated = training([1, 2], [3, 4], copies=30).freeze_training()
    changed_labels = training([1, 2], [3, 4], name='moving_straight_at10').freeze_training()
    assert one['cuts_by_block']['0'] == [1, 2, 3]
    assert repeated['cuts_by_block'] == one['cuts_by_block'] == changed_labels['cuts_by_block']
    assert repeated['train_n_by_panel']['0'] == dict(source=60, local_adapt=2)
    assert one['panel_weights'] == dict(source=.5, local_adapt=.5)


def test_repeated_cuts_merge_and_equality_belongs_to_lower_bin():
    artifact = training([1, 1], [1, 1]).freeze_training()
    assert artifact['cuts_by_block']['0'] == [1]
    assert artifact['cuts_by_block']['1'] is None
    for panel in ('source', 'local_adapt'):
        curve = artifact['training_curves']['0'][panel]
        assert len(curve) == 1
        assert curve[0]['bin'] == 0
        assert curve[0]['n'] == 2
        assert curve[0]['magnitude_mean'] == 1


def test_dev_is_blocked_until_freeze_and_extreme_dev_values_cannot_change_cuts():
    stats = training([1, 2], [3, 4])
    with pytest.raises(ValueError, match='require frozen'):
        add(stats, 'dev', 3, 0, 1000)
    artifact = deepcopy(stats.freeze_training())
    add(stats, 'dev', 3, 0, 1000)
    assert stats.training_artifact == artifact
    with pytest.raises(ValueError, match='cannot change'):
        add(stats, 'local_adapt', 2, 0, .001)


def test_fixed_C2_separates_magnitude_mixture_and_retains_not_slowing_without_a_bin():
    stats = MagnitudeStats([CONTEXT])
    for panel, seed, block, stopped_n, moving_n in (
            ('source', 1, None, 9, 1), ('local_adapt', 2, 0, 1, 9)):
        add(stats, panel, seed, block, 1, n=stopped_n)
        add(stats, panel, seed, block, 3, 'moving_straight_at10', n=moving_n)
        add(stats, panel, seed, block, None, 'moving_straight_at10', trend='not_slowing')
    stats.freeze_training()
    add(stats, 'dev', 3, 0, 1)
    add(stats, 'dev', 3, 0, 3, 'moving_straight_at10', n=9)
    add(stats, 'dev', 3, 0, None, 'moving_straight_at10', trend='not_slowing')
    prior = stats._binary_result()
    result = stats.result(prior)
    row = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    pair = row['pairs']['source_vs_dev']
    assert pair['C1_unweighted'] == pytest.approx(8/11)
    assert pair['cohort_change_unweighted'] == 0
    assert pair['C2_trend_standardized'] == pytest.approx(8/11)
    assert pair['C2_magnitude_standardized'] == 0
    assert pair['magnitude_composition_fixed_C2'] == pytest.approx(8/11)
    assert next(item for item in row['bins'] if item['trend'] == 'not_slowing')['bin'] == 'N/A'
    assert 'pooled' not in result['comparisons']['unlocked']
    changed = deepcopy(prior)
    changed['by_hour'][0]['strata'][0]['counts']['stopped_at10'] += 1
    with pytest.raises(ValueError, match='Previous motion value changed'):
        stats.result(changed)
    json.dumps(result, allow_nan=False)


def test_missing_magnitudes_are_cohort_change_and_empty_seed_is_never_dropped():
    stats = MagnitudeStats([CONTEXT])
    for panel, seed, block, unknown_n in (('source', 1, None, 10), ('local_adapt', 2, 0, 8)):
        add(stats, panel, seed, block, 1)
        add(stats, panel, seed, block, 3, 'moving_straight_at10')
        add(stats, panel, seed, block, None, 'stopped_at10' if panel == 'source' else 'moving_straight_at10', n=unknown_n)
    stats.freeze_training()
    add(stats, 'dev', 3, 0, 1)
    add(stats, 'dev', 3, 0, 3, 'moving_straight_at10')
    add(stats, 'dev', 3, 0, None, 'moving_straight_at10', n=8)
    stats.register_hour('dev', 99, 2)
    result = stats.result(stats._binary_result())
    row = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    assert row['groups']['source']['magnitude_unknown']['n'] == 10
    assert row['groups']['source']['C2_coverage_C1'] == pytest.approx(2/12)
    assert row['pairs']['source_vs_dev']['cohort_change_unweighted'] < 0
    assert row['pairs']['source_vs_dev']['magnitude_composition_fixed_C2'] == 0
    scope = result['comparisons']['unlocked']
    missing = next(seed for seed in scope['by_dev_seed'] if seed['seed'] == 99)
    assert missing['contexts'][0]['pairs']['source_vs_dev']['C2_magnitude_standardized'] is None
    assert scope['equal_seed_summary'][0]['pairs']['source_vs_dev']['C2_magnitude_standardized'] is None
    assert scope['equal_block_summary'][0]['pairs']['source_vs_dev']['C2_magnitude_standardized'] is None
