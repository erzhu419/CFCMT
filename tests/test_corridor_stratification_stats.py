"""Known-state composition, support boundaries and frozen detector CDFs."""
import json

import pytest

from experiments.tsc_corridor_state_stratification_20261109.stratification_stats import StratificationStats

CONTEXT = (True, 3, '<=0.5', False)
A = ('protected', 'zero')
B = ('not_protected', 'positive')
UNKNOWN = ('unknown', 'unknown')


def add(stats, panel, block, state, category, n=1, seed=1, unlocked=True):
    for _ in range(n):
        stats.update(panel, seed, block, CONTEXT, state, category, unlocked)


def context_result(stats, block=0, scope='unlocked'):
    return stats.result()['comparisons'][scope]['by_block'][block]['contexts'][0]


def test_raw_detector_cdf_preserves_category_zero_and_hour_panels():
    stats = StratificationStats([CONTEXT])
    add(stats, 'source', None, A, 0, seed=11)
    add(stats, 'source', None, A, 3, seed=11)
    add(stats, 'local_adapt', 0, A, 10, seed=21)
    add(stats, 'dev', 0, A, 1, seed=31)
    assert not stats.update('dev', 31, 0, (False, 9), A, 1, True)
    result = stats.result()
    source = next(row for row in result['by_hour'] if row['panel'] == 'source')
    assert source['n'] == 2
    assert source['cdf'] == [0, 0]+[.5]*8
    assert source['counts'][0] == source['counts'][3] == 1
    assert {row['panel'] for row in result['by_hour']} == {'source', 'local_adapt', 'dev'}
    assert result['detector_seconds'] == list(range(1, 11))
    json.dumps(result, allow_nan=False)


def test_common_cohort_composition_is_separate_from_unknown_cohort_change():
    stats = StratificationStats([CONTEXT])
    add(stats, 'source', None, A, 1, 9)
    add(stats, 'source', None, B, 0)
    add(stats, 'source', None, UNKNOWN, 0, 10, unlocked=False)
    for panel in ('local_adapt', 'dev'):
        add(stats, panel, 0, A, 1)
        add(stats, panel, 0, B, 0, 9)
    result = context_result(stats)
    source = result['groups']['source']
    assert source['n'] == 20 and source['common_n'] == 10
    assert source['common_support_coverage'] == .5
    assert source['outside_support_n'] == source['unknown_state_n'] == 10
    pair = result['pairs']['source_vs_dev']
    assert pair['full_coarse_gap'] == pytest.approx([.35]*10)
    assert pair['common_unweighted_gap'] == pytest.approx([.8]*10)
    assert pair['standardized_within_gap'] == pytest.approx([0]*10)
    assert pair['composition_gap'] == pytest.approx([.8]*10)
    assert pair['full_to_common_gap_shift'] == pytest.approx([.45]*10)
    for row in result['common_strata']:
        assert row['dev_reference_weight'] == (.1 if row['stratum'] == list(A) else .9)


def test_three_way_support_excludes_missing_states_and_dev_scope_excludes_locked():
    stats = StratificationStats([CONTEXT])
    add(stats, 'source', None, A, 1)
    add(stats, 'source', None, B, 1)
    add(stats, 'local_adapt', 0, A, 0)  # B is absent from the adaptation panel.
    add(stats, 'dev', 0, A, 1)
    add(stats, 'dev', 0, B, 0)
    add(stats, 'dev', 0, UNKNOWN, 1, unlocked=False)
    all_rows, unlocked = context_result(stats, scope='all_observable'), context_result(stats)
    assert [row['stratum'] for row in unlocked['common_strata']] == [list(A)]
    assert all_rows['groups']['dev']['common_support_coverage'] == pytest.approx(1/3)
    assert unlocked['groups']['dev']['common_support_coverage'] == .5
    assert unlocked['groups']['dev']['n'] == 2
    assert unlocked['groups']['source']['n'] == 2
    assert unlocked['pairs']['source_vs_dev']['standardized_within_gap'] == [0]*10
    assert unlocked['pairs']['local_vs_dev']['standardized_within_gap'] == [-1]*10
    empty = StratificationStats([CONTEXT])
    add(empty, 'source', None, A, 1); add(empty, 'local_adapt', 0, B, 0); add(empty, 'dev', 0, A, 1)
    missing = context_result(empty)
    assert missing['common_strata'] == []
    assert missing['groups']['dev']['common_support_coverage'] == 0
    assert missing['pairs']['source_vs_dev']['standardized_within_gap'] is None
    assert missing['pairs']['source_vs_dev']['composition_gap'] is None


def test_block_mean_has_equal_weights_and_remains_distinct_from_pooled_snapshot_cdf():
    stats = StratificationStats([CONTEXT])
    add(stats, 'source', None, A, 1, 2)
    for block in range(3):
        add(stats, 'local_adapt', block, A, 1, seed=10+block)
        add(stats, 'dev', block, A, 0 if block == 0 else 1,
            n=1 if block == 0 else 10, seed=20+block)
    comparisons = stats.result()['comparisons']['unlocked']
    equal = comparisons['equal_block_summary'][0]['pairs']['source_vs_dev']
    pooled = comparisons['pooled'][0]['pairs']['source_vs_dev']
    assert equal['standardized_within_gap'] == pytest.approx([1/3]*10)
    assert pooled['standardized_within_gap'] == pytest.approx([1/21]*10)
    assert equal['blocks_with_common_support'] == 3
    stats.register_hour('dev', 99, 2)  # Preserve a complete hour with zero matching frozen contexts.
    seeds = stats.result()['comparisons']['unlocked']['by_dev_seed']
    missing = next(row for row in seeds if row['seed'] == 99)['contexts'][0]
    assert missing['groups']['source']['n'] == 2
    assert missing['groups']['local_adapt']['n'] == 1
    assert missing['groups']['dev']['n'] == 0
    assert missing['pairs']['source_vs_dev']['standardized_within_gap'] is None
    assert missing['groups']['dev']['common_support_coverage'] is None
    seed_zero = next(row for row in seeds if row['seed'] == 20)['contexts'][0]
    assert seed_zero['pairs']['source_vs_dev']['standardized_within_gap'] == [1]*10
