"""Outcome mass closure and unchanged state/CDF support for the frozen panel."""
from copy import deepcopy
import json

import pytest

from experiments.tsc_corridor_outcome_decomposition_20261109.outcome_stats import OUTCOMES, OutcomeStats
from experiments.tsc_corridor_state_stratification_20261109.stratification_stats import StratificationStats

CONTEXT = (True, 3, '<=0.5', False)
A = ('protected', 'zero')
B = ('not_protected', 'positive')
UNKNOWN = ('unknown', 'unknown')


def pair():
    return OutcomeStats([CONTEXT]), StratificationStats([CONTEXT])


def add(both, panel, seed, block, outcome, state=A, unlocked=True, n=1, category=None):
    category = (3 if outcome == 'straight_by10' else 0) if category is None else category
    for _ in range(n):
        both[0].update(panel, seed, block, CONTEXT, state, category, outcome, unlocked)
        both[1].update(panel, seed, block, CONTEXT, state, category, unlocked)


def test_all_seven_probabilities_close_and_detector_zero_gap_decomposes():
    both = pair()
    for outcome in OUTCOMES:
        add(both, 'source', 1, None, outcome)
        for block in range(3):
            add(both, 'local_adapt', 10+block, block, outcome)
    for block in range(3):
        add(both, 'dev', 20+block, block, 'straight_by10', n=7)
    result = both[0].result(both[1].result())
    assert result['old_reproduction']['raw_counts_and_hour_cdfs']
    source_hour = next(row for row in result['by_hour'] if row['panel'] == 'source')
    assert source_hour['outcomes']['counts'] == {name: 1 for name in OUTCOMES}
    for scope in ('all_observable', 'unlocked'):
        row = result['comparisons'][scope]['by_block'][0]['contexts'][0]
        source = row['groups']['source']['outcomes']
        assert sum(source['full']['rates'].values()) == pytest.approx(1)
        assert source['standardized']['rates'] == pytest.approx({name: 1/7 for name in OUTCOMES})
        gap = row['pairs']['source_vs_dev']['outcome_gaps']['standardized_within']
        assert gap['straight_by10'] == pytest.approx(-6/7)
        assert all(gap[name] == pytest.approx(1/7) for name in OUTCOMES[1:])
        assert row['pairs']['source_vs_dev']['zero_category_decomposition']['standardized_within'][
            'opposite_zero_gap_sum'] == pytest.approx(-6/7)
        equal = result['comparisons'][scope]['equal_block_summary'][0]['pairs']['source_vs_dev']
        assert equal['outcome_gaps']['standardized_within'] == pytest.approx(gap)
    json.dumps(result, allow_nan=False)


def test_unknown_cohort_and_missing_support_are_retained_without_zero_filling():
    both = pair()
    add(both, 'source', 1, None, 'straight_by10', n=3)
    add(both, 'source', 1, None, 'censored', state=UNKNOWN, unlocked=False)
    add(both, 'local_adapt', 10, 0, 'straight_after10')
    add(both, 'dev', 20, 0, 'other_after10')
    add(both, 'dev', 20, 0, 'terminal_by10', state=UNKNOWN, unlocked=False)
    add(both, 'local_adapt', 11, 1, 'terminal_after10', state=B)
    add(both, 'dev', 21, 1, 'straight_by10')  # No three-way known support in this block.
    for stats in both:
        stats.register_hour('dev', 99, 2)
    result = both[0].result(both[1].result())
    all_row = result['comparisons']['all_observable']['by_block'][0]['contexts'][0]
    unlocked = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    assert all_row['groups']['source']['common_support_coverage'] == .75
    assert all_row['groups']['dev']['outcomes']['full']['rates']['terminal_by10'] == .5
    assert unlocked['groups']['dev']['outcomes']['full']['rates']['terminal_by10'] == 0
    assert unlocked['groups']['source']['outcomes']['standardized']['rates']['straight_by10'] == 1
    missing = result['comparisons']['unlocked']['by_block'][1]['contexts'][0]
    assert missing['groups']['source']['outcomes']['standardized']['rates'] is None
    assert missing['pairs']['source_vs_dev']['outcome_gaps']['standardized_within'] is None
    seed = next(row for row in result['comparisons']['unlocked']['by_dev_seed'] if row['seed'] == 99)
    assert seed['contexts'][0]['groups']['dev']['outcomes']['full']['rates'] is None
    assert result['comparisons']['unlocked']['equal_block_summary'][0]['pairs'][
        'source_vs_dev']['outcome_gaps']['standardized_within'] is None


def test_old_raw_counts_and_hour_cdfs_are_exact_but_standardized_roundoff_is_allowed():
    both = pair()
    for panel, seed, block in (('source', 1, None), ('local_adapt', 2, 0), ('dev', 3, 0)):
        add(both, panel, seed, block, 'straight_by10')
    previous = both[1].result()
    tolerant = deepcopy(previous)
    tolerant['comparisons']['unlocked']['by_block'][0]['contexts'][0]['groups']['source'][
        'standardized_cdf'][0] += 5e-13
    both[0].result(tolerant)
    changed = deepcopy(previous); changed['by_hour'][0]['counts'][0] += 1
    with pytest.raises(ValueError, match='Previous stratification value changed'):
        both[0].result(changed)
    changed = deepcopy(previous); changed['by_hour'][0]['cdf'][0] += 5e-13
    with pytest.raises(ValueError, match='Previous stratification value changed'):
        both[0].result(changed)


def test_outcome_and_old_positive_detector_category_must_agree():
    stats, old = pair()
    with pytest.raises(ValueError, match='original detector category'):
        stats.update('source', 1, None, CONTEXT, A, 0, 'straight_by10', True)
    with pytest.raises(ValueError, match='original detector category'):
        stats.update('source', 1, None, CONTEXT, A, 2, 'other_by10', True)
