"""Unchanged magnitude cuts and fixed-cohort PP-plan composition."""
from copy import deepcopy
import json

import pytest

from experiments.tsc_corridor_pp_plan_20261109.plan_stats import PlanStats
from experiments.tsc_corridor_slowing_magnitude_20261109.magnitude_stats import MagnitudeStats

CONTEXT = (True, 3, '<=0.5', False)
STATE = ('protected', 'zero')


def recorders():
    return PlanStats([CONTEXT]), MagnitudeStats([CONTEXT])


def add(stats, panel, seed, block, seconds, name='stopped_at10', n=1):
    slowing = dict(group='slowing', reason='available', old_progress_rate=5, recent_progress_rate=4, delta=-1)
    magnitude = dict(value=1, group='slowing', reason='available')
    kwargs = dict(progress_m=10 if name == 'moving_straight_at10' else None, slowing=slowing, magnitude=magnitude)
    for _ in range(n):
        stats[0].update(panel, seed, block, CONTEXT, STATE, 0, name, True,
                        plan=dict(protected_seconds=seconds) if seconds is not None else None, **kwargs)
        stats[1].update(panel, seed, block, CONTEXT, STATE, 0, name, True, **kwargs)


def freeze(stats):
    artifact = stats[1].freeze_training()
    reproduced = stats[0].freeze_training(artifact)
    assert reproduced['previous_training_reproduced']
    assert reproduced['training'] == artifact
    return artifact


def previous(stats):
    return stats[1].result(stats[1]._binary_result())


def test_fixed_C3_plan_mixture_explains_gap_without_changing_bins_or_cohort():
    stats = recorders()
    add(stats, 'source', 1, None, 0, n=9)
    add(stats, 'source', 1, None, 10, 'moving_straight_at10')
    add(stats, 'local_adapt', 2, 0, 0)
    add(stats, 'local_adapt', 2, 0, 10, 'moving_straight_at10', n=9)
    assert freeze(stats)['cuts_by_block']['0'] == [1]
    add(stats, 'dev', 3, 0, 0)
    add(stats, 'dev', 3, 0, 10, 'moving_straight_at10', n=9)
    result = stats[0].result(previous(stats))
    row = result['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    pair = row['pairs']['source_vs_dev']
    assert pair['C2_unweighted'] == pytest.approx(.8)
    assert pair['cohort_change_unweighted'] == 0
    assert pair['C3_cell_standardized'] == pytest.approx(.8)
    assert pair['C3_plan_standardized'] == 0
    assert pair['plan_composition_fixed_C3'] == pytest.approx(.8)
    no_green = next(plan for plan in row['plans'] if plan['protected_seconds'] == 0)
    assert no_green['common_C3']
    assert no_green['panels']['source']['share'] == .9
    assert no_green['panels']['source']['magnitude_mean'] == 1
    assert no_green['panels']['source']['stopped_at10_rate'] == 1
    assert 'pooled' not in result['comparisons']['unlocked']
    json.dumps(result, allow_nan=False)


def test_unknown_plan_is_cohort_loss_and_does_not_become_plan_composition():
    stats = recorders()
    for panel, seed, block, unknown_n in (('source', 1, None, 10), ('local_adapt', 2, 0, 8)):
        add(stats, panel, seed, block, 0)
        add(stats, panel, seed, block, 10, 'moving_straight_at10')
        add(stats, panel, seed, block, None,
            'stopped_at10' if panel == 'source' else 'moving_straight_at10', n=unknown_n)
    freeze(stats)
    add(stats, 'dev', 3, 0, 0)
    add(stats, 'dev', 3, 0, 10, 'moving_straight_at10')
    add(stats, 'dev', 3, 0, None, 'moving_straight_at10', n=8)
    row = stats[0].result(previous(stats))['comparisons']['unlocked']['by_block'][0]['contexts'][0]
    assert row['groups']['source']['unknown_plan']['n'] == 10
    assert row['groups']['source']['C3_coverage_C2'] == pytest.approx(2/12)
    assert row['pairs']['source_vs_dev']['cohort_change_unweighted'] < 0
    assert row['pairs']['source_vs_dev']['plan_composition_fixed_C3'] == 0


def test_missing_three_way_plan_support_and_empty_seeds_stay_missing():
    stats = recorders()
    add(stats, 'source', 1, None, 0)
    add(stats, 'local_adapt', 2, 0, 10, 'moving_straight_at10')
    freeze(stats)
    add(stats, 'dev', 3, 0, 0)
    for item in stats:
        item.register_hour('dev', 99, 2)
    scope = stats[0].result(previous(stats))['comparisons']['unlocked']
    row = scope['by_block'][0]['contexts'][0]
    assert row['groups']['source']['C3_coverage_C2'] == 0
    assert row['pairs']['source_vs_dev']['C3_plan_standardized'] is None
    assert scope['equal_block_summary'][0]['pairs']['source_vs_dev']['C3_plan_standardized'] is None
    assert scope['equal_seed_summary'][0]['pairs']['source_vs_dev']['C3_plan_standardized'] is None
    assert next(seed for seed in scope['by_dev_seed'] if seed['seed'] == 99)['contexts'][0][
        'pairs']['source_vs_dev']['C3_plan_standardized'] is None


def test_changed_training_artifact_or_previous_magnitude_result_is_rejected():
    stats = recorders()
    add(stats, 'source', 1, None, 0); add(stats, 'local_adapt', 2, 0, 0)
    changed = deepcopy(stats[1].freeze_training())
    changed['cuts_by_block']['0'][0] = 2
    with pytest.raises(ValueError, match='Previous motion value changed'):
        stats[0].freeze_training(changed)
    stats = recorders()
    add(stats, 'source', 1, None, 0); add(stats, 'local_adapt', 2, 0, 0)
    freeze(stats); add(stats, 'dev', 3, 0, 0)
    changed = deepcopy(previous(stats))
    changed['by_hour'][0]['strata'][0]['stopped_at10_n'] += 1
    with pytest.raises(ValueError, match='Previous motion value changed'):
        stats[0].result(changed)
