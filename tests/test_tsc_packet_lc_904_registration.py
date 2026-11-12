"""Pure checks of the 904 registration's new source, weight and reuse rules."""

from collections import Counter
from math import fsum

import pytest

from experiments.tsc_packet_lc_904_registration_20261111.cohort import assign_training_bins, select_cohort
from experiments.tsc_packet_lc_904_registration_20261111.inventory import cost_plan


TRAIN = (76116901, 76116902, 76116903)
REGRESSION = 76116904


def current(direction, speed=10., distance=30., *, present=True):
    return dict(status='RECORDED' if present else 'NO_ADJACENT_CURRENT_STRATUM',
        current_stratum=['lane_a', direction, False, True, ['g'], True, True] if present else None,
        speed_mps=speed, lane_end_distance_m=distance)


def row(seed, time, label, left=None, right=None):
    return dict(seed=seed, time_sec=time, packet_id='same_vehicle', y=label,
        directions={'left':left or current(0, present=False), 'right':right or current(2, present=False)})


def test_904_positive_states_and_opposite_training_labels_do_not_define_bins():
    training = [row(TRAIN[0], 1, 0, current(0), current(2, 2., 2.)),
                row(TRAIN[1], 2, 2, right=current(2, 20., 60.))]
    population = [
        row(REGRESSION, 10, 0, current(0, 2., 2.)),
        row(REGRESSION, 11, 1, current(0, 2., 2.)),
        row(REGRESSION, 12, 1, current(0)),
        row(REGRESSION, 13, 1, right=current(2, 20., 60.)),
        row(REGRESSION, 14, 1, right=current(2, 2., 2.)),
        row(REGRESSION, 15, 1),
    ]
    result = assign_training_bins(population, training)
    assert [value['bin'] for value in result] == ['positive', '00', '10', '01', '00', '00']
    assert result[2]['directions']['left']['matched_positive_keys'] == [[TRAIN[0], 1, 'same_vehicle']]
    assert result[3]['directions']['right']['matched_positive_keys'] == [[TRAIN[1], 2, 'same_vehicle']]
    assert result[5]['directions']['left']['status'] == 'NO_ADJACENT_CURRENT_STRATUM'
    assert len(result) == len(population)
    assert 'bin' not in population[0]
    with pytest.raises(ValueError):
        assign_training_bins(population, [population[0]])


def test_904_sample_uses_rows_and_restores_10000_denominator_and_event_rate():
    population = [dict(seed=REGRESSION, time_sec=time, packet_id='same_vehicle',
                       y=0 if time < 56 else 2, bin='positive') for time in range(117)]
    time = 117
    for name, count in zip(('00', '10', '01', '11'), (1, 2, 3, 9877), strict=True):
        for _ in range(count):
            population.append(dict(seed=REGRESSION, time_sec=time, packet_id='same_vehicle', y=1, bin=name))
            time += 1
    selected, summary = select_cohort(population)
    assert len(selected) == 340
    assert Counter(value['y'] for value in selected) == {0:56, 2:61, 1:223}
    assert len({(value['seed'], value['time_sec'], value['packet_id']) for value in selected}) == 340
    assert {value['packet_id'] for value in selected} == {'same_vehicle'}
    assert {value['seed'] for value in selected} == {REGRESSION}
    sizes = Counter(value['bin'] for value in population if value['y'] == 1)
    samples = Counter(value['bin'] for value in selected if value['y'] == 1)
    for value in selected:
        if value['y'] != 1:
            assert value['raw_weight'] == value['inclusion_probability'] == 1.
            assert value['objective_weight'] == pytest.approx(1/10000)
        else:
            size, count = sizes[value['bin']], samples[value['bin']]
            assert value['inclusion_probability'] == pytest.approx(count/size)
            assert value['raw_weight'] == pytest.approx(size/count)
        assert value['objective_weight'] == pytest.approx(value['raw_weight']/10000)
    assert fsum(value['raw_weight'] for value in selected) == pytest.approx(10000., abs=1e-10)
    assert fsum(value['objective_weight'] for value in selected) == pytest.approx(1., abs=1e-14)
    assert fsum(value['objective_weight'] for value in selected if value['y'] != 1) == pytest.approx(117/10000)


def budget(full_forward):
    return dict(status='REGISTERED_SYMBOLIC_UPPER_BOUND',
        per_actor={'a':dict(current_positive_goals=2), 'b':dict(current_positive_goals=3)},
        physical_resolver_calls_upper_bound=full_forward, lateral_leaf_branches_upper_bound=1000)


def receipt(kind, *, actors=('a', 'b'), particles=None):
    return dict(kind=kind, particle_path=f'{kind}_server_only.json.gz', particle_count=particles,
        stochastic_participants=list(actors), frame_actor_count=5)


def test_paid_exact_then_paid_mixture_precede_missing_case_threshold_without_domain_shrink():
    exact, mixed, q0 = receipt('exact'), receipt('mixture', particles=256), receipt('q0', particles=256)
    plan = cost_plan(budget(512), ['a', 'b'], 5, dict(exact=[exact], mixture=[mixed], q0=[q0]))
    assert plan['proposed_method'] == 'exact'
    assert plan['proposed_reused_cache'] == exact
    assert plan['proposed_new_full_forward_ceiling'] == plan['proposed_new_feature_ceiling'] == 0
    plan = cost_plan(budget(128), ['a', 'b'], 5, dict(exact=[], mixture=[mixed], q0=[q0]))
    assert plan['proposed_method'] == 'mixture256'
    assert plan['proposed_reused_cache'] == mixed
    assert plan['proposed_new_full_forward_ceiling'] == 0
    wrong_domain = receipt('exact', actors=('a',))
    plan = cost_plan(budget(512), ['a', 'b'], 5, dict(exact=[wrong_domain], mixture=[mixed], q0=[]))
    assert plan['proposed_reused_cache'] == mixed
    for full_forward, method, expected in ((128, 'exact', 128), (512, 'mixture256', 256)):
        plan = cost_plan(budget(full_forward), ['a', 'b'], 5, dict(exact=[], mixture=[], q0=[q0]))
        assert plan['proposed_method'] == method
        assert plan['proposed_reused_cache'] is None
        assert plan['proposed_new_full_forward_ceiling'] == expected
