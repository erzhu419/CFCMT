"""Pure census, sampling and symbolic-cost checks; no archived frames are read."""

from collections import Counter, defaultdict
from math import fsum
from types import SimpleNamespace

import pytest

from experiments.tsc_packet_lc_negative_registration_20261111.census import _current_rows, assign_bins
from experiments.tsc_packet_lc_negative_registration_20261111.inventory import cost_plan
from experiments.tsc_packet_lc_negative_registration_20261111.sample import allocate_counts, select_cohort


BINS = ('00', '10', '01', '11')
SEEDS = (76116901, 76116902, 76116903)


def test_waterfill_exhausts_small_bins_and_distributes_remainder_in_fixed_order():
    assert allocate_counts(dict(zip(BINS, (1000, 1000, 1000, 1000)))) == dict(zip(BINS, (56, 56, 56, 55)))
    assert allocate_counts(dict(zip(BINS, (1, 2, 4, 9883)))) == dict(zip(BINS, (1, 2, 4, 216)))
    assert allocate_counts(dict(zip(BINS, (0, 2, 100, 9787)))) == dict(zip(BINS, (0, 2, 100, 121)))


def artificial_population():
    rows = []
    sizes = ((1, 2, 4, 9889), (4000, 3000, 2000, 887), (0, 2, 100, 9784))
    classes = ((51, 53), (51, 62), (63, 51))
    for seed, counts, (left, right) in zip(SEEDS, sizes, classes, strict=True):
        labels = [0]*left+[2]*right
        for time, label in enumerate(labels):
            rows.append(dict(seed=seed, time_sec=time,
                             packet_id='same_vehicle', y=label, bin='positive'))
        time = len(labels)
        for name, count in zip(BINS, counts, strict=True):
            for _ in range(count):
                rows.append(dict(seed=seed, time_sec=time, packet_id='same_vehicle', y=1, bin=name))
                time += 1
    return rows


def test_row_srs_retains_all_positives_and_has_correct_inclusion_and_natural_total_weights():
    population = artificial_population()
    selected, summary = select_cohort(population)
    identity = lambda row: (row['seed'], row['time_sec'], row['packet_id'])
    assert len(selected) == 1000
    assert len({identity(row) for row in selected}) == len(selected)
    assert {identity(row) for row in selected if row['y'] != 1} == {
        identity(row) for row in population if row['y'] != 1}
    assert Counter(row['y'] for row in selected) == {0:165, 2:166, 1:669}
    # Every seed deliberately contains only one vehicle: selection is by archived row.
    assert {row['packet_id'] for row in selected} == {'same_vehicle'}
    assert {row['seed'] for row in selected} == set(SEEDS)
    assert 76116904 not in {row['seed'] for row in selected}
    populations = Counter((row['seed'], row['bin']) for row in population if row['y'] == 1)
    samples = Counter((row['seed'], row['bin']) for row in selected if row['y'] == 1)
    by_seed = defaultdict(list)
    for row in selected:
        by_seed[row['seed']].append(row)
        if row['y'] != 1:
            assert row['raw_weight'] == row['inclusion_probability'] == 1.
        else:
            key = row['seed'], row['bin']
            size, count = populations[key], samples[key]
            assert row['stratum_size'] == size
            assert row['sampled_stratum_size'] == count
            assert row['inclusion_probability'] == pytest.approx(count/size)
            assert row['raw_weight'] == pytest.approx(size/count)
        assert row['objective_weight'] == pytest.approx(row['raw_weight']/30000.)
    for rows in by_seed.values():
        assert sum(row['y'] == 1 for row in rows) == 223
        assert fsum(row['raw_weight'] for row in rows) == pytest.approx(10000., abs=1e-10)
    assert fsum(row['raw_weight'] for row in selected) == pytest.approx(30000., abs=1e-10)
    assert fsum(row['objective_weight'] for row in selected) == pytest.approx(1., abs=1e-14)
    assert fsum(row['objective_weight'] for row in selected if row['y'] != 1) == pytest.approx(331/30000)


def directional_state(direction, *, speed=10., distance=30., lane='lane_a', no_adjacent=False):
    return dict(status='NO_ADJACENT_CURRENT_STRATUM' if no_adjacent else 'RECORDED',
        current_stratum=None if no_adjacent else [lane, direction, False, True, ['g'], True, False],
        speed_mps=speed, lane_end_distance_m=distance)


def population_row(seed, time, packet, label, left=None, right=None):
    return dict(seed=seed, time_sec=time, packet_id=packet, y=label, manifest_ordinal=time,
        directions={'left':left or directional_state(0, no_adjacent=True),
                    'right':right or directional_state(2, no_adjacent=True)})


def test_four_bins_use_original_closed_tolerances_and_keep_no_adjacent_rows():
    rows = [
        population_row(SEEDS[0], 0, 'reused_vehicle', 0, left=directional_state(0)),
        population_row(SEEDS[1], 1, 'right_positive', 2, right=directional_state(2, speed=20., distance=60.)),
        population_row(SEEDS[0], 10, 'reused_vehicle', 1,
            left=directional_state(0), right=directional_state(2, speed=20., distance=60.)),
        population_row(SEEDS[2], 11, 'boundary', 1,
            left=directional_state(0, speed=12., distance=45.)),
        population_row(SEEDS[2], 12, 'right_only', 1,
            left=directional_state(0, speed=12.0001), right=directional_state(2, speed=20., distance=60.)),
        population_row(SEEDS[2], 13, 'neither', 1,
            left=directional_state(0, distance=45.0001)),
        population_row(SEEDS[2], 14, 'no_adjacent', 1),
    ]
    result = assign_bins(rows)
    assert len(result) == len(rows)
    assert [row['bin'] for row in result] == ['positive', 'positive', '11', '10', '01', '00', '00']
    assert result[2]['directions']['left']['matched_positive_keys'] == [[SEEDS[0], 0, 'reused_vehicle']]
    assert result[2]['directions']['right']['matched_positive_keys'] == [[SEEDS[1], 1, 'right_positive']]
    assert result[6]['directions']['left']['status'] == 'NO_ADJACENT_CURRENT_STRATUM'
    assert result[6]['directions']['left']['matched_positive_keys'] == []
    assert 'bin' not in rows[2]


def test_original_lane_direction_and_stratum_categories_are_not_relaxed_by_nearness():
    positive = population_row(SEEDS[0], 0, 'positive', 0, left=directional_state(0))
    rows = [positive]
    for offset, (field, value) in enumerate(((0, 'other_lane'), (1, 2), (2, True), (3, False),
                                            (4, ['G']), (5, False), (6, True)), start=1):
        state = directional_state(0)
        state['current_stratum'][field] = value
        rows.append(population_row(SEEDS[1], offset, f'stay_{offset}', 1, left=state))
    result = assign_bins(rows)
    assert all(row['bin'] == '00' for row in result[1:])


def test_current_census_retains_no_adjacent_row_and_an_undefined_old_cost_bound():
    reference = dict(seed=SEEDS[0], time_sec=10, packet_id='ego', y=1)
    domain = dict(required_actor_count=12, required_participants=['ego'], frame_actor_count=20,
                  eligibility=dict(lane_change=True), physical_calls_upper=None, leaf_visits_upper=None)
    frame = dict(time_s=10, packets={'ego':dict(lane_id='lane_a', speed_mps=3., position_m=25.)})
    topology = SimpleNamespace(lanes={'lane_a':dict(length_m=100.)}, adjacent=lambda lane, direction:None)
    rows = _current_rows(frame, dict(time_sec=10, applied_states={}), [reference],
                         {(SEEDS[0], 10, 'ego'):domain}, topology)
    result = assign_bins(rows)
    assert len(result) == 1
    assert result[0]['required_actor_count'] == 12
    assert result[0]['physical_calls_upper'] is None
    assert result[0]['bin'] == '00'
    for value in result[0]['directions'].values():
        assert value['status'] == 'NO_ADJACENT_CURRENT_STRATUM'
        assert value['lane_end_distance_m'] == 75.
        assert value['matched_positive_keys'] == []


def symbolic_budget(full_forward=240):
    return dict(status='REGISTERED_SYMBOLIC_UPPER_BOUND',
        per_actor={'a':dict(current_positive_goals=2), 'b':dict(current_positive_goals=3)},
        physical_resolver_calls_upper_bound=full_forward, lateral_leaf_branches_upper_bound=900)


def receipt(kind, particles=None):
    return dict(stochastic_participants=['a', 'b'], frame_actor_count=5,
                particle_count=particles, particle_path=f'{kind}_server_only.json.gz')


def test_cost_plan_reuses_only_the_selected_exact_or_corrected_full_mixture_kind():
    exact = receipt('exact')
    q0 = receipt('q0', 256)
    mixed128, mixed256 = receipt('mixture128', 128), receipt('mixture256', 256)
    available = dict(exact=[exact], q0=[q0], mixture=[mixed128, mixed256])
    plan = cost_plan(symbolic_budget(), ['a', 'b'], 5, available)
    assert plan['proposed_method'] == 'exact'
    assert plan['proposed_reused_cache'] == exact
    assert plan['all_exact_full_forward_ceiling'] == 240
    assert plan['all_exact_feature_ceiling'] == 12
    assert plan['all_exact_leaf_ceiling'] == 900
    assert plan['proposed_new_full_forward_ceiling'] == plan['proposed_new_feature_ceiling'] == 0
    assert plan['proposed_new_exact_leaf_ceiling'] == 0
    plan = cost_plan(symbolic_budget(512), ['a', 'b'], 5, dict(exact=[], q0=[q0], mixture=[mixed128]))
    assert plan['proposed_method'] == 'mixture256'
    assert plan['proposed_reused_cache'] is None
    assert plan['proposed_new_full_forward_ceiling'] == 256
    assert plan['proposed_new_feature_ceiling'] == 12
    assert plan['proposed_new_exact_leaf_ceiling'] == 0
    plan = cost_plan(symbolic_budget(512), ['a', 'b'], 5, available)
    assert plan['proposed_reused_cache'] == mixed256
    assert plan['proposed_new_full_forward_ceiling'] == plan['proposed_new_feature_ceiling'] == 0


def test_unknown_symbolic_domain_has_no_invented_cost_or_method():
    plan = cost_plan(dict(status='OUTSIDE_MAX_STOCHASTIC'), [], 20,
                     dict(exact=[], q0=[], mixture=[]))
    assert plan['proposed_method'] is None
    assert plan['proposed_reused_cache'] is None
    assert plan['proposed_new_full_forward_ceiling'] is None
    assert plan['proposed_new_feature_ceiling'] is None
    assert plan['proposed_new_exact_leaf_ceiling'] is None
