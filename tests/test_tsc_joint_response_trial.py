from copy import deepcopy

import pytest

from scripts.cluster.collect_tsc_joint_response_trial import PROTOCOL, validate_spec


def spec_fixture():
    seeds = list(range(1, 15))
    return dict(protocol=PROTOCOL, max_new_branches=224, max_new_fits=30,
        response_horizon_sec=10, cost_horizon_sec=450, checkpoint_search_sec=60,
        source_seeds=seeds[:4], target_seeds=seeds[4:], excluded_measurement_seeds=[100, 101],
        cases=[dict(case_id=f'{seed}_{t}', city='hangzhou' if seed <= 4 else 'jinan',
            seed=seed, checkpoint_sec=t) for seed in seeds for t in [600, 1800]])


def test_whole_seed_shard_retains_both_times_and_city():
    spec = spec_fixture()
    assert [r['checkpoint_sec'] for r in validate_spec(spec, 1)] == [600, 1800]
    assert {r['city'] for r in validate_spec(spec, 14)} == {'jinan'}


def test_duplicate_missing_or_pilot_seeds_never_launch():
    original = spec_fixture()
    for mutation in ['duplicate', 'pilot', 'wrong_anchor']:
        spec = deepcopy(original)
        if mutation == 'duplicate':
            spec['cases'][0] = spec['cases'][1]
        elif mutation == 'pilot':
            spec['excluded_measurement_seeds'].append(1)
        else:
            spec['cases'][0]['checkpoint_sec'] = 900
        with pytest.raises(ValueError):
            validate_spec(spec, 1)
