"""Independent finite-law and score identities on the frozen synthetic cohort."""
import numpy as np
import pytest

from experiments.tsc_packet_conditional_cache_20261110.verify import (
    gradient_directions, parameter_probes, run,
)


@pytest.fixture(scope='module')
def verified():
    return run()


@pytest.mark.parametrize('name', [
    'cap4_lateral', 'cap4_lateral_earlier_neighbour', 'cap4_start_permissions',
    'stopped_open', 'blocked_start', 'yielding_entry', 'blocked_entry',
    'two_actor_lateral', 'intent_and_survival', 'receiving_intent',
    'distant_background', 'binding_background', 'responsive_background',
])
def test_independent_enumeration_and_gradient_match_in_each_registered_case(verified, name):
    row = verified['cases'][name]
    assert row['parameter_probes'] == 2 and row['cache_constructions'] == 1
    assert row['maximum_atom_probability_error'] < 1e-10
    assert row['maximum_log_likelihood_error'] < 1e-10
    assert row['maximum_weighted_score_error'] < 1e-10
    assert row['gradient_direction_checks'] > 0
    assert row['reuse_physical_calls'] == 0
    assert row['input_state_context_and_rng_unchanged']


def test_fixed_probes_and_mixed_nonbias_direction_are_not_parameter_search():
    zero, nonzero = parameter_probes()
    directions = gradient_directions()
    assert not np.any(zero) and np.any(nonzero)
    assert len(directions) == 6 and not np.any(directions[-1][:, 0])
    assert all(np.linalg.norm(direction) == pytest.approx(1.) for direction in directions)


def test_survival_denominator_gradient_and_start_probability_have_closed_form_witness(verified):
    row = verified['closed_forms']
    assert row['start_maximum_error'] < 1e-10
    assert row['survival_probability_at_zero'] == pytest.approx(.7)
    assert row['survival_denominator_nll_gradient'] == pytest.approx(3/14, abs=1e-10)
    assert row['survival_perturbation_maximum_error'] < 1e-10


def test_three_four_actor_cases_and_backgrounds_remain_in_the_complete_cache(verified):
    rows = verified['cases']
    assert sum(row['stochastic_actor_count'] == 4 for row in rows.values()) == 3
    assert rows['cap4_start_permissions']['physical_calls'] == 256
    assert rows['responsive_background']['frame_actor_count'] == 6
    assert rows['responsive_background']['stochastic_actor_count'] == 1
    assert rows['distant_background']['feature_count'] < rows['distant_background']['frame_actor_count']
    assert verified['total_reuse_physical_calls'] == 0


def test_zero_support_stays_zero_without_probability_floor_and_no_paid_work(verified):
    assert sum(row['zero_support_checks'] for row in verified['cases'].values()) > 0
    assert verified['real_record_response_queries'] == 0
    assert verified['new_native_starts'] == 0 and verified['new_fits'] == 0
    assert verified['optimizer_calls'] == 0
