"""Exact cap4 streaming versus full atoms, COW isolation and hard budgets."""
import pytest

from experiments.tsc_packet_conditional_law_20261110.synthetic import cases as larger_cases
from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_joint_law_definition_20261110.law import OutsidePrototype, marginal
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import cases as tiny_cases
from experiments.tsc_packet_conditional_panel_20261110.stream import stream_zero_law
from experiments.tsc_packet_conditional_panel_20261110.synthetic import check_case, check_manual_initializer, four_actor_case


@pytest.mark.parametrize('stopped,earlier', [(False, False), (False, True), (True, False)])
def test_four_actual_random_actors_match_complete_focal_atoms_and_normalize(stopped, earlier):
    kernel, focal = four_actor_case(stopped=stopped, earlier_neighbour=earlier)
    result = check_case(kernel, focal)
    assert result['stochastic_actor_count'] == 4
    assert result['maximum_atom_probability_difference'] < 1e-12
    assert result['survivor_observation_total'] == pytest.approx(1.)
    assert result['input_state_context_and_rng_unchanged']
    if stopped:
        assert result['physical_branches'] == 256
        assert result['event_probabilities']['start']['1'] == pytest.approx(.25, abs=1e-12)


def test_copy_on_write_retains_earlier_neighbour_lc_effect_on_focal_legality():
    first = check_case(*four_actor_case())
    early = check_case(*four_actor_case(earlier_neighbour=True))
    assert first['event_probabilities']['lane_change'] != early['event_probabilities']['lane_change']


@pytest.mark.parametrize('name', ['stopped_open', 'yielding_entry', 'intent_and_survival', 'receiving_intent'])
def test_existing_small_survival_event_and_receiver_prior_cases_are_retained(name):
    result = check_case(*tiny_cases()[name])
    assert result['maximum_event_probability_difference'] < 1e-12
    if name == 'intent_and_survival':
        assert result['survivor_probability'] == pytest.approx(.7)
        assert result['exit_no_speed_probability'] == pytest.approx(.3)
    if name == 'receiving_intent':
        assert result['event_probabilities']['lane_change']['2'] == pytest.approx(.35)


def test_streaming_retains_responsive_background_without_materializing_its_atoms():
    result = check_case(*larger_cases()['responsive_background'])
    response = result['background_response_ranges']['crossing_background']
    assert result['frame_actor_count'] == 6 and result['stochastic_actor_count'] == 1
    assert response['min_travel_m'] == pytest.approx(14., abs=1e-8)
    assert response['max_travel_m'] == 18.


def test_real_runner_manual_constructor_needs_no_prepared_response_attributes():
    result = check_manual_initializer()
    assert result['no_prepared_conditional_attributes']
    assert result['matches_prepared_focal_atoms'] and result['matches_background_response_ranges']
    assert result['input_and_rng_unchanged']


@pytest.mark.parametrize('name,y', [('stopped_open', 0.), ('stopped_open', .05),
    ('stopped_open', .1), ('stopped_open', 3.), ('intent_and_survival', 1.)])
def test_observed_speed_readout_uses_same_mixed_measure_and_survival_denominator(name, y):
    kernel, focal = tiny_cases()[name]
    old = enumerate_conditional(kernel, focal, max_stochastic=4)
    streamed = stream_zero_law(kernel, focal, observed_speed=y)
    expected = marginal(old, focal, 'motion', y)
    assert streamed['observed_speed_likelihood']['likelihood'] == pytest.approx(expected['likelihood'], abs=1e-12)
    assert streamed['observed_speed_likelihood']['survivor_probability'] == pytest.approx(expected['survivor_probability'])
    assert sum(streamed['conditional_class_speed_likelihoods']) == pytest.approx(expected['likelihood'], abs=1e-12)


@pytest.mark.parametrize('limits,field,expected', [
    ({'physical_limit': 1}, 'physical_calls', 1),
    ({'leaf_limit': 1}, 'leaf_visits', 1),
])
def test_budget_exhaustion_rejects_whole_case_before_exceeding_declared_limit(limits, field, expected):
    kernel, focal = four_actor_case()
    with pytest.raises(OutsidePrototype) as caught:
        stream_zero_law(kernel, focal, **limits)
    assert getattr(caught.value, field) == expected
