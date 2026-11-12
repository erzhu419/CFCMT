"""Background presence, conditional response and whole-input obstruction."""
from math import fsum

import pytest

from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_conditional_law_20261110.synthetic import cases, three_lanes
from experiments.tsc_packet_conditional_law_20261110.verify import check_case
from experiments.tsc_packet_joint_law_definition_20261110.law import enumerate_law, marginal, OutsidePrototype
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import cases as tiny_cases, make_kernel, vehicle


def atom_map(law):
    return {tuple((key, tuple(row.items())) for key, row in atom['actors'].items()): atom['probability'] for atom in law['atoms']}


@pytest.mark.parametrize('name', ['stopped_open', 'two_actor_lateral', 'intent_and_survival', 'receiving_intent'])
def test_when_selected_set_is_complete_it_matches_previous_law_atom_by_atom(name):
    kernel, focal = tiny_cases()[name]
    assert atom_map(enumerate_conditional(kernel, focal)) == atom_map(enumerate_law(kernel))


def test_eight_actor_frame_has_two_random_actors_and_keeps_every_conditioned_actor():
    kernel, focal = cases()['distant_background']
    result = check_case(kernel, focal)
    assert result['frame_actor_count'] == 8 and result['stochastic_actor_count'] == 2
    assert result['all_input_actors_retained'] and result['input_and_rng_unchanged']
    assert result['event_probabilities']['lane_change']['0'] == pytest.approx(.125)
    assert result['total_probability'] == pytest.approx(1.)


def test_unselected_background_still_binds_the_selected_neighbour_and_focal_lc():
    kernel, focal = cases()['binding_background']
    law = enumerate_conditional(kernel, focal)
    assert law['stochastic_participants'] == ['focal', 'near'] and 'blocker' in law['background_conditions']
    free = make_kernel(three_lanes(), [vehicle('focal', 'a0', 40, 5), vehicle('near', 'a1', 52, 10)])
    assert marginal(law, focal, 'lane_change', 0)['likelihood'] == pytest.approx(.375)
    assert marginal(enumerate_conditional(free, focal), focal, 'lane_change', 0)['likelihood'] == pytest.approx(.5)


def test_fixed_background_actions_do_not_freeze_its_endpoints_across_local_branches():
    kernel, focal = cases()['responsive_background']
    law = enumerate_conditional(kernel, focal)
    assert law['stochastic_participants'] == ['focal']
    conditions = law['background_conditions']['crossing_background']
    assert conditions['residual_mps'] == 0 and conditions['start_permission'] and conditions['yield_permission']
    r = law['background_response_ranges']['crossing_background']
    assert r['min_travel_m'] == pytest.approx(14., abs=1e-8) and r['max_travel_m'] == 18.
    assert all(atom['actors']['crossing_background']['lane_change'] == 'stay' for atom in law['atoms'])


def test_larger_frames_keep_receiving_intent_weights_and_end_conditioning():
    fixtures = cases()
    kernel, focal = fixtures['receiving_with_background']
    receiving = check_case(kernel, focal)
    assert receiving['stochastic_actor_count'] == 1
    assert receiving['event_probabilities']['lane_change']['2'] == pytest.approx(.35)
    kernel, focal = fixtures['survival_with_background']
    survival = check_case(kernel, focal)
    assert survival['exit_no_speed_probability'] == pytest.approx(.3)
    assert survival['survivor_probability'] == pytest.approx(.7)
    assert survival['survivor_observation_total'] == pytest.approx(1.)


def test_geometry_overflow_prevents_whole_law_instead_of_nearest_actor_truncation():
    kernel, focal = cases()['overflow']
    with pytest.raises(OutsidePrototype) as caught:
        enumerate_conditional(kernel, focal)
    assert caught.value.selection['required_count'] == 5
    assert caught.value.selection['participants'] == []
    assert len(caught.value.selection['required_participants']) == 5


def test_larger_stopped_case_preserves_realized_start_observation_relation():
    kernel, focal = cases()['start_with_background']
    result = check_case(kernel, focal)
    assert result['stochastic_actor_count'] == 1
    assert result['event_probabilities']['start']['1'] == .25
    assert result['survivor_observation_total'] == pytest.approx(1.)
