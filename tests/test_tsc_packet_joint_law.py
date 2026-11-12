"""Physical shared branches, collapsed atoms and conditional observation law."""
from copy import deepcopy
from math import fsum

import pytest

from experiments.tsc_packet_joint_law_definition_20261110.law import enumerate_law, marginal, OutsidePrototype
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import cases, make_kernel, vehicle
from experiments.tsc_packet_joint_law_definition_20261110.verify import check_case
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import lane, road_graph


def test_one_realized_motion_defines_start_and_collapses_permission_residual_branches():
    kernel, focal = cases()['stopped_open']
    law = enumerate_law(kernel)
    assert len(law['atoms']) == 2 and law['diagnostics']['enumerated_leaf_branches'] == 4
    assert sorted(atom['probability'] for atom in law['atoms']) == [.25, .75]
    assert marginal(law, focal, 'start', 1)['likelihood'] == .25
    assert law['diagnostics']['expected_actor_count_start_permitted_but_not_started'] == .25
    checked = check_case(kernel, focal)
    assert checked['start_band_matches_observed_speed'] and checked['survivor_observation_total'] == pytest.approx(1.)


def test_actual_entry_can_be_absent_despite_yield_permission_and_physics_stays_frozen():
    kernels = cases()
    kernel, focal = kernels['yielding_entry']
    law = enumerate_law(kernel)
    assert marginal(law, focal, 'yield_entry', 1)['likelihood'] == .5
    blocked, focal = kernels['blocked_entry']
    joint = enumerate_law(blocked)
    assert joint['diagnostics']['expected_actor_count_yield_permitted_but_not_entered'] > 0
    # Frozen prestate following limits the ego below the one-metre entry
    # distance even in branches where its initially stopped leader starts.
    assert marginal(joint, focal, 'yield_entry', 1)['likelihood'] == 0.


def test_independent_shared_residual_branches_admit_a_joint_lateral_event_without_global_mean_shift():
    kernel, focal = cases()['two_actor_lateral']
    state, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    random_states = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate()) for key, row in kernel.noise._packets.items()}
    law = enumerate_law(kernel)
    assert marginal(law, focal, 'lane_change', 0)['likelihood'] == pytest.approx(.125)
    assert fsum(marginal(law, focal, 'lane_change', y)['likelihood'] for y in (0, 1, 2)) == pytest.approx(1.)
    assert kernel.state == state and kernel._connections == connections
    assert random_states == {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate()) for key, row in kernel.noise._packets.items()}


def test_end_is_not_a_stay_observation_and_survivor_likelihood_keeps_its_denominator():
    kernel, focal = cases()['intent_and_survival']
    law = enumerate_law(kernel)
    exits = [atom for atom in law['atoms'] if not atom['actors'][focal]['survived']]
    assert fsum(atom['probability'] for atom in exits) == pytest.approx(.3)
    assert all(atom['actors'][focal]['physical_speed_mps'] is None and atom['actors'][focal]['lane_change'] is None for atom in exits)
    assert marginal(law, focal, 'motion', 0)['survivor_probability'] == pytest.approx(.7)
    assert check_case(kernel, focal)['survivor_observation_total'] == pytest.approx(1.)


def test_new_receiving_road_goals_are_integrated_instead_of_union_mask_or_rng_draw():
    kernel, focal = cases()['receiving_intent']
    law = enumerate_law(kernel)
    # END prior .7 allows stay/right, next prior .3 allows only stay.
    assert marginal(law, focal, 'lane_change', 2)['likelihood'] == pytest.approx(.35)
    assert marginal(law, focal, 'lane_change', 1)['likelihood'] == pytest.approx(.65)


def test_oversize_complete_input_is_undefined_instead_of_a_truncated_dataset_row():
    static = road_graph([lane('a', 100)], [])
    kernel = make_kernel(static, [vehicle(str(i), 'a', 10+15*i, 1) for i in range(4)])
    with pytest.raises(OutsidePrototype, match='complete stochastic cohort'):
        enumerate_law(kernel)
