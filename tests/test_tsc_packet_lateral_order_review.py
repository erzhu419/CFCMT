"""Independent checks of the declared packet-order lateral witness."""
from copy import deepcopy
from itertools import product

import pytest

from experiments.tsc_packet_discrete_braking_20261110.kernel import PacketKernel
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_lateral_conditioning_20261110.order import forced_lateral_order
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    initial_observation, lane, laws, packet, road_graph,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def two_lateral_actors():
    static = road_graph([lane('a0', 100, edge='a', index=0),
                         lane('a1', 100, edge='a', index=1),
                         lane('a2', 100, edge='a', index=2)], [])
    state = initial_observation(static, [packet('focal', 'a0', 20, 5),
                                         packet('front', 'a1', 22, 5)])
    state['intents'] = {pid: dict(road_id='a', choice='END') for pid in state['packets']}
    kernel = PacketKernel(static, state, {}, laws(static), random_seed=0,
                        conflict_geometry=build_entity_geometry(RoadTopology(static), 2.))
    return kernel, deepcopy(kernel.state['packets']), dict(focal='left', front='left')


def test_front_departure_first_makes_the_following_lateral_action_acceptable():
    kernel, projected, actions = two_lateral_actors()
    result = forced_lateral_order(kernel, projected, actions, dict(front=0, focal=1))
    assert result['all_requested_actions_accepted']
    assert [row['packet_id'] for row in result['steps']] == ['front', 'focal']
    assert result['packets']['front']['lane_id'] == 'a2'
    assert result['packets']['focal']['lane_id'] == 'a1'
    assert projected['front']['lane_id'] == 'a1' and projected['focal']['lane_id'] == 'a0'


def test_later_front_departure_does_not_retroactively_admit_the_focal_action():
    kernel, projected, actions = two_lateral_actors()
    result = forced_lateral_order(kernel, projected, actions, dict(focal=0, front=1))
    assert not result['all_requested_actions_accepted']
    first, second = result['steps']
    assert first['packet_id'] == 'focal' and not first['accepted']
    assert first['allowed_choices'] == ['stay'] and first['zero_logit_probability'] == 0.
    assert second['packet_id'] == 'front' and second['accepted']
    assert 'left' in second['allowed_choices']
    assert result['zero_logit_tuple_mass'] == 0.
    assert result['packets']['focal']['lane_id'] == 'a0'
    assert result['packets']['front']['lane_id'] == 'a2'


def test_complete_static_tuple_set_normalizes_each_fixed_execution_order():
    kernel, projected, unused = two_lateral_actors()
    for order, expected_focal_left in ((dict(front=0, focal=1), .25),
                                        (dict(focal=0, front=1), 0.)):
        total, focal_left = 0., 0.
        # These are all topology/goal-legal classes before dynamic gap masks.
        for focal_action, front_action in product(('stay', 'left'), ('stay', 'left', 'right')):
            actions = dict(focal=focal_action, front=front_action)
            result = forced_lateral_order(kernel, projected, actions, order)
            mass = result['zero_logit_tuple_mass']
            total += mass
            if focal_action == 'left':
                focal_left += mass
            if not result['all_requested_actions_accepted']:
                assert mass == 0.
        assert total == pytest.approx(1.)
        assert focal_left == pytest.approx(expected_focal_left)
