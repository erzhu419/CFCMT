"""Geometry membership, hidden receiving arrivals and no label-based pruning."""
from copy import deepcopy

from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import make_kernel, vehicle
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    lane, laws, parallel_receivers, road_graph,
)


def lanes_three(length=200):
    return road_graph([lane('a0', length, edge='a', index=0),
                       lane('a1', length, edge='a', index=1),
                       lane('a2', length, edge='a', index=2)], [])


def test_complete_eight_vehicle_frame_keeps_only_direct_geometric_interactors():
    rows = [vehicle('focal', 'a0', 20, 5), vehicle('near', 'a1', 24, 5)]
    rows += [vehicle('far'+str(i), 'a'+str(i % 3), 100+15*(i // 3), 1) for i in range(6)]
    kernel = make_kernel(lanes_three(), rows)
    result = select_stochastic(kernel, 'focal')
    assert result['status'] == 'SELECTED_GEOMETRIC_ENVELOPE'
    assert result['participants'] == ['focal', 'near']
    assert len(result['bounds']) == 8  # The physical background is never dropped.


def test_actor_two_source_lanes_away_can_land_in_the_focal_target_lane():
    kernel = make_kernel(lanes_three(), [vehicle('focal', 'a0', 20, 5), vehicle('other', 'a2', 24, 5)])
    result = select_stochastic(kernel, 'focal')
    assert result['participants'] == ['focal', 'other']
    assert 'a1' in result['reasons']['other']


def test_committed_internal_arrival_is_selected_on_its_receiving_road():
    static, history = parallel_receivers()
    kernel = make_kernel(static, [vehicle('focal', 'out0', 10, 1), vehicle('incoming', 'I1', 7, 10)], history, 'GG')
    result = select_stochastic(kernel, 'focal')
    assert result['participants'] == ['focal', 'incoming']
    assert 'out1' in result['bounds']['incoming']['longitudinal_fronts']
    assert result['bounds']['incoming']['longitudinal_fronts']['out1'][0] == 0.


def test_geometry_overflow_reports_the_entire_required_set_without_closest_pruning():
    rows = [vehicle('focal', 'a1', 30, 5), vehicle('left', 'a2', 35, 5),
            vehicle('right', 'a0', 25, 5), vehicle('ahead', 'a1', 45, 5)]
    kernel = make_kernel(lanes_three(), rows)
    result = select_stochastic(kernel, 'focal')
    assert result['status'] == 'OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP'
    assert result['participants'] == []
    assert result['required_participants'] == ['focal', 'left', 'right', 'ahead']
    assert result['required_count'] == 4


def test_bounds_honor_finite_ends_and_selection_does_not_touch_state_rng_or_coefficients():
    static = road_graph([lane('a0', 100, edge='a', index=0), lane('a1', 40, edge='a', index=1)], [])
    kernel = make_kernel(static, [vehicle('focal', 'a0', 80, 20), vehicle('other', 'a1', 35, 20)])
    state, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    random_states = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate()) for key, row in kernel.noise._packets.items()}
    result = select_stochastic(kernel, 'focal')
    assert result['bounds']['focal']['longitudinal_fronts']['a0'] == (80., 100.)
    assert 'a1' not in result['bounds']['focal']['potential_landing_fronts']
    assert result['bounds']['other']['longitudinal_fronts']['a1'] == (35., 40.)
    kernel.model.coefficients = {name: (100.,)*38 for name in kernel.model.coefficients}
    assert select_stochastic(kernel, 'focal') == result
    assert kernel.state == state and kernel._connections == connections
    assert random_states == {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate()) for key, row in kernel.noise._packets.items()}
