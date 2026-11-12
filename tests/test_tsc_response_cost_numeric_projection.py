"""Frozen physical features, exact signal roles and structural PP anchoring."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.tsc_response_cost_readout_20261109 import numeric_projection as p


def state(n, h, m, occupancy, storage):
    return dict(N=n, H=h, M=m, occupancy=occupancy, storage=storage)


def inputs():
    parameters = dict(length_m=5., min_gap_m=2.5, accel_mps2=2.6, decel_mps2=4.5,
        tau_s=1., imperfection=.5, max_speed_mps=11.11, speed_factor=1.2, allowed_speed_mps=12.)
    a = dict(lane_id='a', **state(2, 1, 2.4, .1, 20), length_m=150., speed_limit_mps=10.,
        heads=[dict(present=True, distance_ratio=.1, speed_ratio=1.2),
               dict(present=True, distance_ratio=.5, speed_ratio=.8)], near_count=2,
        parameters={'mean': parameters.copy()}, to_lanes=['r0', 'r1'])
    b = dict(lane_id='b', **state(4, 2, 4., .2, 10), length_m=75., speed_limit_mps=10.,
        heads=[dict(present=True, distance_ratio=.05, speed_ratio=1.1),
               dict(present=True, distance_ratio=.1, speed_ratio=.9)], near_count=4,
        parameters={'mean': parameters.copy()}, to_lanes=['r0'])
    data = dict(can_full_fit=True, lane_nodes=[a, b],
        connections=[dict(from_lane='a', to_lane='r0', link_index=0),
                     dict(from_lane='a', to_lane='r1', link_index=1),
                     dict(from_lane='b', to_lane='r0', link_index=2)],
        movements={'m_a': dict(incoming_lanes=['a'], outgoing_edge='receiving_edge'),
                   'm_b': dict(incoming_lanes=['b'], outgoing_edge='receiving_edge')},
        receiving_nodes={'r0': state(3, 1, 2.4, .1, 30), 'r1': state(0, 0, 0, 0, 10)},
        internal_nodes={'inside': state(2, 1, .4, .2, 4)}, internal_vehicle_count=2,
        relations=dict(foes=[[0, 2], [2, 0]], yields_to=[[0, 2]], shared_to_lane=[]),
        executor0=dict(snapshot=dict(mode='green', green_elapsed_sec=15., remaining_sec=0., pending_all_red_sec=2.),
                       minimum_green_sec=30., feasible_states=['metadata']*8, timings=[]),
        nominal_executor10=dict(snapshot=dict(mode='green', green_elapsed_sec=4., remaining_sec=0., pending_all_red_sec=0.),
                               minimum_green_sec=30., feasible_states=['metadata']), request_differs=True)
    data['global_nodes'] = {'a': deepcopy(a), 'b': deepcopy(b), 'elsewhere': state(3, 0, 3, .3, 30)}
    set_signals(data, {'0': 'G'*10, '1': 'r'*10, '2': 'g'*10})
    return data


def set_signals(data, sequences):
    data['nominal_link_sequences'] = sequences
    for node in data['lane_nodes']:
        links = [row['link_index'] for row in data['connections'] if row['from_lane'] == node['lane_id']]
        node['nominal_signal_fractions'] = [[sum(sequences[str(link)][step] == color for link in links)/len(links)
                                           for color in p.SIGNALS] for step in range(10)]


def column(matrix, row, name):
    return matrix[row, p.DYNAMICS_FEATURE_NAMES.index(name)]


def test_frozen_local_and_joint_columns_keep_partial_connections_and_directed_yields():
    data = inputs()
    lane_ids, matrix = p.project_dynamics(data)
    assert lane_ids == ['a', 'b'] and matrix.shape == (2, 476)
    assert len(set(p.DYNAMICS_FEATURE_NAMES)) == 476
    assert column(matrix, 0, 'rhoN') == .1
    assert column(matrix, 0, 'freeflow_time_10') == 1.5
    assert column(matrix, 0, 'shared_turn') == 0  # Two toLanes still belong to one outgoing edge.
    assert column(matrix, 0, 'head0_speed_ratio') == 1.2
    assert column(matrix, 0, 'accel_10_limit') == 2.6
    assert column(matrix, 0, 'receiving_rhoN') == pytest.approx(.05)
    assert column(matrix, 0, 'internal_vehicles_per_lane') == 2.
    assert column(matrix, 0, 'executor_green_age_min_green') == .5
    assert column(matrix, 0, 'own_t00_G_fraction') == .5
    assert column(matrix, 0, 'own_t00_G_rhoN') == .05
    # Only one of a's two connections is permissive and related to b.
    for relation in ('foes', 'self_yields', 'shared_receiving'):
        assert column(matrix, 0, relation+'_t00_g_rhoN') == pytest.approx(.4*.5/11)
    assert column(matrix, 0, 'other_yields_t00_g_rhoN') == 0
    assert column(matrix, 1, 'other_yields_t00_G_rhoN') == pytest.approx(.1*.5/11)
    red = deepcopy(data)
    set_signals(red, {'0': 'r'*10, '1': 'r'*10, '2': 'g'*10})
    assert column(p.project_dynamics(red)[1], 0, 'foes_t00_g_rhoN') == 0


def test_empty_lanes_use_masks_and_missing_parameters_fail_instead_of_becoming_defaults():
    data = inputs()
    empty = data['lane_nodes'][0]
    empty.update(N=0, H=0, M=0., occupancy=0., near_count=0, parameters={'mean': None},
                 heads=[dict(present=False, distance_ratio=None, speed_ratio=None)]*2)
    matrix = p.project_dynamics(data)[1]
    assert column(matrix, 0, 'empty') == 1
    for name in ('parameters_observed', 'accel_10_limit', 'head0_distance_ratio', 'head0_speed_ratio', 'near_count'):
        assert column(matrix, 0, name) == 0
    data['can_full_fit'] = False
    with pytest.raises(ValueError, match='complete frozen predecision'):
        p.project_dynamics(data)


def test_predicted_state_uses_only_frozen_component_bounds_and_population_phi():
    data = inputs()
    delta = np.array([[-.5, 2., -1., -.2], [.2, .8, .3, 1.1]])
    saved = delta.copy()
    predicted = p.project_predicted_state(data, delta)
    np.testing.assert_allclose(predicted['a'], [0., 0., 0., 0.])
    np.testing.assert_allclose(predicted['b'], [.6, .6, .7, 1.3])  # Occupancy has no upper clipping.
    np.testing.assert_array_equal(delta, saved)
    reader = p.project_reader(data, predicted)
    assert reader.shape == (211,) and len(set(p.READER_FEATURE_NAMES)) == 211
    np.testing.assert_allclose(reader[-16:], [.3, .3, 0, .6, .3, .3, 0, .6,
                                             .35, .35, 0, .7, .65, .65, 0, 1.3])


def test_reader_difference_is_exact_pp_zero_and_shared_t0_tensor_difference():
    action = inputs()
    pp = deepcopy(action)
    pp['request_differs'] = False
    pp['nominal_executor10']['snapshot']['green_elapsed_sec'] = 25.
    set_signals(pp, {'0': 'r'*10, '1': 'r'*10, '2': 'G'*10})
    action_state = p.project_predicted_state(action, np.zeros((2, 4)))
    pp_state = p.project_predicted_state(pp, np.zeros((2, 4)))
    a, baseline = p.project_reader(action, action_state), p.project_reader(pp, pp_state)
    difference = a-baseline
    g0 = np.array([.2, .25/3, .62/3, .2, .25, .125, .26, .15,
                   .05, 1/60, .04, .05])
    np.testing.assert_allclose(difference[15:195], np.outer(g0, difference[:15]).ravel())
    np.testing.assert_array_equal(baseline-baseline, np.zeros(211))
    np.testing.assert_array_equal(difference[-16:], np.zeros(16))
    assert a[p.READER_FEATURE_NAMES.index('halted_weighted_exposure_G')] == pytest.approx(.0125)
    assert a[p.READER_FEATURE_NAMES.index('halted_weighted_exposure_g')] == pytest.approx(.1)


def test_local_ids_order_and_unused_future_payloads_cannot_change_physical_projection():
    original = inputs()
    changed = deepcopy(original)
    rename = {'a': 'z_lane', 'b': 'a_lane', 'r0': 'receiver_9', 'r1': 'receiver_1'}
    links = {0: 7, 1: 4, 2: 9}
    for node in changed['lane_nodes']:
        node['lane_id'] = rename[node['lane_id']]
        node['to_lanes'] = [rename[lane] for lane in reversed(node['to_lanes'])]
    changed['lane_nodes'].reverse()
    for connection in changed['connections']:
        connection['from_lane'], connection['to_lane'] = rename[connection['from_lane']], rename[connection['to_lane']]
        connection['link_index'] = links[connection['link_index']]
    changed['connections'].reverse()
    for movement in changed['movements'].values():
        movement['incoming_lanes'] = [rename[lane] for lane in movement['incoming_lanes']]
        movement['outgoing_edge'] = 'renamed_'+movement['outgoing_edge']
    changed['receiving_nodes'] = {rename[lane]: node for lane, node in reversed(list(changed['receiving_nodes'].items()))}
    for key in ('foes', 'yields_to'):
        changed['relations'][key] = [[links[a], links[b]] for a, b in reversed(changed['relations'][key])]
    changed['nominal_link_sequences'] = {str(links[int(link)]): sequence
                                          for link, sequence in changed['nominal_link_sequences'].items()}
    changed.update(candidate_index=999, actual_signals450=['arbitrary'], costs450=[999],
                   targets={'y': [[999]*4]*2}, endpoint10={'future': True})
    before_ids, before = p.project_dynamics(original)
    after_ids, after = p.project_dynamics(changed)
    for row, lane in enumerate(before_ids):
        np.testing.assert_array_equal(before[row], after[after_ids.index(rename[lane])])
    original_state = p.project_predicted_state(original, np.zeros((2, 4)))
    changed_state = p.project_predicted_state(changed, np.zeros((2, 4)))
    np.testing.assert_array_equal(p.project_reader(original, original_state), p.project_reader(changed, changed_state))
