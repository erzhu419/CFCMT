from copy import deepcopy

import numpy as np
import pytest

from scripts.cluster.tsc_joint_response_features import FEATURE_NAMES, build_joint_features
from scripts.cluster.tsc_joint_response_interface import EXECUTOR_FIELDS


def graph_fixture():
    def lane(name, distances, halted=0, length=200., speed=10., internal=False):
        return dict(lane_id=name, edge_id=name + '_edge', length_m=length,
            speed_limit_mps=speed, vehicle_count=len(distances), distances_to_lane_end_m=distances,
            near_stopline_vehicle_count=sum(d <= speed * 10 for d in distances),
            shared_turn_lane=name == 'a', lane_measurements=None if internal else dict(
                vehicles=len(distances), halted=halted, occupancy_raw=.1 * len(distances),
                mean_speed_mps=4. if distances else None))
    lanes = [lane('a', [10., 40., 180.], 2), lane('b', [15., 120.], 1),
             lane('c', [5., 20., 50., 90., 150.], 3)]
    lanes += [lane(f'out{i}', [50.], length=100.) for i in range(4)]
    lanes += [lane(f':via{i}', [5.] if i == 0 else [], length=15., internal=True) for i in range(4)]
    movements = [dict(movement_id=f'm{i}', incoming_edge=inlet + '_edge', outgoing_edge=f'out{i}_edge',
        incoming_lanes=[inlet], outgoing_lanes=[f'out{i}'], internal_lanes=[f':via{i}'],
        link_indices=[i]) for i, inlet in enumerate(['a', 'a', 'b', 'c'])]
    views = [dict(candidate_index=i, nominal_request_accepted=True,
                  link_sequences={str(link): state * 10 for link, state in enumerate('GgGg')})
             for i in range(8)]
    views[1]['link_sequences']['0'] = 'G' + 'r' * 9
    views[2]['link_sequences']['0'] = 'g' * 10
    views[3]['link_sequences']['0'] = 'Y' * 10
    views[4]['link_sequences']['0'] = 'y' * 10
    views[5]['nominal_request_accepted'] = False
    return dict(protocol='joint-lane-phase-preaction-v3', case_id='fixture', focal_tls='fixture_tls',
        horizon_sec=10, information_time_sec=600., reference_index=6,
        lane_nodes=lanes, movement_nodes=movements,
        link_nodes=[dict(link_index=i, movement_ids=[f'm{i}']) for i in range(4)],
        conflicts=dict(full_context_complete=True, foes=[[0, 2], [1, 2], [0, 3]],
                       response=[[0, 2], [3, 0]]),
        executor_context={name: float(i) for i, name in enumerate(EXECUTOR_FIELDS)},
        candidate_views=views, Y='forbidden labels', costs='forbidden future costs')


def value(features, name, candidate=0, movement=0):
    return features[candidate, movement, FEATURE_NAMES.index(name)]


def test_fixed_dimensions_and_exact_observed_lane_features():
    result = build_joint_features(graph_fixture())
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 838
    assert result.shape == (8, 4, 838) and np.isfinite(result).all()
    assert value(result, 'incoming_vehicle_count') == 3
    assert value(result, 'incoming_halted_count') == 2
    assert value(result, 'incoming_vehicle_within_2s_limit_distance') == 1
    assert value(result, 'incoming_vehicle_within_5s_limit_distance') == 2
    assert value(result, 'incoming_vehicle_within_10s_limit_distance') == 2
    assert value(result, 'incoming_nearest_distance_m') == 10
    assert value(result, 'internal_vehicle_count') == 1
    assert value(result, 'nominal_request_accepted', candidate=5) == 0


def test_directed_conflicts_have_correct_orientation():
    result = build_joint_features(graph_fixture())
    assert value(result, 'yields_to_link_count') == 1
    assert value(result, 'has_priority_over_link_count') == 1
    assert value(result, 'yields_to_t00_priority_vehicle_count') == 2
    assert value(result, 'yields_to_t00_yielding_vehicle_count') == 0
    assert value(result, 'has_priority_over_t00_priority_vehicle_count') == 0
    assert value(result, 'has_priority_over_t00_yielding_vehicle_count') == 5
    assert value(result, 'foe_t00_can_serve_vehicle_count') == 7


def test_shared_lanes_are_counted_once_within_each_service_relation():
    result = build_joint_features(graph_fixture())
    # Both m0 and m1 use a, and both conflict with m2. Their lane supply is not doubled.
    assert value(result, 'foe_link_count', movement=2) == 2
    assert value(result, 'foe_t00_G_link_fraction', movement=2) == .5
    assert value(result, 'foe_t00_g_link_fraction', movement=2) == .5
    assert value(result, 'foe_t00_can_serve_vehicle_count', movement=2) == 3
    assert value(result, 'foe_t00_can_serve_near_vehicle_count', movement=2) == 2
    assert value(result, 'foe_t00_can_serve_halted_count', movement=2) == 2
    assert value(result, 'other_t00_can_serve_vehicle_count', movement=2) == 8
    # Separate character subsets overlap for a shared lane, without an invented split.
    assert value(result, 'foe_t00_priority_vehicle_count', movement=2) == 3
    assert value(result, 'foe_t00_yielding_vehicle_count', movement=2) == 3


def test_signal_time_order_priority_and_yellow_case_remain_visible():
    result = build_joint_features(graph_fixture())
    assert value(result, 'own_t00_G_link_fraction') == value(result, 'own_t00_G_link_fraction', 1)
    assert value(result, 'own_t01_G_link_fraction') == 1
    assert value(result, 'own_t01_G_link_fraction', 1) == 0
    assert value(result, 'own_t00_g_link_fraction', 2) == 1
    assert value(result, 'own_t00_Y_link_fraction', 3) == 1
    assert value(result, 'own_t00_y_link_fraction', 4) == 1
    assert value(result, 'own_t00_can_serve_vehicle_count', 3) == 0


def test_identifier_renaming_and_storage_order_do_not_change_features():
    graph = graph_fixture(); before = deepcopy(graph)
    expected = build_joint_features(graph)
    assert graph == before
    names = {lane['lane_id']: f'cityless_{i}' for i, lane in enumerate(reversed(graph['lane_nodes']))}
    links = {i: 100 - i * 7 for i in range(4)}
    renamed = deepcopy(graph)
    renamed.update(case_id='unrelated_case', focal_tls='renamed_tls', information_time_sec=9999.,
                   reference_index=2, Y=[123456], costs=[-999])
    for lane in renamed['lane_nodes']:
        lane['lane_id'] = names[lane['lane_id']]; lane['edge_id'] = 'unrelated_edge'
    for i, movement in enumerate(renamed['movement_nodes']):
        movement['movement_id'] = f'renamed_m{i}'
        movement['incoming_edge'] = movement['outgoing_edge'] = 'unrelated_edge'
        for field in ['incoming_lanes', 'outgoing_lanes', 'internal_lanes']:
            movement[field] = [names[lane] for lane in movement[field]]
        movement['link_indices'] = [links[link] for link in movement['link_indices']]
    for node in renamed['link_nodes']:
        node['link_index'] = links[node['link_index']]; node['movement_ids'] = ['irrelevant_join_copy']
    for field in ['foes', 'response']:
        renamed['conflicts'][field] = [[links[i], links[j]] for i, j in renamed['conflicts'][field]]
    for view in renamed['candidate_views']:
        view['link_sequences'] = {str(links[int(k)]): v for k, v in view['link_sequences'].items()}
    renamed['lane_nodes'].reverse(); renamed['link_nodes'].reverse(); renamed['candidate_views'].reverse()
    renamed['movement_nodes'].reverse()
    np.testing.assert_array_equal(build_joint_features(renamed)[:, ::-1], expected)


def test_unsupported_signals_and_incomplete_conflicts_stop_instead_of_merging_states():
    graph = graph_fixture(); graph['candidate_views'][0]['link_sequences']['0'] = 'o' * 10
    with pytest.raises(ValueError, match='seven-character'):
        build_joint_features(graph)
    graph = graph_fixture(); graph['conflicts']['full_context_complete'] = False
    with pytest.raises(ValueError, match='complete static conflict'):
        build_joint_features(graph)
