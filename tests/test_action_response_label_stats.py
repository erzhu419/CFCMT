"""Exact full-vector response classes, native cost units and seed weighting."""
import json

import pytest

from experiments.tsc_action_response_label_audit_20261109.label_stats import audit_group, summarize


def group(case_id, seed, Y, stopped450, stopped10=None, lanes=8, pp=0, city='cologne'):
    return audit_group(case_id, city, seed, Y, [value/(450*lanes) for value in stopped450], pp, lanes,
                       stopped10 if stopped10 is not None else [0]*8, stopped450)


def test_exact_equivalence_uses_every_ordered_movement_and_both_count_outputs():
    Y = [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]],
         [[0, 0], [0, 0]], [[2, 0], [0, 1]], [[1, 0], [0, 2]], [[0, 0], [1, 1]]]
    result = group('full', 1, Y, list(range(0, 80, 10)), list(range(8)))
    assert len(result['classes']) == 7
    assert [row['members'] for row in result['classes'] if row['action_count'] > 1] == [[0, 1]]
    assert result['statistics']['identical_Y_pairs'] == 1
    assert result['statistics']['sameY_pair_cost450_different'] == 1
    assert result['statistics']['sameY_pair_cost10_different'] == 1
    assert all(row['range_450']['stopped_vehicle_seconds_span'] == 0
               for row in result['classes'] if row['action_count'] == 1)
    assert result['statistics']['alias_group_450'] == 1
    json.dumps(result, allow_nan=False)


def test_zero_response_all_eight_actions_and_nonzero_PP_index_are_preserved():
    Y = [[[0, 0], [0, 0]] for _ in range(8)]
    result = group('zero', 1, Y, [0, 0, 10, 20, 30, 40, 50, 60], pp=3)
    assert len(result['classes']) == 1
    assert result['classes'][0]['members'] == list(range(8))
    assert result['classes'][0]['range_10']['stopped_vehicle_seconds_span'] == 0
    assert result['statistics']['identical_Y_pairs'] == 28
    assert result['statistics']['sameY_pair_cost450_different'] == 27
    assert result['statistics']['sameY_pair_cost10_equal_but450_different'] == 27
    assert result['statistics']['pp_alias_group_450'] == 1
    deltas = {row['candidate_index']: row for row in result['pp_same_response_deltas']}
    assert 3 not in deltas
    assert deltas[0]['stopped_vehicle_seconds_450'] == -20
    assert deltas[7]['stopped_vehicle_seconds_450'] == 40
    assert deltas[0]['mean_halted_per_lane_450'] == pytest.approx(-20/(450*8))
    tied = group('tied', 2, Y, [20]*8)
    assert tied['statistics']['alias_group_450'] == 0
    assert tied['statistics']['max_class_span_450_vehicle_seconds'] == 0


def test_native_lane_units_and_group_then_seed_equal_weights_keep_cities_separate():
    same = [[[0, 0]] for _ in range(8)]
    unique = [[[index, 0]] for index in range(8)]
    a = group('a', 1, same, [0]*7+[450])
    other_lanes = group('other', 1, same, [0]*7+[450], lanes=144, city='jinan')
    assert a['classes'][0]['range_450']['stopped_vehicle_seconds_span'] == 450
    assert other_lanes['classes'][0]['range_450']['stopped_vehicle_seconds_span'] == 450
    assert a['classes'][0]['range_450']['mean_halted_per_lane_span'] == pytest.approx(1/8)
    assert other_lanes['classes'][0]['range_450']['mean_halted_per_lane_span'] == pytest.approx(1/144)
    rows = [a, group('b', 1, unique, [0]*7+[999]),
            group('c', 2, same, [0]*7+[900]), group('d', 2, same, [0]*7+[900]),
            other_lanes, group('other2', 1, same, [0]*7+[450], lanes=144, city='jinan')]
    result = summarize(rows)
    source = result['cities']['cologne']
    assert source['per_seed'][0]['group_mean_statistics']['max_class_span_450_vehicle_seconds'] == 225
    assert source['per_seed'][1]['group_mean_statistics']['max_class_span_450_vehicle_seconds'] == 900
    assert source['equal_seed_statistics']['max_class_span_450_vehicle_seconds'] == 562.5
    # Eight singleton classes in b contribute one group's zero, not eight extra independent units.
    assert source['equal_seed_statistics']['mean_class_span_450_vehicle_seconds'] == 562.5
    assert result['cities']['jinan']['equal_seed_statistics']['max_class_span_450_vehicle_seconds'] == 450
    assert result['group_count'] == 6 and result['branch_count'] == 48
    assert set(result['cities']) == {'cologne', 'jinan'}
