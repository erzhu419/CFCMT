"""Exact frozen-reference selection and actual physical-path matching only."""
from copy import deepcopy

import pytest

from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import parallel_receivers
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.audit_exclusions import (
    select_references, match_observed_entry)


def ref(time, packet, label):
    return dict(seed=1, time_sec=time, packet_id=packet, y=label)


def test_exact_positive_key_difference_preserves_full_reservoir_and_labels():
    y1, y2, y0 = ref(1, 'a', 1), ref(2, 'b', 1), ref(3, 'c', 0)
    left, stay, right = ref(1, 'd', 0), ref(2, 'e', 1), ref(3, 'f', 2)
    reservoir = {'1': dict(yield_entry=[y1, y2, y0], lane_change=[left, stay, right])}
    kept = dict(yield_entry=[dict(y2, candidate_intents=['END']), y0], lane_change=[stay, right])
    original = deepcopy(reservoir)
    selected = select_references(reservoir, kept)
    assert selected['counts'] == dict(excluded_yield_positive=1, retained_yield_positive=1, excluded_lane_positive=1)
    assert [r['packet_id'] for r in selected['by_seed'][1]['yield_entry']] == ['a', 'b']
    assert selected['by_seed'][1]['lane_change'] == [dict(left, partition='excluded')]
    assert reservoir == original
    kept['yield_entry'][0]['y'] = 0
    with pytest.raises(ValueError, match='unchanged members'):
        select_references(reservoir, kept)


def test_entry_matches_actual_ordinary_lane_after_lateral_and_keeps_G_g_distinct():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    direct = match_observed_entry(topology, ['a0', 'I0', 'out0', 'out1'], {'J': 'Gg'})
    assert direct['event_kind'] == 'stopline_entry' and direct['actual_signal'] == 'G'
    after_lateral = match_observed_entry(topology, ['a0', 'a1', 'I1'], {'J': 'Gg'})
    assert after_lateral['actual_signal'] == 'g'
    assert after_lateral['matches'][0]['from_lane'] == 'a1'
    assert after_lateral['matches'][0]['to_lane'] == 'out1'


def test_ordinary_lane_change_is_not_entry_and_unknown_path_is_not_guessed():
    static, unused = parallel_receivers()
    topology = RoadTopology(static)
    lateral = match_observed_entry(topology, ['a0', 'a1'], {'J': 'Gg'})
    assert lateral == dict(event_kind='ordinary_lane_change_only', actual_signal='no_stopline_entry', matches=[])
    unknown = match_observed_entry(topology, ['a0', 'I1'], {'J': 'Gg'})
    assert unknown['actual_signal'] == 'unknown' and unknown['matches'] == []
