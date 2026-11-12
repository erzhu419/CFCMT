"""Reachable executor ordering and incomplete-information reconstruction cases."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_service_alignment_20261109.service_window import extend_window

FULL = 1023
SUFFIX = FULL ^ ((1 << 4)-1)


def example():
    # Request1 is the current phase. The fixed feasible order is not rotated.
    return dict(t=100, tls='signal', states=['A', 'B', 'C'], pp=0, gp=2,
                base=[18, 20, 30], protected={'north': [SUFFIX, FULL, 0],
                                             'east': [0, 0, SUFFIX]})


def test_recovers_current_from_last_pp_and_extends_queue_units_and_masks():
    row = example(); saved = deepcopy(row)
    result = extend_window(row, 'B')
    meta = result['service_window']
    assert meta['current_request_index'] == 1
    assert meta['clearance_sec'] == 4
    assert meta['queue_demand'] == [3, 2, 5]
    assert meta['base10_reconstructed'] == row['base']
    assert meta['green_seconds_extended'] == [9, 13, 9]
    assert result['base'] == [27, 26, 45]
    assert result['protected']['north'] == [8191 ^ 15, 8191, 0]
    assert result['protected']['east'] == [0, 0, 8191 ^ 15]
    assert all(new & FULL == old for edge in row['protected']
               for old,new in zip(row['protected'][edge], result['protected'][edge]))
    assert row == saved


def test_current_left_phase_need_not_have_any_full_straight_mask():
    row = example()
    row['protected'] = {'north': [SUFFIX, 0, 0], 'east': [0, 0, SUFFIX]}
    result = extend_window(row, 'B')
    assert result['service_window']['current_request_index'] == 1
    assert result['base'] == [27, 26, 45]
    assert result['protected']['north'][1] == 0


def test_incorrect_previous_request_cannot_be_silently_assumed_first():
    with pytest.raises(ValueError, match='preceding'):
        extend_window(example(), None)
    with pytest.raises(ValueError, match='stable green'):
        extend_window(example(), 'A')


def test_non_suffix_or_inconsistent_clearance_is_rejected():
    row = example();row['protected']['east'][2] = 0b1010100000
    with pytest.raises(ValueError, match='suffix'):
        extend_window(row, 'B')
    row = example();row['protected']['east'][2] = FULL ^ 31
    with pytest.raises(ValueError, match='clearance'):
        extend_window(row, 'B')


def test_all_zero_switch_masks_do_not_identify_a_queue_extension():
    row = example();row['protected'] = {'north': [0, FULL, 0]}
    with pytest.raises(ValueError, match='cannot be identified'):
        extend_window(row, 'B')


def test_nonintegral_or_negative_queue_base_is_not_repaired_or_dropped():
    for bad in (19, -6):
        row = example();row['base'][0] = bad
        with pytest.raises(ValueError, match='queue service rate'):
            extend_window(row, 'B')


def test_locked_rows_do_not_need_or_allow_reconstruction():
    row = dict(states=['B'], base=[20], protected={'north': [FULL]})
    with pytest.raises(ValueError, match='unlocked'):
        extend_window(row, 'B')
