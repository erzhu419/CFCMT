"""Use only original PP masks, with no current-state or future substitution."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_pp_plan_20261109.plan_features import pp_plan_feature


def example():
    return dict(pp=1, gp=2, current_request_index=0, current_state='Gr',
                states=['Gr', 'rG', 'GG'], protected={'road': [1023, 1008, 0]}), dict(edge='road')


def test_pp_candidate_is_selected_even_when_gp_and_current_are_different():
    row, flight = example()
    assert pp_plan_feature(row, flight) == dict(protected_seconds=6, mask=1008,
                                               planned_state='rG', pp_index=1)


def test_partial_exposure_is_counted_without_restricting_it_to_zero_four_or_ten():
    row, flight = example()
    row['protected']['road'][1] = 0b1111100000
    assert pp_plan_feature(row, flight)['protected_seconds'] == 5
    row['protected']['road'][1] = (1 << 13)-1
    with pytest.raises(ValueError, match='original ten-bit'):
        pp_plan_feature(row, flight)


def test_zero_and_full_ten_second_plans_are_preserved():
    row, flight = example()
    for seconds, mask in [(0, 0), (10, (1 << 10)-1)]:
        row['protected']['road'][1] = mask
        result = pp_plan_feature(row, flight)
        assert result['protected_seconds'] == seconds and result['mask'] == mask


def test_future_fields_cannot_change_the_plan_and_inputs_are_not_mutated():
    row, flight = example()
    before = deepcopy((row, flight))
    expected = pp_plan_feature(row, flight)
    assert (row, flight) == before
    changed = dict(row, future_signal='rr', realized_green_seconds=0,
                   future_outcome='straight', future_row={'t': 500, 'states': ['rr']})
    assert pp_plan_feature(changed, flight) == expected
