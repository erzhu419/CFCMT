"""Finite conditional LC trees normalize and have exact fixed-mask gradients."""
from itertools import product

import pytest

from experiments.tsc_packet_shared_motion_20261110.categorical import ordered_tuple_probability


def branch(first, second):
    choices = {'left': ['stay', 'right'], 'stay': ['left', 'stay', 'right'], 'right': ['stay']}
    steps = [dict(packet_id='focal', action=first, allowed_choices=['left', 'stay', 'right']),
             dict(packet_id='neighbour', action=second, allowed_choices=choices[first])]
    return dict(all_requested_actions_accepted=second in choices[first], steps=steps)


def test_dynamic_mask_tree_normalizes_with_zero_mass_for_illegal_tuples():
    phi = {'focal': [1., .2]+[0.]*36, 'neighbour': [1., -.5]+[0.]*36}
    left, right = [.2, -.3]+[0.]*36, [-.1, .4]+[0.]*36
    results = [ordered_tuple_probability(branch(a, b), phi, left, right)
               for a, b in product(('left', 'stay', 'right'), repeat=2)]
    assert sum(row['q'] for row in results) == pytest.approx(1., abs=1e-14)
    for name in ('dq_left', 'dq_right'):
        assert [sum(row[name][k] for row in results) for k in range(38)] == pytest.approx([0.]*38, abs=1e-14)
    illegal = ordered_tuple_probability(branch('right', 'left'), phi, left, right)
    assert not illegal['legal'] and illegal['q'] == 0.
    assert illegal['dq_left'] == illegal['dq_right'] == [0.]*38
    assert illegal['logq'] is illegal['dlogq_left'] is illegal['dlogq_right'] is None


def test_shared_tuple_mass_and_log_mass_gradients_match_fixed_mask_finite_differences():
    phi = {'focal': [(k+1)/38 for k in range(38)],
           'neighbour': [(-1.)**k/(k+2) for k in range(38)]}
    left, right = [.1*(k-19)/38 for k in range(38)], [-.07*(k+1)/38 for k in range(38)]
    ordered = branch('left', 'right')
    result = ordered_tuple_probability(ordered, phi, left, right)
    epsilon = 1e-6
    for vector, gradient, log_gradient in ((left, 'dq_left', 'dlogq_left'),
                                            (right, 'dq_right', 'dlogq_right')):
        for k in range(38):
            original = vector[k]
            vector[k] = original+epsilon
            plus = ordered_tuple_probability(ordered, phi, left, right)
            vector[k] = original-epsilon
            minus = ordered_tuple_probability(ordered, phi, left, right)
            vector[k] = original
            assert result[gradient][k] == pytest.approx((plus['q']-minus['q'])/(2*epsilon), abs=1e-9)
            assert result[log_gradient][k] == pytest.approx((plus['logq']-minus['logq'])/(2*epsilon), abs=1e-9)
