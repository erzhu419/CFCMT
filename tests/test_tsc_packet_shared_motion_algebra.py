"""Shared coefficients, saturation and genuine one-sided kink derivatives."""
import numpy as np
import pytest

from experiments.tsc_packet_shared_motion_20261110.algebra import motion_response, shared_gap


def actor(speed_feature, *, x=10., P=3.):
    phi = np.zeros(38); phi[0] = 1.; phi[1] = speed_feature
    return dict(phi=phi, b=4.5, a=2., P=P, x=x, length=5., min_gap=2.)


def test_positive_raw_residual_is_reapplied_to_physical_cap_and_effective_e_is_zero():
    beta = np.zeros(38); beta[0] = 4.
    response = motion_response(beta, actor(2))
    assert response['raw_r'] == 2 and response['effective_e'] == 0
    assert response['physical_proposal_mps'] == 3
    assert response['physical_proposal_jacobian'] == [0.]*38


def test_zero_beta_has_opposite_front_rear_one_sided_effects_and_no_fake_jacobian():
    beta, direction = np.zeros(38), np.zeros(38); direction[1] = -1
    focal, neighbour = actor(2), actor(4, x=9)
    rear = shared_gap(beta, focal, neighbour, 'rear', direction=direction)
    front = shared_gap(beta, focal, neighbour, 'front', direction=direction)
    assert rear['focal']['effective_e_directional'] == -2
    assert rear['neighbour']['effective_e_directional'] == -4
    assert rear['gap_directional'] == 2 and front['gap_directional'] == -2
    assert rear['gap_jacobian'] is None and rear['squared_hinge_jacobian'] is None
    h = 1e-7
    for relation, result in [('rear', rear), ('front', front)]:
        shifted = shared_gap(beta+h*direction, focal, neighbour, relation)
        assert (shifted['gap_margin_m']-result['gap_margin_m'])/h == pytest.approx(result['gap_directional'], abs=1e-7)
        assert (shifted['squared_hinge']-result['squared_hinge'])/h == pytest.approx(result['squared_hinge_directional'], abs=2e-5)


def test_smooth_branch_jacobian_matches_direction_without_independent_actor_coefficients():
    beta, direction = np.zeros(38), np.zeros(38); beta[0] = -1.; direction[1] = -1
    result = shared_gap(beta, actor(2), actor(4, x=9), 'rear', direction=direction)
    assert result['jacobian_available']
    assert np.asarray(result['gap_jacobian'])@direction == result['gap_directional'] == 2
    assert np.asarray(result['squared_hinge_jacobian'])@direction == result['squared_hinge_directional']


def test_declared_lower_clip_and_zero_speed_floor_keep_their_correct_one_sided_rules():
    beta, direction = np.zeros(38), np.zeros(38); beta[0] = -4.5; direction[0] = 1
    lower = motion_response(beta, actor(0, P=6), direction=direction)
    assert lower['effective_e_jacobian'] is None and lower['effective_e_directional'] == 1
    assert motion_response(beta, actor(0, P=6), direction=-direction)['effective_e_directional'] == 0
    beta[0] = -1.
    floor = motion_response(beta, actor(0, P=1), direction=direction)
    assert floor['physical_proposal_jacobian'] is None and floor['physical_proposal_directional'] == 1
    assert motion_response(beta, actor(0, P=1), direction=-direction)['physical_proposal_directional'] == 0

    stopped = motion_response(np.zeros(38), actor(0, P=0), direction=direction)
    assert stopped['physical_proposal_jacobian'] == [0.]*38
    assert stopped['physical_proposal_directional'] == 0
