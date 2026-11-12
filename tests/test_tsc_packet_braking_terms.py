"""Fixed-dt braking algebra; no kernel changes or recorded future inputs."""
import pytest

from experiments.tsc_packet_exclusion_mechanisms_20261110.braking_terms import (
    braking_distance_euler, discrete_safe_speed,
)


def test_one_second_braking_has_zero_travel_at_or_below_one_deceleration_step():
    for speed in (0., .5, 4.5):
        assert braking_distance_euler(speed, 4.5) == 0.
    assert braking_distance_euler(9., 4.5) == 4.5


def test_closed_distance_matches_the_defining_finite_sum():
    for b in (1.2, 3., 4.5):
        for speed in (.3, 4., 4.1, 8., 20.7):
            expected = sum(max(speed-n*b, 0.) for n in range(1, 30))
            assert braking_distance_euler(speed, b) == pytest.approx(expected, abs=1e-12)


def test_distance_is_continuous_and_piecewise_linear_across_b_multiples():
    b, delta = 1.1, 1e-7
    for k in range(1, 5):
        boundary = k*b
        value = b*k*(k-1)/2
        assert braking_distance_euler(boundary, b) == pytest.approx(value, abs=1e-12)
        assert braking_distance_euler(boundary-delta, b) == pytest.approx(value-(k-1)*delta, abs=1e-12)
        assert braking_distance_euler(boundary+delta, b) == pytest.approx(value+k*delta, abs=1e-12)


def test_inverse_round_trip_with_independent_headway_and_heterogeneous_braking():
    for speed, b, tau, leader_speed, leader_b in (
            (.5, 4., .7, .2, 2.), (8., 4., 1., 3.7, 1.1),
            (10.4, 3., 1.3, 8.1, 4.5), (4.2, 1.7, .8, 2.8, .9)):
        gap = tau*speed+braking_distance_euler(speed, b)-braking_distance_euler(leader_speed, leader_b)
        assert gap >= 0.
        recovered = discrete_safe_speed(gap, leader_speed, b, leader_b, tau)
        assert recovered == pytest.approx(speed, abs=1e-12)
        assert tau*recovered+braking_distance_euler(recovered, b) == pytest.approx(
            gap+braking_distance_euler(leader_speed, leader_b), abs=1e-12)


def test_inverse_clips_only_negative_gap_and_keeps_literal_leader_deceleration():
    assert discrete_safe_speed(-5., 10., 4., 4., 1.) == 6.
    assert discrete_safe_speed(0., 10., 4., 5., 1.) == 4.5
    assert discrete_safe_speed(0., 0., 4., 5., 1.) == 0.
