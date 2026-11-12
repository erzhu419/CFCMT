"""Algebraic braking diagnostics for fixed one-second speed-first updates.

These are diagnostic following bounds, not a complete native speed prediction.
They omit acceleration, allowed-speed, signals, receiving and conflict limits.
The leader deceleration is the supplied value: no hidden max(b, bL) or numeric
epsilon is applied. Headway tau remains independent of the fixed dt=1 second.
"""
from math import floor, sqrt


def braking_distance_euler(speed_mps, decel_mps2):
    """D(v,b) = sum(n>=1, max(v-n*b,0)) for dt=1 and positive b."""
    if decel_mps2 <= 0:
        raise ValueError('Braking deceleration must be positive')
    k = max(floor(speed_mps/decel_mps2), 0)
    return k*speed_mps-decel_mps2*k*(k+1)/2


def discrete_safe_speed(net_gap_m, leader_speed_mps, follower_decel_mps2,
                        leader_decel_mps2, tau_s):
    """Invert tau*v+D(v,b)=max(g,0)+D(vL,bL) exactly by linear segments.

    For kb <= v <= (k+1)b, D(v,b)=k*v-b*k*(k+1)/2. Segment k is
    located from the quadratic values at v=kb; the inverse itself is closed
    form and uses no iterative optimizer or fitted threshold. Positive
    follower deceleration and headway make the nonnegative root unique.
    """
    if follower_decel_mps2 <= 0 or tau_s <= 0:
        raise ValueError('Follower braking deceleration and headway must be positive')
    b = follower_decel_mps2
    budget = max(net_gap_m, 0.)+braking_distance_euler(leader_speed_mps, leader_decel_mps2)
    coefficient = 2*tau_s-1
    k = max(floor((sqrt(coefficient**2+8*budget/b)-coefficient)/2), 0)
    return (budget+b*k*(k+1)/2)/(tau_s+k)
