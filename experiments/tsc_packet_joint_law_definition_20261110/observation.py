"""A new observation hypothesis, separate from the physical motion law.

For physical speed mu, censor Normal(mu,1) at zero. If the previous vehicle
was stopped, condition the resulting observation on the physical start band.
The reference measure is delta_0 + Lebesgue on (0,infinity). Observation Y
never drives the physical resolver or lateral gap tests. This is neither the
old untruncated Gaussian score nor a complete physical transition density.
Mu is the Gaussian location before censoring, not the resulting mean of Y.
"""
from math import exp, inf, pi, sqrt

from scipy.special import ndtr


START_THRESHOLD = .1
_NORMAL_SCALE = sqrt(2*pi)


def _parameters(mu, previously_stopped):
    if not previously_stopped:
        return float(ndtr(-mu)), 0., inf, 1.
    if mu >= START_THRESHOLD:
        normalizer = float(ndtr(mu-START_THRESHOLD))
        return 0., START_THRESHOLD, inf, normalizer
    normalizer = float(ndtr(START_THRESHOLD-mu))
    return float(ndtr(-mu))/normalizer, 0., START_THRESHOLD, normalizer


def speed_measure(mu, previously_stopped):
    """Return the zero atom and support of the positive continuous part."""
    atom, lower, upper, unused_normalizer = _parameters(mu, previously_stopped)
    return dict(atom_at_zero=atom, lower_positive=lower, upper_positive=upper)


def speed_likelihood(y, mu, previously_stopped):
    """Evaluate mass at y=0, density at y>0; forbidden bands have mass zero.

    Physical start is determined by mu>=0.1. For an already moving vehicle,
    no start-band conditioning is applied. Sigma remains fixed at1 and is
    not an estimated nuisance parameter.
    """
    atom, lower, upper, normalizer = _parameters(mu, previously_stopped)
    if y == 0.:
        return atom
    if y < 0. or y < lower or y >= upper:
        return 0.
    return exp(-.5*(y-mu)**2)/(_NORMAL_SCALE*normalizer)
