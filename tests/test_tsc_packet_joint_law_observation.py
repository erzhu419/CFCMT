"""Normalization of censored speed atoms and physical start-band support."""
from math import sqrt, pi

import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from experiments.tsc_packet_joint_law_definition_20261110.observation import (
    speed_likelihood, speed_measure,
)


@pytest.mark.parametrize('mu', [0., .099, .1, 2.])
@pytest.mark.parametrize('previously_stopped', [False, True])
def test_zero_atom_plus_continuous_positive_mass_is_one(mu, previously_stopped):
    measure = speed_measure(mu, previously_stopped)
    positive, error = quad(lambda y: speed_likelihood(y, mu, previously_stopped),
        measure['lower_positive'], measure['upper_positive'], epsabs=1e-12, epsrel=1e-12)
    assert measure['atom_at_zero']+positive == pytest.approx(1., abs=1e-10)
    assert error < 1e-9


def test_start_band_is_preserved_including_the_exact_threshold():
    for mu in (0., .099):
        assert speed_likelihood(.05, mu, True) > 0.
        assert speed_likelihood(.1, mu, True) == 0.
        assert speed_likelihood(2., mu, True) == 0.
    for mu in (.1, 2.):
        assert speed_likelihood(0., mu, True) == 0.
        assert speed_likelihood(.099, mu, True) == 0.
        assert speed_likelihood(.1, mu, True) > 0.
    assert speed_likelihood(-.01, 0., False) == 0.


def test_exact_zero_is_an_atom_not_the_positive_density_limit():
    assert speed_likelihood(0., 0., False) == .5
    assert speed_likelihood(1e-12, 0., False) == pytest.approx(1/sqrt(2*pi))
    assert speed_likelihood(0., 0., True) == pytest.approx(.5/ndtr(.1))
    # Moving vehicles have a zero atom even when their physical speed is high;
    # the newly started band deliberately removes that atom.
    assert speed_likelihood(0., 2., False) == pytest.approx(ndtr(-2.))
    assert speed_likelihood(0., 2., True) == 0.
