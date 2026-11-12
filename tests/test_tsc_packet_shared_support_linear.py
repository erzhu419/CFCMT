"""Only explicit matrix products with hand-supplied vectors; no LP solver."""
import numpy as np
import pytest

from experiments.tsc_packet_shared_support_lp_20261110.linear import build_min_l1


def actor(index, *, multiplier=1., x=10., P=5., b=4.5, cell='linear'):
    phi = np.zeros(38); phi[index] = multiplier
    return dict(phi=phi, P=P, x=x, length=5., min_gap=2., b=b, cell=cell)


def vector(beta):
    return np.r_[beta, np.abs(beta)]


def test_linear_gap_direction_and_absolute_auxiliary_matrix_have_no_epsilon():
    actors = dict(focal=actor(0), rear=actor(0, multiplier=2., x=5., P=4.))
    lp = build_min_l1(actors, [dict(label='focal-ahead-rear', front='focal', rear='rear')])
    assert lp['A_ub'].shape == (83, 76)
    assert lp['bounds'][:38] == [(None, None)]*38
    assert lp['bounds'][38:] == [(0., None)]*38
    np.testing.assert_array_equal(lp['c'], np.r_[np.zeros(38), np.ones(38)])
    index = lp['constraint_labels'].index('gap/focal-ahead-rear')
    assert lp['b_ub'][index] == -1. and lp['A_ub'][index, 0] == 1.
    zero = np.zeros(38)
    assert lp['A_ub'][index]@vector(zero) > lp['b_ub'][index]
    beta = zero.copy(); beta[0] = -1.
    assert np.all(lp['A_ub']@vector(beta) <= lp['b_ub'])
    beta[0] += 1e-12
    assert lp['A_ub'][index]@vector(beta) > lp['b_ub'][index]
    beta[0] = -1.
    wrong_abs = vector(beta); wrong_abs[38] = 0.
    assert np.any(lp['A_ub']@wrong_abs > lp['b_ub'])


def test_plateau_zero_floor_and_P0_constant_use_the_registered_affine_cells():
    actors = dict(plateau=actor(0, P=2., cell='plateau'),
                  floor=actor(1, P=2., cell='zero_floor'),
                  constant=actor(2, P=0., cell='plateau'))
    lp = build_min_l1(actors, [])
    beta = np.zeros(38); beta[0] = 1.; beta[1] = -2.; beta[2] = -100.
    assert np.all(lp['A_ub']@vector(beta) <= lp['b_ub'])
    assert not any(label.startswith('constant/') for label in lp['constraint_labels'])
    for key, expected in [('plateau', 12.), ('floor', 10.), ('constant', 10.)]:
        endpoint = lp['actor_endpoints'][key]
        assert endpoint['constant_m']+endpoint['beta_coefficients']@beta == expected
    beta[1] = -1.
    assert np.any(lp['A_ub']@vector(beta) > lp['b_ub'])
    with pytest.raises(ValueError, match='P<=b'):
        build_min_l1(dict(invalid=actor(0, P=5., b=4.5, cell='zero_floor')), [])
