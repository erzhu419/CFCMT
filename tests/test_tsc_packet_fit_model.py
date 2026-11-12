"""Synthetic mixture likelihood, analytic gradients and single-solve receipt."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tsc_target_packet_training_20261109 import fit_model as f


def heads(n=3, k=2):
    rng = np.random.default_rng(42)
    result = {}
    for name in f.HEAD_NAMES:
        X = rng.normal(0, .2, (n, k, 38))
        X[:, :, 0] = 1.
        result[name] = dict(X=X, prior=np.tile([.3, .7], (n, 1)),
                           risk=np.ones((n, k), bool), y=np.array([1., 0., 1.])[:n])
    result['motion'].update(upper=np.full((n, k), 4.), a=np.full((n, k), 2.), b=np.full((n, k), 4.5),
                            y=np.array([1.2, 2.3, 2.8])[:n])
    result['start']['risk'][0, 0] = False
    result['lane_change'].update(mask=np.ones((n, k, 3), bool), y=np.array([0., 1., 2.])[:n])
    if n > 1:
        result['yield_entry']['risk'][1, 1] = False
        result['lane_change']['risk'][1, 0] = False
    result['lane_change']['mask'][0, 0, 0] = False
    return result


def test_analytic_gradient_matches_all190_finite_differences_away_from_clip_kinks():
    data = heads()
    rng = np.random.default_rng(19)
    theta = rng.normal(0, .05, 190)
    theta[0] = -.7  # All motion means strictly inside their projected interval.
    loss, gradient = f.objective(theta, data)
    numerical = np.zeros_like(theta)
    epsilon = 1e-6
    for index in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (f.objective(plus, data)[0]-f.objective(minus, data)[0])/(2*epsilon)
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-8)


def test_structural_zero_intents_retain_prior_mass_and_match_manual_mixture_nll():
    data = heads(n=1)
    for row in data.values():
        row['X'][:] = 0
        row['X'][:, :, 0] = 1
        row['prior'][:] = [.25, .75]
    data['motion'].update(upper=np.array([[2., 4.]]), y=np.array([2.]), risk=np.array([[False, True]]))
    data['start'].update(risk=np.array([[False, True]]), y=np.array([1.]))
    data['yield_entry'].update(risk=np.array([[True, False]]), y=np.array([0.]))
    data['lane_change'].update(risk=np.array([[False, True]]), y=np.array([0.]),
                              mask=np.array([[[False, False, False], [True, True, False]]]))
    theta = np.zeros(190)
    theta[0] = -1.  # Motion means1/3 give identical density at speed2, despite entry riskFalse.
    theta[38], theta[76], theta[114], theta[152] = np.log(3), np.log(2), np.log(2), np.log(5)
    loss, gradient = f.objective(theta, data)
    # All five intercepts are nonzero, yet none contributes an L2 penalty.
    expected = .5*np.log(2*np.pi)+.5-np.log(.75*.75)-np.log(.25/3+.75)-np.log(.75*2/3)
    assert loss == pytest.approx(expected)
    padded = deepcopy(data)
    for name, row in padded.items():
        row['X'] = np.pad(row['X'], ((0, 0), (0, 1), (0, 0)))
        for key in ('prior', 'risk')+ (('upper', 'a', 'b') if name == 'motion' else ()):
            row[key] = np.pad(row[key], ((0, 0), (0, 1)))
        if name == 'lane_change':
            row['mask'] = np.pad(row['mask'], ((0, 0), (0, 1), (0, 0)))
    padded_loss, padded_gradient = f.objective(theta, padded)
    assert padded_loss == pytest.approx(loss)
    np.testing.assert_allclose(padded_gradient, gradient, rtol=0, atol=1e-14)
    changed = deepcopy(data)
    changed['start']['risk'][:] = False
    with pytest.raises(ValueError, match='zero prior-supported likelihood'):
        f.objective(theta, changed)


def test_zero_initial_motion_uses_feasible_side_and_actual_loss_decreases():
    data = heads(n=1)
    data['motion']['X'][:] = 0
    data['motion']['X'][:, :, 0] = 1
    data['motion']['upper'][:] = 3.
    data['motion']['y'][:] = 1.
    zero = np.zeros(190)
    loss, gradient = f.objective(zero, data)
    assert gradient[0] == pytest.approx(2.)
    negative = zero.copy()
    negative[0] = -1e-4
    assert f.objective(negative, data)[0] < loss
    assert (f.objective(negative, data)[0]-loss)/-1e-4 == pytest.approx(gradient[0], abs=1e-4)
    # An observed speed above the immutable physical ceiling cannot be learned
    # by a positive residual; do not produce a spurious ascent-at-kink gradient.
    data['motion']['y'][:] = 4.
    assert f.objective(zero, data)[1][0] == 0
    data['motion']['y'][:] = 1.
    lower = zero.copy()
    lower[0] = -3.
    assert f.objective(lower, data)[1][0] == pytest.approx(-1.)


def test_joint_toy_fit_uses_one_frozen_solve_and_learns_motion_slowdown(monkeypatch):
    data = heads()
    calls = []
    real_minimize = f.minimize

    def once(fun, initial, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(initial, np.zeros(190))
        return real_minimize(fun, initial, **kwargs)
    monkeypatch.setattr(f, 'minimize', once)
    coefficients, receipt = f.fit_joint(data)
    assert len(calls) == receipt['optimizer_calls'] == receipt['logical_fit_attempts'] == 1
    assert calls[0]['method'] == 'L-BFGS-B' and calls[0]['jac'] is True
    assert calls[0]['options'] == dict(maxiter=200, gtol=1e-7)
    assert receipt['success'] and receipt['status'] == 'PASS_JOINT_OPTIMIZATION'
    assert receipt['final_loss'] < receipt['initial_loss']
    assert set(coefficients) == set(f.COEFFICIENT_VECTORS)
    assert all(len(vector) == 38 for vector in coefficients.values())
    assert coefficients['motion'][0] < -.1


def test_nonconvergence_returns_terminal_receipt_without_retry(monkeypatch):
    calls = []

    def fail(fun, initial, **kwargs):
        calls.append(1)
        loss, gradient = fun(initial)
        return SimpleNamespace(success=False, status=1, message='iteration limit',
            nit=200, nfev=201, njev=201, x=initial, fun=loss, jac=gradient)
    monkeypatch.setattr(f, 'minimize', fail)
    coefficients, receipt = f.fit_joint(heads())
    assert coefficients is None
    assert receipt['status'] == 'HOLD_JOINT_OPTIMIZATION'
    assert not receipt['success'] and receipt['iterations'] == 200
    assert len(calls) == receipt['optimizer_calls'] == 1


def test_l2_excludes_all_five_intercepts():
    # Read the objective's difference from changing a zero-exposure coordinate;
    # its only contribution must be the frozen non-intercept penalty.
    data = heads(n=1)
    for row in data.values():
        row['X'][:, :, 37] = 0
    zero = np.zeros(190)
    shifted = zero.copy()
    shifted[37] = 2.
    loss0, _ = f.objective(zero, data)
    loss1, gradient = f.objective(shifted, data)
    assert loss1-loss0 == pytest.approx(1e-3*4)
    assert gradient[37] == pytest.approx(2e-3*2)
