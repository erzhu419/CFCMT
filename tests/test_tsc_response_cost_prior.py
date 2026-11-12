"""Toy-only numerical acceptance; no real response or cost records are read."""
from dataclasses import replace

import numpy as np

from experiments.tsc_response_cost_readout_20261109.prior_ridge import (
    convert_prior, fit_dynamics, fit_reader, fit_source, predict, reader_predict, select_action,
)


def toy():
    X = np.array([[0., 2., 3.1], [1., -1., 3.1], [2., 3., 3.1], [4., 0., 3.1]])
    Y = np.column_stack((2*X[:, 0]-X[:, 1]+10, -X[:, 0]+.5*X[:, 1]-3))
    return X, Y, np.array([.1, .2, .3, .4])


def test_source_function_is_identical_after_analytic_coordinate_conversion():
    X, Y, w = toy()
    source = fit_source(X, Y, w)
    mean, scale = np.array([-2., 7., 1.]), np.array([3., .5, 2.])
    coef, intercept = convert_prior(source, mean, scale)
    probes = np.array([[-1., 4., 3.1], [8., -5., 3.1], [.5, .2, 7.]])
    converted = ((probes-mean)/scale)@coef+intercept
    np.testing.assert_allclose(converted, predict(source, probes), rtol=0, atol=1e-12)
    assert source.scaler.scale[2] == 1  # A constant static field stays well-defined.


def test_target_intercept_freely_corrects_source_mean_without_penalizing_the_offset():
    X, Y, w = toy()
    source = fit_source(X, Y+500, w)
    shifted_source = replace(source, intercept=source.intercept+10000)
    target = Y-20
    fitted = fit_dynamics(X, target, w, prior=source)
    shifted = fit_dynamics(X, target, w, prior=shifted_source)
    np.testing.assert_allclose(fitted.coef, shifted.coef, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted.intercept, shifted.intercept, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.average(predict(fitted, X), axis=0, weights=w),
                               np.average(target, axis=0, weights=w), rtol=0, atol=1e-12)


def test_zero_prior_matches_weighted_standardized_ridge_with_free_intercept():
    X, Y, w = toy()
    fitted = fit_dynamics(X, Y, w)
    mean = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X-mean)**2, axis=0, weights=w)); scale[2] = 1
    Z = (X-mean)/scale
    zmean, ymean = np.average(Z, axis=0, weights=w), np.average(Y, axis=0, weights=w)
    Zc, Yc = Z-zmean, Y-ymean
    coef = np.linalg.solve(Zc.T@(w[:, None]*Zc)+np.eye(3), Zc.T@(w[:, None]*Yc))
    np.testing.assert_allclose(fitted.coef, coef, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted.intercept, ymean-zmean@coef, rtol=0, atol=1e-12)
    zero = replace(fitted, coef=np.zeros_like(fitted.coef), intercept=np.array([100., -200.]))
    np.testing.assert_allclose(predict(fit_dynamics(X, Y, w, prior=zero), X), predict(fitted, X), rtol=0, atol=1e-12)


def test_explicit_weight_mass_is_preserved_inputs_are_unchanged_and_duplicate_halfweights_are_fair():
    X, Y, w = toy()
    originals = X.copy(), Y.copy(), w.copy()
    source = fit_source(X+1, Y-1, w)
    prior_coef = source.coef.copy()
    fitted = fit_dynamics(X, Y, w, prior=source)
    duplicate = fit_dynamics(np.repeat(X, 2, axis=0), np.repeat(Y, 2, axis=0), np.repeat(w/2, 2), prior=source)
    np.testing.assert_allclose(predict(fitted, X), predict(duplicate, X), rtol=0, atol=1e-12)
    heavier = fit_dynamics(X, Y, 2*w, prior=source)
    assert not np.allclose(fitted.coef, heavier.coef)  # No hidden sum-to-one or unit-mean renormalization.
    for actual, original in zip((X, Y, w), originals):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(source.coef, prior_coef)


def test_scale_only_reader_keeps_PP_exact_zero_and_round12_ties_prefer_PP():
    F = np.array([[0., 0., 0.], [1., 3., 0.], [2., -1., 0.], [-2., 2., 0.]])
    delta = np.array([0., -.2, .4, -.1])
    w = np.array([.1, .2, .3, .4])
    reader = fit_reader(F, delta, w)
    predictions = reader_predict(reader, F)
    assert predictions[0] == 0
    np.testing.assert_array_equal(reader_predict(reader, np.zeros((3, 3))), np.zeros(3))
    assert not hasattr(reader, 'intercept')
    assert reader.scale[2] == 1
    scores = np.array([-.4e-12, 7., .2])
    before = scores.copy()
    assert select_action(scores, 1) == 1  # PP is structural zero, and -4e-13 rounds to the same score.
    np.testing.assert_array_equal(scores, before)
    assert select_action(np.array([-.1, 0., -.1]), 1) == 0  # Otherwise original index breaks ties.
    assert select_action(np.array([.1, -10., .2]), 1) == 1
