"""Small weighted Ridge primitives for the prospective response/readout design."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class DynamicsModel:
    scaler: Scaler
    coef: np.ndarray
    intercept: np.ndarray
    single_output: bool


@dataclass(frozen=True)
class ReaderModel:
    scale: np.ndarray
    coef: np.ndarray


def _design(X, Y, w):
    x, y, weights = np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(w, dtype=float)
    if (x.ndim != 2 or y.ndim not in (1, 2) or y.shape[0] != len(x) or weights.shape != (len(x),) or
            not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or
            not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Expected finite aligned rows and nonnegative weights with positive total mass')
    return x, y, weights


def _mean(values, weights):
    return (weights[:, None]*values).sum(axis=0)/weights.sum()


def _scaler(x, weights):
    mean = _mean(x, weights)
    variance = _mean((x-mean)**2, weights)
    # A supported static field can be constant in a complete training fold.
    scale = np.where(np.ptp(x, axis=0) == 0, 1., np.sqrt(variance))
    scale = np.where(scale == 0, 1., scale)
    return Scaler(mean, scale)


def convert_prior(source_model, target_mean, target_scale):
    """Express the identical source function in target-standardized coordinates."""
    mean, scale = np.asarray(target_mean, dtype=float), np.asarray(target_scale, dtype=float)
    if (mean.shape != source_model.scaler.mean.shape or scale.shape != mean.shape or
            not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise ValueError('Target coordinates must match the source feature roster and have positive scales')
    coef = (scale/source_model.scaler.scale)[:, None]*source_model.coef
    intercept = source_model.intercept+((mean-source_model.scaler.mean)/source_model.scaler.scale)@source_model.coef
    return coef, intercept


def fit_dynamics(X, Y, w, prior=None, alpha=1):
    """Minimize weighted squared loss + alpha*||coef-prior_coef||².

    Target-fold scaling and the intercept use target rows only. Supplied weights
    retain their original mass; the caller owns the per-seed weighting rule.
    """
    x, y, weights = _design(X, Y, w)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('This fixed Ridge design requires a positive alpha')
    single = y.ndim == 1
    response = y[:, None] if single else y
    scaler = _scaler(x, weights)
    z = (x-scaler.mean)/scaler.scale
    zmean, ymean = _mean(z, weights), _mean(response, weights)
    zc, yc = z-zmean, response-ymean
    prior_coef = np.zeros((x.shape[1], response.shape[1]))
    if prior is not None:
        prior_coef, unused_intercept = convert_prior(prior, scaler.mean, scaler.scale)
        if prior_coef.shape != (x.shape[1], response.shape[1]):
            raise ValueError('Source and target response output rosters differ')
    gram = zc.T@(weights[:, None]*zc)+alpha*np.eye(x.shape[1])
    rhs = zc.T@(weights[:, None]*yc)+alpha*prior_coef
    coef = np.linalg.solve(gram, rhs)
    intercept = ymean-zmean@coef  # The source intercept is deliberately unpenalized.
    return DynamicsModel(scaler, coef, intercept, single)


def fit_source(X, Y, w):
    return fit_dynamics(X, Y, w, prior=None, alpha=1)


def predict(model, X):
    x = np.asarray(X, dtype=float)
    values = ((x-model.scaler.mean)/model.scaler.scale)@model.coef+model.intercept
    return values[:, 0] if model.single_output else values


def fit_reader(F, deltaC, w, alpha=1):
    """Scale-only target readout: no centering or intercept, so PP stays zero."""
    features, response, weights = _design(F, deltaC, w)
    if response.ndim != 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('The target cost reader requires one delta-cost column and positive alpha')
    scale = np.sqrt(_mean(features**2, weights))
    scale = np.where(scale == 0, 1., scale)
    design = features/scale
    coef = np.linalg.solve(design.T@(weights[:, None]*design)+alpha*np.eye(features.shape[1]),
                           design.T@(weights*response))
    return ReaderModel(scale, coef)


def reader_predict(model, F):
    return (np.asarray(F, dtype=float)/model.scale)@model.coef


def select_action(scores, pp_index):
    """PP has structural score zero; round12 ties prefer PP, then original order."""
    values = np.asarray(scores, dtype=float).copy()
    if values.ndim != 1 or pp_index not in range(len(values)) or not np.all(np.isfinite(values)):
        raise ValueError('Expected finite action scores and a valid canonical PP index')
    values[pp_index] = 0.
    rounded = np.round(values, 12)
    return int(pp_index) if rounded[pp_index] == rounded.min() else int(np.argmin(rounded))
