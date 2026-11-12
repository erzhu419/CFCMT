"""One frozen composite-mixture GLM optimization; no collection or CLI run."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from experiments.tsc_target_packet_kernel_20261109.local_model import (
    COEFFICIENT_VECTORS, FEATURE_NAMES)


HEAD_NAMES = ('motion', 'start', 'yield_entry', 'lane_change')
L2 = 1e-3
MAXITER = 200
GTOL = 1e-7
MOTION_KINK_CONVENTION = (
    'At a clip boundary use the feasible loss-decreasing side: upper/y<mu '
    'and lower/y>mu have derivative1; the opposite side has derivative0. '
    'A zero-width speed interval has derivative0. Mixture responsibilities '
    'weight these component derivatives. Interior derivatives are ordinary.'
)


def _prepare(heads):
    result = {}
    for name in HEAD_NAMES:
        row = heads[name]
        X = np.asarray(row['X'], dtype=float)
        prior = np.asarray(row['prior'], dtype=float)
        risk = np.asarray(row['risk'], dtype=bool)
        y = np.asarray(row['y'], dtype=float)
        if (X.ndim != 3 or X.shape[2] != len(FEATURE_NAMES) or
                X.shape[0] == 0 or X.shape[1] == 0 or prior.shape != X.shape[:2] or
                risk.shape != prior.shape or y.shape != (len(X),)):
            raise ValueError(f'{name}: expected nonempty X[N,K,38],prior/risk[N,K],y[N]')
        if (not np.isfinite(X).all() or not np.isfinite(prior).all() or not np.isfinite(y).all() or
                np.any(prior < 0) or not np.allclose(prior.sum(axis=1), 1., rtol=0, atol=1e-12)):
            raise ValueError(f'{name}: finite features/labels and normalized nonnegative priors required')
        item = dict(X=X, prior=prior, risk=risk, y=y)
        if name == 'motion':
            for key in ('upper', 'a', 'b'):
                item[key] = np.asarray(row[key], dtype=float)
                if item[key].shape != prior.shape or not np.isfinite(item[key]).all():
                    raise ValueError(f'motion: {key} must be finite N-by-K')
            if np.any(item['upper'] < 0) or np.any(item['a'] < 0) or np.any(item['b'] < 0):
                raise ValueError('motion: upper,a,b must be nonnegative')
        elif name == 'lane_change':
            mask = np.asarray(row['mask'], dtype=bool)
            if mask.shape != (*prior.shape, 3) or not np.isin(y, (0, 1, 2)).all():
                raise ValueError('lane_change: require mask[N,K,3] and labels left0/stay1/right2')
            item['mask'] = mask
        elif not np.isin(y, (0, 1)).all():
            raise ValueError(f'{name}: Bernoulli labels must be0/1')
        result[name] = item
    return result


def _mixture(log_likelihood, prior, name):
    with np.errstate(divide='ignore'):
        log_prior = np.log(prior)
    terms = log_prior+log_likelihood
    mixture = logsumexp(terms, axis=1)
    if not np.isfinite(mixture).all():
        raise ValueError(f'{name}: a label has zero prior-supported likelihood')
    responsibilities = np.exp(terms-mixture[:, None])
    return -float(mixture.mean()), responsibilities


def _motion(theta, data):
    X, U, a, b = (data[key] for key in ('X', 'upper', 'a', 'b'))
    y = data['y'][:, None]
    score = np.einsum('nkd,d->nk', X, theta)
    residual = np.clip(score, -b, a)
    proposal = U+residual
    mean = np.clip(proposal, 0., U)
    log_likelihood = -.5*(y-mean)**2-.5*np.log(2*np.pi)
    loss, responsibility = _mixture(log_likelihood, data['prior'], 'motion')
    # risk is NOT entry risk here. A red/incompatible intent can still approach
    # its blocker; upper alone supplies its declared conditional speed bound.
    inner = ((score > -b) & (score < a))
    inner |= (score == -b) & (y > mean)
    inner |= (score == a) & (y < mean)
    outer = (proposal > 0) & (proposal < U)
    outer |= (proposal == U) & (y < mean)
    outer |= (proposal == 0) & (y > mean)
    active = inner & outer & (U > 0)
    derivative = responsibility*(mean-y)*active
    gradient = np.einsum('nk,nkd->d', derivative, X)/len(X)
    return loss, gradient


def _bernoulli(theta, data, name):
    X, y, risk = data['X'], data['y'][:, None], data['risk']
    score = np.einsum('nkd,d->nk', X, theta)
    log_likelihood = np.where(y == 1, -np.logaddexp(0., -score), -np.logaddexp(0., score))
    log_likelihood = np.where(risk, log_likelihood, np.where(y == 0, 0., -np.inf))
    loss, responsibility = _mixture(log_likelihood, data['prior'], name)
    gradient = np.einsum('nk,nkd->d', responsibility*(expit(score)-y)*risk, X)/len(X)
    return loss, gradient


def _lane_change(theta_left, theta_right, data):
    X, prior, risk = (data[key] for key in ('X', 'prior', 'risk'))
    logits = np.stack((np.einsum('nkd,d->nk', X, theta_left),
                       np.zeros(prior.shape), np.einsum('nkd,d->nk', X, theta_right)), axis=2)
    mask = data['mask'].copy()
    # Structurally unable intents produce stay, rather than disappearing from
    # the prior mixture. Padding is also stay for finite numerical evaluation.
    mask[~risk | (prior == 0)] = (False, True, False)
    if np.any(~mask.any(axis=2)):
        raise ValueError('lane_change: a prior-supported intent has no allowed class')
    logits = np.where(mask, logits, -np.inf)
    log_probabilities = logits-logsumexp(logits, axis=2, keepdims=True)
    labels = data['y'].astype(int)
    log_likelihood = np.take_along_axis(log_probabilities,
        np.broadcast_to(labels[:, None, None], (*prior.shape, 1)), axis=2)[:, :, 0]
    loss, responsibility = _mixture(log_likelihood, prior, 'lane_change')
    probabilities = np.exp(log_probabilities)
    gradients = []
    for choice in (0, 2):
        derivative = responsibility*(probabilities[:, :, choice]-(labels[:, None] == choice))
        gradients.append(np.einsum('nk,nkd->d', derivative, X)/len(X))
    return loss, gradients


def _objective(theta, heads):
    coefficients = np.asarray(theta, dtype=float).reshape(5, len(FEATURE_NAMES))
    loss, gradient = 0., np.zeros_like(coefficients)
    part, gradient[0] = _motion(coefficients[0], heads['motion'])
    loss += part
    for index, name in ((1, 'start'), (2, 'yield_entry')):
        part, gradient[index] = _bernoulli(coefficients[index], heads[name], name)
        loss += part
    part, lane_gradients = _lane_change(coefficients[3], coefficients[4], heads['lane_change'])
    loss += part
    gradient[3], gradient[4] = lane_gradients
    loss += L2*float(np.sum(coefficients[:, 1:]**2))
    gradient[:, 1:] += 2*L2*coefficients[:, 1:]
    return float(loss), gradient.ravel()


def objective(theta, heads):
    """Sum four mean mixture NLLs plus L2; X contains already scaled phi."""
    return _objective(theta, _prepare(heads))


def fit_joint(heads):
    """Attempt exactly one fixed L-BFGS-B solve; return None on any failure.

    Receipt failure is terminal for the declared candidate. This function has
    no retry, alternate initialization, optimizer search or real-data loading.
    """
    receipt = dict(status='HOLD_JOINT_OPTIMIZATION', success=False, logical_fit_attempts=1,
        optimizer_calls=0, optimizer='L-BFGS-B', maxiter=MAXITER, gtol=GTOL,
        L2=L2, L2_excludes_intercepts=True, motion_sigma_mps=1.,
        motion_derivative_convention=MOTION_KINK_CONVENTION)
    try:
        prepared = _prepare(heads)
        receipt['head_records'] = {name: len(data['X']) for name, data in prepared.items()}
        initial = np.zeros(5*len(FEATURE_NAMES))
        receipt['initial_loss'] = _objective(initial, prepared)[0]
        receipt['optimizer_calls'] = 1
        result = minimize(lambda theta: _objective(theta, prepared), initial,
            jac=True, method='L-BFGS-B', options=dict(maxiter=MAXITER, gtol=GTOL))
        receipt.update(success=bool(result.success), scipy_status=int(result.status),
            message=str(result.message), iterations=int(result.nit), function_evaluations=int(result.nfev),
            gradient_evaluations=int(result.njev),
            final_loss=float(result.fun) if np.isfinite(result.fun) else None,
            final_gradient_inf=float(np.max(np.abs(result.jac))) if np.isfinite(result.jac).all() else None)
        if (not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all()
                or not np.isfinite(result.jac).all()):
            receipt['success'] = False
            return None, receipt
        receipt['status'] = 'PASS_JOINT_OPTIMIZATION'
        matrix = result.x.reshape(5, len(FEATURE_NAMES))
        motion = prepared['motion']
        raw_residual = np.einsum('nkd,d->nk', motion['X'], matrix[0])
        receipt['motion_positive_residual_prior_fraction'] = float(
            np.sum(motion['prior']*(raw_residual > 0))/len(raw_residual))
        return {name: matrix[index].tolist() for index, name in enumerate(COEFFICIENT_VECTORS)}, receipt
    except Exception as error:
        receipt.update(success=False, error=f'{type(error).__name__}: {error}')
        return None, receipt
