"""Deterministic penalized multinomial likelihood in numpy/scipy."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def probabilities(x, weights):
    logits = x @ weights
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    p = np.maximum(p, 1e-6)
    return p / p.sum(axis=-1, keepdims=True)


def objective(flat, x, target, row_weight, penalty, center):
    w = flat.reshape(x.shape[1], 5)
    logits = x @ w
    normalizer = logsumexp(logits, axis=1)
    logp = logits - normalizer[:, None]
    p = np.exp(logp)
    deviation = w - center
    loss = -np.sum(row_weight[:, None] * target * logp) + penalty * np.sum(deviation ** 2)
    grad = x.T @ (row_weight[:, None] * (p - target)) + 2 * penalty * deviation
    return float(loss), grad.ravel()


def fit_weights(x, target, row_weight, penalty, center, maxiter=100, diagnostics=None):
    if not len(x):
        if diagnostics is not None:
            diagnostics.update(success=True, status=0, nit=0, objective=0.0,
                               max_abs_grad=0.0, message='No training rows; retained center')
        return center.copy()
    result = minimize(objective, center.ravel().copy(),
                      args=(x, target, row_weight, penalty, center),
                      method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30, 'maxcor': 10})
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise FloatingPointError('Nonfinite hierarchical GLM fit')
    if diagnostics is not None:
        diagnostics.update(success=bool(result.success), status=int(result.status),
                           nit=int(result.nit), objective=float(result.fun),
                           max_abs_grad=float(np.max(np.abs(result.jac))), message=str(result.message))
    return result.x.reshape(center.shape)
