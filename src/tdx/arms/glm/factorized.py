"""Opt-in hierarchical win / conditional payout-size multinomial model.

Both heads optimize one joint likelihood on the original five-class targets.
Size gradients are weighted by target win mass, so losses provide no invented
size labels and soft cutoff targets retain their exact conditional information.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .arm import GLMArm


def objective(flat, x, target, row_weight, penalty, center):
    w = flat.reshape(x.shape[1], 6)
    logits = x @ w
    win_logp = logits[:, :2] - logsumexp(logits[:, :2], axis=1, keepdims=True)
    size_logp = logits[:, 2:] - logsumexp(logits[:, 2:], axis=1, keepdims=True)
    positive = target[:, 1:]
    win_mass = positive.sum(axis=1, keepdims=True)
    win_target = np.concatenate((target[:, :1], win_mass), axis=1)
    deviation = w - center
    loss = -np.sum(row_weight[:, None] * win_target * win_logp)
    loss -= np.sum(row_weight[:, None] * positive * size_logp)
    loss += penalty * np.sum(deviation ** 2)
    error = np.concatenate((np.exp(win_logp) - win_target,
                            win_mass * np.exp(size_logp) - positive), axis=1)
    grad = x.T @ (row_weight[:, None] * error) + 2 * penalty * deviation
    return float(loss), grad.ravel()


def fit_weights(x, target, row_weight, penalty, center, maxiter, diagnostics=None):
    if not len(x):
        if diagnostics is not None:
            diagnostics.update(success=True, status=0, nit=0, objective=0.0,
                               max_abs_grad=0.0, message='No training rows; retained center')
        return center.copy()
    result = minimize(objective, center.ravel().copy(),
                      args=(x, target, row_weight, penalty, center),
                      method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6,
                               'maxls': 30, 'maxcor': 10})
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise FloatingPointError('Nonfinite factorized hierarchical GLM fit')
    if diagnostics is not None:
        diagnostics.update(success=bool(result.success), status=int(result.status),
                           nit=int(result.nit), objective=float(result.fun),
                           max_abs_grad=float(np.max(np.abs(result.jac))), message=str(result.message))
    return result.x.reshape(center.shape)


def probabilities(x, weights):
    logits = x @ weights
    heads = []
    for z in (logits[..., :2], logits[..., 2:]):
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        heads.append(p / p.sum(axis=-1, keepdims=True))
    win, size = heads
    p = np.concatenate((win[..., :1], win[..., 1:] * size), axis=-1)
    p = np.maximum(p, 1e-6)
    return p / p.sum(axis=-1, keepdims=True)


class FactorizedGLMArm(GLMArm):
    _num_logits = 6
    _fit_weights = staticmethod(fit_weights)
    _probabilities = staticmethod(probabilities)

    def __init__(self, two_stage=True, **params):
        super().__init__(**params)
        self.params['two_stage'] = True

