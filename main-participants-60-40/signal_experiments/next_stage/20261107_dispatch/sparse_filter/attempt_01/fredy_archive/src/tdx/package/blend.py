"""Versioned numpy inference contract shared by packaging and refitting."""
import numpy as np

CONDITIONS = ('stable', 'short_history', 'drift', 'shared_change')


def probabilities(value, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(f'Expected probabilities {shape}, got {value.shape}')
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError('Invalid probability values')
    if np.max(np.abs(value.sum(axis=-1) - 1)) > 1e-6:
        raise ValueError('Probabilities do not sum to one')
    value = np.clip(value, 1e-6, 1.0)
    return value / value.sum(axis=-1, keepdims=True)


def posterior(split):
    """W5 fallback when an exported condition posterior is unavailable."""
    short = np.all(split.n_obs == 64, axis=1)
    result = np.full((len(short), 4), 1 / 3, dtype=np.float64)
    result[:, 1] = 0
    result[short] = [0, 1, 0, 0]
    return result


def condition_probabilities(value, episodes):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (episodes, 4) or not np.isfinite(value).all()
            or np.any(value < 0) or np.any(value > 1)
            or np.max(np.abs(value.sum(axis=1) - 1)) > 1e-12):
        raise ValueError('Invalid condition posterior')
    return value


def validate(config, arm_ids):
    if config.get('version', 1) != 1:
        raise ValueError('Unsupported blend version')
    if config.get('arms', arm_ids) != arm_ids:
        raise ValueError('Blend arms/order differ from --arms')
    if config.get('calibrator') is not None:
        raise ValueError('Calibrator export is not supported by blend version 1')
    weights = np.asarray(config['weights'], dtype=np.float64)
    if weights.shape not in ((len(arm_ids),), (4, len(arm_ids))):
        raise ValueError('Blend weights must have shape (arms,) or (4, arms)')
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Blend weights must be finite and nonnegative')
    if np.max(np.abs(weights.sum(axis=-1) - 1)) > 1e-12:
        raise ValueError('Blend weights must sum to one')
    return dict(config, version=1, arms=arm_ids, weights=weights.tolist())


def apply(config, predictions, split, cond_posterior=None):
    weights = np.asarray(config['weights'], dtype=np.float64)
    values = np.stack([probabilities(p, (*split.query_ids.shape, 5)) for p in predictions])
    if weights.ndim == 1:
        result = np.einsum('k,kemc->emc', weights, values)
    else:
        pi = posterior(split) if cond_posterior is None else condition_probabilities(cond_posterior, len(split.ids))
        result = np.einsum('eh,hk,kemc->emc', pi, weights, values)
    return probabilities(result)


def _simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    rho = np.flatnonzero(u - css / np.arange(1, len(u) + 1) > 0)[-1]
    return np.maximum(v - css[rho] / (rho + 1), 0)


def fit(predictions, split, reference):
    """Deterministic final refit only; this is not a validation CV estimate.

    Four simplex rows use observable condition posteriors, never true conditions
    during prediction. Loss weights match equal-condition contest averaging.
    """
    values = np.stack(predictions)
    n = len(predictions)
    if n == 1:
        return [1.0]
    pi = posterior(split)
    # Remove the component common to every arm. Simplex constraints make it
    # constant, and retaining it can make almost-identical arms ill-conditioned.
    center = values.mean(axis=0)
    features = np.einsum('eh,kemc->emchk', pi, values - center).reshape(-1, 4 * n)
    target = (split.target - center).reshape(-1)
    ref_error = ((reference - split.target) ** 2).sum(axis=(1, 2)) / 8
    episode_weight = np.zeros(len(split.ids))
    for condition in CONDITIONS:
        mask = split.condition == condition
        if not mask.any():
            raise ValueError(f'Missing validation condition {condition}')
        episode_weight[mask] = 1 / (mask.sum() * max(ref_error[mask].mean(), 1e-9))
    weights = np.repeat(episode_weight, 8 * 5)
    gram = features.T @ (weights[:, None] * features)
    rhs = features.T @ (weights * target)
    step = 1 / max(np.linalg.eigvalsh(gram).max(), 1e-12)
    w = np.full((4, n), 1 / n)
    accelerated, momentum = w.copy(), 1.0
    scale = max(np.max(np.abs(rhs)), 1e-12)
    for _ in range(100000):
        new = np.stack([_simplex(row) for row in (accelerated.ravel() - step * (gram @ accelerated.ravel() - rhs)).reshape(4, n)])
        gradient = gram @ new.ravel() - rhs
        projected = np.stack([_simplex(row) for row in (new.ravel() - step * gradient).reshape(4, n)])
        # Projected-gradient optimality residual, relative to this loss scale.
        if np.max(np.abs(projected - new)) / step < 1e-10 * scale:
            return new.tolist()
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        accelerated = new + ((momentum - 1) / next_momentum) * (new - w)
        w, momentum = new, next_momentum
    raise ValueError('Blend refit did not reach projected-gradient optimality tolerance')
