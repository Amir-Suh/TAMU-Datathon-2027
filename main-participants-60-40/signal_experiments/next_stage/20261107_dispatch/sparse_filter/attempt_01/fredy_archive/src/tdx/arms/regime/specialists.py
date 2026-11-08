"""Causal local frequencies and lag-one conditional counts, with shrinkage."""
import numpy as np

CONDITIONS = ('stable', 'short_history', 'drift', 'shared_change')
DEFAULT_SPECIALISTS = ((None, 20.), (64., 20.), (32., 20.), (16., 20.))


def probabilities(p):
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-6)
    return p / p.sum(axis=-1, keepdims=True)


def legal_history(split):
    return np.where(np.arange(split.y.shape[-1])[None, None, :] < split.cutoff[:, None, None], split.y, -1)


def fit_priors(splits):
    counts = np.full((3, 5), .5, dtype=np.float64)
    for s in splits:
        y = legal_history(s)
        for mt in range(3):
            v = y[s.mtype == mt]
            counts[mt] += np.bincount(v[v >= 0], minlength=5)
    return probabilities(counts)


def forecast(split, priors, half_life=32., strength=20., starts=None, transition_strength=8.):
    """Observed ticks alone contribute; context missing at cutoff falls back to frequencies."""
    y = legal_history(split)
    t = np.arange(y.shape[-1])
    mask = y >= 0
    if starts is not None:
        mask &= t[None, None, :] >= np.asarray(starts)[:, None, None]
    age = split.cutoff[:, None] - 1 - t[None, :]
    w = np.ones_like(age, dtype=np.float64) if half_life is None else np.exp2(-np.maximum(age, 0) / half_life)
    weights = w[:, None, :] * mask
    prior = priors[split.mtype]
    counts = np.stack([(weights * (y == k)).sum(-1) for k in range(5)], -1)
    base = probabilities(counts + strength * prior)
    if transition_strength is None:
        return base
    cat = np.where(y < 0, -1, np.minimum(y, 2))
    prev = np.full_like(cat, -1)
    prev[..., 1:] = cat[..., :-1]
    index = np.maximum(split.cutoff - 1, 0)
    wanted = np.take_along_axis(cat, index[:, None, None].repeat(8, axis=1), axis=2)[..., 0]
    matched = (prev == wanted[..., None]) & (wanted[..., None] >= 0)
    if starts is not None:
        matched &= t[None, None, :] > np.asarray(starts)[:, None, None]
    tw = weights * matched
    tc = np.stack([(tw * (y == k)).sum(-1) for k in range(5)], -1)
    conditional = probabilities(tc + transition_strength * base)
    return np.where((wanted >= 0)[..., None], conditional, base)
