"""Causal float64 design matrices, including a slot for the cutoff tick."""
import numpy as np

from tdx.feat import lag_classes


def observed_history(split):
    """Do not trust storage beyond the query cutoff, including in local fitting."""
    ticks = np.arange(split.y.shape[-1])
    return np.where(ticks[None, None, :] < split.cutoff[:, None, None], split.y, -1)


def feature_names(L=4, neighbours=True, rates=True):
    names = ['bias'] + [f'type_{k}' for k in range(3)]
    for lag in range(1, L + 1):
        names += [f'own_lag{lag}_class{k}' for k in range(5)] + [f'own_lag{lag}_missing']
    if neighbours:
        for lag in range(1, min(L, 3) + 1):
            names += [f'others_lag{lag}_wins', f'others_lag{lag}_high', f'others_lag{lag}_missing']
    if rates:
        for half in (4, 16, 64):
            names += [f'ema{half}_class{k}' for k in range(5)]
    names += ['win_last20_observed']
    return names


def all_features(split, L=4, neighbours=True, rates=True):
    """Return (episode, machine, tick, feature), with ticks 0..T inclusive.

    The value at t uses observations strictly before t. EMA denominators contain
    observed weight only; an empty history gives zero rates. Neighbour counts
    are zero with a separate flag whenever any of the seven inputs is unknown.
    """
    y0 = observed_history(split)
    y = np.pad(y0, ((0, 0), (0, 0), (0, 1)), constant_values=-1)
    E, M, T = y.shape
    x = np.zeros((E, M, T, len(feature_names(L, neighbours, rates))), dtype=np.float64)
    x[..., 0] = 1
    x[..., 1:4] = np.eye(3, dtype=np.float64)[split.mtype][:, :, None, :]
    j = 4
    lagged = lag_classes(y, L)
    for lag in range(L):
        z = lagged[..., lag]
        x[..., j:j + 5] = z[..., None] == np.arange(5)
        x[..., j + 5] = z < 0
        j += 6
    if neighbours:
        known = y >= 0
        wins = y > 0
        high = y >= 3  # class indices 3 and 4 are payouts 5 and 20
        missing = (~known).sum(axis=1, keepdims=True) - (~known)
        counts = [a.sum(axis=1, keepdims=True) - a for a in (wins, high)]
        for lag in range(1, min(L, 3) + 1):
            x[..., j + 2] = 1
            if lag < T:
                unknown = missing[..., :-lag] > 0
                for k, count in enumerate(counts):
                    x[..., lag:, j + k] = np.where(unknown, 0, count[..., :-lag])
                x[..., lag:, j + 2] = unknown
            j += 3
    if rates:
        for half in (4, 16, 64):
            state = np.zeros((E, M, 5), dtype=np.float64)
            decay = np.exp2(-1.0 / half)
            for t in range(T):
                denominator = state.sum(axis=-1, keepdims=True)
                np.divide(state, denominator, out=x[:, :, t, j:j + 5], where=denominator > 0)
                state *= decay
                state += y[:, :, t, None] == np.arange(5)
            j += 5
    # Ring buffers advance on an observed tick, not on elapsed clock ticks.
    ring = np.zeros((E, M, 20), dtype=np.float64)
    count = np.zeros((E, M), dtype=np.int64)
    total = np.zeros((E, M), dtype=np.float64)
    for t in range(T):
        np.divide(total, np.minimum(count, 20), out=x[:, :, t, j], where=count > 0)
        e, m = np.nonzero(y[:, :, t] >= 0)
        slot = count[e, m] % 20
        new = (y[e, m, t] > 0).astype(np.float64)
        total[e, m] += new - ring[e, m, slot]
        ring[e, m, slot] = new
        count[e, m] += 1
    return x


def cutoff_features(split, **params):
    x = all_features(split, **params)
    return x[np.arange(x.shape[0])[:, None], np.arange(x.shape[1])[None, :], split.cutoff[:, None]]


def recency_weights(ticks, tau):
    ticks = np.asarray(ticks)
    return np.exp2((ticks - ticks.max()) / tau) if ticks.size else np.empty(0, dtype=np.float64)
