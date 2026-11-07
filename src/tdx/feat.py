"""Strictly past tick features; missing observations remain unknown."""
import numpy as np

def lag_classes(y, L):
    if L < 0:
        raise ValueError('L must be nonnegative')
    out = np.full((*y.shape, L), -1, dtype=np.int8)
    for lag in range(1, min(L, y.shape[-1] - 1) + 1):
        out[..., lag:, lag - 1] = y[..., :-lag]
    return out

def others_win(y, lags):
    lags = tuple(lags)
    if any(lag < 1 for lag in lags):
        raise ValueError('All lags must be positive')
    out = np.full((*y.shape, len(lags)), -1, np.int8)
    wins = (y > 0).astype(np.int8)
    missing = (y < 0).astype(np.int8)
    counts = wins.sum(axis=1, keepdims=True) - wins
    unknown = missing.sum(axis=1, keepdims=True) - missing
    counts = np.where(unknown > 0, -1, counts)
    for i, lag in enumerate(lags):
        if lag < y.shape[-1]:
            out[..., lag:, i] = counts[..., :-lag]
    return out

def tick_rows(split):
    return np.nonzero((split.y >= 0) & (np.arange(split.y.shape[-1])[None, None, :] < split.cutoff[:, None, None]))
