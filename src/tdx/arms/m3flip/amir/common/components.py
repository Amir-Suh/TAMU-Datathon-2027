"""Building blocks shared by the models.

* Context priors: P(win) and P(size | win) for each machine type x signal cell.
* ``dirichlet_filter``: an online, recency-weighted Dirichlet-multinomial that
  returns, for every tick t, the predictive distribution using ticks < t only.
* ``SizeComponent``: P(payout size | win) per machine, shrunk to its context.
* ``combine``: turn P(win) and P(size | win) into the five payout probabilities.
"""
from __future__ import annotations

import math

import numpy as np

from .data import MISSING, N_CONTEXTS, PAYOUTS, SIGNALS, Split
from .selection import categorical_logloss, grid_search, scoring_mask

N_SIZES = len(PAYOUTS) - 1  # 1x, 2x, 5x, 20x


def decay(half_life: float) -> float:
    """Per-tick multiplier for counts; inf means no forgetting."""
    return 1.0 if math.isinf(half_life) else 2.0 ** (-1.0 / half_life)


def context_priors(train: Split, shrink: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """P(win) [9] and P(size | win) [9, 4] per type x signal, from train histories + labels.

    Each context is shrunk toward its machine type with ``shrink`` pseudo-counts
    (contexts hold ~10k observations, so shrinkage only matters for tiny cells).
    """
    X = train.X_with_labels()
    ctx = np.repeat(train.context[:, None], X.shape[1], axis=1)
    typ = ctx // len(SIGNALS)
    obs = X != MISSING
    win = X > 0

    win_prior = np.empty(N_CONTEXTS)
    size_prior = np.empty((N_CONTEXTS, N_SIZES))
    for c in range(N_CONTEXTS):
        t_mask = obs & (typ == c // len(SIGNALS))
        c_mask = obs & (ctx == c)
        type_rate = win[t_mask].mean()
        win_prior[c] = (win[c_mask].sum() + shrink * type_rate) / (c_mask.sum() + shrink)
        type_sizes = np.bincount(X[t_mask & win] - 1, minlength=N_SIZES) + 0.5
        type_sizes = type_sizes / type_sizes.sum()
        counts = np.bincount(X[c_mask & win] - 1, minlength=N_SIZES)
        size_prior[c] = (counts + shrink * type_sizes) / (counts.sum() + shrink)
    return win_prior, size_prior


def dirichlet_filter(cat: np.ndarray, prior: np.ndarray, strength: float, half_life: float) -> np.ndarray:
    """Online Dirichlet-multinomial with exponential forgetting.

    cat:   int [M, T], category index or MISSING
    prior: [M, K] prior mean per machine
    Returns [M, T, K]: the predictive for tick t built from ticks < t only.
    Counts decay every tick (time-based), whether or not the tick was observed.
    """
    M, T = cat.shape
    lam = decay(half_life)
    counts = np.zeros(prior.shape)
    out = np.empty((M, T, prior.shape[1]))
    rows = np.arange(M)
    for t in range(T):
        counts *= lam
        out[:, t] = (strength * prior + counts) / (strength + counts.sum(axis=1, keepdims=True))
        obs = cat[:, t] != MISSING
        counts[rows[obs], cat[obs, t]] += 1.0
    return out


def combine(p_win: np.ndarray, size: np.ndarray) -> np.ndarray:
    """[M] P(win) and [M, 4] P(size | win) -> [M, 5] probabilities summing to 1."""
    p_win = np.clip(p_win, 1e-6, 1 - 1e-6)
    out = np.concatenate([(1 - p_win)[:, None], p_win[:, None] * size], axis=1)
    return out / out.sum(axis=1, keepdims=True)


class SizeComponent:
    """P(payout size | win) per machine: recency-weighted counts shrunk to the context prior.

    Hyperparameters are chosen on the training split by prequential log-loss
    over each machine's last ``eval_last`` observed wins (plus the label).
    """

    GRID = [{"strength": s, "half_life": h} for s in (5.0, 15.0, 40.0, 100.0) for h in (math.inf, 96.0, 48.0)]

    def __init__(self, eval_last: int = 32):
        self.eval_last = eval_last
        self.params = None
        self.prior = None
        self.selection = []

    @staticmethod
    def _sizes(X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, X - 1, MISSING).astype(np.int8)

    def fit(self, train: Split, size_prior: np.ndarray) -> "SizeComponent":
        self.prior = size_prior
        X = train.X_with_labels()
        cat = self._sizes(X)
        mask = scoring_mask(train, self.eval_last) & (X > 0)
        query_prior = size_prior[train.context]
        neutral_context = train.machine_type * len(SIGNALS) + SIGNALS.index(0)
        historical_prior = size_prior[neutral_context]
        rows = np.arange(train.n)

        def score(p):
            pred = dirichlet_filter(cat, historical_prior, p["strength"], p["half_life"])
            query_pred = dirichlet_filter(cat, query_prior, p["strength"], p["half_life"])
            pred[rows, train.cutoff] = query_pred[rows, train.cutoff]
            return categorical_logloss(pred, cat, mask)

        self.params, self.selection = grid_search(self.GRID, score)
        return self

    def predict(self, split: Split) -> np.ndarray:
        pred = dirichlet_filter(self._sizes(split.X), self.prior[split.context], **self.params)
        return pred[np.arange(split.n), split.cutoff]
