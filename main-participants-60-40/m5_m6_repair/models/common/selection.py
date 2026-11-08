"""Hyperparameter selection within the supplied training partition.

Local pseudo-query histories exclude their target tick. Shared empirical
priors can still be fitted on the full supplied partition, so these losses
are selection diagnostics, not independently held-out generalization scores.
Stacked-model training uses common.crossfit to exclude entire episodes from
all upstream fitting before building held-out features and predictions.
Validation answers are never used by these selection helpers.
"""
from __future__ import annotations

import numpy as np

from .data import MISSING, Split

EPS = 1e-9


def scoring_mask(split: Split, last: int) -> np.ndarray:
    """[M, T] True for observed ticks in [cutoff - last, cutoff], label included."""
    X = split.X_with_labels()
    t = np.arange(X.shape[1])[None, :]
    lo = (split.cutoff - last)[:, None]
    return (X != MISSING) & (t >= lo) & (t <= split.cutoff[:, None])


def binary_logloss(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    p = np.clip(p[mask], EPS, 1 - EPS)
    y = y[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def categorical_logloss(pred: np.ndarray, cat: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask, cat, 0).astype(np.int64)
    p = np.take_along_axis(pred, idx[..., None], axis=-1)[..., 0]
    return float(-np.mean(np.log(np.clip(p[mask], EPS, 1.0))))


PSEUDO_SHIFTS = tuple(range(0, 32, 4))


def pseudo_splits(train: Split, shifts=PSEUDO_SHIFTS) -> list[Split]:
    """Train re-cut at earlier cutoffs, with one labelled query per machine."""
    from .data import truncate
    return [truncate(train, s) for s in shifts]


def pseudo_win_logloss(p_win_fn, splits: list[Split]) -> float:
    """Binary log-loss of P(win) over the labelled pseudo-queries of ``splits``."""
    ps, ys = [], []
    for s in splits:
        ok = s.labels != MISSING
        ps.append(p_win_fn(s)[ok])
        ys.append((s.labels[ok] > 0).astype(float))
    p, y = np.concatenate(ps), np.concatenate(ys)
    return binary_logloss(p, y, np.ones_like(y, dtype=bool))


def grid_search(grid: list[dict], score_fn) -> tuple[dict, list[dict]]:
    """Return the lowest-scoring params and a table of every candidate."""
    table = [{**params, "train_logloss": score_fn(params)} for params in grid]
    best = min(table, key=lambda r: r["train_logloss"])
    return {k: v for k, v in best.items() if k != "train_logloss"}, table
