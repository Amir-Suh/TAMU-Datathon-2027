"""Model 1: hierarchical frequency model (no lag structure).

P(payout) = [1 - P(win), P(win) * P(size | win)], where
* P(win) is a recency-weighted Beta posterior per machine, centred on the
  type x signal win rate learned from train;
* P(size | win) is the shared SizeComponent.
"""
from __future__ import annotations

import math

import numpy as np

from ..common.components import SizeComponent, combine, context_priors, dirichlet_filter
from ..common.data import MISSING, Split
from ..common.selection import binary_logloss, grid_search, scoring_mask


def _win_categories(X: np.ndarray) -> np.ndarray:
    return np.where(X == MISSING, MISSING, (X > 0).astype(np.int8)).astype(np.int8)


class HierarchicalFrequencyModel:
    name = "m1_hierarchical_frequency"
    GRID = [{"strength": s, "half_life": h}
            for s in (2.0, 5.0, 10.0, 20.0, 50.0) for h in (math.inf, 128.0, 64.0, 32.0, 16.0)]

    def __init__(self, eval_last: int = 32):
        self.eval_last = eval_last
        self.win_prior = None
        self.params = None
        self.selection = []
        self.size = SizeComponent(eval_last)

    def _p_win_all(self, X: np.ndarray, split: Split, params: dict) -> np.ndarray:
        m = self.win_prior[split.context]
        prior = np.stack([1 - m, m], axis=1)
        return dirichlet_filter(_win_categories(X), prior, params["strength"], params["half_life"])[..., 1]

    def fit(self, train: Split) -> "HierarchicalFrequencyModel":
        self.win_prior, size_prior = context_priors(train)
        X = train.X_with_labels()
        win = (X > 0).astype(float)
        mask = scoring_mask(train, self.eval_last)
        self.params, self.selection = grid_search(
            self.GRID, lambda p: binary_logloss(self._p_win_all(X, train, p), win, mask))
        self.size.fit(train, size_prior)
        return self

    def predict(self, split: Split) -> np.ndarray:
        p_win = self._p_win_all(split.X, split, self.params)[np.arange(split.n), split.cutoff]
        return combine(p_win, self.size.predict(split))

    def describe(self) -> dict:
        return {"win_params": self.params, "size_params": self.size.params,
                "win_prior_by_context": np.round(self.win_prior, 4).tolist(),
                "win_selection": self.selection, "size_selection": self.size.selection}
