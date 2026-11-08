"""Model 5: gradient-boosted corrector on top of model 4.

A HistGradientBoostingClassifier predicts the next spin's win/loss from model
4's P(win) (as a logit) plus summaries of its posterior (pair confidence, flip
probability, ticks since flip, shared-flip probability, null probability,
history length, type, signal). It can only learn systematic miscalibration of
the base model; the base still does the pattern finding.

Training rows are pseudo-queries on train (each machine re-cut at 8 earlier
cutoffs, label = the observed payout there). The number of boosting rounds is
chosen by 4-fold cross-validation grouped by episode, and the out-of-fold
predictions are kept so the ensemble (model 6) can weigh this model fairly.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from ..common.components import combine
from ..common.data import MISSING, Split
from ..common.selection import binary_logloss, pseudo_splits
from ..m4_lag_pair_flip_shared import SharedFlipLagPairModel

FEATURES = ("logit_p_win", "logit_p_win_independent", "p_shared", "p_null", "max_pair_post", "p_flip",
            "since_flip", "n_obs", "n_recent", "machine_type", "signal")


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class BoostedCorrector:
    name = "m5_boosted_corrector"
    ITER_GRID = (25, 50, 100, 200)
    GBM = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
               early_stopping=False, random_state=0)

    def __init__(self, base=None, folds: int = 4):
        self.base = base
        self.folds = folds
        self.gbm = None
        self.selection = []
        self.oof = None

    @staticmethod
    def _design(f: dict) -> np.ndarray:
        cols = dict(f)
        cols["logit_p_win"] = _logit(f["p_win"])
        cols["logit_p_win_independent"] = _logit(f["p_win_independent"])
        return np.column_stack([cols[k] for k in FEATURES])

    def fit(self, train: Split) -> "BoostedCorrector":
        if self.base is None:
            self.base = SharedFlipLagPairModel().fit(train)
        Xs, ys, gs, base_p = [], [], [], []
        for s in pseudo_splits(train):
            ok = s.labels != MISSING
            f = self.base.features(s)
            Xs.append(self._design(f)[ok])
            ys.append((s.labels[ok] > 0).astype(int))
            gs.append(s.episode_ids[ok])
            base_p.append(f["p_win"][ok])
        X, y, groups = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
        base_p = np.concatenate(base_p)
        all_rows = np.ones_like(y, dtype=bool)
        self.base_logloss = binary_logloss(base_p, y.astype(float), all_rows)

        best = None
        for n_iter in self.ITER_GRID:
            oof = np.empty(len(y))
            for tr_i, te_i in GroupKFold(self.folds).split(X, y, groups):
                m = HistGradientBoostingClassifier(max_iter=n_iter, **self.GBM).fit(X[tr_i], y[tr_i])
                oof[te_i] = m.predict_proba(X[te_i])[:, 1]
            ll = binary_logloss(oof, y.astype(float), all_rows)
            self.selection.append({"max_iter": n_iter, "cv_logloss": ll})
            if best is None or ll < best[0]:
                best = (ll, n_iter, oof)
        self.cv_logloss, self.max_iter, self.oof = best
        self.gbm = HistGradientBoostingClassifier(max_iter=self.max_iter, **self.GBM).fit(X, y)
        return self

    def predict_win(self, split: Split) -> np.ndarray:
        return self.gbm.predict_proba(self._design(self.base.features(split)))[:, 1]

    def predict(self, split: Split) -> np.ndarray:
        return combine(self.predict_win(split), self.base.size.predict(split))

    def describe(self) -> dict:
        return {"base": self.base.name, "features": list(FEATURES), "max_iter": self.max_iter,
                "gbm": self.GBM, "selection": self.selection,
                "pseudo_query_logloss": {"base_model": round(self.base_logloss, 5),
                                         "corrector_out_of_fold": round(self.cv_logloss, 5)}}
