"""Model 5: a replacement win classifier trained on episode-heldout M4 features.

GBM round selection is nested: for each selection fold, the GBM's training
features are themselves generated with M4 fits excluding each feature row's
episode. The selection fold is absent from all those M4 fits and the GBM fit.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..common.components import combine
from ..common.crossfit import (FitCache, episode_folds, factory_config, frame_values,
                               pseudo_frame, split_fingerprint, subset_split)
from ..common.data import Split
from ..common.selection import binary_logloss
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

    def __init__(self, base=None, folds: int = 4, inner_folds: int = 3,
                 base_factory=None, classifier_factory=None, fit_cache: FitCache | None = None,
                 progress=None):
        self.base = base  # only eligible for the final all-episode fit
        self.folds = folds
        self.inner_folds = inner_folds
        self.base_factory = base_factory or SharedFlipLagPairModel
        self.classifier_factory = classifier_factory
        self.fit_cache = fit_cache
        self.progress = progress
        self.gbm = None
        self.selection = []
        self.oof = None

    def _progress(self, message):
        if self.progress is None:
            print(f"[m5] {message}", file=sys.stderr, flush=True)
        else:
            self.progress(f"m5: {message}")

    @staticmethod
    def _design(f: dict) -> np.ndarray:
        cols = dict(f)
        cols["logit_p_win"] = _logit(f["p_win"])
        cols["logit_p_win_independent"] = _logit(f["p_win_independent"])
        return np.column_stack([cols[k] for k in FEATURES])

    def _new_classifier(self, n_iter):
        if self.classifier_factory is not None:
            return self.classifier_factory(n_iter)
        return HistGradientBoostingClassifier(max_iter=n_iter, **self.GBM)

    def _base_fit(self, train: Split, cache: FitCache):
        config = factory_config(self.base_factory)
        return cache.fit("m4-for-m5", train, self.base_factory, config=config)

    def _features(self, base, frame):
        return np.vstack([self._design(base.features(s))[idx]
                          for s, idx in zip(frame.splits, frame.valid)])

    def _crossfit_features(self, train: Split, cache: FitCache, folds: int):
        frame = pseudo_frame(train)
        X = np.empty((len(frame.y), len(FEATURES)))
        assigned = np.zeros(len(frame.y), bool)
        pos = {key: i for i, key in enumerate(frame.row_ids)}
        for tr_i, held_i in episode_folds(train, folds):
            base = self._base_fit(subset_split(train, tr_i), cache)
            held = pseudo_frame(subset_split(train, held_i))
            idx = np.asarray([pos[key] for key in held.row_ids], dtype=int)
            X[idx] = self._features(base, held)
            assigned[idx] = True
        if not assigned.all():
            raise RuntimeError("crossfit feature rows were not all assigned")
        return frame, X

    def fit(self, train: Split) -> "BoostedCorrector":
        cache = self.fit_cache or FitCache()
        n_episodes = len(set(train.episode_ids))
        if n_episodes < 4:
            raise ValueError("M5 nested episode crossfit requires at least four episodes")
        outer_n = min(self.folds, n_episodes)
        full = pseudo_frame(train)
        pos = {key: i for i, key in enumerate(full.row_ids)}
        oof_grid = {n: np.empty(len(full.y)) for n in self.ITER_GRID}
        base_oof = np.empty(len(full.y))
        assigned = np.zeros(len(full.y), bool)

        for fold_no, (tr_i, held_i) in enumerate(episode_folds(train, outer_n), 1):
            fit_part, held_part = subset_split(train, tr_i), subset_split(train, held_i)
            n_inner = min(self.inner_folds, len(set(fit_part.episode_ids)))
            self._progress(f"selection fold {fold_no}/{outer_n}: crossfit M4 training features ({n_inner} folds)")
            fit_frame, X_fit = self._crossfit_features(fit_part, cache, n_inner)
            held_frame = pseudo_frame(held_part)
            held_pos = np.asarray([pos[key] for key in held_frame.row_ids], dtype=int)
            held_base = self._base_fit(fit_part, cache)
            X_held = self._features(held_base, held_frame)
            base_oof[held_pos] = frame_values(held_base, held_frame, "predict_win")
            for n_iter in self.ITER_GRID:
                model = self._new_classifier(n_iter).fit(X_fit, fit_frame.y.astype(int))
                oof_grid[n_iter][held_pos] = model.predict_proba(X_held)[:, 1]
            assigned[held_pos] = True
        if not assigned.all():
            raise RuntimeError("M5 selection predictions were not all assigned")

        mask = np.ones(len(full.y), bool)
        self.base_logloss = binary_logloss(base_oof, full.y, mask)
        self.selection = [{"max_iter": n, "nested_episode_cv_logloss": binary_logloss(oof_grid[n], full.y, mask)}
                          for n in self.ITER_GRID]
        chosen = min(self.selection, key=lambda r: r["nested_episode_cv_logloss"])
        self.max_iter = chosen["max_iter"]
        self.cv_logloss = chosen["nested_episode_cv_logloss"]
        self.oof = oof_grid[self.max_iter].copy()
        self.oof_row_ids = full.row_ids

        self._progress(f"selected {self.max_iter} rounds; crossfit final M4 features")
        final_frame, X_final = self._crossfit_features(train, cache, outer_n)
        if final_frame.row_ids != full.row_ids:
            raise RuntimeError("final pseudo row order changed")
        self.gbm = self._new_classifier(self.max_iter).fit(X_final, full.y.astype(int))
        if self.base is None or getattr(self.base, "_fit_fingerprint", None) != split_fingerprint(train):
            self.base = self._base_fit(train, cache)
        self._progress("complete")
        return self

    def predict_win(self, split: Split) -> np.ndarray:
        return self.gbm.predict_proba(self._design(self.base.features(split)))[:, 1]

    def predict(self, split: Split) -> np.ndarray:
        return combine(self.predict_win(split), self.base.size.predict(split))

    def describe(self) -> dict:
        return {"base": self.base.name, "features": list(FEATURES), "max_iter": self.max_iter,
                "gbm": self.GBM, "selection": self.selection,
                "selection_pseudo_query_logloss": {"heldout_base": round(self.base_logloss, 5),
                                                   "selected_corrector": round(self.cv_logloss, 5)},
                "selection_note": "The minimum over round choices is a selection loss, not an unbiased assessment."}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["fit_cache"] = None
        state["progress"] = None
        return state
