"""Model 6: weights fit to episode-heldout forecasts from models 2–5.

The fitted blend loss is in-sample for its weights. ``evaluate_heldout``
refits the whole weight-selection pipeline inside an additional outer fold.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.optimize import minimize

from ..common.components import combine
from ..common.crossfit import (FitCache, episode_folds, factory_config, frame_values,
                               pseudo_frame, split_fingerprint, subset_split)
from ..common.data import Split
from ..common.selection import binary_logloss


class Ensemble:
    name = "m6_ensemble"
    KEYS = ("m2", "m3", "m4", "m5")

    def __init__(self, members: dict | None = None, folds: int = 4, fit_cache: FitCache | None = None,
                 progress=None, member_factories: dict | None = None):
        self.members = dict(members or {})  # supplied models are used for final deployment only
        self.folds = folds
        self.fit_cache = fit_cache
        self.progress = progress
        self.member_factories = member_factories
        self.weights = None
        self.member_logloss = {}

    def _progress(self, message):
        if self.progress is None:
            print(f"[m6] {message}", file=sys.stderr, flush=True)
        else:
            self.progress(f"m6: {message}")

    def _fit_members(self, train: Split, cache: FitCache, use_supplied: bool) -> dict:
        from ..m2_lag_pair import LagPairModel
        from ..m3_lag_pair_flip import FlipLagPairModel
        from ..m4_lag_pair_flip_shared import SharedFlipLagPairModel
        from ..m5_boosted_corrector import BoostedCorrector

        factories = self.member_factories or {
            "m2": LagPairModel, "m3": FlipLagPairModel, "m4": SharedFlipLagPairModel,
            "m5": BoostedCorrector,
        }
        fitted = {}
        for key in self.KEYS:
            if (use_supplied and key in self.members
                    and getattr(self.members[key], "_fit_fingerprint", None) == split_fingerprint(train)):
                fitted[key] = self.members[key]
                continue
            factory = factories[key]
            if key == "m5" and self.member_factories is None:
                # M4 for this same partition is safe for M5's final model;
                # its nested heldout fits always use independently fitted M4s.
                def new_m5():
                    return BoostedCorrector(base=fitted["m4"], fit_cache=cache, progress=self.progress)
                fitted[key] = cache.fit("member-m5", train, new_m5,
                                        config=(BoostedCorrector.ITER_GRID, BoostedCorrector.GBM))
            else:
                kind = "m4-for-m5" if key == "m4" and self.member_factories is None else f"member-{key}"
                fitted[key] = cache.fit(kind, train, factory,
                                        config=factory_config(factory))
        return fitted

    @staticmethod
    def _weights(P: np.ndarray, y: np.ndarray):
        rows = np.ones_like(y, dtype=bool)

        def loss(z):
            w = np.exp(z - z.max())
            w /= w.sum()
            return binary_logloss(P @ w, y, rows)

        res = minimize(loss, np.zeros(P.shape[1]), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
        w = np.exp(res.x - res.x.max())
        w /= w.sum()
        return w, loss(res.x)

    def fit(self, train: Split) -> "Ensemble":
        cache = self.fit_cache or FitCache()
        n_folds = min(self.folds, len(set(train.episode_ids)))
        if n_folds < 2:
            raise ValueError("ensemble episode crossfit requires at least two episodes")
        frame = pseudo_frame(train)
        pos = {key: i for i, key in enumerate(frame.row_ids)}
        P = np.empty((len(frame.y), len(self.KEYS)))
        assigned = np.zeros(len(frame.y), bool)
        self.fold_choices = []

        for fold_no, (tr_i, held_i) in enumerate(episode_folds(train, n_folds), 1):
            fit_part, held_part = subset_split(train, tr_i), subset_split(train, held_i)
            self._progress(f"meta fold {fold_no}/{n_folds}: fit members on {len(set(fit_part.episode_ids))} episodes")
            members = self._fit_members(fit_part, cache, use_supplied=False)
            held_frame = pseudo_frame(held_part)
            rows = np.asarray([pos[key] for key in held_frame.row_ids], dtype=int)
            for col, key in enumerate(self.KEYS):
                P[rows, col] = frame_values(members[key], held_frame, "predict_win")
            assigned[rows] = True
            self.fold_choices.append({"held_episode_ids": sorted(set(str(x) for x in held_part.episode_ids)),
                                      "m5_max_iter": members["m5"].max_iter})
        if not assigned.all():
            raise RuntimeError("ensemble meta rows were not all assigned")

        P = np.clip(P, 1e-6, 1 - 1e-6)
        self.names = list(self.KEYS)
        rows = np.ones(len(frame.y), bool)
        self.member_logloss = {key: binary_logloss(P[:, col], frame.y, rows)
                               for col, key in enumerate(self.KEYS)}
        self.weights, self.blend_logloss = self._weights(P, frame.y)
        self.meta_oof = P
        self.meta_row_ids = frame.row_ids
        self._progress("weights fit; fit final members on all episodes")
        self.members = self._fit_members(train, cache, use_supplied=True)
        self._progress("complete")
        return self

    def predict_win(self, split: Split) -> np.ndarray:
        return sum(w * self.members[n].predict_win(split) for n, w in zip(self.names, self.weights))

    def predict(self, split: Split) -> np.ndarray:
        return combine(self.predict_win(split), self.members["m4"].size.predict(split))

    def describe(self) -> dict:
        return {"weights": {n: round(float(w), 4) for n, w in zip(self.names, self.weights)},
                "weight_fit_pseudo_query_logloss": {
                    **{n: round(v, 5) for n, v in self.member_logloss.items()},
                    "blend_in_sample_for_weights": round(self.blend_logloss, 5)},
                "meta_feature_provenance": "Every member prediction is episode held out; M5 rounds are selected inside each fit partition.",
                "fold_choices": self.fold_choices,
                "loss_note": "Blend loss fits and scores weights on the same meta rows; use evaluate_heldout for full-pipeline assessment."}

    def evaluate_heldout(self, train: Split, folds: int = 4, progress=None) -> dict:
        """Whole-pipeline outer assessment; no supplied models enter any fold.

        Returns aligned pseudo-query rows for each member and the blend. Rows
        with shift zero are the original labelled cutoff queries.
        """
        if progress is not None:
            self.progress = progress
        cache = self.fit_cache or FitCache()
        frame = pseudo_frame(train)
        pos = {key: i for i, key in enumerate(frame.row_ids)}
        preds = {key: np.empty(len(frame.y)) for key in (*self.KEYS, "m6")}
        assigned = np.zeros(len(frame.y), bool)
        fold_info = []
        for fold_no, (tr_i, held_i) in enumerate(episode_folds(train, folds), 1):
            fit_part, held_part = subset_split(train, tr_i), subset_split(train, held_i)
            self._progress(f"outer assessment fold {fold_no}/{folds}: refit complete ensemble")
            fitted = Ensemble(folds=self.folds, fit_cache=cache, progress=self.progress,
                              member_factories=self.member_factories).fit(fit_part)
            held_frame = pseudo_frame(held_part)
            rows = np.asarray([pos[key] for key in held_frame.row_ids], dtype=int)
            for key in self.KEYS:
                preds[key][rows] = frame_values(fitted.members[key], held_frame, "predict_win")
            preds["m6"][rows] = frame_values(fitted, held_frame, "predict_win")
            assigned[rows] = True
            fold_info.append({"held_episode_ids": sorted(set(str(x) for x in held_part.episode_ids)),
                              "weights": {n: float(w) for n, w in zip(fitted.names, fitted.weights)},
                              "m5_max_iter": fitted.members["m5"].max_iter,
                              "meta_fold_choices": fitted.fold_choices})
        if not assigned.all():
            raise RuntimeError("outer assessment rows were not all assigned")
        return {"row_ids": frame.row_ids, "episode_ids": frame.episode_ids, "y": frame.y,
                "is_endpoint": np.asarray([shift == 0 for _, _, shift in frame.row_ids]),
                "predictions": preds, "folds": fold_info,
                "binary_logloss": {key: binary_logloss(p, frame.y, np.ones(len(frame.y), bool))
                                   for key, p in preds.items()}}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["fit_cache"] = None
        state["progress"] = None
        return state
