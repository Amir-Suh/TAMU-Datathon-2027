"""Model 6: weighted blend of models 2-5.

P(win) = sum_i w_i * P_i(win) with w on the simplex, chosen by minimising
log-loss on train pseudo-queries (each machine re-cut at 8 earlier cutoffs).
Model 5's contribution uses its out-of-fold predictions so it is not
favoured for having been trained on the same rows. Validation is not used.
P(size | win) is taken from model 4.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..common.components import combine
from ..common.data import MISSING, Split
from ..common.selection import binary_logloss, pseudo_splits


class Ensemble:
    name = "m6_ensemble"

    def __init__(self, members: dict | None = None):
        self.members = members or {}
        self.weights = None
        self.member_logloss = {}

    def _ensure_members(self, train: Split) -> None:
        from ..m2_lag_pair import LagPairModel
        from ..m3_lag_pair_flip import FlipLagPairModel
        from ..m4_lag_pair_flip_shared import SharedFlipLagPairModel
        from ..m5_boosted_corrector import BoostedCorrector
        if "m2" not in self.members:
            self.members["m2"] = LagPairModel().fit(train)
        if "m3" not in self.members:
            self.members["m3"] = FlipLagPairModel().fit(train)
        if "m4" not in self.members:
            self.members["m4"] = SharedFlipLagPairModel().fit(train)
        if "m5" not in self.members:
            self.members["m5"] = BoostedCorrector(base=self.members["m4"]).fit(train)

    def fit(self, train: Split) -> "Ensemble":
        self._ensure_members(train)
        self.names = list(self.members)
        splits = pseudo_splits(train)
        cols = []
        for name in self.names:
            model = self.members[name]
            if getattr(model, "oof", None) is not None:          # boosted corrector: out-of-fold
                cols.append(model.oof)
            else:
                cols.append(np.concatenate([model.predict_win(s)[s.labels != MISSING] for s in splits]))
        P = np.clip(np.column_stack(cols), 1e-6, 1 - 1e-6)
        y = np.concatenate([(s.labels[s.labels != MISSING] > 0) for s in splits]).astype(float)
        rows = np.ones_like(y, dtype=bool)
        self.member_logloss = {n: binary_logloss(P[:, i], y, rows) for i, n in enumerate(self.names)}

        def loss(z):
            w = np.exp(z - z.max())
            w /= w.sum()
            return binary_logloss(P @ w, y, rows)

        res = minimize(loss, np.zeros(len(self.names)), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
        w = np.exp(res.x - res.x.max())
        self.weights = w / w.sum()
        self.blend_logloss = float(res.fun)
        return self

    def predict_win(self, split: Split) -> np.ndarray:
        return sum(w * self.members[n].predict_win(split) for n, w in zip(self.names, self.weights))

    def predict(self, split: Split) -> np.ndarray:
        return combine(self.predict_win(split), self.members["m4"].size.predict(split))

    def describe(self) -> dict:
        return {"weights": {n: round(float(w), 4) for n, w in zip(self.names, self.weights)},
                "pseudo_query_logloss": {**{n: round(v, 5) for n, v in self.member_logloss.items()},
                                         "blend": round(self.blend_logloss, 5)}}
