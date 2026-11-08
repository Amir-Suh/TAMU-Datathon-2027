"""Model 2: Bayesian lag-pair model.

Each machine is assumed to follow one hidden rule of the form
    "a win is likely when the wins at lags a and b MATCH"   (direction 0), or
    "a win is likely when the wins at lags a and b DIFFER"  (direction 1),
with any pair 1 <= a < b <= max_lag, plus a "no pattern" null hypothesis.
For every hypothesis and every count memory (half-life) the model keeps
recency-weighted win counts per cell (match / differ), turns them into a
Beta posterior predictive, and weighs hypotheses by their (slowly forgotten)
prequential evidence. The prediction is the posterior-weighted average.

Priors are learned from train with a short EM loop:
* how often each lag pair occurs (pair_prior),
* the typical win rate of the "win" cell (hi) and "lose" cell (lo) per
  machine type x signal.
Nothing about which pairs exist is hard-coded, so a refit on new data can
discover different pairs.

P(size | win) comes from the shared SizeComponent (same as model 1).
"""
from __future__ import annotations

import math

import numpy as np

from ..common.components import SizeComponent, combine, context_priors, decay
from ..common.data import MISSING, N_CONTEXTS, Split
from ..common.selection import binary_logloss, grid_search, scoring_mask


class LagPairModel:
    name = "m2_lag_pair"
    GRID = [{"kappa": k, "evidence_half_life": e} for k in (3.0, 8.0, 20.0) for e in (32.0, 64.0, 128.0)]

    def __init__(self, max_lag: int = 8, count_half_lives=(math.inf, 64.0, 24.0), null_weight: float = 0.1,
                 prior_smoothing: float = 0.1, em_iterations: int = 2, confident: float = 0.9,
                 default_params=None, eval_last: int = 32):
        if not math.isinf(count_half_lives[0]):
            raise ValueError("count_half_lives[0] must be inf (used to estimate priors)")
        self.max_lag = max_lag
        self.pairs = [(a, b) for a in range(1, max_lag + 1) for b in range(a + 1, max_lag + 1)]
        self.count_half_lives = tuple(count_half_lives)
        self.null_weight = null_weight
        self.prior_smoothing = prior_smoothing
        self.em_iterations = em_iterations
        self.confident = confident
        self.default_params = default_params or {"kappa": 8.0, "evidence_half_life": 64.0}
        self.eval_last = eval_last
        self.size = SizeComponent(eval_last)
        self.params = None
        self.selection = []
        self.win_prior = self.hi = self.lo = self.pair_prior = None

    # ------------------------------------------------------------------ core
    def _forward(self, X: np.ndarray, split: Split, params: dict, record_all: bool = False):
        """Run the online filter over all ticks.

        record_all=True  -> [M, T] predictive P(win) at every tick (for selection).
        record_all=False -> dict of quantities at each machine's cutoff.
        """
        M, T = X.shape
        P, H = len(self.pairs), len(self.count_half_lives)
        W = np.where(X == MISSING, -1, X > 0).astype(np.int8)
        lag_a = np.array([p[0] for p in self.pairs])
        lag_b = np.array([p[1] for p in self.pairs])
        lam = np.array([decay(h) for h in self.count_half_lives])
        lam_e = decay(params["evidence_half_life"])
        k = params["kappa"]
        hi, lo, base = self.hi[split.context], self.lo[split.context], self.win_prior[split.context]

        wins = np.zeros((M, P, H, 2))
        trials = np.zeros((M, P, H, 2))
        null_w = np.zeros((M, H))
        null_n = np.zeros((M, H))
        logev = np.zeros((M, P, 2, H))
        logev0 = np.zeros((M, H))
        log_prior = (math.log(1 - self.null_weight) + np.log(self.pair_prior)[:, None, None]
                     + math.log(0.5) - math.log(H)) * np.ones((P, 2, H))
        log_prior0 = math.log(self.null_weight) - math.log(H)

        p_all = np.full((M, T), np.nan) if record_all else None
        out = {"p_win": np.full(M, np.nan), "post_pd": np.zeros((M, P, 2)), "post_null": np.zeros(M),
               "wins_inf": np.zeros((M, P, 2)), "trials_inf": np.zeros((M, P, 2))}

        for t in range(T):
            wins *= lam[None, None, :, None]
            trials *= lam[None, None, :, None]
            null_w *= lam
            null_n *= lam
            logev *= lam_e
            logev0 *= lam_e

            if t >= self.max_lag:
                wa, wb = W[:, t - lag_a], W[:, t - lag_b]
                cvalid = (wa >= 0) & (wb >= 0)
                cell = (wa != wb).astype(np.int64)          # 0 = lags match, 1 = differ
            else:
                cvalid = np.zeros((M, P), bool)
                cell = np.zeros((M, P), np.int64)

            idx = np.broadcast_to(cell[:, :, None, None], (M, P, H, 1))
            w_c = np.take_along_axis(wins, idx, axis=3)[..., 0]
            n_c = np.take_along_axis(trials, idx, axis=3)[..., 0]
            prior_rate = np.stack([np.where(cell == 0, hi[:, None], lo[:, None]),     # direction 0
                                   np.where(cell == 0, lo[:, None], hi[:, None])],    # direction 1
                                  axis=2)
            rate = (k * prior_rate[..., None] + w_c[:, :, None, :]) / (k + n_c[:, :, None, :])
            rate0 = (k * base[:, None] + null_w) / (k + null_n)
            rate = np.where(cvalid[:, :, None, None], rate, rate0[:, None, None, :])

            at_cut = t == split.cutoff
            if record_all or at_cut.any():
                lw = log_prior + logev
                lw0 = log_prior0 + logev0
                mx = np.maximum(lw.reshape(M, -1).max(axis=1), lw0.max(axis=1))
                ew = np.exp(lw - mx[:, None, None, None])
                ew0 = np.exp(lw0 - mx[:, None])
                Z = ew.sum(axis=(1, 2, 3)) + ew0.sum(axis=1)
                pw = ((ew * rate).sum(axis=(1, 2, 3)) + (ew0 * rate0).sum(axis=1)) / Z
                if record_all:
                    p_all[:, t] = pw
                if at_cut.any():
                    r = at_cut
                    out["p_win"][r] = pw[r]
                    out["post_pd"][r] = ew[r].sum(axis=3) / Z[r, None, None]
                    out["post_null"][r] = ew0[r].sum(axis=1) / Z[r]
                    out["wins_inf"][r] = wins[r][:, :, 0, :]
                    out["trials_inf"][r] = trials[r][:, :, 0, :]

            y = W[:, t]
            obs = y >= 0
            if obs.any():
                won = y == 1
                lp = np.where(won[:, None, None, None], np.log(rate), np.log1p(-rate))
                logev += np.where(obs[:, None, None, None], lp, 0.0)
                lp0 = np.where(won[:, None], np.log(rate0), np.log1p(-rate0))
                logev0 += np.where(obs[:, None], lp0, 0.0)
                upd = (cell[..., None] == np.arange(2)) & (obs[:, None] & cvalid)[..., None]
                upd = upd[:, :, None, :].astype(float)
                trials += upd
                wins += upd * won[:, None, None, None]
                null_n += obs[:, None]
                null_w += (obs & won)[:, None]
        return p_all if record_all else out

    # ------------------------------------------------------------------ fitting
    def _update_priors(self, out: dict, split: Split) -> None:
        M, P = split.n, len(self.pairs)
        post = out["post_pd"]
        pair_post = post.sum(axis=2)
        pair_post = pair_post / pair_post.sum(axis=1, keepdims=True)
        eps = self.prior_smoothing
        self.pair_prior = (1 - eps) * pair_post.mean(axis=0) + eps / P

        # Both directions share counts, so once data dominate their posteriors
        # converge; confidence is therefore taken on the pair, and the winning
        # cell is read from the long-memory counts.
        pair_conf = post.sum(axis=2)              # [M, P], includes null mass in the remainder
        p_idx = pair_conf.argmax(axis=1)
        conf = pair_conf.max(axis=1)
        rows = np.arange(M)
        w = out["wins_inf"][rows, p_idx]          # [M, 2]
        n = out["trials_inf"][rows, p_idx]
        rate = (w + 0.5) / (n + 1.0)
        win_cell = (rate[:, 1] > rate[:, 0]).astype(int)
        hi_w, hi_n = w[rows, win_cell], n[rows, win_cell]
        lo_w, lo_n = w[rows, 1 - win_cell], n[rows, 1 - win_cell]
        keep = conf >= self.confident
        ctx = split.context
        hi, lo = np.empty(N_CONTEXTS), np.empty(N_CONTEXTS)
        for c in range(N_CONTEXTS):
            sel = keep & (ctx == c)
            hi[c] = (hi_w[sel].sum() + 2 * 0.8) / (hi_n[sel].sum() + 2)
            lo[c] = (lo_w[sel].sum() + 2 * 0.1) / (lo_n[sel].sum() + 2)
        self.hi, self.lo = np.clip(hi, 0.02, 0.98), np.clip(lo, 0.01, 0.9)
        self.confident_share = float(keep.mean())

    def fit(self, train: Split) -> "LagPairModel":
        self.win_prior, size_prior = context_priors(train)
        self.size.fit(train, size_prior)
        self.hi, self.lo = np.full(N_CONTEXTS, 0.85), np.full(N_CONTEXTS, 0.08)
        self.pair_prior = np.full(len(self.pairs), 1 / len(self.pairs))
        for _ in range(self.em_iterations):
            self._update_priors(self._forward(train.X, train, self.default_params), train)

        X = train.X_with_labels()
        win = (X > 0).astype(float)
        mask = scoring_mask(train, self.eval_last)
        self.params, self.selection = grid_search(
            self.GRID, lambda p: binary_logloss(self._forward(X, train, p, record_all=True), win, mask))
        return self

    # ------------------------------------------------------------------ inference
    def predict_win(self, split: Split) -> np.ndarray:
        return self._forward(split.X, split, self.params)["p_win"]

    def predict(self, split: Split) -> np.ndarray:
        return combine(self.predict_win(split), self.size.predict(split))

    def explain(self, split: Split) -> list[dict]:
        """Most probable rule per machine at its cutoff (for inspection)."""
        out = self._forward(split.X, split, self.params)
        pair_post = out["post_pd"].sum(axis=2)
        p_idx = pair_post.argmax(axis=1)
        m_idx = np.arange(split.n)
        w, n = out["wins_inf"][m_idx, p_idx], out["trials_inf"][m_idx, p_idx]
        rate = (w + 0.5) / (n + 1.0)
        rows = []
        for m in range(split.n):
            rows.append({"query_id": split.query_ids[m], "pair": "%d-%d" % self.pairs[p_idx[m]],
                         "direction": "match->win" if rate[m, 0] >= rate[m, 1] else "differ->win",
                         "match_cell_win_rate": round(float(rate[m, 0]), 3),
                         "differ_cell_win_rate": round(float(rate[m, 1]), 3),
                         "pair_posterior": round(float(pair_post[m, p_idx[m]]), 4),
                         "p_null": round(float(out["post_null"][m]), 4),
                         "p_win": round(float(out["p_win"][m]), 4)})
        return rows

    def describe(self) -> dict:
        order = np.argsort(-self.pair_prior)
        return {"params": self.params, "count_half_lives": [str(h) for h in self.count_half_lives],
                "null_weight": self.null_weight,
                "hi_by_context": np.round(self.hi, 3).tolist(), "lo_by_context": np.round(self.lo, 3).tolist(),
                "top_pairs": [{"pair": "%d-%d" % self.pairs[i], "prior": round(float(self.pair_prior[i]), 4)}
                              for i in order[:6]],
                "confident_share_train": round(self.confident_share, 3),
                "size_params": self.size.params, "selection": self.selection,
                "size_selection": self.size.selection}
