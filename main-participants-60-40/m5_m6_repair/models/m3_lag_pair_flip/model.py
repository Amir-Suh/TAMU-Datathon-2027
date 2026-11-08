"""Model 3: flip-aware Bayesian lag-pair model.

Hypotheses per machine: lag pair (a, b) x current direction d x flip time tau,
where tau = "no flip" or a tick on a grid. Ticks at or after tau follow the
current rule; ticks before tau follow the opposite rule (cells swapped), which
is what drift and shared-change machines do in the training data. Because the
cells are just relabelled, the whole history counts as evidence for the current
rule instead of being forgotten.

Each hypothesis splits observed ticks into a high-win group, a low-win group,
and an "unknown context" group (a needed lag was unobserved); each group has a
Beta prior, so the marginal likelihood is exact (Beta-binomial) and every
hypothesis explains the same ticks. A "no pattern" null explains all ticks with
one win rate.

``share_episode=True`` (model 4) adds an episode-level alternative where all
machines in an episode share one flip time, mixed with the independent model by
its evidence.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import betaln, logsumexp

from ..common.components import SizeComponent, combine, context_priors
from ..common.data import MISSING, N_CONTEXTS, Split
from ..common.selection import grid_search, pseudo_splits, pseudo_win_logloss


def _lagged(W: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(W, -1)
    out[:, lag:] = W[:, :-lag]
    return out


class FlipLagPairModel:
    name = "m3_lag_pair_flip"
    GRID = [{"kappa": k} for k in (8.0, 20.0, 50.0, 120.0)]

    def __init__(self, max_lag: int = 8, tau_step: int = 4, min_segment: int = 8, max_tick: int = 260,
                 null_weight: float = 0.05, prior_smoothing: float = 0.1, tau_smoothing: float = 0.5,
                 em_iterations: int = 2, confident: float = 0.9, share_episode: bool = False,
                 default_kappa: float = 8.0):
        self.max_lag = max_lag
        self.pairs = [(a, b) for a in range(1, max_lag + 1) for b in range(a + 1, max_lag + 1)]
        self.min_segment = min_segment
        self.tau_grid = np.arange(max_lag + min_segment, max_tick, tau_step)
        self.null_weight = null_weight
        self.prior_smoothing = prior_smoothing
        self.tau_smoothing = tau_smoothing
        self.em_iterations = em_iterations
        self.confident = confident
        self.share_episode = share_episode
        self.default_kappa = default_kappa
        self.size = SizeComponent()
        self.params, self.selection = None, []
        self.win_prior = self.hi = self.lo = self.pair_prior = None
        self.p_flip, self.tau_q, self.p_shared = 0.3, None, 0.3

    # ------------------------------------------------------------------ likelihoods
    def _log_tau_prior(self, cutoff: np.ndarray) -> np.ndarray:
        """[M, 1 + nG]: column 0 = no flip, then each grid tick that leaves min_segment ticks after it."""
        ok = self.tau_grid[None, :] <= (cutoff[:, None] - self.min_segment)
        q = self.tau_q[None, :] * ok
        has = q.sum(axis=1) > 0
        q = np.where(has[:, None], q / np.where(has, q.sum(axis=1), 1)[:, None], 0.0)
        p_flip = np.where(has, self.p_flip, 0.0)
        with np.errstate(divide="ignore"):
            return np.concatenate([np.log(1 - p_flip)[:, None], np.log(p_flip[:, None] * q)], axis=1)

    def _core(self, split: Split, kappa: float, store_counts: bool = False) -> dict:
        X = split.X
        M, T = X.shape
        P, nT = len(self.pairs), 1 + len(self.tau_grid)
        W = np.where(X == MISSING, -1, X > 0).astype(np.int8)
        obs, won = W >= 0, W == 1
        rows, cut = np.arange(M), split.cutoff
        ctx = split.context
        k = kappa
        a_hi, b_hi = k * self.hi[ctx], k * (1 - self.hi[ctx])
        a_lo, b_lo = k * self.lo[ctx], k * (1 - self.lo[ctx])
        a_0, b_0 = k * self.win_prior[ctx], k * (1 - self.win_prior[ctx])
        norm_hi, norm_lo, norm_0 = betaln(a_hi, b_hi), betaln(a_lo, b_lo), betaln(a_0, b_0)
        g_idx = np.minimum(self.tau_grid, T)

        llp = np.empty((M, P, 2, nT))      # log P(pair, dir) + log-likelihood, no tau prior
        pred = np.empty((M, P, 2, nT))
        counts = np.empty((M, P, 2, nT, 4), np.float32) if store_counts else None
        base = math.log(1 - self.null_weight) + math.log(0.5)
        zeros = np.zeros((M, 1))

        for pi, (a, b) in enumerate(self.pairs):
            Wa, Wb = _lagged(W, a), _lagged(W, b)
            valid = obs & (Wa >= 0) & (Wb >= 0)
            cell = Wa != Wb                                   # True = lags differ
            unk = obs & ~valid
            n_u, k_u = unk.sum(axis=1), (unk & won).sum(axis=1)
            tot, before = [], []
            for c in (False, True):
                sel = valid & (cell == c)
                cn = np.concatenate([zeros, np.cumsum(sel, axis=1)], axis=1)      # cn[:, g] = count over ticks < g
                ck = np.concatenate([zeros, np.cumsum(sel & won, axis=1)], axis=1)
                tot.append((cn[:, -1:], ck[:, -1:]))
                before.append((np.concatenate([zeros, cn[:, g_idx]], axis=1),
                               np.concatenate([zeros, ck[:, g_idx]], axis=1)))
            ca, cb = W[rows, cut - a], W[rows, cut - b]
            v_star = (ca >= 0) & (cb >= 0)
            c_star = (ca != cb).astype(int)
            l_u = betaln(a_0 + k_u, b_0 + n_u - k_u) - norm_0
            for d in (0, 1):                                    # d = cell that wins now (0 match, 1 differ)
                cw, co = d, 1 - d
                n_h = tot[cw][0] - before[cw][0] + before[co][0]
                k_h = tot[cw][1] - before[cw][1] + before[co][1]
                n_l = tot[co][0] - before[co][0] + before[cw][0]
                k_l = tot[co][1] - before[co][1] + before[cw][1]
                ll = (betaln(a_hi[:, None] + k_h, b_hi[:, None] + n_h - k_h) - norm_hi[:, None]
                      + betaln(a_lo[:, None] + k_l, b_lo[:, None] + n_l - k_l) - norm_lo[:, None]
                      + l_u[:, None])
                llp[:, pi, d] = base + math.log(self.pair_prior[pi]) + ll
                p_h = (a_hi[:, None] + k_h) / (k + n_h)
                p_l = (a_lo[:, None] + k_l) / (k + n_l)
                p_any = (a_0[:, None] + k_u[:, None] + k_h + k_l) / (k + n_u[:, None] + n_h + n_l)
                pred[:, pi, d] = np.where(v_star[:, None], np.where((c_star == cw)[:, None], p_h, p_l), p_any)
                if store_counts:
                    counts[:, pi, d] = np.stack([k_h, n_h, k_l, n_l], axis=-1)

        n_all, k_all = obs.sum(axis=1), won.sum(axis=1)
        lp0 = math.log(self.null_weight) + betaln(a_0 + k_all, b_0 + n_all - k_all) - norm_0
        pred0 = (a_0 + k_all) / (k + n_all)
        return {"llp": llp, "pred": pred, "lp0": lp0, "pred0": pred0, "counts": counts,
                "log_tau_prior": self._log_tau_prior(cut)}

    # ------------------------------------------------------------------ posteriors
    def _independent(self, core: dict) -> dict:
        lp = core["llp"] + core["log_tau_prior"][:, None, None, :]
        M = lp.shape[0]
        Z = np.logaddexp(logsumexp(lp.reshape(M, -1), axis=1), core["lp0"])
        post = np.exp(lp - Z[:, None, None, None])
        post0 = np.exp(core["lp0"] - Z)
        p_win = (post * core["pred"]).sum(axis=(1, 2, 3)) + post0 * core["pred0"]
        return {"post": post, "post0": post0, "p_win": p_win, "log_evidence": Z}

    def _shared(self, core: dict, ind: dict, split: Split) -> tuple[np.ndarray, np.ndarray]:
        """Mix in 'one flip time for the whole episode'; returns P(win) and P(shared) per machine."""
        llp, pred, lp0, pred0 = core["llp"], core["pred"], core["lp0"], core["pred0"]
        M = llp.shape[0]
        L = np.logaddexp(logsumexp(llp.reshape(M, -1, llp.shape[-1]), axis=1), lp0[:, None])    # [M, nT]
        num = (np.exp(llp - L[:, None, None, :]) * pred).sum(axis=(1, 2)) + np.exp(lp0[:, None] - L) * pred0[:, None]
        p_win = ind["p_win"].copy()
        p_shared = np.zeros(M)
        prior_s = math.log(self.p_shared) - math.log(1 - self.p_shared)
        for ep in np.unique(split.episode_ids):
            idx = np.where(split.episode_ids == ep)[0]
            tau_prior = core["log_tau_prior"][idx[0]]
            J = tau_prior + L[idx].sum(axis=0)
            ev_shared = logsumexp(J)
            ev_ind = ind["log_evidence"][idx].sum()
            ps = 1.0 / (1.0 + math.exp(-np.clip(prior_s + ev_shared - ev_ind, -50, 50)))
            w_tau = np.exp(J - ev_shared)
            p_sh = num[idx] @ w_tau
            p_win[idx] = ps * p_sh + (1 - ps) * ind["p_win"][idx]
            p_shared[idx] = ps
        return p_win, p_shared

    def _infer(self, split: Split, kappa: float | None = None, store_counts: bool = False) -> dict:
        core = self._core(split, kappa or self.params["kappa"], store_counts)
        ind = self._independent(core)
        res = {"core": core, "ind": ind, "p_win": ind["p_win"], "p_shared": np.zeros(split.n)}
        if self.share_episode:
            res["p_win"], res["p_shared"] = self._shared(core, ind, split)
        return res

    # ------------------------------------------------------------------ fitting
    def _update_priors(self, res: dict, split: Split) -> None:
        post = res["ind"]["post"]
        M, P = split.n, len(self.pairs)
        pattern = post.sum(axis=(1, 2, 3))
        pair_post = post.sum(axis=(2, 3)) / pattern[:, None]
        eps = self.prior_smoothing
        self.pair_prior = (1 - eps) * pair_post.mean(axis=0) + eps / P
        tau_post = post.sum(axis=(1, 2))                                    # [M, nT]
        self.p_flip = float(np.clip((tau_post[:, 1:].sum(axis=1) / pattern).mean(), 0.02, 0.98))
        flip = tau_post[:, 1:] / np.maximum(tau_post[:, 1:].sum(axis=1, keepdims=True), 1e-300)
        q = flip.mean(axis=0)
        self.tau_q = (1 - self.tau_smoothing) * q / q.sum() + self.tau_smoothing / len(q)
        if self.share_episode:
            self.p_shared = float(np.clip(res["p_shared"].mean(), 0.05, 0.95))

        flat = post.reshape(M, -1)
        best = flat.argmax(axis=1)
        p_i, d_i, t_i = np.unravel_index(best, post.shape[1:])
        conf = pair_post[np.arange(M), p_i] * pattern
        c = res["core"]["counts"][np.arange(M), p_i, d_i, t_i].astype(float)   # k_h, n_h, k_l, n_l
        keep = conf >= self.confident
        hi, lo = np.empty(N_CONTEXTS), np.empty(N_CONTEXTS)
        for ctx in range(N_CONTEXTS):
            s = keep & (split.context == ctx)
            hi[ctx] = (c[s, 0].sum() + 2 * 0.8) / (c[s, 1].sum() + 2)
            lo[ctx] = (c[s, 2].sum() + 2 * 0.1) / (c[s, 3].sum() + 2)
        self.hi, self.lo = np.clip(hi, 0.02, 0.98), np.clip(lo, 0.01, 0.9)
        self.confident_share = float(keep.mean())

    def fit(self, train: Split) -> "FlipLagPairModel":
        self.win_prior, size_prior = context_priors(train)
        self.size.fit(train, size_prior)
        self.hi, self.lo = np.full(N_CONTEXTS, 0.85), np.full(N_CONTEXTS, 0.08)
        self.pair_prior = np.full(len(self.pairs), 1 / len(self.pairs))
        self.tau_q = np.full(len(self.tau_grid), 1 / len(self.tau_grid))
        for _ in range(self.em_iterations):
            self._update_priors(self._infer(train, self.default_kappa, store_counts=True), train)
        splits = pseudo_splits(train)
        self.params, self.selection = grid_search(
            self.GRID, lambda p: pseudo_win_logloss(lambda s: self._infer(s, p["kappa"])["p_win"], splits))
        return self

    # ------------------------------------------------------------------ inference
    def predict_win(self, split: Split) -> np.ndarray:
        return self._infer(split)["p_win"]

    def predict(self, split: Split) -> np.ndarray:
        return combine(self.predict_win(split), self.size.predict(split))

    def features(self, split: Split) -> dict:
        """Per-query summaries of the posterior, used by the boosted corrector (model 5)."""
        res = self._infer(split)
        post = res["ind"]["post"]
        pattern = post.sum(axis=(1, 2, 3))
        pair_post = post.sum(axis=(2, 3))
        tau_post = post.sum(axis=(1, 2))
        flip = tau_post[:, 1:].sum(axis=1)
        since = split.cutoff[:, None] - self.tau_grid[None, :]
        exp_since = (tau_post[:, 1:] * since).sum(axis=1) / np.maximum(flip, 1e-12)
        ticks = np.arange(split.X.shape[1])[None, :]
        observed_history = (split.X != MISSING) & (ticks < split.cutoff[:, None])
        recent = observed_history & (ticks >= split.cutoff[:, None] - 64)
        p_ind = res["ind"]["p_win"]
        return {"p_win": res["p_win"], "p_win_independent": p_ind, "p_shared": res["p_shared"],
                "p_null": res["ind"]["post0"], "max_pair_post": pair_post.max(axis=1),
                "p_flip": flip / np.maximum(pattern, 1e-12), "since_flip": np.where(flip > 0.5, exp_since, -1.0),
                "n_obs": observed_history.sum(axis=1).astype(float), "n_recent": recent.sum(axis=1).astype(float),
                "machine_type": split.machine_type.astype(float), "signal": split.signal.astype(float)}

    def describe(self) -> dict:
        order = np.argsort(-self.pair_prior)
        tq = self.tau_q
        return {"params": self.params, "share_episode": self.share_episode,
                "hi_by_context": np.round(self.hi, 3).tolist(), "lo_by_context": np.round(self.lo, 3).tolist(),
                "top_pairs": [{"pair": "%d-%d" % self.pairs[i], "prior": round(float(self.pair_prior[i]), 4)}
                              for i in order[:6]],
                "p_flip": round(self.p_flip, 3),
                "flip_tick_prior_peak": int(self.tau_grid[int(np.argmax(tq))]),
                "p_shared": round(self.p_shared, 3) if self.share_episode else None,
                "confident_share_train": round(self.confident_share, 3),
                "size_params": self.size.params, "selection": self.selection,
                "size_selection": self.size.selection}
