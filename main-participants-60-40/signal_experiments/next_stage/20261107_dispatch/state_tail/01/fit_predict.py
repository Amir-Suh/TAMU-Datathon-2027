"""Freeze E1/E2 forecasts using training-only fitted parameters."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit, logsumexp
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[4]
EXP = KIT / "signal_experiments"
REPAIR = KIT / "m5_m6_repair"
DATA = KIT / "public"
sys.path[:0] = [str(REPAIR), str(KIT)]

from models.common.crossfit import subset_split
from models.common.data import TYPES, load_split, truncate
from models.m3_lag_pair_flip.model import FlipLagPairModel
from run_repaired_comparison import input_only
from slot_forecasting.io import read_submission, write_submission
import importlib.util

match_spec = importlib.util.spec_from_file_location("split_match", EXP / "04_four_lag_cells/run.py")
match = importlib.util.module_from_spec(match_spec)
sys.modules[match_spec.name] = match
match_spec.loader.exec_module(match)

SEED = 20261107
SHIFTS = (0, 4, 8, 16, 24, 32)
KAPPAS = (-0.85, -0.75, -0.60, -0.45, 0.0)
ALPHAS = (0, 32, 64, 128, 256, 512, np.inf)
ARMS = ("hybrid", "p0", "e1_payout", "e1_decoupled", "e1_coupled",
        "e2_finite", "e1_coupled_e2_finite")
FIT_FILES = tuple(DATA / "train" / name for name in
                  ("queries.csv", "history.csv", "labels.csv", "conditions.csv")) + (EXP / "folds.csv",)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def mask_prefix(split):
    return np.where(np.arange(split.X.shape[1]) < split.cutoff[:, None], split.X, -1)


def count_prefix(split):
    x = mask_prefix(split)
    return np.stack([(x == k).sum(axis=1) for k in range(5)], axis=1)


def fit_tail(split):
    counts = np.stack([count_prefix(subset_split(split, np.flatnonzero(split.machine_type == m))).sum(0)[2:]
                       for m in range(len(TYPES))])
    pooled = counts.sum(axis=0)
    used = np.where(counts.sum(axis=1, keepdims=True) > 0, counts, pooled)
    if not pooled.sum():
        raise ValueError("No training tail payouts")
    return used / used.sum(axis=1, keepdims=True), counts


def fit_beta_binomial(split):
    result = []
    counts = count_prefix(split)
    for m in range(len(TYPES)):
        n_one = counts[split.machine_type == m, 1]
        n_other = counts[split.machine_type == m, 2:].sum(axis=1)
        ok = n_one + n_other > 0
        n_one, n_other = n_one[ok], n_other[ok]

        def unpack(theta):
            return expit(theta[:2]), np.exp(theta[2:4]), expit(theta[4])

        def nll(theta):
            means, concentration, prevalence = unpack(theta)
            a, b = means * concentration, (1 - means) * concentration
            log_mass = betaln(n_one[:, None] + a, n_other[:, None] + b) - betaln(a, b)
            return -logsumexp(log_mass + np.log([1 - prevalence, prevalence]), axis=1).sum()

        center = n_one.sum() / (n_one.sum() + n_other.sum())
        fits = []
        for spread in (0.10, 0.22):
            start = np.r_[logit(np.clip(center + np.array([-spread, spread]), 0.015, 0.985)),
                          np.log([100.0, 100.0]), 0]
            fit = minimize(nll, start, method="L-BFGS-B",
                           bounds=[(-6, 6), (-6, 6), (0, 9), (0, 9), (-4.6, 4.6)],
                           options={"maxiter": 350, "ftol": 1e-10})
            if not np.isfinite(fit.fun):
                raise ValueError("Non-finite beta-binomial likelihood")
            fits.append(fit)
        chosen = min(fits, key=lambda fit: fit.fun)
        means, concentration, prevalence = unpack(chosen.x)
        if means[0] > means[1]:
            means, concentration, prevalence = means[::-1], concentration[::-1], 1 - prevalence
        result.append([*means, prevalence, *concentration])
    return np.asarray(result)


class FactoredGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.payout = nn.Embedding(6, 8)
        self.machine = nn.Embedding(3, 4)
        self.recurrence = nn.GRU(12, 24, batch_first=True)
        self.head = nn.Linear(24, 5)
        self.signal_offset = nn.Embedding(3, 5, padding_idx=1)
        nn.init.zeros_(self.signal_offset.weight)

    def forward(self, x, machine_type):
        machine = self.machine(machine_type)[:, None, :].expand(-1, x.shape[1], -1)
        state, _ = self.recurrence(torch.cat([self.payout(x + 1), machine], dim=2))
        return self.head(state)


def fit_gru(split, epochs):
    torch.manual_seed(SEED)
    model = FactoredGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    x = torch.from_numpy(mask_prefix(split)).long()
    machine_type = torch.from_numpy(split.machine_type).long()
    cutoff = torch.from_numpy(split.cutoff).long()
    signal = torch.from_numpy(split.signal + 1).long()
    labels = torch.from_numpy(split.labels).long()
    for _ in range(epochs):
        for b in torch.randperm(split.n).split(128):
            logits = model(x[b], machine_type[b])
            history_y = x[b, 1:]
            observed = history_y >= 0
            historical_loss = factored_loss(logits[:, :-1][observed], history_y[observed])
            cutoff_logits = logits[torch.arange(len(b)), cutoff[b] - 1] + model.signal_offset(signal[b])
            endpoint_loss = factored_loss(cutoff_logits, labels[b])
            loss = (historical_loss.sum() + 20 * endpoint_loss.sum()) / (observed.sum() + 20 * len(b))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    return model.eval()


def factored_loss(logits, y):
    win = (y > 0).float()
    return F.binary_cross_entropy_with_logits(logits[:, 0], win, reduction="none") + win * (
        F.cross_entropy(logits[:, 1:], (y - 1).clamp_min(0), reduction="none")
    )


def model_probabilities(model, split, shifts):
    x = torch.from_numpy(mask_prefix(split)).long()
    machine_type = torch.from_numpy(split.machine_type).long()
    with torch.inference_mode():
        logits = model(x, machine_type)
        output = {}
        for shift in shifts:
            cutoff = torch.from_numpy(split.cutoff - shift).long()
            signal = torch.from_numpy(split.signal + 1).long() if shift == 0 else torch.ones(split.n, dtype=torch.long)
            out = logits[torch.arange(split.n), cutoff - 1] + model.signal_offset(signal)
            win = out[:, :1].sigmoid()
            output[shift] = torch.cat([1 - win, win * out[:, 1:].softmax(dim=1)], dim=1).double().numpy()
    return output


def fit_win_proxy(train):
    # The proxy's historical priors ignore the endpoint-only signal. The
    # frozen release comparison uses the exact Split-Match M3 win source.
    train = replace(train, signal=np.zeros(train.n, dtype=int))
    model = FlipLagPairModel()
    counts = count_prefix(train)
    wins = counts[:, 1:].sum(axis=1)
    obs = counts.sum(axis=1)
    by_type = [wins[train.machine_type == m].sum() / obs[train.machine_type == m].sum()
               for m in range(len(TYPES))]
    model.win_prior = np.repeat(by_type, 3)
    model.hi, model.lo = np.full(9, 0.85), np.full(9, 0.08)
    model.pair_prior = np.full(len(model.pairs), 1 / len(model.pairs))
    model.tau_q = np.full(len(model.tau_grid), 1 / len(model.tau_grid))
    model.params = {"kappa": 50.0}
    for _ in range(model.em_iterations):
        model._update_priors(model._infer(train, model.default_kappa, store_counts=True), train)
    # Split-Match's size component is irrelevant when extracting p0.
    model.size.prior = np.full((9, 4), 0.25)
    model.size.params = {"strength": 15.0, "half_life": np.inf}
    prior = np.asarray(match.train_contrasts(model, train)["learned_contrast_11_minus_00"])
    return model, prior


def features(split, gru, state):
    counts = count_prefix(split)
    low, high, pi = state[split.machine_type, :3].T
    n_tail = counts[:, 2:].sum(axis=1)
    lr = counts[:, 1] * np.log(high / low) + n_tail * np.log((1 - high) / (1 - low))
    score = gru[:, 1] / gru[:, 1:].sum(axis=1)
    centered = logit(np.clip(score, 1e-7, 1 - 1e-7)) - logit((1 - pi) * low + pi * high)
    return np.column_stack([low, high, pi, lr, centered, n_tail])


def gamma(x, weights):
    return expit(logit(x[:, 2]) + x[:, 3:5] @ weights)


def fit_weights(frames):
    rows = np.concatenate([frame["features"] for frame in frames])
    y = np.concatenate([frame["y"] for frame in frames])
    weight = np.concatenate([frame["weight"] for frame in frames])
    rows, weight, z = rows[y > 0], weight[y > 0], y[y > 0] == 1
    low, high = rows[:, :2].T

    def objective(beta):
        g = gamma(rows, beta)
        p = low + (high - low) * g
        return -np.average(np.where(z, np.log(p), np.log1p(-p)), weights=weight) + 1e-4 * (beta @ beta)

    opt = minimize(objective, [0.05, 1.0], method="L-BFGS-B", bounds=[(0, 3), (0, 8)])
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x


def state_wins(base, pi, kappa):
    """Find state win rates whose prior-weighted mean equals the base win."""
    if kappa == 0:
        return base, base
    odds_multiplier = np.exp(kappa)
    left, right = np.zeros_like(base), np.ones_like(base)
    for _ in range(55):
        low = (left + right) / 2
        high = odds_multiplier * low / (1 - low + odds_multiplier * low)
        above = (1 - pi) * low + pi * high > base
        left, right = np.where(above, left, low), np.where(above, low, right)
    low = np.where((base == 0) | (base == 1), base, (left + right) / 2)
    high = odds_multiplier * low / (1 - low + odds_multiplier * low)
    return low, high


def joint(base_win, x, g, ratios, kappa, coupled):
    low, high, pi = x[:, :3].T
    w_low, w_high = state_wins(base_win, pi, kappa)
    win = (1 - g) * w_low + g * w_high
    one = ((1 - g) * w_low * low + g * w_high * high if coupled
           else win * ((1 - g) * low + g * high))
    return np.column_stack([1 - win, one, (win - one)[:, None] * ratios])


def tail_mix(base, pooled):
    mass = base[:, 2:].sum(axis=1, keepdims=True)
    return np.divide(base[:, 2:], mass, out=pooled.copy(), where=mass > 0)


def shrink(base, pooled, n, alpha):
    if alpha == 0:
        return base.copy()
    ratio = pooled if np.isinf(alpha) else (
        n[:, None] * tail_mix(base, pooled) + alpha * pooled
    ) / (n[:, None] + alpha)
    mass = base[:, 2:].sum(axis=1, keepdims=True)
    return np.column_stack([base[:, :2], mass * ratio])


def train_frame(gru, state, held, win_proxy=None, ratios=None):
    forecasts = model_probabilities(gru, held, SHIFTS)
    rows = {k: [] for k in ("features", "y", "weight", "gru", "episode_id", "query_id", "shift", "cutoff")}
    if win_proxy is not None:
        rows.update({"win": [], "pooled": []})
    for shift in SHIFTS:
        pseudo = truncate(held, shift)
        if shift:
            pseudo = replace(pseudo, signal=np.zeros(pseudo.n, dtype=int))
        ok = pseudo.labels >= 0
        rows["features"].append(features(pseudo, forecasts[shift], state)[ok])
        rows["y"].append(pseudo.labels[ok])
        rows["weight"].append(np.full(ok.sum(), 4.0 if shift == 0 else 1.0))
        rows["gru"].append(forecasts[shift][ok])
        rows["episode_id"].append(pseudo.episode_ids[ok])
        rows["query_id"].append(np.asarray(pseudo.query_ids)[ok])
        rows["shift"].append(np.full(ok.sum(), shift))
        rows["cutoff"].append(pseudo.cutoff[ok])
        if win_proxy is not None:
            model, prior = win_proxy
            m3 = match.predict(model, pseudo, prior)["candidate"]
            rows["win"].append(1 - m3[ok, 0])
            rows["pooled"].append(ratios[pseudo.machine_type[ok]])
    return {key: np.concatenate(values) for key, values in rows.items()}


def forecast_array(path, ids):
    forecast = read_submission(path, ids)
    return np.asarray([forecast[q] for q in ids])


def replay_champion(split, hashes):
    validation = split.name == "validation"
    parents = EXP / "01_fredy_replay" / ("forecasts" if validation else "test_forecasts")
    pair_path = parents / "pair_l31.csv"
    gru_paths = [parents / f"gru128_factor_seed_{i}.csv" for i in (1, 2, 3)]
    win_path = EXP / "04_four_lag_cells" / ("validation_predictions.csv" if validation else "test_predictions.csv")
    reference = EXP / "combined" / ("m3_match_split_win_pair_l31_factor_gru_size" +
                                    ("" if validation else "_test") + ".csv")
    pair = forecast_array(pair_path, split.query_ids)
    gru = np.mean([forecast_array(p, split.query_ids) for p in gru_paths], axis=0)
    size = (pair + gru) / 2
    p0 = forecast_array(win_path, split.query_ids)[:, :1]
    out = np.column_stack([p0, (1 - p0) * size[:, 1:] / size[:, 1:].sum(axis=1, keepdims=True)])
    hashes.update({str(path.relative_to(KIT)): digest(path) for path in [pair_path, *gru_paths, win_path, reference]})
    error = float(np.max(np.abs(out - forecast_array(reference, split.query_ids))))
    if error > 2e-14:
        raise ValueError(f"Champion replay diverged: {error}")
    return out, error


def arms(split, base, gru, state, pooled, beta, kappa, alpha):
    x = features(split, gru, state)
    p = pooled[split.machine_type]
    g = gamma(x, beta)
    tail = tail_mix(base, p)
    w = 1 - base[:, 0]
    result = {
        "hybrid": base,
        "p0": shrink(base, p, x[:, 5], np.inf),
        "e1_payout": joint(w, x, g, tail, 0, False),
        "e1_decoupled": joint(w, x, g, tail, kappa, False),
        "e1_coupled": joint(w, x, g, tail, kappa, True),
        "e2_finite": shrink(base, p, x[:, 5], alpha),
    }
    result["e1_coupled_e2_finite"] = shrink(result["e1_coupled"], p, x[:, 5], alpha)
    return result, x, g


def known_law():
    x = np.asarray([[0.24, 0.67, 0.60, 0, 0, 0]] * 5)
    w = np.asarray([0, 0.15, 0.40, 0.90, 1.0])
    g = np.asarray([0.50, 0.20, 0.50, 0.60, 0.90])
    q = np.asarray([[0.6, 0.3, 0.1]] * len(w))
    kappa = -0.75
    low, high = state_wins(w, x[:, 2], kappa)
    centering = np.max(np.abs((1 - x[:, 2]) * low + x[:, 2] * high - w))
    odds = np.max(np.abs(logit(high[1:-1]) - logit(low[1:-1]) - kappa))
    coupled, decoupled = [joint(w, x, g, q, kappa, flag) for flag in (True, False)]
    covariance = g * (1 - g) * (high - low) * (x[:, 1] - x[:, 0])
    cov_error = np.max(np.abs(coupled[:, 1] - decoupled[:, 1] - covariance))
    finite_at_zero = shrink(decoupled, q, x[:, 5], 64)
    checks = {"win_centering_max_abs": float(centering), "log_odds_offset_max_abs": float(odds),
              "one_x_covariance_max_abs": float(cov_error),
              "zero_count_finite_equals_hard_projection": bool(np.array_equal(
                  finite_at_zero, shrink(decoupled, q, x[:, 5], np.inf))),
              "zero_count_alpha_zero_identity": bool(np.array_equal(
                  decoupled, shrink(decoupled, q, x[:, 5], 0))),
              "zero_win_all_tails_zero": bool(np.all(finite_at_zero[0, 2:] == 0))}
    if max(centering, odds, cov_error) > 1e-12 or not all(
        checks[k] for k in checks if not k.endswith("_abs")
    ):
        raise ValueError(checks)
    return checks


def known_law_recovery():
    """Recover the static joint mechanism on a synthetic law with hidden state."""
    rng = np.random.default_rng(SEED)
    n = 60_000
    pi, mu_low, mu_high, true_kappa = 0.57, 0.21, 0.50, -0.75
    is_high = rng.random(n) < pi
    past_n = rng.integers(2, 16, n)
    past_ones = rng.binomial(past_n, np.where(is_high, mu_high, mu_low))
    evidence = past_ones * np.log(mu_high / mu_low)
    evidence += (past_n - past_ones) * np.log((1 - mu_high) / (1 - mu_low))
    true_gamma = expit(logit(pi) + evidence)
    win_low = rng.uniform(0.10, 0.92, n)
    win_high = expit(logit(win_low) + true_kappa)
    base = (1 - pi) * win_low + pi * win_high
    tail = np.broadcast_to(np.array([0.65, 0.30, 0.05]), (n, 3))
    low_vec = np.column_stack([1 - win_low, win_low * mu_low,
                              (win_low * (1 - mu_low))[:, None] * tail])
    high_vec = np.column_stack([1 - win_high, win_high * mu_high,
                               (win_high * (1 - mu_high))[:, None] * tail])
    realized_law = np.where(is_high[:, None], high_vec, low_vec)
    y = (np.cumsum(realized_law, axis=1) < rng.random(n)[:, None]).sum(axis=1)
    expected = (1 - true_gamma[:, None]) * low_vec + true_gamma[:, None] * high_vec
    x = np.column_stack([np.full(n, mu_low), np.full(n, mu_high), np.full(n, pi),
                         evidence, np.zeros(n), past_n - past_ones])
    losses = {}
    for k in KAPPAS:
        p = joint(base, x, true_gamma, tail, k, True)
        losses[str(k)] = float(-np.log(p[np.arange(n), y]).mean())
    best = min(losses, key=losses.get)
    modeled = joint(base, x, true_gamma, tail, float(best), True)
    decoupled = joint(base, x, true_gamma, tail, float(best), False)
    coupled_sse = float(np.square(modeled - expected).sum(axis=1).mean())
    decoupled_sse = float(np.square(decoupled - expected).sum(axis=1).mean())
    if float(best) != true_kappa or coupled_sse > 1e-20 or decoupled_sse <= 1e-6:
        raise ValueError((best, coupled_sse, decoupled_sse))
    return {"seed": SEED, "queries": n, "true_kappa": true_kappa, "recovered_kappa": float(best),
            "sampled_logloss_by_kappa": losses, "coupled_probability_sse": coupled_sse,
            "decoupled_probability_sse": decoupled_sse}


def choose(frames):
    kappa_losses, alpha_losses = [], []
    for kappa in KAPPAS:
        losses, weights = [], []
        for frame, beta in frames:
            x, y, w = frame["features"], frame["y"], frame["weight"]
            p = joint(frame["win"], x, gamma(x, beta), frame["pooled"], kappa, True)
            losses.extend(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
            weights.extend(w)
        kappa_losses.append(float(np.average(losses, weights=weights)))
    for alpha in ALPHAS:
        losses, weights = [], []
        for frame, _ in frames:
            x, y, w = frame["features"], frame["y"], frame["weight"]
            p = shrink(frame["gru"], frame["pooled"], x[:, 5], alpha)
            q = tail_mix(p, frame["pooled"])
            is_tail = y >= 2
            losses.extend(-np.log(np.clip(q[np.flatnonzero(is_tail), y[is_tail] - 2], 1e-12, 1)))
            weights.extend(w[is_tail])
        alpha_losses.append(float(np.average(losses, weights=weights)))
    kappa = KAPPAS[np.argmin(kappa_losses[:-1])]
    alpha = ALPHAS[1 + np.argmin(alpha_losses[1:-1])]
    return kappa, alpha, {
        "kappa": dict(zip(map(str, KAPPAS), kappa_losses)),
        "unconstrained_best_kappa": KAPPAS[np.argmin(kappa_losses)],
        "alpha": dict(zip(map(str, ALPHAS), alpha_losses)),
        "unconstrained_best_alpha": str(ALPHAS[np.argmin(alpha_losses)]),
    }


def verify_outputs(result):
    errors = {}
    for name, values in result.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"Invalid probability in {name}")
        errors[name] = float(np.abs(values.sum(axis=1) - 1).max())
    pairs = (("hybrid", "p0"), ("hybrid", "e2_finite"), ("e1_coupled", "e1_coupled_e2_finite"))
    tail = {}
    for control, candidate in pairs:
        a, b = result[control], result[candidate]
        tail[candidate] = {
            "p0_p1_max_abs": float(np.abs(a[:, :2] - b[:, :2]).max()),
            "total_tail_max_abs": float(np.abs(a[:, 2:].sum(axis=1) - b[:, 2:].sum(axis=1)).max()),
        }
    payout_p0 = float(np.abs(result["e1_payout"][:, 0] - result["hybrid"][:, 0]).max())
    coupled_p0 = float(np.abs(result["e1_coupled"][:, 0] - result["e1_decoupled"][:, 0]).max())
    if max(*errors.values(), payout_p0, coupled_p0, *(max(x.values()) for x in tail.values())) > 1e-12:
        raise ValueError("Probability or matched-arm mass invariant failed")
    return {"sum_to_one_max_abs": errors, "payout_only_p0_max_abs": payout_p0,
            "coupled_vs_decoupled_p0_max_abs": coupled_p0, "tail": tail}


def perturb(split, cutoff):
    x = split.X.copy()
    future = np.arange(x.shape[1]) >= cutoff[:, None]
    x[future] = (x[future] + 2) % 5
    return replace(split, X=x)


def check_future(model, split, state, pooled, beta, kappa, alpha, base):
    changed = perturb(split, split.cutoff)
    p = model_probabilities(model, split, (0,))[0]
    q = model_probabilities(model, changed, (0,))[0]
    a = arms(split, base, p, state, pooled, beta, kappa, alpha)[0]
    b = arms(changed, base, q, state, pooled, beta, kappa, alpha)[0]
    errors = {key: float(np.abs(a[key] - b[key]).max()) for key in a}
    errors["gru"] = float(np.abs(p - q).max())
    if max(errors.values()) != 0:
        raise ValueError(f"At- and after-cutoff perturbation changed forecasts: {errors}")
    return errors


def check_pseudo(model, split):
    shift = 16
    changed = perturb(split, split.cutoff - shift)
    original = model_probabilities(model, split, (shift,))[shift]
    modified = model_probabilities(model, changed, (shift,))[shift]
    signal_changed = model_probabilities(model, replace(split, signal=-split.signal), (shift,))[shift]
    prefix = count_prefix(truncate(split, shift))
    changed_prefix = count_prefix(truncate(changed, shift))
    if (not np.array_equal(original, modified) or not np.array_equal(original, signal_changed)
            or not np.array_equal(prefix, changed_prefix)):
        raise ValueError("Pseudo-cutoff causal or neutral-signal check failed")
    return {"gru_future_tick_max_abs": float(np.abs(original - modified).max()),
            "gru_historical_signal_max_abs": float(np.abs(original - signal_changed).max()),
            "historical_count_max_abs": int(np.abs(prefix - changed_prefix).max())}


def write_state(path, split, x, g, base, kappa):
    w_low, w_high = state_wins(1 - base[:, 0], x[:, 2], kappa)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("query_id", "episode_id", "type", "cutoff", "n_tail", "mu_low", "mu_high",
                         "prevalence", "binomial_llr", "gru_centered_logit", "gamma", "win_low", "win_high"))
        for i, key in enumerate(split.query_ids):
            writer.writerow((key, split.episode_ids[i], TYPES[split.machine_type[i]], split.cutoff[i],
                             int(x[i, 5]), *x[i, :5], g[i], w_low[i], w_high[i]))


def main():
    started = monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        out = HERE / "smoke_final"
        out.mkdir(exist_ok=True)
    else:
        out = HERE
    if (out / "frozen.json").exists():
        raise ValueError("Forecasts already frozen in this attempt")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    epochs = 1 if args.smoke else 16
    dispatch_path = HERE.parents[1] / "manifest.json"
    dispatch = json.loads(dispatch_path.read_text())
    manifest_hashes = dispatch["input_sha256"]
    train = load_split(DATA, "train")
    with (EXP / "folds.csv").open(newline="") as stream:
        fold_map = {r["episode_id"]: int(r["fold"]) for r in csv.DictReader(stream) if r["split"] == "train"}
    assert set(train.episode_ids) == set(fold_map)
    if args.smoke:
        episodes = set()
        for fold in range(4):
            episodes.update(sorted(e for e, f in fold_map.items() if f == fold)[:3])
        train = subset_split(train, np.flatnonzero(np.isin(train.episode_ids, sorted(episodes))))
    row_fold = np.array([fold_map[e] for e in train.episode_ids])
    group_trace = []
    input_hashes = {str(path.relative_to(KIT)): digest(path) for path in FIT_FILES}
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    group_sets = {(o,): None for o in range(4)}
    group_sets.update({x: None for x in combinations(range(4), 2)})
    group_sets[()] = None
    all_episodes = set(train.episode_ids)

    for excluded in group_sets:
        start_group = monotonic()
        fit = subset_split(train, np.flatnonzero(~np.isin(row_fold, excluded)))
        fit_ids = sorted(set(fit.episode_ids))
        withheld = sorted(all_episodes - set(fit_ids))
        overlap = sorted(set(fit_ids) & set(withheld))
        assert not overlap and set(fit_ids).isdisjoint(e for e, f in fold_map.items() if f in excluded)
        model = fit_gru(fit, epochs)
        state = fit_beta_binomial(fit)
        ratios, counts = fit_tail(fit)
        group_sets[excluded] = (model, state, ratios, counts)
        name = "full" if not excluded else "exclude_" + "_".join(map(str, excluded))
        # The deployment component uses the full train model only. Fold fits
        # are deterministic and reproducible from the fit trace and source.
        if not excluded:
            np.savez(checkpoints / "full_gru.npz", **{
                key: value.detach().cpu().numpy() for key, value in model.state_dict().items()
            })
        group_trace.append({
            "excluded_folds": excluded, "fit_episodes": fit_ids, "held_episodes": withheld,
            "episode_overlap": overlap, "fit_rows": fit.n,
            "fit_source_files": list(input_hashes),
            "parameters": {"beta_binomial": state.tolist(), "tail_counts": counts.tolist()},
            "seconds": monotonic() - start_group,
        })
        print(f"finished {name}: {len(fit_ids)} episodes in {monotonic() - start_group:.2f}s", flush=True)

    outer = []
    fits_for_weights = []
    for o in range(4):
        fit = subset_split(train, np.flatnonzero(row_fold != o))
        held = subset_split(train, np.flatnonzero(row_fold == o))
        model, state, ratios, _ = group_sets[(o,)]
        proxy = fit_win_proxy(fit)
        frame = train_frame(model, state, held, proxy, ratios)
        assert set(frame["episode_id"]).isdisjoint(fit.episode_ids)
        inner_frames = []
        for i in range(4):
            if i == o:
                continue
            inner = subset_split(train, np.flatnonzero(row_fold == i))
            inner_model, inner_state, _, _ = group_sets[tuple(sorted((o, i)))]
            inner_frames.append(train_frame(inner_model, inner_state, inner))
        beta_outer = fit_weights(inner_frames)
        beta_eps = set(np.concatenate([f["episode_id"] for f in inner_frames]))
        assert set(held.episode_ids).isdisjoint(beta_eps)
        group_trace[o].update({
            "inner_weights_fit_episodes": sorted(beta_eps), "outer_held_vs_inner_weight_overlap": [],
            "beta_weights": beta_outer.tolist(), "held_pseudo_rows": len(frame["y"]),
            "held_pseudo_rows_by_shift": {str(s): int(np.count_nonzero(frame["shift"] == s)) for s in SHIFTS},
            "win_proxy": "fold-fitted Split-Match M3 structure, neutral signal, fixed 50 prior strength",
        })
        outer.append((frame, beta_outer))
        fits_for_weights.append(frame)
    kappa, alpha, losses = choose(outer)
    beta = fit_weights(fits_for_weights)
    model, state, pooled, counts = group_sets[()]
    np.savez_compressed(out / "train_oof.npz", **{
        key: np.concatenate([frame[key] for frame, _ in outer]) for key in outer[0][0]
    })
    print(f"Training selection: kappa={kappa}, alpha={alpha}, beta={beta.tolist()}", flush=True)
    diagnostics = {
        "known_law": known_law(),
        "known_law_recovery": known_law_recovery(),
        "pseudo_future_tick": check_pseudo(model, train),
        "training": {
            "kappa": kappa, "finite_alpha": alpha, "beta_weights": beta.tolist(), "cv_logloss": losses,
            "tail_type_order": list(TYPES), "beta_binomial_columns": ["mu_low", "mu_high", "pi_high",
                                                                       "concentration_low", "concentration_high"],
            "beta_binomial": state.tolist(), "tail_counts": counts.tolist(), "tail_ratios": pooled.tolist(),
            "pseudo_shifts": list(SHIFTS), "actual_cutoff_label_weight": 4,
        },
    }

    for name in ("validation", "test"):
        split = input_only(DATA, name)
        assert all_episodes.isdisjoint(split.episode_ids)
        for filename in ("queries.csv", "history.csv"):
            path = DATA / name / filename
            input_hashes[str(path.relative_to(KIT))] = digest(path)
        base, champion_error = replay_champion(split, input_hashes)
        p0_path = EXP / "research_next/P0" / ("raw_counts.csv" if name == "validation" else "raw_counts_test.csv")
        input_hashes[str(p0_path.relative_to(KIT))] = digest(p0_path)
        archived_p0 = forecast_array(p0_path, split.query_ids)
        if args.smoke:
            subset = np.flatnonzero(np.isin(split.episode_ids, np.unique(split.episode_ids)[:12]))
            split, base, archived_p0 = subset_split(split, subset), base[subset], archived_p0[subset]
        gru = model_probabilities(model, split, (0,))[0]
        result, x, g = arms(split, base, gru, state, pooled, beta, kappa, alpha)
        p0_error = float(np.abs(result["p0"] - archived_p0).max())
        if not args.smoke and p0_error > 2e-14:
            raise ValueError(f"P0 hard-projection replay diverged: {p0_error}")
        diagnostics[name] = {
            "episodes": len(set(split.episode_ids)), "queries": split.n, "fit_episode_overlap": [],
            "champion_replay_max_abs": champion_error,
            "p0_replay_max_abs" if not args.smoke else "p0_reduced_train_expected_difference": p0_error,
            "forecast_future_tick_invariance": check_future(model, split, state, pooled, beta, kappa, alpha, base),
            "invariants": verify_outputs(result),
            "gamma_by_type": {
                t: np.quantile(g[split.machine_type == j], [0, 0.25, 0.5, 0.75, 1]).tolist()
                for j, t in enumerate(TYPES)
            },
        }
        for arm, values in result.items():
            path = out / f"{name}_{arm}.csv"
            write_submission(path, dict(zip(split.query_ids, values)))
            saved = forecast_array(path, split.query_ids)
            if np.abs(saved - values).max() > 1e-12:
                raise ValueError(f"CSV round-trip changed forecast for {name} {arm}")
        write_state(out / f"{name}_state.csv", split, x, g, base, kappa)

    for path, value in input_hashes.items():
        if manifest_hashes.get(path) != value:
            raise ValueError(f"Dispatch hash mismatch: {path}")
    source_paths = [
        Path(__file__), HERE / "score_frozen.py", HERE.parents[1] / "coordinator/audit_run.py",
        EXP / "04_four_lag_cells/run.py", EXP / "score.py",
        REPAIR / "run_repaired_comparison.py", KIT / "slot_forecasting/io.py",
        KIT / "slot_forecasting/contest.py",
    ] + sorted((REPAIR / "models").rglob("*.py"))
    source_hashes = {str(p.relative_to(KIT)): digest(p) for p in source_paths}
    dump(out / "fit_trace.json", {
        "stages": group_trace, "new_component_fit_source_sha256": {str(p.relative_to(KIT)): digest(p) for p in FIT_FILES},
        "read_exposure": {
            "new_gru_and_parameters": "train only, nested episode grouping for selection, no validation or test data",
            "archived_eval_parents": "pre-cutoff histories across main splits for pair_l31 and archived factored GRUs, "
                                     "frozen forecasts read in inference only",
            "split_match_m3": "archived win-only CSV source, evaluation; folded train-only proxy selects kappa",
        },
    })
    dump(out / "diagnostics.json", diagnostics)
    dump(out / "manifest.json", {
        "round": dispatch["round"], "season_id": dispatch["season_id"], "owner": "state_tail",
        "cards": ["E1", "E2"], "arms": ARMS, "fit_start_utc": timestamp, "fit_seconds": monotonic() - started,
        "smoke": args.smoke, "seed": SEED, "gru_epochs": epochs, "cpu_threads": 1,
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__},
        "dispatch_manifest_sha256": digest(dispatch_path),
        "fit_scope": "New parameters train only. Frozen transductive parents used in released forecasts.",
        "forecast_fit_trace": "fit_trace.json",
        "input_sha256": input_hashes, "source_sha256": source_hashes,
        "commands": [
            f"python3 -B {HERE.parents[1].relative_to(KIT)}/coordinator/audit_run.py --stage fit-predict "
            f"--trace {out.relative_to(KIT)}/fit_audit.jsonl {HERE.relative_to(KIT)}/fit_predict.py"
            + (" --smoke" if args.smoke else ""),
            f"python3 -B {HERE.parents[1].relative_to(KIT)}/coordinator/audit_run.py --stage score "
            f"--trace {HERE.relative_to(KIT)}/score_audit.jsonl {HERE.relative_to(KIT)}/score_frozen.py",
        ],
    })
    forecast_paths = sorted(out.glob("validation_*.csv")) + sorted(out.glob("test_*.csv"))
    dump(out / "frozen.json", {
        "phase": "forecasts_frozen_before_scoring", "at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {p.name: digest(p) for p in forecast_paths},
    })
    print(f"Frozen forecasts in {out.relative_to(KIT)} after {monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
