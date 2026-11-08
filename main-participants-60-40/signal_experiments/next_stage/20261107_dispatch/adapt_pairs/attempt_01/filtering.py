"""Matched fixed-rule, single-switch and pruned Bayesian run-length filters."""
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from scipy.special import logsumexp

from fixtures import MISSING

PAIRS = np.array(tuple(combinations(range(1, 9), 2)))
RULES = 2 * len(PAIRS)
HAZARD = 1 / 128
BUDGET = 64
STRENGTH = 48.0


@dataclass(frozen=True)
class Head:
    prior: np.ndarray
    rates: np.ndarray  # low, high within type x signal
    default: np.ndarray
    mix: np.ndarray


def lag_inputs(history, t):
    a, b = (t - PAIRS).T
    x, y = history[:, a.clip(0)], history[:, b.clip(0)]
    return (x > 0) ^ (y > 0), (b >= 0) & (x >= 0) & (y >= 0)


def favorable_rules(parity):
    return (parity[..., None] == np.arange(2)).reshape(len(parity), RULES).astype(np.int8)


def evidence(split):
    """Observed parity cells by machine, pair, parity, and win."""
    counts = np.zeros((split.n, len(PAIRS), 2, 2))
    for t in range(2, int(split.cutoff.max())):
        parity, valid = lag_inputs(split.X, t)
        y = split.X[:, t, None]
        for d in range(2):
            cell = valid & (parity == d)
            counts[:, :, d, 0] += cell & (y == 0)
            counts[:, :, d, 1] += cell & (y > 0)
    return counts


def discover(split, main):
    """Two pair-prior EM updates under fixed carry emissions, then fit new cell heads."""
    counts = evidence(split)
    ctx = split.context
    prior = np.full(len(PAIRS), 1 / len(PAIRS))
    # Rule 0 maps parity 0 to high and parity 1 to low.
    p = main.rates[ctx, ::-1]
    log_cell = np.stack((np.log1p(-p), np.log(p)), axis=-1)
    likelihood = np.stack([(counts * log_cell[:, None]).sum(axis=(2, 3)),
                           (counts * log_cell[:, None, ::-1]).sum(axis=(2, 3))], axis=2)
    iterations = [prior.copy()]
    for _ in range(2):
        joint = likelihood + np.log(prior)[None, :, None] - np.log(2)
        post = np.exp(joint - logsumexp(joint, axis=(1, 2), keepdims=True))
        prior = .85 * post.sum(axis=2).mean(axis=0) + .15 / len(PAIRS)
        iterations.append(prior)
    adapt = np.zeros_like(main.rates)
    mix = np.zeros_like(main.mix)
    default = np.zeros_like(main.default)
    for c in range(9):
        rows = ctx == c
        low = (counts[rows, :, 1] * post[rows, :, :1] +
               counts[rows, :, 0] * post[rows, :, 1:]).sum(axis=(0, 1))
        high = (counts[rows, :, 0] * post[rows, :, :1] +
                counts[rows, :, 1] * post[rows, :, 1:]).sum(axis=(0, 1))
        for k, cell in enumerate((low, high)):
            adapt[c, k] = (cell[1] + 2 * main.rates[c, k]) / (cell.sum() + 2)
        observed = split.X[rows]
        observed = observed[observed >= 0]
        positive = observed[observed > 0]
        default[c] = (len(positive) + 2 * main.default[c]) / (len(observed) + 2)
        tally = np.bincount(positive - 1, minlength=4) + .5
        mix[c] = tally / tally.sum()
    return Head(prior, adapt, default, mix), iterations


def warm(main, adapt, omega):
    return replace(adapt, **{key: (1 - omega) * getattr(main, key) + omega * getattr(adapt, key)
                             for key in ("rates", "default", "mix")})


def simple(split, head, single=False):
    """Exact constant-rate reference, with optional one polarity switch."""
    h = split.X
    n, width = h.shape
    last = int(split.cutoff.max())
    before = np.broadcast_to(np.repeat(.5 * head.prior, 2), (n, RULES)).copy()
    after = np.zeros_like(before)
    trace = new_trace(n, last)
    for t in range(last + 1):
        parity, lag_available = lag_inputs(h, t)
        favorable = favorable_rules(parity)
        actual = t == split.cutoff
        ctx = np.where(actual, split.context, split.machine_type * 3 + 1)
        p = np.where(np.repeat(lag_available, 2, axis=1),
                     head.rates[ctx[:, None], favorable], head.default[ctx, None])
        if single and t:
            reset = HAZARD * before[:, np.arange(RULES) ^ 1]
            before *= 1 - HAZARD
        else:
            reset = np.zeros_like(before)
        pred = before + after + reset
        if t == last:
            break
        y = h[:, t, None]
        likelihood = np.where(y == MISSING, 1, np.where(y > 0, p, 1 - p))
        z = (pred * likelihood).sum(axis=1, keepdims=True)
        before = before * likelihood / z
        after = (after + reset) * likelihood / z
        trace["reset"][:, t] = (reset * likelihood).sum(axis=1) / z[:, 0]
        trace["map"][:, t] = (before + after).argmax(axis=1)
    return finish(pred, p, trace)


def new_trace(n, last):
    return {"map": np.zeros((n, last), dtype=np.int8),
            "reset": np.zeros((n, last), dtype=np.float32),
            "pruned": np.zeros((n, last), dtype=np.float32)}


def finish(rule_mass, emissions, trace):
    if not np.allclose(rule_mass.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Cutoff posterior does not normalize")
    return (rule_mass * emissions).sum(axis=1), rule_mass, trace


def reset_mass(mass, prior, swap):
    """Sum over run lengths, then reset to flipped polarity or a different pair."""
    polarity_flip = mass[:, np.arange(RULES) ^ 1]
    if not swap:
        return polarity_flip
    source = mass.reshape(len(mass), len(PAIRS), 2) / (1 - prior[:, None])
    other = (source.sum(axis=1, keepdims=True) - source) * prior[:, None]
    return (1 - swap) * polarity_flip + swap * other.reshape(mass.shape)


def bincount_rules(weights, rules):
    offsets = np.arange(len(weights))[:, None] * RULES
    return np.bincount((rules + offsets).ravel(), weights.ravel(),
                       minlength=len(weights) * RULES).reshape(len(weights), RULES)


def bocpd(split, head, *, swap, gaps, budget=BUDGET, single=False):
    """Keep a deterministic beam of (last reset tick, rule, segment statistics)."""
    h = split.X
    n, last = split.n, int(split.cutoff.max())
    if not np.all(split.cutoff == last) or np.any(h[:, last:] >= 0):
        raise ValueError("BOCPD batch needs a common cutoff and pre-cutoff-only history")
    base_rules = np.broadcast_to(np.arange(RULES), (n, RULES))
    ids = base_rules.copy()
    weight = np.broadcast_to(np.repeat(.5 * head.prior, 2), (n, RULES)).copy()
    age = np.zeros_like(ids)
    stats = np.zeros((*ids.shape, 2, 2), dtype=np.int16)  # parity favorability, loss/win
    trace = new_trace(n, last)
    trace["mean_age"] = np.zeros((n, last), dtype=np.float32)
    for t in range(last + 1):
        parity, lag_available = lag_inputs(h, t)
        favorable = favorable_rules(parity)
        available = np.repeat(lag_available, 2, axis=1)
        y = h[:, t, None]
        actual = t == last
        ctx = split.context if actual else split.machine_type * 3 + 1
        hz = HAZARD * (t > 0) * (actual | (y != MISSING) | gaps)
        base_p = head.rates[ctx[:, None], favorable]
        rate = np.where(available, base_p, head.default[ctx, None])
        f = np.take_along_axis(favorable, ids, axis=1)
        growth_cell = stats[np.arange(n)[:, None], np.arange(ids.shape[1]), f]
        g_rate = ((STRENGTH * np.take_along_axis(base_p, ids, axis=1) + growth_cell[:, :, 1])
                  / (STRENGTH + growth_cell.sum(axis=2)))
        g_rate = np.where(np.take_along_axis(available, ids, axis=1),
                          g_rate, head.default[ctx, None])
        switch_weight = weight * hz
        if single:
            switch_weight *= age == t - 1
        growth = weight - switch_weight
        reset = reset_mass(bincount_rules(switch_weight, ids), head.prior, swap)
        if actual:
            # Predict from the grown beam plus all reset branches before new pruning.
            mass = bincount_rules(growth, ids) + reset
            win = (growth * g_rate).sum(axis=1) + (reset * rate).sum(axis=1)
            if not np.allclose(mass.sum(axis=1), 1, atol=1e-10):
                raise ValueError("BOCPD cutoff posterior does not normalize")
            return win, mass, trace
        g_like = np.where(y == MISSING, 1, np.where(y > 0, g_rate, 1 - g_rate))
        r_like = np.where(y == MISSING, 1, np.where(y > 0, rate, 1 - rate))
        posterior = np.concatenate((growth * g_like, reset * r_like), axis=1)
        posterior /= posterior.sum(axis=1, keepdims=True)
        trace["reset"][:, t] = posterior[:, ids.shape[1]:].sum(axis=1)
        # Stable ordering resolves ties by previous beam order, followed by pair and polarity.
        top = np.argsort(-posterior, axis=1, kind="stable")[:, :budget]
        kept = np.take_along_axis(posterior, top, axis=1)
        trace["pruned"][:, t] = np.maximum(0, 1 - kept.sum(axis=1))
        weight = kept / kept.sum(axis=1, keepdims=True)
        is_growth = top < ids.shape[1]
        grow_slot = np.minimum(top, ids.shape[1] - 1)
        ids = np.take_along_axis(np.concatenate((ids, base_rules), axis=1), top, axis=1)
        stats = stats[np.arange(n)[:, None], grow_slot] * is_growth[:, :, None, None]
        age = (np.take_along_axis(age, grow_slot, axis=1) + (t > 0)) * is_growth
        update = (y != MISSING) & np.take_along_axis(available, ids, axis=1)
        r, k = np.where(update)
        stats[r, k, favorable[r, ids[r, k]], (h[r, t] > 0).astype(int)] += 1
        trace["map"][:, t] = bincount_rules(weight, ids).argmax(axis=1)
        trace["mean_age"][:, t] = (age * weight).sum(axis=1)
