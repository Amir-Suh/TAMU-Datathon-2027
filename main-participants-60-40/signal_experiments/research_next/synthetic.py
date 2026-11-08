"""Deterministic known-law fixtures for the House Edge research experiments."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import sys

import numpy as np
from scipy.special import expit, logit

KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT / "m5_m6_repair"))
from models.common.data import MISSING, Split

COMMON_PAIRS = ((1, 3), (2, 5), (4, 7))
RARE_PAIRS = ((1, 8), (3, 6), (4, 5), (6, 8))
TAIL = np.array([[.8, .196, .004], [.695652, .260870, .043478], [.368421, .473684, .157895]])
TAIL = TAIL / TAIL.sum(axis=1, keepdims=True)


@dataclass
class Fixture:
    split: Split
    pairs: np.ndarray
    polarity: np.ndarray
    state: np.ndarray
    change: np.ndarray


def generate(scenario="stable", *, episodes=24, seed=2607, cutoff=200, lead=16,
             mode="off", rare=False, four_cells=False):
    """Generate exact cutoff distributions with independently seeded histories."""
    scenarios = ("stable", "flip", "ramp", "pair_swap", "double_flip", "shared_flip", "staggered_flip")
    if scenario not in scenarios or mode not in ("off", "static", "sticky", "iid"):
        raise ValueError("Unknown synthetic mechanism.")
    if not 4 <= lead < cutoff - 24:
        raise ValueError("Change age must be inside the generated prefix.")
    rng = np.random.default_rng(seed)
    n = episodes * 8
    keys = [f"synthetic-{seed}-e{i // 8:04d}-m{i % 8:02d}" for i in range(n)]
    ep = np.asarray([f"synthetic-{seed}-e{i:04d}" for i in range(episodes) for _ in range(8)])
    kind = rng.integers(0, 3, n)
    signal = rng.integers(-1, 2, n)
    pairs = np.asarray(RARE_PAIRS if rare else COMMON_PAIRS)
    choice = rng.integers(len(pairs), size=n)
    alt = (choice + rng.integers(1, len(pairs), size=n)) % len(pairs)
    d0 = rng.integers(0, 2, size=n).astype(bool)
    t0 = np.full(n, cutoff - lead)
    if scenario == "shared_flip":
        t0 += np.repeat(rng.integers(-3, 4, episodes), 8)
    if scenario == "staggered_flip":
        t0 += rng.integers(-7, 4, n)
    if scenario == "stable":
        t0[:] = cutoff + 1
    t1 = t0 - 24
    X = np.full((n, cutoff + 1), MISSING, dtype=np.int8)
    modes = np.zeros_like(X)
    modes[:, 0] = rng.integers(0, 2, n) if mode != "off" else 0
    hi = np.array([.95, .84, .59])[kind]
    lo = np.array([.16, .055, .015])[kind]
    qlow = np.array([.4551, .2141, .1208])[kind]
    qhigh = np.array([.7540, .4999, .3351])[kind]
    rows = np.arange(n)
    probability = None
    sample = None
    for t in range(cutoff + 1):
        if t:
            if mode == "sticky":
                modes[:, t] = modes[:, t-1] ^ (rng.random(n) < 1/64)
            elif mode == "iid":
                modes[:, t] = rng.integers(0, 2, n)
            else:
                modes[:, t] = modes[:, t-1]
        pair_now = pairs[np.where(t >= t0, alt, choice)] if scenario == "pair_swap" else pairs[choice]
        if scenario in ("flip", "shared_flip", "staggered_flip"):
            direction = d0 ^ (t >= t0)
        elif scenario == "double_flip":
            direction = d0 ^ ((t >= t1) & (t < t0))
        else:
            direction = d0
        a, b = pair_now.T
        y_a = X[rows, np.maximum(0, t-a)] > 0
        y_b = X[rows, np.maximum(0, t-b)] > 0
        cell = y_a != y_b
        win = np.where(cell == direction, hi, lo)
        if scenario == "ramp":
            ratio = np.clip((t - t0 + 8) / 16, 0, 1)
            win = (1 - ratio) * win + ratio * np.where(cell != direction, hi, lo)
        win = np.where(t >= b, win, .44)
        if four_cells:
            win = np.clip(win + np.where(y_a, .028, -.028) + np.where(y_b, .012, -.012), .002, .998)
        state = modes[:, t]
        win = expit(logit(win) + .35 * signal + (.38 - .76 * state if mode != "off" else 0))
        q1 = np.where(state > 0, qhigh, qlow) if mode != "off" else (qlow + qhigh)/2
        mix = np.column_stack((q1, (1 - q1[:, None]) * TAIL[kind]))
        probability = np.column_stack((1 - win, win[:, None] * mix))
        sample = np.sum(rng.random((n, 1)) > np.cumsum(probability, axis=1), axis=1).astype(np.int8)
        if t < cutoff:
            X[:, t] = sample
    conditions = np.repeat("stable" if scenario == "stable" else
                           "shared_change" if scenario in ("shared_flip", "staggered_flip") else "drift", n)
    split = Split("synthetic", [key + "-next" for key in keys], ep, keys, kind, signal,
                  np.full(n, cutoff), X, labels=sample, conditions=conditions, targets=probability)
    return Fixture(split, pair_now, direction, modes, t0)


def mask_history(split, older, *, recent=40, seed=2107):
    """Retain nested older observations plus the original recent observed ticks."""
    if older < 0 or recent < 0:
        raise ValueError("History budgets must be nonnegative.")
    masked = np.full_like(split.X, MISSING)
    for i, (qid, cutoff) in enumerate(zip(split.query_ids, split.cutoff)):
        start = max(0, cutoff - recent)
        available = np.flatnonzero(split.X[i, :start] != MISSING)
        order = sorted(available, key=lambda t: hashlib.sha256(f"{seed}:{qid}:{t}".encode()).digest())
        retained = order[:older]
        masked[i, retained] = split.X[i, retained]
        masked[i, start:cutoff] = split.X[i, start:cutoff]
    return replace(split, X=masked)


if __name__ == "__main__":
    for scenario in ("stable", "flip", "ramp", "pair_swap", "double_flip", "shared_flip", "staggered_flip"):
        f = generate(scenario, episodes=2, seed=43, mode="sticky")
        assert f.split.n == 16 and np.allclose(f.split.targets.sum(axis=1), 1)
        assert np.all(f.split.X[:, -1] == MISSING)
        counts = [np.count_nonzero(mask_history(f.split, n).X != MISSING) for n in (0, 12, 24, 48)]
        assert counts == [(40+n)*16 for n in (0, 12, 24, 48)]
    print("Seven known-law fixtures checked for cutoffs, normalization, and nested mask sizes.")
