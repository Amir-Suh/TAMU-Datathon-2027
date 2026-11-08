"""E5 schedules with the mode-off history law and payout tail from synthetic.py."""
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit

EXPERIMENTS = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(EXPERIMENTS / "research_next"), str(EXPERIMENTS / "05_adaptation_stress")]
from synthetic import COMMON_PAIRS, MISSING, Split, TAIL
from stress import sparse_prefix

RARE = ((1, 8), (2, 7), (3, 6))
CUTOFF = 200
MASKS = {"full": None, "24+16": 16, "24+8": 8}
SCENARIOS = {"stable": (), "double_flip": (90, 170),
             "swap16": (184,), "swap8": (192,)}
FIT_SEED = 587200
SMOKE_SEED = 587300
EVAL_SEED = 587400


def generate(scenario, seed, episodes, *, score=False):
    """Predict stage receives only pre-cutoff observations and public context."""
    rng = np.random.default_rng(seed)
    n = episodes * 8
    episode = np.repeat([f"e{seed}-{i:04d}" for i in range(episodes)], 8)
    keys = [f"{e}-m{i % 8}" for i, e in enumerate(episode)]
    kind = rng.integers(3, size=n)
    signal = rng.integers(-1, 2, size=n)
    pairs = np.asarray(RARE)
    initial = rng.integers(3, size=n)
    alternate = (initial + rng.integers(1, 3, size=n)) % 3
    direction = rng.integers(2, size=n)
    hi = np.array([.95, .84, .59])[kind]
    lo = np.array([.16, .055, .015])[kind]
    q1 = np.array([(.4551+.7540)/2, (.2141+.4999)/2, (.1208+.3351)/2])[kind]
    mix = np.column_stack((q1, (1 - q1[:, None]) * TAIL[kind]))
    history = np.full((n, CUTOFF + 1), MISSING, dtype=np.int8)
    rows = np.arange(n)
    changes = SCENARIOS[scenario]
    for t in range(CUTOFF + int(score)):
        pair = pairs[alternate if scenario.startswith("swap") and t >= changes[0] else initial]
        polarity = direction ^ int(scenario == "double_flip" and changes[0] <= t < changes[1])
        lhs, rhs = (t - pair).T
        match = (history[rows, lhs.clip(0)] > 0) != (history[rows, rhs.clip(0)] > 0)
        win = np.where(rhs >= 0, np.where(match == polarity, hi, lo), .44)
        win = expit(logit(win) + .35 * signal)
        dist = np.column_stack((1 - win, win[:, None] * mix))
        if t < CUTOFF:
            history[:, t] = (rng.random((n, 1)) > np.cumsum(dist, axis=1)).sum(axis=1)
    split = Split("synthetic", keys, episode, keys, kind, signal, np.full(n, CUTOFF), history)
    if score:
        return split, dist, pairs[initial], pairs[alternate], direction
    return split


def masked(split, tag):
    return split if MASKS[tag] is None else sparse_prefix(split, MASKS[tag])
