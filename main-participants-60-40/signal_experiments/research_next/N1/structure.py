"""Causal per-tick lag posterior, conditional size prior, and sequence inputs."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import logit

PAIRS = np.array(list(combinations(range(1, 9), 2)))
SWITCH = 1 / 256
SIZE_DECAY = 2 ** (-1 / 32)
SIZE_STRENGTH = 24
N_TYPES = 3


@dataclass
class Frames:
    token: np.ndarray
    kind: np.ndarray
    context: np.ndarray
    evidence: np.ndarray
    win_logit: np.ndarray
    size_logit: np.ndarray
    labels: np.ndarray

    def subset(self, indices):
        return Frames(*(value[indices] for value in vars(self).values()))


def size_prior(train):
    counts = np.array([[np.count_nonzero(train.X[train.machine_type == kind] == value)
                        for value in range(1, 5)] for kind in range(N_TYPES)], dtype=float) + .5
    return counts / counts.sum(axis=1, keepdims=True)


def frame_inputs(split, model, size_reference, *, include_cutoff_labels=False):
    n, t_count = split.X.shape
    times = np.arange(t_count)[None, :]
    x = np.where(times < split.cutoff[:, None], split.X, -1)
    token = np.c_[np.zeros(n, dtype=np.int8), x[:, :-1] + 1]
    labels = x.copy()
    rows = np.arange(n)
    if include_cutoff_labels:
        labels[rows, split.cutoff] = split.labels
    history_type = split.machine_type * 3 + 1
    cutoff_type = split.context
    prior = np.broadcast_to(model.pair_prior[None, :, None] / 2, (n, len(PAIRS), 2)).copy()
    null = np.full(n, model.null_weight)
    prior *= 1 - model.null_weight
    counters = np.zeros((n, 4))
    age = np.full(n, 64)
    context = np.zeros((n, t_count, 3), dtype=np.float32)
    evidence = np.empty((n, t_count, 4), dtype=np.float32)
    base_win = np.empty((n, t_count), dtype=np.float32)
    base_size = np.empty((n, t_count, 4), dtype=np.float32)
    left, right = PAIRS.T
    direction = np.array([False, True])

    for t in range(t_count):
        cutoff = t == split.cutoff
        ctx = np.where(cutoff, cutoff_type, history_type)
        age = np.where(token[:, t] > 0, 0, np.minimum(64, age + 1))
        context[:, t, 0] = token[:, t] > 0
        context[:, t, 1] = np.log1p(age) / np.log(65)
        context[:, t, 2] = np.where(cutoff, split.signal, 0)
        lag_a = x[:, (t - left).clip(0)]
        lag_b = x[:, (t - right).clip(0)]
        valid = (t >= right) & (lag_a >= 0) & (lag_b >= 0)
        differ = (lag_a > 0) != (lag_b > 0)
        pattern_win = np.where(differ[..., None] == direction, model.hi[ctx, None, None],
                               model.lo[ctx, None, None])
        default = model.win_prior[ctx]
        win = np.where(valid[..., None], pattern_win, default[:, None, None])
        prediction = (prior * win).sum(axis=(1, 2)) + null * default
        base_win[:, t] = logit(prediction.clip(.005, .995))
        base_size[:, t] = np.log((counters + SIZE_STRENGTH * size_reference[split.machine_type])
                                / (counters.sum(axis=1, keepdims=True) + SIZE_STRENGTH))
        pair_mass = prior.sum(axis=2)
        evidence[:, t, 0] = pair_mass.max(axis=1)
        evidence[:, t, 1] = (pair_mass * valid).sum(axis=1)
        evidence[:, t, 2] = np.abs(prior[:, :, 0] - prior[:, :, 1]).sum(axis=1)
        evidence[:, t, 3] = base_win[:, t] / 4

        outcome = x[:, t]
        observed = outcome >= 0
        won = outcome > 0
        prior *= np.where(observed[:, None, None], np.where(won[:, None, None], win, 1 - win), 1)
        null *= np.where(observed, np.where(won, default, 1 - default), 1)
        scale = prior.sum(axis=(1, 2)) + null
        prior /= scale[:, None, None]
        null /= scale
        prior = (1 - SWITCH) * prior + SWITCH * prior[:, :, ::-1]
        counters *= SIZE_DECAY
        counters[rows[won], outcome[won] - 1] += 1

    return Frames(token, split.machine_type, context, evidence, base_win, base_size, labels)


def endpoints(frames, cutoff):
    rows = np.arange(len(cutoff))
    return frames.win_logit[rows, cutoff], frames.size_logit[rows, cutoff]
