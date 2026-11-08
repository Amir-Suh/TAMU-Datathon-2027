"""Tick-indexed histories and point-in-time tabular features."""

from collections import defaultdict
from itertools import combinations

import numpy as np

from .schema import PAYOUTS, index_history

MISSING = -1
LAGS = tuple(range(1, 9))
LAG_PAIRS = tuple(combinations(LAGS, 2))
WINDOWS = (8, 16, 32, 64, None)


def tick_arrays(queries, history):
    indexed = index_history(queries, history)
    cutoffs = np.array([q.cutoff for q in queries])
    # Include a column for the cutoff itself. It remains missing.
    spins = np.full((len(queries), cutoffs.max() + 1), MISSING, dtype=np.int8)
    for i, q in enumerate(queries):
        for play in indexed.get((q.episode_id, q.machine_id), ()):
            spins[i, play.tick] = PAYOUTS.index(play.payout)
    return spins, cutoffs


def lagged_spins(spins):
    padded = np.pad(spins, ((0, 0), (len(LAGS), 0)), constant_values=MISSING)
    return np.stack([padded[:, len(LAGS) - lag:-lag] for lag in LAGS], axis=-1)


def _window_features(prefix, machines, ticks):
    end = prefix[machines, ticks]
    counts = [end - prefix[machines, np.maximum(ticks - w, 0)] if w else end for w in WINDOWS]
    totals = [values.sum(axis=1, keepdims=True) for values in counts]
    rates = [values / np.maximum(total, 1) for values, total in zip(counts, totals)]
    deltas = [rates[0] - rates[2], rates[1] - rates[-1]]
    return np.column_stack([*rates, *[np.log1p(n) for n in totals], *deltas])


def feature_rows(queries, spins, machines, ticks, *, peers=True):
    """The feature at tick t only uses entries with tick < t."""
    types = np.array([q.machine_type for q in queries])[machines]
    signals = np.array([q.signal for q in queries])[machines]
    prefix = np.pad(np.cumsum(spins[..., None] == np.arange(len(PAYOUTS)), axis=1, dtype=np.int32),
                    ((0, 0), (1, 0), (0, 0)))
    lagged = lagged_spins(spins)[machines, ticks]
    lag_flags = (lagged[..., None] == np.arange(MISSING, len(PAYOUTS))).reshape(len(ticks), -1)
    win = lagged > 0
    win_pairs = np.column_stack([win[:, a - 1] & win[:, b - 1] for a, b in LAG_PAIRS])
    static = np.column_stack([types[:, None] == ("steady", "balanced", "jackpot"),
                              signals[:, None] == (-1, 0, 1)])
    columns = [static, _window_features(prefix, machines, ticks), lag_flags, win_pairs]
    if peers:
        episodes = defaultdict(list)
        for i, q in enumerate(queries):
            episodes[q.episode_id].append(i)
        peer_prefix = np.empty_like(prefix)
        for members in episodes.values():
            own = prefix[members]
            peer_prefix[members] = own.sum(axis=0) - own
        columns.append(_window_features(peer_prefix, machines, ticks))
    return np.column_stack(columns)
