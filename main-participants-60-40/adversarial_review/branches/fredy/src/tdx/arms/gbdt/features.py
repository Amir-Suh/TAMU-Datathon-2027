"""Causal features, vectorized over episodes and machines (never row loops)."""
import numpy as np
from tdx.feat import lag_classes
from tdx.data import PAYOUTS


def features(y, mtype):
    """Return features for all ticks 0..T, including the next query tick.

    Rates use observed values only; empty rates/aggregates are -1. EMA decay
    advances per observation, while divergence windows use calendar ticks.
    Neighbour 1 uses each other machine's last observation; 5/10 pool
    observed other-machine cells in the last N calendar ticks.
    """
    E, M, T = y.shape
    z = np.concatenate((y, np.full((E, M, 1), -1, y.dtype)), axis=2)
    obs = z >= 0
    win = z > 0
    def past_sum(a, width=None):
        cs = np.concatenate((np.zeros((*a.shape[:2], 1)), np.cumsum(a, axis=2)), axis=2)[..., :-1]
        if width is not None:
            cs = cs - np.concatenate((np.zeros((*a.shape[:2], width)), cs[..., :-width]), axis=2)[..., :T+1]
        return cs
    count = past_sum(obs)
    lags = lag_classes(z, 8)
    arrays = [lags.astype(float)]
    names = [f'lag{k}' for k in range(1, 9)]
    groups = ['lag'] * 8
    def add(a, name, group):
        arrays.append(np.asarray(a, dtype=float)[..., None])
        names.append(name)
        groups.append(group)
    add(np.where((lags[..., 0] >= 0) & (lags[..., 1] >= 0), 5*lags[..., 0]+lags[..., 1], -1), 'code12', 'lag')
    for half in (4, 12, 24, 64):
        decay = 2. ** (-1. / half)
        mass = np.zeros((E, M, 5))
        denom = np.zeros((E, M))
        values = np.full((E, M, T+1, 5), -1.)
        for t in range(T+1):
            values[:, :, t] = np.divide(mass, denom[..., None], out=np.full_like(mass, -1.), where=denom[..., None] > 0)
            valid = obs[:, :, t]
            mass *= np.where(valid, decay, 1.)[..., None]
            mass += (z[:, :, t, None] == np.arange(5))
            denom = denom * np.where(valid, decay, 1.) + valid
        arrays.append(values)
        names.extend(f'ema{half}_c{k}' for k in range(5))
        groups.extend(['ema'] * 5)
    # Pack observations in tick order; gathering by prefix count selects only past.
    order = np.argsort(np.where(obs, np.arange(T+1), T+1), axis=2, kind='stable')
    packed = np.take_along_axis(win, order, axis=2)
    prefix = np.concatenate((np.zeros((E, M, 1)), np.cumsum(packed, axis=2)), axis=2)
    for width in (5, 20, 50):
        end = np.take_along_axis(prefix, count.astype(int), axis=2)
        begin = np.take_along_axis(prefix, np.maximum(0, count-width).astype(int), axis=2)
        add(end-begin, f'wins_obs{width}', 'rate')
    add(count, 'n_obs_before', 'rate')
    add(np.broadcast_to(mtype[..., None], count.shape), 'machine_type', 'type')
    payout = np.take(np.asarray(PAYOUTS), np.maximum(z, 0)) * obs
    # Pool all observed neighbour cells; absent cells never act as losses.
    # Carry forward only for the last-observation neighbour group.
    idx = np.maximum.accumulate(np.where(obs, np.arange(T+1), -1), axis=2)
    latest = np.take_along_axis(z, np.maximum(idx, 0), axis=2)
    latest = np.where(idx >= 0, latest, -1)
    for width in (1, 5, 10):
        source = latest if width == 1 else z
        source_obs = source >= 0
        source_win = source > 0
        source_payout = np.take(np.asarray(PAYOUTS), np.maximum(source, 0)) * source_obs
        den = past_sum(source_obs, width)
        den = den.sum(axis=1, keepdims=True) - den
        for a, name in ((source_win, 'win'), (source_payout, 'payout')):
            num = past_sum(a, width)
            num = num.sum(axis=1, keepdims=True)-num
            add(np.divide(num, den, out=np.full_like(num, -1.), where=den > 0), f'other{width}_{name}', 'neighbour')
        mx = np.full((E, M, T+1), -1.)
        for lag in range(1, width+1):
            prev = source[..., :T+1-lag]
            # Eight machine iterations are small fixed axes, not training rows.
            for m in range(M):
                mx[:, m, lag:] = np.maximum(mx[:, m, lag:], np.max(np.delete(prev, m, axis=1), axis=1))
        add(mx, f'other{width}_maxclass', 'neighbour')
    rates = []
    episode_rates = []
    valid_rates, valid_episode = [], []
    for width in (15, 80):
        den, num = past_sum(obs, width), past_sum(win, width)
        valid_rates.append(den > 0)
        rates.append(np.divide(num, den, out=np.zeros_like(num), where=den > 0))
        ed, en = den.sum(axis=1, keepdims=True), num.sum(axis=1, keepdims=True)
        valid_episode.append(np.broadcast_to(ed > 0, count.shape))
        episode_rates.append(np.broadcast_to(np.divide(en, ed, out=np.zeros_like(en), where=ed > 0), count.shape))
    add(np.where(valid_rates[0] & valid_rates[1], rates[0]-rates[1], np.nan), 'self_divergence', 'divergence')
    add(np.where(valid_episode[0] & valid_episode[1], episode_rates[0]-episode_rates[1], np.nan), 'episode_divergence', 'divergence')
    add(np.broadcast_to(np.arange(T+1)/256., count.shape), 'position', 'position')
    return np.concatenate(arrays, axis=-1).astype(np.float32), names, groups


def query_features(split):
    # Mask corrupt future storage explicitly, independently of supplied n_obs.
    y = np.where(np.arange(split.y.shape[-1])[None, None, :] < split.cutoff[:, None, None], split.y, -1)
    x, names, groups = features(y, split.mtype)
    return x[np.arange(len(split.ids))[:, None], np.arange(8)[None, :], split.cutoff[:, None]], names, groups
