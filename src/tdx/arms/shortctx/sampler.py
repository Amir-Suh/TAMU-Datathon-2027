"""Exact-layout sampler and dataset generator for shortctx arm."""

import numpy as np

# Compute offset bucket lookup table: 0..39 individual, log-spaced for 40..255 (64 buckets total)
_edges = np.geomspace(40, 255, 25)
BUCKET_LUT = np.zeros(512, dtype=np.int64)
BUCKET_LUT[:40] = np.arange(40)
BUCKET_LUT[40:256] = 40 + np.digitize(np.arange(40, 256), _edges[1:-1])
BUCKET_LUT[256:] = 63
N_OFFSET_BUCKETS = 64


def sample_layout(y_row, cutoff_prime, rng):
    """Sample layout of 64 observed items before cutoff_prime.

    Args:
        y_row: 1D array of outcomes (-1 for missing, 0..4 for observed).
        cutoff_prime: int pseudo-cutoff tick.
        rng: np.random.RandomState, np.random.Generator, or int seed.

    Returns:
        (tokens, offsets) where:
            tokens: int8 array of shape (64,) with outcomes in {0..4}.
            offsets: int16 array of shape (64,) = cutoff_prime - tick.
        The last 40 items (indices 24..63) are consecutive ticks
        cutoff_prime-40..cutoff_prime-1 (all must be observed).
        The 24 older items (indices 0..23) are drawn without replacement
        from observed ticks < cutoff_prime-40.
        Returns None when the machine does not have 64 observed ticks before cutoff_prime
        (or if any of the recent 40 ticks is missing, or older observed count < 24).
    """
    if cutoff_prime < 64:
        return None

    # Check recent 40 consecutive ticks: cutoff_prime - 40 .. cutoff_prime - 1
    recent_ticks = np.arange(cutoff_prime - 40, cutoff_prime, dtype=np.int16)
    if (y_row[recent_ticks] < 0).any():
        return None

    # Check older ticks: < cutoff_prime - 40
    older_candidates = np.where(y_row[:cutoff_prime - 40] >= 0)[0]
    if len(older_candidates) < 24:
        return None

    if isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(int(rng))

    older_ticks = rng.choice(older_candidates, size=24, replace=False)
    older_ticks = np.sort(older_ticks).astype(np.int16)

    all_ticks = np.concatenate([older_ticks, recent_ticks])
    tokens = y_row[all_ticks].astype(np.int8)
    offsets = (cutoff_prime - all_ticks).astype(np.int16)

    return tokens, offsets


class ShortCtxDataset:
    """Pre-indexed candidate dataset from round histories (strictly ticks < cutoff)."""

    def __init__(self, rnd, spec):
        all_y = []
        all_mt = []
        all_co = []

        for name in spec.tick_splits:
            s = rnd.splits[name]
            all_y.append(s.y.reshape(-1, s.y.shape[-1]))
            all_mt.append(s.mtype.reshape(-1))
            all_co.append(np.repeat(s.cutoff, 8))

        self.y = np.concatenate(all_y, axis=0)  # [N_machines, T]
        self.mtype = np.concatenate(all_mt, axis=0).astype(np.int64)  # [N_machines]
        self.cutoff = np.concatenate(all_co, axis=0).astype(np.int64)  # [N_machines]

        # Strictly mask ticks at or after cutoff
        T = self.y.shape[-1]
        mask = np.arange(T)[None, :] >= self.cutoff[:, None]
        self.y[mask] = -1

        # Identify candidate pseudo-cutoffs cp in range(80, cutoff)
        exact_cands = []
        full_cands = []
        for m_idx in range(len(self.y)):
            co = self.cutoff[m_idx]
            y_row = self.y[m_idx]
            for cp in range(80, co):
                if y_row[cp] < 0:
                    continue
                # Exact layout candidate: recent 40 observed and >= 24 older observed
                if (y_row[cp - 40:cp] >= 0).all() and (y_row[:cp - 40] >= 0).sum() >= 24:
                    exact_cands.append((m_idx, cp))
                # Full history candidate: 64 consecutive observed
                if cp >= 64 and (y_row[cp - 64:cp] >= 0).all():
                    full_cands.append((m_idx, cp))

        self.exact_cands = np.array(exact_cands, dtype=np.int32)
        self.full_cands = np.array(full_cands, dtype=np.int32)


def sample_batch(dataset, batch_size=256, rng=None, exact_ratio=0.5, include_flag=False):
    """Sample a batch of training items with mixed exact-layout and full-history samples.

    Returns:
        (tokens, offsets, mtype, target) if not include_flag
        or (tokens, offsets, mtype, target, flag) if include_flag.
    """
    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(rng if rng is not None else 42)

    B = batch_size
    B_exact = int(round(B * exact_ratio))
    B_full = B - B_exact

    parts_tok = []
    parts_off = []
    parts_mt = []
    parts_tgt = []
    parts_flg = []

    # 1. Full-history samples (flag = 0)
    if B_full > 0:
        idx_f = rng.randint(0, len(dataset.full_cands), size=B_full)
        m_f = dataset.full_cands[idx_f, 0]
        cp_f = dataset.full_cands[idx_f, 1]

        ticks_f = cp_f[:, None] - 64 + np.arange(64)[None, :]
        tok_f = dataset.y[m_f[:, None], ticks_f].astype(np.int8)
        off_f = (cp_f[:, None] - ticks_f).astype(np.int16)
        tgt_f = dataset.y[m_f, cp_f].astype(np.int64)
        mt_f = dataset.mtype[m_f]
        flg_f = np.zeros(B_full, dtype=np.int64)

        parts_tok.append(tok_f)
        parts_off.append(off_f)
        parts_mt.append(mt_f)
        parts_tgt.append(tgt_f)
        parts_flg.append(flg_f)

    # 2. Exact-layout samples (flag = 1)
    if B_exact > 0:
        idx_e = rng.randint(0, len(dataset.exact_cands), size=B_exact)
        m_e = dataset.exact_cands[idx_e, 0]
        cp_e = dataset.exact_cands[idx_e, 1]

        rec_ticks = cp_e[:, None] - 40 + np.arange(40)[None, :]
        rec_tok = dataset.y[m_e[:, None], rec_ticks].astype(np.int8)
        rec_off = (cp_e[:, None] - rec_ticks).astype(np.int16)

        old_tok = np.empty((B_exact, 24), dtype=np.int8)
        old_off = np.empty((B_exact, 24), dtype=np.int16)

        for k in range(B_exact):
            y_m = dataset.y[m_e[k]]
            cp_k = cp_e[k]
            older_obs = np.where(y_m[:cp_k - 40] >= 0)[0]
            chosen = np.sort(rng.choice(older_obs, size=24, replace=False))
            old_tok[k] = y_m[chosen]
            old_off[k] = cp_k - chosen

        tok_e = np.concatenate([old_tok, rec_tok], axis=1)
        off_e = np.concatenate([old_off, rec_off], axis=1)
        tgt_e = dataset.y[m_e, cp_e].astype(np.int64)
        mt_e = dataset.mtype[m_e]
        flg_e = np.ones(B_exact, dtype=np.int64)

        parts_tok.append(tok_e)
        parts_off.append(off_e)
        parts_mt.append(mt_e)
        parts_tgt.append(tgt_e)
        parts_flg.append(flg_e)

    tokens = np.concatenate(parts_tok, axis=0)
    offsets = np.concatenate(parts_off, axis=0)
    mtype = np.concatenate(parts_mt, axis=0)
    target = np.concatenate(parts_tgt, axis=0)
    flag = np.concatenate(parts_flg, axis=0)

    # Shuffle within batch
    perm = rng.permutation(B)
    tokens = tokens[perm]
    offsets = offsets[perm]
    mtype = mtype[perm]
    target = target[perm]
    flag = flag[perm]

    if include_flag:
        return tokens, offsets, mtype, target, flag
    return tokens, offsets, mtype, target


def generate_batches(dataset, batch_size=256, n_batches=100, rng=None, exact_ratio=0.5, include_flag=False):
    """Generator yielding training batches."""
    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(rng if rng is not None else 42)

    for _ in range(n_batches):
        yield sample_batch(dataset, batch_size=batch_size, rng=rng, exact_ratio=exact_ratio, include_flag=include_flag)
