"""Leak-free tail holdout backtest harness."""

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = str(min(12, int(os.environ.get(_var, "12"))))

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import numpy as np

from .data import load_round
from .fit import dev_spec
from .paths import data_root
from .run import arm_module, build_arm, fit_round, parse_set

TAIL_SIGNAL_POLICY = "neutral-at-pseudo-cutoff-v1"


class BacktestResult(float, Mapping):
    def __new__(cls, logloss, n, first, mid, last):
        obj = super().__new__(cls, float(logloss))
        obj.logloss = float(logloss)
        obj.n = int(n)
        obj.first = float(first)
        obj.mid = float(mid)
        obj.last = float(last)
        return obj

    def __getitem__(self, key):
        if key in ('logloss', 'n', 'first', 'mid', 'last'):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(('logloss', 'n', 'first', 'mid', 'last'))

    def __len__(self):
        return 5

    def keys(self):
        return ['logloss', 'n', 'first', 'mid', 'last']

    def values(self):
        return [self.logloss, self.n, self.first, self.mid, self.last]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]

    def asdict(self):
        return dict(self.items())

    def __repr__(self):
        return f"BacktestResult(logloss={self.logloss:.4f}, n={self.n}, first={self.first:.4f}, mid={self.mid:.4f}, last={self.last:.4f})"


def truncate(rnd, hold):
    """Return (rnd_trunc, full_splits) where rnd_trunc has cutoffs reduced by hold
    and y >= cutoff - hold set to -1. Earlier queries use neutral signal."""
    full_splits = rnd.splits
    trunc_splits = {}
    for name, s in full_splits.items():
        new_cutoff = (s.cutoff - hold).astype(s.cutoff.dtype)
        new_y = s.y.copy()
        mask = np.broadcast_to(
            np.arange(new_y.shape[-1])[None, None, :] >= new_cutoff[:, None, None],
            new_y.shape,
        )
        new_y[mask] = -1
        trunc_splits[name] = replace(
            s,
            cutoff=new_cutoff,
            y=new_y,
            signal=np.where(new_cutoff[:, None] == s.cutoff[:, None], s.signal, 0),
            n_obs=(new_y >= 0).sum(axis=-1).astype(s.n_obs.dtype),
            label=None,
            condition=None,
            target=None,
            reference=None,
        )
    rnd_trunc = replace(rnd, splits=trunc_splits)
    return rnd_trunc, full_splits


def make_pseudo_split(val_trunc, full_val, j, short=False, rng=None):
    """Create pseudo-query split from validation at cutoff_trunc + j."""
    pseudo_cutoff = (val_trunc.cutoff + j).astype(val_trunc.cutoff.dtype)
    y_j = full_val.y.copy()
    mask = np.broadcast_to(
        np.arange(y_j.shape[-1])[None, None, :] >= pseudo_cutoff[:, None, None],
        y_j.shape,
    )
    y_j[mask] = -1

    if short:
        if rng is None:
            rng = np.random.RandomState(0)
        E, M, _ = y_j.shape
        for e in range(E):
            c = int(pseudo_cutoff[e])
            n_older = max(0, c - 40)
            if n_older > 24:
                for m in range(M):
                    chosen = rng.choice(n_older, size=24, replace=False)
                    older_mask = np.ones(n_older, dtype=bool)
                    older_mask[chosen] = False
                    y_j[e, m, :n_older][older_mask] = -1

    return replace(
        val_trunc,
        cutoff=pseudo_cutoff,
        y=y_j,
        signal=np.where(pseudo_cutoff[:, None] == full_val.cutoff[:, None], full_val.signal, 0),
        n_obs=(y_j >= 0).sum(axis=-1).astype(val_trunc.n_obs.dtype),
    )


def tail_logloss(arm_name, variant, hold=30, short=False, overrides=None, seed=0, data=None, round_name='main'):
    rnd = load_round(data_root(data, round_name), round_name, answers=False)
    rt, full = truncate(rnd, hold)

    module = arm_module(arm_name)
    params = dict(module.VARIANTS[variant])
    if overrides:
        params.update(overrides)
    arm = build_arm(module, params)
    spec = dev_spec(bool(params.get('transductive', False)))

    # Pass only authorized supervision. For arms expecting split.label (like PooledArm),
    # supply dummy missing labels (-1) if split.label is None so training relies on history.
    fitted_rt = fit_round(rt, spec)
    splits = dict(fitted_rt.splits)
    for name in spec.label_splits:
        s = splits[name]
        if s.label is None:
            splits[name] = replace(s, label=np.full(s.query_ids.shape, -1, dtype=np.int8))
    fitted_rt = replace(fitted_rt, splits=splits)
    arm.fit(fitted_rt, spec)

    val_trunc = rt.splits['validation']
    val_full = full['validation']
    cutoff_trunc = val_trunc.cutoff
    E = len(val_trunc.ids)

    rng = np.random.RandomState(seed) if short else None
    losses_by_j = []
    for j in range(hold):
        split_j = make_pseudo_split(val_trunc, val_full, j, short=short, rng=rng)
        pred = arm.predict(split_j)

        target_tick = cutoff_trunc + j
        target_y = val_full.y[np.arange(E)[:, None], np.arange(8)[None, :], target_tick[:, None]]
        valid_mask = target_y >= 0
        if not np.any(valid_mask):
            losses_by_j.append(np.array([], dtype=np.float64))
            continue
        outcomes = target_y[valid_mask]
        e_idx, m_idx = np.nonzero(valid_mask)
        probs = np.clip(pred[e_idx, m_idx, outcomes], 1e-6, 1.0)
        losses_by_j.append(-np.log(probs))

    flat_losses = np.concatenate(losses_by_j)
    mean_loss = float(np.mean(flat_losses)) if len(flat_losses) > 0 else float('nan')
    n = len(flat_losses)

    if hold >= 30:
        first = float(np.mean(np.concatenate(losses_by_j[:10])))
        mid_start = (hold - 10) // 2
        mid = float(np.mean(np.concatenate(losses_by_j[mid_start:mid_start + 10])))
        last = float(np.mean(np.concatenate(losses_by_j[hold - 10:hold])))
    else:
        k = min(10, hold)
        first = float(np.mean(np.concatenate(losses_by_j[:k]))) if k > 0 else mean_loss
        mid = mean_loss
        last = float(np.mean(np.concatenate(losses_by_j[max(0, hold - k):]))) if k > 0 else mean_loss

    return BacktestResult(logloss=mean_loss, n=n, first=first, mid=mid, last=last)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--hold', type=int, default=30)
    p.add_argument('--short', action='store_true')
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--data')
    p.add_argument('--round', default='main')
    p.add_argument('--seed', type=int, default=0)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    overrides = parse_set(a.set)
    result = tail_logloss(
        a.arm,
        a.variant,
        hold=a.hold,
        short=a.short,
        overrides=overrides,
        seed=a.seed,
        data=a.data,
        round_name=a.round,
    )
    short_int = 1 if a.short else 0
    print(f"backtest {a.arm}/{a.variant} hold={a.hold} short={short_int} logloss={result.logloss:.4f} n={result.n} first={result.first:.4f} mid={result.mid:.4f} last={result.last:.4f}")


if __name__ == '__main__':
    main()
