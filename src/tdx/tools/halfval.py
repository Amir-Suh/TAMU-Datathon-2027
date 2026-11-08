"""Matched test: do TRUE validation probabilities used as soft targets help on the OTHER half of validation?

Design (everything matched except the soft-target term):
  * validation episodes are split in two halves, stratified by condition, at random (`--split` picks the shuffle seed);
  * a model is fit on train+validation+test histories with soft targets from half `--role` (w_cut > 0), the held-out half
    has NO targets; the baseline (w_cut = 0) is the ordinary dev-mode model with the SAME torch seed;
  * both are scored on the held-out half only.

usage:
  python -m tdx.tools.halfval fit --seed 1 --split 0 --role 0 --wcut 20 --out DIR     # one fit (GPU, ~3.5 min)
  python -m tdx.tools.halfval report --dir DIR                                        # paired differences
Files written: DIR/s{seed}_p{split}_r{role}_w{wcut}.npy  (validation forecasts [96,8,5]).
Baselines (wcut 0) do not depend on the role, so they are written once as s{seed}_base.npy.
"""
import argparse
import glob
import os
import re
from dataclasses import replace
from pathlib import Path
import numpy as np
from tdx.data import load_round, with_answers
from tdx.fit import FitSpec, dev_spec
from tdx.run import fit_round


def halves(condition, split_seed):
    rng = np.random.RandomState(1000 + split_seed)
    side = np.zeros(len(condition), int)
    for c in np.unique(condition):
        ii = np.where(condition == c)[0]
        rng.shuffle(ii)
        side[ii[len(ii) // 2:]] = 1
    return side


def fit_one(seed, split_seed, role, wcut, out, data=None):
    from tdx.arms.seq import build
    rnd = with_answers(load_round(data or 'data/main', answers=False))
    v = rnd.splits['validation']
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    arm = build(h=128, msg=False, epochs=int(os.environ.get('HALFVAL_EPOCHS', 100)), transductive=True, factor=True, seed=seed, w_cut=float(wcut or 20.))
    if wcut:
        side = halves(np.array(v.condition), split_seed)
        target = v.target.copy()
        target[side != role] = np.nan                     # held-out half: no supervision at all
        rnd2 = replace(rnd, splits={**rnd.splits, 'validation': replace(v, target=target)})
        spec = FitSpec(('train', 'validation', 'test'), ('train',), ('validation',))
        name = f's{seed}_p{split_seed}_r{role}_w{wcut}.npy'
    else:
        rnd2, spec = rnd, dev_spec(True)
        name = f's{seed}_base.npy'
    arm.fit(fit_round(rnd2, spec), spec)
    pred = arm.predict(replace(v, label=None, condition=None, target=None, reference=None))
    np.save(out / name, pred)
    print('saved', out / name)


def subset_points(p, target, reference, condition, mask):
    err = lambda q: ((q - target) ** 2).sum(-1).mean(1)
    e, ref = err(p), err(reference)
    R = np.mean([e[mask & (condition == c)].mean() / ref[mask & (condition == c)].mean() for c in np.unique(condition)])
    return 60 / (1 + R)


def report(directory, data=None):
    rnd = with_answers(load_round(data or 'data/main', answers=False))
    v = rnd.splits['validation']
    cond = np.array(v.condition)
    rows = []
    for f in sorted(glob.glob(str(Path(directory) / 's*_p*_r*_w*.npy'))):
        m = re.search(r's(\d+)_p(\d+)_r(\d+)_w(\d+)\.npy$', f)
        seed, split_seed, role, wcut = map(int, m.groups())
        base = Path(directory) / f's{seed}_base.npy'
        if not base.exists():
            continue
        held = halves(cond, split_seed) != role
        a = subset_points(np.load(f), v.target, v.reference, cond, held)
        b = subset_points(np.load(base), v.target, v.reference, cond, held)
        rows.append((wcut, seed, split_seed, role, a, b))
    print('w_cut seed split role  with-targets  baseline   diff   (scored on the held-out half only)')
    for r in rows:
        print('%5d %4d %5d %4d   %10.3f %9.3f %+7.3f' % (r[0], r[1], r[2], r[3], r[4], r[5], r[4] - r[5]))
    for w in sorted({r[0] for r in rows}):
        d = np.array([r[4] - r[5] for r in rows if r[0] == w])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
        print('w_cut %3d: mean diff %+.3f  se %.3f  (%d cells, %d positive)' % (w, d.mean(), se, len(d), (d > 0).sum()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--seed', type=int, required=True)
    f.add_argument('--split', type=int, default=0)
    f.add_argument('--role', type=int, default=0, choices=(0, 1))
    f.add_argument('--wcut', type=int, default=20)
    f.add_argument('--out', required=True)
    f.add_argument('--data')
    r = sub.add_parser('report')
    r.add_argument('--dir', required=True)
    r.add_argument('--data')
    a = p.parse_args(argv)
    if a.cmd == 'fit':
        fit_one(a.seed, a.split, a.role, a.wcut, a.out, a.data)
    else:
        report(a.dir, a.data)


if __name__ == '__main__':
    main()
