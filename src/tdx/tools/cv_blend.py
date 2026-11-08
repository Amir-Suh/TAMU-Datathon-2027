"""Repeated grouped k-fold evaluation of factored blends, exactly as frozen in claude_output/notes/frozen_candidates.json.

Weights (win and size simplex vectors per candidate) are fit by SLSQP on the TRAINING folds' squared error against the
validation true probabilities and scored on the reserved fold. The 4 reserved folds are concatenated into one 96-episode
out-of-fold forecast set per repeat and scored with the official scorer; the reported number is the mean over repeats.

usage: python -m tdx.tools.cv_blend [--preds weights/val_preds] [--spec claude_output/notes/frozen_candidates.json]
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from tdx.data import load_round, with_answers
from tdx.score import score


def split(p):
    pw = 1 - p[..., 0]
    return pw, p[..., 1:] / np.maximum(pw[..., None], 1e-9)


def blend(S, a, b):
    pw = sum(w * s[0] for w, s in zip(a, S))
    mix = sum(w * s[1] for w, s in zip(b, S))
    return np.concatenate([(1 - pw)[..., None], pw[..., None] * mix], -1)


def fit_weights(S, T, idx):
    n = len(S)
    f = lambda w: ((blend(S, w[:n], w[n:])[idx] - T[idx]) ** 2).sum()
    cons = [{'type': 'eq', 'fun': lambda w: w[:n].sum() - 1}, {'type': 'eq', 'fun': lambda w: w[n:].sum() - 1}]
    r = minimize(f, np.full(2 * n, 1 / n), bounds=[(0, 1)] * (2 * n), constraints=cons, method='SLSQP')
    return r.x[:n], r.x[n:]


def folds(condition, k, seed):
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(condition), int)
    for c in np.unique(condition):
        ii = np.where(condition == c)[0]
        rng.shuffle(ii)
        fold[ii] = np.arange(len(ii)) % k
    return fold


def oof(S, T, cond, k, seed):
    fold = folds(cond, k, seed)
    out = np.zeros_like(T)
    ws = []
    for f in range(k):
        tr = np.where(fold != f)[0]
        a, b = fit_weights(S, T, tr)
        te = fold == f
        out[te] = blend(S, a, b)[te]
        ws.append(np.concatenate([a, b]))
    return out, np.mean(ws, 0)


def episode_error(p, T):
    return ((p - T) ** 2).sum(-1).mean(1)


def points_from_errors(err, ref_err, cond, idx_by_c):
    R = np.mean([err[ix].mean() / ref_err[ix].mean() for ix in idx_by_c])
    return 60 / (1 + R)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preds', default='weights/val_preds')
    ap.add_argument('--spec', default='claude_output/notes/frozen_candidates.json')
    ap.add_argument('--data', default='data/main')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    P = {k: np.load(Path(a.preds) / f'{k}.npy') for k in spec['members']}
    rnd = with_answers(load_round(a.data, answers=False))
    v = rnd.splits['validation']
    T, ref, cond = v.target, v.reference, np.array(v.condition)
    ref_err = episode_error(ref, T)
    idx_by_c = [np.where(cond == c)[0] for c in np.unique(cond)]
    seeds = range(5)
    names = list(spec['candidates'])
    res, errs, weights = {}, {}, {}
    for n in names:
        S = [split(P[m]) for m in spec['candidates'][n]]
        pts, es, ws = [], [], []
        for s in seeds:
            p, w = oof(S, T, cond, 4, s)
            pts.append(score(rnd, p)['points'])
            es.append(episode_error(p, T))
            ws.append(w)
        res[n] = np.array(pts)
        errs[n] = np.mean(es, 0)            # per-episode error averaged over repeats (for the paired bootstrap)
        weights[n] = np.mean(ws, 0)
    base = 'F_submitted_recipe'
    print('Out-of-fold validation points, 4-fold grouped CV x 5 repeats (weights never fit on the scored episodes)')
    print('%-22s %9s %8s   %s' % ('candidate', 'mean pts', 'sd rep', 'members'))
    for n in names:
        print('%-22s %9.3f %8.3f   %s' % (n, res[n].mean(), res[n].std(ddof=1), '+'.join(spec['candidates'][n])))
    print('\nPaired episode bootstrap of the gain over %s (stratified by condition, %d resamples):' % (base, a.boot))
    rng = np.random.RandomState(0)
    for n in names:
        if n == base:
            continue
        d = []
        for _ in range(a.boot):
            idx = [rng.choice(ix, len(ix)) for ix in idx_by_c]
            d.append(points_from_errors(errs[n], ref_err, cond, idx) - points_from_errors(errs[base], ref_err, cond, idx))
        lo, hi = np.percentile(d, [2.5, 97.5])
        gain = res[n].mean() - res[base].mean()
        ok = gain >= 0.2 and lo > 0
        print('  %-22s gain %+6.3f  95%% interval [%+.3f, %+.3f]  ->  %s' % (n, gain, lo, hi, 'REPLACES F (rule met)' if ok else 'keep F'))
    if 'A_ce12_matched' in names and 'B_size2_12_matched' in names:
        d = []
        for _ in range(a.boot):
            idx = [rng.choice(ix, len(ix)) for ix in idx_by_c]
            d.append(points_from_errors(errs['B_size2_12_matched'], ref_err, cond, idx) - points_from_errors(errs['A_ce12_matched'], ref_err, cond, idx))
        print('\nMatched-seed test of the size x2 loss (B - A, same seeds 41-52): gain %+.3f, 95%% interval [%+.3f, %+.3f]' % (
            res['B_size2_12_matched'].mean() - res['A_ce12_matched'].mean(), *np.percentile(d, [2.5, 97.5])))
    print('\nMean fold weights (win | size) over the members in order:')
    for n in names:
        k = len(spec['candidates'][n])
        print('  %-22s win %s | size %s' % (n, np.round(weights[n][:k], 2), np.round(weights[n][k:], 2)))
    # non-CV references (reported for context only)
    S = [split(P[m]) for m in spec['candidates'][base]]
    ref_p = blend(S, [0.9017, 0.0414, 0.0569], [0.2465, 0.4948, 0.2587])
    print('\nReference, NOT cross-validated: clean-refit weights (Andrew) on recipe F members: %.3f' % score(rnd, ref_p)['points'])


if __name__ == '__main__':
    main()
