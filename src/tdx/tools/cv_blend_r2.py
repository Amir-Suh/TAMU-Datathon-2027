"""Round-two blend check on the NEW validation set: fixed main-round weights (candB_main) vs weights tuned by grouped k-fold.

Tuned weights are fit (SLSQP, squared error vs the true validation probabilities) on the training folds only and scored on the
reserved fold; the folds are concatenated into one 96-episode out-of-fold forecast set per repeat. Fixed weights need no fitting.
Gain is the paired, condition-stratified episode bootstrap of points on the 40-point scale; gate (frozen): gain >= 0.2, interval > 0.
usage: python -m tdx.tools.cv_blend_r2 --data <public> --gru <seqens model dir> --tfm <tfmens model dir> [--emit blend.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
from tdx.data import load_round, with_answers
from tdx.fit import dev_spec
from tdx.tools.cv_blend import split, blend, oof, episode_error, fit_weights
from tdx.paths import data_root
from tdx.run import fit_round
from tdx.tools.round2_blend import PRESETS


def pts(err, ref, idx):
    return 40 / (1 + np.mean([err[i].mean() / ref[i].mean() for i in idx]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round', default='adaptation'); ap.add_argument('--data', required=True)
    ap.add_argument('--gru', required=True); ap.add_argument('--tfm', required=True)
    ap.add_argument('--repeats', type=int, default=5); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--emit'); a = ap.parse_args(argv)
    from tdx.arms.m3flip import build as build_m3
    from tdx.arms.seqens.arm import SeqEnsArm
    from tdx.arms.tfmens.arm import TfmEnsArm
    rnd = load_round(data_root(a.data, a.round), a.round, answers=False)
    val = rnd.splits['validation']
    m3 = build_m3(); m3.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    P = [m3.predict(val), SeqEnsArm.load(a.gru).predict(val), TfmEnsArm.load(a.tfm).predict(val)]
    truth = with_answers(rnd); v = truth.splits['validation']
    T, ref, cond = v.target, v.reference, np.array(v.condition)
    S = [split(p) for p in P]
    ref_err = episode_error(ref, T)
    idx = [np.where(cond == c)[0] for c in np.unique(cond)]
    w, s = PRESETS['candB_main']
    fixed = episode_error(blend(S, w, s), T)
    eq = episode_error(blend(S, [1 / 3] * 3, [1 / 3] * 3), T)
    tuned_errs, tuned_pts, ws = [], [], []
    for seed in range(a.repeats):
        p, wv = oof(S, T, cond, 4, seed)
        e = episode_error(p, T); tuned_errs.append(e); tuned_pts.append(pts(e, ref_err, idx)); ws.append(wv)
    tuned = np.mean(tuned_errs, 0)
    print('Validation points (40 scale), 96 episodes. tuned = 4-fold grouped CV x %d repeats, weights never fit on scored episodes' % a.repeats)
    print('fixed candB_main (no fitting): %.3f' % pts(fixed, ref_err, idx))
    print('equal thirds    (no fitting): %.3f' % pts(eq, ref_err, idx))
    print('CV-tuned out-of-fold        : %.3f  (sd over repeats %.3f)' % (np.mean(tuned_pts), np.std(tuned_pts, ddof=1)))
    rng = np.random.RandomState(0)
    for name, base in (('fixed candB_main', fixed), ('equal thirds', eq)):
        d = []
        for _ in range(a.boot):
            ix = [rng.choice(i, len(i)) for i in idx]
            d.append(pts(tuned, ref_err, ix) - pts(base, ref_err, ix))
        lo, hi = np.percentile(d, [2.5, 97.5]); g = np.mean(tuned_pts) - pts(base, ref_err, idx)
        print('tuned - %-17s gain %+.3f  95%% interval [%+.3f, %+.3f] -> %s' % (name, g, lo, hi, 'TUNED WINS (gate met)' if g >= 0.2 and lo > 0 else 'keep fixed'))
    wm = np.mean(ws, 0)
    print('mean fold weights win [m3,gru,tfm]', np.round(wm[:3], 3), 'size', np.round(wm[3:], 3))
    print('candB_main weights        win', np.round(w, 3), 'size', np.round(s, 3))
    if a.emit:
        win, size = np.maximum(wm[:3], 0), np.maximum(wm[3:], 0)
        cfg = dict(version=1, kind='factored', arms=['m3flip/m3', 'seqens/r2gru', 'tfmens/r2tfm'],
                   weights=(win / win.sum()).tolist(), size_weights=(size / size.sum()).tolist())
        Path(a.emit).write_text(json.dumps(cfg, indent=2) + '\n'); print('wrote', a.emit)


if __name__ == '__main__':
    main()
