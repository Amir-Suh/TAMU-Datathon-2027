"""Choose the round-two blend among a few fixed presets by dev-mode score on the NEW round's validation.

usage: python -m tdx.tools.round2_blend --round adaptation --data <public> --gru <seqens export dir> --tfm <tfmens export dir> --out blend.json

Presets are the structures validated on main (m3 / GRU ensemble / transformer ensemble, factored win-vs-size blend).
Default: the FROZEN candidate-4 (B) weights from the main-round cross-validation (preset candB_main); every preset is still
scored on the new validation and printed as a diagnostic. With --select the best preset by new-validation points is used
instead (selection on 96 episodes, optimistic). If the new validation has no true probabilities, candB_main is used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tdx.data import load_round, with_answers
from tdx.fit import dev_spec
from tdx.paths import data_root
from tdx.run import fit_round
from tdx.score import score

ARMS = ['m3flip/m3', 'seqens/%s', 'tfmens/%s']
# name -> (win weights, size weights) over (m3, gru ensemble, transformer ensemble)
PRESETS = {
    'm3_only': ([1, 0, 0], [1, 0, 0]),
    'gru_only': ([0, 1, 0], [0, 1, 0]),
    'tfm_only': ([0, 0, 1], [0, 0, 1]),
    'neural_50_50': ([0, .5, .5], [0, .5, .5]),
    'equal_3': ([1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]),
    'cand2_like': ([.7, .3, 0], [0, 1, 0]),
    'cand3b': ([.77, .23, 0], [.08, .60, .32]),
    # candidate 4 (B): fold-averaged weights of the frozen 4-fold x5 CV on main (runs/entry_cand4_B/blend.json), over (m3, size2 GRUs, transformers)
    'candB_main': ([0.5565144069948885, 0.23549875382172186, 0.20798683918338962], [1.713394446426447e-05, 0.5647747750721468, 0.43520809098338903]),
    'neural_win_m3_small': ([.3, .35, .35], [0, .5, .5]),
}


def split(p):
    pw = 1 - p[..., 0]
    return pw, p[..., 1:] / np.maximum(pw[..., None], 1e-9)


def blend(preds, win_w, size_w):
    S = [split(p) for p in preds]
    pw = sum(w * s[0] for w, s in zip(win_w, S))
    mix = sum(w * s[1] for w, s in zip(size_w, S))
    return np.concatenate([(1 - pw)[..., None], pw[..., None] * mix], -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round', default='adaptation')
    ap.add_argument('--data', required=True)
    ap.add_argument('--gru', required=True, help='folder with member_* (seqens export model dir)')
    ap.add_argument('--tfm', required=True, help='folder with member_* (tfmens export model dir)')
    ap.add_argument('--gru-id', default='seqens/r2gru')
    ap.add_argument('--tfm-id', default='tfmens/r2tfm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--default', default='candB_main')
    ap.add_argument('--select', action='store_true', help='pick the best preset on the new validation instead of --default')
    a = ap.parse_args(argv)

    from tdx.arms.m3flip import build as build_m3
    from tdx.arms.seqens.arm import SeqEnsArm
    from tdx.arms.tfmens.arm import TfmEnsArm

    root = data_root(a.data, a.round)
    rnd = load_round(root, a.round, answers=False)
    val = rnd.splits['validation']
    m3 = build_m3()
    m3.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    preds = [m3.predict(val), SeqEnsArm.load(a.gru).predict(val), TfmEnsArm.load(a.tfm).predict(val)]
    arm_ids = ['m3flip/m3', a.gru_id, a.tfm_id]
    choice, scores = a.default, {}
    try:
        truth = with_answers(rnd)
        if truth.splits['validation'].target is None:
            raise ValueError('no validation targets')
        for name, (w, s) in PRESETS.items():
            r = score(truth, blend(preds, w, s))
            scores[name] = (r['points'], {k: round(v, 4) for k, v in r['by_condition'].items()})
        if a.select:
            choice = max(scores, key=lambda n: scores[n][0])
        print('%-22s %8s  per-condition loss ratios' % ('preset', 'points'))
        for n, (pts, byc) in sorted(scores.items(), key=lambda kv: -kv[1][0]):
            print('%-22s %8.3f  %s%s' % (n, pts, byc, '   <-- chosen' if n == choice else ''))
    except Exception as exc:  # no targets: keep the main-round preset
        print(f'no preset scoring ({type(exc).__name__}: {exc}); using default preset {choice}')
    w, s = PRESETS[choice]
    cfg = dict(version=1, kind='factored', arms=arm_ids, weights=list(map(float, w)), size_weights=list(map(float, s)))
    Path(a.out).write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'BLEND {choice} -> {a.out}')


if __name__ == '__main__':
    main()
