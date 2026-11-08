"""Overfitting diagnostics for the GRU: held-out-tail loss against training epoch, and SVG curve plots.

  python -m tdx.tools.epoch_curves tail --variant ce --seed 41 --out DIR     # refit with the last 30 ticks hidden, checkpoint every 10
                                                                            # epochs, score each checkpoint on held-out tail ticks
  python -m tdx.tools.epoch_curves plot --tail DIR [--curves runs/curves] --out FILE.svg

`tail` uses no validation answers (realized outcomes at hidden history ticks only; the GRU does not use `signal`).
`plot` also draws, if --curves is given, the VALIDATION score of each checkpoint of the full-data runs (runs/curves/<variant>_<seed>/model/ckpt_*.npz),
a diagnostic that uses the validation true probabilities (never used to fit anything).
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path
import numpy as np
from tdx.backtest import truncate, make_pseudo_split
from tdx.data import load_round, with_answers
from tdx.fit import dev_spec
from tdx.run import fit_round
from tdx.score import score

WINDOWS = (0, 6, 12, 18, 24, 29)        # pseudo-cutoffs inside the 30 hidden ticks


def parts(P, outs):
    valid = outs >= 0
    P = np.clip(P, 1e-6, 1)
    oc = np.clip(outs, 0, 4)
    p0 = P[..., 0]
    pw = np.clip(1 - p0, 1e-6, 1)
    po = np.take_along_axis(P, oc[..., None], -1)[..., 0]
    win = outs > 0
    wl = np.where(outs == 0, -np.log(p0), -np.log(pw))[valid].mean()
    sl = (-np.log(np.clip(po / pw, 1e-6, 1)))[win].mean()
    brier = ((P - np.eye(5)[oc]) ** 2).sum(-1)[valid].mean()
    return dict(win=float(wl), size=float(sl), total=float((-np.log(po))[valid].mean()), brier=float(brier))


def tail_run(variant, seed, out, hold=30, data=None):
    from tdx.arms.seqloss import VARIANTS, build
    rnd = load_round(data or 'data/main', answers=False)
    rt, full = truncate(rnd, hold)
    kw = {**VARIANTS[variant], 'seed': seed, 'ckpt_every': int(os.environ.get('CKPT_EVERY', 10))}
    if os.environ.get('EPOCHS'):
        kw['epochs'] = int(os.environ['EPOCHS'])        # smoke tests only
    arm = build(**kw)
    spec = dev_spec(True)
    arm.fit(fit_round(rt, spec), spec)
    vt, vf = rt.splits['validation'], full['validation']
    E = len(vt.ids)
    splits = [make_pseudo_split(vt, vf, j) for j in WINDOWS]
    outs = np.stack([vf.y[np.arange(E)[:, None], np.arange(8)[None, :], (vt.cutoff + j)[:, None]] for j in WINDOWS])
    result = dict(variant=variant, seed=seed, hold=hold, train_loss=arm.history, epochs={})
    for ep, w in sorted(arm.checkpoints.items()):
        arm.w = w
        P = np.stack([arm.predict(s) for s in splits])
        result['epochs'][str(ep)] = parts(P, outs)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f'tail_{variant}_{seed}.json').write_text(json.dumps(result))
    print('saved', Path(out) / f'tail_{variant}_{seed}.json', {e: round(v['total'], 4) for e, v in result['epochs'].items()})


def svg(panels, path, width=420, height=260):
    colors = {'ce': '#1f77b4', 'size2': '#d62728'}
    pad = 46
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * len(panels)}" height="{height + 30}" font-family="sans-serif" font-size="11">',
           '<rect width="100%" height="100%" fill="white"/>']
    for k, (title, series) in enumerate(panels):
        ox = k * width
        xs = [x for s in series.values() for l in s for x, _ in l]
        ys = [y for s in series.values() for l in s for _, y in l]
        if not xs:
            continue
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        y0, y1 = y0 - 0.05 * (y1 - y0 + 1e-9), y1 + 0.05 * (y1 - y0 + 1e-9)
        fx = lambda x: ox + pad + (x - x0) / (x1 - x0 + 1e-9) * (width - pad - 14)
        fy = lambda y: 24 + (1 - (y - y0) / (y1 - y0 + 1e-9)) * (height - 24 - 34)
        out.append(f'<text x="{ox + width / 2}" y="14" text-anchor="middle" font-size="12">{title}</text>')
        out.append(f'<rect x="{ox + pad}" y="24" width="{width - pad - 14}" height="{height - 58}" fill="none" stroke="#888"/>')
        for t in np.linspace(y0, y1, 5):
            out.append(f'<text x="{ox + pad - 4}" y="{fy(t) + 3}" text-anchor="end">{t:.3f}</text>')
        for t in np.linspace(x0, x1, 6):
            out.append(f'<text x="{fx(t)}" y="{height - 14}" text-anchor="middle">{t:.0f}</text>')
        out.append(f'<text x="{ox + width / 2}" y="{height + 6}" text-anchor="middle">epoch</text>')
        for name, lines in series.items():
            for line in lines:
                pts = ' '.join(f'{fx(x):.1f},{fy(y):.1f}' for x, y in line)
                out.append(f'<polyline points="{pts}" fill="none" stroke="{colors.get(name, "#444")}" stroke-width="1.2" opacity="0.65"/>')
        for i, name in enumerate(series):
            out.append(f'<text x="{ox + pad + 8 + 70 * i}" y="38" fill="{colors.get(name, "#444")}">{name}</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out))


def plot(tail_dir, curves, path, data=None):
    train, tail, valid = {}, {}, {}
    for f in sorted(glob.glob(str(Path(tail_dir) / 'tail_*_*.json'))):
        d = json.load(open(f))
        train.setdefault(d['variant'], []).append([(i + 1, v) for i, v in enumerate(d['train_loss'])])
        tail.setdefault(d['variant'], []).append([(int(e), m['total']) for e, m in d['epochs'].items()])
    if curves:
        from tdx.arms.seq.arm import SeqArm
        rnd = with_answers(load_round(data or 'data/main', answers=False))
        v = rnd.splits['validation']
        for d in sorted(glob.glob(str(Path(curves) / '*_*/model'))):
            name = Path(d).parent.name.rsplit('_', 1)[0]
            arm = SeqArm.load(d)
            line = []
            for ck in sorted(glob.glob(d + '/ckpt_*.npz')):
                with np.load(ck) as z:
                    arm.w = {k: z[k] for k in z.files}
                line.append((int(re.search(r'ckpt_(\d+)', ck).group(1)), score(rnd, arm.predict(v))['points']))
            if line:
                valid.setdefault(name, []).append(line)
    panels = [('training loss per epoch (tail-hidden refit)', train), ('held-out tail log-loss (30 hidden ticks)', tail)]
    if valid:
        panels.append(('validation points by epoch (full-data runs)', valid))
        print({n: [round(l[-1][1], 2) for l in ls] for n, ls in valid.items()})
    svg(panels, path)
    print('wrote', path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tail')
    t.add_argument('--variant', default='ce')
    t.add_argument('--seed', type=int, required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--data')
    q = sub.add_parser('plot')
    q.add_argument('--tail', required=True)
    q.add_argument('--curves')
    q.add_argument('--out', required=True)
    q.add_argument('--data')
    a = p.parse_args(argv)
    if a.cmd == 'tail':
        tail_run(a.variant, a.seed, a.out, data=a.data)
    else:
        plot(a.tail, a.curves, a.out, a.data)


if __name__ == '__main__':
    main()
