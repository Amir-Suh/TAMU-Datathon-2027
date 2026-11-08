"""Refit a model on histories truncated by `hold` ticks and save its forecasts for the held-out tail ticks.
usage: tail_preds.py NAME [SEED]   NAME in gru_fac, gru_plain, gbdt, m3, pooled_recent"""
import sys
import numpy as np
from dataclasses import replace
from tdx.backtest import truncate, make_pseudo_split
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round, arm_module, build_arm

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 30
WIN = 30   # always score 30 ticks: [cutoff-HOLD, cutoff-HOLD+30)
rnd = load_round('data/main', answers=False)
rt, full = truncate(rnd, HOLD)

if name == 'gru_fac':
    from tdx.arms.seq import build
    arm = build(h=128, msg=False, epochs=100, transductive=True, factor=True, seed=seed)
    spec = dev_spec(True)
elif name == 'gru_plain':
    from tdx.arms.seq import build
    arm = build(h=128, msg=False, epochs=100, transductive=True, factor=False, seed=seed)
    spec = dev_spec(True)
elif name == 'tfm':
    module = arm_module('tfm2')
    params = dict(module.VARIANTS['conv9_alibi']); params.update(d=128, L=3, epochs=100, seed=seed)
    arm = build_arm(module, params)
    spec = dev_spec(True)
elif name == 'gbdt':
    module = arm_module('gbdt')
    params = dict(module.VARIANTS['pair_l31'])
    params['threads'] = 6
    arm = build_arm(module, params)
    spec = dev_spec(True)
elif name == 'm3':
    module = arm_module('m3flip')
    arm = build_arm(module, {})
    spec = dev_spec(False)
elif name == 'pooled_recent':
    module = arm_module('pooled')
    arm = build_arm(module, {'method': 'pooled_recent'})
    spec = dev_spec(False)
else:
    raise SystemExit('unknown model')

# honest train labels for the earlier (truncated) cutoff: the realized outcome at that tick
tr_full, tr = full['train'], rt.splits['train']
E = len(tr.ids)
lab = tr_full.y[np.arange(E)[:, None], np.arange(8)[None, :], tr.cutoff[:, None]].astype(np.int8)
splits = dict(rt.splits)
splits['train'] = replace(tr, label=lab, condition=tr_full.condition)
fitted = fit_round(replace(rt, splits=splits), spec)
arm.fit(fitted, spec)

val_t, val_f = rt.splits['validation'], full['validation']
Ev = len(val_t.ids)
preds, outs = [], []
for j in range(WIN):
    sj = make_pseudo_split(val_t, val_f, j)
    preds.append(arm.predict(sj))
    tick = val_t.cutoff + j
    outs.append(val_f.y[np.arange(Ev)[:, None], np.arange(8)[None, :], tick[:, None]])
tag = f'{name}_{seed}' if name.startswith(('gru','tfm')) else name
tag = tag + (f'_h{HOLD}' if HOLD != 30 else '')
np.savez(f'/tmp/claude-1000/aug/tail_{tag}.npz', preds=np.array(preds), outs=np.array(outs))
print('saved', tag)
