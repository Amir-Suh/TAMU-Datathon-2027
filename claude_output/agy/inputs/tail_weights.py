import glob, csv, numpy as np
from scipy.optimize import minimize

A = '/tmp/claude-1000/aug/'
def L(tag): return np.load(A + f'tail_{tag}.npz')

holds = [h for h in (30, 60, 90) if glob.glob(A + f'tail_m3{"" if h == 30 else f"_h{h}"}.npz')]
def tg(base, h, s=None):
    n = base if s is None else f'{base}_{s}'
    return n + ('' if h == 30 else f'_h{h}')
P = {'m3': [], 'gru': [], 'tfm': []}; OUT = []
for h in holds:
    P['m3'].append(L(tg('m3', h))['preds'])
    P['gru'].append(np.mean([L(tg('gru_fac', h, s))['preds'] for s in (1, 2, 3)], 0))
    P['tfm'].append(np.mean([L(tg('tfm', h, s))['preds'] for s in (1, 2)], 0))
    OUT.append(L(tg('m3', h))['outs'])
M = {k: np.concatenate(v, 0) for k, v in P.items()}            # [W, E, 8, 5], W = 30 * windows
outs = np.concatenate(OUT, 0); W, E, _ = outs.shape
print('windows (hold):', holds, ' outcomes scored:', int((outs >= 0).sum()))
valid = outs >= 0
onehot = np.eye(5)[np.clip(outs, 0, 4)]
names = ['m3', 'gru', 'tfm']

def split(p):
    pw = 1 - p[..., 0]; return pw, p[..., 1:] / np.maximum(pw[..., None], 1e-9)
S = {k: split(v) for k, v in M.items()}

def build(a, b, use):
    pw = sum(a[i] * S[k][0] for i, k in enumerate(use))
    mix = sum(b[i] * S[k][1] for i, k in enumerate(use))
    return np.concatenate([(1 - pw)[..., None], pw[..., None] * mix], -1)

def brier(p, mask=None):
    m = valid if mask is None else mask
    return ((p - onehot) ** 2).sum(-1)[m].mean()

def logloss(p, mask=None):
    m = valid if mask is None else mask
    q = np.take_along_axis(np.clip(p, 1e-6, 1), np.clip(outs, 0, 4)[..., None], -1)[..., 0]
    return (-np.log(q))[m].mean()

def fit(use, mask):
    n = len(use)
    f = lambda w: brier(build(w[:n], w[n:], use), mask)
    cons = [{'type': 'eq', 'fun': lambda w: w[:n].sum() - 1}, {'type': 'eq', 'fun': lambda w: w[n:].sum() - 1}]
    r = minimize(f, np.full(2 * n, 1 / n), bounds=[(0, 1)] * (2 * n), constraints=cons, method='SLSQP')
    return r.x[:n], r.x[n:]

# 8-fold CV grouped by episode (all windows of an episode stay together)
rng = np.random.RandomState(0); fold = rng.permutation(E) % 8
def cv(use):
    out = np.zeros_like(M['m3'])
    for k in range(8):
        tr = (fold != k)[None, :, None] & valid; te = (fold == k)
        a, b = fit(use, tr)
        out[:, te] = build(a, b, use)[:, te]
    return out

fixed = {
    'm3 alone': M['m3'], 'GRU x3 alone': M['gru'], 'transformer x2 alone': M['tfm'],
    'plain average of 3': (M['m3'] + M['gru'] + M['tfm']) / 3,
    'plain 50/50 m3 + GRU': (M['m3'] + M['gru']) / 2,
    'shipped cand2: win .7 m3 + .3 GRU, size GRU': build([0.7, 0.3], [0.0, 1.0], ['m3', 'gru']),
}
print('%-46s %9s %9s' % ('selection on held-out tail, 8-fold CV by episode', 'Brier', 'log-loss'))
for n, p in fixed.items():
    print('%-46s %9.5f %9.4f' % (n, brier(p), logloss(p)))
res = {}
for use in (['m3', 'gru'], ['m3', 'gru', 'tfm']):
    p = cv(use); res[tuple(use)] = p
    print('%-46s %9.5f %9.4f' % ('fitted split-duty on ' + '+'.join(use) + ' (CV)', brier(p), logloss(p)))
base = brier(M['m3'])
print('Brier relative to m3 alone: shipped %.2f%%, fitted 2-model %.2f%%, fitted 3-model %.2f%%' % tuple(
    100 * (brier(x) / base - 1) for x in (fixed['shipped cand2: win .7 m3 + .3 GRU, size GRU'], res[('m3', 'gru')], res[('m3', 'gru', 'tfm')])))

# full-data weights (for use) and their stability across windows
for use in (['m3', 'gru'], ['m3', 'gru', 'tfm']):
    a, b = fit(use, valid)
    print('weights fitted on all tail data, %s: win %s | size %s' % ('+'.join(use), dict(zip(use, a.round(2))), dict(zip(use, b.round(2)))))
    for wi in range(len(holds)):
        sl = slice(wi * 30, (wi + 1) * 30); msk = np.zeros_like(valid); msk[sl] = valid[sl]
        a2, b2 = fit(use, msk)
        print('   window hold=%d: win %s | size %s' % (holds[wi], a2.round(2), b2.round(2)))
np.save(A + 'tail_weights_3.npy', np.array(fit(['m3', 'gru', 'tfm'], valid)))
np.save(A + 'tail_weights_2.npy', np.array(fit(['m3', 'gru'], valid)))
