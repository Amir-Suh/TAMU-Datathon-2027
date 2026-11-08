import glob
import numpy as np

A = '/tmp/claude-1000/aug/'
def load(tag):
    z = np.load(A + f'tail_{tag}.npz'); return z['preds'], z['outs']

models = {}
outs = None
for tag in ('pooled_recent', 'm3', 'gbdt'):
    p, o = load(tag); models[tag] = p; outs = o
gru = [load(f'gru_fac_{s}')[0] for s in (1, 2, 3) if glob.glob(A + f'tail_gru_fac_{s}.npz')]
if gru:
    models['gru_fac (mean of %d)' % len(gru)] = np.mean(gru, 0)
    models['gru_fac single seeds avg'] = None
    models['gru + m3 (50/50)'] = (np.mean(gru, 0) + models['m3']) / 2
    models['gru + m3 + gbdt (equal)'] = (np.mean(gru, 0) + models['m3'] + models['gbdt']) / 3

J, E, M = outs.shape
valid = outs >= 0
o = np.clip(outs, 0, 4)
win = valid & (outs > 0)

def parts(p):
    p = np.clip(p, 1e-6, 1.0)
    p0 = p[..., 0]
    pw = 1 - p0
    po = np.take_along_axis(p, o[..., None], -1)[..., 0]
    wl = np.where(outs == 0, -np.log(np.clip(p0, 1e-6, 1)), -np.log(np.clip(pw, 1e-6, 1)))
    sl = -np.log(np.clip(po / np.clip(pw, 1e-6, 1), 1e-6, 1))
    tl = -np.log(po)
    return wl, sl, tl

def row(name, p):
    wl, sl, tl = parts(p)
    ep = np.broadcast_to(np.arange(E)[None, :, None], outs.shape)
    return name, wl[valid].mean(), sl[win].mean(), tl[valid].mean(), (wl, sl)

print('Held-out tail (30 ticks), models refit on histories truncated 30 ticks earlier. n outcomes=%d, wins=%d' % (valid.sum(), win.sum()))
print('%-30s %10s %12s %10s' % ('model', 'win part', 'size part', 'total'))
base = None
res = {}
for name, p in models.items():
    if p is None:
        continue
    r = row(name, p); res[name] = r
    print('%-30s %10.4f %12.4f %10.4f' % r[:4])
if gru:
    singles = [row('g', g) for g in gru]
    print('%-30s %10.4f %12.4f %10.4f' % ('gru_fac single seed (avg of %d)' % len(gru), *[np.mean([s[i] for s in singles]) for i in (1, 2, 3)]))

# paired bootstrap over episodes for size-part differences (per-win averages)
rng = np.random.RandomState(0)
def size_by_ep(name):
    wl, sl, _ = parts(models[name])
    num = (sl * win).sum(axis=(0, 2)); den = win.sum(axis=(0, 2)); return num, den
def boot(a, b, B=2000):
    na, da = size_by_ep(a); nb, db = size_by_ep(b); d = []
    for _ in range(B):
        i = rng.randint(0, E, E); d.append(na[i].sum() / da[i].sum() - nb[i].sum() / db[i].sum())
    return np.percentile(d, [2.5, 97.5]).round(4), round(float(np.mean(d)), 4)
gname = [k for k in models if k.startswith('gru_fac (')][0] if gru else None
if gname:
    for other in ('pooled_recent', 'm3', 'gbdt'):
        print('size part, %s - %s : mean diff %s, 95%% CI %s' % (gname, other, boot(gname, other)[1], boot(gname, other)[0]))
