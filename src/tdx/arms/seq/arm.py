"""Per-machine GRU over all histories (self-supervised next-tick), torch fit, numpy float64 inference."""
import json
from pathlib import Path
import numpy as np
from tdx.arm import Arm


def _masked(split):
    t = np.arange(split.y.shape[-1])[None, None, :]
    return np.where(t < split.cutoff[:, None, None], split.y, -1).astype(np.int64)


def _short_history(y, co, prob, recent=40, older=24):
    """Mask a random share of episodes like short_history: last 40 ticks + 24 sampled older ticks per machine."""
    import torch
    B, M, T = y.shape
    idx = torch.arange(T, device=y.device)[None, None, :]
    c = co[:, None, None]
    is_recent = (idx >= c - recent) & (idx < c)
    is_old = idx < c - recent
    score = torch.where(is_old, torch.rand(B, M, T, device=y.device), torch.full((B, M, T), 2., device=y.device))
    kth = score.kthvalue(older, dim=-1, keepdim=True).values
    keep = is_recent | (is_old & (score <= kth))
    pick = (torch.rand(B, 1, 1, device=y.device) < prob)
    return torch.where(pick & ~keep, torch.full_like(y, -1), y)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_np(w, y, mt, steps, H, msg, factor=False):
    """y [E,8,T] tokens (-1 missing), returns logits [E,8,steps,5]. Mirrors torch GRUCell exactly."""
    E, M, _ = y.shape
    h = np.zeros((E, M, H)); m = np.zeros((E, M, H)); te = w['typ'][mt]; out = []
    wih, whh, bih, bhh = w['wih'], w['whh'], w['bih'], w['bhh']
    for t in range(steps):
        parts = [w['tok'][y[:, :, t] + 1], te] + ([m] if msg else [])
        x = np.concatenate(parts, -1)
        gi = x @ wih.T + bih; gh = h @ whh.T + bhh
        r = _sigmoid(gi[..., :H] + gh[..., :H]); z = _sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        n = np.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
        h = (1 - z) * n + z * h
        m = np.broadcast_to(h.mean(1, keepdims=True), h.shape)
        o = h @ w['hw'].T + w['hb']
        if factor:  # win/no-win logit in column 0, size logits (1,2,5,20) in columns 1:5
            wl = o[..., :1]
            sl = o[..., 1:] - o[..., 1:].max(-1, keepdims=True)
            lsm = sl - np.log(np.exp(sl).sum(-1, keepdims=True))
            o = np.concatenate([-np.logaddexp(0, wl), -np.logaddexp(0, -wl) + lsm], -1)
        out.append(o)
    return np.stack(out, 2)


class SeqArm(Arm):
    def __init__(self, h=128, msg=False, epochs=100, lr=2e-3, wd=1e-2, batch=32, seed=7,
                 transductive=True, w_cut=20., aug=0.0, factor=False, device=None, smoke=False):
        self.p = dict(h=h, msg=bool(msg), epochs=2 if smoke else epochs, lr=lr, wd=wd, batch=batch,
                      seed=seed, transductive=transductive, w_cut=w_cut, aug=aug, factor=bool(factor))
        self.device = device; self.smoke = smoke; self.w = None

    def fit(self, rnd, spec):
        import torch, torch.nn as nn, torch.nn.functional as F
        p = self.p
        dev = self.device if self.device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(p['seed']); np.random.seed(p['seed'])
        Y, MT, CO, LAB, SOFT = [], [], [], [], []
        for name in spec.tick_splits:
            s = rnd.splits[name]
            Y.append(_masked(s)); MT.append(s.mtype.astype(np.int64)); CO.append(s.cutoff.astype(np.int64))
            n = len(s.ids)
            lab = np.full((n, 8, 5), np.nan)
            if name in spec.soft_splits and s.target is not None:
                lab = s.target.astype(np.float64)
            elif name in spec.label_splits and s.label is not None:
                ok = s.label >= 0
                lab = np.where(ok[..., None], np.eye(5)[np.clip(s.label, 0, 4)], np.nan)
            LAB.append(lab)
        Y = torch.tensor(np.concatenate(Y)).to(dev); MT = torch.tensor(np.concatenate(MT)).to(dev)
        CO = torch.tensor(np.concatenate(CO)).to(dev)
        LAB = torch.tensor(np.concatenate(LAB), dtype=torch.float32).to(dev)
        has_cut = bool((~torch.isnan(LAB[..., 0])).any())
        H = p['h']

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.tok = nn.Embedding(6, 8); s.typ = nn.Embedding(4, 4)
                s.cell = nn.GRUCell(12 + (H if p['msg'] else 0), H)
                s.drop = nn.Dropout(0.1); s.head = nn.Linear(H, 5)

            def forward(s, y, mt):
                B, M, T = y.shape
                h = torch.zeros(B, M, H, device=y.device, dtype=s.head.weight.dtype); m = torch.zeros_like(h)
                te = s.typ(mt); out = []
                for t in range(T):
                    x = torch.cat([s.tok(y[:, :, t] + 1), te] + ([m] if p['msg'] else []), -1)
                    h = s.cell(x.reshape(B * M, -1), h.reshape(B * M, -1)).reshape(B, M, -1)
                    m = h.mean(1, keepdim=True).expand_as(h)
                    o = s.head(s.drop(h))
                    if p['factor']:
                        wl = o[..., :1]
                        o = torch.cat([F.logsigmoid(-wl), F.logsigmoid(wl) + F.log_softmax(o[..., 1:], -1)], -1)
                    out.append(o)
                return torch.stack(out, 2)

        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
        N = Y.shape[0]
        for ep in range(p['epochs']):
            net.train(); perm = torch.randperm(N, device=dev)
            for i in range(0, N, p['batch']):
                b = perm[i:i + p['batch']]; y = Y[b]
                if p['aug'] > 0:
                    y = _short_history(y, CO[b], p['aug'])
                lg = net(y, MT[b])
                tgt = y[:, :, 1:]; msk = tgt >= 0
                num = F.cross_entropy(lg[:, :, :-1][msk], tgt[msk], reduction='sum'); den = msk.sum().float()
                if has_cut:
                    idx = (CO[b] - 1).view(-1, 1, 1, 1).expand(-1, 8, 1, 5)
                    cl = F.log_softmax(lg.gather(2, idx).squeeze(2), -1); lab = LAB[b]
                    ok = ~torch.isnan(lab[..., 0])
                    num = num + p['w_cut'] * (-(lab[ok] * cl[ok]).sum(-1)).sum(); den = den + p['w_cut'] * ok.sum()
                loss = num / den
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.double().eval()
        with torch.no_grad():
            ref = net(Y[:3], MT[:3]).cpu().numpy()
        sd = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
        self.w = dict(tok=sd['tok.weight'], typ=sd['typ.weight'], wih=sd['cell.weight_ih'], whh=sd['cell.weight_hh'],
                      bih=sd['cell.bias_ih'], bhh=sd['cell.bias_hh'], hw=sd['head.weight'], hb=sd['head.bias'])
        got = forward_np(self.w, Y[:3].cpu().numpy(), MT[:3].cpu().numpy(), Y.shape[-1], H, p['msg'], p['factor'])
        self.parity = float(np.abs(got - ref).max())
        if self.parity > 1e-9:
            raise RuntimeError(f'numpy/torch parity failed: {self.parity:.3e}')

    def step_outputs(self, y, mtype, steps):
        """Unmasked per-input-tick outputs from the float64 NumPy forward pass."""
        return np.asarray(forward_np(self.w, y, mtype, steps, self.p['h'],
                                     self.p['msg'], self.p.get('factor', False)), dtype=np.float64)

    def predict(self, split):
        y = _masked(split); co = split.cutoff.astype(np.int64)
        lg = forward_np(self.w, y, split.mtype.astype(np.int64), int(co.max()), self.p['h'], self.p['msg'], self.p.get('factor', False))
        sel = lg[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        e = np.exp(sel - sel.max(-1, keepdims=True)); pr = (e / e.sum(-1, keepdims=True)).clip(1e-6, None)
        return pr / pr.sum(-1, keepdims=True)

    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        (d / 'meta.json').write_text(json.dumps(self.p, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory):
        d = Path(directory); p = json.loads((d / 'meta.json').read_text())
        a = cls(**p)
        with np.load(d / 'model.npz') as z:
            a.w = {k: z[k] for k in z.files}
        return a
