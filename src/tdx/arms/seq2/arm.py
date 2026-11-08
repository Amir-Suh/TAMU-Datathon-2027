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
    masked_y = torch.where(pick & ~keep, torch.full_like(y, -1), y)
    return masked_y, pick.squeeze(-1).squeeze(-1), keep


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_np(w, y, mt, steps, H, msg, factor=False, short_flag=False):
    """y [E,8,T] tokens (-1 missing), returns logits [E,8,steps,5]. Mirrors torch GRUCell exactly."""
    E, M, _ = y.shape
    h = np.zeros((E, M, H)); m = np.zeros((E, M, H)); te = w['typ'][mt]; out = []
    wih, whh, bih, bhh = w['wih'], w['whh'], w['bih'], w['bhh']
    if short_flag:
        flag = (np.cumsum(y < 0, axis=-1) > 0).astype(np.int64)
    for t in range(steps):
        parts = [w['tok'][y[:, :, t] + 1], te]
        if short_flag:
            parts.append(w['flg'][flag[:, :, t]])
        if msg:
            parts.append(m)
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


class Seq2Arm(Arm):
    def __init__(self, h=128, msg=False, epochs=100, lr=2e-3, wd=1e-2, batch=32, seed=7,
                 transductive=True, w_cut=20., aug=0.0, aug_target='masked', short_flag=False,
                 distill=0.0, tokdrop=0.0, factor=False, device=None, smoke=False):
        self.p = dict(h=h, msg=bool(msg), epochs=2 if smoke else epochs, lr=lr, wd=wd, batch=batch,
                      seed=seed, transductive=transductive, w_cut=w_cut, aug=float(aug),
                      aug_target=str(aug_target), short_flag=bool(short_flag),
                      distill=float(distill), tokdrop=float(tokdrop), factor=bool(factor))
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
                if p['short_flag']:
                    s.flg = nn.Embedding(2, 4)
                in_dim = 12 + (4 if p['short_flag'] else 0) + (H if p['msg'] else 0)
                s.cell = nn.GRUCell(in_dim, H)
                s.drop = nn.Dropout(0.1); s.head = nn.Linear(H, 5)

            def forward(s, y, mt, flag=None):
                B, M, T = y.shape
                h = torch.zeros(B, M, H, device=y.device, dtype=s.head.weight.dtype); m = torch.zeros_like(h)
                te = s.typ(mt); out = []
                if p['short_flag']:
                    # Causal feature: a missing token has occurred in the
                    # prefix ending at this input tick.
                    flag_steps = torch.cumsum((y < 0).long(), dim=-1) > 0
                for t in range(T):
                    parts = [s.tok(y[:, :, t] + 1), te]
                    if p['short_flag']:
                        parts.append(s.flg(flag_steps[:, :, t].long()))
                    if p['msg']:
                        parts.append(m)
                    x = torch.cat(parts, -1)
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
                b = perm[i:i + p['batch']]; orig_y = Y[b]
                if p['aug'] > 0:
                    y_aug, b_aug, keep = _short_history(orig_y, CO[b], p['aug'])
                else:
                    y_aug = orig_y
                    b_aug = torch.zeros(len(b), dtype=torch.bool, device=dev)
                    keep = torch.ones_like(orig_y, dtype=torch.bool)

                flag = None  # Net derives the causal flag from the actual input.

                if p['tokdrop'] > 0:
                    drop_mask = torch.rand_like(y_aug, dtype=torch.float32) < p['tokdrop']
                    y_in = torch.where(drop_mask, torch.full_like(y_aug, -1), y_aug)
                else:
                    y_in = y_aug

                lg = net(y_in, MT[b], flag=flag)
                orig_tgt = orig_y[:, :, 1:]
                if p['aug_target'] == 'masked':
                    msk = (orig_tgt >= 0) & (~b_aug[:, None, None] | keep[:, :, 1:])
                else:
                    msk = orig_tgt >= 0
                tgt = orig_tgt

                num = F.cross_entropy(lg[:, :, :-1][msk], tgt[msk], reduction='sum'); den = msk.sum().float()
                if has_cut:
                    idx = (CO[b] - 1).view(-1, 1, 1, 1).expand(-1, 8, 1, 5)
                    cl = F.log_softmax(lg.gather(2, idx).squeeze(2), -1); lab = LAB[b]
                    ok = ~torch.isnan(lab[..., 0])
                    num = num + p['w_cut'] * (-(lab[ok] * cl[ok]).sum(-1)).sum(); den = den + p['w_cut'] * ok.sum()
                loss = num / den

                if p['distill'] > 0 and b_aug.any():
                    msk_aug = msk[b_aug]
                    if msk_aug.any():
                        net.eval()
                        with torch.no_grad():
                            teacher_lg = net(orig_y[b_aug], MT[b][b_aug])
                        net.train()
                        s_lp = F.log_softmax(lg[b_aug, :, :-1][msk_aug], -1)
                        t_p = F.softmax(teacher_lg[:, :, :-1][msk_aug], -1)
                        t_lp = F.log_softmax(teacher_lg[:, :, :-1][msk_aug], -1)
                        kl = (t_p * (t_lp - s_lp)).sum(-1)
                        loss = loss + p['distill'] * kl.mean()

                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()

        net.double().eval()
        with torch.no_grad():
            ref = net(Y[:3], MT[:3]).cpu().numpy()
        sd = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
        self.w = dict(tok=sd['tok.weight'], typ=sd['typ.weight'], wih=sd['cell.weight_ih'], whh=sd['cell.weight_hh'],
                      bih=sd['cell.bias_ih'], bhh=sd['cell.bias_hh'], hw=sd['head.weight'], hb=sd['head.bias'])
        if p['short_flag']:
            self.w['flg'] = sd['flg.weight']
        got = forward_np(self.w, Y[:3].cpu().numpy(), MT[:3].cpu().numpy(), Y.shape[-1], H, p['msg'], p['factor'], p['short_flag'])
        self.parity = float(np.abs(got - ref).max())
        if self.parity > 1e-9:
            raise RuntimeError(f'numpy/torch parity failed: {self.parity:.3e}')

    def predict(self, split):
        y = _masked(split); co = split.cutoff.astype(np.int64)
        lg = self.step_outputs(y, split.mtype.astype(np.int64), int(co.max()))
        sel = lg[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        e = np.exp(sel - sel.max(-1, keepdims=True)); pr = (e / e.sum(-1, keepdims=True)).clip(1e-6, None)
        return pr / pr.sum(-1, keepdims=True)

    def step_outputs(self, y, mtype, steps):
        """Return unmasked float64 logits for the first ``steps`` ticks."""
        out = forward_np(self.w, np.asarray(y), np.asarray(mtype), int(steps),
                         self.p['h'], self.p['msg'], self.p.get('factor', False),
                         self.p.get('short_flag', False))
        return np.asarray(out, dtype=np.float64)

    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        meta = dict(self.p, flag='causal_v1')
        (d / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory):
        d = Path(directory); p = json.loads((d / 'meta.json').read_text())
        flag = p.pop('flag', None)
        if flag != 'causal_v1':
            raise ValueError("seq2 model was trained with the non-causal flag; retrain before loading")
        a = cls(**p)
        with np.load(d / 'model.npz') as z:
            a.w = {k: z[k] for k in z.files}
        return a


SeqArm = Seq2Arm
