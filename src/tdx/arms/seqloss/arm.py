"""`seq` GRU with alternative training losses (inference, save and load are inherited, so exports are plain seq models).

Losses (all on the observed next tick of every history; `factor=True` heads output log-probs [win-or-not, size | win]):
  loss='ce'        cross-entropy (== seq)
  loss='brier'     sum_c (onehot_c - p_c)^2, the contest's squared-error risk up to a constant
  loss='ce_brier'  ce + beta * brier
  size_w=lam       win BCE + lam * size CE on wins (lam=1 is ce); needs factor=True
  recency_tau=tau  weight exp(-(cutoff-1-t)/tau): ticks closer to the cutoff count more (0 = off)
  len_pow=a        machine-row weight (256 / observed ticks)^a: short observed histories count more (0 = off)
  val_pow=a        tick weight payout_value^a with payouts (0,1,2,5,20) -> weight (1,1,2,5,20)^a
Weights multiply and the loss is the weighted mean.
"""
import numpy as np
from ..seq.arm import SeqArm, forward_np, _masked, _short_history  # noqa: F401


class SeqLossArm(SeqArm):
    def __init__(self, loss='ce', beta=2.0, size_w=1.0, recency_tau=0.0, len_pow=0.0, val_pow=0.0, **kw):
        super().__init__(**kw)
        if loss not in ('ce', 'brier', 'ce_brier'):
            raise ValueError(f'unknown loss {loss}')
        if size_w != 1.0 and not self.p['factor']:
            raise ValueError('size_w needs factor=True')
        self.p.update(loss=loss, beta=float(beta), size_w=float(size_w), recency_tau=float(recency_tau),
                      len_pow=float(len_pow), val_pow=float(val_pow))

    def fit(self, rnd, spec):
        import torch, torch.nn as nn, torch.nn.functional as F
        p = self.p
        dev = self.device if self.device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(p['seed']); np.random.seed(p['seed'])
        Y, MT, CO = [], [], []
        for name in spec.tick_splits:
            s = rnd.splits[name]
            Y.append(_masked(s)); MT.append(s.mtype.astype(np.int64)); CO.append(s.cutoff.astype(np.int64))
        Y = torch.tensor(np.concatenate(Y)).to(dev); MT = torch.tensor(np.concatenate(MT)).to(dev)
        CO = torch.tensor(np.concatenate(CO)).to(dev)
        H = p['h']
        value = torch.tensor([1., 1., 2., 5., 20.], device=dev)

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
        N = Y.shape[0]; T = Y.shape[-1]
        tick = torch.arange(T - 1, device=dev).view(1, 1, -1)                 # position of the input tick t
        for ep in range(p['epochs']):
            net.train(); perm = torch.randperm(N, device=dev)
            for i in range(0, N, p['batch']):
                b = perm[i:i + p['batch']]; y = Y[b]
                if p['aug'] > 0:
                    y = _short_history(y, CO[b], p['aug'])
                lg = net(y, MT[b])[:, :, :-1]                                  # [B,8,T-1,5], predicts tick t+1
                tgt = y[:, :, 1:]; msk = tgt >= 0
                tg = tgt.clamp(min=0)
                logp = F.log_softmax(lg, -1)
                lp_y = logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)           # log p(realized class)
                if p['size_w'] != 1.0:
                    lse_w = torch.logsumexp(logp[..., 1:], -1)                 # log(1 - p0)
                    win = tg > 0
                    per = torch.where(win, -lse_w - p['size_w'] * (lp_y - lse_w), -lp_y)
                else:
                    per = -lp_y
                if p['loss'] in ('brier', 'ce_brier'):
                    pr = logp.exp()
                    onehot = F.one_hot(tg, 5).to(pr.dtype)
                    brier = ((onehot - pr) ** 2).sum(-1)
                    per = brier if p['loss'] == 'brier' else per + p['beta'] * brier
                w = torch.ones_like(per)
                if p['recency_tau'] > 0:
                    w = w * torch.exp(-((CO[b] - 1).view(-1, 1, 1) - tick).clamp(min=0).to(per.dtype) / p['recency_tau'])
                if p['len_pow'] > 0:
                    n_obs = (y >= 0).sum(-1).clamp(min=1).to(per.dtype).unsqueeze(-1)
                    w = w * (256.0 / n_obs) ** p['len_pow']
                if p['val_pow'] > 0:
                    w = w * value[tg] ** p['val_pow']
                w = w * msk
                loss = (per * w).sum() / w.sum().clamp(min=1e-9)
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

    def save(self, directory):
        """Write a plain `seq` export (meta.json holds only the keys SeqArm understands) plus loss.json with the loss options."""
        import json
        from pathlib import Path
        extra = ('loss', 'beta', 'size_w', 'recency_tau', 'len_pow', 'val_pow')
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        (d / 'meta.json').write_text(json.dumps({k: v for k, v in self.p.items() if k not in extra}, indent=2, sort_keys=True))
        (d / 'loss.json').write_text(json.dumps({k: self.p[k] for k in extra}, indent=2, sort_keys=True))
