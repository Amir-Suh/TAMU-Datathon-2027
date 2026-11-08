"""Causal Transformer over all histories (self-supervised next-tick), torch fit, numpy float64 inference."""
import json
import math
from pathlib import Path
import numpy as np
import scipy.special
from tdx.arm import Arm


def _masked(split):
    t = np.arange(split.y.shape[-1])[None, None, :]
    return np.where(t < split.cutoff[:, None, None], split.y, -1).astype(np.int64)


def forward_np(w, y, mt, steps, d, L, heads, factor=False):
    """y [E,8,T] tokens (-1 missing), returns logits [E,8,steps,5]. Exact float64 parity with torch."""
    E, M, _ = y.shape
    BM = E * M
    head_dim = d // heads
    scale = 1.0 / math.sqrt(head_dim)
    y_sub = y[:, :, :steps]

    tok_w = w['tok.weight']
    pos_w = w['pos.weight'][:steps]
    typ_w = w['typ.weight']
    x = tok_w[y_sub + 1] + pos_w[None, None, :, :] + typ_w[mt][:, :, None, :]
    x = x.reshape(BM, steps, d)

    mask = np.triu(np.full((steps, steps), -np.inf), 1)

    for i in range(L):
        # LayerNorm 1
        m1 = x.mean(-1, keepdims=True); v1 = x.var(-1, keepdims=True)
        xn = (x - m1) / np.sqrt(v1 + 1e-5) * w[f'blocks.{i}.ln1.weight'] + w[f'blocks.{i}.ln1.bias']

        # QKV linear projection
        qkv = xn.reshape(-1, d) @ w[f'blocks.{i}.in_proj.weight'].T + w[f'blocks.{i}.in_proj.bias']
        q = qkv[:, :d].reshape(BM, steps, heads, head_dim).swapaxes(1, 2)
        k = qkv[:, d:2 * d].reshape(BM, steps, heads, head_dim).swapaxes(1, 2)
        v = qkv[:, 2 * d:].reshape(BM, steps, heads, head_dim).swapaxes(1, 2)

        # Causal Self-Attention
        scores = (q @ np.ascontiguousarray(k.swapaxes(-2, -1))) * scale
        scores += mask
        scores -= scores.max(-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(-1, keepdims=True)
        attn_out = (scores @ v).swapaxes(1, 2).reshape(BM * steps, d)
        attn_proj = attn_out @ w[f'blocks.{i}.out_proj.weight'].T + w[f'blocks.{i}.out_proj.bias']
        x = x + attn_proj.reshape(BM, steps, d)

        # LayerNorm 2
        m2 = x.mean(-1, keepdims=True); v2 = x.var(-1, keepdims=True)
        xn = (x - m2) / np.sqrt(v2 + 1e-5) * w[f'blocks.{i}.ln2.weight'] + w[f'blocks.{i}.ln2.bias']

        # MLP with GELU
        h = xn.reshape(-1, d) @ w[f'blocks.{i}.fc1.weight'].T + w[f'blocks.{i}.fc1.bias']
        h = h * 0.5 * (1.0 + scipy.special.erf(h / math.sqrt(2.0)))
        mlp_out = h @ w[f'blocks.{i}.fc2.weight'].T + w[f'blocks.{i}.fc2.bias']
        x = x + mlp_out.reshape(BM, steps, d)

    # Final LayerNorm
    mf = x.mean(-1, keepdims=True); vf = x.var(-1, keepdims=True)
    xn = (x - mf) / np.sqrt(vf + 1e-5) * w['ln_f.weight'] + w['ln_f.bias']

    # Output head
    o = (xn.reshape(-1, d) @ w['head.weight'].T + w['head.bias']).reshape(E, M, steps, 5)

    if factor:
        wl = o[..., :1]
        sl = o[..., 1:] - o[..., 1:].max(-1, keepdims=True)
        lsm = sl - np.log(np.exp(sl).sum(-1, keepdims=True))
        o = np.concatenate([-np.logaddexp(0, wl), -np.logaddexp(0, -wl) + lsm], -1)

    return o


class TfmArm(Arm):
    def __init__(self, d=64, L=2, heads=4, nhead=None, epochs=100, lr=1e-3, wd=1e-2, batch=32, seed=7,
                 transductive=True, w_cut=20., factor=False, device=None, smoke=False):
        actual_heads = nhead if nhead is not None else heads
        self.p = dict(d=d, L=L, heads=actual_heads, epochs=2 if smoke else epochs, lr=lr, wd=wd, batch=batch,
                      seed=seed, transductive=transductive, w_cut=w_cut, factor=bool(factor))
        self.device = device
        self.smoke = smoke
        self.w = None
        self.parity = None

    def fit(self, rnd, spec):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        p = self.p
        dev = self.device if self.device in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(p['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(p['seed'])
        np.random.seed(p['seed'])

        Y, MT, CO, LAB = [], [], [], []
        for name in spec.tick_splits:
            s = rnd.splits[name]
            Y.append(_masked(s))
            MT.append(s.mtype.astype(np.int64))
            CO.append(s.cutoff.astype(np.int64))
            n = len(s.ids)
            lab = np.full((n, 8, 5), np.nan)
            if name in spec.soft_splits and s.target is not None:
                lab = s.target.astype(np.float64)
            elif name in spec.label_splits and s.label is not None:
                ok = s.label >= 0
                lab = np.where(ok[..., None], np.eye(5)[np.clip(s.label, 0, 4)], np.nan)
            LAB.append(lab)

        Y = torch.tensor(np.concatenate(Y), device=dev)
        MT = torch.tensor(np.concatenate(MT), device=dev)
        CO = torch.tensor(np.concatenate(CO), device=dev)
        LAB = torch.tensor(np.concatenate(LAB), dtype=torch.float32, device=dev)
        has_cut = bool((~torch.isnan(LAB[..., 0])).any())

        d_model = p['d']
        L_layers = p['L']
        n_heads = p['heads']
        assert d_model % n_heads == 0, f"d ({d_model}) must be divisible by heads ({n_heads})"
        head_dim = d_model // n_heads

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.in_proj = nn.Linear(d_model, 3 * d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.ln2 = nn.LayerNorm(d_model)
                self.fc1 = nn.Linear(d_model, 4 * d_model)
                self.fc2 = nn.Linear(4 * d_model, d_model)
                self.drop = nn.Dropout(0.1)

            def forward(self, x):
                BM, T, _ = x.shape
                # Pre-LN Self-Attention
                xn = self.ln1(x)
                qkv = self.in_proj(xn)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(BM, T, n_heads, head_dim).transpose(1, 2)
                k = k.view(BM, T, n_heads, head_dim).transpose(1, 2)
                v = v.view(BM, T, n_heads, head_dim).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(BM, T, d_model)
                x = x + self.drop(self.out_proj(attn_out))

                # Pre-LN MLP
                xn = self.ln2(x)
                mlp_out = self.fc2(self.drop(F.gelu(self.fc1(xn))))
                x = x + self.drop(mlp_out)
                return x

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = nn.Embedding(6, d_model)
                self.pos = nn.Embedding(256, d_model)
                self.typ = nn.Embedding(4, d_model)
                self.blocks = nn.ModuleList([Block() for _ in range(L_layers)])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, 5)

            def forward(self, y, mt):
                B, M, T = y.shape
                tok_emb = self.tok(y + 1)
                pos_emb = self.pos(torch.arange(T, device=y.device))[None, None, :, :]
                typ_emb = self.typ(mt)[:, :, None, :]
                x = tok_emb + pos_emb + typ_emb
                x = x.view(B * M, T, d_model)
                for block in self.blocks:
                    x = block(x)
                xn = self.ln_f(x)
                o = self.head(xn).view(B, M, T, 5)
                if p['factor']:
                    wl = o[..., :1]
                    o = torch.cat([F.logsigmoid(-wl), F.logsigmoid(wl) + F.log_softmax(o[..., 1:], -1)], -1)
                return o

        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
        N = Y.shape[0]
        steps_per_epoch = max(1, (N + p['batch'] - 1) // p['batch'])
        total_steps = p['epochs'] * steps_per_epoch
        warmup_steps = int(0.05 * total_steps)
        step = 0

        for ep in range(p['epochs']):
            net.train()
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, p['batch']):
                if warmup_steps > 0 and step < warmup_steps:
                    lr_scale = float(step + 1) / float(warmup_steps)
                else:
                    progress = float(step + 1 - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in opt.param_groups:
                    pg['lr'] = p['lr'] * lr_scale
                step += 1

                b = perm[i:i + p['batch']]
                y = Y[b]
                lg = net(y, MT[b])
                tgt = y[:, :, 1:]
                msk = tgt >= 0
                num = F.cross_entropy(lg[:, :, :-1][msk], tgt[msk], reduction='sum')
                den = msk.sum().float()
                if has_cut:
                    idx = (CO[b] - 1).view(-1, 1, 1, 1).expand(-1, 8, 1, 5)
                    cl = F.log_softmax(lg.gather(2, idx).squeeze(2), -1)
                    lab = LAB[b]
                    ok = ~torch.isnan(lab[..., 0])
                    num = num + p['w_cut'] * (-(lab[ok] * cl[ok]).sum(-1)).sum()
                    den = den + p['w_cut'] * ok.sum()
                loss = num / den
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        # Parity check on 3 episodes in float64
        net.double().eval()
        with torch.no_grad():
            ref = net(Y[:3], MT[:3]).cpu().numpy()
        self.w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        got = forward_np(self.w, Y[:3].cpu().numpy(), MT[:3].cpu().numpy(), Y.shape[-1],
                         d_model, L_layers, n_heads, p['factor'])
        self.parity = float(np.abs(got - ref).max())
        if self.parity > 1e-6:
            raise RuntimeError(f'numpy/torch parity failed: {self.parity:.3e}')

    def predict(self, split):
        y = _masked(split)
        co = split.cutoff.astype(np.int64)
        lg = forward_np(self.w, y, split.mtype.astype(np.int64), int(co.max()),
                        self.p['d'], self.p['L'], self.p['heads'], self.p.get('factor', False))
        sel = lg[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        e = np.exp(sel - sel.max(-1, keepdims=True))
        pr = (e / e.sum(-1, keepdims=True)).clip(1e-6, None)
        return pr / pr.sum(-1, keepdims=True)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        (d / 'meta.json').write_text(json.dumps(self.p, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        p = json.loads((d / 'meta.json').read_text())
        a = cls(**p)
        with np.load(d / 'model.npz') as z:
            a.w = {k: z[k] for k in z.files}
        return a
