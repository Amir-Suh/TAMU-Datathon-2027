"""Causal Transformer arm with ALiBi relative bias and causal 1D convolution."""
import json
import math
from pathlib import Path
import numpy as np
import scipy.special
from tdx.arm import Arm


def _masked(split):
    t = np.arange(split.y.shape[-1])[None, None, :]
    return np.where(t < split.cutoff[:, None, None], split.y, -1).astype(np.int64)


def forward_np(w, y, mt, steps, d, L, heads, conv_k=0, alibi=False):
    """y [E,8,T] tokens (-1 missing), returns logits [E,8,steps,5]. Exact float64 parity with torch."""
    E, M, _ = y.shape
    BM = E * M
    head_dim = d // heads
    scale = 1.0 / math.sqrt(head_dim)
    y_sub = y[:, :, :steps]

    tok_w = w['tok.weight']
    typ_w = w['typ.weight']
    tok_emb = tok_w[y_sub + 1]

    if conv_k > 0:
        dw_w = w['conv_dw.weight'][:, 0, :]
        dw_b = w['conv_dw.bias']
        pw_w = w['conv_pw.weight'][:, :, 0]
        pw_b = w['conv_pw.bias']
        tok_flat = tok_emb.reshape(BM, steps, d)
        tok_pad = np.pad(tok_flat, ((0, 0), (conv_k - 1, 0), (0, 0)), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(tok_pad, conv_k, axis=1)
        dw_out = (windows * dw_w).sum(axis=-1) + dw_b
        conv_out = dw_out @ pw_w.T + pw_b
        tok_emb = tok_emb + conv_out.reshape(E, M, steps, d)

    typ_emb = typ_w[mt][:, :, None, :]
    if not alibi:
        pos_w = w['pos.weight'][:steps]
        x = tok_emb + pos_w[None, None, :, :] + typ_emb
    else:
        x = tok_emb + typ_emb

    x = x.reshape(BM, steps, d)

    mask = np.triu(np.full((steps, steps), -np.inf), 1)
    if alibi:
        slopes = np.array([2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)], dtype=np.float64)[:, None, None]
        dist = np.arange(steps)[None, :] - np.arange(steps)[:, None]
        attn_mask = slopes * dist + mask
    else:
        attn_mask = mask

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
        scores = (q @ np.ascontiguousarray(k.swapaxes(-2, -1))) * scale + attn_mask
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
    return o


class Tfm2Arm(Arm):
    def __init__(self, d=64, L=2, heads=4, nhead=None, epochs=60, lr=2e-3, wd=1e-2, batch=32, seed=7,
                 transductive=True, w_cut=20., conv=0, alibi=False, warmup=True, dropout=0.1, size_w=1.0,
                 device=None, smoke=False, **kwargs):
        actual_heads = nhead if nhead is not None else heads
        self.p = dict(d=d, L=L, heads=actual_heads, epochs=2 if smoke else epochs, lr=lr, wd=wd, batch=batch,
                      seed=seed, transductive=transductive, w_cut=w_cut, conv=int(conv),
                      alibi=bool(alibi), warmup=bool(warmup), dropout=float(dropout), size_w=float(size_w))
        self.history = []
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

        class Block(nn.Module):
            def __init__(self, d_model, n_heads, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.ln1 = nn.LayerNorm(d_model)
                self.in_proj = nn.Linear(d_model, 3 * d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.ln2 = nn.LayerNorm(d_model)
                self.fc1 = nn.Linear(d_model, 4 * d_model)
                self.fc2 = nn.Linear(4 * d_model, d_model)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, attn_mask):
                BM, T, _ = x.shape
                # Pre-LN Self-Attention
                xn = self.ln1(x)
                qkv = self.in_proj(xn)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(BM, T, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(BM, T, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(BM, T, self.n_heads, self.head_dim).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(BM, T, self.d_model)
                x = x + self.drop(self.out_proj(attn_out))

                # Pre-LN MLP
                xn = self.ln2(x)
                mlp_out = self.fc2(self.drop(F.gelu(self.fc1(xn))))
                x = x + self.drop(mlp_out)
                return x

        class Net(nn.Module):
            def __init__(self, d_model=64, L_layers=2, n_heads=4, conv_k=0, alibi=False, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.conv_k = conv_k
                self.alibi = alibi
                self.n_heads = n_heads

                self.tok = nn.Embedding(6, d_model)
                self.typ = nn.Embedding(4, d_model)
                if not alibi:
                    self.pos = nn.Embedding(256, d_model)
                else:
                    self.pos = None

                if conv_k > 0:
                    self.conv_dw = nn.Conv1d(d_model, d_model, kernel_size=conv_k, groups=d_model)
                    self.conv_pw = nn.Conv1d(d_model, d_model, kernel_size=1)
                else:
                    self.conv_dw = None
                    self.conv_pw = None

                self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(L_layers)])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, 5)

                if alibi:
                    slopes = [2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)]
                    self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32).view(1, n_heads, 1, 1))
                else:
                    self.slopes = None

            def forward(self, y, mt):
                B, M, T = y.shape
                tok_emb = self.tok(y + 1)

                if self.conv_k > 0:
                    tok_flat = tok_emb.view(B * M, T, self.d_model).transpose(1, 2)
                    padded = F.pad(tok_flat, (self.conv_k - 1, 0))
                    conv_out = self.conv_pw(self.conv_dw(padded)).transpose(1, 2).view(B, M, T, self.d_model)
                    tok_emb = tok_emb + conv_out

                typ_emb = self.typ(mt)[:, :, None, :]
                if not self.alibi:
                    pos_emb = self.pos(torch.arange(T, device=y.device))[None, None, :, :]
                    x = tok_emb + pos_emb + typ_emb
                else:
                    x = tok_emb + typ_emb

                x = x.view(B * M, T, self.d_model)

                causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=y.device, dtype=tok_emb.dtype), diagonal=1)
                if self.alibi:
                    q_idx = torch.arange(T, device=y.device, dtype=tok_emb.dtype).view(1, 1, T, 1)
                    k_idx = torch.arange(T, device=y.device, dtype=tok_emb.dtype).view(1, 1, 1, T)
                    dist = k_idx - q_idx
                    alibi_bias = self.slopes * dist
                    attn_mask = alibi_bias + causal_mask.view(1, 1, T, T)
                else:
                    attn_mask = causal_mask.view(1, 1, T, T)

                for block in self.blocks:
                    x = block(x, attn_mask=attn_mask)

                xn = self.ln_f(x)
                o = self.head(xn).view(B, M, T, 5)
                return o

        net = Net(d_model=d_model, L_layers=L_layers, n_heads=n_heads,
                  conv_k=p['conv'], alibi=p['alibi'], dropout=p['dropout']).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['wd'])

        N = Y.shape[0]
        steps_per_epoch = max(1, (N + p['batch'] - 1) // p['batch'])
        total_steps = p['epochs'] * steps_per_epoch
        warmup_steps = int(0.10 * total_steps) if p['warmup'] else 0
        step = 0

        for ep in range(p['epochs']):
            net.train()
            perm = torch.randperm(N, device=dev)
            ep_sum, ep_n = 0.0, 0
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
                if p.get('size_w', 1.0) == 1.0:
                    num = F.cross_entropy(lg[:, :, :-1][msk], tgt[msk], reduction='sum')
                else:
                    # win BCE + size_w * size CE on wins, computed from the 5-way log-probabilities
                    lp = F.log_softmax(lg[:, :, :-1][msk], -1); tg_ = tgt[msk]
                    lp_y = lp.gather(-1, tg_.unsqueeze(-1)).squeeze(-1)
                    lse_w = torch.logsumexp(lp[:, 1:], -1)
                    num = torch.where(tg_ > 0, -lse_w - p['size_w'] * (lp_y - lse_w), -lp_y).sum()
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
                ep_sum += float(loss.detach()); ep_n += 1
            self.history.append(ep_sum / max(ep_n, 1))

        # Parity check on 3 episodes in float64
        net.double().eval()
        with torch.no_grad():
            ref = net(Y[:3], MT[:3]).cpu().numpy()
        self.w = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
        got = forward_np(self.w, Y[:3].cpu().numpy(), MT[:3].cpu().numpy(), int(Y.shape[-1]),
                         d_model, L_layers, n_heads, p['conv'], p['alibi'])
        self.parity = float(np.abs(got - ref).max())
        print(f"tfm2 parity max abs difference: {self.parity:.3e}")
        if self.parity > 1e-6:
            raise RuntimeError(f'numpy/torch parity failed: {self.parity:.3e}')

    def step_outputs(self, y, mtype, steps):
        """Unmasked per-input-tick logits from the float64 NumPy forward pass."""
        return np.asarray(forward_np(self.w, y, mtype, steps, self.p['d'], self.p['L'],
                                     self.p['heads'], self.p['conv'], self.p['alibi']), dtype=np.float64)

    def predict(self, split):
        y = _masked(split)
        co = split.cutoff.astype(np.int64)
        lg = forward_np(self.w, y, split.mtype.astype(np.int64), int(co.max()),
                        self.p['d'], self.p['L'], self.p['heads'],
                        self.p['conv'], self.p['alibi'])
        sel = lg[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        e = np.exp(sel - sel.max(-1, keepdims=True))
        pr = (e / e.sum(-1, keepdims=True)).clip(1e-6, None)
        return pr / pr.sum(-1, keepdims=True)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / 'model.npz', **self.w)
        (d / 'meta.json').write_text(json.dumps(self.p, indent=2, sort_keys=True))
        (d / 'history.json').write_text(json.dumps({'train_loss_per_epoch': getattr(self, 'history', [])}))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        p = json.loads((d / 'meta.json').read_text())
        a = cls(**p)
        with np.load(d / 'model.npz') as z:
            a.w = {k: z[k] for k in z.files}
        return a
