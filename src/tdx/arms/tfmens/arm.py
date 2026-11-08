"""Equal-weight probability average of trained `tfm2` transformers, numpy float64 inference only (no torch).

Members are exported tfm2 model folders (model.npz + meta.json) copied under member_NN/. Training happens in the
tfm2 arm; this arm only averages them. The forward pass below is a verbatim copy of tdx.arms.tfm2.arm.forward_np
so the shipped import graph never touches torch (tfm2.arm imports torch inside fit, which the package check
forbids); a test asserts identical predictions.
"""
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


class _Member:
    def __init__(self, directory):
        d = Path(directory)
        self.p = json.loads((d / 'meta.json').read_text())
        with np.load(d / 'model.npz') as z:
            self.w = {k: z[k] for k in z.files}

    CHUNK = 12   # episodes per forward pass: bounds attention memory (all 1,536 sequences at once peaked near 16 GB)

    def predict(self, split):
        y = _masked(split)
        co = split.cutoff.astype(np.int64)
        mt = split.mtype.astype(np.int64)
        out = np.empty((len(co), 8, 5))
        for a in range(0, len(co), self.CHUNK):
            b = min(a + self.CHUNK, len(co))
            lg = forward_np(self.w, y[a:b], mt[a:b], int(co[a:b].max()),
                            self.p['d'], self.p['L'], self.p['heads'], self.p['conv'], self.p['alibi'])
            sel = lg[np.arange(b - a)[:, None], np.arange(8)[None, :], (co[a:b] - 1)[:, None]]
            e = np.exp(sel - sel.max(-1, keepdims=True))
            pr = (e / e.sum(-1, keepdims=True)).clip(1e-6, None)
            out[a:b] = pr / pr.sum(-1, keepdims=True)
        return out


class TfmEnsArm(Arm):
    def __init__(self):
        self.members = []

    def fit(self, rnd, spec):
        raise NotImplementedError('train members with the tfm2 arm, then export with tdx.tools.make_ensemble --arm tfmens')

    def predict(self, split):
        if not self.members:
            raise RuntimeError('no members loaded')
        return np.mean([m.predict(split) for m in self.members], axis=0)

    def save(self, directory):
        raise NotImplementedError('use tdx.tools.make_ensemble to assemble an export')

    @classmethod
    def load(cls, directory):
        arm = cls()
        dirs = sorted(p for p in Path(directory).glob('member_*') if p.is_dir())
        if not dirs:
            raise ValueError(f'no member_* folders in {directory}')
        arm.members = [_Member(p) for p in dirs]
        return arm
