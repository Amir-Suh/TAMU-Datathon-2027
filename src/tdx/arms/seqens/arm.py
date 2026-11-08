"""Equal-weight probability average of trained `seq` GRUs, numpy float64 inference only (no torch).

Members are exported `seq` model folders (model.npz + meta.json) copied under member_NN/. Training
happens in the `seq` arm; this arm only packages and averages them. The numpy forward pass duplicates
tdx.arms.seq.arm.forward_np on purpose so the shipped import graph never touches torch; a test asserts
the two give identical predictions.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from tdx.arm import Arm


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _forward(w, y, mt, steps, H, msg, factor):
    E, M, _ = y.shape
    h = np.zeros((E, M, H)); m = np.zeros((E, M, H)); te = w['typ'][mt]; out = []
    for t in range(steps):
        x = np.concatenate([w['tok'][y[:, :, t] + 1], te] + ([m] if msg else []), -1)
        gi = x @ w['wih'].T + w['bih']; gh = h @ w['whh'].T + w['bhh']
        r = _sigmoid(gi[..., :H] + gh[..., :H]); z = _sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        n = np.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
        h = (1 - z) * n + z * h
        m = np.broadcast_to(h.mean(1, keepdims=True), h.shape)
        o = h @ w['hw'].T + w['hb']
        if factor:
            wl = o[..., :1]
            sl = o[..., 1:] - o[..., 1:].max(-1, keepdims=True)
            lsm = sl - np.log(np.exp(sl).sum(-1, keepdims=True))
            o = np.concatenate([-np.logaddexp(0, wl), -np.logaddexp(0, -wl) + lsm], -1)
        out.append(o)
    return np.stack(out, 2)


class _Member:
    def __init__(self, directory):
        d = Path(directory)
        self.p = json.loads((d / 'meta.json').read_text())
        with np.load(d / 'model.npz') as z:
            self.w = {k: z[k] for k in z.files}

    def predict(self, split):
        t = np.arange(split.y.shape[-1])[None, None, :]
        y = np.where(t < split.cutoff[:, None, None], split.y, -1).astype(np.int64)
        co = np.asarray(split.cutoff, dtype=np.int64)
        lg = _forward(self.w, y, np.asarray(split.mtype, dtype=np.int64), int(co.max()),
                      self.p['h'], self.p['msg'], self.p.get('factor', False))
        sel = lg[np.arange(len(co))[:, None], np.arange(8)[None, :], (co - 1)[:, None]]
        e = np.exp(sel - sel.max(-1, keepdims=True))
        pr = (e / e.sum(-1, keepdims=True)).clip(1e-6, None)
        return pr / pr.sum(-1, keepdims=True)


class SeqEnsArm(Arm):
    def __init__(self):
        self.members = []

    def fit(self, rnd, spec):
        raise NotImplementedError('train members with the seq arm, then export with tdx.tools.make_ensemble')

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


def assemble(member_dirs, out_model_dir):
    """Copy exported seq model folders into out_model_dir/member_NN (used by tdx.tools.make_ensemble)."""
    out = Path(out_model_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(member_dirs):
        dst = out / f'member_{i:02d}'
        shutil.copytree(src, dst)
