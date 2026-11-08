"""Wrapper around a teammate's Bayesian lag-pair + flip model (vendored in ./amir, see NOTICE.md).

The model infers, per machine, which lag pair and direction its win rule follows and when the rule
flipped, from that machine's own history only; train labels set priors. It fits in about 12 s, so the
entry refits it on the released train split at inference time (RUNTIME_FIT) instead of shipping weights.
"""
import json
from pathlib import Path
import numpy as np
from tdx.arm import Arm
from .amir.common.data import Split as AmirSplit
from .amir.m3_lag_pair_flip import FlipLagPairModel


def to_amir(split, name):
    """Convert a tdx Split ([E,8,T] histories, one cutoff per episode) to the teammate's per-query Split."""
    E, M, T = split.y.shape
    q = E * M
    cutoff = np.repeat(np.asarray(split.cutoff, dtype=np.int64), M)
    X = np.full((q, int(cutoff.max()) + 1), -1, dtype=np.int8)
    width = min(T, X.shape[1])
    X[:, :width] = np.asarray(split.y, dtype=np.int8).reshape(q, T)[:, :width]
    X[np.arange(X.shape[1])[None, :] >= cutoff[:, None]] = -1
    out = AmirSplit(name=name, query_ids=[str(i) for i in np.asarray(split.query_ids).ravel()],
                    episode_ids=np.repeat(np.asarray(split.ids), M),
                    machine_ids=[str(i) for i in range(q)],
                    machine_type=np.asarray(split.mtype, dtype=np.int64).ravel(),
                    signal=np.asarray(split.signal, dtype=np.int64).ravel(), cutoff=cutoff, X=X)
    label = getattr(split, 'label', None)
    if label is not None:
        out.labels = np.asarray(label, dtype=np.int8).ravel()
    cond = getattr(split, 'condition', None)
    if cond is not None:
        out.conditions = np.repeat(np.asarray(cond, dtype=object), M)
    return out


class M3FlipArm(Arm):
    def __init__(self):
        self.model = None

    def fit(self, rnd, spec):
        train = rnd.splits['train']
        if train.label is None:
            raise ValueError('m3flip needs train labels')
        self.model = FlipLagPairModel().fit(to_amir(train, 'train'))
        return self

    def predict(self, split):
        if self.model is None:
            raise RuntimeError('m3flip must be fitted (it refits on the train split at inference time)')
        p = self.model.predict(to_amir(split, split.name))
        return np.asarray(p, dtype=np.float64).reshape(*split.y.shape[:2], 5)

    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        (d / 'meta.json').write_text(json.dumps({'runtime_fit': True}) + '\n')

    @classmethod
    def load(cls, directory):
        return cls()
