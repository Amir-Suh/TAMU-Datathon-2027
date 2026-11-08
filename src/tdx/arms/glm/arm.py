"""Hierarchical pooled + episode-machine softmax arm."""
import json
from pathlib import Path

import numpy as np

from tdx.arm import Arm
from tdx.feat import tick_rows
from .features import all_features, recency_weights
from .model import fit_weights, probabilities


class GLMArm(Arm):
    _num_logits = 5

    @staticmethod
    def _fit_weights(*args, **kwargs):
        return fit_weights(*args, **kwargs)

    @staticmethod
    def _probabilities(*args, **kwargs):
        return probabilities(*args, **kwargs)

    def __init__(self, L=4, lambda_=10.0, tau=96.0, neighbours=True, rates=True,
                 transductive=False, w_cut=20.0, pooled_maxiter=1500, local_maxiter=150,
                 lambda0=1.0):
        if int(L) != L or L < 1 or tau <= 0 or lambda_ <= 0 or lambda0 <= 0 or w_cut < 0:
            raise ValueError('Invalid GLM hyperparameters')
        self.params = dict(L=int(L), lambda_=float(lambda_), tau=float(tau),
                           neighbours=bool(neighbours), rates=bool(rates),
                           transductive=bool(transductive), w_cut=float(w_cut),
                           pooled_maxiter=int(pooled_maxiter), local_maxiter=int(local_maxiter),
                           lambda0=float(lambda0))
        self.W0 = None
        self.fit_diagnostics = {}

    def _features(self, split):
        return all_features(split, **{k: self.params[k] for k in ('L', 'neighbours', 'rates')})

    def fit(self, rnd, spec):
        xs, ys, ws = [], [], []
        eye = np.eye(5, dtype=np.float64)
        for name, split in rnd.splits.items():
            if name not in (*spec.tick_splits, *spec.label_splits, *spec.soft_splits):
                continue
            x = self._features(split)
            if name in spec.tick_splits:
                e, m, t = tick_rows(split)
                xs.append(x[e, m, t])
                ys.append(eye[split.y[e, m, t]])
                latest = np.max(np.where((split.y >= 0) &
                    (np.arange(split.y.shape[-1])[None, None, :] < split.cutoff[:, None, None]),
                    np.arange(split.y.shape[-1]), -1), axis=-1)
                ws.append(np.exp2((t - latest[e, m]) / self.params['tau']))
            qx = x[np.arange(x.shape[0])[:, None], np.arange(x.shape[1])[None, :], split.cutoff[:, None]]
            if name in spec.label_splits and split.label is not None:
                ok = split.label >= 0
                xs.append(qx[ok])
                ys.append(eye[split.label[ok]])
                ws.append(np.full(ok.sum(), self.params['w_cut'], dtype=np.float64))
            if name in spec.soft_splits and split.target is not None:
                xs.append(qx.reshape(-1, qx.shape[-1]))
                ys.append(np.asarray(split.target, dtype=np.float64).reshape(-1, 5))
                ws.append(np.full(split.target.shape[0] * split.target.shape[1], self.params['w_cut'], dtype=np.float64))
        if not xs:
            raise ValueError('FitSpec supplies no GLM training rows')
        x, target, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        self.fit_diagnostics = {}
        self.W0 = self._fit_weights(x, target, weight, self.params['lambda0'],
                              np.zeros((x.shape[-1], self._num_logits)), self.params['pooled_maxiter'],
                              diagnostics=self.fit_diagnostics)

    def predict(self, split):
        if self.W0 is None:
            raise RuntimeError('Fit or load the GLM before prediction')
        x = self._features(split)
        E, M = split.y.shape[:2]
        result = np.empty((E, M, 5), dtype=np.float64)
        eye = np.eye(5, dtype=np.float64)
        ticks = np.arange(split.y.shape[-1])
        for e in range(E):
            for m in range(M):
                good = (split.y[e, m] >= 0) & (ticks < split.cutoff[e])
                t = ticks[good]
                local = self._fit_weights(x[e, m, t], eye[split.y[e, m, t]],
                                    recency_weights(t, self.params['tau']), self.params['lambda_'],
                                    self.W0, self.params['local_maxiter'])
                result[e, m] = self._probabilities(x[e, m, int(split.cutoff[e])], local)
        return result

    def save(self, directory):
        if self.W0 is None:
            raise RuntimeError('Cannot save an unfitted GLM')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / 'model.npz', W0=self.W0,
                 params=np.array(json.dumps(self.params, sort_keys=True)),
                 fit_diagnostics=np.array(json.dumps(self.fit_diagnostics, sort_keys=True)))

    @classmethod
    def load(cls, directory):
        with np.load(Path(directory) / 'model.npz', allow_pickle=False) as z:
            params = json.loads(str(z['params']))
            two_stage = params.pop('two_stage', False)
            if two_stage:
                from .factorized import FactorizedGLMArm
                model = FactorizedGLMArm(**params)
            else:
                model = cls(**params)
            model.W0 = z['W0'].copy()
            if 'fit_diagnostics' in z:
                model.fit_diagnostics = json.loads(str(z['fit_diagnostics']))
        return model


VARIANTS = {
    'l2': {'L': 2},
    'l4': {'L': 4},
    'l8': {'L': 8},
    'lambda1': {'L': 4, 'lambda_': 1},
    'lambda100': {'L': 4, 'lambda_': 100},
    'tau32': {'L': 4, 'lambda_': 10, 'tau': 32},
    'tauinf': {'L': 4, 'lambda_': 10, 'tau': 1e9},
}


def build(device='auto', smoke=False, two_stage=False, **params):
    # The shared sweep passes these hints; numpy fitting is always CPU and uses
    # the requested iteration limits unchanged, including during smoke runs.
    if two_stage:
        from .factorized import FactorizedGLMArm
        return FactorizedGLMArm(**params)
    return GLMArm(**params)
