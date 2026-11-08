"""Shared Arm implementation. Dev fit never opens evaluation answers."""
import json
from pathlib import Path
import numpy as np
from tdx.arm import Arm
from .specialists import DEFAULT_SPECIALISTS, fit_priors, forecast, probabilities
from .classifier import ConditionModel
from .changepoint import ChangeModel


class RegimeArm(Arm):
    def __init__(self, method='d2_routed', specialists=None, transductive=False):
        if method not in ('d2_routed', 'd3_changepoint', 'd2_d3_mix'):
            raise ValueError('Unknown regime method')
        self.method = method
        self.specialists = [list(p) for p in (specialists or DEFAULT_SPECIALISTS)]
        if len(self.specialists) != 4 or any(len(p) != 2 or p[1] <= 0 or (p[0] is not None and p[0] <= 0) for p in self.specialists):
            raise ValueError('Expected four positive (half_life, strength) pairs; null means infinite half-life')
        self.transductive = transductive
        self.posteriors = {}
        self.diagnostics = {}

    def fit(self, rnd, spec):
        supervised = [rnd.splits[n] for n in spec.label_splits]
        self.condition_model = ConditionModel().fit(supervised)
        self.change_model = ChangeModel().fit(supervised)
        self.priors = fit_priors([rnd.splits[n] for n in spec.tick_splits])
        self.posteriors = {}
        self.diagnostics = {}
        return self

    def condition_posterior(self, split):
        return self.condition_model.predict_proba(split)

    def predict(self, split):
        posterior = self.condition_posterior(split)
        key = 'val' if split.name == 'validation' else split.name
        self.posteriors[key] = posterior
        starts, fires = self.change_model.detect(split)
        self.diagnostics[key] = {'ids': list(split.ids), 'change_ticks': starts.tolist(), 'fires': fires.tolist()}
        if self.method != 'd3_changepoint':
            experts = np.stack([forecast(split, self.priors, half, strength) for half, strength in self.specialists], axis=1)
            routed = np.einsum('ec,ecmk->emk', posterior, experts)
        if self.method != 'd2_routed':
            # Full-history fallback; only triggered episodes use post-change ticks and half-life eight.
            base = forecast(split, self.priors, None, 20.)
            changed = forecast(split, self.priors, 8., 20., starts=starts)
            cp = np.where(fires[:, None, None], changed, base)
        if self.method == 'd2_routed':
            return probabilities(routed)
        if self.method == 'd3_changepoint':
            return probabilities(cp)
        return probabilities((routed + cp) / 2)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        value = {'version': 1, 'method': self.method, 'specialists': self.specialists,
                 'transductive': self.transductive, 'priors': self.priors.tolist(),
                 'condition_model': self.condition_model.to_dict(), 'change_model': self.change_model.to_dict()}
        (directory / 'model.json').write_text(json.dumps(value, indent=2) + '\n')
        # Harness calls save after both predictions; artifacts follow custom --out too.
        for name, posterior in self.posteriors.items():
            if name in ('val', 'test'):
                np.save(directory.parent / f'cond_posterior_{name}.npy', posterior)
        report = {'classifier': self.condition_model.report, 'changepoint': self.change_model.report,
                  'predictions': self.diagnostics}
        (directory.parent / 'regime_report.json').write_text(json.dumps(report, indent=2) + '\n')

    @classmethod
    def load(cls, directory):
        d = json.loads((Path(directory) / 'model.json').read_text())
        if d['version'] != 1:
            raise ValueError('Unknown regime model version')
        obj = cls(d['method'], d['specialists'], d['transductive'])
        obj.priors = np.asarray(d['priors'], dtype=np.float64)
        obj.condition_model = ConditionModel.from_dict(d['condition_model'])
        obj.change_model = ChangeModel.from_dict(d['change_model'])
        return obj
