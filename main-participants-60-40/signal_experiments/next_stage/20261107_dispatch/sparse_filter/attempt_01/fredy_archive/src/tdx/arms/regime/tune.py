"""Explicit released-validation evaluator; never called by Arm.fit or dev sweep.

Select each specialist on the kit tuning half, then report its untouched audit
half. Pass the printed --set argument to tdx.run; no implicit artifact loading.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round
from tdx.paths import data_root
from . import build
from .specialists import CONDITIONS, forecast


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round', default='main', choices=('main', 'adaptation'))
    p.add_argument('--data')
    p.add_argument('--out', default='runs/regime/tuning')
    a = p.parse_args(argv)
    rnd = load_round(data_root(a.data, a.round), a.round, answers=True)
    arm = build().fit(fit_round(rnd, dev_spec()), dev_spec())
    s = rnd.splits['validation']
    grid = [(half, strength) for half in (8., 16., 32., 64., None) for strength in (5., 20., 60.)]
    preds = np.stack([forecast(s, arm.priors, *param) for param in grid])
    loss = ((preds - s.target[None]) ** 2).mean(axis=(2, 3))
    ref = ((s.reference - s.target) ** 2).mean(axis=(1, 2))
    params, report = [], {}
    for condition in CONDITIONS:
        eps = [i for i, c in enumerate(s.condition) if c == condition]
        eps.sort(key=lambda i: hashlib.sha256(s.ids[i].encode()).hexdigest())
        tuning, audit = eps[::2], eps[1::2]
        chosen = int(np.argmin(loss[:, tuning].mean(axis=1)))
        params.append(grid[chosen])
        report[condition] = {'chosen': grid[chosen],
                             'tuning_episodes': [s.ids[i] for i in tuning],
                             'audit_episodes': [s.ids[i] for i in audit],
                             'tuning_ratio': float(loss[chosen, tuning].mean() / ref[tuning].mean()),
                             'audit_ratio': float(loss[chosen, audit].mean() / ref[audit].mean()),
                             'grid_tuning_mse': loss[:, tuning].mean(axis=1).tolist()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    value = {'specialists': params, 'grid': grid, 'conditions': report,
             'protocol': 'Per-condition SHA256-order alternating tuning/audit halves; select tuning MSE; audit once after selection.'}
    (out / 'tuning.json').write_text(json.dumps(value, indent=2) + '\n')
    # Evaluation diagnostics may read released conditions; training never does.
    for name in ('validation', 'test'):
        split = rnd.splits[name]
        posterior = arm.condition_posterior(split)
        np.save(out / ('cond_posterior_val.npy' if name == 'validation' else 'cond_posterior_test.npy'), posterior)
    starts, fires = arm.change_model.detect(s)
    value['validation_classifier_accuracy'] = float(np.mean(np.asarray(CONDITIONS)[arm.condition_posterior(s).argmax(1)] == s.condition))
    value['validation_confusion'] = [[int(np.sum((s.condition == c) & (arm.condition_posterior(s).argmax(1) == k))) for k in range(4)] for c in CONDITIONS]
    value['validation_change_fire_rates'] = {c: float(fires[s.condition == c].mean()) for c in CONDITIONS}
    (out / 'tuning.json').write_text(json.dumps(value, indent=2) + '\n')
    print('specialists=' + json.dumps(params, separators=(',', ':')))
    print('validation classifier accuracy=' + str(value['validation_classifier_accuracy']))
    print(json.dumps({c: {k: v for k, v in report[c].items() if k in ('chosen', 'tuning_ratio', 'audit_ratio')} for c in CONDITIONS}))


if __name__ == '__main__':
    main()
