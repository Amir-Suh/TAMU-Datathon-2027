"""Verify frozen, fitted models under changes to prediction batching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from models.common.crossfit import subset_split
from models.common.data import MISSING
from run_repaired_comparison import DEFAULT_DATA, digest, input_only, save_json
from slot_forecasting.io import read_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    freeze = json.loads((args.out / 'freeze.json').read_text())
    checkpoint = args.out / 'models.joblib'
    assert digest(checkpoint) == freeze['checkpoint_sha256']
    models = joblib.load(checkpoint)
    split = input_only(args.data, 'validation')
    expected = {name: np.asarray([p[q] for q in split.query_ids]) for name in models
                for p in [read_submission(args.out / name / 'validation_predictions.csv', set(split.query_ids))]}
    order = np.random.default_rng(271828).permutation(split.n)
    reordered = subset_split(split, order)
    padded = subset_split(split, np.arange(split.n))
    padded.X = np.pad(padded.X, ((0, 0), (0, 127)), constant_values=MISSING)
    episodes = sorted(set(split.episode_ids),
                      key=lambda e: int(split.cutoff[split.episode_ids == e].max()))
    chosen = [episodes[i] for i in np.linspace(0, len(episodes)-1, 8, dtype=int)]
    checks = [('row_permutation', reordered, order),
              ('extra_missing_padding', padded, np.arange(split.n))]
    for ep in chosen:
        rows = np.flatnonzero(split.episode_ids == ep)
        single = subset_split(split, rows)
        single.X = single.X[:, :int(single.cutoff.max())+1]
        checks.append(('isolated_trimmed_episode_' + str(ep), single, rows))
    report = {'checkpoint_sha256': freeze['checkpoint_sha256'],
              'scope': 'Actual frozen models; full-episode batching preserves M4 shared inference.',
              'tolerance': 1e-10, 'checks': {}}
    with threadpool_limits(limits=2):
        for label, changed, rows in checks:
            deltas = {name: float(np.max(np.abs(model.predict(changed) - expected[name][rows])))
                      for name, model in models.items()}
            report['checks'][label] = deltas
            print(label, deltas, flush=True)
            if any(delta > report['tolerance'] for delta in deltas.values()):
                raise AssertionError('Prediction batching changed forecasts: ' + label)
    report['passed'] = True
    save_json(args.out / 'inference_verification.json', report)


if __name__ == '__main__':
    main()
