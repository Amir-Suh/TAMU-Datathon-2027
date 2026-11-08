"""ZIP-root inference entry point (also executed by the builder)."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from tdx.data import _read_split, Round
from tdx.fit import FitSpec
from tdx.package.blend import apply, probabilities, validate
from slot_forecasting.io import write_submission


def no_torch():
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('PyTorch imported in inference graph')


def fallback_record(errors, selected):
    record = dict(errors=errors, selected=selected)
    Path('/tmp/predict_fallback.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PREDICT FALLBACK: ' + json.dumps(record), file=sys.stderr, flush=True)


def run(root, data, out):
    root, data = Path(root), Path(data).resolve()
    config = json.loads((root / 'config.json').read_text())
    meta = json.loads((data / 'metadata.json').read_text())
    if meta['round'] != config['round'] or meta['season_id'] != config['season_id']:
        raise ValueError('Entry was built for another round/season; refit before use')
    split = _read_split(data, 'test')
    shape = (*split.query_ids.shape, 5)
    records = {a['id']: a for a in config['arms']}
    loaded, predictions = {}, {}

    def predict(arm_id):
        if arm_id in predictions:
            return predictions[arm_id]
        if arm_id == 'pooled_recent':
            from tdx.arms.pooled import build
            arm = build(method='pooled_recent')
            train = _read_split(data, 'train')
            rnd = Round(meta['round'], data, {'train': train, 'test': split}, float(meta['max_points']))
            arm.fit(rnd, FitSpec(('train',), ('train',), ()))
        else:
            record = records[arm_id]
            cls = getattr(importlib.import_module(record['module']), record['class'])
            arm = cls.load(root / record['model'])
            if record.get('fit_mode') == 'runtime':
                train = _read_split(data, 'train')
                arm.fit(Round(meta['round'], data, {'train': train, 'test': split}, float(meta['max_points'])),
                        FitSpec(('train',), ('train',), ()))
        value = probabilities(arm.predict(split), shape)
        no_torch()
        loaded[arm_id] = arm
        predictions[arm_id] = value
        return value

    errors = []
    try:
        arm_ids = config['primary']
        blend = validate(json.loads((root / 'blend.json').read_text()), arm_ids)
        values = [predict(name) for name in arm_ids]
        pi = None
        provider = blend.get('condition_arm')
        if provider:
            predict(provider)
            pi = loaded[provider].condition_posterior(split)
        result = apply(blend, values, split, pi)
    except Exception as exc:
        errors.append({'entry': 'primary', 'error': f'{type(exc).__name__}: {exc}'})
        print('Primary forecast failed: ' + str(exc), file=sys.stderr, flush=True)
        for name in config['fallback']:
            try:
                result = predict(name)
                fallback_record(errors, name)
                break
            except Exception as fallback_exc:
                errors.append({'entry': name, 'error': f'{type(fallback_exc).__name__}: {fallback_exc}'})
                print(f'Fallback {name} failed: {fallback_exc}', file=sys.stderr, flush=True)
        else:
            fallback_record(errors, None)
            raise RuntimeError('All forecasts failed')
    no_torch()
    write_submission(out, {str(q): tuple(p) for q, p in zip(split.query_ids.flat, result.reshape(-1, 5))})
    print(f'INFERENCE OK rows={split.query_ids.size} torch_imported=False', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run(Path(__file__).resolve().parent, args.data, args.out)


if __name__ == '__main__':
    main()
