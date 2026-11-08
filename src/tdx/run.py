"""Run one arm with explicit fit permissions and kit-backed scoring."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = str(min(12, int(os.environ.get(_var, "12"))))
import argparse
from dataclasses import replace
import importlib
import inspect
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .data import load_round, with_answers
from .fit import dev_spec, final_spec
from .paths import ROOT, data_root
from .score import score, prediction_map
from .log import append
from slot_forecasting.io import write_submission, write_json

def fit_round(rnd, spec):
    """Pass only authorized supervision to an arm; scoring retains its own Round."""
    return replace(rnd, splits={name: replace(s,
        label=s.label if name in spec.label_splits else None,
        condition=s.condition if name in spec.label_splits else None,
        target=s.target if name in spec.soft_splits else None,
        reference=None) for name, s in rnd.splits.items()})

def parse_set(values):
    out = {}
    for item in values:
        key, value = item.split('=', 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out

def arm_module(name):
    if not name.isidentifier():
        raise ValueError('Invalid arm name')
    return importlib.import_module(f'tdx.arms.{name}')

def build_arm(module, params, device=None, smoke=None):
    params = dict(params)
    sig = inspect.signature(module.build)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for name, value in [('device', device), ('smoke', smoke)]:
        if value is not None and (name in sig.parameters or accepts_kwargs):
            params[name] = value
    return module.build(**params)

def execute(rnd, arm_name, variant, mode='dev', overrides=None, out=None, device=None, smoke=None):
    start = perf_counter()
    module = arm_module(arm_name)
    params = dict(module.VARIANTS[variant])
    params.update(overrides or {})
    # Diagnostic override: --set tick_splits=train,test fits on exactly these splits' histories (dev mode only; labels from train).
    ticks = params.pop('tick_splits', None)
    arm = build_arm(module, params, device, smoke)
    spec = final_spec() if mode == 'final' else dev_spec(bool(params.get('transductive', False)))
    if ticks is not None:
        if mode != 'dev':
            raise ValueError('tick_splits override is for dev mode only')
        names = tuple(str(ticks).split(','))
        if not set(names) <= {'train', 'validation', 'test'} or 'train' not in names:
            raise ValueError('tick_splits must be a comma list of train/validation/test containing train')
        spec = dev_spec(False).__class__(names, ('train',), ())
    arm.fit(fit_round(rnd, spec), spec)
    out = Path(out or ROOT / 'runs' / arm_name / variant)
    out.mkdir(parents=True, exist_ok=True)
    result = None
    if mode == 'dev':
        s = rnd.splits['validation']
        pred = arm.predict(replace(s, label=None, condition=None, target=None, reference=None))
        mapping = prediction_map(s, pred)
        np.save(out / 'val_pred.npy', pred)
        write_submission(out / 'val_pred.csv', mapping)
        result = score(with_answers(rnd), pred)
        write_json(out / 'score.json', result)
    else:
        # Reusing an output directory must not leave stale dev score artifacts.
        for filename in ('val_pred.npy', 'val_pred.csv', 'score.json'):
            (out / filename).unlink(missing_ok=True)
    s = rnd.splits['test']
    pred = arm.predict(s)
    write_submission(out / 'test_pred.csv', prediction_map(s, pred))
    np.save(out / 'test_pred.npy', pred)
    try:
        arm.save(out / 'model')
    except NotImplementedError:
        pass
    seconds = perf_counter() - start
    meta = dict(arm=arm_name, variant=variant, mode=mode, params=params, round=rnd.name,
                data=str(rnd.root), data_root=str(rnd.root), max_points=rnd.max_points, seconds=seconds,
                fit_spec=dict(tick_splits=spec.tick_splits, label_splits=spec.label_splits, soft_splits=spec.soft_splits),
                device=device, smoke=smoke)
    meta['metadata_sha256'] = hashlib.sha256((rnd.root / 'metadata.json').read_bytes()).hexdigest()
    meta['validation_queries_sha256'] = hashlib.sha256((rnd.root / 'validation' / 'queries.csv').read_bytes()).hexdigest()
    write_json(out / 'meta.json', meta)
    append(arm_name, variant, mode, result, seconds)
    text = f'{arm_name}/{variant} {mode}: '
    print(text + (f'{result["points"]:.6f} points' if result else 'test predictions saved') + f' ({seconds:.3f}s)')
    return result

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--mode', choices=('dev', 'final'), default='dev')
    p.add_argument('--round', choices=('main', 'adaptation'), default='main')
    p.add_argument('--data')
    p.add_argument('--out')
    p.add_argument('--set', action='append', default=[])
    return p

def main(argv=None):
    a = parser().parse_args(argv)
    rnd = load_round(data_root(a.data, a.round), a.round, answers=a.mode == 'final')
    execute(rnd, a.arm, a.variant, a.mode, parse_set(a.set), a.out)

if __name__ == '__main__':
    main()
