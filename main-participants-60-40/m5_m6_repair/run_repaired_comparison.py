"""Freeze repaired M3/M5/M6 forecasts before scoring released validation answers.

Run prepare first, then score against the frozen predictions. This is a fixed
repair experiment, not a validation-driven hyperparameter sweep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from models.common.data import Split, MISSING, TYPES, PAYOUT_INDEX, PROB_COLUMNS, load_split
from models.common.evaluate import DEFAULT_DATA, write_predictions, to_dict
from models.m6_ensemble import Ensemble
from models.common.crossfit import FitCache
from slot_forecasting.contest import contest_score
import slot_forecasting.contest as contest_module
import slot_forecasting.scoring as scoring_module
import slot_forecasting.schema as schema_module
from slot_forecasting.io import read_targets, read_submission, read_rows, read_queries
from slot_forecasting.rounds import VERSION, round_metadata

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ('m3', 'm4', 'm5', 'm6')
PREVIOUS_NAMES = {'m3': 'm3_lag_pair_flip', 'm5': 'm5_boosted_corrector', 'm6': 'm6_ensemble'}
CONDITIONS = ('stable', 'short_history', 'drift', 'shared_change')


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT / 'models').rglob('*.py')) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def input_only(data, split_name):
    """Read query/history files only, never validation labels/targets/reference."""
    folder = data / split_name
    with (folder / 'queries.csv').open(newline='') as stream:
        queries = list(csv.DictReader(stream))
    cutoff = np.array([int(q['cutoff']) for q in queries])
    matrix = np.full((len(queries), int(cutoff.max()) + 1), MISSING, dtype=np.int8)
    index = {(q['episode_id'], q['machine_id']): i for i, q in enumerate(queries)}
    with (folder / 'history.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            i = index[(row['episode_id'], row['machine_id'])]
            t = int(row['tick'])
            if not 0 <= t < cutoff[i] or matrix[i, t] != MISSING:
                raise ValueError('Invalid or duplicated pre-cutoff observation')
            matrix[i, t] = PAYOUT_INDEX[int(row['payout'])]
    return Split(split_name, [q['query_id'] for q in queries],
                 np.array([q['episode_id'] for q in queries]), [q['machine_id'] for q in queries],
                 np.array([TYPES.index(q['machine_type']) for q in queries]),
                 np.array([int(q['signal']) for q in queries]), cutoff, matrix)


def binary_metrics(p, y, episode_ids, mask):
    episodes = sorted(set(episode_ids[mask]))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -y * np.log(p) - (1-y) * np.log(1-p)
    square = np.square(p-y)
    return {'episodes': len(episodes), 'rows': int(mask.sum()),
            'binary_logloss': float(np.mean([loss[mask & (episode_ids == e)].mean() for e in episodes])),
            'binary_brier': float(np.mean([square[mask & (episode_ids == e)].mean() for e in episodes]))}


def save_outer(out, result):
    y = np.asarray(result['y'], dtype=float)
    groups = np.asarray(result['episode_ids'])
    endpoint = np.asarray(result['is_endpoint'], dtype=bool)
    predictions = result['predictions']
    np.savez_compressed(out / 'train_outer_predictions.npz',
                        y=y, episode_ids=groups, row_ids=np.asarray(result['row_ids']),
                        is_endpoint=endpoint, **predictions)
    summary = {'scope': 'Entire pipeline refitted inside each outer fit partition; diagnostics use noisy train outcomes.',
               'outer_folds': result['folds'],
               'metrics': {name: {label: binary_metrics(np.asarray(p), y, groups, mask)
                                  for label, mask in [('all_pseudo_queries', np.ones(len(y), bool)),
                                                      ('released_cutoffs', endpoint),
                                                      ('historical_pseudo_queries', ~endpoint)]}
                           for name, p in predictions.items()}}
    save_json(out / 'train_outer_report.json', summary)
    return summary


def prepare(data, out, previous, outer_folds):
    if out.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(out))
    metadata = round_metadata(data)
    if metadata['round'] != 'main':
        raise ValueError('This frozen M3 repair comparison is for the main round')
    out.mkdir(parents=True)
    started = perf_counter()
    def progress(message):
        line = f'{perf_counter()-started:.1f}s {message}'
        print(line, flush=True)
        with (out / 'progress.log').open('a') as stream:
            stream.write(line + '\n')
    plan = {'experiment': 'Repair cutoff-relative features and all learned layers of M5/M6 episode exclusion.',
            'outer_evaluation_folds': outer_folds, 'numerical_threads': 2,
            'validation_policy': 'No validation answers loaded before forecast freeze; single fixed repair comparison.',
            'main_comparator': 'Unchanged M3; M4 diagnostic only.',
            'outer_caveat': 'Two outer folds fit on half of train. These diagnostics do not estimate full-train test score.',
            'source_sha256': source_hashes(),
            'scorer_sha256': {Path(m.__file__).name: digest(Path(m.__file__)) for m in (contest_module, scoring_module, schema_module)},
            'input_sha256': {str(p.relative_to(data)): digest(p) for p in [
                data/'metadata.json', data/'train/queries.csv', data/'train/history.csv', data/'train/labels.csv',
                data/'validation/queries.csv', data/'validation/history.csv',
                data/'test/queries.csv', data/'test/history.csv']},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__},
            'command': sys.argv}
    save_json(out / 'plan.json', plan)
    progress('Read train only for model fitting and selection')
    train = load_split(data, 'train')
    model = Ensemble(fit_cache=FitCache(), progress=progress)
    with threadpool_limits(limits=2):
        progress('Begin whole-pipeline outer training evaluation')
        outer = model.evaluate_heldout(train, folds=outer_folds, progress=progress)
        save_outer(out, outer)
        progress('Begin final fit on all training episodes')
        model.fit(train)
        models = {name: model.members[name] for name in ('m3', 'm4', 'm5')}
        models['m6'] = model
        descriptions = {name: m.describe() for name, m in models.items()}
        save_json(out / 'fitted_configurations.json', descriptions)
        # Model __getstate__ removes run-local cache/progress callbacks.
        joblib.dump(models, out / 'models.joblib')
        forecasts = {}
        for split_name in ('validation', 'test'):
            split = input_only(data, split_name)
            for name, fitted in models.items():
                progress(f'Predict {name} on {split_name} inputs without answers')
                probability = fitted.predict(split)
                path = out / name / ('validation_predictions.csv' if split_name == 'validation' else 'submission.csv')
                write_predictions(path, to_dict(split, probability))
                read_submission(path, set(split.query_ids))
                forecasts[str(path.relative_to(out))] = digest(path)
        # Re-load the exact saved checkpoint and check actual deployed inference.
        loaded = joblib.load(out / 'models.joblib')
        validation = input_only(data, 'validation')
        checkpoint_deltas = {}
        for name, fitted in loaded.items():
            expected = read_submission(out / name / 'validation_predictions.csv', set(validation.query_ids))
            current = fitted.predict(validation)
            checkpoint_deltas[name] = float(np.max(abs(current - np.array([expected[k] for k in validation.query_ids]))))
            if checkpoint_deltas[name] > 1e-8:
                raise ValueError('Checkpoint replay failed: ' + name)
    previous_hashes = {}
    m3_delta = None
    if previous is not None:
        for name, folder in PREVIOUS_NAMES.items():
            source = previous / folder / 'validation_predictions.csv'
            destination = out / 'previous' / name / 'validation_predictions.csv'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            previous_hashes[str(destination.relative_to(out))] = digest(destination)
        old = read_submission(out / 'previous/m3/validation_predictions.csv')
        new = read_submission(out / 'm3/validation_predictions.csv', set(old))
        m3_delta = max(abs(a-b) for key in old for a,b in zip(old[key],new[key]))
        if m3_delta > 1e-8:
            raise ValueError('M3 comparator changed: ' + str(m3_delta))
    if plan['source_sha256'] != source_hashes():
        raise ValueError('Source changed during experiment')
    freeze = {'forecast_sha256': forecasts, 'previous_sha256': previous_hashes,
              'checkpoint_sha256': digest(out/'models.joblib'), 'source_sha256': plan['source_sha256'],
              'plan_sha256': digest(out/'plan.json'), 'checkpoint_replay_max_abs_delta': checkpoint_deltas,
              'unchanged_m3_max_abs_delta': m3_delta, 'elapsed_seconds': perf_counter()-started,
              'scope': 'All training decisions and forecasts frozen before validation-answer scoring.'}
    save_json(out / 'freeze.json', freeze)
    progress('Frozen forecasts and checkpoints; ready for score phase')


def paired(candidate, control, targets, groups, reference, repeats=10000):
    conditions = {}
    values = {}
    for target in targets:
        k = target.query_id
        truth = np.asarray(target.probabilities)
        row = (np.square(candidate[k]-truth).sum() - np.square(control[k]-truth).sum(),
               np.square(reference[k]-truth).sum())
        values.setdefault(target.episode_id, []).append(row)
        conditions[target.episode_id] = groups[k]
    episodes = {e: np.mean(v, axis=0) for e,v in values.items()}
    draws = np.zeros(repeats); deltas = {}; rng=np.random.default_rng(314159)
    for condition in CONDITIONS:
        data = np.array([episodes[e] for e in sorted(episodes) if conditions[e] == condition])
        means = data.mean(0); deltas[condition] = float(means[0]/means[1])
        samples = data[rng.integers(len(data), size=(repeats,len(data)))].mean(1)
        draws += samples[:,0]/samples[:,1]/4
    return {'delta_R': float(np.mean(list(deltas.values()))),
            'fixed_forecast_episode_bootstrap_95pct': np.quantile(draws,[.025,.975]).tolist(),
            'condition_delta_R': deltas,
            'episodes_better': sum(v[0]<-1e-12 for v in episodes.values()),
            'episodes_worse': sum(v[0]>1e-12 for v in episodes.values()),
            'caveat': 'Fixed forecasts on already used public validation; not a fresh audit or selection-adjusted interval.'}


def score(data, out):
    if (out/'comparison.json').exists():
        raise FileExistsError('Comparison already exists')
    frozen=json.loads((out/'freeze.json').read_text())
    if digest(out/'plan.json') != frozen['plan_sha256']:
        raise ValueError('Experiment plan changed')
    plan=json.loads((out/'plan.json').read_text())
    for name, sha in plan['input_sha256'].items():
        if digest(data/name) != sha:raise ValueError('Data changed: '+name)
    for module in (contest_module, scoring_module, schema_module):
        if digest(Path(module.__file__)) != plan['scorer_sha256'][Path(module.__file__).name]:
            raise ValueError('Scorer changed')
    for name, sha in {**frozen['forecast_sha256'],**frozen['previous_sha256']}.items():
        if digest(out/name)!=sha:raise ValueError('Frozen prediction changed: '+name)
    if digest(out/'models.joblib')!=frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    queries=read_queries(data/'validation/queries.csv');ids={q.query_id for q in queries}
    targets=read_targets(data/'validation/targets.csv')
    groups={r['query_id']:r['condition'] for r in read_rows(data/'validation/conditions.csv',('query_id','condition'))}
    reference={k:np.asarray(v) for k,v in read_submission(data/'validation/reference.csv',ids).items()}
    predictions={name:{k:np.asarray(v) for k,v in read_submission(out/name/'validation_predictions.csv',ids).items()} for name in MODEL_NAMES}
    for name in PREVIOUS_NAMES:
        path=out/'previous'/name/'validation_predictions.csv'
        if path.exists():predictions['previous_'+name]={k:np.asarray(v) for k,v in read_submission(path,ids).items()}
    scores={name:contest_score(p,targets,groups,reference,max_points=60,version=VERSION) for name,p in predictions.items()}
    comparisons={name+'_minus_m3':paired(p,predictions['m3'],targets,groups,reference)
                 for name,p in predictions.items() if name!='m3'}
    for name in ('m5','m6'):
        previous='previous_'+name
        if previous in predictions:
            comparisons[name+'_minus_'+previous]=paired(predictions[name],predictions[previous],targets,groups,reference)
    result={'scope':'Single fixed repair experiment on all 96 previously used validation episodes; not a new audit.',
            'scores':scores,'paired_comparisons':comparisons,'freeze_sha256':digest(out/'freeze.json'),
            'validation_answer_sha256':{name:digest(data/'validation'/name) for name in ('targets.csv','reference.csv','conditions.csv')},
            'test_csvs':'All four forecasts validated for exact test query coverage and probability constraints in prepare.'}
    save_json(out/'comparison.json',result)
    lines=['# Repaired M5/M6 compared with unchanged M3','',result['scope'],'',
           '| Model | Points / 60 | R | Stable | Short history | Drift | Shared change |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in scores.items():
        values=' | '.join(f"{row['by_condition'][c]['loss_ratio']:.6f}" for c in CONDITIONS)
        lines.append(f"| {name} | {row['technical_points']:.6f} | {row['mean_loss_ratio']:.6f} | {values} |")
    lines += ['','| Repaired candidate minus M3 | ΔR | Descriptive 95% interval |','|---|---:|---|']
    for name in ('m5','m6'):
        row=comparisons[name+'_minus_m3'];lo,hi=row['fixed_forecast_episode_bootstrap_95pct']
        lines.append(f"| {name} | {row['delta_R']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', 'Lower R is better. Intervals resample episodes within condition with fixed forecasts.',
              'See train_outer_report.json for separately held-out whole-pipeline training diagnostics,',
              'fitted_configurations.json for selections, and freeze.json for hashes and replay checks.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','score'))
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--outer-folds',type=int,default=2)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.data.resolve(),args.out.resolve(),args.previous.resolve() if args.previous else None,args.outer_folds)
    else:score(args.data.resolve(),args.out.resolve())


if __name__=='__main__':
    main()
