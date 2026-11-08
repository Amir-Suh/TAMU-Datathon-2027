"""Independently rescore saved forecasts; never fit or select new candidates."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'main-participants-60-40/public/validation'
COLS = ['p_0', 'p_1', 'p_2', 'p_5', 'p_20']
CONDITIONS = ['stable', 'short_history', 'drift', 'shared_change']

def rows(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))

def read_prediction(path, ids):
    records = rows(path)
    keys = [r['query_id'] for r in records]
    if len(keys) != len(set(keys)) or set(keys) != set(ids):
        raise ValueError('Query coverage mismatch')
    keyed = {r['query_id']: r for r in records}
    matrix = np.array([[float(keyed[k][c]) for c in COLS] for k in ids])
    assert np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 1).all()
    assert np.max(abs(matrix.sum(axis=1) - 1)) < 1e-6
    return matrix / matrix.sum(axis=1, keepdims=True)

queries = {r['query_id']: r for r in rows(DATA / 'queries.csv')}
groups = {r['query_id']: r['condition'] for r in rows(DATA / 'conditions.csv')}
tuning = json.loads((ROOT / 'runs/model-comparison/comparison.json').read_text())['tuning_episodes']
# Both splits below have already been scored. No new candidate is scored on audit.
targets = {r['query_id']: r for r in rows(DATA / 'targets.csv')}
references = {r['query_id']: r for r in rows(DATA / 'reference.csv')}

def context(ids):
    truth = np.array([[float(targets[k][c]) for c in COLS] for k in ids])
    reference = np.array([[float(references[k][c]) for c in COLS] for k in ids])
    episodes = sorted({queries[k]['episode_id'] for k in ids})
    masks = [np.array([queries[k]['episode_id'] == e for k in ids]) for e in episodes]
    conditions = [groups[ids[np.flatnonzero(mask)[0]]] for mask in masks]
    ref = np.array([np.square(reference[mask] - truth[mask]).sum(axis=1).mean() for mask in masks])
    return truth, masks, conditions, ref

def losses(matrix, ctx):
    truth, masks, _, _ = ctx
    return np.array([np.square(matrix[mask] - truth[mask]).sum(axis=1).mean() for mask in masks])

def score(loss, ctx):
    _, _, cond, ref = ctx
    ratios = {c: float(loss[np.array(cond) == c].mean() / ref[np.array(cond) == c].mean()) for c in CONDITIONS}
    ratio = np.mean(list(ratios.values()))
    return {'points': float(60 / (1 + ratio)), 'R': float(ratio), 'condition_R': ratios,
            'max_condition_R': max(ratios.values()),
            'worst_episode_squared_error': float(loss.max())}

def comparison(candidate, control, ctx):
    _, _, cond, ref = ctx
    delta = candidate - control
    rng = np.random.default_rng(314159)
    draws = np.zeros(10000)
    cond_delta = {}
    for c in CONDITIONS:
        ix = np.flatnonzero(np.array(cond) == c)
        cond_delta[c] = float(delta[ix].mean() / ref[ix].mean())
        sampled = ix[rng.integers(len(ix), size=(len(draws),len(ix)))]
        draws += delta[sampled].mean(axis=1) / ref[sampled].mean(axis=1) / 4
    return {'delta_R': float(np.mean(list(cond_delta.values()))),
            'conditional_95pct_bootstrap_delta_R': np.quantile(draws,[.025,.975]).tolist(),
            'episodes_better': int((delta < -1e-12).sum()), 'episodes_worse': int((delta > 1e-12).sum()),
            'leave_one_condition_out_delta_R': {c: float(np.mean([v for k,v in cond_delta.items() if k != c])) for c in CONDITIONS},
            'caveat': 'Fixed forecast episode resampling; not model selection adjusted or a test-score prediction interval.'}

ids = sorted(k for k,q in queries.items() if q['episode_id'] in tuning)
ctx = context(ids)
unique = {}
skipped = []
by_path = {}
for path in sorted((ROOT/'runs').rglob('*.csv')):
    if 'audit-switch-96' in path.parts:
        continue
    records = rows(path)
    if not records or set(records[0]) != {'query_id',*COLS}:
        continue
    if {r['query_id'] for r in records} != set(ids):
        skipped.append(str(path.relative_to(ROOT)));continue
    matrix = read_prediction(path,ids)
    loss = losses(matrix,ctx)
    key = hashlib.sha256(matrix.tobytes()).hexdigest()
    result = unique.setdefault(key, {**score(loss,ctx),'paths':[]})
    name=str(path.relative_to(ROOT));result['paths'].append(name);by_path[name]=loss
ranked=sorted(unique.values(),key=lambda r:r['R'])
base='runs/bayesian-lag-pairs/replay/standalone.csv'
learned='runs/bayesian-lag-pair-tuning/replay/win_only.csv'
boost='runs/06-combine-forecasts/replay/fixed__boosted_3072_logistic_head.csv'
paired={
 'bayesian_minus_boosted_tuning': comparison(by_path[base],by_path[boost],ctx),
 'learned_prior_minus_uniform_tuning': comparison(by_path[learned],by_path[base],ctx)}
audit_dir=ROOT/'runs/bayesian-lag-pairs/audit-switch-96'
freeze=json.loads((audit_dir/'freeze.json').read_text())
audit_ids=sorted(freeze['query_episodes']);actx=context(audit_ids)
audit_losses={};audit_scores={}
for name,digest in freeze['prediction_sha256'].items():
 path=audit_dir/f'{name}.csv';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 loss=losses(read_prediction(path,audit_ids),actx);audit_losses[name]=loss;audit_scores[name]=score(loss,actx)
paired['bayesian_minus_boosted_existing_audit']=comparison(audit_losses['bayesian'],audit_losses['previous_lead'],actx)
cv=json.loads((ROOT/'runs/bayesian-lag-pair-tuning/replay/train_cv.json').read_text())
outer={name:{metric:float(np.mean([f['metrics'][name]['cutoffs'][metric]['brier'] for f in cv['outer']])) for metric in ('joint','win')} for name in ('control','win_only','selected')}
result={'scope':'Existing saved tuning forecasts and two previously frozen/scored audit forecasts only',
        'unique_prediction_arrays':len(ranked),'files_scored':len(by_path),'skipped_coverage':skipped,
        'tuning_ranking':ranked,'existing_audit_scores':audit_scores,'paired_diagnostics':paired,
        'nested_training_outer_cutoff_brier':outer}
(OUT/'verified_scores.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='tuning_ranking'},indent=2))
print('Top saved tuning forecasts:')
for r in ranked[:8]:print(round(r['points'],6), r['paths'][0])
