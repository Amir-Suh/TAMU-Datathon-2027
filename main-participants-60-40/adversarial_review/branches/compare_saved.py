"""Compare already-saved branch forecasts on identical released validation rows."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'main-participants-60-40/public'
COLS=['p_0','p_1','p_2','p_5','p_20'];CONDS=['stable','short_history','drift','shared_change']
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def forecast(p):
 r=rows(p);assert list(r[0])==['query_id',*COLS]
 d={x['query_id']:np.array([float(x[c]) for c in COLS]) for x in r};assert len(d)==len(r)
 a=np.array(list(d.values()));assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all() and np.max(abs(a.sum(1)-1))<1e-6
 return {k:v/v.sum() for k,v in d.items()}
queries={r['query_id']:r for r in rows(DATA/'validation/queries.csv')}
ids=sorted(queries);episodes=sorted({q['episode_id'] for q in queries.values()})
groups={r['query_id']:r['condition'] for r in rows(DATA/'validation/conditions.csv')}
conditions=np.array([groups[next(k for k in ids if queries[k]['episode_id']==e)] for e in episodes])
ep_index=np.array([episodes.index(queries[k]['episode_id']) for k in ids])
truth_map={r['query_id']:np.array([float(r[c]) for c in COLS]) for r in rows(DATA/'validation/targets.csv')}
truth=np.array([truth_map[k] for k in ids]);ref_map=forecast(DATA/'validation/reference.csv');ref=np.array([ref_map[k] for k in ids])
def episode_error(a):return np.bincount(ep_index,weights=np.square(a-truth).sum(1))/np.bincount(ep_index)
ref_error=episode_error(ref)
tuning=set(json.loads((ROOT/'runs/model-comparison/comparison.json').read_text())['tuning_episodes'])
splits={'full_validation':np.ones(len(episodes),bool),'previous_tuning_half':np.array([e in tuning for e in episodes]),'previous_audit_half':np.array([e not in tuning for e in episodes])}
def score(loss,mask):
 ratios={c:float(loss[mask&(conditions==c)].mean()/ref_error[mask&(conditions==c)].mean()) for c in CONDS};R=np.mean(list(ratios.values()))
 return {'points':float(60/(1+R)),'R':float(R),'condition_R':ratios,'episodes':int(mask.sum())}
arrays={};paths={}
for run in ['2026-11-07_m1_m2','2026-11-07_m1_to_m6']:
 for p in sorted((HERE/'amir/results'/run).glob('*/validation_predictions.csv')):
  pred=forecast(p);assert set(pred)==set(ids)
  name=run+'/'+p.parent.name
  arrays[name]=np.array([pred[k] for k in ids]);paths[name]=str(p.relative_to(HERE))
for name,tp,ap in [('previous_bayesian','runs/bayesian-lag-pairs/replay/standalone.csv','runs/bayesian-lag-pairs/audit-switch-96/bayesian.csv'),('previous_boosted_logistic','runs/06-combine-forecasts/replay/fixed__boosted_3072_logistic_head.csv','runs/bayesian-lag-pairs/audit-switch-96/previous_lead.csv')]:
 pred=forecast(ROOT/tp);audit=forecast(ROOT/ap);assert not (pred.keys() & audit.keys());pred.update(audit);assert set(pred)==set(ids)
 arrays[name]=np.array([pred[k] for k in ids]);paths[name]=[tp,ap]
losses={name:episode_error(a) for name,a in arrays.items()}
results={name:{'source':paths[name],'scores':{s:score(losses[name],m) for s,m in splits.items()},'p0_mse':float(np.square(a[:,0]-truth[:,0]).mean()),'class_mse':np.square(a-truth).mean(0).tolist()} for name,a in arrays.items()}
def paired(a,b,mask):
 diff=losses[a]-losses[b];rng=np.random.default_rng(314159);draws=np.zeros(10000);ds={}
 for c in CONDS:
  ix=np.flatnonzero(mask&(conditions==c));ds[c]=float(diff[ix].mean()/ref_error[ix].mean());sample=ix[rng.integers(len(ix),size=(10000,len(ix)))];draws+=diff[sample].mean(1)/ref_error[sample].mean(1)/4
 return {'delta_R':float(np.mean(list(ds.values()))),'fixed_forecast_episode_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist(),'condition_delta_R':ds,'episodes_better':int((diff[mask]<-1e-12).sum()),'episodes_worse':int((diff[mask]>1e-12).sum()),'leave_one_condition_out_delta_R':{c:float(np.mean([d for k,d in ds.items() if k!=c])) for c in CONDS}}
prefix='2026-11-07_m1_to_m6/'
pairs=[(prefix+'m3_lag_pair_flip','previous_bayesian'),(prefix+'m6_ensemble',prefix+'m3_lag_pair_flip'),(prefix+'m4_lag_pair_flip_shared',prefix+'m3_lag_pair_flip'),(prefix+'m5_boosted_corrector',prefix+'m3_lag_pair_flip'),(prefix+'m2_lag_pair','previous_bayesian')]
comparisons={a+' MINUS '+b:{s:paired(a,b,m) for s,m in splits.items()} for a,b in pairs}
test_ids={q['query_id'] for q in rows(DATA/'test/queries.csv')};test_checks={}
for p in (HERE/'amir/results').glob('*/*/submission.csv'):
 pred=forecast(p);assert set(pred)==test_ids;test_checks[str(p.relative_to(HERE))]={'valid_rows':len(pred),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
data_checks={}
for p in (HERE/'amir/main-participants-60-40/public').rglob('*'):
 if p.is_file():
  rel=p.relative_to(HERE/'amir/main-participants-60-40/public');q=DATA/rel;data_checks[str(rel)]={'byte_identical':p.read_bytes()==q.read_bytes(),'content_identical':json.loads(p.read_text())==json.loads(q.read_text()) if p.suffix=='.json' else p.read_bytes()==q.read_bytes()}
assert all(v['content_identical'] for v in data_checks.values())
output={'caution':'All Amir models were already evaluated on all 96 validation episodes. Reusing old tuning/audit partition is descriptive, NOT a fresh independent audit. Bootstrap holds saved forecasts fixed and is not selection-adjusted.','data_files_identical':data_checks,'results':results,'comparisons':comparisons,'saved_test_format_checks':test_checks}
(HERE/'cross_branch_scores.json').write_text(json.dumps(output,indent=2)+'\n')
for name,row in sorted(results.items(),key=lambda x:x[1]['scores']['full_validation']['R']):print(name,*[round(row['scores'][s]['points'],6) for s in splits])
for name,row in comparisons.items():print(name,json.dumps(row['full_validation']))
print('Identical data files',len(data_checks),'valid test CSVs',len(test_checks))
