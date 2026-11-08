"""Public-input-only distribution audit. Never opens targets, labels, or reference."""
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(path):
 with path.open() as f:return list(csv.DictReader(f))
def summarize(a):
 a=np.asarray(a,dtype=float)
 if not len(a):return None
 return {'mean':float(a.mean()),'quantiles_0_10_50_90_100':np.quantile(a,[0,.1,.5,.9,1]).tolist()}
def endpoint_stats(qs, histories, historical=False):
 records=[]
 for q in qs:
  seq=histories[(q['episode_id'],q['machine_id'])]; cutoff=int(q['cutoff'])
  ticks=sorted(seq)[-32:] if historical else [cutoff]
  for t in ticks:
   prior=[k for k in seq if k<t]
   n8=sum(t-lag in seq for lag in range(1,9));n40=sum(k>=t-40 for k in prior)
   n32=sum(k>=t-32 for k in prior);n64=sum(k>=t-64 for k in prior)
   recent=[seq[k]>0 for k in prior if k>=t-32]; older=[seq[k]>0 for k in prior if t-64<=k<t-32]
   records.append(dict(episode=q['episode_id'],machine_type=q['machine_type'],signal=int(q['signal']),cutoff=cutoff,tick=t,offset=cutoff-t,n_prior=len(prior),density=len(prior)/max(t,1),n_lag8=n8,n_40=n40,n_32=n32,n_64=n64,last_observed_gap=t-max(prior) if prior else t+1,win32=np.mean(recent) if recent else 0,rate_change=abs(np.mean(recent)-np.mean(older)) if recent and older else 0,short=int(len(seq)<100)))
 out={'rows':len(records),'machine_type':dict(Counter(r['machine_type'] for r in records)),'signals':dict(Counter(r['signal'] for r in records))}
 for k in ('cutoff','tick','offset','n_prior','density','n_lag8','n_40','n_32','n_64','last_observed_gap','win32','rate_change'):
  out[k]=summarize([r[k] for r in records])
 out['fraction_any_missing_lag8']=np.mean([r['n_lag8']<8 for r in records]);out['fraction_n40_lt40']=np.mean([r['n_40']<40 for r in records])
 for short in (0,1):
  vals=[r for r in records if r['short']==short];out['short' if short else 'long']={k:summarize([r[k] for r in vals]) for k in ('n_prior','density','n_lag8','n_40','win32','rate_change')}
 return out,records
report={};sets={};allrecords={}
for split in ('train','validation','test'):
 p=ROOT/'public'/split;qs=read(p/'queries.csv'); hs=read(p/'history.csv'); history=defaultdict(dict)
 for r in hs:history[(r['episode_id'],r['machine_id'])][int(r['tick'])]=int(r['payout'])
 sets[split]={q['episode_id'] for q in qs};end,records=endpoint_stats(qs,history);allrecords[split]=records
 cutoffs=defaultdict(set)
 for q in qs:cutoffs[q['episode_id']].add(int(q['cutoff']))
 report[split]={'queries':len(qs),'episodes':len(sets[split]),'history_rows':len(hs),'endpoints':end,'all_episodes_single_cutoff':all(len(v)==1 for v in cutoffs.values()),'unique_history_fingerprints':len({hashlib.sha256(str(sorted(v.items())).encode()).hexdigest() for v in history.values()}),'short_history_stratum_machine_count':dict(Counter('short' if len(v)<100 else 'long' for v in history.values()))}
 if split!='test':
  groups={r['query_id']:r['condition'] for r in read(p/'conditions.csv')};ec={q['episode_id']:groups[q['query_id']] for q in qs}
  report[split]['condition_episode_counts']=dict(Counter(ec.values()));report[split]['by_condition']={c:endpoint_stats([q for q in qs if groups[q['query_id']]==c],history)[0] for c in sorted(set(groups.values()))}
 if split=='train':report[split]['historical_training'],allrecords['train_historical']=endpoint_stats(qs,history,True)
report['episode_overlap']={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
# Episode bootstrap of difference in simple visible test vs train summaries.
rng=np.random.default_rng(71829);differences={}
for key in ('cutoff','density','n_prior','n_40','win32','rate_change'):
 arrays=[]
 for split in ('train','test'):
  values=defaultdict(list)
  for r in allrecords[split]:values[r['episode']].append(r[key])
  arrays.append(np.array([np.mean(v) for v in values.values()]))
 train,test=arrays;d=np.mean(test)-np.mean(train)
 boot=test[rng.integers(len(test),size=(3000,len(test)))].mean(axis=1)-train[rng.integers(len(train),size=(3000,len(train)))].mean(axis=1)
 differences[key]={'test_minus_train':float(d),'episode_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
report['test_minus_train_observables']=differences
out=ROOT/'adversarial_review'/'baselines_distribution_audit.json';out.write_text(json.dumps(report,indent=2))
for split in ('train','validation','test'):
 d=report[split];e=d['endpoints'];print(split,d['episodes'],d['history_rows'],'short',d['short_history_stratum_machine_count'])
 for key in ('cutoff','density','n_prior','n_lag8','n_40','win32','rate_change'):print(key,e[key])
 print('missinglag8',e['fraction_any_missing_lag8'])
print('TRAIN HISTORICAL')
for key in ('cutoff','tick','offset','density','n_prior','n_lag8','n_40','win32','rate_change'):print(key,report['train']['historical_training'][key])
print('DIFFERENCES',json.dumps(differences));print(out)
