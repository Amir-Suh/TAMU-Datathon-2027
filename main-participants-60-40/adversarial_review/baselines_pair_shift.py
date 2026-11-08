"""Frozen Bayesian posterior diagnostics from public histories; no scoring or refitting."""
import sys,json,csv
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.bayesian_lag_pairs import BayesianLagPairForecaster,LagPairConfig,lag_pair_rollout
from slot_forecasting.causal import tick_arrays,LAG_PAIRS
from slot_forecasting.io import load_inputs
model=BayesianLagPairForecaster(LagPairConfig(evidence_half_life=None,hazard=1/96))
model.prior=FrequencyForecaster.load(ROOT.parent/'runs/bayesian-lag-pairs/replay/population_prior.json')
learned=json.loads((ROOT.parent/'runs/bayesian-lag-pair-tuning/replay/report.json').read_text())['configuration']['pair_prior']
top3=np.argsort(learned)[-3:];rng=np.random.default_rng(481117)
report={'settings':{'hazard':1/96,'pair_prior':'uniform','frozen_population_prior':'runs/bayesian-lag-pairs/replay/population_prior.json','dominant_learned_prior_pairs':[LAG_PAIRS[i] for i in top3]},'note':'No labels/targets/reference read; no fitting; forecasts discarded; aggregate descriptive diagnostics only.'}
rows_by_split={}
def summary(q, posterior):
 mass=posterior.sum(axis=1);conditional=posterior/np.maximum(mass[:,None],1e-300);leading=conditional.argmax(axis=1)
 episodes=defaultdict(list)
 for i,query in enumerate(q):episodes[query.episode_id].append(i)
 in3=conditional[:,top3].sum(axis=1);ep=np.array([in3[v].mean() for v in episodes.values()]);boot=ep[rng.integers(len(ep),size=(3000,len(ep)))].mean(axis=1)
 return {'machines':len(q),'mean_total_pair_mass':float(mass.mean()),'fraction_total_pair_mass_gt_half':float(np.mean(mass>.5)),'conditional_pair_mean_mass':dict(zip(map(str,LAG_PAIRS),conditional.mean(axis=0).tolist())),'leading_pair_count':dict(Counter(str(LAG_PAIRS[i]) for i in leading)),'conditional_max_mass_quantiles':np.quantile(conditional.max(axis=1),[0,.1,.5,.9,1]).tolist(),'top3_conditional_mass_mean':float(in3.mean()),'top3_conditional_mass_episode_ci95':np.quantile(boot,[.025,.975]).tolist(),'top3_argmax_fraction':float(np.mean(np.isin(leading,top3))),'top3_conditional_mass_among_pair_mass_gt_half':float(in3[mass>.5].mean())}
for split in ('train','validation','test'):
 q,h=load_inputs(ROOT/'public'/split);spins,cutoffs=tick_arrays(q,h)
 _,diagnostics=lag_pair_rollout(spins,model.prior_rows(q),model.config,diagnostic_ticks=cutoffs)
 pair=diagnostics['pair_posterior'];short=(spins!=-1).sum(axis=1)<100
 report[split]={'all':summary(q,pair),'short_history':summary([v for v,s in zip(q,short) if s],pair[short]),'long_history':summary([v for v,s in zip(q,short) if not s],pair[~short])}
 rows=[]
 for i,query in enumerate(q):
  recent=spins[i,query.cutoff-32:query.cutoff];rows.append({'episode':query.episode_id,'context':f'{query.machine_type}|{query.signal}','win':float(np.mean(recent>0))})
 rows_by_split[split]=rows
 print(split,{s:{k:v for k,v in summary.items() if k not in ('conditional_pair_mean_mass','leading_pair_count')} for s,summary in report[split].items()},flush=True)
contexts=sorted(set(r['context'] for r in rows_by_split['train']));counts=Counter(r['context'] for r in rows_by_split['test']);means={}
for split,rows in rows_by_split.items():
 means[split]={c:float(np.mean([r['win'] for r in rows if r['context']==c])) for c in contexts}
report['recent32_win_context_means']=means
report['test_context_reweighted_train_recent32_win']=sum(means['train'][c]*counts[c] for c in contexts)/len(rows_by_split['test'])
report['test_recent32_win']=np.mean([r['win'] for r in rows_by_split['test']])
report['test_minus_context_reweighted_train_recent32_win']=report['test_recent32_win']-report['test_context_reweighted_train_recent32_win']
print('context-adjusted test minus train',report['test_minus_context_reweighted_train_recent32_win'])
out=ROOT/'adversarial_review/baselines_pair_shift.json';out.write_text(json.dumps(report,indent=2));print(out)
