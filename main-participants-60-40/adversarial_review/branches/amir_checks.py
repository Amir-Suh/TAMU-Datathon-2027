"""Small structural checks; no model search, training, or new scoring."""
from pathlib import Path
import sys, json
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parent/'amir'
sys.path.insert(0,str(ROOT))
from models.common.data import load_split,Split
from models.common.components import context_priors
from models.m1_hierarchical_frequency.model import HierarchicalFrequencyModel
from models.m2_lag_pair.model import LagPairModel
from models.m3_lag_pair_flip.model import FlipLagPairModel
from models.m4_lag_pair_flip_shared.model import SharedFlipLagPairModel

def subset(s,ids):
 ids=np.asarray(ids);width=int(s.cutoff[ids].max())+1
 return Split(s.name,[s.query_ids[i] for i in ids],s.episode_ids[ids], [s.machine_ids[i] for i in ids],s.machine_type[ids],s.signal[ids],s.cutoff[ids],s.X[ids,:width],None if s.labels is None else s.labels[ids],None if s.conditions is None else s.conditions[ids],None if s.targets is None else s.targets[ids],None if s.reference is None else s.reference[ids])

def configured(cls):
 m=cls();m.win_prior=np.full(9,.4);m.size.prior=np.tile([.5,.3,.15,.05],(9,1));m.size.params={'strength':100.,'half_life':float('inf')}
 if isinstance(m,HierarchicalFrequencyModel):m.params={'strength':5.,'half_life':16.}
 else:
  m.hi=np.full(9,.85);m.lo=np.full(9,.08);m.pair_prior=np.full(len(m.pairs),1/len(m.pairs))
  if isinstance(m,FlipLagPairModel):m.params={'kappa':120.};m.tau_q=np.full(len(m.tau_grid),1/len(m.tau_grid))
  else:m.params={'kappa':3.,'evidence_half_life':64.}
 return m

val=load_split(ROOT/'main-participants-60-40/public','validation')
low_ep=val.episode_ids[np.argmin(val.cutoff)];high_ep=val.episode_ids[np.argmax(val.cutoff)]
low_ids=np.where(val.episode_ids==low_ep)[0];high_ids=np.where(val.episode_ids==high_ep)[0]
low=subset(val,low_ids);both=subset(val,np.r_[low_ids,high_ids])
result={'scope':'Fixed synthetic parameter configurations using real public validation histories. Not saved model replay or performance scoring.','low_cutoff':int(low.cutoff[0]),'high_cutoff':int(both.cutoff.max()),'metadata_invariance':{},'unrelated_episode_probability_invariance':{}}
for cls in [HierarchicalFrequencyModel,LagPairModel,FlipLagPairModel,SharedFlipLagPairModel]:
 m=configured(cls)
 p=m.predict(both)
 changed=replace(both,labels=(both.labels+1)%5,conditions=np.full(both.n,'scrambled'),targets=both.targets[:,::-1],reference=both.reference[:,::-1])
 result['metadata_invariance'][m.name]=float(np.max(np.abs(p-m.predict(changed))))
 result['unrelated_episode_probability_invariance'][m.name]=float(np.max(np.abs(m.predict(low)-p[:low.n])))
 if isinstance(m,FlipLagPairModel):
  fsmall=m.features(low);fbig=m.features(both)
  result[m.name+'_n_recent']={'alone':fsmall['n_recent'].tolist(),'with_unrelated_later_episode':fbig['n_recent'][:low.n].tolist(),'expected_last64':[(low.X[i,max(0,int(c)-64):int(c)]>=0).sum().item() for i,c in enumerate(low.cutoff)]}

t=np.arange(val.X.shape[1])[None,:];actual=((val.X>=0)&(t>=val.cutoff[:,None]-64)&(t<val.cutoff[:,None])).sum(1);implemented=(val.X[:,-64:]>=0).sum(1)
result['full_validation_n_recent_mismatch']={'rows':int(np.count_nonzero(actual!=implemented)),'total':val.n,'maximum_undercount':int((actual-implemented).max()),'implemented_minmax':[int(implemented.min()),int(implemented.max())],'actual_minmax':[int(actual.min()),int(actual.max())]}
train=load_split(ROOT/'main-participants-60-40/public','train');wp,sp=context_priors(train);changed=train.labels.copy();changed[0]=1 if changed[0]==0 else 0;wp2,sp2=context_priors(replace(train,labels=changed))
result['single_training_cutoff_label_changes_own_prior']={'query':train.query_ids[0],'win_prior_max_abs_delta':float(np.max(np.abs(wp2-wp))),'size_prior_max_abs_delta':float(np.max(np.abs(sp2-sp))),'own_context_before_after':[float(wp[train.context[0]]),float(wp2[train.context[0]])]}
out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
