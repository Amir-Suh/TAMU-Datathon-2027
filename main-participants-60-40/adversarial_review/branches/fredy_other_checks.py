"""Python 3.9 numeric checks with annotation-only compatibility; no source mutations."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,runpy,tempfile,json,types,contextlib,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BRANCH=HERE/'fredy'
KIT=HERE.parents[1]
sys.path[:0]=[str(BRANCH/'src'),str(BRANCH/'tests/glm'),str(KIT)]
sys.dont_write_bytecode=True
# Test functions are invoked directly because pytest is absent. Only its raises
# context helper is needed; production numerical dependencies are unmodified.
@contextlib.contextmanager
def raises(kind,match=None):
 try:yield
 except kind as ex:
  if match:assert re.search(match,str(ex)),str(ex)
 else:raise AssertionError('Expected '+str(kind))
pytest=types.ModuleType('pytest');pytest.raises=raises;sys.modules['pytest']=pytest
# LightGBM is absent. Feature tests do not use its engine. A later separate test
# captures engine arguments to check holdout and supervision construction only.
lgb=types.ModuleType('lightgbm');sys.modules['lightgbm']=lgb
# The installed 3.10 SciPy cannot load its _propack extension. Working 3.9
# numerics are used, postponing annotations only while loading tdx modules.
import importlib.abc, importlib.machinery, __future__
class AnnotationLoader(importlib.machinery.SourceFileLoader):
 def source_to_code(self,data,path,*,_optimize=-1):
  return compile(data,path,'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True,optimize=_optimize)
class AnnotationFinder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if not fullname.startswith('tdx'):return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  if spec and isinstance(spec.loader,importlib.machinery.SourceFileLoader):spec.loader=AnnotationLoader(fullname,spec.origin)
  return spec
sys.meta_path.insert(0,AnnotationFinder())
import numpy as np, scipy, sklearn
results=[]
def run(name,fn,*args):
 fn(*args);results.append(name);print('PASS',name,flush=True)
modules={}
with tempfile.TemporaryDirectory(prefix='fredy-other-review-') as tmp:
 tmp=Path(tmp)
 for group,file in [('glm','test_glm.py'),('factorized','test_factorized.py'),('gbdt','test_gbdt.py'),('classifier','test_classifier.py'),('specialists','test_specialists.py'),('changepoint','test_changepoint.py')]:
  folder='glm' if group in ('glm','factorized') else 'gbdt' if group=='gbdt' else 'regime'
  modules[group]=runpy.run_path(str(BRANCH/'tests'/folder/file))
 for name in ('test_shared_harness_device_and_smoke_arguments','test_feature_causality_and_cutoff_storage','test_missing_neighbours_rates_and_observed_window','test_gradient','test_planted_lag2'):
  run('glm/'+name,modules['glm'][name])
 run('glm/save_load_causal',modules['glm']['test_probabilities_save_load_and_prediction_causality'],tmp/'glm')
 for name in ('test_factorized_soft_and_hard_gradient','test_losses_provide_no_size_gradient','test_factorized_planted_lag2','test_default_dispatch_is_exactly_unchanged'):
  run('factorized/'+name,modules['factorized'][name])
 run('factorized/save_load_causal',modules['factorized']['test_factorized_probabilities_save_load_and_causality'],tmp/'factorized')
 for name in ('test_three_episode_scalar_reference','test_feature_future_perturbation','test_sparse_divergence_requires_observations_in_both_windows','test_neighbour_one_uses_each_other_machines_last_observation'):
  run('gbdt/'+name,modules['gbdt'][name])
 for name,fn in modules['classifier'].items():
  if name.startswith('test_'):run('classifier/'+name,fn)
 for name,fn in modules['specialists'].items():
  if name.startswith('test_'):run('specialists/'+name,fn)
 for name,fn in modules['changepoint'].items():
  if name.startswith('test_'):run('changepoint/'+name,fn)
 # GBDT dataset/holdout capture. This is NOT a LightGBM training test.
 from dataclasses import replace
 from tdx.data import Round
 from tdx.fit import dev_spec,final_spec
 from tdx.arms.gbdt import build
 calls=[]
 class Dataset:
  def __init__(self,data,label,weight,**kwargs):self.data,self.label,self.weight=data,label,weight
 class Booster:best_iteration=1
 def capture(params,data,**kw):calls.append((data,kw));return Booster()
 lgb.Dataset=Dataset;lgb.train=capture;lgb.early_stopping=lambda *a,**k:None
 train=modules['gbdt']['sample'](10);s=modules['gbdt']['sample'](3)
 val=replace(s,name='validation',ids=['val'+i for i in s.ids],target=np.broadcast_to(np.array([.05,.1,.15,.3,.4]),(3,8,5)).copy())
 test=replace(s,name='test',ids=['test'+i for i in s.ids],label=None,target=None)
 rnd=Round('synthetic',tmp,{'train':train,'validation':val,'test':test},60.)
 dev=build(rounds=2,w_cut=7).fit(rnd,dev_spec());dev_rows=dev.fit_rows
 calls.clear();final=build(rounds=2,w_cut=7).fit(rnd,final_spec())
 pilot,held=calls[0][0],calls[0][1]['valid_sets'][0];full=calls[-1][0]
 assert len(pilot.label)+len(held.label)==len(full.label)
 assert np.array_equal(full.weight[-val.target.size:],7*val.target.ravel())
 assert np.array_equal(full.label[-val.target.size:],np.tile(np.arange(5),24))
 assert final.fit_rows>dev_rows
 results.append('gbdt/captured_final_soft_weights_and_holdout_partition')
 print('PASS gbdt/captured_final_soft_weights_and_holdout_partition',flush=True)
 from tdx.package.blend import fit,apply
 # Fallback routing has only two distinguishable groups, not four independently
 # estimable regimes: nonshort rows are all [1/3,0,1/3,1/3].
 from types import SimpleNamespace
 rng=np.random.default_rng(713)
 q=np.array([[f'q{e}_{m}' for m in range(8)] for e in range(8)])
 target=rng.dirichlet(np.ones(5),size=(8,8));condition=np.array(['stable','short_history','drift','shared_change']*2)
 n_obs=np.full((8,8),200);n_obs[condition=='short_history']=64
 s=SimpleNamespace(ids=list(map(str,range(8))),n_obs=n_obs,query_ids=q,condition=condition,target=target)
 a=target.copy();b=target.copy();a[condition!='short_history']=.2;b[condition=='short_history']=.2
 weights=np.asarray(fit([a,b],s,np.full_like(target,.2)))
 np.testing.assert_allclose(weights[0],weights[2]);np.testing.assert_allclose(weights[0],weights[3])
 pred=apply({'weights':weights.tolist()},[a,b],s)
 assert np.max(abs(pred-target))<1e-7
 results.append('blend/fallback_has_two_identifiable_groups')
 report={'python':sys.version,'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'passed':results,'count':len(results),'limitations':['Python 3.9 with annotation-only import compatibility; native 3.10 SciPy extension failed to load.','No pytest installation: original test functions invoked directly with minimal raises context helper.','No LightGBM installation: only feature oracle/causality and dataset capture tested; no native booster training, determinism, or serialization claim.','No full public dataset training or scoring, no source mutations, no package reproduction.'],'gbdt_captured_dev_rows':dev_rows,'gbdt_captured_final_rows':final.fit_rows,'blend_fallback_weights':weights.tolist()}
 (HERE/'fredy_other_checks.json').write_text(json.dumps(report,indent=2))
 print('DONE',len(results),'checks',flush=True)
