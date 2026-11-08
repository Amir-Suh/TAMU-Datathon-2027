"""Record exact reviewed branch tips, source hashes, and duplicated kit code."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
manifest=json.loads((HERE/'manifest.json').read_text())
for item in manifest:
 folder=Path(item['snapshot']);item['source_sha256']={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*.py'))}
item={'name':'main','ref':'origin/main','sha':'c7da3c55ac14b65448aa9f293601467178d3593e','scope':'README only; no model implementations'}
manifest.append(item)
kit=HERE/'amir/main-participants-60-40/slot_forecasting'
duplicates={p.name:p.read_bytes()==(ROOT/'main-participants-60-40/slot_forecasting'/p.name).read_bytes() for p in kit.glob('*.py')}
coverage={
'amir_workspace':{'models':['m1_hierarchical_frequency','m2_lag_pair','m3_lag_pair_flip','m4_lag_pair_flip_shared','m5_boosted_corrector','m6_ensemble'],'additional':['common components and selection','two saved run batches','baseline controls'], 'excluded_duplicates':duplicates},
'fredy/bakeoff-gbdt-seq':{'arms':['pooled','glm','gbdt','seq','regime'],'additional':['all declared variants','supervision permissions and scoring','blend/refit pipeline and packaging'],'limitations':['Saved pilot prediction arrays and fitted models are git-ignored and absent from branch snapshot; scores for learned arms may be log-only.']},
'main':{'models':[]},
'previous_checkout':{'scope':'Previously reviewed custom slot_forecasting suite excluded; saved Bayesian/boosted forecasts used only as fixed benchmarks.'}}
(HERE/'coverage.json').write_text(json.dumps({'remote_enumeration_complete':True,'remote_heads_verified_via_GitHub':True,'worktrees_found':1,'branches':manifest,'coverage':coverage},indent=2)+'\n')
print('Original kit identical implementations:',sum(duplicates.values()),'of',len(duplicates))
for name in ['seq','gbdt','glm','regime','pooled']:
 print('\nFREDY',name)
 print((HERE/'fredy/src/tdx/arms'/name/'__init__.py').read_text())
