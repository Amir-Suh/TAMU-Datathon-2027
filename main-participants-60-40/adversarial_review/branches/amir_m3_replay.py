"""Replay current M3 on fixed released train, compare saved validation forecasts."""
from pathlib import Path
from time import perf_counter
import sys,csv,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent/'amir'
sys.path.insert(0,str(ROOT))
from models.common.data import load_split
from models.m3_lag_pair_flip.model import FlipLagPairModel
train=load_split(ROOT/'main-participants-60-40/public','train')
val=load_split(ROOT/'main-participants-60-40/public','validation')
t=perf_counter()
with threadpool_limits(limits=2):
 model=FlipLagPairModel().fit(train)
 print('Fit completed',perf_counter()-t,flush=True)
 probs=model.predict(val)
print('Prediction completed',perf_counter()-t,flush=True)
p=ROOT/'results/2026-11-07_m1_to_m6/m3_lag_pair_flip/validation_predictions.csv'
rows={r['query_id']:[float(r[f'p_{k}']) for k in (0,1,2,5,20)] for r in csv.DictReader(p.open())}
expected=np.array([rows[q] for q in val.query_ids])
result={'scope':'Fresh default M3 train fit and validation replay; no alternate hyperparameter grid or new scoring.','max_absolute_probability_delta':float(np.max(np.abs(probs-expected))),'elapsed_seconds':perf_counter()-t,'params':model.describe(),'saved_predictions_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'python':sys.version,'numpy':np.__version__}
out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(Path(__file__).with_suffix('.npz'),query_ids=val.query_ids,probabilities=probs)
print(json.dumps({k:v for k,v in result.items() if k!='params'},indent=2),flush=True)
