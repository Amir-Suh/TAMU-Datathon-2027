#!/usr/bin/env bash
# Follow-ups from the 2026-11-08 02:00 addendum in claude_output/notes/EVAL_PROTOCOL_FROZEN.md (GPU; PAR fits at a time, default 2).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; PAR="${PAR:-2}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p runs/disent runs/ce80 runs/moe_ens
stage() { xargs -P "$PAR" -I{} bash -c "{}"; }
# 1) disentangle the transductive gap (GRU CE seeds 41-43)
{ for s in 41 42 43; do for v in B:train,test D:train,validation; do n=${v%%:*}; ts=${v##*:}
  [ -f runs/disent/${n}_$s/score.json ] || echo "$PY -m tdx.run --arm seqloss --variant ce --mode dev --set transductive=true --set tick_splits=$ts --set seed=$s --out runs/disent/${n}_$s > runs/disent/${n}_$s.log 2>&1"; done; done; } | stage
# 2) plain CE at 80 epochs, seeds 41-52, then average forecasts for the frozen blend test
{ for s in $(seq 41 52); do [ -f runs/ce80/s$s/score.json ] || echo "$PY -m tdx.run --arm seqloss --variant ce --mode dev --set epochs=80 --set seed=$s --out runs/ce80/s$s > runs/ce80/s$s.log 2>&1"; done; } | stage
"$PY" - <<'PYEOF'
import numpy as np
from pathlib import Path
from tdx.data import load_round
from tdx.arms.seqens.arm import SeqEnsArm
from tdx.tools.make_ensemble import main as mk
mk(['--arm', 'seqens', '--variant', 'ce80', '--runs', 'runs/ce80/export', '--members'] + [f'runs/ce80/s{s}/model' for s in range(41, 53)])
v = load_round('data/main', answers=False).splits['validation']
np.save('weights/val_preds/gru_ce80_41_52.npy', SeqEnsArm.load('runs/ce80/export/seqens/ce80/model').predict(v))
PYEOF
"$PY" -m tdx.tools.cv_blend --spec claude_output/notes/frozen_addendum3_ce80.json | tee claude_output/notes/cv_blend_ce80_results.txt
# 3) transductive-off for the other members
{ for s in 21 22 23; do [ -f runs/transd_off_tfm/s$s/score.json ] || echo "mkdir -p runs/transd_off_tfm; $PY -m tdx.run --arm tfm2 --variant d128l3 --mode dev --set tick_splits=train --set seed=$s --out runs/transd_off_tfm/s$s > runs/transd_off_tfm/s$s.log 2>&1"; done
  for s in 41 42 43; do [ -f runs/transd_off_size2/s$s/score.json ] || echo "mkdir -p runs/transd_off_size2; $PY -m tdx.run --arm seqloss --variant size2 --mode dev --set tick_splits=train --set seed=$s --out runs/transd_off_size2/s$s > runs/transd_off_size2/s$s.log 2>&1"; done; } | stage
# 4) MoE arms with the causal flag, frozen addendum 2
{ for s in $(seq 41 48); do for v in k3_ref:k3 k4_nocond:k4; do n=${v##*:}
  [ -f runs/moe_ens/${n}_$s/model/model.npz ] || echo "$PY -m tdx.tools.dummy 2>/dev/null; $PY -m tdx.run --arm moegate4 --variant ${v%%:*} --mode dev --set seed=$s --out runs/moe_ens/${n}_$s > runs/moe_ens/${n}_$s.log 2>&1"; done; done; } | sed 's#\$PY -m tdx.tools.dummy 2>/dev/null; ##' | stage
"$PY" - <<'PYEOF'
import numpy as np
from pathlib import Path
from tdx.data import load_round
from tdx.arms.moegate4 import MoeGate4Arm
v = load_round('data/main', answers=False).splits['validation']
for k in ('k3', 'k4'):
    ps = [MoeGate4Arm.load(d).predict(v) for d in sorted(Path('runs/moe_ens').glob(f'{k}_*/model'))]
    print(k, len(ps), 'members'); np.save(f'weights/val_preds/moe_{k}_8.npy', np.mean(ps, 0))
PYEOF
"$PY" -m tdx.tools.cv_blend --spec claude_output/notes/frozen_addendum2_moegate.json | tee claude_output/notes/cv_blend_moegate_results.txt
