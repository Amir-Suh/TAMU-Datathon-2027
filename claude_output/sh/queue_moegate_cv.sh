#!/usr/bin/env bash
# moegate K=3 (variant k3_ref) and K=4 without condition loss (variant k4_nocond), 8 seeds each (41-48, from the frozen seed range),
# dev mode; then average their validation forecasts into weights/val_preds/moe_k{3,4}_8.npy and run the frozen cross-validated blend test.
#   bash claude_output/sh/queue_moegate_cv.sh         (GPU; waits for the transformer/curves queue; PAR fits at a time, default 2)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; PAR="${PAR:-2}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p runs/moe_ens
while pgrep -f "queue_tfm_size2_and_curves.sh" >/dev/null; do sleep 30; done
for s in $(seq 41 48); do for v in k3_ref:k3 k4_nocond:k4; do
  [ -f runs/moe_ens/${v##*:}_$s/model/model.npz ] || echo "$PY -m tdx.run --arm moegate4 --variant ${v%%:*} --mode dev --set seed=$s --out runs/moe_ens/${v##*:}_$s > runs/moe_ens/${v##*:}_$s.log 2>&1"
done; done | xargs -P "$PAR" -I{} bash -c "{}"
"$PY" - <<'PYEOF'
import numpy as np
from pathlib import Path
from tdx.data import load_round
from tdx.arms.moegate4 import MoeGate4Arm
v = load_round('data/main', answers=False).splits['validation']
for k in ('k3', 'k4'):
    ps = [MoeGate4Arm.load(d).predict(v) for d in sorted(Path('runs/moe_ens').glob(f'{k}_*/model'))]
    print(k, len(ps), 'members')
    np.save(f'weights/val_preds/moe_{k}_8.npy', np.mean(ps, 0))
PYEOF
"$PY" -m tdx.tools.cv_blend --spec claude_output/notes/frozen_addendum2_moegate.json | tee claude_output/notes/cv_blend_moegate_results.txt
