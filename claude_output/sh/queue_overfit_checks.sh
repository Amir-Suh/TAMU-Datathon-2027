#!/usr/bin/env bash
# Overfitting diagnostics (see the 2026-11-08 01:00 addendum in claude_output/notes/EVAL_PROTOCOL_FROZEN.md).
#   bash claude_output/sh/queue_overfit_checks.sh      (GPU; PAR fits at a time, default 1; waits for the transformer/curves queue)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; PAR="${PAR:-1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p runs/epoch_tail runs/transd_off
while pgrep -f "queue_tfm_size2_and_curves.sh" >/dev/null; do sleep 30; done
{
  for s in 41 42 43; do for v in ce size2; do [ -f runs/epoch_tail/tail_${v}_$s.json ] || echo "$PY -m tdx.tools.epoch_curves tail --variant $v --seed $s --out runs/epoch_tail > runs/epoch_tail/log_${v}_$s.txt 2>&1"; done; done
  for s in 41 42 43; do [ -f runs/transd_off/s$s/score.json ] || echo "$PY -m tdx.run --arm seqloss --variant ce --mode dev --set transductive=false --set seed=$s --out runs/transd_off/s$s > runs/transd_off/s$s.log 2>&1"; done
} | xargs -P "$PAR" -I{} bash -c "{}"
"$PY" -m tdx.tools.epoch_curves plot --tail runs/epoch_tail --curves runs/curves --out runs/epoch_tail/curves.svg
"$PY" - <<'PYEOF'
import json
import numpy as np
on = [json.load(open(f'runs/ce_ens/s{s}/score.json'))['points'] for s in (41, 42, 43)]
off = [json.load(open(f'runs/transd_off/s{s}/score.json'))['points'] for s in (41, 42, 43)]
print('transductive ON  (train+val+test histories):', np.round(on, 2), 'mean %.2f' % np.mean(on))
print('transductive OFF (train histories only)    :', np.round(off, 2), 'mean %.2f' % np.mean(off))
print('paired difference (off - on): %+.2f' % (np.mean(off) - np.mean(on)))
PYEOF
