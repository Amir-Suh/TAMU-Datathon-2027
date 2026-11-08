#!/usr/bin/env bash
# (1) 8 transformers with the size-x2 loss on the SAME seeds 21-28 as the plain ensemble, exported as a tfmens ensemble;
# (2) re-run GRU seeds 41-43 (CE and size x2) with per-epoch train loss + checkpoints every 10 epochs for loss/validation curves.
#   bash claude_output/sh/queue_tfm_size2_and_curves.sh      (GPU; PAR fits at a time, default 2)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; PAR="${PAR:-2}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p runs/tfm_s2 runs/curves
{
  for s in $(seq 21 28); do [ -f runs/tfm_s2/s$s/model/model.npz ] || echo "$PY -m tdx.run --arm tfm2 --variant d128l3_size2 --mode dev --set seed=$s --out runs/tfm_s2/s$s > runs/tfm_s2/s$s.log 2>&1"; done
  for s in 41 42 43; do for v in ce size2; do [ -f runs/curves/${v}_$s/model/history.json ] || echo "$PY -m tdx.run --arm seqloss --variant $v --mode dev --set seed=$s --set ckpt_every=10 --out runs/curves/${v}_$s > runs/curves/${v}_$s.log 2>&1"; done; done
} | xargs -P "$PAR" -I{} bash -c "{}"
M=(); for s in $(seq 21 28); do [ -d runs/tfm_s2/s$s/model ] && M+=("runs/tfm_s2/s$s/model"); done
"$PY" -m tdx.tools.make_ensemble --arm tfmens --variant tfm_s2_8 --runs runs/tfm_s2/export --members "${M[@]}"
grep -h points runs/tfm_s2/s*.log runs/curves/*.log | cut -c1-80
