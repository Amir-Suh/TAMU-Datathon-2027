#!/usr/bin/env bash
# Train a 12-seed GRU ensemble with the size-x2 loss (arm seqloss, variant size2, dev mode) and assemble it as a seqens export.
#   bash claude_output/sh/size2_ensemble.sh [OUTDIR]   (GPU; PAR fits at a time, default 2; about 5 min per fit)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; OUT="${1:-$REPO/runs/size2_ens}"; PAR="${PAR:-2}"; N="${N:-12}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p "$OUT"
for s in $(seq 41 $((40 + N))); do [ -f "$OUT/s$s/model/model.npz" ] && continue
  echo "$PY -m tdx.run --arm seqloss --variant size2 --mode dev --set seed=$s --out $OUT/s$s > $OUT/s$s.log 2>&1"; done | xargs -P "$PAR" -I{} bash -c "{}"
M=(); for s in $(seq 41 $((40 + N))); do [ -d "$OUT/s$s/model" ] && M+=("$OUT/s$s/model"); done
"$PY" -m tdx.tools.make_ensemble --arm seqens --variant size2_ens --runs "$OUT/export" --members "${M[@]}"
grep -h points "$OUT"/s*.log | cut -c1-80
