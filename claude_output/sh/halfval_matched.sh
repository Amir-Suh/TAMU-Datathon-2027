#!/usr/bin/env bash
# Matched half-validation soft-target test (GPU, about 35 minutes with PAR=4).
#   bash claude_output/sh/halfval_matched.sh [OUTDIR]
# 2 baseline fits + (2 random splits x 2 roles x 2 weights x 2 seeds) = 18 fits; then a paired report on the held-out half.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; OUT="${1:-$REPO/runs/halfval}"; PAR="${PAR:-4}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p "$OUT"; J="$OUT/jobs.txt"; : > "$J"
for s in 1 2; do echo "$PY -m tdx.tools.halfval fit --seed $s --wcut 0 --out $OUT > $OUT/log_base_$s.txt 2>&1" >> "$J"; done
for s in 1 2; do for p in 0 1; do for r in 0 1; do for w in 5 20; do
  echo "$PY -m tdx.tools.halfval fit --seed $s --split $p --role $r --wcut $w --out $OUT > $OUT/log_s${s}_p${p}_r${r}_w${w}.txt 2>&1" >> "$J"
done; done; done; done
xargs -a "$J" -P "$PAR" -I{} bash -c "{}"
"$PY" -m tdx.tools.halfval report --dir "$OUT" | tee "$OUT/report.txt"
