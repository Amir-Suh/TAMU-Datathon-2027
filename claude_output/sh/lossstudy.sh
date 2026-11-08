#!/usr/bin/env bash
# Loss study for the GRU (arm `seqloss`): baseline CE vs Brier, CE+Brier, win/size reweighting (lam 2, 4), then recency, history-length
# and payout-value weights. Leak-free tail (hold 30, seeds 1-3, scored with tdx.tools.tail_eval) + dev-mode validation (seeds 1-2).
#   bash claude_output/sh/lossstudy.sh [OUTDIR]      (GPU, about 40 fits; re-running skips finished outputs)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; OUT="${1:-$REPO/runs/lossstudy}"; PAR="${PAR:-4}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"; mkdir -p "$OUT/tail" "$OUT/logs"; J="$OUT/jobs.txt"; : > "$J"
declare -A VAR=([slbase]=ce [slbrier]=brier [slcebrier]=ce_brier2 [slsize2]=size2 [slsize4]=size4 [slrec]=rec64 [sllen]=lenw [slval]=valw)
MAIN="slbase slbrier slcebrier slsize2 slsize4"; EXTRA="slrec sllen slval"
add_tail() { for m in $1; do for s in 1 2 3; do [ -f "$OUT/tail/tail_${m}_${s}_h30.npz" ] && continue; echo "$PY -m tdx.tools.tail_eval predict --model $m --seed $s --hold 30 --out $OUT/tail > $OUT/logs/tail_${m}_$s.txt 2>&1" >> "$J"; done; done; }
add_dev()  { for m in $1; do for s in 1 2; do grep -q points "$OUT/logs/dev_${m}_$s.txt" 2>/dev/null && continue; echo "$PY -m tdx.run --arm seqloss --variant ${VAR[$m]} --mode dev --set seed=$s --out $OUT/dev_${m}_$s > $OUT/logs/dev_${m}_$s.txt 2>&1" >> "$J"; done; done; }
add_tail "$MAIN"; add_dev "$MAIN"; add_tail "$EXTRA"; add_dev "$EXTRA"
wc -l < "$J"
xargs -a "$J" -P "$PAR" -I{} bash -c "{}"
ALL="slbase,slbrier,slcebrier,slsize2,slsize4,slrec,sllen,slval"
"$PY" -m tdx.tools.tail_eval report --dir "$OUT/tail" --models "$ALL" --holds 30 --baseline slbase | tee "$OUT/tail_report.txt"
echo "--- dev-mode validation points (seeds 1-2)"; grep -h "points" "$OUT"/logs/dev_*.txt | cut -c1-100 | tee "$OUT/dev_points.txt"
