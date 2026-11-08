#!/usr/bin/env bash
# One command for round two (adaptation): train GRU + transformer seeds on the NEW data, blend with the FROZEN candidate-4 (B)
# weights from the main-round cross-validation (no selection on the new validation unless BLEND_SELECT=1), build the entry,
# check it and reproduce it offline. Budget: the whole round is 90 minutes.
# Recipe = candidate 4 (B): m3 + 12 size-x2 GRUs (arm seqloss, variant size2, seeds 41..52) + 8 transformers (tfm2 d128l3, seeds 21..28).
#
#   bash claude_output/sh/round2.sh <new public dir>                 # real run (GPU)
#   ROUND=main N_GRU=2 N_TFM=2 EP_GRU=3 EP_TFM=3 bash claude_output/sh/round2.sh <main public dir>   # plumbing rehearsal
#
# Knobs (env): ROUND (adaptation), N_GRU (12), N_TFM (8), EP_GRU (100), EP_TFM (100), PAR (5 parallel trainings),
#              OUT (runs/r2_<time>), SKIP_REPRO (0), BLEND_SELECT (0; 1 = pick the best preset on the new validation), TDX_PY.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$(readlink -f "${1:?new public dir (the folder that holds metadata.json, train/, validation/, test/)}")"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"
ROUND="${ROUND:-adaptation}"; N_GRU="${N_GRU:-12}"; N_TFM="${N_TFM:-8}"; BLEND_SELECT="${BLEND_SELECT:-0}"; PAR="${PAR:-5}"
EP_GRU="${EP_GRU:-100}"; EP_TFM="${EP_TFM:-100}"; SKIP_REPRO="${SKIP_REPRO:-0}"
OUT="${OUT:-$REPO/runs/r2_$(date +%H%M)}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"
cd "$REPO"; mkdir -p "$OUT"
T0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - T0) / 60 )) min] $*"; }

META_ROUND="$("$PY" -c "import json,sys;print(json.load(open('$DATA/metadata.json'))['round'])")"
[ "$META_ROUND" = "$ROUND" ] || { echo "metadata round is '$META_ROUND' but ROUND='$ROUND' (set ROUND=$META_ROUND for a rehearsal)"; exit 1; }
stamp "round=$ROUND data=$DATA out=$OUT  GRU x$N_GRU ($EP_GRU ep), transformer x$N_TFM ($EP_TFM ep), $PAR parallel"

# 1) trainings (dev mode: histories only, no validation targets), all in one parallel pool
JOBS="$OUT/jobs.txt"; : > "$JOBS"
for s in $(seq 1 "$N_GRU"); do
  echo "$PY -m tdx.run --arm seqloss --variant size2 --mode dev --round $ROUND --data $DATA --set seed=$((40+s)) --set epochs=$EP_GRU --out $OUT/seq_$s > $OUT/seq_$s.log 2>&1" >> "$JOBS"
done
for s in $(seq 1 "$N_TFM"); do
  echo "$PY -m tdx.run --arm tfm2 --variant d128l3 --mode dev --round $ROUND --data $DATA --set seed=$((20+s)) --set epochs=$EP_TFM --out $OUT/tfm_$s > $OUT/tfm_$s.log 2>&1" >> "$JOBS"
done
# tfm2 first: it is the slower model
tac "$JOBS" > "$JOBS.rev"; mv "$JOBS.rev" "$JOBS"
xargs -a "$JOBS" -P "$PAR" -I{} bash -c "{}"
for s in $(seq 1 "$N_GRU"); do grep -q "points" "$OUT/seq_$s.log" || echo "WARN seq_$s did not report a score: $(tail -n 2 $OUT/seq_$s.log | tr '\n' ' ')"; done
for s in $(seq 1 "$N_TFM"); do grep -q "points" "$OUT/tfm_$s.log" || echo "WARN tfm_$s did not report a score: $(tail -n 2 $OUT/tfm_$s.log | tr '\n' ' ')"; done
stamp "trainings done"; grep -h "points" "$OUT"/seq_*.log "$OUT"/tfm_*.log | cut -c1-90

# 2) assemble the two ensembles
GRU_M=(); for s in $(seq 1 "$N_GRU"); do [ -d "$OUT/seq_$s/model" ] && GRU_M+=("$OUT/seq_$s/model"); done
TFM_M=(); for s in $(seq 1 "$N_TFM"); do [ -d "$OUT/tfm_$s/model" ] && TFM_M+=("$OUT/tfm_$s/model"); done
"$PY" -m tdx.tools.make_ensemble --arm seqens --variant r2gru --runs "$OUT/ens" --members "${GRU_M[@]}"
"$PY" -m tdx.tools.make_ensemble --arm tfmens --variant r2tfm --runs "$OUT/ens" --members "${TFM_M[@]}"
stamp "ensembles assembled (${#GRU_M[@]} GRU, ${#TFM_M[@]} transformer)"

# 3) blend: frozen candidate-4 (B) main-CV weights by default; every preset is still scored on the new validation as a diagnostic
"$PY" -m tdx.tools.round2_blend --round "$ROUND" --data "$DATA" --gru "$OUT/ens/seqens/r2gru/model" --tfm "$OUT/ens/tfmens/r2tfm/model" --out "$OUT/blend.json" $([ "$BLEND_SELECT" = "1" ] && echo --select)
stamp "blend chosen"

# 4) build the entry (also runs predict.py once) and check the CSV
"$PY" -m tdx.package.build --arms m3flip/m3,seqens/r2gru,tfmens/r2tfm --blend "$OUT/blend.json" --round "$ROUND" \
  --data "$DATA" --runs "$OUT/ens" --out "$OUT/entry" --zip "$OUT/entry_$ROUND.zip" --csv "$OUT/submission.csv" --allow-dev
stamp "entry built"
"$PY" kit/check_submission.py "$OUT/submission.csv" --data "$DATA" | head -3

# 5) fresh-folder offline reproduction (two runs; the longest check)
if [ "$SKIP_REPRO" = "1" ]; then echo "SKIP_REPRO=1: reproduction NOT run"; else
  bash claude_output/sh/repro_check.sh "$OUT/entry_$ROUND.zip" "$DATA" "$OUT/submission.csv"
  stamp "reproduction OK"
fi
echo "UPLOAD: $OUT/entry_$ROUND.zip  and  $OUT/submission.csv"
sha256sum "$OUT/entry_$ROUND.zip" "$OUT/submission.csv"
stamp "TOTAL"
