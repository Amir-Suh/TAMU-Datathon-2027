#!/usr/bin/env bash
# Overnight chain (user-approved 2026-11-08): (1) full-size round-two rehearsal on the MAIN data (timing + plumbing; its main-validation
# scores are NOT evidence: main validation already steered recipe and blend choices), then (2) leak test for the moegate4 variants,
# then (3) the frozen addendum-2 MoE test (K=3 k3_ref, K=4 k4_nocond, plain CE, seeds 41-48, vs the extra-GRU control).
#   bash claude_output/sh/overnight_chain.sh      (GPU; run once; log: runs/overnight_chain.log)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"
DATA=/home/ganfwe/Downloads/main-participants-60-40/public
echo "[$(date +%T)] start rehearsal"
ROUND=main OUT="$REPO/runs/r2_rehearsal" bash claude_output/sh/round2.sh "$DATA" > runs/r2_rehearsal.log 2>&1
echo "[$(date +%T)] rehearsal exit=$? (see runs/r2_rehearsal.log)"
for v in k3_ref k4_nocond; do
  "$PY" -m tdx.tools.leak_test --arm moegate4 --variant $v > runs/leak_moegate4_$v.log 2>&1 || { echo "[$(date +%T)] LEAK TEST FAILED for $v: MoE queue NOT started"; exit 1; }
  echo "[$(date +%T)] leak test passed for $v"
done
bash claude_output/sh/queue_moegate_cv.sh > runs/moe_queue.log 2>&1
echo "[$(date +%T)] MoE queue exit=$? (results: claude_output/notes/cv_blend_moegate_results.txt)"
