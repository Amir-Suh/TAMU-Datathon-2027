#!/usr/bin/env bash
# Read-only status of the overnight queue (overnight_chain.sh -> queue_moegate_cv.sh -> queue_epoch150.sh).  bash claude_output/sh/overnight_status.sh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"; date +%T
echo "== chain =="; cat runs/overnight_chain.log 2>/dev/null
echo "== rehearsal (stage stamps; TOTAL line = full-size wall time vs the 90-min budget) =="; grep -E "^\[[0-9]+ min\]|WARN|REPRO|UPLOAD|Error|Traceback" runs/r2_rehearsal.log 2>/dev/null | tail -12
echo "== MoE runs: finished model dirs / 16, failed logs =="; ls -d runs/moe_ens/*/model 2>/dev/null | wc -l; grep -l "Traceback" runs/moe_ens/*.log 2>/dev/null | wc -l
[ -f claude_output/notes/cv_blend_moegate_results.txt ] && cut -c1-160 claude_output/notes/cv_blend_moegate_results.txt | head -14
echo "== epoch150 tail curves (files / 3) =="; ls runs/epoch_tail150/tail_size2_*.json 2>/dev/null | wc -l; cat runs/epoch150_queue.log 2>/dev/null
echo "== python jobs =="; ps -eo args | awk '$1 ~ /python$/' | cut -c1-120 | head -8
