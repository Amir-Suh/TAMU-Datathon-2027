#!/usr/bin/env bash
# Diagnostic (user-approved 2026-11-08): does the size x2 GRU keep improving past 100 epochs? Tail-hidden refits (last 30 ticks hidden,
# no validation answers), seeds 41-43, 150 epochs, checkpoints every 10; waits for overnight_chain.sh to finish.
# Reading rule (reused from addendum 3, not a new cut-off): the epoch count is changed only if the best checkpoint ABOVE 100 beats
# epoch 100 by >= 0.003 tail log-loss in all three seeds; that would then be a new frozen decision with a matched-seed CV re-evaluation.
# Constant learning rate and deterministic seeds, so the epoch-100 checkpoint should equal runs/epoch_tail's size2 epoch 100 (plumbing check).
#   bash claude_output/sh/queue_epoch150.sh      (GPU; output runs/epoch_tail150/)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"
mkdir -p runs/epoch_tail150
for s in 41 42 43; do [ -f runs/epoch_tail150/tail_size2_$s.json ] || echo "EPOCHS=150 $PY -m tdx.tools.epoch_curves tail --variant size2 --seed $s --out runs/epoch_tail150 > runs/epoch_tail150/log_size2_$s.txt 2>&1"; done | xargs -P 2 -I{} bash -c "{}"
echo "[$(date +%T)] epoch150 done"
