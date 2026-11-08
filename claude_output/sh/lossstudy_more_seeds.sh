#!/usr/bin/env bash
# Extra dev-mode seeds (3, 4) for the baseline CE and size-x2 loss, to test the size2 gain with 4 matched seeds each.
# Waits until the main loss study is finished, then runs 4 fits two at a time.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"; OUT="$REPO/runs/lossstudy"; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src"
while pgrep -f "claude_output/sh/lossstudy.sh" >/dev/null; do sleep 20; done
for s in 3 4; do for v in ce:slbase size2:slsize2; do
  echo "$PY -m tdx.run --arm seqloss --variant ${v%%:*} --mode dev --set seed=$s --out $OUT/dev_${v##*:}_$s > $OUT/logs/dev_${v##*:}_$s.txt 2>&1"
done; done | xargs -P 2 -I{} bash -c "{}"
grep -H points "$OUT"/logs/dev_slbase_*.txt "$OUT"/logs/dev_slsize2_*.txt | sed 's#.*/dev_##'
