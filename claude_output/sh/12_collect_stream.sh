#!/usr/bin/env bash
# Copy a finished stream's owned paths from its clone back into the main checkout (no git needed), merge runs/LOG.md.
#   bash claude_output/sh/12_collect_stream.sh W1 [--dry-run]
set -euo pipefail
source "$(dirname "$0")/streams.sh"
ID="${1:?usage: 12_collect_stream.sh W0..W6 [--dry-run]}"; DRY=""; [ "${2:-}" = "--dry-run" ] && DRY="--dry-run"
CLONE="$(clone_dir "$ID")"; [ -f "$CLONE/.tdx_clone" ] || { echo "no clone at $CLONE"; exit 1; }
for p in ${OWNED[$ID]}; do
  [ -e "$CLONE/$p" ] || { echo "skip (absent in clone): $p"; continue; }
  if [ -d "$CLONE/$p" ]; then mkdir -p "$MAIN/$p"; rsync -a $DRY --itemize-changes --exclude '__pycache__' "$CLONE/$p/" "$MAIN/$p/" | sed 's/^/  /'
  else mkdir -p "$(dirname "$MAIN/$p")"; rsync -a $DRY --itemize-changes "$CLONE/$p" "$MAIN/$p" | sed 's/^/  /'; fi
done
if [ -f "$CLONE/runs/LOG.md" ] && [ -z "$DRY" ]; then
  touch "$MAIN/runs/LOG.md"
  grep -vxFf "$MAIN/runs/LOG.md" "$CLONE/runs/LOG.md" >> "$MAIN/runs/LOG.md" || true
fi
echo "collected $ID. Changed paths outside the owned list (should be none):"
( cd "$CLONE" && rsync -a --dry-run --itemize-changes --exclude '.git' --exclude '__pycache__' --exclude '.tdx_clone' --exclude 'data' --exclude 'outputs' --exclude '.cache' --exclude 'runs/LOG.md' ./ "$MAIN/" | grep -v '^\.' | awk '{print $2}' | while read -r f; do
  ok=0; for p in ${OWNED[$ID]}; do case "$f" in "$p"|"$p"/*) ok=1;; esac; done; [ $ok = 1 ] || echo "  OUTSIDE: $f"; done ) || true
