#!/usr/bin/env bash
# Launch ONE Codex stream in its own copy of the working tree, under the spending guard.
#   bash claude_output/sh/10_launch_stream.sh W1 [--dry-run] [--fresh]
# Override per run: CODEX_MODEL=gpt-6-astra CODEX_REASONING_EFFORT=high CODEX_SANDBOX=workspace-write
# The copy is a plain rsync (no git needed): ../TAMU-Datathon-2027-w1 . Uncommitted specs/launchers are included.
set -euo pipefail
source "$(dirname "$0")/streams.sh"
ID="${1:?usage: 10_launch_stream.sh W0..W6 [--dry-run] [--fresh]}"; shift || true
DRY=0; FRESH=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --fresh) FRESH=1;; *) echo "unknown arg $a"; exit 1;; esac; done
[ -n "${SPEC[$ID]:-}" ] || { echo "unknown stream $ID"; exit 1; }
TAG="$CODEX_PROJECT_TAG"; lid="${ID,,}"
model="${CODEX_MODEL:-${MODEL[$ID]}}"; effort="${CODEX_REASONING_EFFORT:-${EFFORT[$ID]}}"; sandbox="${CODEX_SANDBOX:-workspace-write}"
spec="$MAIN/claude_output/specs/${SPEC[$ID]}.md"; contract="$MAIN/claude_output/specs/COMMON_CONTRACT.md"
[ -f "$spec" ] && [ -f "$contract" ] || { echo "missing spec or contract"; exit 1; }
CLONE="$(clone_dir "$ID")"

# In --dry-run, missing prerequisites are warnings so the commands can be inspected before setup.
need() { if [ "$DRY" = 1 ]; then echo "WARN: $1"; else echo "ERROR: $1"; exit 1; fi; }
command -v codex-guarded >/dev/null || { echo "codex-guarded not on PATH"; exit 1; }
codex debug models 2>/dev/null | python3 -c "import json,sys; s={m['slug'] for m in json.load(sys.stdin)['models']}; sys.exit(0 if '$model' in s else 1)" \
  || { echo "model $model not in 'codex debug models'"; exit 1; }
[ -x "$PY" ] || need "venv missing: run 00_setup_env.sh"
[ -e "$MAIN/data/main/metadata.json" ] || need "data not linked: run 01_link_data.sh"
if [ "$ID" != W0 ] && [ ! -f "$MAIN/src/tdx/run.py" ]; then need "W0 not merged (src/tdx/run.py missing): run W0, then 12_collect_stream.sh W0"; fi

echo "== guard status =="; codex-guard --status | head -4
echo "== $ID: model=$model effort=$effort sandbox=$sandbox clone=$CLONE =="
echo "   report : /tmp/${TAG}_${lid}_orchestrator.txt (written when the stream ends)"
echo "   log    : tail -f /tmp/${TAG}_${lid}_stdout.log"

if [ -e "$CLONE" ]; then
  if [ "$FRESH" = 1 ] && [ -f "$CLONE/.tdx_clone" ]; then [ "$DRY" = 1 ] || rm -rf "$CLONE"
  else echo "clone exists ($CLONE). Use --fresh to recreate it (only deletes folders carrying the .tdx_clone marker)."; [ "$DRY" = 1 ] || exit 1; fi
fi
if [ "$DRY" = 0 ]; then
  rsync -a --exclude '.git' --exclude 'outputs' --exclude '*.zip' --exclude '__pycache__' --exclude '.venv' "$MAIN/" "$CLONE/"
  touch "$CLONE/.tdx_clone"
fi

prompt="$(cat "$contract" "$spec")"
cmd=(codex-guarded --log "/tmp/${TAG}_${lid}_orchestrator.txt.guard" -- env PYTHONDONTWRITEBYTECODE=1
     codex exec --skip-git-repo-check -m "$model" -c "model_reasoning_effort=\"$effort\"" -s "$sandbox" -C "$CLONE"
     -o "/tmp/${TAG}_${lid}_orchestrator.txt" "$prompt")
if [ "$DRY" = 1 ]; then
  echo "DRY RUN. Command (prompt = $(printf %s "$prompt" | wc -c) chars, contract + $(basename "$spec")):"
  printf '  %q' "${cmd[@]:0:${#cmd[@]}-1}"; echo ' "<prompt>" </dev/null'
  exit 0
fi
# </dev/null: codex exec otherwise waits on stdin.
"${cmd[@]}" </dev/null > "/tmp/${TAG}_${lid}_stdout.log" 2>&1
echo "stream $ID finished; report: /tmp/${TAG}_${lid}_orchestrator.txt ; next: bash claude_output/sh/12_collect_stream.sh $ID"
