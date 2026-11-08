#!/usr/bin/env bash
# Read-only independent review of a stream's clone by a DIFFERENT model than the implementer.
#   bash claude_output/sh/13_review_stream.sh W1 [--dry-run]
set -euo pipefail
source "$(dirname "$0")/streams.sh"
ID="${1:?usage: 13_review_stream.sh W0..W6}"; DRY="${2:-}"
CLONE="$(clone_dir "$ID")"; [ -f "$CLONE/.tdx_clone" ] || { echo "no clone"; exit 1; }
impl="${MODEL[$ID]}"; rev="${REVIEW_MODEL:-gpt-6-astra}"; [ "$rev" = "$impl" ] && rev=gpt-6.1-sol
TAG="$CODEX_PROJECT_TAG"; lid="${ID,,}"
prompt="Independent review. Spec: claude_output/specs/${SPEC[$ID]}.md and COMMON_CONTRACT.md. Review the files this stream created (${OWNED[$ID]}). Look for: leakage of ticks >= cutoff, use of validation labels/targets in dev mode, signal used at tick level, episodes split across folds, nondeterminism, numpy/torch parity gaps, results that cannot be reproduced from the command in the report. Run the spec's acceptance commands and compare their output with the stream's report. Do NOT modify existing files; write scratch only under /tmp/tdx_${lid}_review_scratch and pass --out there. Output: a numbered list of verified findings, each with path:line, a one-sentence defect, and the command that shows it; say UNVERIFIED for anything you did not run. At most 12 findings, most severe first. No summary paragraph."
cmd=(codex-guarded --log "/tmp/${TAG}_${lid}_review.txt.guard" -- env PYTHONDONTWRITEBYTECODE=1 codex exec -m "$rev" -c 'model_reasoning_effort="high"' -s workspace-write -C "$CLONE" -o "/tmp/${TAG}_${lid}_review.txt" "$prompt")
if [ "$DRY" = "--dry-run" ]; then echo "DRY RUN review of $ID with $rev -> /tmp/${TAG}_${lid}_review.txt"; exit 0; fi
"${cmd[@]}" </dev/null > "/tmp/${TAG}_${lid}_review_stdout.log" 2>&1
echo "review: /tmp/${TAG}_${lid}_review.txt"
