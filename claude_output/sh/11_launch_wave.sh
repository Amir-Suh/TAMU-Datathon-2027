#!/usr/bin/env bash
# Launch several streams in parallel, one tmux window each.
#   bash claude_output/sh/11_launch_wave.sh w0            # W0 harness only
#   bash claude_output/sh/11_launch_wave.sh pilots        # W1 W2 W3 W4 W6 (after W0 is collected)
#   bash claude_output/sh/11_launch_wave.sh stack         # W5
#   bash claude_output/sh/11_launch_wave.sh pilots --dry-run
set -euo pipefail
source "$(dirname "$0")/streams.sh"
WAVE="${1:?usage: 11_launch_wave.sh w0|pilots|seq|stack [--dry-run]}"; EXTRA="${2:-}"
case "$WAVE" in w0) STREAMS="W0";; pilots) STREAMS="W1 W2 W4 W6";; seq) STREAMS="W3";; stack) STREAMS="W5";; *) echo "unknown wave"; exit 1;; esac
S="tdx-$WAVE"
if [ "$EXTRA" = "--dry-run" ]; then for id in $STREAMS; do bash "$MAIN/claude_output/sh/10_launch_stream.sh" "$id" --dry-run; done; exit 0; fi
tmux has-session -t "$S" 2>/dev/null && { echo "tmux session $S exists: tmux attach -t $S"; exit 1; }
first=1
for id in $STREAMS; do
  cmd="bash $MAIN/claude_output/sh/10_launch_stream.sh $id --fresh; echo; echo '[$id done - press enter]'; read"
  if [ $first = 1 ]; then tmux new-session -d -s "$S" -n "$id" "$cmd"; first=0; else tmux new-window -t "$S" -n "$id" "$cmd"; fi
done
tmux new-window -t "$S" -n watch "watch -n 30 'codex-guard --status | head -4; echo; tail -n 12 $MAIN/runs/LOG.md 2>/dev/null; for f in /tmp/tdx_w*_stdout.log; do echo; echo \$f; tail -n 2 \$f; done'"
echo "started tmux session $S ($STREAMS). Attach: tmux attach -t $S   Detach: ctrl-b d"
echo "Each stream's result is in its clone; merge with: bash claude_output/sh/12_collect_stream.sh <W>"
