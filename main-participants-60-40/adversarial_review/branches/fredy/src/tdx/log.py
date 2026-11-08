"""Append-only experiment ledger."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
from .paths import ROOT

CONDITIONS = ('stable', 'short_history', 'drift', 'shared_change')
HEADER = '| time | arm | variant | mode | points | stable | short | drift | shared | seconds |\n|---|---|---|---|---|---|---|---|---|---|\n'

def append(arm, variant, mode, result, seconds, path=None):
    path = Path(path or ROOT / 'runs' / 'LOG.md')
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [f'{result["points"]:.6f}', *(f'{result["by_condition"][c]:.6f}' for c in CONDITIONS)] if result else ['—'] * 5
    row = [datetime.now(timezone.utc).isoformat(timespec='seconds'), arm, variant, mode, *vals, f'{seconds:.3f}']
    with path.open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(HEADER)
        f.write('| ' + ' | '.join(row) + ' |\n')
