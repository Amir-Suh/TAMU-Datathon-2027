#!/usr/bin/env bash
# Refit the main-chosen variants against the new adaptation kit, then verify ZIP.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NEW="$(readlink -f "${1:?new public dir}")"
shift
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src"
cd "$REPO"
exec "$PY" -m tdx.pipeline --round adaptation --data "$NEW" "$@"
