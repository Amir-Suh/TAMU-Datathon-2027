#!/usr/bin/env bash
# Fresh Python 3.13 venv; exact requirements only; offline cached wheels.
# TDX_WHEELHOUSE may name a folder of original wheels; otherwise uv cache is used.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ZIP="$(readlink -f "${1:?entry zip}")"
DATA="$(readlink -f "${2:?public dir}")"
UPLOADED="$(readlink -f "${3:?uploaded csv}")"
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src"
cd "$REPO"
exec "$PY" -m tdx.package.repro "$ZIP" "$DATA" "$UPLOADED"
