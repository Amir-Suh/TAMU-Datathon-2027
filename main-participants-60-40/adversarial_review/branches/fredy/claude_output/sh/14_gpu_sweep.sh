#!/usr/bin/env bash
# Run an arm's full sweep on the RTX 5080 OUTSIDE the Codex sandbox (the sandbox hides /dev/nvidia*). No model tokens used.
#   bash claude_output/sh/14_gpu_sweep.sh [arm=seq] [mode=dev] [extra sweep args...]
# Run it in the main checkout after 12_collect_stream.sh W3. Log: /tmp/tdx_gpu_<arm>.log ; results in runs/<arm>/ and runs/LOG.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM="${1:-seq}"; MODE="${2:-dev}"; shift 2 2>/dev/null || shift $# || true
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA not available on host"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
"$PY" -m tdx.sweep --arm "$ARM" --mode "$MODE" --device cuda "$@" 2>&1 | tee "/tmp/tdx_gpu_${ARM}.log"
"$PY" -m tdx.tools.compare runs/ --baseline pooled_recent | tail -25
