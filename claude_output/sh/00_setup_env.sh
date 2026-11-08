#!/usr/bin/env bash
# One-time: build the shared venv OUTSIDE every sandbox (Codex has no network). ~5-10 min, torch is the big download.
# Run:  ! bash claude_output/sh/00_setup_env.sh
set -euo pipefail
VENV="${TDX_VENV:-$HOME/.venvs/tdx}"
command -v uv >/dev/null || { echo "uv not found"; exit 1; }
# Python 3.13 matches the organizer image (README section 7). uv fetches it if missing.
uv venv --python 3.13 "$VENV"
# CPU-side science stack. Versions are frozen to requirements-dev.lock afterwards so the packaging step can pin exact versions.
uv pip install --python "$VENV/bin/python" numpy scipy scikit-learn lightgbm pytest pandas
# torch for the RTX 5080 (Blackwell, sm_120) needs a CUDA 12.8+ wheel.
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" - <<'PY'
import numpy, scipy, sklearn, lightgbm, torch, sys
print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__,
      "sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__, "torch", torch.__version__,
      "cuda", torch.cuda.is_available())
PY
cd "$(dirname "$0")/../.."
uv pip freeze --python "$VENV/bin/python" > requirements-dev.lock
echo "wrote requirements-dev.lock ($(wc -l < requirements-dev.lock) packages)"
