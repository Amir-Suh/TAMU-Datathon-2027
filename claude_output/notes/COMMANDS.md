# Commands worth knowing (safety: R = read-only, W = writes files, $ = expensive/GPU)
| Command | What it does | Tag |
|---|---|---|
| `PYTHONPATH=src ~/.venvs/tdx/bin/python -m tdx.tools.cv_blend [--emit CAND --emit-out blend.json]` | Frozen repeated grouped-CV blend test on cached forecasts; --emit writes a candidate's fold-averaged weights | R (W with --emit) |
| `python -m tdx.tools.leak_test --arm A --variant V --set epochs=1 --set h=16` | Cutoff + step-causality leak test (perturbs values/adds/removes/moves ticks) with a tiny fit | W (tiny fit) |
| `python -m tdx.tools.tail_eval predict/report ...` | Held-out-tail evaluation (last ticks hidden, neutral signal) with paired bootstrap | $ (predict), R (report) |
| `bash claude_output/sh/repro_check.sh ZIP PUBLIC CSV` | Fresh Python 3.13 venv, offline, 2 cores reproduction of an entry (run twice, compares to CSV) | W, ~10 min |
| `python -m tdx.package.build --arms ... --blend ... --allow-dev ...` | Build an entry ZIP + CSV from exported ensembles | W, ~5 min |
| `bash claude_output/sh/queue_disentangle.sh` | Follow-up GPU queue: CE 80 epochs, transductive-off for other members, causal-flag MoE | $ GPU, ~2 h |
| `bash claude_output/sh/round2.sh <public dir>` | One-command round-two refit (train, ensembles, blend preset, build, check, reproduce) | $ GPU, untested at full size |
| `codex-guard --status \| head -4` | Codex usage window and whether the guard sees a run | R |
| `ps -eo args \| awk '$1 ~ /python$/'` | What python jobs are running | R |
| `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader` | GPU load and power | R |
