# Antigravity job (resume): finish and score moegate4 (the laptop reset; /tmp was wiped). Budget 75 minutes, no time gate.

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; `PYTHONPATH=src`). Run `date` and write it to /tmp/agy_moegate4/start3.txt (mkdir -p). Use at most ONE GPU job at a time (the laptop is on battery power and cannot sustain more; another job also trains).

## State
- `claude_output/agy/moegate4.md` is the full task description (question, rules, variants, tests, scoring, final report format): follow it. Ignore its "do not start before 21:55" gate.
- Already in the repo from earlier sessions: `src/tdx/arms/moegate4/{__init__,arm.py}` and `tests/moegate4/test_moegate4.py` (the leak test `python -m tdx.tools.leak_test --arm moegate4 --variant k4_cond03` passed earlier, but it took ~20 minutes on a busy GPU: run it with a 25-minute timeout and do not wait on it before starting the scoring).
- LOST (everything under /tmp was wiped): your scoring wrapper, orchestrator and all forecast files. Rebuild them: a wrapper script under /tmp/agy_moegate4/ that fits a moegate4 variant on the truncated rounds with the TRAIN split's `condition` restored (moegate4.md explains why) and writes forecasts in the same .npz format and naming as `tdx.tools.tail_eval` / `tdx.tools.short_backtest` (read those tools: `run_predict`, `run_short_predict`), plus an orchestrator that runs only ONE GPU job at a time and logs to /tmp/agy_moegate4/orchestration.log.
- NOTE for scoring: a teammate has fixed a signal-timing problem in the evaluation tools on branch `origin/andrew-exp`; we have NOT merged it, so `m3` comparisons from the local tools are slightly optimistic for m3 (about 0.2% Brier). The GRU, transformer and moegate arms do not use signal, so comparisons among them are unaffected. State this caveat in the report next to any m3 number.

## Do, in order
1. `$PY -m pytest -q tests/moegate4` (fix failures; you may edit only the moegate4 files).
2. Rebuild wrapper + orchestrator, then score: variants k3_ref, k4_nocond, k4_cond03, k4_cond10; seeds 1, 2; tail holds 30, 60, 90; incumbents pooled_recent, m3, gru_fac seeds 1-3 for the same windows via the standard `tail_eval predict`; short_backtest only if time remains (the local short_backtest has a known label-leak design flaw being fixed, so label any short-history numbers "unreliable"); validation fit of each variant and the expert-usage-by-condition table.
3. Final report exactly as in moegate4.md. If the budget runs out, report what finished.
