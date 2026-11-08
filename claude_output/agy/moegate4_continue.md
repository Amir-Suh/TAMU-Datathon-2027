# Antigravity job (continuation): finish and score moegate4 NOW (no time gate)

You have no conversation context. A previous session of this job already wrote code and then waited for a GPU gate that is now lifted. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; `PYTHONPATH=src`). GPU is free for you except two or three other trainings run by a different job (use at most 2 GPU jobs at a time yourself). Budget: 70 minutes. Run `date` and write it to /tmp/agy_moegate4/start2.txt.

## Read first (they already exist)
- The original task description: `claude_output/agy/moegate4.md` (question, rules, variants, tests, scoring, final report format). Follow it exactly, EXCEPT ignore its "do not start before 21:55" gate: start now.
- Files already written by the previous session: `src/tdx/arms/moegate4/{__init__,arm.py}`, `tests/moegate4/test_moegate4.py`, `/tmp/agy_moegate4/scoring_wrapper.py`, `/tmp/agy_moegate4/orchestrator.py` (look at what is there and what is missing; fix bugs you find).

## Do, in order
1. `$PY -m pytest -q tests/moegate4` and `$PY -m tdx.tools.leak_test --arm moegate4 --variant k4_cond03`; fix failures (you may edit only the files listed in moegate4.md).
2. Launch the scoring with the orchestrator (variants k3_ref, k4_nocond, k4_cond03, k4_cond10; seeds 1, 2; tail holds 30/60/90; short_backtest seed 1; incumbents pooled_recent, m3, gru_fac seeds 1-3 for the same windows; validation fit and the expert-usage-by-condition table for each variant).
3. Final report exactly as specified in moegate4.md (tables, usage by condition, one-sentence verdict, failures, elapsed minutes). If the budget runs out, report what finished.
