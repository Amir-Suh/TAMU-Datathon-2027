# W0 report

P1–P6: done. No commits or network access.

| Package | Status | Proof |
|---|---|---|
| P1 data/cache | done | `~/.venvs/tdx/bin/python -m pytest -q tests`: 16 passed; shapes 96/96/192, 192 short train machines, 1000 CSV round-trips |
| P2 causal features | done | Same test command: brute-force three-episode references and future-corruption checks pass |
| P3 scoring/run/sweep/log | done | Same test command: reference 30.0, four ratios 1.0, final writes no validation score; sweep runs four variants |
| P4 pooled | done | `~/.venvs/tdx/bin/python -m tdx.run --arm pooled --variant pooled_recent --mode dev`: 30.825854 points; pooled: 30.000000 |
| P5 tools | done | `~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm pooled --variant pooled_recent`: 5 episodes, max difference 0.0, forbidden opens 0 |
| P6 notes | done | W0_harness.md records built API, deviations, cold/warm timings |

Every exact acceptance command and its printed output is in [W0_acceptance.txt](W0_acceptance.txt). Environment: PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. Both kit/check_submission.py commands return exit=0: test 1536 queries, validation 768 queries and 30.825854 points.

Pilot rows copied from runs/LOG.md (latest dev entry per variant):

| variant | points | stable | short | drift | shared | seconds |
|---|---:|---:|---:|---:|---:|---:|
| pooled_recent | 30.825854 | 0.958227 | 1.000561 | 0.946405 | 0.880479 | 0.060 |
| pooled | 30.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.057 |
| recent | 30.743191 | 0.985914 | 1.017846 | 0.938333 | 0.864513 | 0.054 |
| context | 29.685186 | 0.990280 | 1.079960 | 1.029656 | 0.984945 | 0.050 |

Surprises and open risks:
- Causal-test correction: reduce cutoff by four before corrupting future storage; the original last four observed ticks were legal inputs.
- Root symlink `tdx -> src/tdx` is an extra path needed for bare module commands without installation; collection must preserve it or set PYTHONPATH=src.
- Cache deliberately excludes validation answers; cold 0.719s, warm 0.009s with answers attached.
- Independent review found kwargs forwarding and dataset-identity checks missing; both fixed and independently rechecked.
- No adaptation data kit to exercise; all round/data selection is parameterized.
- No upload requested; final packaging and fresh-folder reproduction remain with the submission stream.

| Package | Model | Effort | Why |
|---|---|---|---|
| P1–P3 implementation, tests, P6 integration | Primary Codex session (inherited model) | Session default | Shared core implemented in the active stream |
| P4 equality to kit | gpt-6.1-sol | medium | Independent numeric parity and persistence checks |
| P5 tools/leak audit | gpt-6.1-sol | high | Audit-hook lifetime and bootstrap pairing |
| Independent read-only review | gpt-6-astra | high | Different model reviewed integration and verified fixes |

Model selection mechanism: explicit `collaboration.spawn_agent` model/effort arguments with `fork_turns="none"` for delegated packages, following the user's routing table. P1–P3 stayed in the inherited primary session; its model was not changed.

Files created/edited and explicit source/report paths to collect (leave uncommitted):

- `src/tdx/__init__.py`
- `src/tdx/arm.py`
- `src/tdx/arms/__init__.py`
- `src/tdx/arms/pooled/__init__.py`
- `src/tdx/arms/pooled/arm.py`
- `src/tdx/data.py`
- `src/tdx/feat.py`
- `src/tdx/fit.py`
- `src/tdx/log.py`
- `src/tdx/paths.py`
- `src/tdx/run.py`
- `src/tdx/score.py`
- `src/tdx/sweep.py`
- `src/tdx/tools/__init__.py`
- `src/tdx/tools/compare.py`
- `src/tdx/tools/leak_test.py`
- `tests/test_data.py`
- `tests/test_feat.py`
- `tests/test_leak.py`
- `tests/test_score.py`
- `tdx`
- `runs/LOG.md`
- `claude_output/notes/W0_harness.md`
- `claude_output/notes/W0_report.md`
- `claude_output/notes/W0_acceptance.txt`
- `claude_output/notes/W0_timings.json`

Generated experiment artifacts (git-ignored run directories):

- `runs/pooled/pooled/meta.json`
- `runs/pooled/pooled/model/model.json`
- `runs/pooled/pooled/score.json`
- `runs/pooled/pooled/test_pred.csv`
- `runs/pooled/pooled/test_pred.npy`
- `runs/pooled/pooled/val_pred.csv`
- `runs/pooled/pooled/val_pred.npy`
- `runs/pooled/pooled_recent/meta.json`
- `runs/pooled/pooled_recent/model/model.json`
- `runs/pooled/pooled_recent/score.json`
- `runs/pooled/pooled_recent/test_pred.csv`
- `runs/pooled/pooled_recent/test_pred.npy`
- `runs/pooled/pooled_recent/val_pred.csv`
- `runs/pooled/pooled_recent/val_pred.npy`
- `runs/pooled/recent/meta.json`
- `runs/pooled/recent/model/model.json`
- `runs/pooled/recent/score.json`
- `runs/pooled/recent/test_pred.csv`
- `runs/pooled/recent/test_pred.npy`
- `runs/pooled/recent/val_pred.csv`
- `runs/pooled/recent/val_pred.npy`
- `runs/pooled/context/meta.json`
- `runs/pooled/context/model/model.json`
- `runs/pooled/context/score.json`
- `runs/pooled/context/test_pred.csv`
- `runs/pooled/context/test_pred.npy`
- `runs/pooled/context/val_pred.csv`
- `runs/pooled/context/val_pred.npy`

Generated local cache: `.cache/main_7979323052df471e.npz`. Do not include it in source collection.
