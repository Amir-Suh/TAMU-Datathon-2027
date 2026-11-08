# W9 leakage fixes
Status: P1 done; P2 done; P3 done; P4 done. Nothing unfinished; no commits, real retraining, scoring pilots, or submission artifacts.
P1: five public unmasked float64 `step_outputs` adapters; SeqLossArm inherits SeqArm's adapter.
P2: moved cutoffs get neutral signals and recounted observations; exact values/remove/add/move audits use full original validation layouts at k=32 and min(cutoff)-8; cutoff/dev-file audits remain; CLI accepts --set overrides.
P3: NumPy and Torch flags use per-tick missing-prefix detection; all three arms export causal_v1 metadata and reject legacy metadata, including unflagged seq2 exports.
P4: observed-only layouts with true offsets, dedicated masked PAD embedding, token bounds checks, and migration of legacy five-row shortctx embeddings; training samples unchanged.
Verification environment: `export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`; Python below is `~/.venvs/tdx/bin/python`.
Acceptance: `~/.venvs/tdx/bin/python -m pytest -q -p no:cacheprovider tests/test_leak.py tests/moegate tests/moegate4 tests/seq2 tests/shortctx tests/seq tests/tfm2 tests/seqloss` → `56 passed in 69.18s (0:01:09)`.
`~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm seq --variant gru128_factor --set epochs=1 --set h=16` → `PASS seq/gru128_factor`, cutoff delta 0.0, forbidden opens 0, step_causality 0.0.
`~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm tfm2 --variant d128l3 --set epochs=1 --set d=16 --set L=1` → `PASS tfm2/d128l3`, cutoff delta 0.0, forbidden opens 0, step_causality 0.0; parity 8.882e-16.
`~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm moegate --variant pilot --set epochs=1 --set h=16` → `PASS moegate/pilot`, cutoff delta 0.0, forbidden opens 0, step_causality 0.0.
`~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm moegate4 --variant k3_ref --set epochs=1 --set h=16` → `PASS moegate4/k3_ref`, cutoff delta 0.0, forbidden opens 0, step_causality 0.0.
`~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm seq2 --variant flag_aug5all --set epochs=1 --set h=16` → `PASS seq2/flag_aug5all`, cutoff delta 0.0, forbidden opens 0, step_causality 0.0.
P1 print command: `~/.venvs/tdx/bin/python -m pytest -q -s -p no:cacheprovider tests/moegate/test_moegate.py::test_f_initialized_weights_are_step_causal_and_float64 tests/moegate4/test_moegate4.py::test_g_initialized_weights_are_step_causal_and_float64 tests/seq2/test_seq2.py::test_step_flag_is_causal_and_legacy_flagged_export_rejected` → `3 passed in 0.10s`; every arm printed values/remove/add/move `0.000000000`.

| Arm | Before remove/add (audit P1) | After remove/add/move |
| --- | --- | --- |
| moegate | 0.005350618 / 0.005350618 | 0.000000000 / 0.000000000 / 0.000000000 |
| moegate4 | 0.002312858 / 0.002312858 | 0.000000000 / 0.000000000 / 0.000000000 |
| seq2 | 0.173931807 / 0.173931807 | 0.000000000 / 0.000000000 / 0.000000000 |

Adapted P2 random-weight probe: retained nonzero moved signals 28 → 0; missing_vs_jackpot maxdelta 0.0 → 0.000000021493; PAD offset maxdelta 0.0. Shortctx suite: `~/.venvs/tdx/bin/python -m pytest -p no:cacheprovider -q tests/shortctx` → `6 passed in 3.83s`.
Repeatability: two independent one-epoch CPU fits on three training episodes produced maxdelta `0.000000000` for seq, tfm2, moegate, moegate4, seq2.

| Pilot variant | Points | Stable | Short | Drift | Shared | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| None (repair-only task; runs/LOG.md unchanged) | N/A | N/A | N/A | N/A | N/A | N/A |

Surprises/risks: truncating short layouts before the step probe hid count==64 leaks; integration regression now catches that. Old MoE/seq2 exports require refitting; contest score effects remain unmeasured.
Outside owned list: `.pytest_cache/v/cache/nodeids` was updated by the initial pytest run; all subsequent pytest runs disabled the cache. No other outside source/artifact edits found.
Delegation used explicit `collaboration.spawn_agent(model=..., reasoning_effort="high", fork_turns="none")`; cheaper implementers received a different, stronger reviewer.

| Package | Model | Effort | Why |
| --- | --- | --- | --- |
| P1 seq/tfm2, P2, integration/report | Primary Codex (inherited) | Inherited | Shared interfaces and acceptance verification |
| P1 MoE/seq2, P3 | gpt-6-luna | high | Bounded implementation, verified by tests and stronger review |
| P4 | gpt-6-luna | high | Independent PAD repair, verified by tests and stronger review |
| Independent review | gpt-6.1-sol | high | Different model reviewed harness, flags, PAD, and legacy loading; no remaining issues |

Changed paths: `src/tdx/tools/leak_test.py`, `src/tdx/arms/seq/arm.py`, `src/tdx/arms/tfm2/arm.py`, `src/tdx/arms/moegate/arm.py`, `src/tdx/arms/moegate4/arm.py`, `src/tdx/arms/seq2/arm.py`, `src/tdx/arms/shortctx/arm.py`, `tests/test_leak.py`, `tests/moegate/test_moegate.py`, `tests/moegate4/test_moegate4.py`, `tests/seq2/test_seq2.py`, `tests/shortctx/test_shortctx.py`, `claude_output/notes/W9_leakfix.md`; incidental cache path above.
