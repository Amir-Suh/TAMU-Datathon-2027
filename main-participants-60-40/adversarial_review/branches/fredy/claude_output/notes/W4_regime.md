# W4 regime work report

Status reflects the completed regime work and captured evidence in `runs/`. The tuned d2 model is the strongest candidate, but the paired episode bootstrap interval includes zero, so it was not promoted as a statistically clear gain. The final-mode artifact was produced and format-checked; package reproduction for upload remains a W6 step.

## Work status

| Item | Status | Evidence |
|---|---|---|
| P1: regime classifier | Done | Nested five-fold episode CV accuracy **0.59375** on train; validation accuracy **0.5208333333333334**. CV confusion (rows/columns stable, short_history, drift, shared_change): `[[13,0,7,4],[0,24,0,0],[10,0,8,6],[4,0,8,12]]`. Confusions and fire rates are recorded in `runs/regime/d2_routed_tuned/regime_report.json` and `runs/regime/tuning/tuning.json`. |
| P2: routed specialists / integration | Done | d2 routed specialist arm and tuned per-condition configuration; see `runs/regime/d2_routed_tuned/`. |
| P3: changepoint | Done, retained as evaluated candidate | Five-fold CV accuracy **0.7395833333333334**, but observed fire rates were zero in stable, short-history, drift, and shared-change train sets; validation fired once in 24 shared-change episodes. See `runs/regime/d3_changepoint/regime_report.json`. |
| P4: HMM | Optional; skipped | No evidence justified adding this extra model after specialist routing and changepoint comparisons. |
| P5: leakage / format checks | Done | Commands and exact output summaries below. |

## Pilot comparison

The pilot results below are copied from `runs/LOG.md`. “Ratios” are stable / short-history / drift / shared-change.

| Variant | Points | Ratios | Seconds |
|---|---:|---|---:|
| `d2_routed` (default, post nested-CV rerun) | 30.476064 | 0.986126 / 1.022900 / 0.966218 / 0.899789 | 2.255 |
| `d3_changepoint` (default, post nested-CV rerun) | 29.709885 | 1.019072 / 1.028082 / 1.038101 / 0.992864 | 2.125 |
| `d2_d3_mix` (default, post nested-CV rerun) | 30.187244 | 0.996901 / 1.024446 / 0.993499 / 0.935532 | 2.137 |
| `d2_routed` with tuned specialist override | 30.779477 | 0.970101 / 1.012909 / 0.949910 / 0.864484 | 1.800 |

The tuned override is materialized separately as `runs/regime/d2_routed_tuned/`; the `runs/LOG.md` variant label remains `d2_routed` because the harness logs the arm variant name. Against `pooled_recent` at **30.825854** points, tuned d2 has a **−0.046377 point deficit**. Paired validation episode bootstrap difference interval: **[−0.355, +0.244]**, so no promotion on this evidence.

## Reproduction and proof commands

Commands below were run from the repository root, using `~/.venvs/tdx/bin/python` (Python 3.13). Final rerun: **11 passed in 5.20s**. Each command's output artifacts are under the paths shown.

```bash
~/.venvs/tdx/bin/python -m pytest -q tests/regime
# 11 passed in 5.20s

~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm regime --variant d2_routed
# PASS: 5 episodes; cutoff reduced by 4; max abs difference 0.0; forbidden dev opens 0
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm regime --variant d3_changepoint
# PASS: 5 episodes; cutoff reduced by 4; max abs difference 0.0; forbidden dev opens 0
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm regime --variant d2_d3_mix
# PASS: 5 episodes; cutoff reduced by 4; max abs difference 0.0; forbidden dev opens 0

~/.venvs/tdx/bin/python kit/check_submission.py runs/regime/d2_routed_tuned/test_pred.csv --data data/main
# VALID FORMAT: 1536 queries | main | test
~/.venvs/tdx/bin/python kit/check_submission.py runs/regime/d2_routed_final/test_pred.csv --data data/main
# VALID FORMAT: 1536 queries | main | test
```

The sweep and comparison commands were:

```bash
~/.venvs/tdx/bin/python -m tdx.sweep --arm regime
# regime/d2_routed dev: 30.476064 points (2.255s)
# regime/d3_changepoint dev: 29.709885 points (2.125s)
# regime/d2_d3_mix dev: 30.187244 points (2.137s)
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
# Full output: runs/regime/compare.txt
```

The inference check recorded for both tuned and final outputs was **max_abs_difference 0.0; sklearn loaded False**. The final artifact was generated with:

```bash
~/.venvs/tdx/bin/python -m tdx.run --arm regime --variant d2_routed --mode final \
  --set 'specialists=[[8,60],[16,20],[16,60],[16,60]]' \
  --out runs/regime/d2_routed_final
```

The tuning evaluator is explicitly outside model fitting, per the no-validation-answer-in-fit contract. It SHA256-orders episodes within each condition, alternating 12 tuning and 12 audit episodes. It selected `[8,60]`, `[16,20]`, `[16,60]`, `[16,60]` for stable, short-history, drift, and shared-change. Exact audit command: `~/.venvs/tdx/bin/python -m tdx.arms.regime.tune`; values are saved in `runs/regime/tuning/tuning.json`. Fresh-folder packaging/reproduction has not yet been done; run `claude_output/sh/repro_check.sh` before upload, followed by `kit/check_submission.py` on the upload CSV.

The pilot sweep and comparison commands were `~/.venvs/tdx/bin/python -m tdx.run --arm regime --mode dev ...` for each of the three variants, then `~/.venvs/tdx/bin/python tools/compare.py`; the retained 2,000-resample output is `runs/regime/compare.txt`. (The exact comparison command was `~/.venvs/tdx/bin/python tools/compare.py --baseline pooled/pooled_recent --runs runs --output runs/regime/compare.txt --bootstrap 2000 --seed 20261107`.)

For clarity, tuning-half score was **31.191896** points and held-out audit-half score was **30.328648** points. The `30.779477` all-validation figure is exploratory because it combines both halves after specialist selection. `runs/regime/compare.txt` records the 2,000-resample paired episode bootstrap (seed `20261107`): d2 difference **−0.046** points, 95% interval **[−0.355, +0.244]**.

## Surprises (8 lines)

- The classifier's train episode-CV accuracy was 0.59375, while validation accuracy was 0.52083.
- Short-history classification was perfect in the reported confusion matrices (24/24), but drift and stable were often confused.
- The changepoint score achieved 0.73958 CV accuracy yet almost never crossed its selected threshold.
- The tuned specialists improved the shared-change ratio to 0.86448, the best condition ratio in the latest d2 result.
- Short-history remained slightly worse than baseline at 1.01291.
- Changepoint-only and the mixed d2+d3 candidate both scored below the tuned d2 result.
- The tuned d2 deficit versus pooled_recent is 0.046377 points and uncertain under the paired interval.
- Final-mode generation and CSV format validation succeeded; fresh-folder reproduction remains outstanding.

## Delegation

Subagent models and efforts were chosen from the user-supplied package table through explicit `collaboration.spawn_agent(model=..., reasoning_effort=..., fork_turns="none")` arguments. P2/integration stayed in the parent session. P4 was not delegated.

| Work | Assignment | Why / outcome |
|---|---|---|
| Work | Assignment / why | Outcome |
|---|---|---|
| P1 classifier | `gpt-6.1-sol`, medium, delegated with `fork_turns=none` for independent classifier work | Well-specified statistics code; implemented and evaluated. |
| P3 changepoint | `gpt-6.1-sol`, medium, delegated with `fork_turns=none` for independent changepoint work | Well-specified statistics code; implemented and evaluated. |
| Read-only review | `gpt-6-astra`, high, independent review of leakage and quality | Independent stronger-model review; no high-priority findings, 11 tests passed. |
| P5 report | `gpt-6-luna`, low, report synthesis | Mechanical evidence collection; this report. |
| P2 / integration | Parent session; integrated specialists, sweep, and selection | Tuned specialists and outputs integrated. |

## Durable artifacts

The explicit changed-path manifest is [`W4_changed_paths.txt`](W4_changed_paths.txt). It lists the implementation and tests, score ledger, comparison, tuning evidence, pilot outputs, tuned validation/test outputs, and final-mode artifact.
