# W2 — `gbdt`: global LightGBM on tick rows

## Role and objective
Implement arm `gbdt` in `src/tdx/arms/gbdt/` and run its pilot grid. Goal: a strong global learner whose features expose lags, multi-scale rates, neighbour aggregates and regime divergence.

## Why
Gemini's family 1, kept because it is the standard strong baseline for tabular lag features and gives a different error pattern from the per-machine GLM. Read first: `claude_output/02_BAKEOFF_PLAN.md` section 3B, `src/tdx/{data,feat,arm}.py`, `claude_output/notes/W0_harness.md`.

## Decisions (do not reopen)
- Rows: every observed `(e, m, t)` with at least 8 observed ticks before `t` (fewer allowed in short_history only if the 8 most recent observed ticks exist). Target = class of `y[e,m,t]`.
- Features (all strictly past): own lags 1..8 as class codes (−1 missing); `code12 = 5*lag1 + lag2` when both observed else −1; EMA class rates (5 values each) at half-lives 4, 12, 24, 64 over observed ticks; counts of wins in the last 5/20/50 observed ticks; `n_obs_before`; machine type; for the other 7 machines at last observed tick before `t` and aggregated over the last 5 and 10 ticks: mean win rate, max payout class, mean payout value; difference between short (15) and long (80) tick win rates for self and for the episode mean; position `t/256`.
- Row weights: `2^(-(t_end - t)/τ)` where `t_end` is the last observed tick of that machine in its split, τ variant parameter; samples from the cutoff-query rows (train labels) are added with weight `w_cut` (default 20) in `final`; validation soft targets in `final` enter as five weighted rows per query (one per class, weight `w_cut * target_prob`); no sampling.
- LightGBM params baseline: multiclass, `num_leaves=15`, `min_data_in_leaf=200`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `lambda_l2=10`, `num_threads=1`, `deterministic=True`, `force_row_wise=True`, `seed=7`, early stopping on a held-out set of whole train episodes (20%), then refit on all rows with the found number of rounds.
- Output must be clipped and renormalised. `save` writes the LightGBM text model plus feature metadata; `load` reproduces predictions to 1e-12.

## Preconditions
Hard: W0 merged. Soft: if LightGBM nondeterminism appears across two runs, switch to `num_threads=1` + `deterministic=True` + fixed bagging seeds and report it.

## Owned files
`src/tdx/arms/gbdt/*`, `tests/gbdt/*`, `runs/gbdt/**`, `claude_output/notes/W2_*`. Forbidden: everything else.

## Work packages
- **P1 features and arm**: vectorised builders (no per-row Python loops over 140k rows), `build`, `VARIANTS`, `save/load`. Tests: leak perturbation on features; feature matrix for 3 episodes equals a slow reference implementation; deterministic across two runs.
- **P2 pilot grid** (≤ 8 variants, each ≤ 6 min with 8 threads allowed in training): recency τ ∈ {inf, 64, 24}; last-N-ticks-only ∈ {64, 128, all}; leaves ∈ {7, 15}; transductive 0/1. Then a feature-group ablation on the best variant (drop neighbour aggregates; drop regime divergence).
- **P3 leak test** passes; **P4 notes** `claude_output/notes/W2_gbdt.md` with the LOG table, feature importance top 15, per-condition behaviour.

## Acceptance
```
~/.venvs/tdx/bin/python -m pytest -q tests/gbdt
~/.venvs/tdx/bin/python -m tdx.sweep --arm gbdt
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm gbdt
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
python kit/check_submission.py runs/gbdt/<best>/test_pred.csv
```
Time box 90 minutes.

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1 | gpt-6.1-sol | medium | feature code with a slow reference test |
| P2 driving, P4 notes | gpt-6-luna | low | mechanical |
| Review (read-only) | gpt-6-astra | high | different model |
