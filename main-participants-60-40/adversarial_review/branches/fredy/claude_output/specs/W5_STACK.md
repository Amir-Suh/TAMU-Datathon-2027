# W5 — `stack`: blend + cutoff-level calibrator (launch after the pilot wave is read)

## Role and objective
Build `src/tdx/stack/`: take the `dev`-mode validation predictions of the kept arms, fit condition-aware convex blend weights and an optional cutoff-level calibrator that uses `signal`, report an honest grouped-CV score, and apply the frozen weights to `final`-mode test predictions.

## Why
Averaging probability forecasts from models with different errors is the cheapest reliable gain under squared error, and `signal` can only enter at the cutoff. Read first: `claude_output/02_BAKEOFF_PLAN.md` section 3E, `claude_output/runs_selection.md` (written by Claude after the pilots; lists the kept arms/variants and the reading rule), `runs/READING_RULE.md`.

## Decisions (do not reopen)
- Inputs: for each kept `(arm, variant)`: `runs/<arm>/<variant>/val_pred.npy` (dev) and `test_pred.npy` (final); `cond_posterior_{val,test}.npy` from `regime/<variant>`.
- Stage 1: for condition posterior `π[e, c]`, prediction `Σ_c π[e,c] Σ_k w[c,k] p_k`, weights `w[c,:]` on the simplex, minimising squared error to validation `target`. Solve with SLSQP (scipy) with 20 random restarts or an exact QP via projected gradient; report the solver residual.
- Stage 2: multinomial logit on `[log p1, type one-hot (3), signal one-hot (3), type×signal one-hot (9)]` with L2 strength chosen by grouped CV, fitted to validation soft targets (cross-entropy against the 5-vector). Keep only if grouped-CV points improve by more than the paired-bootstrap interval from `tools.compare`; otherwise skip stage 2 and say so.
- Honest score: 8-fold grouped CV over validation episodes (folds by `episode_id` hash, stratified by condition): fit stages 1–2 on 7 folds, predict the 8th, concatenate, score with `tdx.score.score`. This is the number to compare to single arms (their own scores are already out-of-sample).
- Final: fit on all validation episodes; apply to `test_pred.npy` of the `final`-mode arms; write `runs/stack/<name>/{test_pred.csv, weights.json, cv_report.json}`.
- Stage-1 weights from `dev` arm outputs are applied to `final`-mode arm outputs (which also trained on validation). That shifts calibration slightly; to quantify, also report the blend weights recomputed with arms' grouped-CV-style predictions if the arms provide them (not required).

## Preconditions
Hard: at least two kept arms with `val_pred.npy` in `runs/`; `runs_selection.md` exists. Soft: `cond_posterior` missing → use the hard `n_obs==64` flag plus uniform over the other three.

## Owned files
`src/tdx/stack/*`, `tests/stack/*`, `runs/stack/**`, `claude_output/notes/W5_*`.

## Work packages
- **P1** stage 1 + CV; **P2** stage 2 + CV; **P3** apply + export `weights.json` (consumed by W6); **P4** notes with: per-condition ratios of the blend vs each member, weights table, CV score with interval.

## Acceptance
```
~/.venvs/tdx/bin/python -m pytest -q tests/stack
~/.venvs/tdx/bin/python -m tdx.stack.fit --config claude_output/runs_selection.json --name s1
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
python kit/check_submission.py runs/stack/s1/test_pred.csv
```
Time box 60 minutes.

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1–P3 | gpt-6.1-sol | high | CV leakage is the main risk; fold hashing must match tools |
| P4 | gpt-6-luna | low | tables from files |
| Review | gpt-6-astra | high | different model |
