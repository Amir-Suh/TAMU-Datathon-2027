# W1 — `glm`: hierarchical per-machine multinomial logit (primary bet)

## Role and objective
Implement arm `glm` in `src/tdx/arms/glm/` and run its pilot grid. Goal: beat `pooled_recent` (30.826 validation points) by exploiting machine-specific lag structure (lags up to 8) and neighbour wins, with shrinkage to a pooled fit.

## Why
Measured: true cutoff probabilities are near-deterministic for most machines (445/768 below 0.2 or above 0.8 in true `p_0`), lag effects differ per machine and sometimes flip sign (`claude_output/01_ADVERSARIAL_REVIEW.md` rows 2–4). Read first: that file, `claude_output/02_BAKEOFF_PLAN.md` section 3A, `src/tdx/{data,feat,arm}.py`, `claude_output/notes/W0_harness.md`.

## Decisions (do not reopen)
- Model: 5-class softmax regression, features φ(e, m, t) built only from `y[e, :, :t]`:
  1. bias; machine-type one-hot (3);
  2. own payout class one-hot at lags 1..L, plus a missing flag per lag (class codes via `feat.lag_classes`);
  3. neighbour wins: for lags 1..min(L,3), the count of the other 7 machines with payout > 0, and the count with payout ≥ 5 (0 and a missing flag when any neighbour tick is unobserved);
  4. recency-weighted own class rates over observed ticks before `t` at half-lives 4, 16, 64 (5 values each, renormalised by the weight sum);
  5. total own win rate over the last 20 observed ticks.
- Hierarchy: Level 0 = pooled weights `W0` fit on all tick rows of the fit spec (L2 λ0=1, rows weighted by recency from the end of each machine's history with half-life τ). Level 1 = per episode-machine deviation `Δ` fit on that machine's own tick rows, penalty `λ‖Δ‖²`, prediction uses `W0+Δ`. Fit Level 1 with a fixed number of Newton steps (≤ 8) per machine; vectorise across machines with batched numpy where easy, otherwise `scipy.optimize.minimize(method="L-BFGS-B")`.
- Short-history machines (n_obs == 64) use the same code; fewer rows automatically shrink toward `W0` (no special branch).
- Fit modes: `dev` uses `dev_spec(transductive)`; Level 0 trains on all tick rows of the allowed splits; Level 1 is fit at predict time per query machine on its own history (valid for train/validation/test splits alike). `final` uses `final_spec()`; additionally, cutoff-label rows (train labels, validation labels) are appended to Level 0 as hard rows with weight `w_cut` (default 20) and the validation target vectors enter as soft rows (cross-entropy with the soft vector) with the same weight.
- Numpy float64 only (the package must run without torch). Provide `save/load` (npz of `W0` and hyperparameters) so W6 can ship it.

## Preconditions
Hard: W0 merged (`python -m tdx.run --arm pooled --variant pooled_recent --mode dev` prints 30.826). Soft: if `feat.others_win` lacks a lag you need, add a local helper inside your arm and report it.

## Owned files
Create/edit: `src/tdx/arms/glm/*`, `tests/glm/*`, `runs/glm/**`, `claude_output/notes/W1_*`. Forbidden: everything else.

## Work packages
- **P1 core**: `features.py` (vectorised φ for all tick rows and for the cutoff row of any split), `model.py` (Level 0 fit, Level 1 per-machine fit, predict), `arm.py` with `build`, `VARIANTS`, `save/load`. Tests: φ at tick `t` unchanged when `y[..., t:]` is corrupted; predictions sum to 1; `save`→`load` reproduces predictions exactly; a synthetic episode with a planted lag-2 rule is recovered (predicted probability within 0.1 of the planted one).
- **P2 pilot grid**: run `python -m tdx.sweep --arm glm` over VARIANTS (≤ 12): L ∈ {2, 4, 8}; λ ∈ {1, 10, 100} (at L=4); τ ∈ {32, 96, 1e9} (at best λ so far, choose by points on validation; record the choice); neighbours off; transductive 0 vs 1 at the best setting; and one two-stage variant (win vs size) only if time remains. Each variant ≤ 5 min.
- **P3 leak test** `python -m tdx.tools.leak_test --arm glm` passes.
- **P4 notes**: `claude_output/notes/W1_glm.md` with the table from `runs/LOG.md`, per-condition behaviour of the best variant, and one paragraph on which feature groups mattered (ablate groups 3 and 4 on the best variant; two extra runs).

## Acceptance
```
~/.venvs/tdx/bin/python -m pytest -q tests/glm
~/.venvs/tdx/bin/python -m tdx.sweep --arm glm
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm glm
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
python kit/check_submission.py runs/glm/<best>/test_pred.csv
```
Time box 90 minutes total. Predict (all splits) must stay under 5 min on 2 threads.

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1 | gpt-6.1-sol | high | numerics and vectorisation need care |
| P2 sweep driving and P4 notes | gpt-6-luna | low | mechanical, numbers come from tools |
| Review of P1 (read-only) | gpt-6-astra | high | different model |
