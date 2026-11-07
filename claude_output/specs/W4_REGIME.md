# W4 — `regime`: condition posterior, adaptive windows, changepoint, HMM

## Role and objective
Implement arm `regime` in `src/tdx/arms/regime/`. It has two jobs: (a) produce a condition posterior per episode (`cond_posterior[E, 4]`, order `stable, short_history, drift, shared_change`) that the stacker uses; (b) provide specialist forecasters whose memory length adapts to the inferred regime.

## Why
Score weights the four conditions equally against the pooled reference, so the weakest condition matters as much as the strongest. Gemini's hand thresholds for drift and shared_change were unsupported (review row 10); `conditions.csv` gives 96 + 96 labelled episodes to learn them. Read first: `claude_output/02_BAKEOFF_PLAN.md` section 3D, `kit/docs/DATA.md`.

## Decisions (do not reopen)
- D1 classifier: features per episode from history only: `n_obs==64` flag; for windows w ∈ {10, 20, 40, 80}: mean payout value and win rate of the last w observed ticks versus the preceding w; cross-machine synchrony (average pairwise correlation of the 8 machines' win indicators over the last 40 ticks, and the same over the earlier part); variance across time of the 8-machine mean win rate (rolling 20); per-type means. Model: multinomial logistic regression (sklearn, standardised features, C chosen by grouped 5-fold CV over train episodes), final fit on train (dev) or train+validation (final). Report held-out accuracy by grouped CV and a confusion matrix, and the accuracy on validation episodes in dev mode. Save `cond_posterior` for validation and test to `runs/regime/<variant>/cond_posterior_{val,test}.npy`. short_history is overridden to probability 1 when `n_obs==64` for all 8 machines.
- D2 specialists: counts-based forecasters (reuse the logic of `kit/slot_forecasting/baseline.py` and `transition.py`, vectorised) with parameters (half-life, prior strength) per condition class; route by posterior-weighted mixture of the four specialists' outputs. Tune the per-condition parameters on validation targets restricted to that condition's episodes (dev) with the two-half protocol; do not tune on `test`.
- D3 changepoint: for each episode compute, over candidate change ticks `c` in the last 120 ticks (step 4), the log-likelihood gain of fitting separate pooled class distributions before/after `c` for the episode mean (all 8 machines); if the penalised gain exceeds a threshold learned on train episodes (choose the threshold maximising CV condition accuracy for shared_change), forecast from post-change ticks only (weights decay with half-life 8 after the change). Report how often it fires per condition.
- D4 optional HMM (only if D1–D3 are done with ≥ 30 min left): K=3 states, emissions conditioned on lag-1 class (3 categories as in `transition.py`), numpy forward algorithm, EM on train history with 5 restarts, per-machine state posterior at the cutoff then emission mixture.
- Variants: `d2_routed`, `d3_changepoint`, `d2_d3_mix` (average of routed and changepoint), `hmm3` (if D4). All numpy/sklearn; no torch. `save/load` as JSON/npz.

## Preconditions
Hard: W0 merged. Soft: sklearn available (else implement logistic regression by Newton in numpy).

## Owned files
`src/tdx/arms/regime/*`, `tests/regime/*`, `runs/regime/**`, `claude_output/notes/W4_*`. Forbidden: everything else.

## Work packages
- **P1** D1 classifier + CV report + posterior files. Test: posterior rows sum to 1; leak perturbation test on the features.
- **P2** D2 specialists + tuning + variant `d2_routed`.
- **P3** D3 changepoint + variants `d3_changepoint`, `d2_d3_mix`.
- **P4** D4 optional.
- **P5** leak test; notes `claude_output/notes/W4_regime.md` with CV accuracy, confusion matrix, per-condition ratios of each variant.

## Acceptance
```
~/.venvs/tdx/bin/python -m pytest -q tests/regime
~/.venvs/tdx/bin/python -m tdx.sweep --arm regime
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm regime
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
python kit/check_submission.py runs/regime/<best>/test_pred.csv
```
Time box 90 minutes.

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1–P3 | gpt-6.1-sol | medium | well-specified statistics code |
| P4 | gpt-6.1-sol | high | EM details |
| P5 notes | gpt-6-luna | low | mechanical |
| Review (read-only) | gpt-6-astra | high | different model |
