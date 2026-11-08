# Assignment 06 · Combine win probability and payout size

## Goal

Build and verify a small combiner using existing tuning predictions. On the 48 tuning episodes, keeping the adaptive model's win odds and using the self-only logistic model's positive payout mix scored **33.374**, versus **33.224516** for the adaptive model. A 90% adaptive whole-distribution blend scored 33.149. Reproduce this distinction, then test whether a small combination generalizes across held-out tuning folds. This assignment can start before any new model is ready. [Adaptive forecasts](../runs/model-comparison/adaptive_lag.csv), [two-stage forecasts](../runs/model-comparison/two_stage_self.csv).

Stacked regression fits constrained combination weights on held-out predictions, and probabilistic forecast combination preserves uncertainty by mixing distributions. Here a factorized combination lets the strongest model supply win mass while another estimates the relative frequencies of positive payouts. [Breiman, Stacked Regressions](https://doi.org/10.1023/A%3A1018046112532), [Raftery et al. on probabilistic forecast ensembles](https://doi.org/10.1175%2FMWR2906.1).

## Build

1. Load existing `adaptive_lag.csv` and `two_stage_self.csv` with the kit's submission reader. Assert identical tuning IDs and exact column order. Join by query ID. For each component let `w=1-p_0` and `r=p_positive/w`. Guard against zero win mass by using the adaptive conditional mix where defined and a documented normalized training prior if both components have zero win mass.
2. Compare four prespecified configurations. Score adaptive unchanged, the fixed head swap `w_adaptive × r_logistic`, a 50/50 conditional positive mix with fixed adaptive win mass, and a fixed 90/10 whole-distribution blend. The resulting row always has `p_0=1-w` and four entries `w*r` for a head-wise combination.
3. Try at most one learned blend for the initial pass. Hold win mass at `w_adaptive` and learn a single weight for mixing the positive heads on the official normalized squared loss. The objective is quadratic in that weight. For base forecast `p`, alternative `a`, target `q` and official row weights `u`, compute `clip(sum(u*(a-p)·(q-p)) / sum(u*||a-p||²), 0, 1)`; use weight zero when the denominator is effectively zero. Here `u = 1/(4 * B[condition] * episodes_in_condition * queries_in_episode)`, with the scorer's denominator floor on `B`. Use four fixed folds of tuning episodes stratified by condition, fitting only on the other folds and recomputing the reference denominators there. Assemble one out-of-fold forecast per tuning query and score the assembled predictions with the official scorer. If retained, fit one final scalar on all tuning rows and save it separately from its out-of-fold score.
4. Add new models only after their standalone evaluation. Give each model a precise role in the blend and compare it against its own standalone forecast. Report which comparisons were exploratory on tuning. If exploring a larger or context-dependent ensemble later, use grouped train out-of-fold predictions or a separate selection protocol.

## Verification

- Reproduce the adaptive score to within `1e-5` points and the fixed head-swap score to within `0.002` points. A head-only mix must preserve the input adaptive `p_0` for every query. Endpoints 0 and 1 must reproduce the corresponding fixed combinations. Reject duplicate or missing IDs and invalid probabilities before blending.
- On a toy table, compare the clipped scalar solution with a dense grid and check the zero-denominator case. Randomize row order and verify identical scores and normalized predictions. Assert zero overlap of fitting and evaluation episode IDs in every tuning fold.
- Report tuning points and condition ratios, positive-class squared errors, and fitted weight per fold. For fixed combinations, report paired episode intervals against adaptive. Treat an interval computed from learned out-of-fold predictions as descriptive unless the resampling also refits the weights. Label any full-tuning fitted score as in-sample.

## Execution and handoff

Own `slot_forecasting/combine_forecasts.py` and `run_combine_forecasts.py`. Write to `runs/06-combine-forecasts/<run-id>/`. Use only the 48 tuning episodes through `run_model_comparison.tuning_data()` and the official `contest_score`. Use the tuning soft targets for scalar combination fitting only under the fold protocol above. Reserve audit and test. Return the command, split IDs, learned and fixed weights, checks, component hashes, tuning CSVs, and `report.json` with separate fixed, out-of-fold and any in-sample scores. See the [shared experiment contract](README.md#shared-experiment-contract).
