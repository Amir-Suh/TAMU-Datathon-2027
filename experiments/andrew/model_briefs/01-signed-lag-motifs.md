# Assignment 01 · Bayesian signed lag motifs

## Goal

Implement a machine-local posterior over a small dictionary of lag effects. Test whether learning an explicit direction and strength beats the current adaptive count experts, especially on stable and short-history episodes. The current model already selects among one- and two-lag counts with causal Brier weighting. This assignment changes the expert representation and how its evidence accumulates. The tuning bar is **33.224516/60**, `R = 0.805895`. [Current model](../slot_forecasting/adaptive.py), [comparison](../runs/model-comparison/comparison.json).

The related problem in Bayesian context trees is one-step prediction for a discrete process with unknown memory. The paper integrates over alternative context structures. Here, nonconsecutive lag interactions and a prior over signed log-odds effects are a proposed adaptation to the kit. [Kontoyiannis et al., Bayesian Context Trees](https://arxiv.org/abs/2007.14900).

## Build

1. Use `causal.tick_arrays()` and `lagged_spins()`. For each machine, create the binary history `z[t] = 1(payout[t] > 0)` plus the observation mask. Define `s_i(t) = 2*z[t-i]-1` for an observed lag. Use eight singleton motifs `s_i` and 28 pair parity motifs `s_i*s_j`. A parity motif lets either agreement or disagreement raise the win odds. Missing required lags make that motif unavailable.
2. Use the 12-tick, strength-8 pooled-recent win probability `b_t` as a causal offset. Build it from the type/signal prior and machine counts strictly before `t`. Include a null hypothesis whose prediction is exactly `b_t`. For motif value `x_t`, each signed amplitude `a` predicts `sigmoid(logit(clip(b_t)) + a*x_t)`. Use `a ∈ {-2, -1, +1, +2}` initially. Give the null, singleton family and pair family prior mass `(0.5, 0.25, 0.25)` and distribute a family's mass uniformly within it.
3. At each tick, predict with the posterior before observing `z[t]`. For observed outcomes, add each hypothesis's Bernoulli log likelihood relative to the null. For unavailable motifs, use the null prediction and add zero relative evidence. For missing outcome ticks, add no likelihood. Maintain log weights and normalize stably. Evaluate constant memory versus a 48-tick evidence half-life, using `log_weight = log_prior + discounted_log_evidence` so forgetting returns weights toward the original prior. Use the same rollout for historical checks and cutoff inference.
4. Keep the positive payout distribution identical to the adaptive model's conditional positive mix. Emit `[1-w, w*r_1, w*r_2, w*r_5, w*r_20]`. Compare the null, singleton-only and singleton-plus-pair variants. If parity helps, add at most one extension with a three-bin contrast such as `payout >= 2` versus `payout < 2`, starting with singleton motifs. Keep the initial grid small and select the variant with four episode-grouped folds within train.

## Verification

- On seeded toy machines generated with opposite effects from the same lag pair, such as lags 2 and 5, the model should concentrate posterior mass on that pair and opposite signs given enough observations. Include a zero-effect toy machine and report held-out Brier against the null.
- Set all hypothesis mass to the null. The cutoff win forecast must match the recency control within `1e-12`. Flip future or same-tick values in a synthetic prefix and assert that the earlier forecast is invariant. On a completely missing prefix verify finite prior predictions and zero evidence updates.
- Compare against the existing adaptive and fixed-weight lag CSVs. Report stable and short-history ratios, support by active motif, the distribution of max posterior mass, and `p_0` error. Check whether gains persist after leaving out the largest-improvement episode in each condition. A concentration plot without score improvement is a diagnostic result.

## Execution and handoff

Own `slot_forecasting/signed_motifs.py` and `run_signed_motifs.py`. Write to a fresh `runs/01-signed-motifs/<run-id>/`. Use organizer train only for fitting. Select via the kit's four training episode folds, then score the exact 48-episode tuning split with `run_model_comparison.tuning_data()` and the official `contest_score`. Reserve audit and test. Return a runnable command, fitted settings or checkpoint, verification results, `tuning_predictions.csv`, and `report.json` with condition ratios, points, paired episode interval against 33.224516, and runtime. See the [shared experiment contract](README.md#shared-experiment-contract).
