# Bayesian lag-pair forecasting plan

## Objective

Build a Bayesian win model that can learn which two lags matter for each machine, how strongly they interact, and whether the interaction changes during the observed history. Carry forward the self-only logistic conditional payout mix so differences in official score reflect changes in win probability. Compare against the frozen **50.273957 / 60** tuning lead and keep the audit reserved. [Lead results](EXPERIMENT_RESULTS.md).

## What the signed-motif experiment established

The existing model averages a no-lag expert, eight singleton motifs, and 28 pair motifs. For lags `i < j), it uses the product of the two win/loss signs, so it can learn that either agreement or disagreement raises win odds. Each motif has four possible signed log-odds effects, `{-2, -1, +1, +2}`, applied to the 12-tick recency forecast. The training-selected model used a 48-tick evidence half-life. It scored **42.026** with its original payout mix and **42.402** with the same logistic payout mix used by the lead. [Model source](slot_forecasting/signed_motifs.py), [signed-motif report](runs/01-signed-motifs/replay/report.json), [head-swap report](runs/06-combine-forecasts/replay/report.json).

| Forecast using the same logistic payout mix | Points / 60 | Stable R | Short history R | Drift R | Shared change R |
| --- | ---: | ---: | ---: | ---: | ---: |
| Signed motifs | 42.402 | **0.1111** | **0.1822** | 0.7610 | 0.6058 |
| Boosted adaptive lead | **50.274** | 0.1527 | 0.2191 | **0.1912** | **0.2108** |

On the 48 tuning episodes, the existing model assigned mean posterior mass of **94.1%** to pairs on stable episodes and **77.3%** on short histories, versus **14.3%** on drift and **31.3%** on shared change. Among pair hypotheses, 86.4% of stable and 83.6% of short-history posterior mass sat at the grid edges, `|a| = 2`, versus 2.9% on drift and 5.4% on shared change. Median usable historical observations per pair were 35 on short histories and 208.5–224 on the other conditions. These are diagnostics of the fitted model, and the condition labels are used only to summarize them. They motivate testing wider effects separately from testing changes in effect direction. They also leave open the possibility that drift and shared change involve patterns other than binary parity. [Signed-motif runner](run_signed_motifs.py), [data contract](main-participants-60-40/DATA.md).

The earlier episode-cross-fitted blend of boosted and signed-motif win probabilities with the logistic payout mix scored **50.237**. The next model needs to improve the win forecast itself or supply complementary information that survives a matched comparison. [Prior combination experiment](runs/06-combine-forecasts/replay/report.json).

## Research that informs the design

| Primary source | Application here |
| --- | --- |
| Bayesian Context Trees integrates over discrete Markov contexts and integrates out each context's categorical parameters under conjugate priors. | Use a small, explicitly enumerated family of candidate lag pairs and average its forecasts by posterior weight. A conjugate pair table gives a transparent control. Our candidate pairs can skip ticks, so the usual contiguous suffix tree does not directly express the intended hypothesis space. [Kontoyiannis et al.](https://arxiv.org/abs/2007.14900) |
| Bayesian higher-order Markov modeling can select a sparse set of relevant lags while capturing interactions among them. | Include all 28 pairs up to lag eight, allowing nonconsecutive lags. Compare two parity groups with a four-cell joint context that can express asymmetric effects. Enumeration fits this small hypothesis space. [Sarkar and Dunson](https://arxiv.org/abs/1506.06268) |
| Expert switching can be represented as a hidden Markov model and computed by forward filtering. | Keep a pair's identity fixed within each hypothesis and let its coefficient change through a small transition matrix. This targets direction changes inside a pair, whereas the earlier regime experiment modified the already-averaged win forecast. [Koolen and de Rooij](https://arxiv.org/abs/0802.2015) |
| Online changepoint detection updates predictions from past observations using a hazard. Work on discrete changepoints also combines changing regimes with Bayesian context models. | Start with a constant reset hazard on the coefficient and check reversals on synthetic data. A run-length model or switching pair identity is a follow-up if the simpler filter leaves a clear failure mode. The segmentation paper uses retrospective inference, while this task requires forward prediction. [Adams and MacKay](https://arxiv.org/abs/0710.3742), [Lungu et al.](https://arxiv.org/abs/2203.04341) |

## Model to implement

Add `slot_forecasting/bayesian_lag_pairs.py` and `run_bayesian_lag_pairs.py` at the repository root. Reuse the type/signal priors and recency counts from the signed model, the tick arrays and masks from `slot_forecasting/causal.py`, and the existing grouped folds and tuning loader. Keep a `fit(queries, history, labels)` and `predict(queries, history)` interface. Run on CPU with NumPy and SciPy. [Existing rollout](slot_forecasting/signed_motifs.py), [tick alignment](slot_forecasting/causal.py).

### 1. Reproduce and widen the signed model

For an observed lag, let `s_i(t) = 2*1(payout[t-i] > 0) - 1`. Under a pair hypothesis the win probability at tick `t` is `sigmoid(logit(b_t) + a*s_i(t)*s_j(t))`, where `b_t` is the current 12-tick recency win forecast from the prefix. If either lag is missing, use `b_t` for that hypothesis. Keep the null/singleton/pair prior mass at `0.50 / 0.25 / 0.25` and distribute mass uniformly among the lags or pairs within each family.

First reproduce the existing 48-tick evidence model. Then add one wider-grid variant for **pairs only**. Give `±1` and `±2` probability 0.20 each, `±3` probability 0.075 each, and `±4` probability 0.025 each within a pair. Keep the singleton grid, the discount, and the family priors fixed. This isolates the effect of letting strong interactions move farther away from the base odds.

### 2. Let the sign and magnitude change within a pair

Use the original four amplitudes and start with uniform conditional mass `rho(a)` within a pair. Carry joint posterior mass `q[p,a]` over pair identity and amplitude, alongside static null and singleton hypotheses. Before each prediction tick, redistribute only within each pair.

```text
pair_mass[p] = sum_a q_post[p,a]
q_pre[p,a] = (1-h)*q_post[p,a] + h*pair_mass[p]*rho[a]
win[t] = sum_hypotheses q_pre[hypothesis] * P_hypothesis(win at t)
```

For an observed outcome, multiply each predicted mass by its Bernoulli likelihood and normalize across **all** hypotheses. For a missing outcome, apply the transition without a likelihood update. Missing lag slots use the null emission for that pair, so the current outcome gives that pair zero evidence relative to null. Advance the transition and recency decay in real ticks, including gaps. Update recency counts after emitting the forecast and processing the likelihood. Average probabilities across states. Use stable log-space normalization.

Compare `h = 0, 1/96, 1/32` per tick, with the evidence discount disabled for this family. The `h=0` filter must match the existing **undiscounted** signed model. The 48-tick discounted model stays a separate control. Test the best nonzero hazard with the widened grid only if both changes help independently on grouped training. The core test asks whether a machine can keep its pair identity while changing the direction or size of the effect.

### 3. Compare with a conjugate four-cell lag model

Fit a simpler Bayesian table to test whether binary parity is too restrictive. For each pair keep wins `W_c` and losses `L_c` for either two parity groups or four binary lag states `00, 01, 10, 11`. Use the same state prior for both controls. Let `m` be the type/signal prior win rate estimated on the training fold and start at `kappa = 8`.

```text
theta_c ~ Beta(kappa*m, kappa*(1-m))
P(win | cell c, history) = (W_c + kappa*m) / (W_c + L_c + kappa)
```

Include a no-lag Beta-Bernoulli expert and singleton tables. Use their prequential predictive likelihoods to update model weights. If a lag is unavailable, use the no-lag forecast for that table and skip its cell update. A missing target updates no counts or likelihoods. Run both table variants with the same fixed priors and without decay. The extra cells buy flexibility at the cost of fewer observations per cell. Use the existing logistic payout mix on tuning for both tables.

Consider a pooled prior on *pair identity* only after the uniform-prior comparison. Estimate prevalence from machine posteriors in training episodes only, shrink strongly toward uniform, and keep signs symmetric. Fit the prior again inside each training fold. This is an optional adaptation experiment.

## Evaluation protocol

1. **Select on grouped train.** Use the same four condition-stratified folds from `run_model_comparison._train_folds` with all eight machines in an episode kept together. Use the observed training cutoff outcomes and up to the last 32 observed historical labels per machine. A held-out machine can update local state from its own prefix. Fit type/signal priors and any shared settings on other episodes. Choose one Bayesian candidate by episode-weighted cutoff win Brier. Report win log loss, historical Brier, and condition slices as diagnostics. Freeze the grid and selection rule before scoring tuning. The initial grid is the 48-tick control, wider grid, three hazard settings, and the two conjugate tables. [Current fold and scoring helpers](run_model_comparison.py).
2. **Run the official tuning comparison.** Use only `tuning_data(DEFAULT_PUBLIC / "validation")` and assert 48 episodes, 12 per condition, with exactly 384 matching query IDs. Join the selected model's win probability `w` with the existing self-only logistic positive mix `r`, derived by dividing its four positive probabilities by their sum. Use the combiner's training-prior fallback if that sum is zero. Emit `[1-w, w*r_1, w*r_2, w*r_5, w*r_20]`. Load the frozen lead by query ID and verify it scores **50.273957** and `R = 0.1934608563` before comparing. Record its SHA-256 hash, `0530eb2bbd4e3738c7df88c3412e5501a65da7e21f23355e0db9585c4d1d6fd5`. Call `contest_score` with the round's maximum points and scorer version. Score the signed-motif control with the same logistic head. [Frozen lead CSV](runs/06-combine-forecasts/replay/fixed__boosted_3072_logistic_head.csv), [logistic forecast CSV](runs/model-comparison/two_stage_self.csv), [official scorer](slot_forecasting/contest.py).
3. **Measure a blend separately.** Prespecify one fixed blend with 25% candidate win probability and 75% lead win probability, holding the logistic payout mix fixed. If fitting a scalar blend, use the existing episode-cross-fitted combiner protocol. Fit its weight and frozen-reference denominators on the other three tuning folds, then assemble held-out forecasts. Report that result separately as exploratory tuning evidence, alongside the standalone and fixed-blend scores. [Combiner](run_combine_forecasts.py).
4. **Report gains and tradeoffs.** Report official points, overall `R`, each condition ratio, squared error by payout, raw win calibration, and the paired episode bootstrap interval for `R(candidate) - R(lead)` using `paired_ratio_interval`. Record pair posterior entropy, expected signed amplitude, edge mass, usable lag support, and regime transition behavior. Consider a candidate for a later audit if tuning `R` improves by at least 0.01 and no condition ratio increases by more than 0.05. Report the interval and treat repeated comparisons on this tuning set as exploratory. [Paired interval helper](slot_forecasting/tuning_metrics.py).

The runner must select the tuning IDs before loading or decoding validation history, target, and reference values. Use condition labels for folds and reports only. Keep audit histories and answers out of model input, fitting, calibration, and scoring. Use no test data for this experiment. The general starter runner and full-validation checker evaluate beyond the requested tuning subset. [Tuning loader](run_model_comparison.py).

## Verification and deliverables

- **Known answers.** Reproduce the existing signed model with the original grid and discount. Confirm the zero-hazard limit against the undiscounted signed model within `1e-12`. Hand-check the table update and verify that the sum of log predictive probabilities matches the Beta marginal likelihood for an ordered, fully observed toy sequence. Compare a tiny switching example with exhaustive enumeration of the amplitude state paths.
- **Synthetic discrimination.** Use seeded machines with the same planted pair and opposite effects, a sign reversal within one machine, an asymmetric four-cell pattern, and a null. Include histories with realistic sparse older ticks. On reversal simulations report recovery delay and post-change Brier against the static model. Check that flexible variants preserve stable and short-history performance.
- **Causality and output checks.** Change a payoff at or after a synthetic cutoff and assert the earlier prediction stays fixed. Verify that missing lags and recorded 0× produce different behavior, that a missing target contributes no likelihood, and that all rows remain finite and normalized. Exercise the tuning filter with synthetic audit rows with invalid numeric values and assert those rows never enter the scoring loader. Verify exactly one output row per selected query.
- **Replay.** Save the selected settings, fold IDs, evidence diagnostics, tuning predictions, per-condition scores, paired intervals, file hashes, and a command under `runs/bayesian-lag-pairs/<run-id>/`. Save distinct standalone and blend CSVs. The runner should support `--data` and `--out`, with a default data path from `DEFAULT_PUBLIC`. Replay to a fresh directory with two CPU threads and require at most `1e-8` normalized difference. Keep the frozen lead CSV unchanged and reserve the audit for a separate decision.
