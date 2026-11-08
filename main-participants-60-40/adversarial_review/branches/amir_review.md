# Adversarial review: Amir models M1–M6

Reviewed snapshot: `amir_workspace`, commit `c339b469de33bbc2b2c68ea739eeec835139d42b`. Source paths below are relative to `adversarial_review/branches/amir/`. No model source was edited. The fixed M3 implementation was fitted once for a replay after explicit coordination with the parent reviewer; no new configuration search was introduced.

## Decision

**The explicit flip family, M3/M4, is the strongest credible new family.** Its broad improvement over the previous Bayesian switch model is supported by real, reproducible forecasts. M3 is the simplest recommended submission candidate; M4 is a reasonable alternative with a very small favorable observed difference. Do not promote M6 solely because it has the highest displayed score.

The parent review independently rescored all saved forecasts on the same 96 validation episodes. M3 scores 55.268370 versus 53.037228 for the prior Bayesian champion. Its paired delta R is -.045669, descriptive 95% episode interval [-.061022, -.031343], improving all four conditions and 79/96 episodes. The two old validation halves give M3 55.409835 and 55.123644. **Those halves are no longer independent audits of the Amir research process:** this branch previously evaluated all 96 episodes. They show consistency, not a newly untouched holdout.

Fresh M3 fitting plus validation prediction reproduced every saved probability within **1.33e-14** in **9.20 seconds** with two threads. The replay used the unchanged default fit, including its existing fixed training-selection grid. This substantially strengthens confidence that the saved score represents the reviewed code.

| Model | Full-validation result | Review disposition |
|---|---:|---|
| M1 hierarchical frequency | 31.09 | Valid weak fallback; does not model the dominant lag rule. |
| M2 Bayesian lag pair | 52.60 | Strong plausible model, but direction changes blur its cell rates. Superseded by M3/M4. |
| M3 explicit flip | 55.268370 | Recommended simple candidate; exact replay passed. |
| M4 shared flip | 55.28 | Clean extension; its small gain over M3 is supported on this validation sample, but practical benefit is tiny. |
| M5 boosted classifier | 54.99 | Negative result. Worse than M3/M4; CV is not fully cross-fitted and a history feature is incorrect. |
| M6 ensemble | 55.30 | Highest observed score; advantage over M3 is uncertain and it inherits M5 concerns. |

For M4 minus M3, parent rescore gives delta R approximately -.000287, interval [-.000566, -.0000684]. This is a small favorable interval, so describing them as an unqualified statistical tie would be inaccurate. For M6 minus M3, delta R approximately -.000554 has interval [-.001284, +.000161], including no improvement. M5 is worse than M3 by +.005536 R, interval [+.001256, +.010020]. These are descriptive intervals for already evaluated fixed forecasts, not selection-adjusted winner probabilities.

## Material findings

### 1. M5/M6 history feature depends on unrelated rows in the prediction batch

**Location:** `models/m3_lag_pair_flip/model.py:235-241`; consumed by `models/m5_boosted_corrector/model.py:25-26`, `:60-61`, and `:84-85`.

`recent = obs[:, -64:].sum(axis=1)` counts the last 64 columns of the *batch array*, whose width is the largest cutoff plus one. It does not count each machine's 64 ticks before its own cutoff, and it also includes the empty cutoff column even for the latest machine. In the released validation set this feature is incorrect on **749/768 rows**, with maximum undercount **64**. Actual last-64 support ranges from 40 to 64; implemented support ranges from 0 to 63.

A targeted check keeps all eight machines of a cutoff-192 episode and every observation unchanged. On its own, its feature values are 41–45. Adding an unrelated cutoff-256 episode changes all eight to zero. M3/M4 direct probabilities remain exactly unchanged; only their feature summaries are wrong. M5 consumes this feature and M6 gives M5 positive weight, so their inference need not be invariant to unrelated batches. This is an actual feature-construction bug, not target leakage. Saved forecasts use full-split batches consistently, so it does not make their existing validation scores fictitious.

Remedy before relying on M5/M6 in a general inference interface: count observed ticks satisfying `cutoff - 64 <= tick < cutoff` per row, then retrain affected learners. Do not patch the feature only at inference while retaining a classifier fitted to the old definition.

### 2. M5's advertised out-of-fold evidence is only out-of-fold for the final classifier

**Location:** `models/m5_boosted_corrector/model.py:54-75`; base fitting in `models/m3_lag_pair_flip/model.py:205-215`.

The M4 base is fitted on the entire training split before pseudo-query features are generated. Only then does `GroupKFold` separate episodes for the GBM. The base's context priors, pair prior, high/low rates, flip-time prior, shared prior and selected kappa already include the held-out episode, including its future outcomes relative to historical pseudo-cutoffs. The classifier's held-out episode has therefore contributed to its input-generating fit. The same flaw remains if a pre-fitted M4 is supplied by `models/run.py:40-43`.

The grouped split correctly keeps an episode's eight pseudo-cutoffs and machines together for GBM fitting. It does **not** make the whole stack out of episode. Fixing the evidence requires fitting M4 inside each outer training fold and building its training features with inner exclusion (or a fixed disjoint base-training set). This finding reduces trust in the CV and ensemble-weight justification, not the independently generated external validation predictions.

### 3. M1–M4 training-selection scores are not fresh out-of-sample observations

**Locations:** `models/common/selection.py:3-6`; `models/common/components.py:26-48`, `:99-110`; `models/m1_hierarchical_frequency/model.py:40-46`; `models/m2_lag_pair/model.py:179-191`; `models/m3_lag_pair_flip/model.py:205-215`.

The online filters correctly exclude the next observation from local counts, and `truncate` correctly hides pseudo-cutoff and later observations. However, `context_priors(train)` uses all training histories and cutoff labels first. M2/M3 then estimate empirical Bayes parameters using complete histories, before evaluating earlier pseudo-cutoffs from those same machines. This lets future within-machine outcomes and the evaluation labels influence the population parameters used to score those rows. The training score is useful for development but optimistic as generalization evidence. A check flipping only one actual training cutoff label changed its own win prior by 5.75e-5; EM parameter reuse is an additional path beyond that small context-prior effect.

This is **not** validation/test target leakage: fitting population parameters on all available train data is appropriate for final deployment. The misleading part is calling every scored training tick a fresh out-of-sample test. Episode-held-out population fitting would address the distinction.

### 4. M6's small training advantage lacks an outer assessment

**Location:** `models/m6_ensemble/model.py:41-66`.

Weights are fitted and assessed on the same pseudo-query outcome vector. M2/M3/M4 columns were generated from full-train fitted bases; M5's column has the partial cross-fitting problem above. M5's winning round count was selected using this same OOF outcome vector. The optimizer then reports its minimum loss without an outer holdout for weight selection.

Saved weights are **83.48% M3 + 16.52% M5**, effectively zero M2/M4. Rounded pseudo-query log loss improves only from .30325 for M3 to .30306 for the blend. The external validation gain is also very small and its paired interval versus M3 crosses zero. This is insufficient evidence to accept extra classifier complexity and the defective recent-history feature as a reliable test improvement.

### 5. M5 is a replacement classifier, not an offset residual correction

**Location:** `models/m5_boosted_corrector/model.py:47-52`, `:73-85`.

The training label `(observed payout > 0)` is valid. But the base logit is just a feature in a standard `HistGradientBoostingClassifier`, whose returned class probability replaces the base. There is no fixed base offset, no residual target, and no identity fallback at zero correction. Thus the statement that it “can only learn systematic miscalibration” (`:6`) is stronger than the implementation permits. Relearning a strong base from roughly 6,100 noisy outcomes can make it worse; the saved training and validation results both show that. This is a modeling-description and design limitation, not an invalid-label bug.

## Why M3 is materially different and plausible

The old champion's signed-lag model starts from a recency-based win logit, adds one of a small fixed set of signed amplitudes, and uses a per-tick hazard to redistribute amplitude mass while retaining pair mass (`slot_forecasting/bayesian_lag_pairs.py:50-57`, `:134-147` in the main checkout).

M3 instead enumerates **pair × current direction × one flip time**. It learns context-specific high/low win rates and population pair/flip priors from train. For each flip hypothesis it relabels observations before the flip into the opposite current cell (`models/m3_lag_pair_flip/model.py:114-117`) and uses Beta-integrated likelihoods (`:118-125`). This retains useful evidence from both sides of a direction reversal instead of forcing old and new regimes into one recent baseline. M4 adds posterior evidence for an episode-level common flip time (`:145-165`).

This mechanism is consistent with the branch's reported train-only discovery of direction reversals and with the strong external drift/shared-change improvement. Inference uses only the supplied pre-cutoff history. Remaining structural limits are one direction flip, an unchanged pair, a four-tick flip grid, and no flips within eight ticks of cutoff (`:42-49`, `:63-71`). These are genuine limits if the hidden regime changes, not evidence that the current score is invalid.

## Leakage and reproducibility checks that passed

- `models/common/data.py:88-92` rejects history at or after each machine's cutoff. `truncate` at lines 72–79 removes pseudo-target/future values and drops inappropriate target/reference arrays.
- Prediction functions for M1–M4 never read `labels`, `targets`, `reference` or `conditions`. Small checks with real history and fixed synthetic parameters scrambled those fields and obtained exactly identical predictions for all four models.
- Adding an unrelated later episode leaves M1–M4 direct predictions exactly unchanged when all machines of the focal episode are preserved. The M5 feature issue is isolated above.
- M2 emits its forecast before consuming that tick's outcome (`models/m2_lag_pair/model.py:112-144`). M3's likelihood evaluates all observations in the provided prefix, which is causal for its requested endpoint; using the entire *available prefix* is legitimate.
- A fresh complete M3 fit followed by validation prediction matched the saved CSV within **1.3322676295501878e-14**. Runtime was 9.20 seconds with two threads. This used Python 3.9.6, NumPy 1.26.4, SciPy 1.13.1; despite differing from pinned package versions the probabilities matched numerically.
- The branch's runner actually writes test submissions and checks their IDs/probabilities (`models/run.py:69-74`, `:94-97`). Unlike the earlier main-branch tuning-only runners, the test-output workflow exists. Saved `params.json` descriptions contain rounded/incomplete learned parameters, however, and M5 does not save its classifier; reproducibility depends on deterministic refitting rather than reloading those JSON descriptions.

## Review artifacts

- `adversarial_review/branches/amir_checks.py` and `.json`: small invariance checks, exact recent-history mismatch counts, and self-inclusion of training cutoff labels in context priors. Synthetic fixed parameters are explicitly used for the invariance checks; these are not performance claims.
- `adversarial_review/branches/amir_m3_replay.py`, `.json`, `.npz`: unchanged default M3 fit and saved-forecast replay, including probability arrays and artifact hash.

No demonstrated direct validation-target leakage or inference lookahead invalidates M3's improvement. Its full-validation superiority is credible; selection among M3, M4 and M6 remains much less certain than the broad improvement over the previous champion.
