# Fredy branch review: GLM, GBDT, regime, pooled, and stack

Snapshot: `fredy/bakeoff-gbdt-seq`, commit `5ba591c91d3589a2511b946dcf908b194e1390aa`. Reviewed the snapshot under `adversarial_review/branches/fredy`, after reading its `AGENTS.md`. Paths and line references below are relative to that snapshot. This review made no model-source changes, installed nothing, ran no full-data training, and opened no hidden test answers. Sequence-model internals, its package blocker, the shared score harness, and cross-branch score reconciliation belong to the other reviewers.

## Decision

None of these four model families is a credible replacement for the leading sequence/Bayesian forecasts on current evidence. Fredy's **transductive GBDT is clearly strongest among the four reviewed families**. The hierarchical GLM offers no convincing gain over a cheap recent-frequency baseline; regime routing is tied with or worse than that baseline. The factorized GLM is implemented and numerically checked, but has no recorded real-data score and should remain unranked.

The table contains the branch's **reported development scores on all 96 validation episodes**, not the other suite's 48-episode tuning scores. The logged runs were not freshly rescored here because trained artifacts and prediction files are git-ignored and absent from this snapshot.

| Family / variant | Reported points / 60 | Evidence interpretation |
|---|---:|---|
| GBDT `trans_64_l15` | **36.395373** | Repeated best recorded run; 36.225856 and 36.191118 also appear under the same label. |
| GBDT `all_inf_l15` | 35.123478 | Best nontransductive tree configuration in the retained pilot table. |
| GBDT `all_64_l15` | 34.361767 | Other tree settings range down to 33.087621. |
| GLM `l8` | 31.093246 | +.267 points versus pooled recent; paired interval [-.426,+.997], so no clear improvement. |
| GLM `l4`, `lambda100`, `l2` | 30.963001 / 30.878279 / 30.844023 | All reported paired intervals versus pooled recent contain zero. |
| Pooled `pooled_recent` | **30.825854** | Best low-complexity baseline; useful fallback. |
| Regime tuned `d2_routed` | 30.779477 | -.046377 versus pooled recent, interval [-.355,+.244]. Exploratory whole-validation score after specialist tuning. |
| Pooled `recent` | 30.743191 | Slightly below pooled recent. |
| Regime default `d2_routed` | 30.476064 | Below baseline; default result must not be confused with the tuned override. |
| Regime `d2_d3_mix` | 30.187244 | Adds an ineffective changepoint forecast. |
| Pooled `pooled` | 30.000000 | Frozen-reference reproduction. |
| GLM `lambda1` | 29.758615 | Weak local shrinkage hurts; reported interval versus pooled recent excludes zero negatively. |
| Regime `d3_changepoint` | 29.709885 | Worse than pooled recent and even the frozen pooled reference. |
| Pooled `context` | 29.685186 | Static population forecast is insufficient. |
| GLM factorized / `tauinf` | No retained real-data score | Do not infer performance from synthetic correctness tests. |

Evidence: `claude_output/notes/pilot_results.md`, `runs/LOG.md`, and `claude_output/notes/W4_regime.md`. GLM `tau32` has both 29.315303 and a corrected later 30.745459 entry; the latter is the retained pilot figure. Log labels omit material overrides, seeds, smoke settings, and source revisions, so they are not uniquely reproducible model identifiers.

## Findings that affect interpretation or use

### 1. Final fitting is a different estimator from the scored development model

`src/tdx/fit.py:10–14` permits train-only labels during development, then all three histories, train+validation labels, and validation true probability vectors in final mode. This is explicitly authorized by the branch's rules and is **not hidden-answer leakage**. But the reported development score does not measure the final recipe.

The difference is especially large for GBDT: `src/tdx/arms/gbdt/arm.py:42–56` adds **no cutoff supervision at all in development**, then adds hard cutoff labels and expanded soft targets in final. A validation cutoff is represented once by its realized outcome and once by its exact distribution, both with total weight `w_cut`. This intentionally blends noisier one-hot supervision with the available soft distribution; it is not equivalent to fitting the true distribution alone. The final tree is then refitted on all rows at the selected iteration (`:71–74`). The GLM also appends both kinds of supervision (`glm/arm.py:55–64`).

`src/tdx/pipeline.py:59–67` fits blend weights to development validation forecasts, then swaps in final-refitted arms. These weights need not remain optimal after that change. The code correctly prints that no CV score is claimed for the final blend (`:64`); preserve that distinction in any winner claim.

### 2. The stack's default routing has two effective groups, not four learned regimes

`src/tdx/package/blend.py:19–25` sets the posterior to `[0,1,0,0]` for exactly-64-observation episodes and `[1/3,0,1/3,1/3]` otherwise. `fit` always uses this fallback posterior (`:82`), and the standard pipeline supplies no trained `condition_arm` (`pipeline.py:62`). Consequently stable, drift, and shared-change weight rows are mathematically unidentifiable separately: only their average matters. Starting from identical rows, the optimizer leaves those three rows equal. A synthetic direct test confirmed this.

This is a valid short-versus-long-history blend. It should not be described as successfully learning four regime-specific model weights. Inference can consume an explicit condition provider (`package/predict.py:66–71`), but weights fitted with fallback routing should not simply be paired with a different posterior without matching the fitting procedure. No such mismatch was established for the saved default pipeline.

### 3. Regime-classification accuracy is not evidence of forecasting improvement

The condition classifier uses genuine past-only episode summaries, standardizes inside each CV fit, and reports nested five-fold episode CV (`regime/classifier.py:101–107,134–173`). The deterministic 64-observation shortcut is valid for this kit (`:88–89`). Hard routing is restricted to that known short-history case; the other condition predictions enter a **soft** weighted mixture (`regime/arm.py:41–43`).

However, classifier CV accuracy is .59375, with validation accuracy .520833, and stable/drift/shared conditions are frequently confused (recorded `W4_regime.md`). The specialist forecasts are only frequency-plus-lag-one counts with different decay/strength settings (`regime/specialists.py:27–53`). Even perfect condition labels would not supply the missing higher-order machine-specific lag mechanism.

The changepoint threshold is selected for unweighted binary accuracy of `shared_change` versus everything else (`regime/changepoint.py:21–28,103–113`), not probability-forecast loss. Because shared change is 25% of episodes, always predicting no shared change achieves .75 accuracy. Reported changepoint CV accuracy .739583 is below that trivial baseline, and the fitted threshold fires on no training episodes and only one of 24 validation shared-change episodes. Thus the detector largely falls back to full-history counts (`regime/arm.py:45–48`). That explains the poor score; its high-looking accuracy is not useful discrimination.

The specialist tuning utility is separate from development fit and correctly selects 12 episodes per condition before scoring the other 12 (`regime/tune.py:33–43`). The recorded tuned score is 31.191896 on tuning, 30.328648 on the reserved half, and 30.779477 on their combined set. The combined score and its bootstrap remain exploratory after tuning, as the notes acknowledge.

### 4. GLM adaptation is causal, but the feature family is poorly matched to pair mechanisms

Unlike a global logistic model, Fredy's GLM genuinely fits separate episode-machine coefficients at prediction time, penalized toward global `W0` (`glm/arm.py:80–88`). It can therefore learn opposite machine-specific **single-lag** signs; both its existing planted-lag tests and numerical probes pass.

Its lag feature matrix is nevertheless additive: individual lag-category indicators, peer counts, EMA rates, and last-20-observed win rate (`glm/features.py:13–24,41–82`). There are no explicit pair products, parity states, or per-machine pair posterior features. A general XOR/parity relation is not representable by additive lag logits alone. Lag histories can correlate with its other features, so this is an expressiveness limitation, not a claim that every pair-driven example is unlearnable. Weakening local shrinkage to `lambda1` increases noise and worsens the recorded score.

Cutoff `signal` is absent from both historical and final GLM feature matrices. This avoids inventing historical signals, as required by `AGENTS.md`, but also discards a valid cutoff covariate. A separate cutoff-only signal correction was not implemented. Factorizing the likelihood into win and payout size is mathematically correct (`glm/factorized.py:14–29`) and retains positive-mass weighting; it does not itself add missing lag interactions.

The optimizer only rejects nonfinite outputs, not a nonconverged finite solution (`glm/model.py:34–44`, `glm/factorized.py:38–49`). Global diagnostics are persisted, while local fits do not retain diagnostics (`glm/arm.py:85–87`). The historical review documents raised caps and improved convergence, but this code still cannot certify every local final fit converged. This is a monitoring limitation, not an observed failing probe in this review.

### 5. GBDT features and episode holdout are sound; transfer still depends on aggregate evidence

GBDT uses real tick lags, past-only counts, observed-time EMAs, calendar-window divergence, and neighbor observations (`gbdt/features.py:19–95`). Missing windows remain missing rather than becoming losses. Its last-observation neighbor feature is allowed to carry an older observation forward, while larger windows pool actual observed cells. The scalar reference checked every feature on all 2,328 synthetic rows, including empty prefixes and the next tick, and passed. Future perturbations leave earlier features and cutoff features unchanged.

The early-stopping split removes **whole train episodes** via hashes (`gbdt/arm.py:58–72`). Validation/test histories can supply permitted self-supervised fitting rows in transductive development, but do not become early-stopping target episodes. No shared learned feature normalizer or full-episode target encoding leaks held-out outcomes into those features. Final iteration count is refitted on all allowed rows.

There is no machine ID or per-machine trained parameter in the tree. Its only explicit joint code is lag1/lag2 (`gbdt/features.py:33`); trees can combine any supplied lags, but must learn pooled interactions and infer latent machine behavior indirectly from rate summaries. That is weaker structure than a machine-local pair-state model. The large transductive advantage in the log is credible as additional allowed histories, not proof the same gain would survive a task that withheld test histories.

Native LightGBM was not installed in the available environments, so this review does **not** independently certify native booster fitting, save/load determinism, or the reported tree scores. An engine-argument capture did verify exact final soft-target expansion/weights and that pilot training plus held-out rows partition final fitting rows.

## Signal use, priors, transforms, and holdout integrity

- GLM, GBDT, regime classifier, and regime specialists do not use query `signal` as a historical tick feature. Changing signal cannot affect their history feature matrices. This complies with the branch instruction that only cutoff signals are released. Their final forecasts also do not exploit signal.
- The pooled implementation intentionally groups a machine's training history and cutoff outcome under its **cutoff context** `(type,signal)` to reproduce the released kit baseline (`pooled/arm.py:32–65`). It is an aggregate cutoff-conditioned estimator, not a fictitious stream of tick-level signal labels. Its design can dilute the latest signal's relation with the next outcome, but matches the benchmark's construction.
- Pooled population fitting reads only explicitly authorized `label_splits`; it deliberately ignores transductive tick-only splits (`pooled/arm.py:12–16,45–56`). Pooled inference masks every observation at/after cutoff and recomputes observation counts (`:37–40,79–88`), rather than trusting cached `n_obs`.
- Regime priors pool only `spec.tick_splits` (`regime/arm.py:27`), while its condition/changepoint classifiers read only explicitly supplied labeled splits (`:24–26`). Nested classifier scalers are trained within each fold. No held-out outcome-derived prior was found in this path.
- Missing values remain unknown in local transition matching, lag features, EMA denominators, and changepoint counts. Clipping at `1e-6` followed by normalization is applied consistently in the arms; it is a small probability floor, not the scoring objective.
- GLM and GBDT train likelihoods, whereas the contest ranks squared probability error against true distributions with reference scaling. Likelihood is a reasonable surrogate, but it makes tiny pilot/CV differences less persuasive. GBDT's early stopping uses log loss on historical train episodes, not the final cutoff distribution-error metric.
- Both transductive use of public histories and final use of released validation targets are explicitly allowed by this snapshot's `AGENTS.md`. They should be disclosed, not mislabeled as hidden-test leakage. Development hyperparameter comparisons on the same full validation set still create selection optimism; paired intervals for fixed fitted candidates do not adjust for searching variants.

## Targeted verification and limits

`adversarial_review/branches/fredy_other_checks.py` directly invoked **24 original test functions** plus **2 focused checks**, all passing; machine-readable details are in `fredy_other_checks.json`.

Coverage included GLM and factorized weighted gradients, loss-only payout-gradient behavior, single-lag recovery, save/load equality, future-storage invariance, GBDT scalar feature oracle and missingness/neighbor regressions, classifier nested-CV serialization and legal supervision checks, specialist post-change behavior, changepoint recovery/causality, GBDT fitting-row capture, and the blend's two identifiable routing groups.

The installed Python 3.10 NumPy worked, but importing its SciPy/sklearn failed because the installed `_propack` extension could not load. The working Python 3.9 environment supplied NumPy 1.26.4, SciPy 1.13.1, and scikit-learn 1.6.1. A read-only import loader postponed type-annotation evaluation for `tdx` modules so Python 3.9 could execute the unchanged numerical source. Pytest was absent, so test functions were called directly with a small equivalent `raises` context helper. LightGBM was absent; its engine was only stubbed for the explicitly labeled fitting-row capture, never for a numerical training claim.

This is targeted correctness evidence, not the required Python 3.13 offline package reproduction. The independent sequence reviewer owns the concrete package rejection caused by the static import graph scanning its training-only PyTorch imports. The parent reviewer owns the current pipeline configuration and score-harness audit. No new upload files were generated by this review.
