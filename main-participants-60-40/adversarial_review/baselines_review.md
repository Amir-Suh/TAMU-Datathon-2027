# Adversarial review: baselines, logistic models, peers, regimes, and public input shift

Reviewed 2026-11-07. Source paths below are relative to the repository root (`tamudatathon27`), not the participant-kit directory. This review changed no model source and performed no training or new target scoring. New diagnostics used public query/history inputs only. The Bayesian diagnostic loaded the already frozen population prior, ran the uniform-pair switch-96 model, discarded forecasts, and saved aggregate posterior summaries. No test labels, conditions, reference, or targets were available or accessed. Validation target rows were not opened by these diagnostics.

## Recommendation and ranking

The frozen Bayesian switch-96 model with the logistic positive-payout head remains the most defensible test choice given the previously recorded audit advantage reported by the parent review. The new rho=.8 learned pair-prior variant has a credible transfer mechanism, and public test histories do not contradict its three dominant lag pairs, but this descriptive check cannot replace its unmet promotion gate or supply a new independent audit.

Among the families reviewed directly here, the evidence ranks adaptive rate 3072 as the useful control; peer corrections are indistinguishable from it; regime filtering offers no convincing improvement and one variant clearly hurts. Logistic regression is useful as a positive-payout head, but its standalone win forecasts, simple frequency estimators, and fixed transitions are far behind the leading adaptive/Bayesian models. The learned transition-retrieval result is stronger than those static baselines but below adaptive 3072. The broad tuning order across the existing reports is Bayesian switch96 (53.382758), boosted3072 plus logistic head (50.273957), boosted3072 (49.744), adaptive/peers (~44.10), regime (~43.95), signed motifs (~42.03), learned retrieval (~41.23), boosted48 (~37.78), adaptive48 (~33.22), then CNN/logistic/fixed retrieval/frequency (~31–32). The rho=.8 candidate scores 53.799644 on tuning but is not established as the test winner. This is an evidence ranking, not an observed test leaderboard.

## Material findings

1. **Historical pseudo endpoints have materially less usable history than actual cutoffs, especially in the short-history condition.** `slot_forecasting/win_payout.py:15–26` and `slot_forecasting/peer_lags.py:97–102` take the last 32 observed historical outcomes plus the real cutoff. Historical samples are on average 16.5 ticks earlier and contribute 32/33 of the logistic fitting rows. In training, actual cutoffs have 40/40 recent observations for every machine. Pseudo endpoints average 36.3986/40, with a minimum of 9. For short-history machines the mismatch is 25.5946/40 versus 40/40, and 47.5 total preceding observations versus 64. This can change confidence, smoothing, and feature distributions; it is not label leakage. Reported historical performance should not be treated as a direct estimate of next-spin cutoff performance. The runners' decision to select by held-out **cutoff** Brier mitigates this, but does not change the supervised fitting-row mixture. Every sampled pseudo endpoint still has all preceding 8 lags observed, so missing lag flags themselves are not the mismatch here.

2. **The regime runner exports a nonbase filter even when training selects the base.** `run_regime_filter.py:193–198` correctly includes the base in its selected winner, but `run_regime_filter.py:267–270` chooses `lead` from nonbase variants and writes that forecast to `tuning_predictions*.csv`. Both rate-48 and rate-3072 reports name the base as the training-CV winner. At rate 3072, fixed-state filtering scores 43.949860 versus base 44.097892; its paired delta-R is +.004583, CI [-.003478, +.013412]. Constant-hazard filtering scores 43.692490, delta-R +.012624, CI [+.002461, +.024470]. The report is candid, but a downstream reader must not interpret the conveniently named prediction CSV as the overall training-selected winner. `OnlineRegimeForecaster` also defaults to a direction/hazard filter (`slot_forecasting/regime_filter.py:26–29,84–87`), not the winning no-filter control.

3. **Peer corrections have no demonstrated incremental signal and substantial selection noise.** `slot_forecasting/peer_lags.py:185–225` estimates candidates causally, but ranks 56 individual peer-lag choices using only the last 12 eligible scored observations (`:195,208,216–220`) and chooses the highest gains (`:230–245`). Counts reported as support are cumulative rather than the effective exponentially weighted support (`:223–225`); large displayed counts do not mean the selected recent gain is precise. At rate3072, individual peers gain only .018861 points over self-only: delta-R -.000582, CI [-.006087,+.004583]. Aligned versus 11-tick shifted peers also has a null interval: -.001595, CI [-.011177,+.007150]. The selected peer mechanism activates on 328/384 endpoints, exactly the same total activation count as the shifted control. Neither placebo nor matched-control evidence justifies promoting peers. The shortlist gate correctly rejects the candidate (`run_peer_lags.py:220–222`).

4. **The standalone logistic model lacks a mechanism for identifying each machine's sign and lag structure.** Its static machine information is type and signal, followed by aggregate rates, recent lag values, and pair indicators (`slot_forecasting/causal.py:41–62`). The heads are global linear logistic regressions (`slot_forecasting/win_payout.py:30–34,61–78`). It can represent a common pair effect, but does not infer a machine-specific pair/sign hypothesis from that machine's preceding transition evidence. Opposing machine-specific effects can average away. The existing tuning scores corroborate this limitation: self-only two-stage 31.443817, direct five-class 31.353549, peer two-stage 31.305904, versus adaptive3072 44.097892. Its pooled conditional positive payout head is still valuable when attached to a much stronger win predictor; failure of the standalone win head does not invalidate the payout head.

5. **Population priors overweight long histories, whereas scoring balances conditions and episodes.** `slot_forecasting/baseline.py:38–45` pools every observed outcome, so training's 192 short-history machines supply only 12,288 of 139,944 history observations (8.78%), despite representing 25% of machines/episodes. Adding one cutoff label per machine barely changes this imbalance. That is a possible population-prior bias, not leakage; per-machine recent evidence and context-specific priors reduce its effect. Frequency baselines cannot express lag/parity dependence; `TransitionForecaster` restricts matching to fixed one-to-three lags and a ternary payout category (`slot_forecasting/transition.py:6–13,16–25`), and shrinks toward an all-history local baseline (`:34–45`). That makes drift and discovering unknown machine-specific lag pairs difficult. No comparable frozen standalone score for every simple transition setting was established in this review.

6. **Bayesian switching only changes amplitude within a pair.** `slot_forecasting/bayesian_lag_pairs.py:50–57,134–137` resets amplitude mass while preserving each pair's total mass. A real change in lag-pair identity could therefore be slow to recover from. The observed public data strongly support stable pair identities, so this remains a structural risk rather than an observed test failure. Posterior concentration is model evidence, not ground-truth verification.

## Leakage, missingness, and scoring checks

- `slot_forecasting/schema.py:138–155` rejects duplicate histories and observations at/after each machine's cutoff. `causal.py:45–47` constructs a padded prefix that excludes tick t from feature t; lag indexing reads t-1 through t-8. Peer prefixes sum only machines in the same episode and exclude self (`:55–62`). Missing observations get an explicit lag category (`:48`); rolling totals accompany zero-filled rates (`:32–38`). The combination of flags and totals avoids silently equating all missing values with actual losses.
- `run_model_comparison.py:58–68,83–112` splits by episode with condition-stratified folds, fits standardization only inside each fold via `WinPayoutForecaster.fit_rows`, and selects C by cutoff Brier. All peers of an episode are in the same fold. Building purely causal, label-free features before the fold mask is safe here. No episode-ID feature or cross-split episode overlap was found.
- Peer and regime updates occur after each forecast; endpoint regime outcomes explicitly exclude the endpoint (`regime_filter.py:70–76`). Their population priors are fitted outside the held-out training episodes by the corresponding runners. These checks found no material future-observation or episode-label leakage in the reviewed paths.
- The event metric is equal-weight condition mean of probability-error/reference-error, with episode means inside each condition (`slot_forecasting/contest.py:31–42`; `scoring.py:67–93`). One-hot Brier on training outcomes is an unbiased proxy for probability error up to irreducible outcome noise, but is noisier and lacks condition-specific reference scaling. Balanced folds help; tiny cutoff-CV differences remain weak evidence.
- The 20x payout does **not** receive 20 times the probability-error weight. With win probability w and conditional payout mix s, the prediction is `(1-w, w*s)`. If the payout mix is correct, squared probability error is `(1 + sum(s**2)) * (w-u)**2`. This explains why improving win probability dominates and why a useful payout head can add a smaller but genuine improvement. Expected-payout MAE is diagnostic, not the ranking objective.

## Public train/validation/test distribution

| Visible property | Train | Validation | Test |
|---|---:|---:|---:|
| Episodes / queries | 96 / 768 | 96 / 768 | 192 / 1536 |
| Cutoff range | 192–256 | 192–256 | 192–256 |
| Mean cutoff | 222.667 | 222.802 | 223.828 |
| Mean history observations | 182.219 | 184.104 | 184.026 |
| Machines with exactly 64 observations | 25% | 25% | 25% |
| Available lags 1–8 at final cutoff | 8/8 for all | 8/8 for all | 8/8 for all |
| Observed prior 40 ticks at final cutoff | 40/40 for all | 40/40 for all | 40/40 for all |
| Mean prior-32 win fraction | .429240 | .406820 | .393778 |

Training and validation have exactly 24 episodes in each released condition. Metadata promises equal test-condition counts, but test conditions are private, so this review does not infer their identities beyond the directly visible 64-observation stratum. All machines within each episode share a cutoff. Split episode IDs are disjoint.

There is a modest marginal behavioral shift: test minus train recent-32 win fraction is -.035461 with an episode-bootstrap 95% interval [-.057170,-.013871]. Matching training's nine type/signal context means to test's type/signal frequencies reduces the difference to -.025719. This is an observable history difference, not a measurement of hidden next-spin probabilities. It argues for local adaptation and cautions against relying heavily on a static global win prior. It does not establish a new test-generating process: cutoff/exposure patterns, short-history prevalence, and recent rate-change proxies are similar.

The final cutoff distribution is consequently **not interchangeable with pseudo historical endpoints**, particularly their available older history. However, final training, validation, and test endpoints are closely matched on the visible exposure design. No evidence here supports an abrupt hidden-target-specific distribution switch.

## Pair-prevalence transfer diagnostic

The already frozen Bayesian switch96 model was applied with a **uniform pair prior** to all public histories. This avoids assuming the rho=.8 learned prior while testing whether its preferred pairs recur. The three dominant learned-prior pairs are (1,3), (2,5), and (4,7).

| Posterior diagnostic | Train | Validation | Test |
|---|---:|---:|---:|
| Mean conditional pair mass on those three pairs | 94.93% | 93.68% | 93.99% |
| Episode-bootstrap 95% interval | 93.04–96.65% | 91.41–95.79% | 92.48–95.33% |
| Fraction with leading pair among those three | 96.09% | 95.18% | 95.44% |
| Three-pair conditional mass, long history | 98.48% | 98.50% | 98.35% |
| Three-pair conditional mass, short history | 84.27% | 79.20% | 80.89% |
| Mean total pair-family posterior mass | 94.62% | 93.28% | 93.18% |

The three test pair masses are 31.85%, 31.88%, and 30.25%, compared with training's 33.99%, 31.05%, and 29.89%. There is no visible pair-prevalence break. Test short-history three-pair mass has an episode-bootstrap interval of 77.54–84.17%, and its total pair-family mass is only 77.95%, so uncertain short histories remain the main place a learned prior can help or mislead. These descriptive findings support the plausibility of transferring the learned prior, but do not certify rho=.8 or justify retuning against test inputs.

## Reproducible artifacts

- `main-participants-60-40/adversarial_review/baselines_distribution_audit.py` and `.json`: public-input exposure, missingness, context, recent-outcome summaries, and episode bootstrap.
- `main-participants-60-40/adversarial_review/baselines_pair_shift.py` and `.json`: aggregate frozen-model pair-posterior summaries and type/signal reweighting; no forecast CSVs.
- Existing evidence consulted: `runs/model-comparison/comparison.json`, `runs/03-peer-lags/main-rate3072/report.json`, `runs/02-regime-filter/replay/report.json`, `runs/05-transition-retrieval/replay/report.json`, and the frozen Bayesian tuning reports.
