# Adversarial model review — November 7, 2026

Scope note: this review covered the original checkout only. The subsequent [all-branch review](branches/REVIEW.md) found Amir's M3 to be a stronger candidate and supersedes the overall recommendation below. The original measurements remain valid for their stated model versions and splits.

Recommend the **frozen switching Bayesian lag-pair model (`switch_96`, uniform pair prior) with the self-only logistic positive payout head** for the main round. It has the strongest evidence for generalization among the evaluated candidates: 53.382758 tuning points and 52.665123 on a separate, previously frozen audit, compared with 50.273957 and 49.212506 for the previous boosted/logistic leader. Test answers are unavailable; these are observed validation scores, not a promised test score.

Three GPT-6 Astra subagents at xhigh reasoning independently reviewed the lead/selection pipeline, temporal families, and baselines/distribution shift. The root reviewer independently rescored saved forecasts, replayed the selected Bayesian experiment and checked the targeted test suite. This review changes documentation and creates review artifacts; it does not change the forecasting algorithms or submit an entry.

## What the evidence supports

| Candidate/family | Tuning points / 60 | Existing audit / 60 | Decision |
|---|---:|---:|---|
| Switching Bayesian, uniform pair prior + logistic payout head | 53.382758 | 52.665123 | Recommended frozen model |
| Switching Bayesian, learned pair prior `rho=.8` + logistic head | 53.799644 | Not evaluated | Plausible small improvement; keep as challenger |
| Bayesian/boosted blend, scalar fitted outside each tuning fold | 53.392796 | Not evaluated | Negligible advantage over Bayesian alone; added selection uncertainty |
| 25% Bayesian / 75% previous lead | 51.712276 | Not evaluated | Dilutes the stronger model |
| Boosted adaptive 3072 win odds + logistic payout head | 50.273957 | 49.212506 | Credible fallback |
| Boosted correction, adaptive 3072 | 49.743869 | Not evaluated | Strong gain, weaker payout head |
| Individual peer lags, adaptive 3072 | 44.117 | Not evaluated | No clear incremental gain |
| Adaptive rate 1536 / rate 3072 | 44.104 / 44.098 | Not evaluated | Practically tied; retain train-selected3072 |
| Regime filter, adaptive 3072 | 43.950 | Not evaluated | Does not improve matched control |
| Signed motifs, original / logistic-head version | 42.026 / 42.402 | Not evaluated | Drift/shared-change weakness |
| Learned retrieval | 41.233 | Not evaluated | Better than fixed retrieval, far below leader |
| Boosted correction, adaptive 48 | 37.775 | Not evaluated | Superseded |
| Original adaptive 48 | 33.225 | Not evaluated | Original control |
| Self-only causal CNN | 31.597 | Not evaluated | This implementation/training regime is not competitive |
| Self-only two-stage logistic | 31.444 | Not evaluated | Weak win model; valuable conditional payout model |
| Fixed retrieval | 31.438 | Not evaluated | No competitive advantage |
| Direct five-class logistic | 31.354 | Not evaluated | Cannot express machine-specific changing effects adequately here |
| Two-stage logistic with peers / unblended | 31.306 / 31.290 | Not evaluated | No demonstrated peer benefit |
| Fixed lag / frequency / no-context controls | 31.237 / 31.133 / 31.133 | Not evaluated | Baselines |

All tuning results use the same 48 episodes, 12 per condition, and 384 queries. The audit uses the other 48 episodes and was already scored before this review. Existing model-selection variants and blend arrays are included in the [complete independent score inventory](verified_scores.json): 152 CSVs, 72 distinct normalized prediction arrays. These arrays include replicas, related configurations and in-sample fitted blends; the count is not 72 independent statistical tests.

The unselected Bayesian variants were compared using realized binary cutoff Brier on grouped training folds: switch_96 .082812; switch_32 .087086; discounted signed .142102; widened discounted signed .143056; static signed .187806; parity table .194872; four-cell table .199040. This supports the switching mechanism specifically. Binary Brier must not be compared numerically with another model's five-class Brier.

## Why the Bayesian recommendation survives challenge

The frozen model infers each machine's lag-pair identity and signed strength, allows the strength to reset with probability 1/96 per tick, and uses a separate trained conditional payout distribution. This matches the released problem's heterogeneous temporal relationships and changing effects. A pooled logistic or small CNN has to learn these machine-dependent dynamics indirectly. The recommendation is driven by held-out results, not architecture alone.

| Audit condition | Bayesian R | Boosted/logistic R | Bayesian minus boosted |
|---|---:|---:|---:|
| Stable | .093617 | .076837 | +.016780 |
| Short history | .225938 | .277504 | -.051566 |
| Drift | .114058 | .271122 | -.157064 |
| Shared change | .123483 | .251346 | -.127863 |
| Equal-condition mean | .139274 | .219202 | -.079928 |

The audit advantage is 3.452618 points. The saved paired 95% episode-bootstrap interval for delta R is [-.124380, -.041979]. Independent 10,000-resample verification gives [-.123317, -.041710], consistent with the saved 2,000-resample interval. The Bayesian model improves 30 of 48 audit episodes; its advantage remains when any one condition is excluded. Its worst audit episode squared error is .074895 versus .163966 for boosted. Stable is a real exception, so this is not dominance on every condition or episode.

Do not use the stable audit result to invent a condition-specific blend. Test condition labels are unavailable, and adapting the choice to audit observations would consume the independence that makes this evidence persuasive.

## The learned prior: promising, but a smaller claim

The latest model adds a training-estimated pair-frequency prior with `rho=.8`. It keeps nonzero support for all 28 pairs and leaves hazard, amplitude grid and logistic payout head unchanged. Tuning improves by .416885 points, with delta R -.008709 and saved descriptive interval [-.014173, -.004372]. Short-history R falls from .181968 to .150495; stable and shared change worsen very slightly.

This is positive evidence, not proof of overfitting. However:

- It misses the prespecified tuning advancement criterion `delta R <= -.01`; its own report says `advances.win_only=false`.
- Nested outer training evaluation improves binary cutoff Brier only .08281191 to .08245183 and five-class Brier .37213232 to .37168536. Two folds improve, one worsens, and one is unchanged. One outer training selection retains the uniform prior. A much larger improvement on historical endpoints should not be substituted for the released-cutoff result.
- The improvement is concentrated in sparse histories, where a concentrated prior helps common patterns but can delay learning rare patterns. The temporal review includes a controlled synthetic example of this tradeoff; it is a stress test, not a test-set performance estimate.
- No independent post-development audit exists for this variant. The original audit is already consumed. The strongest tuning score alone is insufficient to replace the frozen default under the existing selection rule.

The learned-prior model may achieve a slightly better main-test score if pair prevalence persists, and observable diagnostics support that possibility. The audited uniform model remains the more defensible submission choice. The promotion threshold is a decision rule, not a test proving the challenger has no benefit.

The tuning implementation also differs from the written plan: it adds a .001 Brier minimum gain and majority-fold requirement. These suppress the reset 144/expanded-grid follow-up that the original plan's looser rule would have permitted. The frozen manifest records the implemented rule. Do not retrospectively loosen it after seeing tuning scores; reconcile the plan before any genuinely new experiment. See [temporal review](temporal_review.md).

## Generalization and leakage checks

No direct causal leakage invalidating the recommended model was found. Forecasts are emitted before each outcome update; lag features access earlier ticks; missing ticks remain distinct from zero payouts. Boosted correction excludes held-out episodes from population priors and uses inner episode exclusion for correction-training features. The logistic head's grouped training split keeps machines from an episode together. Synthetic future-outcome and held-out-label invariance checks passed.

The existing audit script has separate prediction-freezing and scoring phases. Forecast/model hashes are recorded before targets are loaded, and the saved audit scores match independent rescoring. This is evidence from the code and saved records; it cannot reconstruct every historical interaction with the public validation data.

Repeated tuning still matters. Scalar cross-fitting only excludes each fold while fitting its scalar, not while selecting upstream architectures and model families. Bootstrap intervals resample fixed episode losses; they do not include every tuning decision, retraining variation, or a hidden-test score forecast. All model families other than the two frozen audit candidates rely on development evidence here.

Observable distribution checks used only released queries and histories:

- Train, validation and test have cutoff range 192–256 and exactly 25% short histories. Every final query has all previous 40 ticks and all 8 lag inputs observed.
- Historical training endpoints from the last 32 observations are on average 16.5 ticks earlier. Their previous 40-tick windows average 36.40 observed outcomes and can contain only 9, versus 40 at final cutoffs. This mismatch can hurt learned corrections, retrieval and sequence models; cutoff validation is essential.
- Test's last 32-tick win rate is .39378 versus train's .42924. Type/signal composition explains about .00974 of the .03546 difference; a residual remains. These are observed past outcomes, not future labels. Avoid assuming this alone proves a particular probability recalibration would help.
- Under the frozen uniform-prior Bayesian model, the three common pairs `(1,3),(2,5),(4,7)` carry 94.93% of conditional pair mass in train, 93.68% in validation and 93.99% in test. Test short histories carry 80.89%, versus 98.35% for long histories. No large prevalence break is visible, but inferred posterior identity is not ground truth.

The principal Bayesian limitation is fixed pair identity within each hypothesis. The hazard resets amplitude inside a pair rather than switching pair identity. Posterior weights can still move between pairs, but accumulated evidence can slow recovery if the true lag pair changes. Amplitude-grid limits, the plug-in win offset and imperfect payout prediction also leave error. None of the main-round rankings establish performance after the adaptation round changes recurring relationships.

## Reproducibility and remaining delivery gap

The current review reran all 19 targeted tests across Bayesian lag pairs, pair tuning, boosted correction and peer lags; all passed. A fresh Bayesian run selected switch_96 and reproduced standalone, signed-control, fixed-blend and cross-fitted-blend predictions with maximum absolute difference 0.0. [Fresh replay report](bayesian_replay/report.json).

The saved audit CSVs, frozen checkpoint and logistic model hashes match their manifest. A forecast-only replay of the existing audit inputs also checks the current source against the saved model bundle; see [checkpoint replay](frozen_checkpoint_replay.json). This reruns existing forecasts only and does not select a new audit candidate.

There is a delivery gap: the model runners write tuning forecasts, and the audit runner hardcodes validation with `prepare`/`score` phases. A submission still needs a forecast-only entry point loading the frozen Bayesian and logistic models, accepting test query/history paths, checking all 1,536 IDs, and recreating its CSV offline within organizer limits. The frozen trained ingredients exist. The generic starter command generates the pooled starter, not the recommended model. No test CSV or code ZIP was submitted by this review.

The boosted runner also excludes zero correction when naming its selected correction. That would be unsafe to consume blindly on another release if every correction lost. It does not explain the saved main result: the selected 60-tree correction improves all four training folds. This is a follow-up robustness issue, not grounds to invalidate the comparison. Similarly, the regime-filter runner exports its best nonbase filter even though grouped training selects the base at both adaptive rates; its saved filter CSV must be treated as a rejected experiment, not the training-selected deployment model. See the baseline review for source locations.

## Evidence and replay

- [Independent scores, paired diagnostics and nested-fold aggregates](verified_scores.json)
- [Independent scoring script](verify_saved_forecasts.py)
- [Lead and selection review with source locations](leader_selection_review.md)
- [Temporal-family review and stress tests](temporal_review.md)
- [Baseline/distribution review](baselines_review.md)
- [Public-input distribution measurements](baselines_distribution_audit.json)
- [Frozen Bayesian pair-prevalence diagnostics](baselines_pair_shift.json)
- [Original Bayesian tuning report](../../runs/bayesian-lag-pairs/replay/report.json)
- [Original frozen audit report](../../runs/bayesian-lag-pairs/audit-switch-96/report.json)
- [Learned-prior tuning report](../../runs/bayesian-lag-pair-tuning/replay/report.json)

From the repository root, using an environment with the model dependencies:

```sh
python -B -m unittest test_bayesian_lag_pairs test_bayesian_pair_tuning test_boosted_correction test_peer_lags
python -B run_bayesian_lag_pairs.py --out /tmp/house-edge-review-fresh --compare-to runs/bayesian-lag-pairs/replay
python -B main-participants-60-40/adversarial_review/verify_saved_forecasts.py
```

Choose a fresh replay output directory. The independent scorer deliberately recomputes the two already-scored audit forecasts; it does not score additional candidates there.
