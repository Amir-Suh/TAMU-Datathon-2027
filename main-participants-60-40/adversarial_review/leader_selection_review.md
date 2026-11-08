# Lead-model adversarial review

Scope: adaptive lag model, nested boosted correction, forecast combinations, episode bootstrap, tuning and previously produced frozen audit records. No new audit scoring or reserved target access; no source changes or full training. Paths below are relative to the parent repository.

## Recommendation

Use the frozen Bayesian `switch_96` with the self-only logistic positive payout head as the current main-round candidate. Retain boosted rate-3072 plus the same head as a fallback and benchmark. The advertised 50.273957 tuning leader is superseded by direct independent evidence: the existing audit is 52.665123 versus 49.212506, a 3.452618 point advantage. Paired episode-resampled delta R is -0.079928, with a descriptive 95% interval [-0.124380, -0.041979]. Evidence: `runs/bayesian-lag-pairs/audit-switch-96/report.json:4`, `:52`, `:88`.

This supports a likely winner among these compared candidates, not a guaranteed hidden-test score. Only 48 audit episodes, 12 per condition, are available. Bayesian loses the stable audit stratum (R .093617 versus .076837), while winning drift (.114058 versus .271122), shared change (.123483 versus .251346), and short history (.225938 versus .277504). Do not create a condition-aware hybrid from these audit outcomes.

The newer learned-pair-prior candidate (`rho=.8`) reaches 53.799644 tuning versus the frozen control's 53.382758, but delta R = -.008709 misses its specified -.01 promotion threshold. `run_bayesian_lag_pair_tuning.py:344` implements that gate; `runs/bayesian-lag-pair-tuning/replay/report.json:2` records `advances.win_only=false`. Its largest gain is short history. Keep it a challenger rather than replacing the audited model solely on tuning rank.

## Findings and limitations

1. **Outdated leader claims are materially misleading.** `EXPERIMENT_RESULTS.md:7` calls 50.273957 the best evaluated configuration, and `:3`/`:9` say the audit is still reserved. Both conflict with the existing Bayesian audit. README had the same issue at line 83 before the parent review's update. Separate tuning results from the already consumed audit and identify the frozen configuration.

2. **Tuning bootstrap intervals do not establish selection-adjusted winner probabilities.** `run_combine_forecasts.py:105` creates 15 fixed comparisons, `:119` defines three scalar strategies, and `:205` selects the strongest categories after scoring. `slot_forecasting/tuning_metrics.py:34` reduces fixed forecast differences to episode means and `:46` resamples those losses; it neither retrains models nor reselects strategies. `slot_forecasting/combine_forecasts.py:73` correctly fits each scalar only outside its held-out episodes, but all upstream configurations and the decision to use logistic heads were selected earlier. Consequently the out-of-fold blend is out-of-fold for that scalar, not for the entire research pipeline. This is a limitation, not an observed data-leakage bug; the combination report explicitly labels these intervals descriptive. The existing frozen audit is the stronger evidence.

3. **Frozen deployment ingredients exist, but the standard lead runners remain tuning-only.** `run_boosted_correction.py:137` and `run_combine_forecasts.py:68` load tuning inputs. The public reproduction sequence in `EXPERIMENT_RESULTS.md:38` produces tuning CSVs; it does not predict arbitrary test query IDs. The saved audit checkpoint is real and contains Bayesian, adaptive, trees and logistic models (`runs/bayesian-lag-pairs/audit-switch-96/run_frozen_audit.py:139`). Its `_predict` function at line 96 is reusable, but its CLI at line 204 only offers `prepare` and `score` and the script hardcodes validation inputs. Before submission, expose a forecast-only entry point using frozen models and arbitrary query/history directories, validate exact test IDs, and avoid a fresh tuning sweep. This is an integration gap, not missing learned parameters.

4. **The boost runner always names a correction even if the base wins.** `run_boosted_correction.py:110` excludes `zero_correction` from lead selection, and `:150` writes that correction to `tuning_predictions.csv`. The later `shortlist` flag prevents a favorable report when the base wins, but a caller that consumes the CSV without checking the flag can still deploy the worse correction. This does not affect the saved main-rate3072 result: trees_60 actually wins all four folds. Treat as a low-priority runner robustness issue.

## Evidence against causal-leakage allegations

- `run_boosted_correction.py:68` excludes the outer held-out episodes from the base prior. Lines 71-74 exclude both outer and inner held-out episodes when constructing correction-training features. Lines 99-101 train each correction only on other outer folds. The structure is valid nested base-feature cross-fitting.
- `slot_forecasting/causal.py:45` builds a padded prefix, and `_window_features` indexes at the endpoint, excluding that tick's outcome. Lag features index only prior ticks. `slot_forecasting/peer_lags.py:73` records the adaptive forecast before updating it at lines 80-92.
- `run_model_comparison.py:45` selects tuning episode IDs before reading target values at line 50. No direct audit leakage was found in this reviewed training path.
- The audit script freezes models and prediction hashes in its prepare phase (`run_frozen_audit.py:138`), then reads audit targets only in the separate score phase (`:180`). Its reported tuning replay deltas are 1.11e-16 for Bayesian, logistic, and previous lead.
- The selected boost is supported beyond tuning: cutoff multiclass Brier falls from .434141 to .395256; every outer fold improves. Note the Bayesian report uses binary Brier, so its .082812 must not be numerically compared with boost's multiclass .395256.

## Verification completed

- `python -B -m unittest test_boosted_correction -v`: all three tests passed, covering weighted Newton updates with supplied offsets, future-payout invariance, and outer-label exclusion from base/calibration fitting.
- SHA256 values independently checked against the existing freeze/report: Bayesian audit CSV, previous-lead audit CSV, frozen model checkpoint, freeze manifest, and logistic checkpoint all match.
- Existing boosted reports show deterministic replay maximum deltas of zero and adaptive-trace/control delta 5.55e-16. Existing combination replay reports show zero deltas for all saved CSVs. These are inspected report evidence, not full training reruns during this review.

No direct model math or leakage bug was established that invalidates the 50.273957 tuning number. Its weakness is being an older selected tuning winner, while a later frozen model has already beaten it on the reserved audit.
