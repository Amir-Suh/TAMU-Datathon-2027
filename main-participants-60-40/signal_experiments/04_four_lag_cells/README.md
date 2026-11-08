# Split M3's matching lag contexts

M3's predictive win rate pools two lag losses (00) with two lag wins (11). This experiment splits that match cell while keeping M3's fitted pair, direction, flip-time and null posterior weights fixed. The two differing contexts continue to share their M3 prediction.

For each candidate lag pair and flip time, count observed 00 and 11 contexts with observed current outcomes in that hypothesis's post-flip history. The no-flip hypothesis uses the entire pre-cutoff history. Both lag payouts and the outcome must be observed. Before a hypothesized flip, M3's grouped evidence still contributes to its parent prediction.

The contrast is the empirical win rate for 11 minus that for 00. Estimate a global contrast separately when a match is the current high-win context and when it is the low-win context. Average over the main training histories, weighting by each hypothesis's frozen M3 posterior and effective paired count, `n00 * n11 / (n00 + n11)`. Add 200 effective counts at zero to regularize the global estimates. The resulting contrasts are −0.02368 and −0.00209, respectively.

For each current match hypothesis, blend the machine's local contrast with 20 effective counts at the learned global contrast. Shrink further by `n / (n + 8)`, where `n = n00 + n11`. Center the two child predictions around M3's parent under the smoothed subtype frequencies and constrain the contrast so both predictions are valid. Mix the refined win forecasts with M3's original hypothesis weights. Reuse the fitted M3 conditional payout-size distribution.

All settings were fixed before validation scoring. The script reads only the query and history files for training the contrast and producing the forecasts. It reads released validation targets after saving the candidate files.

| Forecast | Points / 60 | R | Stable R | Short-history R | Drift R | Shared-change R |
|---|---:|---:|---:|---:|---:|---:|
| Frozen M3 | 55.268370 | 0.085612 | 0.077244 | 0.106894 | 0.079817 | 0.078493 |
| Split 00/11 | 55.350776 | 0.083996 | 0.072578 | 0.105652 | 0.079494 | 0.078259 |

The gain is 0.0824 points. The change in R is −0.001616 with a condition-stratified paired episode bootstrap 95% interval of [−0.003049, −0.000364]. P(0×) MSE fell from 0.004711 to 0.004468. The conditional payout-size probabilities agree with frozen M3 after CSV parsing to within 2.22 × 10⁻¹⁶.

| Current context of M3's highest-posterior lag pair | Queries | Mean absolute win adjustment | Five-class SSE change per query |
|---|---:|---:|---:|
| Match | 466 | 0.008705 | −0.000513 |
| 00 | 288 | 0.006284 | −0.000310 |
| 11 | 178 | 0.012622 | −0.000841 |
| Differ | 302 | 0.00000116 | −0.000000580 |

Rows where the top lag pair differs can still have a small posterior weight on other lag pairs that match. The prediction is exactly unchanged wherever no matching hypothesis has weight. There were zero rows with missing current lags for the top pair on main validation. A synthetic all-missing case verified the fallback, and 80 comparisons against direct pre-cutoff enumeration verified the suffix counts. The script also checked rejection of an at-cutoff payout, exact fallback when child history was empty, and numerical centering on M3's match parent.

Main validation is reused development data. The paired interval describes episode variation conditional on this fit. `test_predictions.csv` passed the local checker for all 1,536 queries. Hidden main test and adaptation scores are unavailable.

Run from the participant-kit root with the frozen checkpoint and its original dependencies in `m5_m6_repair/requirements-repaired-models.txt`.

```sh
d=signal_experiments/04_four_lag_cells
python3 -B "$d/run.py" --test
python3 -B signal_experiments/score.py --out "$d/paired_scores.json" \
  "four_lag_cells=$d/validation_predictions.csv"
python3 -B check_submission.py "$d/validation_predictions.csv" --split validation
python3 -B check_submission.py "$d/test_predictions.csv"
```
