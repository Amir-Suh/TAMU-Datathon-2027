# S2 · Sum over missing historical wins

## Released main validation

| Forecast | Points / 60 | Overall R | Short-history R | ΔR vs hybrid, paired 95% interval |
|---|---:|---:|---:|---|
| Frozen hybrid | 56.341022 | .064943 | .092936 | — |
| Drop, constant | 35.760100 | .677848 | .088262 | +.612904 [+.558920, +.673933] |
| Expected, constant | 34.080857 | .760519 | .418948 | +.695576 [+.638928, +.757931] |
| Exact, constant | 35.773903 | .677200 | .085673 | +.612257 [+.558141, +.672968] |
| Drop, one flip | 55.864158 | .074034 | .107089 | +.009090 [+.005339, +.013594] |
| Expected, one flip | 51.821538 | .157820 | .442233 | +.092876 [+.076146, +.110460] |
| Exact, one flip | 55.927330 | .072821 | .102237 | +.007877 [+.004457, +.012030] |

Exact one-flip gains .063172 points over its matched drop control. The paired ΔR is −.001213 [−.003376, +.000307], with all of the material change coming from short histories. Constant-polarity arms give poor forecasts on main drift and shared-change episodes. The fixed-emission exact one-flip arm remains .413692 points behind the frozen hybrid. These comparisons support testing marginalization within the stronger hybrid win model, where local rate adaptation and a denser flip grid would need to be retained.

### Nested masks on released validation

Official points with the same query cutoffs and frozen full-prefix payout mix for each budget. All 96 episodes appear in every row.

| Older ticks | Drop, one flip | Expected, one flip | Exact, one flip | Exact ΔR vs drop |
|---:|---:|---:|---:|---:|
| 0 | 55.606804 | 55.430946 | 55.625212 | −.000357 |
| 12 | 55.590509 | 52.738528 | 55.669651 | −.001534 |
| 24 | 55.667614 | 42.010735 | 55.833173 | −.003196 |
| 48 | 55.652078 | 34.418311 | 55.943864 | −.005623 |
| All released | 55.864158 | 51.821538 | 55.927330 | −.001213 |

## Models and data

The comparison freezes the train-only M3 high, low and marginal win rates by machine type and cutoff signal, its 28-pair prior and its null mass. M3's original fit used main training histories and labels only. The first scope fixes polarity throughout the prefix. The second allows one polarity reversal at 8, 16, 40, 80 or 120 ticks before cutoff, with uniform mass over eligible ages and the frozen M3 total flip mass. Both scopes integrate across all pairs, both current polarities, and the marginal-rate null. The bounded age grid and fixed emissions make these models distinct from the frozen hybrid's M3-derived win head, which estimates local rates and uses a denser flip grid.

- **Drop** uses the marginal win rate for observed outcomes whose lag cells are incomplete and skips missing outcomes.
- **Expected** imputes unavailable lag bits independently from the fixed marginal and accumulates fractional high/low counts for each observed outcome. Its weighted log likelihood is an approximation. In uncertain contexts, averaging log emissions penalizes the model more than logging the average emission, and repeated uncertain observations can shift mass to the null.
- **Exact** propagates a distribution over the last `b` win bits. At missing ticks both possible successors retain their probability and time advances. At observed ticks it keeps the compatible branch, normalizes and records the predictive log likelihood. Early ticks with unavailable prehistory use the marginal win rate in all arms. The C kernel uses two shift successors per state and switches to a deterministic state after `b` consecutive observed ticks.

Main forecasts preserve the frozen hybrid's query-level conditional positive-payout probabilities. The shared payout head includes Fredy components trained on pre-cutoff ticks in all main splits. The win arms use released query and history files for prediction and rely on a train-only fitted checkpoint. Main is reused public development. The real-data masking exercise freezes the original full-prefix payout mix to measure sensitivity of the win arm only. It retains the same cutoffs, targets and official reference and uses the last 40 ticks plus nested old observations. The `all` row keeps every released pre-cutoff observation, including naturally incomplete short histories.

Synthetic stable and 16-tick flip scenarios each use 24 independent episodes with exact generated odds. Masks hold the last 40 ticks and add 0, 12, 24, 48 or all older observations. All synthetic forecasts share one type-level payout mix estimated from public main training histories. Synthetic scores use the frozen marginal win rate and that payout mix as a separate reference and are comparable only within this simulation. Pair recovery treats the null as an incorrect pair when its posterior is at least that of the leading pair.

## Generated histories

Five-class SSE per query, lower is better. The stable columns use the constant-polarity scope. The recent-flip columns use the one-flip scope.

| Older ticks | Stable drop | Stable expected | Stable exact | Recent flip drop | Recent flip expected | Recent flip exact |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .001384 | .001695 | .001162 | .005301 | .006386 | .005145 |
| 12 | .001937 | .006691 | .001382 | .005351 | .027447 | .003850 |
| 24 | .001511 | .044693 | .000546 | .005396 | .092418 | .003048 |
| 48 | .001483 | .140651 | .000489 | .003378 | .158651 | .001236 |
| All | .000155 | .000155 | .000155 | .000158 | .000158 | .000158 |

With 24 older observations, stable pair recovery rose from 95.3% to 96.9%. The paired SSE change was −.000964 [−.002599, +.000010]. For recent flips, recovery rose from 97.4% to 97.9% and the paired SSE change was −.002348 [−.005189, −.000228]. The intervals use 10,000 paired whole-episode bootstrap draws. The simulated flip age is represented exactly in the one-flip grid. With a constant polarity on complete recent-flip histories, all three arms had SSE .689087 even though pair recovery reached 93.8%. Correctly identifying the pair cannot compensate for using the wrong current polarity.

## Verification and replay

All six validation CSVs passed the format checker. The 12-episode smoke took 0.82 seconds and the full run took 58 seconds. The native filter matched exhaustive path enumeration for 60 combinations of known and missing outcomes with largest lag up to four. The checks include repeated gaps, completely unobserved prefixes, and prefixes shorter than the largest lag. Complete-history likelihoods and cutoff forecasts matched across methods within numerical tolerance for every simulated and released complete prefix and every hypothesis. Other checks cover missing-padding and row-order invariance, injection of arbitrary post-cutoff outcomes, nested masks, and the unchanged hybrid conditional payout mix.

Run from the participant kit root with a C99 compiler and the packages in [the repaired model requirements](../../../m5_m6_repair/requirements-repaired-models.txt). On macOS the script builds with the active Xcode SDK. The native library is compiled in a temporary directory.

```sh
d=signal_experiments/research_next/S2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 python3 -B "$d/run.py"
for forecast in "$d"/validation_*.csv; do
  case "$forecast" in *_lag_posteriors.csv) continue ;; esac
  python3 -B check_submission.py "$forecast" --split validation
done
```

[Run manifest and hashes](plan.json) · [Main scores vs hybrid](scores.json) · [Matched main comparisons](matched_scores.json) · [Main nested mask scores](masked_public_scores.json) · [Synthetic scores](synthetic_scores.json) · [Exact one-flip forecast](validation_predictions.csv).
