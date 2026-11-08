# S1 · Blend forecasts using observed support

Main validation contains 768 queries from 96 episodes. The 40-triple gate scored **56.419747/60**, only **0.001142 points** above the global blend. Its paired ΔR versus global was −0.000022 overall and −0.000021 on short histories, with both intervals spanning zero. The global blend is the simpler development candidate for a fresh-episode comparison.

| Forecast | Points / 60 | R | Stable R | Short-history R | Drift R | Shared-change R | ΔR vs hybrid, paired 95% interval |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen hybrid | 56.341022 | .064943 | .051074 | .092936 | .055314 | .060449 | — |
| Frozen full blend | 56.064085 | .070204 | .061924 | .091776 | .063370 | .063745 | +.005260 [+.000813, +.009554] |
| Equal | 56.397504 | .063877 | .051701 | .087477 | .056652 | .059677 | −.001067 [−.003443, +.001083] |
| Global | 56.418604 | .063479 | .050434 | .088530 | .055550 | .059402 | −.001464 [−.003010, −.000072] |
| Support | 56.419747 | .063457 | .050331 | .088509 | .055324 | .059665 | −.001486 [−.003220, +.000110] |
| Entropy | 56.405187 | .063732 | .050339 | .089664 | .055471 | .059454 | −.001211 [−.002696, +.000115] |

| Slice | Queries | Episodes | Support ΔR vs global | Paired 95% interval |
|---|---:|---:|---:|---|
| Overall | 768 | 96 | −.000022 | [−.000566, +.000532] |
| Short history | 192 | 24 | −.000021 | [−.001918, +.001932] |
| Complete history | 576 | 72 | −.000022 | [−.000366, +.000370] |
| M3 top pair (4,7) | 248 | 86 | +.000420 | [−.000515, +.001389] |

On short histories, the global blend beat the hybrid by .004406 in R [−.008848, −.000523 for ΔR]. The equal blend scored lowest on that slice, at .087477. The global blend lowered overall p0 MSE from .004468 to .004143 and increased the win-mass-weighted conditional payout error from .006448 to .006616. The 40-triple split placed 191 of 192 short-history queries below the threshold and every complete-history query above it. The standalone ≥40 slice gives the single short-history query's condition one quarter of its weight, so the complete-history row is the cleaner readout for long prefixes.

## Fit and scope

Both parents forecast all five classes. The hybrid uses split-match M3 win mass and the 31-leaf tree plus three factored GRUs for conditional payout size. The full blend averages the 31-leaf tree, factored GRU mean and repaired M6. Their CSVs and the frozen M3 checkpoint are verified against their original manifests. The shared Fredy components trained on observed pre-cutoff ticks from all main splits. This is reused main development with transductive parents. The four condition-balanced outer folds hold out episodes for fitting stack weights. The upstream forecasts and checkpoint retain their original fit scopes.

The mixture is hybrid plus the full-blend weight times the difference between the two parents. In each fold, fit the weight on the other 72 episodes by minimizing the scorer's condition-balanced, reference-scaled sum of squared probability errors. Reference scales come from those 72 episodes. Each gated weight adds a quadratic penalty toward the global weight with curvature equal to eight average fitting episodes. The gate falls back to the fold's global weight for missing values or values outside the corresponding bin's fitting range. All weights are constrained to [0, 1].

Support counts observed outcomes with both lagged outcomes present before cutoff, using the highest-posterior pair from frozen M3. The independent entropy arm uses normalized Shannon entropy over the M3 lag-pair posterior and null, split at .01. An initial median-entropy trial fell around 10⁻¹⁵, where the split responds to posterior rounding. The final run normalizes the posterior mass and uses the fixed .01 boundary. Condition enters fitting and scoring, while application takes the fitted coefficients, the two parent vectors, and the history-derived feature.

| Held fold | Global full-blend weight | Support <40 | Support ≥40 | Support fallback queries |
|---|---:|---:|---:|---:|
| 0 | .354 | .503 | .271 | 0 |
| 1 | .282 | .362 | .232 | 0 |
| 2 | .321 | .491 | .222 | 8 |
| 3 | .330 | .582 | .207 | 8 |

Intervals use 10,000 paired episode bootstrap draws stratified by condition with reference denominators recomputed for each resample. They describe episode variability for these fixed cross-fitted predictions. They exclude fitting and model-selection uncertainty.

## Replay

From the participant kit root, use the packages pinned in [the repaired model requirements](../../../m5_m6_repair/requirements-repaired-models.txt).

```sh
d=signal_experiments/research_next/S1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 python3 -B "$d/run.py"
for forecast in "$d"/validation_*.csv; do
  python3 -B check_submission.py "$forecast" --split validation
done
```

All four CSVs passed the format checker. The runner checks exact keyed feature invariance to row order, missing padding and post-cutoff outcome injection, all 768 support counts against direct enumeration, and blend invariance to held-out truth, condition and reference poisoning. It also checks row permutation, fallback, valid probabilities and score agreement. On this machine the full replay took about 2.2 seconds with two numerical threads.

[Run manifest and hashes](plan.json) · [Fold coefficients](coefficients.json) · [Per-query features and weights](query_weights.csv) · [Official scores](scores.json) · [Paired subgroup scores](subgroup_scores.json) · [Support-gated predictions](validation_predictions.csv) · [Global predictions](validation_global.csv).
