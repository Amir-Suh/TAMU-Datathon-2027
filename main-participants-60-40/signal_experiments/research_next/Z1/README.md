# Z1 · Factored GRU payout state as a win offset

Retain the frozen hybrid on grouped main-validation development. The intercept plus state offset scored **56.025064 / 60**, down 0.315959 points from the raw hybrid. The state term increased R versus the intercept in all four conditions.

| Forecast | Points / 60 | Official R | Paired ΔR vs. raw [95% interval] | p0 MSE |
|---|---:|---:|---:|---:|
| Raw hybrid | 56.341022 | 0.0649434 | | 0.00446810 |
| Intercept | 56.175599 | 0.0680794 | +0.003136 [+0.000091, +0.005997] | 0.00492961 |
| Intercept + GRU state | 56.025064 | 0.0709493 | +0.006006 [+0.002199, +0.009805] | 0.00533895 |

State versus intercept ΔR was **+0.002870 [+0.001272, +0.004707]**. Intervals are based on 10,000 paired episode bootstrap draws stratified by condition and hold the fitted forecasts fixed. All three arms had win-mass-weighted conditional payout SSE of 0.00644843. [Official scores and fits](scores.json)

| Forecast | Stable R | Short-history R | Drift R | Shared-change R |
|---|---:|---:|---:|---:|
| Raw hybrid | 0.051074 | 0.092936 | 0.055314 | 0.060449 |
| Intercept | 0.054890 | 0.091813 | 0.061166 | 0.064449 |
| Intercept + GRU state | 0.057431 | 0.095070 | 0.065316 | 0.065980 |

Against the intercept, the state term changed official R by +0.000814 [+0.000182, +0.001532] from short-history rows and +0.000044 [−0.000430, +0.000472] from the 248 inferred (4,7) rows. These contributions use the full evaluation denominators. On short histories, p0 MSE was 0.006791 raw, 0.006649 with intercept, and 0.007071 with state. On inferred (4,7) rows it was 0.006453, 0.006887, and 0.006891, respectively. [Slice and fixed raw-win-bin diagnostics](diagnostics.csv)

## Win bias by payout mode

Bias is mean forecast win probability minus released win probability, in percentage points. The released payout modes are diagnostic labels assigned after forecasting from the two conditional 1× levels within each type.

| Machine type and mode | Queries | Raw bias | Intercept bias | Intercept + state bias |
|---|---:|---:|---:|---:|
| All, low | 353 | −3.20 | −4.52 | −4.73 |
| All, high | 415 | +2.72 | +1.50 | +1.68 |
| Balanced, low | 103 | −3.35 | −4.65 | −4.91 |
| Balanced, high | 131 | +2.52 | +1.39 | +1.38 |
| Jackpot, low | 118 | −4.35 | −5.84 | −6.14 |
| Jackpot, high | 147 | +3.81 | +2.39 | +2.76 |
| Steady, low | 132 | −2.04 | −3.24 | −3.32 |
| Steady, high | 137 | +1.76 | +0.66 | +0.80 |

The feature's mode AUC was 0.888 for balanced, 0.860 for jackpot, and 0.886 for steady. Its correlation with the raw win residual (released win odds minus forecast) was −0.288 after demeaning both variables within machine type and fixed raw win bins. The feature carries a descriptive win residual association on these episodes. The four outcome-fitted slopes were −0.043, +0.059, +0.130, and −0.041 log-odds per standard deviation. Mean realized win rate was 42.58% versus 43.84% from the released odds. The fitted intercepts were all negative, and the state offset raised observed win log loss from 0.31896 to 0.32132 relative to intercept. [State features and fold assignments](state_features.csv) · [Scores and per-type diagnostics](scores.json)

The earlier signal calibration is a separate negative comparison on the standalone factored GRU. Signal versus intercept had ΔR −0.001033 [−0.004216, +0.002152], and both adjusted forecasts scored below their raw base. [Previous cutoff signal result](../../03_signal_calibration/results/fredy_factored_gru_mean/official_signal_vs_intercept.json)

## Post-hoc soft-odds sensitivity

Using the same frozen inputs, folds and penalties, a secondary diagnostic replaced the realized outcome with released win odds from the other three folds. The soft intercept scored 56.324903 and the soft state offset scored 56.521828. State ΔR was −0.003407 [−0.005275, −0.001635] versus raw and −0.003711 [−0.005577, −0.001934] versus the soft intercept. The four state slopes were −0.174, −0.171, −0.154 and −0.155. Held soft-odds perturbation left each fit and its predictions bitwise identical. This test was chosen after seeing the primary result, and the released main validation odds had already informed the hypothesis. It measures a development sensitivity to supervision and leaves the actual-cutoff-only result as the primary comparison. [Secondary intercept](secondary_soft_intercept.csv) · [Secondary state forecast](secondary_soft_intercept_state.csv) · [Paired scores](scores.json)

## Fit and replay

Each of the four published folds holds out 24 episodes and fits on the 576 realized cutoff outcomes from the other 72 validation episodes. The raw parent is the [frozen split-match M3 and tree/factored GRU hybrid](../../combined/m3_match_split_win_pair_l31_factor_gru_size.csv). The feature is the [three-seed factored GRU mean's](../../01_fredy_replay/forecasts/gru128_factor_mean_3.csv) conditional 1× probability, standardized within type using the fitting episodes. The offset is an equal-episode-weighted Bernoulli fit with prior SD 1.0 for both intercepts and 0.5 per standard deviation for the state slope. The primary fit uses actual outcomes only and has no interaction arm.

This is reused main-validation development with transductive exposure to pre-cutoff histories in the frozen parents. The GRU used train cutoff outcomes. The held validation cutoff outcomes and released odds were excluded from each prediction's fitting steps. The condition and inferred lag pair are used for evaluation and the preassigned grouping only. In every fold, flipping all held realized win labels left the fitted coefficients, scalers and predictions bitwise identical. Conditional payout mix changed by at most 2.22e−16 after CSV export and reread. [Input hashes, settings and versions](plan.json) · [Parent training scope](../../01_fredy_replay/README.md)

Run from the participant-kit root. The runner caps numerical libraries at two threads. The complete run took 2.27 seconds on this host. A separate temporary-directory replay matched all outputs byte for byte, and the kit checker accepted the forecasts.

```sh
d=signal_experiments/research_next/Z1
python3 -B "$d/run.py"
for f in raw.csv intercept.csv validation_predictions.csv secondary_soft_intercept.csv secondary_soft_intercept_state.csv; do
  python3 -B check_submission.py "$d/$f" --split validation
done
```
