# Cutoff-only signal calibration

Each base five-class prediction provides a win probability and a conditional distribution over 1×, 2×, 5×, and 20×. A fixed-offset logistic regression fits an intercept, with an optional linear term for the query signal (−1, 0, 1). The zero-centered Gaussian penalties have standard deviations 1.0 for the intercept and 0.5 for the signal slope. Training uses the observed payout at the actual cutoff, coded as win or zero, with equal episode weight. All four conditional payout probabilities are carried through unchanged. The observed-label fit is separate from scoring against the simulator odds.

For the grouped validation development comparison, the same fixed base predicts every main validation query. Each fold fits its adjustment on 576 observed cutoffs from 72 other validation episodes. The fold manifest balances the four conditions, and the adjustments use the base prediction and query signal. The GBDT development fit consumes histories and never uses any actual cutoff labels. An additional GBDT check fits the adjustment on its 768 training-cutoff forecasts and holds all validation labels out. Those training episodes contributed histories to the GBDT. The GRU used training cutoff labels, so its calibration uses grouped validation only. The M3 comparison fits the correction on the archived 768 genuine training-cutoff out-of-fold predictions and applies it once to all validation episodes.

| Frozen base and scope | Raw points | Intercept points | Intercept + signal points | Signal vs. intercept ΔR | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| Fredy pair_l31 GBDT, validation cross-fit | 54.28989 | 54.09749 | 54.03749 | +0.001232 | [−0.001810, +0.004307] |
| Fredy pair_l31 GBDT, train cutoff labels held out from base | 54.28989 | 54.27644 | 54.22677 | +0.001013 | [−0.001907, +0.003911] |
| Fredy no-message GRU-128, seven replay seeds, validation cross-fit | 55.41919 | 55.23294 | 55.24239 | −0.000186 | [−0.003067, +0.002695] |
| Fredy no-message GRU-128 seed 2, validation cross-fit | 53.77811 | 53.63676 | 53.52257 | +0.002387 | [−0.001403, +0.006317] |
| Signal-aware repaired M3, train OOF calibration | 55.26837 | 55.12163 | 55.08835 | +0.000658 | [−0.000264, +0.001627] |

At commit `bbff737`, the GBDT and GRU use histories and machine type without an explicit signal input. Across validation, simulator mean win probability was 7.35 percentage points higher at signal +1 than at −1. The raw GBDT, seven-seed GRU, and M3 forecast gaps were 5.00, 3.68, and 6.89 percentage points respectively. These unadjusted group means show some signal-associated variation in the forecasts and a positive residual gap for the Fredy bases. [GBDT residuals](results/fredy_pair_l31/signal_residuals.csv) · [GRU residuals](results/fredy_gru_mean/signal_residuals.csv)

The fitted slopes were positive in every GBDT and GRU fold. Compared with the intercept, the slope increased the held-episode observed win Brier for both bases, increased official loss for the GBDT, and slightly reduced official loss for the seven-seed GRU. The paired intervals for these slope contrasts all crossed zero. The held-cutoff-label GBDT fit produced a +0.208 log-odds slope per signal step with the same direction in official loss. The M3 training slope was +0.105 with similarly inconclusive validation differences. Each raw model scored higher than its calibrated versions. [GBDT paired contrast](results/fredy_pair_l31/official_signal_vs_intercept.json) · [GRU paired contrast](results/fredy_gru_mean/official_signal_vs_intercept.json)

The primary GRU result uses the equal mean of seven newly fitted 100-epoch no-message models. Seed 2 is a sensitivity check and is one of the seven ensemble members. See the [replay provenance](../01_fredy_replay/README.md) for the training devices and historical checkpoint availability.

Each results directory contains the base and both adjusted CSVs, coefficients, observed label metrics, descriptive residuals by signal, and official comparisons. Paired intervals sample whole episodes within condition, keeping the cross-fitted predictions fixed.

## Reproduce

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2
cal=signal_experiments/03_signal_calibration
out="$cal/results/pair_l31_replay"
python -B "$cal/calibrate.py" --model 'Fredy pair_l31 GBDT at bbff737' \
  --validation-forecast signal_experiments/01_fredy_replay/forecasts/pair_l31.csv \
  --folds signal_experiments/folds.csv --out "$out"
python -B signal_experiments/score.py --out "$out/official_vs_raw.json" \
  --baseline "$out/raw.csv" intercept="$out/intercept.csv" intercept_signal="$out/intercept_signal.csv"
python -B signal_experiments/score.py --out "$out/official_signal_vs_intercept.json" \
  --baseline "$out/intercept.csv" intercept_signal="$out/intercept_signal.csv"
python -B -m unittest discover -s "$cal" -v
```

For the archived M3 mode, pass `--train-cache m5_m6_repair/results/repaired-m5-m6/train_outer_predictions.npz` instead of `--folds` and use the M3 validation forecast from the same result directory.

The three-seed factored GRU mean was scored in an additional grouped check. Raw scored 55.406231, intercept 55.227612, and intercept plus signal 55.280198. Signal versus intercept ΔR was -0.001033 with paired 95% interval [-0.004216, +0.002152]. [Saved comparison](results/fredy_factored_gru_mean/official_vs_raw.json).
