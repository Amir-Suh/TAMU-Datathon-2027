# House Edge experiments

The five experiments use frozen M3 as the common main-round control. The official scorer reproduces **55.268370 / 60**, with **R = 0.085611888** across 96 public validation episodes and 768 queries.

The first round's public validation data already informed development. New supervised layers use whole-episode folds or separate training cutoff labels as documented in each run. The four folds in [folds.csv](folds.csv) each hold 24 episodes in both train and validation. Each fold has six episodes per condition. The fold stratification uses condition only to assign episodes to folds and score forecasts.

[score.py](score.py) validates forecast IDs and probabilities with the kit, reports official condition-weighted loss and win and payout diagnostics, and computes paired 95% intervals from 10,000 episode resamples within condition. Each resample uses the same episodes for all forecasts and the frozen reference. The intervals hold forecasts fixed and describe variation across the released episodes. [Control and data hashes](control.json).

To score a candidate from the kit root:

```sh
python3 -B signal_experiments/score.py candidate=/path/to/validation_predictions.csv
```

## Individual experiments

1. [Fredy retrains, blend reconstruction, and payout-head swaps](01_fredy_replay/README.md). The 31-leaf tree and historical 15-leaf tree reproduced their logged validation scores to six decimals. Historical GRU predictions and some seed identities were unavailable. New fits on this host reconstruct the seven-seed blend design.
2. [History-only payout-state head](02_payout_state/README.md). The four-fold history classifier achieved 0.731 AUC and **55.278099**. Its change in R against M3 was −0.000191 (paired 95% interval −0.002194 to +0.001924). The extra weighting check scored 55.262742. Both preserve M3's win mass.
3. [Cutoff signal calibration](03_signal_calibration/README.md). A regularized win-logit offset fits actual cutoff labels and preserves each model's positive payout mix. For the 31-leaf tree, the shared validation folds scored **54.289888** raw, **54.097491** with intercept, and **54.037486** with intercept and signal. The train-cutoff sensitivity shows the same ordering. The tree's historical training uses the episode histories and leaves the actual train cutoff labels for this calibration.
4. [Split M3's two matching lag cases](04_four_lag_cells/README.md). Separating 00 and 11 scored **55.350776**, with lower loss in all four conditions. ΔR was −0.001616 (−0.003049 to −0.000364). The original lag-pair and flip posterior and the original payout-size model remain the basis of the ablation.
5. [Lag-prior and shared-flip transfer stress](05_adaptation_stress/README.md). The adaptation release was unavailable at the time of these runs. Under the main-round mask with 8 recent and 24 older observations, M4 scored **38.456** against M3's **38.263**. In the generated rare-pair shift with 16 recent and 24 older observations, refitting the lag prior reduced squared distribution error from **0.212595 to 0.056175**. Full histories had effectively identical error under both priors. The runner supports the adaptation kit and refits M3 and M4 on its training episodes.

[Combined win and payout heads](combined/README.md) compares the separate win and payout improvements in one forecast. Both the fixed forecast scores and the bootstrap intervals use previously examined main validation episodes. Official test checks validate CSV structure and probability rows.

## Leading development forecasts

| Forecast | Main validation points / 60 | Change from M3 |
|---|---:|---:|
| Split-match M3 win + tree/factored-GRU conditional payout mix | 56.341022 | +1.072652 |
| M3 win + tree/factored-GRU conditional payout mix | 56.259032 | +0.990662 |
| Split-match M3 win + tree/plain-GRU conditional payout mix | 56.254033 | +0.985663 |
| M3 win + tree/GRU conditional payout mix | 56.172553 | +0.904183 |
| 31-leaf tree + factored GRU mean + repaired M6, full forecast average | 56.064085 | +0.795715 |
| 31-leaf tree + plain GRU mean + repaired M6, full forecast average | 55.973765 | +0.705395 |
| Split-match M3 win + 31-leaf tree conditional payout mix | 55.833268 | +0.564898 |
| 31-leaf tree + frozen M3 win, with only tree payout mix | 55.751846 | +0.483476 |
| Frozen M3 | 55.268370 | 0 |

The original three-way recipe with the historical 15-leaf tree and pre-repair M6 scored **55.860278** after fresh GRU training. The branch had reported 55.92 with the original GRUs. The seven-seed mean also supports the calibration result: raw **55.419191**, intercept **55.232942**, and intercept plus signal **55.242387**. The signal term's incremental ΔR was −0.000186 (−0.003067 to +0.002695).
