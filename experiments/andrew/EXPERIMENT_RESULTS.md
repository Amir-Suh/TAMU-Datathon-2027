# House Edge model suite · Main tuning results

All comparisons use the same **48 tuning episodes and 384 queries**. The other validation episodes remain reserved for audit. The official score is `60 / (1 + R)`, where smaller `R` is better. The original adaptive control uses weight rate 48 and scores **33.224516**.

## Lead forecast

The best evaluated configuration scored **50.273957 / 60** with `R = 0.193461`. It takes the win probability from the 60-tree boosted model at rate 3072 and the conditional positive payout mix from the existing self-only logistic model. The change preserved every no-payout probability and added **0.530 points** over the boosted model. Paired `ΔR` was `-0.012718` with a 95% interval of `[-0.019674, -0.006553]`. All four condition ratios improved to 0.1527 stable, 0.2191 short history, 0.1912 drift and 0.2108 shared change. [Lead tuning predictions](runs/06-combine-forecasts/replay/fixed__boosted_3072_logistic_head.csv), [combination report](runs/06-combine-forecasts/replay/report.json).

The strongest episode-cross-fitted combination reached **50.237** by blending boosted and signed-motif win probabilities before applying the same logistic payout mix. The fixed head swap is simpler and scored slightly higher on these tuning episodes. The fixed configuration was selected after comparing tuning results, so these scores describe tuning performance. The independent audit remains reserved.

## Standalone results

| Candidate | Points / 60 | R | Stable | Short history | Drift | Shared change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Boosted correction, rate 3072](runs/04-boosted-correction/main-rate3072/report.json) | **49.744** | **0.2062** | 0.1753 | 0.2232 | 0.2068 | 0.2194 |
| [Individual peer lags, rate 3072](runs/03-peer-lags/main-rate3072/report.json) | 44.117 | 0.3600 | 0.2455 | 0.3618 | 0.4221 | 0.4106 |
| [Adaptive rate 1536, highest tuning score in the sweep](runs/adaptive-weight-rate/main/report.json) | 44.104 | 0.3604 | 0.2410 | 0.3718 | 0.4257 | 0.4032 |
| [Adaptive rate 3072, selected on grouped training](runs/adaptive-weight-rate/main/report.json) | 44.098 | 0.3606 | 0.2411 | 0.3717 | 0.4215 | 0.4082 |
| [Regime filter, rate 3072](runs/02-regime-filter/replay/report.json) | 43.950 | 0.3652 | 0.2423 | 0.3764 | 0.4247 | 0.4173 |
| [Signed lag motifs](runs/01-signed-motifs/replay/report.json) | 42.026 | 0.4277 | 0.1315 | 0.1881 | 0.7795 | 0.6116 |
| [Learned transition retrieval](runs/05-transition-retrieval/replay/report.json) | 41.233 | 0.4552 | 0.4610 | 0.4817 | 0.4670 | 0.4109 |
| [Boosted correction, rate 48](runs/04-boosted-correction/main-v1/report.json) | 37.775 | 0.5884 | 0.3861 | 0.6905 | 0.6801 | 0.5967 |
| [Original adaptive, rate 48](runs/model-comparison/comparison.json) | 33.225 | 0.8059 | 0.6866 | 0.8854 | 0.8497 | 0.8018 |
| [Self-only causal CNN](runs/05-temporal-cnn/replay/report.json) | 31.597 | 0.8989 | 0.9093 | 0.8655 | 0.9736 | 0.8473 |
| [Fixed transition retrieval](runs/05-transition-retrieval/replay/report.json) | 31.438 | 0.9085 | 0.9588 | 0.9330 | 0.8858 | 0.8564 |

The rate sweep held half-lives at 12 and 48 ticks and prior strength at 8. It used realized cutoff Brier across four episode-grouped training folds to choose rate 3072. The tuning difference between 1536 and 3072 was 0.006 points. A paired episode bootstrap for rate 1536 minus 3072 gave a 95% interval for `ΔR` of approximately `[-0.0030, +0.0028]`. [Full sweep](runs/adaptive-weight-rate/main/report.json).

The 60-tree model fits small log-odds corrections to the adaptive win probability. Base features are built before each historical label and population priors are cross-fitted by episode inside the training folds. Against rate 3072, boosting reduced `R` by **0.1544**, with a paired 95% interval of `[-0.1810, -0.1281]`, and reduced raw `p_0` squared error from 0.0434 to 0.0206. It improved every condition. The standalone score is **16.519 points above the original adaptive control**. [Forecasts](runs/04-boosted-correction/main-rate3072/tuning_predictions.csv), [grouped training evidence](runs/04-boosted-correction/main-rate3072/train_folds.json).

The individual peer model and regime filter gave no clear gain over their matched rate-3072 control. The lagged peer check compared aligned peers with an 11-tick shifted control. The peer-triggered regime reset was gated out after the local filter lost on grouped training. [Peer results](runs/03-peer-lags/main-rate3072/report.json), [regime results](runs/02-regime-filter/replay/report.json).

## Reproduce the lead experiment

Run from the repository root with the model dependencies installed. Use fresh output paths.

```sh
python -B run_adaptive_weight_sweep.py \
  --out runs/adaptive-weight-rate/fresh
python -B run_boosted_correction.py \
  --adaptive-rate 3072 \
  --control runs/adaptive-weight-rate/fresh/tuning_predictions.csv \
  --out runs/04-boosted-correction/fresh-rate3072
python -B run_combine_forecasts.py \
  --adaptive-3072 runs/adaptive-weight-rate/fresh/tuning_predictions.csv \
  --boosted-3072 runs/04-boosted-correction/fresh-rate3072/tuning_predictions.csv \
  --out runs/06-combine-forecasts/fresh
```

The final file in that run is `fixed__boosted_3072_logistic_head.csv`. The individual reports record their settings, commands, training folds, timing, probability hashes and replay checks. The saved forecasts were independently rescored with the official tuning scorer. Run artifacts are stored under `runs/`. Historical report commands and source hashes record their original run locations.
