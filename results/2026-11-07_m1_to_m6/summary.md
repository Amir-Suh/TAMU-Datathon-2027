# Run `2026-11-07_m1_to_m6`

Validation scores. Every model is fitted on **train only**; hyperparameters are chosen on train. Points use the kit's own scorer (`contest_score`, max 60). Loss ratio = model error / frozen pooled reference error for that condition (lower is better, 1.00 = reference).

| Model | Points /60 | Δ vs reference | Stable | Short history | Drift | Shared change | Mean ratio |
|---|---|---|---|---|---|---|---|
| `m6_ensemble` | **55.30** | +25.30 | 0.077 | 0.106 | 0.080 | 0.078 | 0.085 |
| `m4_lag_pair_flip_shared` | **55.28** | +25.28 | 0.077 | 0.106 | 0.080 | 0.078 | 0.085 |
| `m3_lag_pair_flip` | **55.27** | +25.27 | 0.077 | 0.107 | 0.080 | 0.078 | 0.086 |
| `m5_boosted_corrector` | **54.99** | +24.99 | 0.079 | 0.112 | 0.089 | 0.085 | 0.091 |
| `m2_lag_pair` | **52.60** | +22.60 | 0.082 | 0.119 | 0.211 | 0.151 | 0.141 |
| `m1_hierarchical_frequency` | **31.09** | +1.09 | 0.962 | 0.979 | 0.933 | 0.845 | 0.930 |
| `baseline_pooled_recent` | **30.83** | +0.83 | 0.958 | 1.001 | 0.946 | 0.880 | 0.946 |
| `baseline_pooled` | **30.00** | +0.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Extra diagnostics (validation)

| Model | Mean Σ sq. error | Error: steady | Error: balanced | Error: jackpot | Log-loss on observed outcomes | Runtime (s) |
|---|---|---|---|---|---|---|
| `m6_ensemble` | 0.0162 | 0.0191 | 0.0176 | 0.0122 | 0.7650 | 96.0 |
| `m4_lag_pair_flip_shared` | 0.0163 | 0.0191 | 0.0175 | 0.0124 | 0.7654 | 107.5 |
| `m3_lag_pair_flip` | 0.0164 | 0.0191 | 0.0176 | 0.0125 | 0.7655 | 101.2 |
| `m5_boosted_corrector` | 0.0174 | 0.0199 | 0.0192 | 0.0132 | 0.7672 | 42.5 |
| `m2_lag_pair` | 0.0271 | 0.0332 | 0.0270 | 0.0208 | 0.7874 | 57.1 |
| `m1_hierarchical_frequency` | 0.1783 | 0.2396 | 0.2083 | 0.0897 | 1.0392 | 0.6 |
| `baseline_pooled_recent` | 0.1815 | 0.2396 | 0.2130 | 0.0948 | 1.0530 | 6.1 |
| `baseline_pooled` | 0.1922 | 0.2447 | 0.2210 | 0.1136 | 1.0692 | 6.4 |

## Files

* `<model>/validation_predictions.csv`: validation forecasts (score with `python check_submission.py <file> --split validation` from the kit folder).
* `<model>/submission.csv`: test forecasts in upload format; format-checked against the test queries. Test answers are private, so these are not scored.
* `<model>/params.json`: fitted priors, chosen hyperparameters and the train-only selection table.
* `m2_lag_pair/validation_rules.csv`: the lag pair and direction model 2 believes each validation machine follows.
* `loss_ratio_by_condition.png`: the table above as a chart.
* `scores.json`: everything above in machine-readable form.
