# Repaired M5/M6 compared with unchanged M3

Single fixed repair experiment on all 96 previously used validation episodes; not a new audit.

| Model | Points / 60 | R | Stable | Short history | Drift | Shared change |
|---|---:|---:|---:|---:|---:|---:|
| m3 | 55.268370 | 0.085612 | 0.077244 | 0.106894 | 0.079817 | 0.078493 |
| m4 | 55.282990 | 0.085325 | 0.077314 | 0.105820 | 0.079818 | 0.078347 |
| m5 | 54.841763 | 0.094057 | 0.083822 | 0.116690 | 0.088514 | 0.087201 |
| m6 | 55.282043 | 0.085343 | 0.077150 | 0.106681 | 0.079534 | 0.078008 |
| previous_m3 | 55.268370 | 0.085612 | 0.077244 | 0.106894 | 0.079817 | 0.078493 |
| previous_m5 | 54.987969 | 0.091148 | 0.078697 | 0.111637 | 0.088928 | 0.085329 |
| previous_m6 | 55.296582 | 0.085058 | 0.076507 | 0.105740 | 0.079860 | 0.078125 |

| Repaired candidate minus M3 | ΔR | Descriptive 95% interval |
|---|---:|---|
| m5 | +0.008445 | [+0.003679, +0.013916] |
| m6 | -0.000268 | [-0.000796, +0.000292] |

Lower R is better. Intervals resample episodes within condition with fixed forecasts.
See train_outer_report.json for separately held-out whole-pipeline training diagnostics,
fitted_configurations.json for selections, and freeze.json for hashes and replay checks.
