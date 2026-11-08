# Held-episode training diagnostics

Each outer model was fitted on 48 of the 96 training episodes. All model fitting, M5 tree-count selection, and M6 weight selection were refitted inside that partition. Every machine and pseudo cutoff from a held episode remained excluded from those decisions.

| Model | All 6,144 pseudo queries: binary log-loss | 768 released cutoffs: binary log-loss | Released cutoffs: binary Brier |
|---|---:|---:|---:|
| M3 | 0.303219 | 0.266357 | 0.076629 |
| M4 | 0.303378 | 0.267504 | 0.077028 |
| M5 | 0.313175 | 0.277766 | 0.079130 |
| M6 | 0.303447 | 0.267517 | 0.076982 |

Lower is better. These metrics use observed win/loss outcomes, whereas the contest scores five-class predictions against true simulator probabilities. The outer results support retaining M3: M5 is worse on both losses, and M6 provides no observed improvement. They are not direct estimates of the full-data model’s hidden-test score.

M6 fitted weights were 94.4753% M3 and 5.5247% M5 in the first outer fold, and effectively 100% M4 in the second. M5 selected 50 and 100 trees respectively. The difference in weights illustrates selection variability between these training partitions.

Full metrics and fold membership are recorded in `train_outer_report.json`; row-aligned predictions are in `train_outer_predictions.npz`.
