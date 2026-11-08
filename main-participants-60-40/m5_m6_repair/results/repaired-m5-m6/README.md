# M5/M6 repair results

**Keep M3 as the main-round recommendation.** The repaired M6 gains only 0.013672 validation points over M3 and does not improve held-episode training diagnostics. Repaired M5 is worse on both comparisons. These results do not establish a hidden-test winner.

## Official validation comparison

All 96 validation episodes and 768 queries; higher points are better.

| Model | Points / 60 | Change versus M3 |
|---|---:|---:|
| M3 | 55.268370 | +0.000000 |
| M4 | 55.282990 | +0.014620 |
| M5 | 54.841763 | -0.426608 |
| M6 | 55.282043 | +0.013672 |

M6's paired loss-ratio change is −0.000268, with a descriptive 95% episode-bootstrap interval of [−0.000796, +0.000292]. M5's change is +0.008445 [ +0.003679, +0.013916 ]. Lower loss ratio is better. The public validation set was already used in earlier model development, and these fixed-forecast intervals are not adjusted for model selection.

Before repair, M5 scored 54.987969 and M6 scored 55.296582. Repairing the implementation lowers these saved validation scores by 0.146206 and 0.014539 points respectively. The repair fixes correctness and evaluation provenance; it does not improve predictive performance in this experiment.

See [comparison.md](comparison.md) and [comparison.json](comparison.json) for condition scores, paired comparisons, and exact values.

## Held-episode assessment of the complete pipeline

Every member, tree-count choice, and blend-weight fit was repeated inside two outer training partitions. Each outer model trained on 48 episodes and predicted the other 48. All machines and pseudo cutoffs of each episode stayed together.

| Model | Binary log-loss: all 6,144 pseudo queries | Binary log-loss: 768 original cutoffs |
|---|---:|---:|
| M3 | 0.303219 | 0.266357 |
| M5 | 0.313175 | 0.277766 |
| M6 | 0.303447 | 0.267517 |

Lower is better. These diagnostics use noisy binary win/loss outcomes, whereas the contest uses five true probabilities. Half-size fits do not directly estimate the final full-train model's test score. They nevertheless provide no evidence that M5 or M6 improves on M3. See [train_outer_summary.md](train_outer_summary.md) and [train_outer_report.json](train_outer_report.json).

## Repairs and fitted configuration

- Recent-history counts now use each row's own 64 ticks before its cutoff. Total observation counts also exclude at/after-cutoff storage. M3's direct forecast calculation is unchanged.
- M5 uses nested episode crossfit: every M4 feature fit excludes the episode it predicts, and tree-count selection excludes its held episodes from every training layer. The final classifier learns from crossfit features and uses a full-training M4 at inference.
- M6 builds all four meta columns from held-episode member fits. Each M5 member is fully fitted and selected inside that partition. An additional outer assessment refits blend weights as well as members.
- Cached and supplied fits are guarded by training-data fingerprints, including history and labels. Selected M5 loss and fitted M6 weight loss are explicitly labeled as selection/training diagnostics.

The original feature list, classifier settings, tree-count grid, and blend optimizer were retained. Final M5 chose **100 trees**. Final M6 weights are approximately **67.04% M3, 21.67% M4, and 11.29% M5**, with negligible M2 weight. Exact fitted models are in `models.joblib`; readable settings are in [fitted_configurations.json](fitted_configurations.json).

## Verification and artifacts

- All **10 regression tests pass** in the installed checkout. Independent review found no blocking leakage or alignment defect; see [REPAIR_REVIEW.md](REPAIR_REVIEW.md).
- Validation/test inference used query and history files only. Code, data, configuration, checkpoint, and forecast hashes were recorded before validation scoring; see [plan.json](plan.json) and [freeze.json](freeze.json).
- Checkpoint replay has maximum absolute probability error **2.23e-16**. M3 differs from its archived validation forecasts by at most **1.34e-14** and test forecasts by at most **4.08e-14**.
- Actual fitted models pass row-permutation, extra-padding, and eight isolated complete-episode checks. The maximum change across all four models is **2.61e-14**; see [inference_verification.json](inference_verification.json).
- All four test CSVs pass the official checker for **1,536 queries**; see [test_csv_checks.txt](test_csv_checks.txt). Test targets are private, so no test score was computed.

Fitted test forecasts: [M3](m3/submission.csv), [M5](m5/submission.csv), [M6](m6/submission.csv). Training and freeze took about **14 minutes** with two numerical-library threads. Reproduction commands are in [M5_M6_REPAIR.md](../../M5_M6_REPAIR.md). No upload was performed.
