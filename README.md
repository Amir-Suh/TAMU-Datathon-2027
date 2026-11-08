# House Edge · TAMU Datathon

Given a simulated slot machine's observed history, machine type, and signal, predict the probabilities of **0×, 1×, 2×, 5×, and 20×** on its next spin. The prediction tick is the query's cutoff. An episode groups eight machines, and every input spin must precede that cutoff. Some histories contain gaps, and a machine's win rule can change. [Problem and data format](main-participants-60-40/DATA.md) · [Organizer rules](main-participants-60-40/RULES.md)

The scorer compares all five predictions with the simulator's true probabilities. It averages squared error by episode and normalizes it against a frozen pooled forecast in each of four equally weighted conditions. The resulting ratio, `R`, gives `60 / (1 + R)` main points or `40 / (1 + R)` adaptation points. Adaptation has fresh episodes, changed patterns, and its own frozen reference. [Scoring and round details](main-participants-60-40/RULES.md#how-points-work)

## Main-round result

**W1a + P0 scored 56.425331 / 60** on the 96 released main validation episodes, versus 56.341022 for the frozen hybrid. W1a refines Amir's M3 win forecast by distinguishing ordered lag outcomes. Fredy's lag-pair LightGBM and three factored GRUs estimate payout sizes. P0 then preserves 0× and 1× and redistributes the remaining mass across 2×, 5×, and 20× using training-history ratios by machine type. All four condition scores improved. Paired ΔR was −0.001591 with a 95% interval of [−0.002449, −0.000821]. [Experiment report and data-access audit](main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/REPORT.md)

The **main** test CSV and source ZIP reproduced all 1,536 query forecasts offline under Linux Python 3.13. The maximum normalized difference was 1.38e−14. Fredy’s latest 12-GRU blend is reported at about 56.58. A local three-GRU reconstruction scored 56.481285, with an inconclusive paired gain and regressions in two conditions. Public validation guided development. The adaptation mechanism tests use synthetic data, and a separate released-round evaluation is required to choose an adaptation entry. [Main test CSV](main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/coordinator/assembly/test_W1a_P0.csv) · [Verified code ZIP](main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/package/02_w1a_p0/w1a_p0_code.zip) · [Replay instructions](main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/package/01_frozen_controls/REPLAY.md)

## Experiment tracks

| Track | Location |
| --- | --- |
| Andrew's early models | Frequency and logistic baselines, causal lag features, Bayesian filters, boosting, CNN, and forecast blends. [Runners and saved experiments](experiments/andrew/README.md) |
| Amir's models | M1–M6, including lag-pair polarity, inferred flip times, and synchronized changes. [Models](models/README.md) · [Original results](results/README.md) |
| Fredy's bake-off | Pooled, GLM, LightGBM, GRU, regime, short-history study, Transformers, and factored ensemble packaging. [Source](src/tdx/arms) · [Run log](runs/LOG.md) · [Research notes](claude_output/00_README.md) |
| Joint comparisons | M5/M6 repairs, component replays, hybrid assembly, E1–E6, and the known-law adaptation work. [Unified experiment index](docs/EXPERIMENTS.md) |

## Check the archived main forecast

From the repository root, the kit checker uses the Python standard library.

```sh
kit=main-participants-60-40
run="$kit/signal_experiments/next_stage/20261107_dispatch"
python3 -B "$kit/check_submission.py" \
  "$run/coordinator/assembly/validation_W1a_P0.csv" --split validation
python3 -B "$kit/check_submission.py" "$run/coordinator/assembly/test_W1a_P0.csv"
```

The validation check computes points. The test check verifies query coverage and probability format. To compare the archived models on identical queries and conditions, use the common scoreboard command in the [experiment index](docs/EXPERIMENTS.md#reproduce-the-common-validation-comparison).
