# House Edge experiment index

This branch combines the three model tracks and the joint validation work. Amir's and Fredy's original layouts remain runnable at the repository root. Andrew's earlier experiments live together under `experiments/andrew/`. The organizer kit, the repaired M5/M6 implementation, and the cross-track experiments retain their kit-relative paths.

| Source | Revision or snapshot | Contents |
| --- | --- | --- |
| Amir | `bb0357e` on `amir_workspace` | [Model source](../models/) · [Lag and shared-change findings](../models/FINDINGS.md) · [Original M1–M6 results](../results/2026-11-07_m1_to_m6/summary.md) |
| Fredy | `562f826` on `fredy/bakeoff-gbdt-seq` | [Experiment plans and notes](../claude_output/00_README.md) · [Arm source](../src/tdx/arms/) · [Run log](../runs/LOG.md) |
| Andrew | Local early-model experiments | [Runners, model briefs, and archived run outputs](../experiments/andrew/README.md) |
| Joint | November 7, 2026 dispatch | [Full report](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/REPORT.md) · [Run manifest](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/manifest.json) · [Cross-branch review](../main-participants-60-40/adversarial_review/branches/REVIEW.md) |

## Common main-validation scoreboard

96 episodes, 768 queries, and four equally weighted conditions. These rows have saved predictions on the same release. Branch log entries with unavailable predictions and earlier tuning/audit subsets are reported separately.

| Configuration | Points / 60 | Scope |
| --- | ---: | --- |
| Pooled reference | 30.000000 | Official denominator |
| Andrew's switching Bayesian win with logistic size | 53.037228 | Best earlier standalone combination on full validation |
| Fredy's 31-leaf lag-pair LightGBM | 54.289888 | Local archived checkpoint replay |
| Amir's M3 explicit flip-time model | 55.268370 | Frozen Bayesian control |
| Amir's M4 shared-flip model | 55.282990 | Episode-synchronized comparison |
| Repaired M6 ensemble | 55.282043 | Fold-refitted M5/M6; paired comparison with M3 was inconclusive |
| Fredy's three factored GRUs | 55.406231 | Local archived checkpoint replay |
| Fredy's seven plain GRUs | 55.419191 | Local archived checkpoint replay |
| Equal M3 and factored GRU probability blend | 56.071861 | Local reconstruction with archived weights, using fetched source |
| Split-Match M3 with LightGBM/GRU payout mix | 56.341022 | Frozen champion hybrid |
| Champion hybrid + P0 | 56.385873 | Hard train-history tail projection |
| **W1a + P0** | **56.425331** | Selected main package; improves all conditions and passes offline replay |
| Fredy’s factored weights with three local GRUs | 56.481285 | 70% M3/30% GRU win probability, GRU payout mix. Inconclusive paired gain vs W1a + P0. |

W2 + P0 reached 56.426385. Its paired improvement over W1a + P0 was inconclusive. Fredy’s published `seq2` and Transformer runs and the held-tail studies use their own reporting scopes. His notes report 56.580–56.581 for the 12-GRU blend. The fetched branch documents the weight rule and omits the full trained member set and prediction CSV. [Branch refresh and dispatch results](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/REPORT.md) · [Earlier control lineage](../main-participants-60-40/HOUSE_EDGE_AI_HANDOFF.md)

## Experiment families

| Family | Question and entry point |
| --- | --- |
| Andrew's causal baselines | How much do type, signal, lagged outcomes, peer history, or learned regime changes add? See the [runner catalog](../experiments/andrew/README.md). |
| Amir M1–M6 | Can each machine infer its own lag pair, polarity, and flip time? Compare the Bayesian family, synchronized flips, and the ensemble. [Model documentation](../models/README.md) |
| Repaired M5/M6 | Does the ensemble survive cutoff-aligned recent history and episode-heldout component fits? [Repair results and reproduction](../main-participants-60-40/m5_m6_repair/README.md) |
| Fredy's LightGBM and sequence models | Compare lag-pair features, causal GRUs, Transformers, conditional payout estimates, and masked-history training. [Published logs](../runs/LOG.md) · [Replayed local checkpoints](../main-participants-60-40/signal_experiments/01_fredy_replay/README.md) |
| Hybrid and structural follow-ups | Combine the Bayesian win model with the tree/GRU payout mix, then test ordered cells, state coupling, tail projection, and missing-history filtering. [Combined forecasts](../main-participants-60-40/signal_experiments/combined/README.md) · [Research scoreboard](../main-participants-60-40/signal_experiments/research_next/README.md) |
| Adaptation readiness | Test changed lag pairs, repeated flips, and sparse inputs against known synthetic probabilities. E5 remains a synthetic mechanism result. E6's first comparison used main data. [E5](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/adapt_pairs/attempt_01/REPORT.md) · [E6](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/neural/attempt_01/REPORT.md) |

## Reproduce the common validation comparison

Run from the repository root with NumPy available. The default frozen M3 forecast is the baseline for paired comparisons, while the official pooled reference remains the denominator for `R`.

```sh
kit=main-participants-60-40
d="$kit/signal_experiments"
run="$d/next_stage/20261107_dispatch"
python3 -B "$d/score.py" \
  "amir_m4=results/2026-11-07_m1_to_m6/m4_lag_pair_flip_shared/validation_predictions.csv" \
  "fredy_tree=$d/01_fredy_replay/forecasts/pair_l31.csv" \
  "fredy_factored_gru=$d/01_fredy_replay/forecasts/gru128_factor_mean_3.csv" \
  "hybrid=$d/combined/m3_match_split_win_pair_l31_factor_gru_size.csv" \
  "fredy_factored_blend=$run/coordinator/fredy_factored_validation.csv" \
  "candidate=$run/coordinator/assembly/validation_W1a_P0.csv"
```

For Amir's refits, install the [model requirements](../requirements.txt) and run `python -B -m models.run --models m3 --data main-participants-60-40/public --out <new-dir>`. For Fredy's runner, use its recorded environment and pass the explicit data directory with `python -B -m tdx.run --arm pooled --variant pooled_recent --data main-participants-60-40/public --out <new-dir>`. The `kit` symlink resolves Fredy's organizer-kit imports. [Amir's model runner](../models/run.py) · [Fredy's setup and runner](../claude_output/00_README.md)

To run the M5/M6 repair from its own directory, use `PYTHONPATH=.. python -B run_repaired_comparison.py prepare --data ../public --out <new-dir> --previous ../../results/2026-11-07_m1_to_m6`, then `PYTHONPATH=.. python -B run_repaired_comparison.py score --data ../public --out <new-dir>`. Use the [pinned repair environment](../main-participants-60-40/m5_m6_repair/requirements-repaired-models.txt).

The main CSV and ZIP are frozen audit artifacts. To create a fresh candidate ZIP, run the [W1a + P0 builder](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/package/02_w1a_p0/build.py) with `--out <new-zip>` in the repaired-model environment. The organizer replay uses Python 3.13 and the ZIP's own dependency pins. Temporary smoke outputs, native objects, duplicate replays, and the regenerable E5 synthetic prediction archive are excluded. Retained historical logs can reference their original host paths. [Package replay](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/package/01_frozen_controls/REPLAY.md) · [E5 regeneration commands](../main-participants-60-40/signal_experiments/next_stage/20261107_dispatch/adapt_pairs/attempt_01/REPORT.md)

## Comparing new runs

Record the round, dataset fingerprint, upstream model revisions, training split, seeds, prediction files, and exact replay command. Keep episodes together in grouped folds. Use only pre-cutoff outcomes to construct inputs, and record any self-supervised exposure to validation or test histories. Score released odds separately from historical outcome log loss and synthetic SSE. Compare against a fixed baseline using paired episode draws. For adaptation, use the round's own reference and reserve complete validation episodes for the final comparison. [Evaluation and promotion rules](../main-participants-60-40/signal_experiments/EXPERIMENT_PLAN.md#evaluation-and-promotion)
