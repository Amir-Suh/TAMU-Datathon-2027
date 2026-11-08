# Model 6: ensemble

Model 6 blends the win probabilities of Models 2–5 with nonnegative weights that sum to one. P(size | win) comes from Model 4.

## Weight fitting

The meta-training rows are the eight pseudo cutoffs from each training query. Four episode folds produce every meta column. For a given held episode, each of Models 2, 3, and 4 is fitted and tuned on the other episodes. Model 5 is fitted on those same other episodes, including its own nested episode crossfit and round selection. No supplied prefit member is used to construct held-episode meta rows.

Weights minimize binary log-loss on that held-episode prediction matrix. The reported blend loss is **in-sample for the weights** because those same rows both fit and score the weights. The individual member losses use held-episode forecasts, but do not assess the effect of weight fitting.

`Ensemble.evaluate_heldout(train, folds=...)` assesses the entire procedure: in each additional outer episode fold, it refits all four members, their selections, and the weights within the outer training episodes, then predicts the held episodes. It returns aligned pseudo-query IDs, episode IDs, labels, endpoint flags, member and blend predictions, and each fold's weights and Model 5 round choice. Its binary log-loss is a separate held-episode assessment of the full pipeline.

Final deployment members are fitted on all training episodes. Supplied prefit members are reused only if their data fingerprints match that final partition. Current validation and test results are reported by the repaired comparison run; scores from the archived implementation do not apply to this fit.

## Repaired run

The fixed repair run scores **55.282043 / 60** on the previously used public validation set, versus M3 at **55.268370**. Weights are approximately **67.04% M3, 21.67% M4, 11.29% M5**, and negligible M2. The 0.014-point gain over M3 has a paired interval crossing zero and no improvement in held-episode training diagnostics. See the [repair results](../../results/repaired-m5-m6/README.md).
