# Frozen evaluation rules (written 2026-11-08 00:20 CT, before computing the results they govern)

Why: until now most choices (seeds, epochs, blend weights, presets) were judged on the full 96-episode validation score and then reported on it. That is optimistic, and a 12-seed GRU ensemble moves by about +-0.3 points with the seed set alone (CE seeds 11-22: 55.77; CE seeds 41-52: 56.29; same recipe). These rules apply from now on.

1. **Freeze first.** The candidate list, member sets, blend structure, fold seeds, metric and decision rule are in `frozen_candidates.json` and are committed BEFORE results are computed. No candidate is added or changed after seeing scores.
2. **Cross-validate by episode.** 4 folds grouped by episode and stratified by condition, repeated 5 times with different shuffles. Blend weights are fit on three folds and scored on the fourth; the reported number is the average out-of-fold score. A model/blend is never scored on episodes used to fit its weights.
3. **Matched seeds for any claim about a training change** (a loss term, an architecture): same seeds, paired differences, bootstrap interval. Differences between ensembles built from different seed sets are not attributed to the training change.
4. **Intervals, not point gaps.** Decide with a paired episode bootstrap (2000 resamples, stratified by condition) and a minimum gain of 0.2 points; otherwise keep the incumbent.
5. **Leak-free tail for anything that needs outcomes.** Held-out-tail evaluations use `tdx.tools.tail_eval` with the neutral-signal policy (Andrew's fix); forecast files made before that fix are rejected by the tool. Short-history backtests use the corrected episode-disjoint `short_backtest`.
6. **Report with caveats.** A validation figure (even cross-validated) is an estimate for 96 episodes; the test set has 192 unseen episodes and will differ.
7. **Shipping.** Ship the average of the fold weights (every shipped number was scored in a fold), not an unscored refit on all 96.
