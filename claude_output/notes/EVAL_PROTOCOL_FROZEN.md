# Frozen evaluation rules (written 2026-11-08 00:20 CT, before computing the results they govern)

Why: until now most choices (seeds, epochs, blend weights, presets) were judged on the full 96-episode validation score and then reported on it. That is optimistic, and a 12-seed GRU ensemble moves by about +-0.3 points with the seed set alone (CE seeds 11-22: 55.77; CE seeds 41-52: 56.29; same recipe). These rules apply from now on.

1. **Freeze first.** The candidate list, member sets, blend structure, fold seeds, metric and decision rule are in `frozen_candidates.json` and are committed BEFORE results are computed. No candidate is added or changed after seeing scores.
2. **Cross-validate by episode.** 4 folds grouped by episode and stratified by condition, repeated 5 times with different shuffles. Blend weights are fit on three folds and scored on the fourth; the reported number is the average out-of-fold score. A model/blend is never scored on episodes used to fit its weights.
3. **Matched seeds for any claim about a training change** (a loss term, an architecture): same seeds, paired differences, bootstrap interval. Differences between ensembles built from different seed sets are not attributed to the training change.
4. **Intervals, not point gaps.** Decide with a paired episode bootstrap (2000 resamples, stratified by condition) and a minimum gain of 0.2 points; otherwise keep the incumbent.
5. **Leak-free tail for anything that needs outcomes.** Held-out-tail evaluations use `tdx.tools.tail_eval` with the neutral-signal policy (Andrew's fix); forecast files made before that fix are rejected by the tool. Short-history backtests use the corrected episode-disjoint `short_backtest`.
6. **Report with caveats.** A validation figure (even cross-validated) is an estimate for 96 episodes; the test set has 192 unseen episodes and will differ.
7. **Shipping.** Ship the average of the fold weights (every shipped number was scored in a fold), not an unscored refit on all 96.

## Addendum (2026-11-08 01:00 CT): overfitting diagnostics, written before they were run
- **Epoch curves** (`tdx.tools.epoch_curves`): GRU, plain CE and size x2, seeds 41-43. (a) held-out-tail log-loss at checkpoints every 10 epochs on tail-hidden refits (no validation answers); (b) validation points by epoch on the full-data runs (a diagnostic only, never used to fit). These are DIAGNOSTICS: the epoch count (100) is changed only if the tail log-loss at the best checkpoint beats epoch 100 by at least 0.003 in all three seeds for the same variant; that change would then be a new frozen decision with a matched-seed re-evaluation.
- **Transductive on/off** (matched seeds 41-43, plain CE): validation points with `transductive=false` (train histories only) vs the existing `true` runs (`runs/ce_ens`). Reported as paired differences; it measures how much the models rely on seeing a machine's own history, not a candidate for shipping.

## Decision log (2026-11-08 ~01:20 CT)
- **Candidates G and H (size x2 transformer) were NOT scored in the blend test.** Reason, decided by the team before looking at any blend score: the matched-seed single-model result is already negative (8 seeds, same seeds 21-28: plain 54.54 vs size x2 54.08, paired difference -0.46, se 0.22, size x2 higher in 1 of 8). The forecasts needed (`tfm_s2_8`) are in `runs/tfm_s2` if anyone wants the formal test; nothing is shipped from it.
- The adversarial leakage audit (`claude_output/codex/LEAKAGE_REPORT_2026-11-08.md`) found future-observedness leakage in the research arms moegate, moegate4 and seq2 and a missing-token bug in shortctx; none of these arms is in any submitted entry. Their earlier results (moegate +0.15% Brier, moegate4, shortctx negative result) are therefore not reliable and the planned moegate K=3/K=4 blend test is on hold until the flag is made causal.

## Addendum 3 (2026-11-08 02:00 CT): results of the two diagnostics and the follow-up design, written before the follow-ups ran
**Observed (seeds 41-43, plain-CE GRU, validation points / held-out tail):**
- Transductive ON (train+validation+test histories) 54.65 vs OFF (train histories only) **34.29**: a 20-point gap. The baseline is 30. Without studying the validation/test machines' histories the GRU barely learns the rule.
- Held-out-tail log-loss by epoch (mean of 3 seeds): CE 0.7578 (70), 0.7570 (80), 0.7589 (90), 0.7632 (100); best epoch per seed 80/60/80, gain of the best checkpoint over epoch 100: 0.0087, 0.0071, 0.0042 -> the pre-registered epoch rule (>= 0.003 in all 3 seeds) IS met for plain CE. Size x2: best at 100/100/90 (rule not met). The win part of CE turns up after epoch 80 (0.3331 -> 0.3384); the size part keeps improving to 90. Train loss and tail loss differ by only ~0.004: little classic memorisation. Validation points at epoch 80 vs 100 (full-data runs): CE +0.22 (3/3 seeds higher), size x2 -0.34.

**Follow-ups (fixed now):**
1. **Disentangle the 20-point gap** (GRU CE, seeds 41-43, `--set tick_splits=...`): B = train+test histories only (validation machines unseen, 3x the episodes of A); D = train+validation histories only. Reference: A = train only 34.29, C = all 54.65. Pre-registered reading: if B - A >= 10 points the gap is mostly data volume; if B - A <= 3 it is mostly the model studying each machine's own history; in between, both.
2. **Epoch rule triggered for CE:** retrain plain CE at 80 epochs on seeds 41-52 (training is deterministic with a constant learning rate, so this equals the epoch-80 checkpoint) and evaluate with the frozen cross-validated blend as candidate A80 vs A (matched seeds); the usual decision rule (gain >= 0.2, interval excluding 0) applies; otherwise 100 epochs stays.
3. **Same diagnostics for the other members:** transductive-off for the plain transformer (seeds 21-23) and size x2 GRU (41-43).
4. **MoE arms were refitted-from-scratch required** (causal flag fix, Codex W9): the 8 partial runs from the old non-causal flag are invalid. Retrain moegate4 k3_ref and k4_nocond seeds 41-48 and run the already-frozen addendum 2 test (extra-GRU control) unchanged.

## Result of follow-up 1 (2026-11-08 ~02:00 CT) - CORRECTS the earlier "own-history" reading
GRU plain CE, seeds 41-43, validation points (measured, `runs/disent`, `runs/transd_off`, `runs/ce_ens`):
- A train histories only (96 episodes): 34.29 | D train+validation (192): 50.68 | B train+test, validation machines UNSEEN (288): **53.92** | C all (384): 54.65.
- B - A = +19.63; C - B = +0.73. By the pre-registered reading (B - A >= 10) the gap is **data volume**, not familiarity with each machine's own history. The model learns a rule that transfers to machines it has not seen, as long as it is trained on enough machine histories (learning curve 34.3 -> 50.7 -> 53.9 -> 54.7, saturating).
- My earlier statements that the model "gets most of its skill from studying each machine's own history" and that unseen-machine performance would be about 34 were wrong; the 34 is a data-starved model. Consequence for round two: train on ALL released histories of the round (train+validation+test), and consider adding the main-round histories ("main data may be reused"), judged on the new validation.

## Decisions log, 2026-11-08 ~02:15 CT
- (default I picked, needs user confirmation) Build candidate 4 = recipe D (m3 + 12 CE GRUs + 12 size x2 GRUs + 8 transformers) with the fold-averaged weights of the frozen cross-validation, as a hedged "best available" entry. The frozen replacement rule was NOT met by any candidate (D gain +0.171 vs F, below 0.2); choosing D is a judgment call, not a rule outcome. Not submitted.
- (user, earlier) Transformer size x2 blend test (G/H) skipped: matched-seed single-model result negative.
- (user) Rules above (frozen candidates, 4-fold grouped CV x5, matched seeds, intervals) apply from now on.
