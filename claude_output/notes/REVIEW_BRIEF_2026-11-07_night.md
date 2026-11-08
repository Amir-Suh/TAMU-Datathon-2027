# CORRECTION (23:55 CT): the size x2 claim in section 3 is NOT supported

A same-seed comparison was run after this brief was first pushed: 12-member GRU ensembles on seeds 41-52, plain CE (`runs/ce_ens`) vs size x2 (`weights/seqens_size2_ens`), validation, dev mode.
- Ensembles: CE 56.294 vs size x2 56.405 (difference +0.11; seed bootstrap 95% interval [-0.15, +0.35]).
- Single seeds: CE mean 54.608 (sd 0.49) vs size x2 54.612 (sd 0.41); paired per-seed difference +0.005, se 0.19, positive in 6 of 12.
- The earlier "+0.63" (55.77 vs 56.41) compared DIFFERENT seeds: the baseline ensemble used seeds 11-22 (single-seed mean 53.75), the size x2 ensemble seeds 41-52 (54.61). The plain-CE ensemble on seeds 41-52 alone scores 56.29. Random 12-subsets of the 24 CE members score 56.06 (sd 0.09, range 55.87-56.25), so a 12-member ensemble's validation score moves by about +-0.3 with the seed set.
- Treat the 4-matched-seed result (+0.60) and the tail log-loss bootstrap (3 seeds, -0.0036) as likely seed luck until reproduced on more seeds; pooled paired difference over all 16 matched pairs is about +0.15 (not significant).
- Consequence: gains of 0.1-0.3 points reported elsewhere in this brief (transformer, moegate, blend weights) are of the same size as the seed-set noise of a 12-member ensemble. The submitted 3b uses the weaker of the two baseline GRU ensembles (55.77); an ensemble of all 24 CE members scores 56.16.

# Adversarial review brief — state at 2026-11-07 23:05 CT (for the other member of technical staff)

Branch: `fredy/bakeoff-gbdt-seq`. Please try to break the claims below. Everything is reproducible from this branch unless marked "files not in git". Main closes Sun 10:30 CT; round two 10:30–12:00.

## 1. What has been submitted (user-reported) and what is proposed
| Entry | What | Validation (dev mode, reported) |
|---|---|---|
| candidate 3b (original) | m3 + 12 GRUs + 8 conv/ALiBi transformers, factored win/size blend | 56.90 |
| candidate 3b clean refit (Andrew, branch `andrew-exp`) | same members, m3 with signal-timing fix, blend refit with a reserved fold | 56.75; 56.42 on the 24 reserved episodes. **Submitted by the user ~22:58 CT** |
| proposed next: 3b + size×2 GRUs | clean-refit recipe with the 12 GRUs trained with the size-weight-2 loss | not built yet; see section 3 |

Verified by me for the clean refit (not recomputed: its 56.75 / 56.42): checksums, `kit/check_submission.py` VALID FORMAT, fresh-venv offline reproduction (2 identical runs, max diff vs CSV 2e-14, no torch, 1.2 GB, 306–326 s).

## 2. Known flaws found in our own evaluation (please look for more)
1. **Signal timing (Andrew, `a7d6e80`).** In the held-out-tail tests, earlier pseudo-cutoffs carried the REAL cutoff's `signal`, which m3 uses. I measured the effect on m3, hold 30, 23,040 outcomes: Brier 0.36681 (ours) vs 0.36759 (his neutral signal), win part 0.3053 vs 0.3073, mean forecast difference 0.006. So m3's tail scores, and blend weights fit on them, were slightly optimistic (~0.2% Brier). GRU/transformer arms do not use signal. The submitted test predictions do not use leaked information (the real signal is legitimate at the real cutoff).
2. **Label leakage in the pooled short-history backtest (Andrew, `f23598f`).** The old `short_backtest` pooled train/validation/test short episodes. Treat all earlier "real short_history backtest" numbers from this branch (m3 0.3645, shortctx 0.393, moegate 0.370 Brier) as unreliable. NOT independently reproduced by me.
3. **Validation reuse.** The 96 validation episodes have steered many choices (seeds, epochs, blend presets). The tail test is leak-free for models that do not use signal, but one hold window with ~96 episodes is a small sample; three windows (30/60/90) for the blend fits.
4. Several earlier statements of mine were wrong and were corrected in the conversation: "tail is leak-free" (see 1); "the win rule is mostly solved" (error is ~44% win / 55% size by oracle substitution on validation).

## 3. The new claim to attack: size×2 loss improves the GRU
Arm `src/tdx/arms/seqloss` (subclass of `seq`; inference and exports unchanged). `size_w=2` means loss = win BCE + 2 × size CE on wins (factored head). Evidence:
- Tail, hold 30, 3 seeds, 23,040 outcomes (log-loss; paired episode bootstrap vs plain CE, `tdx.tools.tail_eval`): total −0.0036 [−0.0060, −0.0010], size part −0.0050 [−0.0071, −0.0028]. size×4 over-corrects (total +0.0064).
- Dev validation points, 4 matched seeds (1–4): CE 54.39, 54.12, 54.22, 53.81 (mean 54.14); size2 54.31, 55.65, 54.38, 54.61 (mean 54.74). 3 of 4 seeds higher; the paired mean +0.60 has a large spread.
- Ensembles of 12 (CE seeds 11–22 vs size2 seeds 41–52, DIFFERENT seeds, `claude_output/sh/size2_ensemble.sh`): 55.77 vs 56.41 (stable .055→.047, short .107→.096, drift .074→.056, shared .066→.056). In the full blend with fixed weights, 56.90→57.18 (original 3b weights) and 56.72→56.96 (clean-refit weights) — computed with the OLD-signal m3 on this branch.
Attack surface: (a) different seeds in the 12 vs 12 comparison (seed luck? baseline single-seed mean 53.75 vs size2 54.62); (b) is the gain really from the size term or from effectively longer/higher effective LR (loss scale change)? (c) validation-only evidence for the ensemble result, no tail ensemble yet; (d) did we try size×2 on the transformer? no (its head is a plain 5-way softmax; the win/size split can be computed algebraically).
Other loss variants (tail, hold 30): Brier-only and CE+2×Brier tie CE; recency, history-length and payout-value weights are worse (validation 50–54).

## 4. Other research results (all negative or marginal)
- shortctx (exact 40+24 layout specialist): worse than the GRU (but short backtest unreliable, see 2.2; validation 48.2).
- moegate (K=3 soft mixture of experts): expert usage ≈ uniform; adding it to the 3b-style blend lowers tail Brier 0.00059 [0.00037, 0.00082] (3 seeds each, m3 old-signal). K=4 + auxiliary condition loss (`moegate4`) is being scored by an agy job; no results yet.
- Signal: stage-2 recalibration with type×signal cells +0.02 points (noise). Models' mean predictions already track true P(win) by type×signal; jackpot machines show the largest signal effect.
- Using half of the validation TRUE probabilities as soft targets hurt the other half (−0.38 mean over 8 fits); matched version `src/tdx/tools/halfval.py` is written, not run.
- Seeds: the 12-GRU ensemble plateaus (~55.8); more seeds do not help.

## 5. Safety/packaging facts worth re-checking
- Inference is numpy/scipy only: static torch ban in `src/tdx/package/adapters.py`, runtime `no_torch()` in `predict.py`, fresh Python 3.13 venv with only numpy/scipy pins in `claude_output/sh/repro_check.sh`.
- Transformer inference runs in 12-episode chunks (full batch peaked at 16.3 GB, `tfmens/arm.py`).
- Round two: `claude_output/sh/round2.sh` is written and rehearsed only at toy scale (3 min 44 s, 2 seeds each, 2 epochs). The full-size timing rehearsal (14 trainings) has NOT been run: we do not know that round two fits in 90 minutes.

## 6. Files not in git (git-ignored `runs/`, or wiped /tmp)
Model weights (`runs/seqens/ens12_dev`, `runs/tfmens/ens8_dev`, `runs/size2_ens/`), loss-study forecasts (`runs/lossstudy/`), and tail forecast files. The submission zips for candidates 1/2/3b are in `submissions/`. Regenerate with `claude_output/sh/size2_ensemble.sh`, `claude_output/sh/lossstudy.sh`, `python -m tdx.tools.tail_eval ...`.

## 7. Suggested checks for the reviewer
1. Recompute the size2-vs-CE comparison on the SAME seeds for the 12-member ensembles (run `size2_ensemble.sh` variant with seeds 11–22 for `size2` and `ce` in `seqloss`).
2. Rebuild the proposed candidate on `andrew-exp` and run `repro_check.sh`; confirm no import of torch and memory < 16 GiB.
3. Look for any other place a cutoff-only quantity (`signal`, `n_obs`, labels) is reused at earlier ticks, in all arms and tools.
4. Challenge the claim that GRU/transformer arms are independent of `signal`.
