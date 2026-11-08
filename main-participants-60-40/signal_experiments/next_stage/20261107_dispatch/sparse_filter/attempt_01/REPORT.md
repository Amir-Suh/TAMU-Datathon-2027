# E4 · Fixed-emission gap filtering and a support gate

**Main disposition: hold the filter for sparse adaptation research and reject the gate for main.** The strongest new released-history arm, the constant blend, scored **56.267897 / 60** versus the frozen champion's **56.341022**. Its paired ΔR was **+0.001384** [−0.002254, +0.004760]. It improved the short-history condition and trailed the champion in the other three. The support gate lost to the constant blend overall. No E4 arm cleared the main promotion gate or entered assembly.

| Released main arm | Points | Overall R | Stable R | Short-history R | Drift R | Shared-change R | p0 MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen hybrid champion | 56.341022 | .064943 | .051074 | .092936 | .055314 | .060449 | .004468 |
| Frozen native M3 | 55.268370 | .085612 | .077244 | .106894 | .079817 | .078493 | .004711 |
| Frozen native M4 | 55.282990 | .085325 | .077314 | .105820 | .079818 | .078347 | .004673 |
| M3 win with hybrid payout | 56.259032 | .066495 | .055460 | .094103 | .055721 | .060698 | .004711 |
| M4 win with hybrid payout | 56.274094 | .066210 | .055531 | .093034 | .055722 | .060553 | .004673 |
| Archived S1 global | 56.418604 | .063479 | .050434 | .088530 | .055550 | .059402 | .004143 |
| Fixed drop | 55.864158 | .074034 | .059986 | .107089 | .060257 | .068803 | .005791 |
| Fixed forward | 55.927330 | .072821 | .059986 | .102237 | .060257 | .068803 | .005621 |
| Forward with constant donor weight | 56.267897 | .066327 | .055255 | .091288 | .057629 | .061138 | .004666 |
| Forward with support gate | 55.870309 | .073916 | .070984 | .091198 | .066561 | .066920 | .005724 |

Forward gained **−0.001213 ΔR** [−0.003376, +0.000307] over drop overall and **−0.004853** [−0.013402, +0.001208] on the 192 released short-history queries across 24 episodes. All differences above rounding came from that condition. Forward trailed the released champion by **+0.007877 ΔR** [+0.004457, +0.012030]. The constant blend improved forward by **−0.006493** [−0.011072, −0.002360] overall and by **−0.010949** [−0.023102, −0.001172] on the short-history slice. The gate degraded the constant by **+0.007588** [+0.004716, +0.011071] overall. Their short-history difference was **−0.000090** [−0.000281, +0.000124]. For the 248 frozen M3 top-pair (4,7) queries from 86 episodes, the gate degraded the constant by **+0.008359** [+0.002636, +0.015321]. The support-below-40 slice coincided with the 192 released short-history queries. [Paired hybrid scores](scores_vs_hybrid.json) · [Native M3 comparisons](scores_vs_m3.json) · [Matched scores](scores_vs_matched.json) · [Slices and payout class errors](slices.csv)

## Deterministic history stress

All masks retain the same 768 queries and cutoff signals. The 24+16 mask contains 40 observations per query and the nested 24+8 mask contains 32. SHA-256 rank chooses a common pool of 24 observed ticks before cutoff−16 by the tuple seed, query ID, cutoff and tick. The final 16 or 8 observed suffix ticks are kept. The frozen tier holds the released hybrid's conditional payout head. The recomputed tier runs the frozen Bayesian, tree and GRU checkpoints against masked prefixes and regenerates the conditional payout head. Both tiers recompute the donor win input.

| Mask tier | Drop points | Forward points | Same-mask hybrid points | Forward ΔR vs drop, paired 95% interval | Forward ΔR vs same-mask hybrid, paired 95% interval |
|---|---:|---:|---:|---|---|
| 24+16 frozen payout | 51.408770 | 52.226468 | 52.158483 | −.018273 [−.035439, −.003405] | −.001497 [−.020700, +.016849] |
| 24+16 recomputed | 50.760017 | 51.566969 | 51.490239 | −.018497 [−.035591, −.003787] | −.001734 [−.020808, +.016517] |
| 24+8 frozen payout | 39.669618 | 41.559047 | 40.003399 | −.068764 [−.097148, −.041760] | −.056144 [−.085391, −.027948] |
| 24+8 recomputed | 39.172168 | 41.013555 | 39.495888 | −.068769 [−.097139, −.041645] | −.056215 [−.085539, −.027745] |

The recomputed forward forecast gained **−0.047802** [−0.106608, −0.009197] and **−0.088679** [−0.171309, −0.015087] versus matched drop on the originally short-history queries under 24+16 and 24+8. Its gain versus the same-mask hybrid in that slice had intervals spanning zero for both masks. The gated blend beat the constant in both recomputed masks by **−0.006428** [−0.009341, −0.003718] and **−0.016140** [−0.023883, −0.008707] overall. Both blends trailed pure forward in points for each mask. Recomputing the payout cost forward 0.660 and 0.545 points at 24+16 and 24+8 respectively. The matched ΔR changed by 0.000224 and 0.000005. Against the unmasked champion, even the strongest severe-mask arms lost points. [Validation mask manifest](validation_mask_manifest.csv) · [Test mask manifest](test_mask_manifest.csv) · [Per-query support and gate weights](validation_support_diagnostics.csv)

## Known-law synthetic histories

Each scenario uses 24 eight-machine episodes and exact generated odds. Both arms use the same S2 one-flip grid and train type-level positive payout prior. This table reports raw five-class SSE.

| Scenario | Budget | Drop SSE | Forward SSE | Drop pair recovery | Forward pair recovery |
|---|---|---:|---:|---:|---:|
| Stable | Complete | .000670 | .000670 | 100.0% | 100.0% |
| Stable | 24+16 | .011022 | .010970 | 88.5% | 88.0% |
| Stable | 24+8 | .086469 | .076785 | 57.8% | 60.9% |
| Flip at age 16 | Complete | .000158 | .000158 | 100.0% | 100.0% |
| Flip at age 16 | 24+16 | .020056 | .017277 | 88.5% | 92.7% |
| Flip at age 16 | 24+8 | .089568 | .087031 | 64.1% | 68.8% |

The stable 24+8 paired SSE interval was [−0.018895, −0.000067]. The other sparse fixture intervals span zero. [Synthetic scores](synthetic_scores.json)

## Fitting, exposure, and verification

- Fixed drop and forward use the same train-only frozen M3 emissions by type and signal, 28-pair prior, two polarities, null arm and S2 flip ages. The native shift-register kernel sums over missing bits, with up to 256 states, and uses marginal emissions before the lag context can exist. On observed ticks it normalizes one time and accumulates that evidence. Drop substitutes the same marginal in incomplete contexts. Exact released drop and forward replay the archived S2 files byte for byte.
- The new gate trains on 9,216 sampled win outcomes at shifts 0, 8, 16 and 24 across all three history budgets. Each of four published training folds holds out 24 episodes and refits the Bayesian rates and priors on the remaining 72 before forecasting its rows. Neutral signal applies before the released cutoff. Each episode receives equal total weight. The regularized gate is sigmoid(0.283823 − 0.028099·(40 − effective support) + 0.046647·pair entropy). Effective support counts observed complete triples, averaged over the 28-pair posterior normalized without the null. Entropy is in nats. The constant weight is 0.541167.
- The tree and three factored GRU donors keep their original main pre-cutoff transductive history exposure. The GRUs also saw the main train cutoff labels. Their earlier observations can include training pseudo-cutoff outcomes, and their weights stay frozen across training folds. Gate-only held-fold Brier improved by .00145–.00166 in those diagnostics. The donor exposure and the non-nested emission refits make these training checks development evidence. The archived S1 global uses public validation odds for its own fold fits and appears here solely as a historical comparator.
- Audited fit and prediction completed before the format checker or official scoring read validation answers. The forecast freeze records exact hashes. Only the score stage read validation targets, condition and reference files within the audited stages. The S2 kernel matched 60 exhaustive path examples and its drop control on every complete history at the per-hypothesis likelihood and cutoff forecast levels. Twelve validation episodes passed query-order, padding, post-cutoff outcome poisoning and mask nesting checks. All 48 validation and test forecasts passed the official format checker. Released validation and test donor replays agreed with frozen exports within 6.7e−16. The two early smoke invocations exposed Python 3.9 annotation and tree-source version mismatches. The successful smoke and full runs used the compatible archived Fredy source and an annotation-only import shim.
- Fit, prediction, checks and fixture scoring took 187.56 seconds on one numerical thread. Released inference and output took 6.82 seconds for 768 validation queries and 14.09 seconds for 1,536 test queries. Each validation stress mask took about 13.4 seconds including all arms. The team reserved concurrent capacity for E5/E6. Two-core package timing was unnecessary once the main promotion gate failed.

[Forecast freeze](forecast_freeze.json) · [Training folds and rates](episode_fold_manifest.json) · [Gate coefficients and fold checks](gate.json) · [Train pseudo-cutoff rows](train_grouped_gate_rows.csv) · [Fit audit](fit_predict_audit.jsonl) · [Score audit](score_audit.jsonl) · [Format checks](format_checks.log) · [Run commands](commands.sh) · [Run manifest](manifest.json)

## Handoff

Keep the frozen main champion in assembly. A separate adaptation-development comparison can carry the exact forward/drop mechanism under adaptation-fitted rates and compare it with the matched adaptation incumbent on genuinely sparse histories. The severe main masks establish development evidence for the mechanism. The archived fixed-rate E4 implementation is independent of E3, whose four-cell arm did not advance.

[Released forward forecast](validation_released_frozen_payout_forward.csv) · [Strongest new released arm](validation_released_frozen_payout_constant.csv) · [Unscored constant test forecast](test_released_frozen_payout_constant.csv) · [24+8 recomputed forward forecast](validation_24plus8_recomputed_forward.csv)
