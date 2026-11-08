# E3 ordered lag-cell experiment

**Decision.** Reject E3Frozen, E3Full and E3Shared for main promotion. Keep W1a as the simpler ordered-cell control. The incremental W1b and W2 intervals against W1a include zero. No four-cell rates handoff to E4.

## Reused main validation

All arms use the exact hybrid conditional payout mix. Intervals use 10,000 paired condition-stratified episode resamples. This is development evidence on 768 queries from 96 episodes. The shared payout donor inherits the transductive 31-leaf and factored GRU fits across the released histories.

| Arm | Points | R | ΔR vs hybrid | Paired 95% interval | p0 MSE |
|---|---:|---:|---:|---|---:|
| Hybrid | 56.341022 | 0.064943404 | 0 | | 0.004468097 |
| W1a | 56.380468 | 0.064198334 | −0.000745070 | [−0.001434263, −0.000123790] | 0.004360195 |
| W1b | 56.379653 | 0.064213708 | −0.000729696 | [−0.001480910, −0.000029099] | 0.004365575 |
| W2 | 56.381529 | 0.064178300 | −0.000765104 | [−0.001453834, −0.000143642] | 0.004357344 |
| E3Frozen | 56.337675 | 0.065006677 | +0.000063273 | [−0.001615234, +0.001774084] | 0.004481001 |
| E3Full | 56.325706 | 0.065232990 | +0.000289586 | [−0.001470053, +0.002095712] | 0.004509491 |
| E3Shared | 56.337082 | 0.065017887 | +0.000074483 | [−0.001680987, +0.001885331] | 0.004480753 |

E3Full changed only hypothesis weights relative to E3Frozen, with ΔR +0.000226313 and interval [−0.000045903, +0.000569602]. Its ΔR versus W1a was +0.001034656 with interval [−0.000760343, +0.002894085]. E3Shared recovered 0.000215103 R versus E3Full with interval [−0.000485960, −0.000026209]. Its ratio still worsened against the hybrid in drift, shared-change, and short-history episodes. The pair entropy including the null changed from 0.0610745 to 0.0612216 nats relative to the strict M3 parent, with one MAP pair change.

The slice masks come from the archived M3 control replay. They are fixed across arms. The per-query E3 diagnostics separately show the strict E3 parent and the new posterior.

| Frozen slice | Queries | Episodes | Hybrid p0 MSE | W1a p0 MSE | E3Frozen p0 MSE | E3Full p0 MSE | E3Shared p0 MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top pair (4,7) | 248 | 86 | 0.00645253 | 0.00620984 | 0.00632726 | 0.00637001 | 0.00633101 |
| Pair mass ≥0.99 | 717 | 96 | 0.00403321 | 0.00391767 | 0.00407093 | 0.00407335 | 0.00406664 |

E3Full changed R by −0.000164905 versus the hybrid in (4,7), compared with W1a’s −0.001104452. The ≥0.99 slice had +0.000151028 versus the hybrid. See the score JSONs for condition ratios, payout metrics and control-specific intervals. The slice CSV also includes ≥0.90, short-history and ordered-cell slices and all five class MSE contributions.

## Fit, held episodes and generated histories

The primary arm used a pre-flip XOR on the t-a bit. Fit posterior-weighted pooled logit offsets on the ordered cells around the M3 type/signal × parity means, with a 200-count centered regularizer. The pair deviations used the card’s initial 150-count regularizer. Both directions are fit independently. M3 kappa supplies the prior strength S0. Each of four grouped folds refitted the M3 parent and ordered offsets on 72 episodes and predicted eight cutoffs from each of the 24 held episodes. The actual cutoff has its released signal. Earlier pseudo-cutoffs have neutral signal. The final strict win fit used all 96 training episodes and selected kappa 50. E3Frozen and E3Full have identical local posterior means, M3 unavailable-context likelihood and M3 cutoff fallback. E3Shared uses the same E3Full likelihood with fixed prior 0.3 for M4 episode sharing.

On 6,144 held training cutoffs, episode-averaged win Brier was 0.090166490 for strict M3, 0.090197410 for E3Frozen, 0.090265966 for E3Full, and 0.090299887 for E3Shared. The t-b XOR was prespecified as training-only and scored 0.089732917. Archived W1a’s grouped score was 0.089816396. Actual-cutoff Brier was 0.076650137 for archived W1a and 0.076382124 for E3Full. These grouped scores isolate the win model.

The separate synthetic run fit and held out independently seeded histories. In the ordered flip fixture with 24 older observations, E3Full recovered 92 of 96 pairs and localized all 96 flips within four ticks. M3 recovered 93 pairs and also localized all flips. The shared generator’s ordered law adds raw-bit offsets in both segments. The exact XOR law was checked separately with the chronological oracle.

## Reproduction and checks

- Coordinator-audited forecast-only replay reproduced W1a, W1b and W2 exactly on validation and test. The reconstructed W0 hybrid agreed within 3.33 × 10⁻¹⁶ and the replay’s conditional payout mix within 2.22 × 10⁻¹⁶. W1 zero contrast reproduced its parent, W2 zero tilt reproduced the M3 posterior, and W2’s 69 fixed-exposure conditional enumerations agreed to 8.89 × 10⁻¹⁶.
- New forecasts, weights and per-hypothesis predictions were exactly invariant to changing observations at and after a pseudo-cutoff. Additional missing padding preserved predictions. XOR mapping conserved all trial and win counts. Both XOR variants matched the chronological oracle for all 256 eight-tick paths over 12 hypotheses. Maximum joint likelihood error was 3.61 × 10⁻¹⁶, maximum normalization error was 1.11 × 10⁻¹⁵, and maximum posterior mean error was 1.11 × 10⁻¹⁶.
- The fit/predict and score processes are separate. The fit freeze preceded opening validation histories. The forecast freeze preceded opening targets, reference and conditions for scoring. The score runner verified all new forecast hashes before reading outcomes. No validation labels or test outcomes were read in the audited runs.
- All six validation and test CSV pairs passed the kit format checker. The audited forecast-only replay took 13.95 seconds. The grouped fits, invariance tests and new forecast generation took 54.19 seconds on one numerical thread, plus the separate synthetic work within the same audited process.

The [manifest](manifest.json) contains exact commands, inputs, trace paths and hashes. [Fold trace](episode_fold_trace.json) lists all fit and held episode IDs. [Validation diagnostics](validation_cell_diagnostics.csv) list raw pre/post and canonical support for each audited hypothesis, plus per-query forecasts and posterior summaries. [Hypothesis grid](validation_hypothesis_grid.npz) contains the full pair/polarity/flip weights. [Train rates](train_rates.npz) is a diagnostic export from the rejected fit. [Slices](slices.csv) and [matched scores](scores_vs_matched.json) contain the comparisons. Run [commands.sh](commands.sh) from the kit root to reproduce the phases with fresh traces.
