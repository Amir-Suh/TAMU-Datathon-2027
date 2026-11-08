# W1 · Four ordered lag cells

The pooled ordered-cell refinement (W1a, strength 20) improved the frozen hybrid by **0.039446 points** on released main validation and passed the grouped training gate for W2. Pair-specific shrinkage (W1b) did not improve on W1a in grouped training or released main validation. The machine-type arm (W1c) was skipped under its gate.

## Matched comparison

W0 is the published M3 split-match win head plus the frozen 31-leaf tree and factored GRU conditional payout donor. W1a keeps the W0 00/11 predictions and adds a pooled 10-minus-01 contrast. W1b adds regularized pair-specific deviations in both sibling groups. All arms hold the posterior over pair, direction, flip time and null fixed. Local contrasts count complete ordered triples at or after each hypothesis's flip and strictly before the cutoff. The M3 parent retains all its original pre-flip information.

Contrasts are fitted separately for match-high and differ-high directions by weighting rate differences by hypothesis posterior and effective paired counts. The pooled prior uses 200 effective counts at zero. W1b shrinks each pair toward the pooled prior using 200 effective counts. Local differ-cell shrinkage uses the chosen strength, plus the W0 history discount of 8. W1b's additional match adjustment uses the same strength around the inherited W0 contrast. The original W0 match formula remains in place when a matching child is absent. Every incremental W1 contrast is zero if either child has zero support. Centering under the smoothed local child fractions preserves each hypothesis's parent rate.

## Grouped train selection

Each published train fold held out 24 complete episodes and refitted M3's context, rate, pair, flip and kappa parameters and the contrast tables on the other 72. Each held episode contributed realized outcomes at the true cutoff and seven earlier cutoffs, with a separately truncated prefix each time. Earlier cutoffs use a neutral signal because the release provides a signal only at the actual cutoff. The real cutoff uses the recorded signal. The grouped metrics average first within episode. Released exact validation probabilities are read after candidate predictions and settings are saved.

| Win head | Episode-averaged Brier | Top-pair (4,7) Brier | Actual-cutoff Brier | Episode-averaged log loss |
|---|---:|---:|---:|---:|
| W0 | 0.089904702 | 0.090606376 | 0.076466362 | 0.304872757 |
| W1a, strength 8 | 0.089836743 | 0.090649967 | 0.076911215 | 0.304853197 |
| W1a, strength 20 | **0.089816396** | 0.090570469 | 0.076650137 | 0.304622732 |
| W1a, strength 50 | 0.089836207 | 0.090560247 | 0.076520495 | 0.304627069 |
| W1b, strength 20 | 0.089835397 | 0.090614004 | 0.076701946 | 0.305010887 |

Selection minimizes W1a's overall Brier and carries one shared strength into W1b. W1a's grouped improvement is small and comes from historical cutoffs. Actual-cutoff Brier worsened while actual-cutoff log loss improved from 0.267096 to 0.266735. The pooled 10-minus-01 priors from full training are −0.00313 for match-high and −0.01364 for differ-high.

## Released main validation

| Model | Points / 60 | R | Paired ΔR 95% interval | p0 MSE |
|---|---:|---:|---:|---:|
| W0 frozen hybrid | 56.341022 | 0.064943404 | | 0.004468097 |
| W1a pooled | **56.380468** | **0.064198334** | [−0.0014343, −0.0001238] | **0.004360195** |
| W1b pair-specific | 56.379653 | 0.064213708 | [−0.0014809, −0.0000291] | 0.004365575 |

W1b's increment over W1a was +0.0000154 R with a paired interval [−0.0002232, +0.0003062]. W1a lowered every condition's loss, with the largest improvement in stable (−0.0021582 R). In the inferred (4,7) slice (248 rows), W1a reduced p0 MSE from 0.0064525 to 0.0062098 and slice R by 0.0011045. In short history (192 rows), it reduced p0 MSE from 0.0067912 to 0.0066912 and R by 0.0006315. Both slice intervals span zero. W1a's top-cell 01 slice (170 rows) had a paired ΔR of −0.0020634 with interval [−0.0043120, −0.0001005]. All intervals use 10,000 paired condition-stratified episode resamples.

This is reused development data. The gain is smaller than the plan's 0.002 fresh-lockbox threshold. W1a is the matched W2 research arm, while W0 remains the promotion control. The 1,536-row test files for W1a and W1b passed format checks and have private scores.

## Mechanism and replay

The separate generated fit and held-out episodes tested stationary parity, shared ordered rates, and pair-specific ordered rates. W1a reduced ordered-case p0 MSE from 0.01970 (W0) to 0.00588. In the pair-specific law, W1b reduced p0 MSE from W1a's 0.01220 to 0.00670. The fitted parity differ-cell contrasts shrank below 0.003 in absolute value, with a residual MSE cost from noisy local corrections (W0 0.00483, W1a 0.00503). These fixed-strength generated checks use the default M3 kappa of 8.

W0 reproduced the saved hybrid within 3.33 × 10⁻¹⁶ on validation and test. Candidate conditional payout mixes agreed with the donor within 2.22 × 10⁻¹⁶. The runner also checks 1,536 child counts against explicit enumeration, exact parent fallback, valid probability ranges, row-order and padding invariance, and invariance after perturbing held outcomes at and after earlier cutoffs. The full run takes under one minute with two numerical threads on this host. The local Git branch has an unborn HEAD, so the provenance records file hashes and a null commit.

Run from the participant-kit root using the dependencies in `m5_m6_repair/requirements-repaired-models.txt`.

```sh
d=signal_experiments/research_next/W1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python3 -B "$d/run.py" --test
python3 -B check_submission.py "$d/W1a_validation_predictions.csv" --split validation
python3 -B check_submission.py "$d/W1a_test_predictions.csv"
python3 -B check_submission.py "$d/W1b_validation_predictions.csv" --split validation
python3 -B check_submission.py "$d/W1b_test_predictions.csv"
```

The runner also accepts `--forecast-only --out <fresh-directory>` for replay without opening exact released targets.

[Run manifest](plan.json) · [Grouped selection](grouped_selection.json) · [Fitted contrasts](train_contrasts.json) · [Validation scores](scores.json) · [Synthetic scores](synthetic_scores.json) · [Causality and replay checks](checks.json) · [Cell support by query](validation_cell_diagnostics.csv)
