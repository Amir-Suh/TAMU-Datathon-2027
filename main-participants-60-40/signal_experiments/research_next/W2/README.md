# W2 · Ordered lag-cell evidence

W2 changed M3's hypothesis weights while holding W1a's per-hypothesis win forecasts fixed. It added **0.001061 points** over W1a on released main validation. The paired interval includes zero, no top pair changed on main, and the known-law gains depended on the case. W1a remains the simpler ordered-cell research arm.

## Evidence update

Under the M3 parent model with fixed child exposures, an allocation of the parent group's post-flip wins between its ordered cells has a central hypergeometric distribution. W2 tilts that allocation by a training-only log-odds contrast. The contrast comes from the frozen pooled W1a table by centering its child rates around each M3 context and polarity rate. For a right-cell win count `j`, the tilted count weight is proportional to

```text
choose(n_left, total_wins - j) * choose(n_right, j) * exp(log_odds * j)
```

The implementation sums over the exact feasible support and adds the tilted-to-central log probability ratio for the observed allocation to M3's evidence. It applies this separately for 00/11 and 01/10 to complete post-flip triples. M3 retains the group totals, pre-flip history, unknown-context model, null, 28 lag pairs, direction support, flip grid and priors. W1a's local child prediction formula and the hybrid conditional payout donor remain fixed.

The tilt is an exact conditional distribution for fixed exposure and win totals. In actual histories the exposure counts depend on earlier outcomes. The conditional construction does not supply a new joint chronological law over every possible adaptive history. The separate known-law comparison measures behavior in that setting.

## Held episodes

The four published train folds hold out 24 episodes each and refit upstream M3 win parameters and W1's pooled contrasts on the other 72. The control reproduced all 6,144 W1a grouped forecasts exactly. Selection stayed at W1a's strength 20. Earlier pseudo-cutoffs use a neutral signal and strictly truncated histories. Actual cutoffs use the recorded signal.

| Win head | Overall Brier | W1a top-pair (4,7) Brier | Actual-cutoff Brier | Overall log loss |
|---|---:|---:|---:|---:|
| W1a | 0.089816396 | 0.090570469 | 0.076650137 | 0.304622732 |
| W2 | 0.089787274 | 0.090533872 | 0.076506064 | 0.304562190 |

## Released main validation

| Model | Points / 60 | R | p0 MSE |
|---|---:|---:|---:|
| Frozen hybrid | 56.341022 | 0.064943404 | 0.004468097 |
| W1a | 56.380468 | 0.064198334 | 0.004360195 |
| W2 | 56.381529 | 0.064178300 | 0.004357344 |

W2 versus W1a had ΔR −0.00002003 with a paired 95% interval of [−0.00004593, +0.00000233]. The (4,7) and short-history slice ΔR values were −0.00002834 and −0.00005348, with both intervals spanning zero. W2 versus the frozen hybrid had ΔR −0.00076510 with interval [−0.00145383, −0.00014364]. Each interval uses 10,000 paired, condition-stratified episode resamples. Main validation is reused development data, and the increment is below the proposed fresh-lockbox threshold.

The top pair stayed the same for all 768 queries. Mean pair entropy including the null moved from 0.059915 to 0.059818 nats overall and from 0.210239 to 0.209880 in short history. The mean absolute win change from W1a was 0.0000659.

## Known-law and replay checks

The shared generator used 64 fitting episodes per law and 24 held episodes per case, with separate seeds. It varied stationary parity versus ordered rates, stationary versus 8-, 16- and 40-tick-old flips, and nested history budgets of 0, 24 or all older observations plus the last 40. Both arms used the same exact conditional payout mix so the comparison isolates win inference.

In the ordered law with a flip 16 ticks before cutoff and 24 older observations, W2 lowered p0 MSE by 0.00173 and increased top-pair recovery by one query out of 192. Correct localization within four ticks rose from 93.75% to 95.31%. It raised p0 MSE by 0.00754 in the ordered stationary case with full history and selected a false flip for 3.125% of those rows, up from zero. Pair recovery changed in both directions across cases, and the parity case showed little change overall. The synthetic table includes direction and flip localization as well as pair recovery and exact probability errors.

Explicitly enumerating fixed-design outcome assignments matched 69 conditional distributions with maximum error 8.89 × 10⁻¹⁶. A zero contrast reproduced the M3 posterior exactly. Earlier-cutoff predictions were invariant to outcome perturbations at and after the cutoff, as well as row order and missing padding. W1a replay and conditional payout donor agreement were each within 2.22 × 10⁻¹⁶. The 768-row validation and 1,536-row test forecasts pass the kit format checker and replay byte-for-byte in a fresh folder. The full two-thread run, including fold refits and known-law checks, took about 79 seconds. Test scores are private.

Run from the participant-kit root using the dependencies in `m5_m6_repair/requirements-repaired-models.txt`.

```sh
d=signal_experiments/research_next/W2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python3 -B "$d/run.py" --test
python3 -B check_submission.py "$d/validation_predictions.csv" --split validation
python3 -B check_submission.py "$d/test_predictions.csv"
```

Use `--forecast-only --out <fresh-directory>` to rerun without reading released exact targets.

[Run manifest](plan.json) · [Grouped results](grouped_metrics.json) · [Validation scores](scores.json) · [Posterior diagnostics](posterior_diagnostics.json) · [Known-law scores](synthetic_scores.json) · [Causality and replay checks](checks.json)
