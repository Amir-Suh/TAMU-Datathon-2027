# Z2 · Static and sticky joint payout states

The train-history selection chose 128 ticks for both sticky families. Its joint forecast scored **55.172876 / 60** against **56.341022** for the frozen hybrid. The selected sticky joint model's paired ΔR versus static joint was −0.001003 [−0.004428, +0.002652]. Its paired ΔR versus the same-dwell payout-only control was +0.004768 [−0.000809, +0.010585]. The static joint model also trailed its payout-only control by +0.005743 [+0.000180, +0.011183] in R. [Official scores and paired comparisons](scores.json)

| Forecast | Points / 60 | ΔR vs. frozen hybrid [paired 95% interval] | p0 MSE | Win-weighted payout SSE |
|---|---:|---:|---:|---:|
| Frozen hybrid | 56.341022 | | 0.004468 | 0.006448 |
| Static payout-only | 55.414442 | +0.017807 [+0.013383, +0.022637] | 0.004468 | 0.009958 |
| Static joint | 55.122059 | +0.023550 [+0.017056, +0.030143] | 0.004939 | 0.010428 |
| Sticky 128 payout-only | 55.415852 | +0.017779 [+0.013829, +0.021884] | 0.004468 | 0.009959 |
| Sticky 128 joint | 55.172876 | +0.022547 [+0.015812, +0.029581] | 0.004434 | 0.010901 |

The 16- and 64-tick runs are recorded alongside the selected model. Sticky 64 joint had the lowest p0 MSE of the tested arms, 0.004329. The selected sticky joint model raised R in every condition. Its short-history and inferred (4,7) contributions to ΔR were +0.004743 [+0.000912, +0.008413] and +0.004871 [+0.000752, +0.009348], respectively. The pair slice uses the prototype's own MAP lag assignment. The full evaluation denominators apply to these contributions. [All arms, condition ratios and slice results](scores.json)

## Known-law checks

The simulations fit and evaluate on independently seeded sets of 24 episodes each, with the generating lag and polarity supplied. The masked evaluation retains the last 40 ticks and 24 older observations. The static and sticky generators use a −0.76 high-versus-low state shift on the win logit. The joint fits recovered −0.802 under the static law and −0.799 under the 64-tick sticky law.

| Generating law | Forecast | Five-class SSE, full | Five-class SSE, 64 observed | State AUC, full |
|---|---|---:|---:|---:|
| Static | Static joint | 0.000029 | 0.002032 | 0.999 |
| Static | Static payout-only | 0.002292 | 0.002583 | 0.998 |
| Static | Sticky 64 joint | 0.003713 | 0.003860 | 0.929 |
| Sticky 64 | Static joint | 0.012666 | 0.010814 | 0.606 |
| Sticky 64 | Sticky 64 joint | 0.006985 | 0.007074 | 0.845 |
| Sticky 64 | Sticky 64 payout-only | 0.008237 | 0.008241 | 0.826 |

Under the independent-state generator, the persistent-state joint arms had 0.01144–0.01203 full-history cutoff SSE. The synthetic odds condition on the simulator's actual cutoff state, so uncertainty about that state remains in the score. Explicit path enumeration agreed with the forward likelihood to 1.39e−17 on a short sequence containing a missing tick. The analytic gradient matched finite differences within 1.24e−6. State label swap, missing tick transition, normalization and payout-only win invariance checks also passed. [Synthetic results](synthetic_scores.json)

## Fitting and prediction scope

The state likelihood uses three emissions per tick, 0×, 1×, and larger wins. Training fits two ordered conditional 1× emissions per type, a history win intercept per type, and one shared win-logit state contrast. The payout-only controls fix the contrast at zero. A symmetric transition advances through every tick, including missing observations. Training starts the ordered 1× emissions from the pooled type-specific history 1× rate among wins. The positive tail uses raw 2×/5×/20× counts from training histories.

Main history win logits use an M3 pair, direction and flip-time plug-in inferred from the entire available prefix, with a marginal type win rate when a required lag is missing or the null hypothesis wins. The training-history likelihood is conditional on these prefix-level plug-ins. Historical ticks have no recorded signal, so the M3 structure fit and history filter use zero signal. The frozen hybrid win probability serves as the state-free cutoff logit, and the history intercept applies only inside filtering. The forecast averages the joint payout distributions over the predicted state probabilities. [Fit settings and hashes](plan.json) · [Validation state posteriors](state_posteriors.csv)

Dwell selection uses the four published training episode folds. Each fold refits the M3 structure parameters and state emissions on its fitting histories, then compares negative log likelihood per observed tick averaged over its held episodes. The final fits use all main training histories. This is public main development against a frozen parent with transductive history exposure. Z2's fitting and dwell selection use no cutoff labels or validation odds. Changing cutoff and future payouts leaves the validation history design identical. The paired main intervals use 10,000 condition-stratified episode resamples with forecasts fixed.

## Replay

The 12-episode simulation fit and checks took under 0.1 seconds after imports. The full run took 27.86 seconds on this host with numerical libraries capped at two threads. A separate temporary-directory replay matched every generated file byte for byte, and the kit checker accepted all eight arm forecasts and the selected forecast alias.

```sh
d=signal_experiments/research_next/Z2
python3 -B "$d/run.py" --smoke
python3 -B "$d/run.py"
python3 -B check_submission.py "$d/validation_predictions.csv" --split validation
```
