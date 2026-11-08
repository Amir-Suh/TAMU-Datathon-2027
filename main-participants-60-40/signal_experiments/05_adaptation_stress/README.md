# Adaptation stress tests

The adaptation kit opens at 9 PM Central on November 7. The current results use public main data and a separate generated experiment. The adaptation command uses the new round's training data, validation odds, and reference once they are available.

## Main validation proxy

The transfer comparison refits M3 on the same 24 target-proxy train episodes selected by a fixed episode hash. Both arms use those fitted settings and differ only in the prior over all 28 lag pairs. One uses the learned prior from the frozen full-main checkpoint and the other uses the prior estimated from the 24 episodes. The carry prior has access to more main training data, including the proxy subset. This isolates the impact of a smaller same-round prior estimate.

| Pair prior | Points | Loss ratio R | p0 MSE |
|---|---:|---:|---:|
| Carry frozen full-main prior | 55.26080 | 0.085761 | 0.004739 |
| Refit on 24 main train episodes | 55.28414 | 0.085302 | 0.004682 |

The refit-minus-carry ΔR was −0.000458 with a paired 95% interval of [−0.001176, +0.000042]. Short histories accounted for nearly all of that difference. The learned and carried priors placed 90.2% and 90.5% of their mass on the same three pairs. In a separate uniform-prior posterior diagnostic, 707 of 768 validation machines had at least 0.90 mass on a single pair. Of these, 216 selected (1, 3), 259 selected (2, 5), and 232 selected (4, 7). These are prefix-inferred counts. [Prior transfer](results/main_proxy/prior_transfer.json) · [All 28 pairs and prefix support](results/main_proxy/pair_prevalence.csv)

The M3–M4 comparison uses their frozen main checkpoints and all 96 validation episodes. A deterministic mask preserves each actual cutoff and 24 observed ticks earlier than the last 40 ticks, plus the most recent 16 or 8 observed contiguous ticks in the original prefix. Masks and forecasts use query IDs, tick indices and pre-cutoff histories. Each condition keeps its original simulator odds and frozen reference. The normal replay reproduces the published forecasts to floating-point precision.

| Input | M3 points | M4 points | ΔR, M4 minus M3 | Paired 95% interval |
|---|---:|---:|---:|---:|
| Released histories | 55.26837 | 55.28299 | −0.000287 | [−0.000571, −0.000072] |
| 24 older + 16 recent | 50.03294 | 50.08748 | −0.001306 | [−0.002987, +0.000312] |
| 24 older + 8 recent | 38.26344 | 38.45602 | −0.007853 | [−0.013850, −0.002133] |

M4's advantage reached 0.193 points under the strongest mask, where both models lost nearly 17 points relative to their released-history scores. The shared-flip mixture improved win forecasts modestly, with the same conditional payout error as M3 in each regime. [Normal](results/main_proxy/m3_m4_normal.json) · [16 recent](results/main_proxy/m3_m4_sparse16.json) · [8 recent](results/main_proxy/m3_m4_sparse8.json)

## Generated rare-pair shift

A generated train set and a disjoint generated evaluation set each have 24 episodes. Every machine follows a fixed binary match or mismatch rule for one of the four pairs with the smallest main prior mass. The generator supplies exact cutoff odds. The target refit assigned 91.4% total mass to those four pairs, compared with 1.5% in the main prior. Both forecasts otherwise use identical settings fit on generated training data.

| Generated evaluation input | Carry main SSE | Refit prior SSE | Paired ΔSSE with 95% interval |
|---|---:|---:|---:|
| Full prefixes | 0.000605 | 0.000605 | Within floating-point precision |
| 24 older + 16 recent | 0.212595 | 0.056175 | −0.156419 [−0.192853, −0.120499] |
| 24 older + 8 recent | 0.284113 | 0.205446 | −0.078667 [−0.108571, −0.048816] |

Full prefixes identified the generated rule with either prior. Under sparse histories the same deliberate shift made the prior consequential. The generated mechanism and its SSE use no contest reference. [Generated experiment](results/main_proxy/synthetic_rare_pair_shift.json)

## Reproduce

Run from the participant kit. The output directory must be fresh. The runner caps numerical thread pools at two.

```bash
python -B signal_experiments/05_adaptation_stress/stress.py \
  --data public --fit-episodes 24 --synthetic \
  --out signal_experiments/05_adaptation_stress/results/replay
```

For the actual adaptation kit, use its public directory and all released training episodes. The transfer comparison replaces only the pair prior of target-fitted M3. The separate normal and masked comparisons refit both M3 and M4 on the same adaptation train episodes. The path was smoke-tested with main data and temporary adaptation metadata.

```bash
python -B signal_experiments/05_adaptation_stress/stress.py \
  --data /path/to/adaptation/public --max-points 40 \
  --out signal_experiments/05_adaptation_stress/results/adaptation
```
