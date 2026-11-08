# Model 3: flip-aware Bayesian lag-pair model

**Validation: 55.27 / 60** (model 2: 52.60; reference: 30.00). Code: [model.py](model.py). Background: [FINDINGS.md](../FINDINGS.md) §1–2.

## What changed from model 2

In drift and shared-change episodes a machine's rule flips direction partway through its history, and the lag pair stays the same. Model 2 could only react by forgetting old data. Model 3 models the flip explicitly. Ticks **before** the flip follow the opposite rule, so the model swaps their match/differ cells and keeps them as evidence for the current rule. The whole history is used instead of being forgotten.

## How it works

For each machine it scores every hypothesis below with an exact Beta-binomial marginal likelihood. Only the machine's own ticks before the cutoff are used.

| Part | Options |
|---|---|
| Lag pair (a, b) | all 28 pairs with 1 ≤ a < b ≤ 8 |
| Current direction | "match → win" or "differ → win" |
| Flip time τ | no flip, or any tick on a 4-tick grid leaving ≥ 8 ticks after it |
| Null | "no pattern": one win rate for every tick |

Under a hypothesis, each observed tick falls into one of three groups:
- the **high-win** group (the current winning cell after τ, the other cell before τ);
- the **low-win** group;
- an **unknown-context** group, for ticks where a needed lag was unobserved.

Each group has a Beta prior. The high group is centred on the learned *hi* rate for the machine's type × signal, the low group on *lo*, and the unknown group on the base win rate. Every hypothesis explains exactly the same ticks, so their evidence is directly comparable. The forecast is the posterior-weighted average of each hypothesis's predicted P(win) at the cutoff. P(size | win) is the shared size component.

There is no online loop over ticks: all flip times are scored at once with cumulative sums, so the model is exact rather than approximate.

## Priors learned from train (EM, 2 iterations)

- **Pair prior:** (1,3) 33%, (2,5) 29%, (4,7) 28%, about 0.4% for each other pair (mixed 90/10 with uniform).
- **Flip probability and flip-time prior:** learned, then smoothed 50/50 with uniform.
  - The learned flip probability (0.62) is inflated by "flips" near the start of a history. These are almost indistinguishable from no flip and do not affect forecasts.
- ***hi* / *lo* per type × signal:** taken from the most probable hypothesis of confident machines (96% of train). They are much sharper than model 2's because flips no longer blur them.

| Type | *hi* (signal −1 / 0 / +1) | *lo* (signal −1 / 0 / +1) |
|---|---|---|
| steady | 0.93 / 0.94 / 0.96 | 0.12 / 0.15 / 0.21 |
| balanced | 0.78 / 0.84 / 0.87 | 0.04 / 0.05 / 0.07 |
| jackpot | 0.50 / 0.59 / 0.65 | 0.01 / 0.02 / 0.02 |

## Hyperparameters (train only)

**κ**, the strength of the Beta priors, is chosen by log-loss on train pseudo-queries: each machine re-cut at 8 earlier cutoffs (0, 4, …, 28 ticks back), using the observed payout there as the label.
- **Grid:** {8, 20, 50, 120}. **Chosen:** 120.
- The train log-loss is nearly flat above 20. A κ of 20 gives 55.25 on validation, about the same.
- A large κ means each machine's cell rates stay close to the learned *hi* / *lo*.

These are fixed by design: maximum lag 8, flip grid step 4, minimum 8 ticks after a flip, null weight 0.05.

## Results (validation loss ratio vs reference; lower is better)

| | Stable | Short history | Drift | Shared change |
|---|---|---|---|---|
| Model 2 | 0.082 | 0.119 | 0.211 | 0.151 |
| **Model 3** | **0.077** | **0.107** | **0.080** | **0.078** |

Drift and shared change are now as good as stable. Short history is the weakest condition, since 64 observed ticks give less evidence.

## Limits

- **Single flip:** it assumes at most one flip. A machine that flips twice would be modeled poorly.
- **Rule shape:** it assumes the rule is match/differ on win/loss. A refinement not yet modeled: in "match → win" machines, two earlier losses win less often (~74%) than two earlier wins (~86%).
- **Runtime:** about 100 s for fit + predict on a laptop CPU. It needs `scipy`.

## Run it

```sh
python -m models.run --models m3
```
