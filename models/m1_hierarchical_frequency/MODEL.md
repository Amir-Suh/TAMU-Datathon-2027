# Model 1: Hierarchical frequency model

**Validation: 31.09 / 60** (pooled reference: 30.00, kit `pooled_recent`: 30.83). Code: [model.py](model.py).

## What it does

It predicts the next payout from how often each payout has happened recently, without using the order of spins. It is the "no lag structure" model: a stronger, tuned version of the kit's `pooled_recent` baseline. It exists as a fallback and as the foundation that model 2 builds on.

The forecast is split into two independent parts:

```
P(0x)            = 1 - P(win)
P(1x, 2x, 5x, 20x) = P(win) * P(size | win)
```

Splitting this way helps because win rate and payout size behave differently. Every spin tells you about the win rate. Only winning spins tell you about size, so size needs more shrinkage. Our analysis also found that payout size depends on machine type but not on the lag pattern.

## How it works

1. **Context priors (from train).** For each of the 9 machine type × signal cells it estimates the win rate and the size distribution among wins, from every training spin. A cell is shrunk toward its machine type only if it is small.
2. **Per-machine win rate.** It keeps a recency-weighted Beta posterior for each machine:
   `P(win) = (strength · prior + recent wins) / (strength + recent spins)`.
   Counts decay with a half-life in ticks. Missing ticks add nothing, but decay still applies.
3. **Per-machine payout size.** It keeps a recency-weighted Dirichlet posterior over {1×, 2×, 5×, 20×} for each machine, centred on the context prior (the shared `SizeComponent` in [common/components.py](../common/components.py)).
4. It combines the two parts into five probabilities that sum to 1.

Everything runs as an online filter: the forecast for tick *t* uses only ticks before *t*, so it can never see the future.

## Hyperparameters and how they were chosen

Selection uses the train split only. Each machine's last 32 observed ticks plus its labelled outcome act as out-of-sample tests, and the setting with the lowest log-loss wins. Validation is used only for the final score.

| Part | Grid | Chosen |
|---|---|---|
| Win rate | strength {2, 5, 10, 20, 50} × half-life {∞, 128, 64, 32, 16} | strength 5, half-life 16 ticks |
| Size given win | strength {5, 15, 40, 100} × half-life {∞, 96, 48} | strength 100, half-life ∞ |

These choices say that a machine's win rate moves quickly, so a short memory is chosen. The size distribution is mostly a property of the machine type, so a heavy prior with no forgetting is chosen.

## Results (validation, loss ratio vs reference; lower is better)

| Stable | Short history | Drift | Shared change |
|---|---|---|---|
| 0.962 | 0.979 | 0.933 | 0.845 |

## Strengths and limits

- **Strengths:** It is fast (about 1 s) and simple. It is a good fallback when a machine shows no lag pattern.
- **Main limit:** It ignores the order of spins. In this simulator a machine's win depends on whether the wins at two specific earlier lags matched, so a frequency model cannot get close to the true odds. On many spins the true win probability is above 90% or below 10%, while this model predicts something near the machine's average. That is why it only beats the reference by about 1 point.

## Run it

```sh
python -m models.run --models m1
```
