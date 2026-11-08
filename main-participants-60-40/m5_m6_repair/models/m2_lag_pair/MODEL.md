# Model 2: Bayesian lag-pair model

**Validation: 52.60 / 60** (pooled reference: 30.00; model 1: 31.09). Code: [model.py](model.py).

## The pattern it exploits

In the training data, each machine follows a hidden rule built from **two earlier spins at lags a and b**:

- **"match → win":** a win is likely when spins *t−a* and *t−b* were both wins or both losses.
- **"differ → win":** a win is likely when exactly one of them was a win.

On stable training machines the "win" cell pays out about 81–94% of the time (by type) and the "lose" cell 1–17%. The same lag pair is selected in a machine's first and second half 93% of the time on stable machines, against 5% for shuffled data. Almost all machines use the pairs (1,3), (2,5) or (4,7). The pair and direction are unrelated to machine type and signal, so they can only be found from each machine's own history. Type and signal set how strong the effect is.

This is also why lag 7 looked important in the exploratory charts: it is half of the (4,7) pair. A single lag on its own predicts almost nothing out of sample.

## What the model does

It does not hard-code any pair. For every machine it considers a set of hypotheses:

| Hypothesis family | Count |
|---|---|
| Lag pair (a, b), 1 ≤ a < b ≤ 8 | 28 |
| × direction (match→win / differ→win) | 2 |
| × count memory (half-life ∞, 64, 24 ticks) | 3 |
| + "no pattern" null (one win rate), per memory | 3 |

For each hypothesis it keeps recency-weighted win counts for the two cells ("lags match" and "lags differ"). It turns these into a Beta posterior predictive whose prior mean is the learned "win cell" rate (*hi*) or "lose cell" rate (*lo*) for that machine's type × signal. Hypotheses are weighted by their **prequential evidence**: how well each one predicted that machine's past spins, one spin at a time. The forecast is the posterior-weighted average of P(win). P(size | win) comes from the shared `SizeComponent`, the same as model 1.

Two kinds of forgetting cover the changing conditions:

- **Count memory (half-lives ∞ / 64 / 24):** short-memory hypotheses track a rule whose strength is drifting. Hypotheses with different memories compete, so each machine effectively picks its own memory.
- **Evidence forgetting (half-life 64 ticks):** old evidence fades, so if a relationship reverses (drift) or changes across the episode (shared change), the posterior can move to the new rule instead of staying locked on the old one.

Missing ticks, as in short histories, are handled honestly. When a lag needed for a pair is unobserved, that hypothesis falls back to the machine's overall win rate for that spin. Every hypothesis is then scored on the same spins.

## Learning the priors (train only)

A short EM loop over the training machines (two iterations):

1. Run the filter with flat priors and read each machine's posterior over pairs at its cutoff.
2. **Pair prior:** the average posterior over pairs, mixed 90/10 with uniform so that unseen pairs remain possible. The learned prior puts about 32% on (1,3), 29% on (2,5), 29% on (4,7) and about 0.4% on each other pair.
3. ***hi* / *lo* per type × signal:** pooled from machines whose most probable pair has ≥ 90% posterior (90% of train machines). Their "win" cell is whichever cell won more often.

| Type | *hi* (signal −1 / 0 / +1) | *lo* (signal −1 / 0 / +1) |
|---|---|---|
| steady | 0.77 / 0.80 / 0.85 | 0.29 / 0.31 / 0.38 |
| balanced | 0.59 / 0.67 / 0.68 | 0.18 / 0.19 / 0.23 |
| jackpot | 0.35 / 0.37 / 0.50 | 0.08 / 0.09 / 0.14 |

These are less extreme than the stable-machine rates because drift and shared-change machines are included. They are only priors, so a machine's own data quickly overrides them.

## Hyperparameters and how they were chosen

These are chosen on train only, by log-loss on each machine's last 32 observed ticks plus its labelled outcome:

| Parameter | Grid | Chosen |
|---|---|---|
| `kappa`: prior strength of the cell rates | {3, 8, 20} | 3 |
| `evidence_half_life` | {32, 64, 128} | 64 |

These are fixed by design: maximum lag 8 (the rules state interactions reach 8 ticks), count half-lives {∞, 64, 24}, null weight 0.1, and pair-prior smoothing 0.1.

## Results (validation, loss ratio vs reference; lower is better)

| Stable | Short history | Drift | Shared change |
|---|---|---|---|
| **0.082** | **0.119** | **0.211** | **0.151** |

The error is roughly 7–12× lower than the reference on every condition. The rule it identifies for each validation machine is saved in `results/<run>/m2_lag_pair/validation_rules.csv`. It is confident on most machines: the mean pair posterior is 0.98 on stable machines and 0.89 on short-history machines.

## Strengths and limits

- **Strengths:**
  - It captures the main mechanism and is fully probabilistic, so it is calibrated by construction.
  - It degrades gracefully: with little data it falls back to the type × signal prior.
  - It learns which pairs exist, so it can be refit for round two.
- **Limit, drift and shared change:** These are now the weakest conditions, with ratios of 0.21 and 0.15. The model only reacts to change through forgetting. It does not model a gradual reversal directly, and it does not share change information across the machines in an episode.
- **Limit, rule shape:** It assumes the rule is "match vs differ" on two lags. If round two used another shape, such as "both won", this hypothesis family would need extending. The diagnostic in the run folder (`validation_rules.csv`) and the pair prior in `params.json` are the first things to check on new data.
- **Limit, payout size:** Size given a win is modeled without lags. The training analysis supports this, but it has not been verified against the round-two data.
- **Limit, runtime:** It takes about 1 minute for fit plus predict on a laptop CPU, well within the 30-minute rerun budget.

## Run it

```sh
python -m models.run --models m2
```
