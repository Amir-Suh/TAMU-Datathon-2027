# Findings: how the simulator behaves

These come from analysis of **train** data unless marked otherwise. The model designs are based on them.

## 1. Wins follow a per-machine lag-pair rule (models 2–6)

- Each machine's chance of a win depends on whether its wins at two earlier lags **matched** (both wins or both losses) or **differed**.
- Almost every machine uses one of the pairs **(1,3), (2,5) or (4,7)**.
- Half the machines use "match → win" and half "differ → win". Neither the pair nor the direction is related to machine type or signal.
- **Stability check:** the best pair in a stable machine's first half is also best in its second half 93% of the time, against 5% for shuffled histories.
- **Effect size:** the "win" cell pays out about 78–96% of the time, depending on type and signal, and the "lose" cell 1–21%.
- **Win means any payout above 0×.** Two wins of different sizes count as a match, just like two equal ones.
- A single lag on its own predicts almost nothing out of sample. Lag 7 only looked important because it is half of (4,7).

## 2. Drift and shared change = the rule flips direction (models 3–6)

Average match-minus-differ win rate per 32-tick window, signed by each machine's early direction:

| Window (ticks) | 0–128 | 128–160 | 160–192 | 192+ |
|---|---|---|---|---|
| Stable | +0.73 | +0.74 | +0.73 | +0.76 |
| Drift | +0.68 | +0.29 | −0.31 | −0.65 |
| Shared change | +0.60 | +0.19 | −0.37 | −0.52 |

- **The pair stays the same and the direction reverses,** for 89% of drift and 94% of shared-change machines.
- **The flip is mostly sudden:** a step fits best for 65% of drift and 89% of shared-change machines, and a 16-tick ramp for most of the rest.
- **Timing:** flips happen around tick 140–150 (sd ~35), so at the cutoff the flip is usually 36–216 ticks in the past (median ~70).
- **Shared-change machines flip only loosely together:** flip times spread by 23 ticks within an episode, against 35 overall. Modeling one shared flip time added almost nothing (model 4).

## 3. Payout size: no usable signal beyond type × signal (step 1, negative result)

Checks on train:
- Size given a win does **not** depend on the lag-pair cell, on earlier payout sizes (lag 1 or the pair lags), or on a per-machine lag-pair rule of its own. For that last check, first-half/second-half pair agreement was 3–4%, the same as shuffled data.
- **Per-machine size counts barely help:** held-out log-loss 1.0356 against 1.0396 for the type × signal prior alone. Pooling sizes across an episode makes it worse (1.05–1.07).

Diagnosis on validation, which looked at the true probabilities but tuned nothing on them:
- The true size mix **does** vary within a machine type (sd ≈ 0.10–0.15 on the 1× share).
- That variation is unrelated to the rule cell, the condition or the win rate, so it reflects hidden state the logs don't reveal.
- With a perfect win probability, about 38% of model 2's error would remain from size alone, and most of that looks irreducible.

**Decision:** keep the existing size component (per-machine counts shrunk strongly to type × signal, strength 100, no forgetting).

## 4. Is model 2 or 3 overfitting? (validation, after the fact)

- **Bootstrap:** model 2's 95% interval over validation episodes is 52.0–53.1 points.
- **Shuffled validation histories:** model 2 drops to 29.9, the baseline level, so it does not report confidence that isn't there.
- **Fit on shuffled train:** model 2 still scores 51.2 on validation, because the rules are discovered from each machine's own history at prediction time. Train only supplies priors.
