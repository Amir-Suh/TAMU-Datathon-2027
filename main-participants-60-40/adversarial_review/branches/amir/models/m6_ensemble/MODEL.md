# Model 6: ensemble

**Validation: 55.30 / 60** (best single model, model 4: 55.28; model 3: 55.27). Code: [model.py](model.py).

## Idea

Blend the win probabilities of models 2–5: `P(win) = Σ wᵢ · Pᵢ(win)`, with non-negative weights that sum to 1. P(size | win) comes from model 4.

## Choosing the weights (train only)

The weights minimise log-loss on train pseudo-queries: every machine re-cut at 8 earlier cutoffs, about 6,100 labelled spins. Model 5 contributes its **out-of-fold** predictions, so it isn't favoured for having been trained on those rows. Validation is not used.

| Member | Pseudo-query log-loss | Weight |
|---|---|---|
| m2 lag-pair (online) | 0.3659 | 0.00 |
| m3 flip-aware | 0.3033 | **0.83** |
| m4 flip + shared | 0.3034 | 0.00 |
| m5 boosted corrector | 0.3082 | 0.17 |
| **Blend** | **0.3031** | |

## Result

| | Stable | Short history | Drift | Shared change | Points |
|---|---|---|---|---|---|
| Model 3 | 0.077 | 0.107 | 0.080 | 0.078 | 55.27 |
| Model 6 | 0.077 | 0.106 | 0.080 | 0.078 | 55.30 |

**A gain of 0.03 points, which is noise.** Ensembling helps when members make *different* mistakes:

- **Models 3 and 4 are nearly the same model,** so blending them adds nothing.
- **Model 2 is uniformly worse,** so it gets no weight.
- **Model 5 is built on model 4.**

A useful ensemble would need a genuinely different strong model. A sequence neural network trained on pseudo-queries is the obvious candidate.

## Run it

```sh
python -m models.run --models m2 m3 m4 m5 m6
```
