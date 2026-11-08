# Model 5: gradient-boosted corrector

**Validation: 54.99 / 60** (its base, model 4: 55.28). **Negative result: the corrector makes things slightly worse.** Code: [model.py](model.py).

## Idea

Boosting can learn where a strong base model is systematically off. A scikit-learn `HistGradientBoostingClassifier` predicts win/loss for the next spin from:

- model 4's P(win), as a logit, and its independent-only P(win);
- posterior summaries: top pair confidence, P(null), P(flip), expected ticks since the flip, P(shared flip);
- context: observed ticks (total and in the last 64), machine type, signal.

P(size | win) comes from model 4.

## Training (train only)

- **Rows:** train pseudo-queries, with every machine re-cut at 8 earlier cutoffs and the observed payout there as the label (about 6,100 rows).
- **Rounds:** chosen from {25, 50, 100, 200} by 4-fold cross-validation **grouped by episode**. 100 was chosen.
- **GBM settings:** learning rate 0.05, 15 leaves, at least 50 samples per leaf, L2 regularization 1.0.
- **Out-of-fold predictions** are kept so that model 6 can weigh this model fairly.

## Result

| | Pseudo-query log-loss (train) | Validation points |
|---|---|---|
| Model 4 (base) | 0.3034 | 55.28 |
| Model 5, out of fold | 0.3082 | 54.99 |

The corrector was already worse than its base on held-out train folds, and validation agrees. **Why:**

- **The base is well calibrated:** it is a full Bayesian posterior, so there is little systematic error left to learn.
- **The labels are noisy:** they are single sampled outcomes, not true probabilities. About 6,000 noisy labels are not enough to reliably correct an error that is already small.

Kept for transparency and as an ensemble member. It is not recommended on its own.

## Run it

```sh
python -m models.run --models m4 m5
```
