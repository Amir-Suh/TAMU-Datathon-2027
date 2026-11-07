# Models

These are forecasting models for the House Edge challenge. Each model lives in its own folder with its code and a `MODEL.md` that explains what it does and how well it scores.

| Model | Idea | Validation points /60 |
|---|---|---|
| kit `pooled` (reference) | Frequency counts shrunk to type × signal | 30.00 |
| kit `pooled_recent` | Same, with recency weighting | 30.83 |
| [m1_hierarchical_frequency](m1_hierarchical_frequency/MODEL.md) | Separate win-rate and size parts, tuned shrinkage and recency, no lags | 31.09 |
| [m2_lag_pair](m2_lag_pair/MODEL.md) | Bayesian averaging over "lag pair (a,b), match/differ → win" rules per machine | 52.60 |
| [m3_lag_pair_flip](m3_lag_pair_flip/MODEL.md) | Model 2 plus an explicit flip time: before the flip the rule is reversed | **55.27** |
| [m4_lag_pair_flip_shared](m4_lag_pair_flip_shared/MODEL.md) | Model 3 plus "one flip time for the whole episode" | 55.28 (no real gain) |
| [m5_boosted_corrector](m5_boosted_corrector/MODEL.md) | Gradient boosting on model 4's outputs | 54.99 (worse) |
| [m6_ensemble](m6_ensemble/MODEL.md) | Weighted blend of models 2–5, weights chosen on train | 55.30 (no real gain) |

**Recommended: model 3.** It is the simplest of the top group. Models 4 and 6 are within noise of it (about ±0.5 points).

How the simulator behaves, and why the models look like this: [FINDINGS.md](FINDINGS.md). Latest results: [results/](../results/README.md).

## Layout

```
models/
  common/
    data.py         load a kit split into numpy arrays (history matrix X[m, t])
    components.py   context priors, online Dirichlet filter, payout-size component
    selection.py    train-only hyperparameter selection (prequential and pseudo-query log-loss)
    evaluate.py     kit scorer wrapper, test-file check, extra diagnostics
  m1_hierarchical_frequency/   model.py + MODEL.md
  m2_lag_pair/                 model.py + MODEL.md
  m3_lag_pair_flip/            model.py + MODEL.md
  m4_lag_pair_flip_shared/     model.py (subclass of m3) + MODEL.md
  m5_boosted_corrector/        model.py + MODEL.md
  m6_ensemble/                 model.py + MODEL.md
  FINDINGS.md       what the data analysis found (rules, flips, payout size)
  run.py            fit -> predict -> score -> write a results folder
```

## Model contract

Every model is a class with:

- `fit(train: Split) -> self`: learns everything from the train split only.
- `predict(split: Split) -> np.ndarray[n, 5]`: probabilities for 0×, 1×, 2×, 5× and 20× at each query's cutoff.
- `describe() -> dict`: fitted priors and the hyperparameter selection table, saved as `params.json`.

## Rules every model follows

- **No future data:** a forecast for tick *t* reads only ticks before *t*. `load_split` rejects any history at or after the cutoff, and every model runs as an online filter.
- **Missing ticks are unknown,** never treated as a 0× payout.
- **Validation is only for scoring.** Hyperparameters, boosting rounds and ensemble weights are chosen on train:
  - **Models 1–2:** prequential log-loss over each machine's last 32 observed ticks plus its label.
  - **Models 3–6:** pseudo-queries, with each train machine re-cut at 8 earlier cutoffs.
- **Official scoring:** validation points come from the kit's own `contest_score`, so they match `check_submission.py --split validation`.

## Running

From the repository root:

```sh
pip install -r requirements.txt
python -m models.run                     # kit baselines + all models -> results/<date>_m1_m2_m3_m4_m5_m6/
python -m models.run --models m2 --out results/m2-experiment
```

The runner refuses to overwrite an existing results folder, so earlier runs are kept.

## Adding a model

1. Create `models/mN_<name>/` with `model.py`, `__init__.py` and `MODEL.md`.
2. Implement the contract above, reusing `common/` where possible.
3. Register it in `MODELS` in `run.py` and run it.

## Not yet done

- **Upload packaging:** the code ZIP with `predict.py --data --out` and `reproduce.json` is not built yet. The runner writes test CSVs that already pass the kit's format check.
- **Test predictions use priors from train only.** Refitting on train + validation for the final upload is a possible small gain.
