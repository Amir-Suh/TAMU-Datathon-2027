# M5/M6 repair experiment

The repaired model code is based on `amir_workspace` at `c339b469de33bbc2b2c68ea739eeec835139d42b`. The archived branch snapshot remains unchanged. The runnable source, tests, and completed results are installed at the repository root. A staging copy remains in `main-participants-60-40/m5_m6_repair/`.

The recent-history feature now counts observed ticks in each query's own `[cutoff - 64, cutoff)` window. Missing ticks remain missing, and counts exclude ticks at or after the cutoff. M3's fitting and direct prediction formulas are unchanged.

M5 selects its tree count using episode folds, with another episode crossfit inside each fold to construct classifier training features. Every M4 feature model is fitted without the episode it predicts. M6 fits its weights from held-episode forecasts of all four members; the entire M5 fitting and selection procedure runs inside each M6 training partition. An additional outer split refits the complete ensemble, including its weights, to assess that procedure on excluded episodes.

The repair preserves the original model family, feature list, tree settings, candidate tree counts, and blend optimizer. It does not establish that those choices are optimal. The experiment uses two outer assessment folds and the original four internal model-selection folds. Each outer model sees half of the training episodes, so its binary outcome diagnostics do not directly estimate the final model's contest score.

## Reproduce

From the repository root, in an environment with the pinned dependencies:

```sh
python -m pip install -r requirements-repaired-models.txt
python -B run_repaired_comparison.py prepare --out results/repaired-fresh --outer-folds 2 \
  --previous main-participants-60-40/adversarial_review/branches/amir/results/2026-11-07_m1_to_m6
python -B run_repaired_comparison.py score --out results/repaired-fresh
python -B verify_repaired_inference.py --out results/repaired-fresh
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

For this staged directory, the `--previous` path is `../adversarial_review/branches/amir/results/2026-11-07_m1_to_m6`. The recorded run used Python 3.9.6 and two numerical-library threads. Use a fresh output directory for a new run; existing experiments are never overwritten.

`prepare` records code, input, and scorer hashes before fitting. It uses training data for all fitting decisions, reads only query/history files for validation and test inference, saves and replays the fitted checkpoint, verifies the unchanged M3 comparator, and freezes predictions. `score` verifies the frozen artifacts before reading validation answers. The public validation set was already used in earlier model development; this is a fixed development comparison, not a fresh audit.

## Artifacts

- `results/repaired-m5-m6/comparison.md` and `comparison.json`: official validation scores and paired episode-bootstrap comparisons.
- `train_outer_report.json` and `train_outer_predictions.npz`: predictions and binary outcome metrics from the outer training folds.
- `fitted_configurations.json`: selected tree count, blend weights, and selection diagnostics. M5's minimum selection loss and M6's weight-fitting loss are not independent performance estimates.
- `models.joblib`: fitted M3, M4, M5, and M6 models.
- `<model>/validation_predictions.csv` and `<model>/submission.csv`: frozen validation and test forecasts.
- `plan.json`, `freeze.json`, and `inference_verification.json`: provenance, checkpoint replay, and prediction-batching checks.
- `results/repaired-m5-m6/REPAIR_REVIEW.md`: independent source review, conducted before the full comparison.

Test targets are private. Test CSV checks establish format and coverage, not test performance. Upload packaging remains a separate task.
