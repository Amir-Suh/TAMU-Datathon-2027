# Models

House Edge forecasting models from `amir_workspace` at `c339b469de33bbc2b2c68ea739eeec835139d42b`, with repaired M5/M6 features and episode cross-validation. Existing local checkpoint folders are preserved alongside this source.

**Recommended: M3.** Repaired M6 adds only 0.014 validation points and does not improve held-episode training diagnostics. M5 remains worse. See the [repair report](../results/repaired-m5-m6/README.md) for uncertainty and provenance.

| Model | Approach | Validation points / 60 | Evidence |
|---|---|---:|---|
| [M1](m1_hierarchical_frequency/MODEL.md) | Hierarchical frequency | 31.092 | Archived branch run |
| [M2](m2_lag_pair/MODEL.md) | Lag-pair rule mixture | 52.598 | Archived branch run |
| [M3](m3_lag_pair_flip/MODEL.md) | Explicit flip-time lag pairs | 55.268370 | Fresh fit; unchanged comparator |
| [M4](m4_lag_pair_flip_shared/MODEL.md) | Shared episode flip time | 55.282990 | Fresh diagnostic fit |
| [M5](m5_boosted_corrector/MODEL.md) | Boosted win classifier on M4 features | 54.841763 | Repaired and retrained |
| [M6](m6_ensemble/MODEL.md) | Episode-crossfit blend of M2–M5 | 55.282043 | Repaired and retrained |

These are full-public-validation development scores, not hidden-test scores. Archived M5=54.987969 and M6=55.296582 scores describe the pre-repair implementation only.

## Training and inference

Each model implements `fit(train)`, `predict(split)` returning five payout probabilities, and `describe()`. Data loading rejects at/after-cutoff history and keeps missing ticks distinct from zero payouts. M4/M5/M6 use episode peers, so prediction batching must keep complete episodes together.

`common/crossfit.py` provides episode folds, aligned pseudo-query rows, data fingerprints, and a session-local fit cache. M5 constructs held-episode M4 features for both classifier fitting and nested tree-count selection. M6 creates held-episode forecasts for every member before fitting blend weights. `Ensemble.evaluate_heldout()` refits the entire pipeline inside additional outer training folds.

The original M1–M4 internal training-selection losses are development diagnostics; they can reuse priors learned from their supplied training partition. M5's minimum tree-selection loss and M6's fitted weight loss are also not independent assessments. The separately held-episode outer results assess the complete M5/M6 fitting procedure.

The simulator analysis is in [FINDINGS.md](FINDINGS.md). The fixed repair retained the original model families and hyperparameter grids.

## Run the repaired comparison

From the repository root:

```sh
python -m pip install -r requirements-repaired-models.txt
python -B run_repaired_comparison.py prepare --out results/repaired-fresh --outer-folds 2 \
  --previous main-participants-60-40/adversarial_review/branches/amir/results/2026-11-07_m1_to_m6
python -B run_repaired_comparison.py score --out results/repaired-fresh
python -B verify_repaired_inference.py --out results/repaired-fresh
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Use a fresh output directory. `prepare` freezes all training decisions and predictions before `score` reads validation answers. The saved run includes fitted checkpoints and validated test CSVs; submission packaging is separate. See [M5_M6_REPAIR.md](../M5_M6_REPAIR.md) for the exact workflow and files.
