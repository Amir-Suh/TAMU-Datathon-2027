# Independent review of the M5/M6 repair

Reviewed the staged source after the feature and crossfit changes, before the full comparison run. The archived source and prior adversarial findings remain in `adversarial_review/branches/amir/` and `adversarial_review/branches/amir_review.md`.

## Decision

**No blocking leakage or alignment defect found in the repaired pipeline.** The repaired M5 and M6 can be trained and compared with the unchanged M3 using the frozen runner. This is a source and targeted-test review; the separate comparison report contains the measured results.

## Checks

- M3's only implementation change is in `features()`: `n_recent` now counts each row's observed ticks in `[cutoff-64, cutoff)`, and `n_obs` excludes at/after-cutoff storage. Its fit and direct prediction paths are unchanged. The feature tests compare a scalar tick-by-tick oracle, batch padding, row order, and fixed pre-repair prediction values.
- M5 round-selection folds keep every machine and pseudo cutoff of an episode together. For each held selection fold, its M4 feature fit excludes that fold. Its classifier training design is crossfit again inside the selection training partition, so no M4 model that constructs those rows has seen the selection fold. The final classifier learns from crossfit features; its deployment base fits the full training partition.
- M6 creates every M2/M3/M4/M5 meta column by fitting the member on the other episodes. M5 is fully refit, including round selection, inside each meta partition. Meta rows use explicit `(episode, query, pseudo shift)` keys and are assigned back into one aligned matrix. Supplied prefit members cannot enter these held-episode rows.
- `evaluate_heldout()` constructs a fresh ensemble for each outer partition. Member fitting, M5 round choice, and M6 weights all occur inside that partition. Its returned forecasts for the outer episodes therefore assess the complete procedure. Final-model supplied prefit bases/members are reused only when a training-data fingerprint matches the exact fit partition.
- The fit cache hashes all model training fields, including labels and histories. Mutating a cutoff label or a history value while retaining the same episode and query IDs creates a new cache entry.
- The runner writes a plan and input/source hashes before fitting. It generates validation and test forecasts from query/history inputs only, checks submission IDs and probabilities, reloads the saved checkpoint, checks the unchanged M3 forecast against the archived CSV, and freezes forecast hashes before loading validation targets for scoring. Its paired `delta_R` routine reproduces the archived M6-minus-M3 value `-0.0005538681185313333` from the old CSVs.

Ten targeted tests pass with the project interpreter (`PYTHONPATH=m5_m6_repair /Users/andrewvong/ml_env/bin/python -m unittest discover -s m5_m6_repair/tests -p 'test_*.py' -v`). Two independent metamorphic tests in `tests/test_review_invariance.py` use label-sensitive fake base and classifier models: changing an episode's cutoff labels leaves that episode's M5 heldout forecasts exactly equal, and the cache invalidates same-ID data mutations. The other tests check nested feature provenance, whole-pipeline outer-fold label invariance, row alignment, serialization, and recent-history behavior. These lightweight tests do not substitute for the full real-model run.

## Interpretation and follow-up

- The public validation set has already been used for model development. Its M3/M5/M6 comparison is a fixed, matched **development** comparison, not an untouched test. The unchanged M3 reference is `55.268370 / 60` on all 96 validation episodes.
- The two outer train folds fit on about half the training episodes. Their pseudo-query binary log-loss assesses held episodes for that smaller fitted pipeline; it is not a direct estimate of the full-train contest score. M5's minimum round-selection loss and M6's fitted blend loss are selection/training diagnostics, not independent performance estimates.
- The staged `models/README.md` still displays the **archived** M5 `54.99` and M6 `55.30` validation scores as if current. Replace those rows with repaired results or label them archived after the frozen run. The repaired M5/M6 `MODEL.md` files already distinguish the old scores from current results.
