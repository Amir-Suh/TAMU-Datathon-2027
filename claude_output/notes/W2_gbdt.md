# W2 gbdt report

Status: P1 done; P2 done (8 pilots + 2 ablations, no timeouts); P3 done; P4 done.

Selected pilot: `runs/gbdt/trans_64_l15_no_neighbour`, 36.395373 points versus pooled_recent 30.825854; paired 95% points-gain interval [4.760, 6.374]. Command:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/tdx/bin/python -m tdx.run --arm gbdt --variant trans_64_l15 --set drop_neighbour=true --out runs/gbdt/trans_64_l15_no_neighbour
```

## Proof by package

- P1: `~/.venvs/tdx/bin/python -m pytest -q tests/gbdt` → `8 passed in 3.18s`. Independent scalar reference covers every row of three episodes; future perturbation, final weights/counts, cross-process final determinism, probability constraints and save/load pass.
- P1 full-data repeat: `PYTHONPATH=src ~/.venvs/tdx/bin/python runs/gbdt/verify.py trans_64_l15` → 539424 rows, 400 rounds, validation/test repeat and save/load maximum difference 0.0. Selected no-neighbour saved-model parity is also exactly 0.0 on both splits.
- P2: `~/.venvs/tdx/bin/python -m tdx.sweep --arm gbdt` → eight outputs in `runs/gbdt/pilot_output.txt`, copied below. Ablations use the selected raw-grid variant with `--set drop_neighbour=true` and `--set drop_divergence=true`, distinct `--out` directories.
- P2 comparison: `~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent` → selected 36.395, delta +5.570, 95% interval [+4.760, +6.374]. Full table: `runs/gbdt/compare.txt`; anchor comparison: `runs/gbdt/compare_anchor.txt`.
- P3: `~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm gbdt` and `--variant trans_64_l15` → each PASS, 5 episodes, cutoff reduced by 4, max abs difference 0.0, forbidden dev opens 0.
- Format: `~/.venvs/tdx/bin/python kit/check_submission.py runs/gbdt/trans_64_l15_no_neighbour/test_pred.csv --data data/main` → VALID FORMAT: 1536 queries | main | test; SHA256 8c0dc3f88fceb97f923f2e12537bc521cd6346c7f084de0f5835507139a65175.
- P4: this report plus `claude_output/notes/W2_pilot.md`, `W2_review.md`, `W2_acceptance.txt`; all measurements retained in the run directories.

## Pilot results

Numbers below are the six-decimal points/ratios and three-decimal seconds written by the harness to runs/LOG.md. The last two rows retain the LOG variant name trans_64_l15; directory suffixes identify their override.

| Variant / output directory | Points | Stable | Short | Drift | Shared | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| all_inf_l15 | 35.123478 | 0.660494 | 0.789811 | 0.749401 | 0.633330 | 26.948 |
| all_64_l15 | 34.361767 | 0.689012 | 0.837107 | 0.787863 | 0.670525 | 25.567 |
| all_24_l15 | 33.505709 | 0.779938 | 0.855951 | 0.842091 | 0.684979 | 15.961 |
| last64_24_l15 | 33.087621 | 0.798631 | 0.866397 | 0.869726 | 0.718713 | 5.359 |
| last128_64_l15 | 34.253452 | 0.723100 | 0.840617 | 0.782389 | 0.660487 | 16.501 |
| all_64_l7 | 33.525941 | 0.763289 | 0.862991 | 0.839460 | 0.692897 | 22.012 |
| trans_64_l15 | 36.225856 | 0.620367 | 0.734134 | 0.683614 | 0.586987 | 99.106 |
| trans_24_l7 | 34.180138 | 0.737179 | 0.832522 | 0.789933 | 0.661988 | 94.561 |
| trans_64_l15_no_neighbour | 36.395373 | 0.603969 | 0.725280 | 0.682277 | 0.582719 | 96.282 |
| trans_64_l15_no_divergence | 36.191118 | 0.613067 | 0.737972 | 0.696892 | 0.583531 | 97.902 |

## Feature importance and behaviour

Selected 37-feature model, LightGBM gain importance:

| Feature | Gain |
|---|---:|
| ema4_c0 | 207375.518 |
| machine_type | 169549.668 |
| ema12_c0 | 54850.108 |
| wins_obs5 | 40245.196 |
| lag7 | 32336.350 |
| ema64_c1 | 30524.409 |
| wins_obs20 | 28149.239 |
| lag4 | 22017.133 |
| lag6 | 20786.099 |
| lag8 | 20340.920 |
| lag3 | 18989.361 |
| lag5 | 18111.277 |
| ema64_c2 | 17144.801 |
| lag2 | 16839.329 |
| code12 | 12731.224 |

All four selected-model condition ratios improve on pooled_recent. Short history remains weakest (0.725280); shared change is strongest (0.582719). Transductive histories improve on train-only pilots; aggressive recency and shorter training windows hurt this grid. Removing neighbours is a simpler, slightly higher-scoring choice: its gain over the base model is +0.170, 95% interval [-0.013, +0.355], a tie. Removing divergence scores 36.191118 (delta -0.035, interval [-0.213, +0.153]).

## Surprises and open risks

- Three review defects fixed: hash-dependent final row order, sparse neighbour snapshots, and missing divergence treated as zero-win rates; re-review found no blockers.
- W2 final-only cutoff supervision takes precedence over the broader dev permission. Signal is unused, as specified for tick features.
- The selected/base transductive fits reach the 400-round search cap; no larger-round refinement was run.
- Pilot selection and ablations reuse public validation; reported uncertainty does not adjust for model selection.
- Adaptation data was unavailable; round/data paths are parameterized but adaptation is untested.
- W6 must merge `src/tdx/arms/gbdt/requirements.txt` (lightgbm==4.7.0) into package requirements.
- No entry ZIP, fresh-folder entry reproduction or upload was performed; these belong to final packaging/upload.
- Initial new schema-size test assertion was corrected 47→46 to match the specified feature count; the independent scalar oracle was retained.

## Delegation

| Package | Model | Effort | Why |
|---|---|---|---|
| P1 implementation and integration | Parent runtime, inherited | Inherited | Implement causal arm and resolve review findings |
| P1 independent regressions | gpt-6.1-sol | medium | Scalar/edge cases, final soft weights and cross-process regression verification |
| P2 driving, P4 pilot notes | gpt-6-luna | low | Mechanical sweep, ablations and durable result summaries |
| Read-only review | gpt-6-astra | high | Different model checked causality and independently reproduced hash-seed failure/fix |

Subagent models were chosen using explicit `collaboration.spawn_agent` model/reasoning_effort overrides with fork_turns="none", following the user package table and tool-advertised model identifiers. Parent runtime was not switched.

## Changed paths

Source: `src/tdx/arms/gbdt/__init__.py`, `src/tdx/arms/gbdt/arm.py`, `src/tdx/arms/gbdt/features.py`, `src/tdx/arms/gbdt/README.md`, `src/tdx/arms/gbdt/requirements.txt`.

Tests: `tests/gbdt/test_gbdt.py`. Notes: `claude_output/notes/W2_gbdt.md`, `claude_output/notes/W2_report.md`, `claude_output/notes/W2_pilot.md`, `claude_output/notes/W2_review.md`, `claude_output/notes/W2_acceptance.txt`.

Artifacts: exact file inventory `claude_output/notes/W2_changed_paths.txt` covers all files under runs/gbdt. Shared `runs/LOG.md` was appended only by the harness. No commits were made.
