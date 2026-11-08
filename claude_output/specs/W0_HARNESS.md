# W0 — Harness (blocking; everything else depends on it)

## Role and objective
Implement the shared package `src/tdx/` exactly as in COMMON_CONTRACT section 1, plus the pooled baselines as the first two "arms", so every later stream can run, score and compare variants with one command.

## Why
Five arms must be scored identically against the kit scorer and compared with paired uncertainty. Read first: `AGENTS.md`, `claude_output/02_BAKEOFF_PLAN.md` sections 2 and 4, `kit/slot_forecasting/{io,schema,contest,experiments,baseline}.py`.

## Decisions (do not reopen)
- Payout class order `(0,1,2,5,20)`; machine type codes steady 0, balanced 1, jackpot 2; `T=256`, 8 machines per episode, machine order = sorted `machine_id`.
- Scoring uses `slot_forecasting.contest.contest_score` with `max_points` from `metadata.json` and the validation `reference.csv`; episode-level values come from `slot_forecasting.scoring.score_predictions(...)["episode_errors"]`.
- Cache: `.cache/<round>_<metadata-sha>.npz`; `load_round` rebuilds when the metadata hash differs. Data root comes from `--data` or `TDX_DATA_ROOT` (default `data/main`). Adaptation data in `data/adaptation`.
- Two-half tuning/audit scores reuse `slot_forecasting.experiments.episode_folds`.
- Bootstrap in `compare`: 2000 resamples of whole validation episodes within each condition (stratified), seed 20261107; the statistic is the difference in points vs the baseline run; print 95% percentile interval. PROPOSED default reading rule, shown but not enforced.

## Preconditions
- Hard: `data/main/{train,validation,test}/*.csv` exist; `~/.venvs/tdx/bin/python -c "import numpy, scipy, lightgbm, torch"` works.
- Soft: if `torch` import fails, report it and continue (W0 does not need torch).

## Owned files
Create: `src/tdx/{__init__,paths,data,fit,arm,feat,score,run,sweep,log}.py`, `src/tdx/arms/__init__.py`, `src/tdx/arms/pooled/{__init__,arm}.py`, `src/tdx/tools/{__init__,compare,leak_test}.py`, `tests/test_data.py`, `tests/test_score.py`, `tests/test_leak.py`, `tests/test_feat.py`, `runs/LOG.md` header.
Do not touch: `kit/`, `AGENTS.md`, other `src/tdx/arms/*`.

## Work packages
- **P1 data**: `load_round`, `Split`, `Round`, cache. Test: shapes for main (96/96/192 episodes), `y` is −1 for all `t >= cutoff`, `n_obs==64` count equals 192 train machines, class codes round-trip with the CSVs (sample 1000 rows).
- **P2 feat**: `lag_classes`, `others_win`, `tick_rows`. Test: brute-force Python reference on 3 episodes matches exactly; lag at `t` never reads `y[..., t:]` (assert by corrupting `y[..., t:]` with 4 and checking outputs).
- **P3 score/run/sweep/log**: `score` wraps the kit scorer; `run` builds the arm, fits per mode, predicts validation (dev) and test, writes the files in the contract, appends `runs/LOG.md`; `sweep` loads data once and runs the arm's `VARIANTS`, forwarding `--device` and `--smoke` to `build` when its signature accepts them. `final` mode scores nothing on validation. Test: scoring a `reference.csv`-equal prediction returns points 30.000 and four ratios 1.0.
- **P4 pooled arms**: `pooled` arm with variants `pooled`, `pooled_recent` (and `recent`, `context`) reimplemented vectorised in numpy to match `FrequencyForecaster` exactly, with `transductive` ignored. Acceptance: `python -m tdx.run --arm pooled --variant pooled_recent --mode dev` prints points 30.826 (±0.001) and `pooled` 30.000; `test_pred.csv` passes `python kit/check_submission.py <csv>`.
- **P5 tools**: `compare` (table sorted by points, columns: arm, variant, points, four ratios, seconds, Δpoints vs baseline with interval) and `leak_test` (perturb history at ticks ≥ cutoff−4 for 5 episodes, require identical predictions, also require dev mode never opens validation `labels/targets/conditions`: implement by running the fit under a file-open audit hook listing those paths).
- **P6 notes**: write `claude_output/notes/W0_harness.md` (≤ 40 lines): API as built, any deviation from the contract, timings of `load_round` cold/warm.

## Acceptance (runnable)
```
~/.venvs/tdx/bin/python -m pytest -q tests
~/.venvs/tdx/bin/python -m tdx.run --arm pooled --variant pooled_recent --mode dev
~/.venvs/tdx/bin/python -m tdx.run --arm pooled --variant pooled --mode dev
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm pooled --variant pooled_recent
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/
python kit/check_submission.py runs/pooled/pooled_recent/test_pred.csv
python kit/check_submission.py runs/pooled/pooled_recent/val_pred.csv --split validation
```
Time box: 45 minutes. `load_round` warm < 3 s.

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1–P3 implementation | gpt-6.1-sol | medium | well-specified modules with tests |
| P4 equality to the kit | gpt-6.1-sol | medium | numeric match check is the proof |
| P5 tools + leak test | gpt-6.1-sol | high | audit-hook logic is subtle |
| Independent review of the diff | gpt-6-astra | high | different model, read-only |
