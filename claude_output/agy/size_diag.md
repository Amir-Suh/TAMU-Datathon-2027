# Antigravity job: what predicts payout size given a win? (diagnostic table, new files only)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root).

## Rules
- Create ONLY `tools_size_diag.py` inside `/tmp/agy_sizediag/` (create the folder) and write all outputs there. Do not edit anything in the repo. No git, no network, no pip, no rm. Run scripts with `PYTHONPATH=src $PY ...`.
- Read once: `src/tdx/data.py` (load_round, Split fields), `src/tdx/arms/gbdt/features.py` (`features(y, mtype)` returns `x [E,8,T+1,F], names, groups`; column row index t describes the state BEFORE tick t, so it is causal), `src/tdx/feat.py`.

## Question
Given that a machine wins at tick t (class code y[e,m,t] in {1,2,3,4}, meaning payout 1x, 2x, 5x, 20x), which causal history features predict WHICH size it is? Class 0 is a loss and is excluded. Known so far: machine type matters (1x share 0.61 / 0.37 / 0.24 for types 0/1/2); the signal column is not available at tick level, do not use it.

## Data
`rnd = load_round('data/main', answers=False)`. Use the three splits train, validation, test together; each has `y [E,8,256]` with -1 for unobserved ticks and `mtype [E,8]`. Only ticks `< cutoff` are observed (already -1 beyond). Build rows (episode e, machine m, tick t) for every observed win tick with t >= 8. Label = y[e,m,t] - 1 (0..3). Group id = (split name, episode index) for cross-validation: use 5-fold GroupKFold by episode (sklearn is installed).

## Feature groups (all causal; compute from `features()` columns by name or group, plus the extras below)
- G0 `type`: the column `machine_type` only (this is the baseline prior).
- G1 `last_sizes`: columns `lag1`..`lag8` (class of the own previous 8 ticks; -1 missing).
- G2 `rates`: columns whose group is `ema` or `rate`.
- G3 `pair`: columns whose group is `pair`.
- G4 `neighbour`: columns whose group is `neighbour`.
- G5 `gaps` (write these yourself, vectorised with numpy, no per-row Python loops over rows): ticks since the machine's last win of any size; ticks since its last win of size class 3 or 4 (5x or 20x); size class of its latest win and of the win before it (-1 if none); number of wins in the last 10 and last 40 calendar ticks; ticks since its last 20x.
- G6 `position`: the column `position` (tick/256) and `n_obs_before`.

## Experiment
For feature set S in: G0; G0+G1; G0+G2; G0+G3; G0+G4; G0+G5; G0+G6; and ALL groups: train `lightgbm.LGBMClassifier(objective='multiclass', n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=7, n_jobs=6, verbose=-1)` with 5-fold GroupKFold, and report out-of-fold multiclass log-loss on the win rows (use `sklearn.metrics.log_loss`, labels=[0,1,2,3]). Also report the log-loss of the empirical class frequencies by machine type alone (no model) and the number of win rows. Finally, for the ALL model fit on all rows, print the top 15 features by gain importance.

## Final report (print only this)
A table: feature set, out-of-fold log-loss, change vs G0 (negative = better); the type-only frequency baseline; the number of rows; the top-15 importance list; any step that failed (last 5 log lines). Do not interpret beyond one sentence.
