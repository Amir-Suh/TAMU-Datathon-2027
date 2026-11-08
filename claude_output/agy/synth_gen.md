# Antigravity job: synthetic generator from the teammate's rule (UNDERSTANDING ONLY, never training data; 40-minute budget)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; `PYTHONPATH=src`). No GPU needed.

## Purpose
Test whether our understanding of the simulator is complete. If the Bayesian model `m3` (which encodes the rule) scores about as well on synthetic data built from that rule as it does on the real validation set (55.3 points), our picture is complete; if it scores far higher, real data has extra mechanisms (e.g. in payout size) we have not captured. The synthetic data must NEVER be used to train anything that ships.

## Rules
- Create ONLY: `src/tdx/tools/synth_gen.py`, `tests/test_synth_gen.py`, and files under `/tmp/agy_synth/`. Do NOT edit existing files. No git, no network, no pip, no rm. Write the synthetic datasets only under `/tmp/agy_synth/`.
- Read once: `git show origin/amir_workspace:models/FINDINGS.md` is not available to you (no git); instead read `src/tdx/arms/m3flip/amir/m3_lag_pair_flip/model.py` (the rule encoded), `src/tdx/arms/m3flip/arm.py`, `src/tdx/data.py` (`_read_split` and the kit file layout under `/home/ganfwe/Downloads/main-participants-60-40/public/{train,validation,test}/`: `queries.csv`, `history.csv`, `labels.csv`, `conditions.csv`, `targets.csv`, `reference.csv`; inspect the first 3 lines of each to copy the exact columns), and `kit/docs/DATA.md`.

## Generator (calibrate to the facts below; keep every constant as a named parameter)
- Episodes: 24 per condition (stable, short_history, drift, shared_change) per split for train and validation (96 episodes each) and 48 per condition for test (192), 8 machines per episode (mix of machine types steady/balanced/jackpot and signals -1/0/1 like the real queries.csv: sample types and signals from the real marginal frequencies), one cutoff per episode in 192..256, 256 ticks. short_history: only the last 40 ticks before the cutoff plus 24 random older ticks are written to history.csv.
- Win rule (per machine): choose a lag pair uniformly from (1,3), (2,5), (4,7) and a direction (match->win or differ->win) with probability 1/2. At tick t, "match" = (win at t-a) == (win at t-b), win meaning payout > 0; if either earlier tick is before tick 0, use a loss. P(win) is `hi` in the winning cell and `lo` in the other cell, with hi ~ Beta around 0.85 and lo ~ Beta around 0.08 per machine-type x signal cell (use hi 0.78-0.96 and lo 0.01-0.21 spread by type/signal; make it an explicit table).
- Flips: for drift and shared_change machines the direction reverses once at a tick drawn from Normal(145, 35) clipped to [100, 230]; for shared_change use the SAME flip tick for all 8 machines of an episode (plus Normal(0,1.2) jitter); for drift use a flip tick = episode mean + Normal(0, 14) per machine. Stable machines never flip.
- Payout size given a win: a per-machine hidden 4-vector (1x,2x,5x,20x) drawn once per machine from a Dirichlet centred on the type mean (1x share 0.61 steady, 0.37 balanced, 0.24 jackpot; split the rest 2x/5x/20x with plausible fixed ratios per type) with std about 0.12 on the 1x share; sizes are drawn iid given a win (no dependence on history).
- Write kit-format folders `/tmp/agy_synth/data_seedN/{train,validation,test}` (plus `metadata.json` copied from the real one with the same season id and round), `labels.csv` = the realized outcome at the cutoff (sample one draw from the true probabilities), `targets.csv` = TRUE probability vectors at the cutoff (validation only), `reference.csv` = the pooled baseline forecast (copy the logic from the kit: fit the kit's `slot_forecasting.baseline.FrequencyForecaster` pooled on the synthetic train and predict validation), `conditions.csv`.

## Experiment
For two seeds (N = 1, 2): generate the data; then run `$PY -m tdx.run --arm m3flip --variant m3 --mode dev --data /tmp/agy_synth/data_seedN --out /tmp/agy_synth/runs_seedN/m3` and likewise `--arm pooled --variant pooled_recent`; if `tdx.run` cannot take a custom `--data` for validation scoring, score with the kit scorer on the synthetic `targets.csv` and `reference.csv` yourself (same formula: per-condition mean of squared error ratio, points = 60/(1+R)). Also, if time allows, one GRU: `--arm seq --variant gru128_nomsg --set epochs=60`.

## Tests
(a) the generator reproduces the kit file layout (same column names as the real files); (b) fraction of ticks that are wins in the winning cell is within 0.05 of `hi`; (c) shared_change flip ticks within an episode spread < 3 ticks; (d) short_history episodes have exactly 64 history rows per machine.

## Final report (print only this)
Files created; test results; table: seed | model | validation points | stable | short_history | drift | shared_change ratios; the real-data reference values m3 55.3 (ratios 0.077, 0.107, 0.080, 0.078) and pooled_recent 30.8; one sentence: is the synthetic m3 score close to 55 or far from it?
