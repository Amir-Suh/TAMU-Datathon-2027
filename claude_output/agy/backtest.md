# Antigravity job: leak-free tail backtest harness (new files only)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root).

## Rules
- Create ONLY these files: `src/tdx/backtest.py` and `tests/test_backtest.py`. Do not edit any other file. No git, no network, no pip, no rm.
- Read once before coding: `src/tdx/data.py` (Split, Round, load_round), `src/tdx/run.py` (execute, fit_round, arm_module, build_arm), `src/tdx/fit.py`, `src/tdx/arms/pooled/arm.py`, `src/tdx/arms/seq/arm.py` (how `predict(split)` uses `split.y`, `split.cutoff`, `split.mtype`).

## Goal
Score an arm on many realized outcomes instead of 768 queries, without leaking. Idea ("tail holdout"): pretend each episode's cutoff was `hold` ticks earlier, fit the arm only on that truncated data, then score its forecasts at every tick of the held-out tail against the realized outcome with log-loss.

## Spec
`truncate(rnd, hold)` returns `(rnd_trunc, full_splits)`:
- For every split, copy it with `cutoff = cutoff - hold` and `y[..., t] = -1` for all `t >= cutoff - hold`. Keep `full_splits` = the original splits (their `y` holds the realized tail). Keep all other fields unchanged. Set `label`, `target`, `condition`, `reference` to None on the truncated copy so nothing at or after the true cutoff can be used.

`tail_logloss(arm_name, variant, hold=30, short=False, overrides=None, seed=0)`:
1. `rnd = load_round('data/main', answers=False)`; `rt, full = truncate(rnd, hold)`.
2. Build the arm exactly as `tdx.run.execute` does (module = arm_module(arm_name); params = dict(module.VARIANTS[variant]); update with overrides; `build_arm`). Fit it with `dev_spec(params.get('transductive', False))` on `fit_round(rt, spec)`.
3. For each j in range(hold): make a pseudo-query split from the VALIDATION split of the truncated round with `cutoff = cutoff_trunc + j` and `y` restored from `full['validation'].y` for ticks `< cutoff_trunc + j` only (everything at or after is -1). If `short=True`, additionally mask each machine's history to the short_history layout: keep the last 40 ticks before the pseudo cutoff plus 24 randomly sampled older ticks (numpy RandomState(seed)), set all others to -1. Call `arm.predict(split)` -> array [E, 8, 5].
4. The realized outcome is `full['validation'].y[e, m, cutoff_trunc + j]`; skip machines where it is -1 (not observed). Log-loss = mean of `-log(clip(pred[e, m, outcome], 1e-6, 1))` over all (e, m, j) scored; also report it for the first 10 / mid 10 / last 10 values of j and the number of outcomes scored.

CLI: `python -m tdx.backtest --arm A --variant V [--hold 30] [--short] [--set key=json ...]` printing one line:
`backtest A/V hold=30 short=0 logloss=<x.xxxx> n=<count> first=<x> mid=<x> last=<x>`.

## Tests (tests/test_backtest.py)
1. `truncate`: for every split, every `y` entry at index >= new cutoff is -1; `full` still holds the originals.
2. No-leak: with pooled/pooled_recent, corrupt `full['validation'].y` at ticks >= cutoff_trunc + 5 (set to 4) and show the pseudo-query forecasts for j = 0..4 do not change.
3. Sanity: `tail_logloss('pooled','pooled_recent')` and `tail_logloss('pooled','pooled')` are finite and between 0.5 and 2.0.

## Proof (run all, paste results in your final report)
```
$PY -m pytest -q tests/test_backtest.py
$PY -m tdx.backtest --arm pooled --variant pooled_recent --hold 30
$PY -m tdx.backtest --arm pooled --variant pooled_recent --hold 30 --short
$PY -m tdx.backtest --arm gbdt --variant pair_trans_64_l15 --hold 30
$PY -m pytest -q tests 2>&1 | tail -2
```

## Final report (print only this)
Files created; test result lines; the three `backtest ...` lines; the full-suite tail line; anything that did not work.
