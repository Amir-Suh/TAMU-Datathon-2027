# Antigravity job: short_history study on a new arm `seq2` (implementation + grid)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; GPU is available to this interpreter).

## Rules
- Create ONLY: `src/tdx/arms/seq2/__init__.py`, `src/tdx/arms/seq2/arm.py`, `tests/seq2/test_seq2.py`, and files under `/tmp/agy_short/`. Do NOT edit any existing file (the `seq` arm must stay exactly as is). No git, no network, no pip, no rm.
- Read once before coding: `src/tdx/arms/seq/arm.py` and `src/tdx/arms/seq/__init__.py` (copy them as the starting point for seq2: same fit/predict/save/load interface, numpy float64 inference, torch-vs-numpy parity check), `src/tdx/backtest.py` (the tail backtest: `python -m tdx.backtest --arm A --variant V --hold 30 --short`, prints `logloss=`), `tests/seq/test_seq.py`.

## Background
`short_history` episodes show each machine exactly 64 observed ticks before the cutoff: the last 40 consecutive ticks plus 24 older ticks sampled at random; every other tick is missing (-1). Our GRU arm `seq` scores worst on that condition. We want training tricks that help it, measured by the tail backtest with `--short` (23,040 outcomes, much more power than the 24 validation short_history episodes).

## seq2 options (all default OFF so `seq2` with defaults equals `seq`; keep the winning ones composable)
1. `aug` (float, prob per training episode of masking inputs to the short layout: keep the last 40 ticks before the cutoff and 24 randomly sampled older ticks per machine; fresh random draw every batch) and `aug_target`: `'masked'` (targets only at kept ticks, as in seq's existing aug) or `'all'` (inputs masked but targets are the ORIGINAL next-tick outcomes at every observed tick).
2. `short_flag` (bool): add a learned 4-dim embedding of a per-machine flag `flag = (number of observed ticks in that machine's history row == 64)` as an extra GRU input at every step. Compute the flag from the (possibly masked) `y` actually fed to the network, in training AND in `predict` (do not use `n_obs`).
3. `distill` (float weight): for augmented rows only, also run the unmasked history through the same network in no-grad eval mode as a teacher, and add `distill * KL(teacher_probs || student_probs)` averaged over ticks where the student's target is observed, so the short-view predictions are pulled toward the full-view predictions.
4. `tokdrop` (float): independently drop each input token to missing (-1) with this probability on ALL training rows (inputs only; targets unchanged).
5. `factor` (bool): the win/size factored head already implemented in seq (copy it).
Keep numpy/torch parity < 1e-9 (the flag embedding must be in the numpy forward pass too) and the save/load round trip exact.

## Tests (tests/seq2/test_seq2.py)
(a) parity + round trip for a tiny model with `short_flag=True, factor=True, epochs=1`; (b) predictions ignore ticks at or after the cutoff; (c) with defaults, a 1-epoch `seq2` fit and a 1-epoch `seq` fit with the same seed give predictions within 1e-5 on the validation split.

## Grid (run after the tests pass; at most 4 jobs at a time; each backtest refits the model, about 3 minutes on the GPU)
All with `h=128, msg=False, epochs=100, transductive=True`. Register each as a VARIANT in `seq2/__init__.py`:
- `base` (all off), `aug3` (aug 0.3, aug_target masked), `aug3all` (aug 0.3, aug_target all), `aug5all` (aug 0.5, all), `flag_aug5all` (short_flag + aug 0.5 all), `flag_aug5all_dist` (+ distill 0.5), `flag_aug5all_fac` (+ factor), `tokdrop2_flag_aug5all` (tokdrop 0.2 + short_flag + aug 0.5 all), `all_in` (short_flag, aug 0.5 all, distill 0.5, factor, tokdrop 0.1).
For each variant run BOTH:
  `$PY -m tdx.backtest --arm seq2 --variant V --hold 30 --short` (and the same without `--short`)
  `$PY -m tdx.run --arm seq2 --variant V --mode dev --out runs/seq2/V` (prints validation points; the per-condition ratios are in `runs/seq2/V/score.json` under audit/by_condition or in runs/LOG.md).
Write logs to `/tmp/agy_short/<variant>_<kind>.log`.

## Final report (print only this)
One table: variant | backtest logloss short=1 | backtest logloss short=0 | validation points | short_history ratio | stable ratio | drift ratio | shared ratio. Then the three variants with the lowest short=1 logloss, the test and parity result lines, and any failed step with its last 5 log lines. No interpretation beyond one sentence.
