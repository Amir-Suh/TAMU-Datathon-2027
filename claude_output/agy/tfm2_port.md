# Antigravity job: port the working transformer `tfm2` to numpy inference + train 8 seeds (30-minute budget)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; GPU available).

## Time budget
Run `date` now and write it to /tmp/agy_tfm2port/start.txt (mkdir -p). Hard limit 30 minutes: code + tests within 12 minutes, then launch the seed farm, report at 30.

## Rules
- You may edit ONLY `src/tdx/arms/tfm2/__init__.py`, `src/tdx/arms/tfm2/arm.py`, create `tests/tfm2/test_tfm2.py`, and write under `/tmp/agy_tfm2port/` and `runs/tfm2/`. Do NOT edit any other file (the `tfm` and `seq` arms stay as is). No git, no network, no pip, no rm.
- Read once: `src/tdx/arms/tfm2/arm.py` (the working torch model: causal conv front-end `conv`, ALiBi bias `alibi`, pre-LayerNorm attention blocks, head of 5 logits), `src/tdx/arms/tfm/arm.py` (it already has a numpy `forward_np`, `save`, `load`, parity check for the plain transformer: reuse that style), `src/tdx/arms/seq/arm.py` (data conventions, parity check, `save`/`load`), `src/tdx/arms/seqens/arm.py` (the torch-free numpy-only loader pattern we ship).

## Task 1: numpy inference for tfm2
1. Add a pure-numpy float64 forward pass that reproduces the torch model EXACTLY, including the causal depthwise+pointwise conv front-end (left padded, residual), the ALiBi distance bias with its per-head slopes, LayerNorms, the MLP with the same activation as torch, and the final head. Predictions for a query are the output after consuming ticks < cutoff (as in `seq`); clip to >= 1e-6 and renormalise like `seq.predict`.
2. After every `fit`, run a parity check: torch float64 (eval mode) vs the numpy forward on 3 episodes must agree to < 1e-6 max abs on the logits; raise if not (as `seq` does).
3. `predict` must use the numpy forward only. `save(dir)` writes `model.npz` (all weights, float64) + `meta.json` (all hyper-parameters); `load(dir)` rebuilds without torch (the loading path and predict must not import torch; torch is imported lazily inside `fit` only).
4. Register the variant `d128l3` = {d:128, L:3, heads:4, epochs:100, lr:2e-3, wd:1e-2, batch:32, transductive:True, conv:9, alibi:True, warmup:True, dropout:0.1}.
5. Tests `tests/tfm2/test_tfm2.py` (tiny model, 1 epoch, d=16, L=1): (a) numpy-vs-torch parity; (b) save/load round trip gives identical predictions; (c) predictions ignore ticks at or after the cutoff; (d) causality: changing the token at tick t does not change logits at ticks < t; (e) `import tdx.arms.tfm2` followed by loading and predicting from a saved folder does not import torch (check `'torch' not in sys.modules` in a subprocess).
6. Measure and print the numpy inference time for the 1,536 test queries for one d128l3 model with `taskset -c 0,1` and OMP/OPENBLAS threads = 2.

## Task 2: seed farm (after the tests pass)
Train 8 seeds (21..28), at most 4 at a time:
`$PY -m tdx.run --arm tfm2 --variant d128l3 --mode dev --set seed=S --out runs/tfm2/d128l3_seed_S > /tmp/agy_tfm2port/seed_S.log 2>&1`
(each took about 4 minutes in earlier tests; they produce `runs/tfm2/d128l3_seed_S/model` via save and `meta.json`).

## Final report (print only this)
- test result lines; the parity max abs difference printed by one fit; the 2-core inference seconds for one model.
- table: seed | validation points | stable | short_history | drift | shared ratios | fit seconds, then the mean of the 8 validation points and the min/max.
- any failure with its last 5 log lines.
