# Antigravity job: causal Transformer arm `tfm` (implementation + grid)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; run python with the default environment, the GPU is available but shared with other jobs: run at most 3 trainings at a time).

## Rules
- Create ONLY: `src/tdx/arms/tfm/__init__.py`, `src/tdx/arms/tfm/arm.py`, `tests/tfm/test_tfm.py`, and files under `/tmp/agy_tfm/`. Do NOT edit any existing file. No git, no network, no pip, no rm.
- Read once: `src/tdx/arms/seq/arm.py` and `src/tdx/arms/seq/__init__.py` (the interface to copy: `fit(rnd, spec)`, `predict(split)`, `save(dir)`, `load(dir)`, torch training with numpy float64 inference and a torch-vs-numpy parity check, deterministic seeds, the `factor` head), `tests/seq/test_seq.py`, `src/tdx/backtest.py` (CLI `python -m tdx.backtest --arm A --variant V --hold 30 [--short]`, prints `logloss=`).

## Goal
A second sequence model with a different inductive bias than the GRU, to add diversity to the ensemble. A causal Transformer can attend directly to specific earlier ticks (our game rule depends on the outcomes at two particular earlier lags, e.g. lags (1,3), (2,5), (4,7), and flips direction partway through the history).

## Spec (`tfm`)
- Same data conventions as `seq`: inputs `y[e,m,t]` in {-1 (missing), 0..4}; machine type embedding; targets are the next-tick outcome at every observed tick (cross-entropy); the prediction for a query is the output after consuming ticks < cutoff. Self-supervised training on train, validation and test histories (`transductive=True`) exactly as `seq` does; mask ticks at or after the cutoff.
- Architecture: per machine a sequence of length T=256 (batch over episodes x 8 machines; machines are independent, no cross-machine mixing in v1). Token embedding (6 entries, d) + learned absolute position embedding (256) + type embedding, then `L` causal self-attention blocks (pre-LayerNorm, `nhead` heads, MLP width 4d, dropout 0.1), output head of 5 logits per tick. Support the `factor` head (win logit + size softmax) like `seq`. Missing tokens are a normal input token.
- Variants to register: `tfm_d64_l2` (d=64, L=2, heads=4), `tfm_d128_l3` (d=128, L=3, heads=4), `tfm_d128_l3_fac` (same with factor head). AdamW lr 1e-3 with 5% linear warmup then cosine decay, weight decay 1e-2, batch 32 episodes, epochs param (start 100; ok to lower for speed), gradient clip 1.0, seed param.
- Numpy float64 inference for the whole forward pass (embedding gathers, LayerNorm, multi-head causal attention with a mask, GELU MLP) with torch-vs-numpy max abs difference < 1e-6 on 3 episodes checked after every fit (float64 torch for the check). `save`/`load` exact round trip (`model.npz` + `meta.json`). Inference must run on the 1,536 test queries in under 60 seconds on 2 CPU cores for one model (report the timing).

## Tests (tests/tfm/test_tfm.py)
(a) parity + round trip on a tiny model, 1 epoch; (b) predictions ignore ticks at or after the cutoff (corrupt them, predictions unchanged); (c) causality: changing the token at tick t does not change outputs at ticks < t.

## Grid (after the tests pass)
For each of the 3 variants with seeds 1 and 2: `$PY -m tdx.run --arm tfm --variant V --mode dev --set seed=S --out runs/tfm/V_seedS` (prints validation points; per-condition ratios are in runs/LOG.md) and `$PY -m tdx.backtest --arm tfm --variant V --hold 30 --set seed=S` (and with `--short`). Logs to `/tmp/agy_tfm/`.

## Final report (print only this)
A table: variant | seed | validation points | stable | short_history | drift | shared_change ratios | backtest logloss (full) | backtest logloss (--short) | fit seconds. Then: the numpy inference timing on 2 cores, test and parity results, and any failed step with its last 5 log lines. No interpretation beyond one sentence.
