# Antigravity job: FAST transformer fix test (hard 15-minute budget, torch-only experiment arm)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root with default env; the GPU is available).

## Time budget (important)
Run `date` now and write the time in /tmp/agy_tfm2/start.txt. You have 15 MINUTES in total. At 10 minutes after the start, stop writing code and launch whatever you have; at 15 minutes print the final report with what has finished. Do not run unit tests beyond one smoke fit. Do not optimise anything.

## Rules
- Create ONLY `src/tdx/arms/tfm2/__init__.py`, `src/tdx/arms/tfm2/arm.py` and files under `/tmp/agy_tfm2/`. Do NOT edit any existing file (the `tfm` arm stays as is). No git, no network, no pip, no rm.
- Start from `src/tdx/arms/tfm/arm.py` (copy it) and `src/tdx/arms/seq/arm.py` for the data conventions. The arm only needs `fit(rnd, spec)` and `predict(split)`; use TORCH for predict too (no numpy forward, no save/load, no parity test). `python -m tdx.run --arm tfm2 --variant V --mode dev --out runs/tfm2/V` must work (run.py calls arm.save inside a try/except NotImplementedError, so make `save` raise NotImplementedError).

## Why the existing `tfm` failed
All six runs score 29-31 validation points (the pooled baseline is 30.0); the GRU `seq` scores 53-54. Our game rule depends on the outcomes at two particular earlier lags, e.g. (1,3), (2,5), (4,7) (and flips direction around tick 140-150). Learned absolute position embeddings make "the tick k steps back" hard to learn from this little data; a GRU gets relative lags for free.

## Changes to make in tfm2 (flags, all in the same arm)
1. `alibi` (bool): drop the learned absolute position embedding and add an ALiBi-style per-head linear distance bias to the attention scores (slopes 2^(-8*(h+1)/H)); keep the causal mask. Use `scaled_dot_product_attention` with an explicit float `attn_mask` (bias + causal -inf).
2. `conv` (int k, 0 = off): before the first attention block, a causal depthwise-plus-pointwise 1D convolution over the token embeddings with kernel size k=9 (left-pad k-1 so no future leakage), added residually. This gives every position direct access to its last 8 outcomes.
3. `warmup`: linear lr warmup for the first 10% of steps then cosine decay (peak lr 2e-3).
4. Keep `factor` out (not needed). d=64, 2 layers, 4 heads, dropout 0.1, batch 32, `epochs` param.

## Variants (register in VARIANTS, all `transductive=True`)
- `conv9` : conv=9, alibi=False (but keep learned absolute positions), warmup on, 60 epochs
- `alibi` : alibi=True, conv=0, warmup on, 60 epochs
- `conv9_alibi` : conv=9, alibi=True, warmup on, 60 epochs

## Run (all three in parallel, seed 7, as soon as the code runs once)
`$PY -m tdx.run --arm tfm2 --variant V --mode dev --out runs/tfm2/V > /tmp/agy_tfm2/V.log 2>&1`
If a variant crashes, fix it once; if it still fails, report the error and move on.

## Final report (print only this)
A table: variant | validation points | stable | short_history | drift | shared_change ratios (from the printed score line or runs/LOG.md) | fit seconds. State the elapsed minutes since start.txt. List any failure with its last 5 log lines. No interpretation beyond one sentence.
