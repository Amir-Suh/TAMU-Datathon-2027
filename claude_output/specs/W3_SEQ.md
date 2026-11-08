# W3 — `seq`: small causal sequence model (GPU training, numpy inference)

## Role and objective
Implement arm `seq` in `src/tdx/arms/seq/`: a tick-synchronous recurrent model over the 8 machines of an episode that predicts the next payout distribution for every machine at every tick. Train with torch (try GPU), export weights, and implement a numpy float64 forward pass used by `predict`.

## Why
Family 3 of the bake-off. The data gives ~560k observed next-tick labels, which a recurrent model can use densely; the risk is generalisation across regimes with only 96 train episodes. Read first: `claude_output/02_BAKEOFF_PLAN.md` section 3C, `claude_output/01_ADVERSARIAL_REVIEW.md` rows 5, 13, 16, `src/tdx/{data,fit,arm}.py`.

## Decisions (do not reopen)
- Input per (episode, tick, machine): payout token ∈ {0..4, MISSING=5} (embedding 8), type embedding (4). Machine identity within an episode is not used (order-invariant across the 8 machines).
- Per-machine GRU cell (shared weights) with hidden size H ∈ {64, 128}. Cross-machine message: at each tick the mean over machines of the projected hidden states (and optionally the max), concatenated into every machine's GRU input at the next tick. A variant without messages.
- Head: linear to 5 logits from the machine's hidden state at tick `t-1`, predicting tick `t`. Loss: cross-entropy on observed next ticks only (mask MISSING); optional Brier term. In `final` mode add the validation soft-target cross-entropy at the cutoff positions with weight `w_cut` (default 20) and train labels at cutoff positions likewise.
- Training: AdamW lr 2e-3, weight decay 1e-2, batch of 16 episodes, up to 40 epochs, early stopping on 12 held-out whole train episodes (train episodes only, in every mode), gradient clip 1.0, seed 7, dropout 0.1. Truncated windows of 128 ticks sampled from each episode; evaluation always runs the full history to `cutoff-1`.
- Prediction at the cutoff = head output after consuming ticks `< cutoff` (the state at `cutoff-1` predicts `cutoff`). Missing ticks inside short_history are MISSING tokens; the GRU keeps stepping.
- Transductive: tick rows from validation and test histories added to training (same loss) in addition to train.
- Export: `save` writes `model.npz` (float32 trained, converted to float64 on load) and `meta.json`; `predict` uses a numpy forward pass. Parity test: torch float64 vs numpy float64 max abs diff < 1e-9 on 5 episodes.
- CUDA is NOT available inside your sandbox (verified: `/dev/nvidia*` is hidden). Write device-agnostic code (`--device auto|cpu|cuda`), smoke-test every variant on CPU with a tiny budget (2 epochs, 8 episodes, < 2 min), and do NOT run full training. The user runs the full sweep on the RTX 5080 outside the sandbox with `claude_output/sh/14_gpu_sweep.sh` (it calls `python -m tdx.sweep --arm seq --device cuda`). Your `sweep` must accept `--device` and `--smoke` and write the same `runs/seq/<variant>/` files.

## Preconditions
Hard: W0 merged; `~/.venvs/tdx/bin/python -c "import torch; print(torch.__version__)"` works. Soft: CUDA availability (see above).

## Owned files
`src/tdx/arms/seq/*`, `tests/seq/*`, `runs/seq/**`, `claude_output/notes/W3_*`. Forbidden: everything else.

## Work packages
- **P1 numpy forward + torch model + parity test.**
- **P2 training loop** (device-agnostic, `--smoke` mode) with early stopping, deterministic seeds (`torch.manual_seed`, `use_deterministic_algorithms(True)` where possible), `build`, `VARIANTS`: `gru64_msg`, `gru64_nomsg`, `gru128_msg`, each with transductive 0/1 → 6 variants; full GPU budget per variant ≤ 10 min (run by the user, not you).
- **P3 leak test** passes (perturbing ticks ≥ cutoff must not change predictions).
- **P4 notes** `claude_output/notes/W3_seq.md`: LOG table, learning curves summary, the exact GPU command, parameter counts, checkpoint size, numpy inference time for 1,536 test queries on 2 threads.

## Acceptance
```
~/.venvs/tdx/bin/python -m pytest -q tests/seq
~/.venvs/tdx/bin/python -m tdx.sweep --arm seq --smoke --device cpu
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm seq
~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent
python kit/check_submission.py runs/seq/<best>/test_pred.csv
```
Time box 60 minutes for you (CPU smoke only); the GPU sweep is separate.

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1–P2 | gpt-6-astra | high | torch/numpy parity and training stability are the risky parts |
| P3–P4 | gpt-6.1-sol | low | mechanical |
| Review (read-only) | gpt-6.1-sol | high | different model |
