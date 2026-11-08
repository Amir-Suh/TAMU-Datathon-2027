# House Edge: Team Timecasters (TAMU Datathon 2026)

We predict the true next-spin payout probabilities (0x, 1x, 2x, 5x, 20x) of simulated slot machines. Scores compare our error with a pooled baseline's error, so lower error is the only thing that counts.

## The model
One blend of three model families, with the win probability and the payout-size mix blended separately:
- **M3**: a Bayesian per-machine model with learned lag effects (refit at predict time, CPU only).
- **12 GRU sequence models** (size-x2 loss) and **8 transformers**, trained on next-tick prediction from every released history (ticks before each cutoff only) and on train labels only. No validation or test answers are used.
- **Fixed blend weights** found by 4-fold cross-validation grouped by whole episode on the main round. The adaptation round re-checked them by the same k-fold on its validation set; tuning added about 0.04 points (interval includes 0), so they stayed.

What we learned: the large gain comes from data volume, not from memorising one machine's history (96 episodes of history scored 34.3; 384 scored 54.7). Mixture-of-experts and 150 epochs did not help; the simpler model won.

## Layout
- `src/tdx/`: training arms (`arms/`), packaging (`package/`), evaluation and diagnostic tools (`tools/`). `tdx/` is a symlink to it.
- `submissions/`: the exact files we uploaded for each round, with SHA-256 sums and a README (`adaptation_timecasters/` is round two; `candidate4_B_uploaded/` is round one).
- `kit/`: the organizers' released checker and helpers (the code imports `slot_forecasting` from here).
- `tests/`, `config/`, `requirements-dev.lock`.

## Reproduce
Data is not included. Each `entry_*.zip` is self-contained: unzip it, install the pinned `requirements.txt` (numpy, scipy), and run `python predict.py --data <public dir> --out submission.csv` (about 4 minutes, CPU only). Check a CSV with `python kit/check_submission.py <csv> --data <public dir>`. Both uploaded entries reproduced bit-for-bit in a fresh folder.
