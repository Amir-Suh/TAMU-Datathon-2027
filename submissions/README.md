# Main-round submission candidates (House Edge)

Each folder holds the exact `entry_main.zip` (code + models, offline-reproducible) and `submission.csv` (1,536 test forecasts) built on 2026-11-07, plus `SHA256SUMS`. Validation numbers are dev-mode (no validation answers used in fitting) and optimistic: the 96 validation episodes were used to compare candidates.

| Folder | What it is | Validation points | Notes |
|---|---|---|---|
| `candidate1_even_blend` | 50/50 probability average of Bayesian `m3` and a 12-seed GRU ensemble | 56.17 | first GRU+m3 entry |
| `candidate2_split_duty` | win probability 70% `m3` + 30% GRU ensemble; payout-size mix from the GRU ensemble only | 56.58 | "factored" blend, `src/tdx/package/blend.py` |
| `candidate3b_with_transformer` | candidate 2 plus an 8-seed conv+ALiBi transformer ensemble (win `[.77,.23,0]`, size `[.08,.60,.32]` over m3 / GRU / transformer) | 56.90 | weights fit on a leak-free held-out tail; transformer inference runs in 12-episode chunks (peak RSS 1.2 GB) |

A candidate "3a" (same blend, unchunked transformer inference) was NOT kept: its reproduction peaked at 16.3 GB, over the organizer's 16 GiB limit.

## Verify a candidate completely
1. Check the hashes: `cd submissions/candidate3b_with_transformer && sha256sum -c SHA256SUMS`.
2. Format: `python kit/check_submission.py submissions/candidate3b_with_transformer/submission.csv --data <public dir>` (expects `VALID FORMAT: 1536 queries`).
3. Offline reproduction (fresh Python 3.13 venv, exact pins, two CPU cores, no network; compares two runs and the uploaded CSV):
   `bash claude_output/sh/repro_check.sh submissions/candidate3b_with_transformer/entry_main.zip <public dir> submissions/candidate3b_with_transformer/submission.csv`
   Expected for 3b: `max|run1-run2|=0`, `max|run1-uploaded|=0`, ~225 s per run, peak RSS ~1.2 GB, no torch imported.
4. By hand: unzip, `python predict.py --data <public dir> --out out.csv` in a Python 3.13 venv with `numpy==2.5.4` and `scipy==1.18.1`, compare `out.csv` with `submission.csv` (about 4 minutes).

Source for everything inside the zips is this branch (`src/tdx/`; the zip's copy matches it byte for byte except `tdx/paths.py` and `tdx/package/__init__.py`, which the builder rewrites). `slot_forecasting/` in the zips is the organizer kit's package.
