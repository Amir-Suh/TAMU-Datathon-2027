# Main-round submission candidates (House Edge)

Each folder holds the exact `entry_main.zip` (code + models, offline-reproducible) and `submission.csv` (1,536 test forecasts) built on 2026-11-07, plus `SHA256SUMS`. Validation numbers are dev-mode (no validation answers used in fitting) and optimistic: the 96 validation episodes were used to compare candidates.

| Folder | What it is | Validation points | Notes |
|---|---|---|---|
| `candidate1_even_blend` | 50/50 probability average of Bayesian `m3` and a 12-seed GRU ensemble | 56.17 | first GRU+m3 entry |
| `candidate2_split_duty` | win probability 70% `m3` + 30% GRU ensemble; payout-size mix from the GRU ensemble only | 56.58 | "factored" blend, `src/tdx/package/blend.py` |
| `candidate3b_with_transformer` | candidate 2 plus an 8-seed conv+ALiBi transformer ensemble (win `[.77,.23,0]`, size `[.08,.60,.32]` over m3 / GRU / transformer) | 56.90 | original tail fit carried the later query signal into earlier pseudo-queries |
| `candidate3b_clean_refit` | M3 with corrected signal timing, 12 GRUs and eight transformers with refitted blend weights | 56.753567 | 56.417694 on 24 reserved validation episodes |

A candidate "3a" (same blend, unchunked transformer inference) was NOT kept: its reproduction peaked at 16.3 GB, over the organizer's 16 GiB limit.

## Verify a candidate completely
1. Check the hashes: `cd submissions/candidate3b_with_transformer && sha256sum -c SHA256SUMS`.
2. Format: `python kit/check_submission.py submissions/candidate3b_with_transformer/submission.csv --data <public dir>` (expects `VALID FORMAT: 1536 queries`).
3. Offline reproduction (fresh Python 3.13 venv, exact pins, two CPU cores, no network; compares two runs and the uploaded CSV):
   `bash scripts/repro_check.sh submissions/candidate3b_with_transformer/entry_main.zip <public dir> submissions/candidate3b_with_transformer/submission.csv`
   Expected for 3b: `max|run1-run2|=0`, `max|run1-uploaded|=0`, ~225 s per run, peak RSS ~1.2 GB, no torch imported.
4. By hand: unzip, `python predict.py --data <public dir> --out out.csv` in a Python 3.13 venv with `numpy==2.5.4` and `scipy==1.18.1`, compare `out.csv` with `submission.csv` (about 4 minutes).

Each ZIP contains a snapshot of the inference source and the organizer kit's
`slot_forecasting/` package. The clean refit uses train-only calibration
models and 45,616 observed historical outcomes from validation folds 0–2.
M3's internal historical win and payout tuning and the blend-tuning backtest
use neutral signal at earlier cutoffs. Real queries use their recorded
signals. The 24 episodes in fold 3 contribute no outcomes to the blend fit.

The clean-refit CSV has 1,536 forecasts and reproduces byte for byte from a
fresh extraction on Python 3.13 on macOS, using one CPU inference thread.
Verify the exact files with the `SHA256SUMS` in the candidate folder.

## Uploaded main-round entry (Team Timecasters, Round 1)
`candidate4_B_uploaded`: the entry uploaded to the main Form on 2026-11-08 (README inside the zip names the team and round). Recipe: m3 + 12 size-x2 GRUs + 8 transformers, blend weights fixed from the main-round grouped cross-validation (out-of-fold 57.283 vs 57.106 for the 3b recipe; the CV differences are small, about 0.1 point). Fresh-folder reproduction passed (246 / 233 s) on the earlier-built zip; the uploaded zip differs only in the README line.

## Uploaded adaptation entry (Team Timecasters, Round 2)
`adaptation_timecasters`: the round-two entry. Same recipe as candidate 4 (B) refit on the adaptation kit: m3 (refit at predict time) + 12 size-x2 GRUs + 8 transformers, trained on the new round's histories (train labels only), with the frozen main-round blend weights. Validation (96 episodes, 40-point scale): 38.19; a 4-fold grouped CV re-tuning of the blend gave +0.04 (interval includes 0), so the weights were kept. Reproduction: run 1 = run 2 = uploaded CSV exactly (226 s, 1.2 GB).
