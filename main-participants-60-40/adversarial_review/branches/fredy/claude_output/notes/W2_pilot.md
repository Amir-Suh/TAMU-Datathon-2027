# W2 GBDT pilot

## Results

Main-round validation results from `runs/LOG.md` (points and condition loss ratios; lower ratios are better). The eight configured sweep variants and two requested group ablations completed. Ablation log rows retain variant key `trans_64_l15`; distinct output directories identify the drop setting.

| Variant / output | Points | Stable | Short history | Drift | Shared change | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| all_inf_l15 | 35.123478 | 0.660494 | 0.789811 | 0.749401 | 0.633330 | 26.948 |
| all_64_l15 | 34.361767 | 0.689012 | 0.837107 | 0.787863 | 0.670525 | 25.567 |
| all_24_l15 | 33.505709 | 0.779938 | 0.855951 | 0.842091 | 0.684979 | 15.961 |
| last64_24_l15 | 33.087621 | 0.798631 | 0.866397 | 0.869726 | 0.718713 | 5.359 |
| last128_64_l15 | 34.253452 | 0.723100 | 0.840617 | 0.782389 | 0.660487 | 16.501 |
| all_64_l7 | 33.525941 | 0.763289 | 0.862991 | 0.839460 | 0.692897 | 22.012 |
| trans_64_l15 | 36.225856 | 0.620367 | 0.734134 | 0.683614 | 0.586987 | 99.106 |
| trans_24_l7 | 34.180138 | 0.737179 | 0.832522 | 0.789933 | 0.661988 | 94.561 |
| trans_64_l15, drop_neighbour | 36.395373 | 0.603969 | 0.725280 | 0.682277 | 0.582719 | 96.282 |
| trans_64_l15, drop_divergence | 36.191118 | 0.613067 | 0.737972 | 0.696892 | 0.583531 | 97.902 |

The sweep's best raw score was `trans_64_l15` (36.225856). In the two exploratory ablations, dropping neighbour features scored 36.395373 and dropping divergence scored 36.191118. `runs/gbdt/compare.txt` gives paired episode bootstrap intervals against `pooled_recent`: the full transductive model was +5.400 points, 95% interval [+4.631, +6.157]. The best non-smoke sweep score was clearly above the pooled baseline under the predeclared interval rule. The compare command ranks both ablations using the shared log variant key and displays their shared variant key; use the dedicated output artifacts and table above for the actual ablation comparison.

**Selection:** use `trans_64_l15` with `drop_neighbour=true`, from `runs/gbdt/trans_64_l15_no_neighbour`. It has the highest observed points of all ten runs and removes a feature group. Using `--baseline runs/gbdt/trans_64_l15`, `runs/gbdt/compare_anchor.txt` gives no-neighbour minus full model +0.170 points, 95% interval [-0.013, +0.355], and no-divergence minus full model -0.035, interval [-0.213, +0.153]. Both are ties under the reading rule; prefer the simpler 37-feature no-neighbour model provisionally. Its ratios improved in all four conditions relative to the base transductive pilot. Parent verified saved-model prediction parity and checker validity for this selection. Full repeat-fit determinism and the transductive leakage audit were verified on its base trans_64_l15 variant.

## Selected model feature importance

Top 15 LightGBM split gain features for the selected no-neighbour saved model:

1. `ema4_c0` — 207375.518
2. `machine_type` — 169549.668
3. `ema12_c0` — 54850.108
4. `wins_obs5` — 40245.196
5. `lag7` — 32336.350
6. `ema64_c1` — 30524.409
7. `wins_obs20` — 28149.239
8. `lag4` — 22017.133
9. `lag6` — 20786.099
10. `lag8` — 20340.920
11. `lag3` — 18989.361
12. `lag5` — 18111.277
13. `ema64_c2` — 17144.801
14. `lag2` — 16839.329
15. `code12` — 12731.224

## Behavior and limits

The top features concentrate on short and medium exponential payout-class histories, machine type, recent win counts, and lag identity. The transductive variants add only legal validation/test history rows in dev; cutoff supervision is final-only. The tau64/leaves15 transductive variant won the unablated grid; both transductive variants take about 95–100 seconds versus 5–27 seconds for non-transductive fits. Neighbour ablation slightly improved observed score and reduced feature count; divergence ablation was close to the base. These estimates use only 96 validation episodes, and ablation intervals against the anchor cross zero in the saved anchor comparison. The 36.395 score is therefore a pilot selection signal, not evidence that neighbour features are reliably harmful. Smoke output is preliminary and excluded from all conclusions.

Commands used:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/tdx/bin/python -m tdx.sweep --arm gbdt
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/tdx/bin/python -m tdx.run --arm gbdt --variant trans_64_l15 --set drop_neighbour=true --out runs/gbdt/trans_64_l15_no_neighbour
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/tdx/bin/python -m tdx.run --arm gbdt --variant trans_64_l15 --set drop_divergence=true --out runs/gbdt/trans_64_l15_no_divergence
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline pooled_recent > runs/gbdt/compare.txt
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/tdx/bin/python -m tdx.tools.compare runs/ --baseline runs/gbdt/trans_64_l15 > runs/gbdt/compare_anchor.txt
```
