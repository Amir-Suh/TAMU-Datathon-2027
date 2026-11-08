# Fredy replay and payout-head swap

## Main validation

| Forecast | Points / 60 | ΔR vs M3 | Paired 95% interval for ΔR |
|---|---:|---:|---:|
| Frozen M3 | 55.268370 | 0.000000 | |
| Historical pair GBDT, 15 leaves | 53.945873 | +0.026614 | [+0.014780, +0.039317] |
| Current pair GBDT, 31 leaves | 54.289888 | +0.019566 | [+0.007591, +0.032282] |
| Seven plain GRUs, mean | 55.419191 | −0.002954 | [−0.012432, +0.006501] |
| Historical three-way blend reconstruction | 55.860278 | −0.011503 | [−0.017166, −0.006318] |
| Current pair GBDT + GRU mean + repaired M6 | 55.973765 | −0.013681 | [−0.019259, −0.008503] |
| M3 win + current three-way blend payout mix | 55.994410 | −0.014076 | [−0.016825, −0.011437] |
| M3 win + current pair GBDT payout mix | 55.751846 | −0.009414 | [−0.013891, −0.005134] |
| M3 win + current pair GBDT and plain GRU payout mix | 56.172553 | −0.017475 | [−0.021149, −0.013836] |
| Three factored GRUs, mean | 55.406231 | −0.002701 | [−0.011752, +0.006851] |
| Current pair GBDT + factored GRU mean + repaired M6 | 56.064085 | −0.015408 | [−0.020616, −0.010517] |
| M3 win + current pair GBDT and factored GRU payout mix | **56.259032** | **−0.019116** | **[−0.023045, −0.015197]** |

The historical two-way pair GBDT and M6 blend scored 55.389536, which rounds to Fredy's reported 55.39. The reconstructed three-way forecast scored 0.059722 points below the reported 55.92. The seven plain GRU runs ranged from 53.034659 to 54.557521.

The M3 win head with current pair GBDT and plain GRU payout mix reduced `R` versus M3 by 0.019020 on stable, 0.011526 on short history, 0.022366 on drift and 0.016987 on shared change. Its win-probability MSE stayed at M3's 0.004711. The win-mass-weighted conditional payout SSE fell from 0.010198 to 0.006784. Its 0.198788-point observed gain over the current full blend had a paired ΔR interval of [−0.008473, +0.001145]. Replacing only the win head of the full blend gained 0.020645 points, with a paired ΔR interval of [−0.004896, +0.004466].

## What was replayed

- Both LightGBM models use the source at Fredy commit `bbff7379e57984eb623588273ed1c958bb8b4f47`, pinned LightGBM 4.7.0, NumPy 2.5.4, SciPy 1.18.1 and narwhals 2.28.0 on Python 3.13.11. The runtime used two threads. The 15-leaf pair model was defined at `237118c9d61553f5b4fd838a152c56b131810d96` and the 31-leaf variant was introduced in the later commit. The rescored outputs match the branch's corresponding reported points to six decimal places.
- The plain GRU replay uses seven 100-epoch no-message, 128-hidden models. Fredy's log and commit document seeds 1–3 for the first three runs. The remaining four seed IDs and the original CUDA checkpoints were absent from the supplied branch. Seeds 4–7 complete the replay. The archived source runs directly on CPU for seeds 1, 3, 5 and 7. Seeds 2, 4 and 6 use a fused sequence GRU on Apple MPS with the same weights, gates, training inputs, objective and NumPy inference. That change and the device affect the training trajectory. GRU runtime was Python 3.10.20, PyTorch 2.11.0 and NumPy 2.2.6.
- The current factored GRU uses the branch's `gru128_factor` variant and CPU seeds 1–3. It is measured separately because the historical three-way blend predates this variant.
- The fit uses the main development contract. LightGBM learns from observed pre-cutoff ticks in all three splits and uses only held training episodes for early stopping. It consumes zero cutoff labels in dev mode. GRU additionally uses the train cutoff outcomes. Neither arm consumes validation cutoff outcomes, exact validation probabilities or query signal. Validation targets enter only the scorer. The GBDT train CSVs expose the same frozen models at train cutoffs for a calibration fit using the released train outcomes.

## Fixed formulas

Let `P` be a pair GBDT forecast, `G` the equal mean of the seven plain GRU forecasts, and `A` a frozen M6 forecast. The full blend is `(P + G + A) / 3`. The reported 55.92 used the 15-leaf pair model. The historical M6 input here is the prior forecast with SHA-256 `bc965bc34aaf3dc173e16a9e78622e925b17bca837d537d96624f1b52061a101`, matching Amir commit `c339b46`. The modern comparison uses the later 31-leaf pair model and repaired M6.

For a source mixture `F`, the head swap takes M3's exact loss probability and uses `p(k) = (1 - p_M3(0)) * F(k) / sum(F(1:))` for each positive payout. The `_gru_size` forecast uses the Fredy-only mixture `(P + G) / 2`. The `_gru_m6_*_size` forecast uses the conditional mix of the full three-way blend, isolating the win-head replacement. The `_gru_equal_size` diagnostic conditions the pair and GRU forecasts separately before averaging their conditional distributions.

## Artifacts and commands

- `forecasts/` contains keyed main-validation forecast CSVs. `test_forecasts/` contains the corresponding main-test outputs for candidates using repaired M6 or no M6. The supplied historical M6 CSV covers validation.
- `checkpoints/` contains every fitted component, run metadata and the original Fredy scorer's output. `scores.json` independently scores saved CSVs with the public kit on all 96 validation episodes. Its intervals use 10,000 paired, condition-stratified whole-episode bootstrap draws and describe fixed predictions on the released split.
- Use `PYTHON=<path> bash replay.sh pair_l31` or `PYTHON=<path> bash replay.sh gru128_nomsg 2 mps` from this directory after provisioning the dependencies above. The script archives Fredy's Git commit into a temporary directory, so it leaves the active checkout intact. The MPS run applies `fused_gru.patch` only in that temporary archive. Use `python3 -B compose.py` to rebuild derived forecasts and score all components.
- To check frozen inference, archive the same commit in a writable temporary directory, put its `src` on `PYTHONPATH`, use the LightGBM runtime above and run `python3 -B verify.py`. The verifier compares every reloaded component against its saved validation and test forecasts with a maximum allowed absolute difference of `1e-8`. The observed differences are in `verification.json`.
