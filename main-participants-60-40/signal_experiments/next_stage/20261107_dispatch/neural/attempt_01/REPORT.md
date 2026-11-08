# E6 main-development proxy

**Main decision** Reject the tested challenger. The best frozen hybrid swap scored **56.057023 / 60** against **56.341022 / 60** for the incumbent. Its paired ΔR was **+0.005395** [0.000334, 0.010778], with deterioration in all four condition ratios. Adaptation evaluation depends on E5's released-round, fold-safe offset export.

| Frozen forecast | Points / 60 | R | p0 MSE | R on 64 observations | R on full prefixes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Archived incumbent hybrid | 56.341022 | 0.064943 | 0.004468 | 0.092936 | 0.055612 |
| E6 win swap into archived hybrid | 56.057023 | 0.070339 | 0.005277 | 0.101193 | 0.060054 |
| Archived factored GRU | 55.406231 | 0.082911 | 0.007103 | 0.121242 | 0.070134 |
| Repaired M3 | 55.268370 | 0.085612 | 0.004711 | 0.106894 | 0.078518 |
| E6 signal and offset | 55.242544 | 0.086119 | 0.005277 | 0.113939 | 0.076846 |
| E6 signal, offset and mask augmentation | 55.196745 | 0.087021 | 0.005302 | 0.115349 | 0.077578 |
| Frozen N1 proxy structure | 54.769686 | 0.095497 | 0.005971 | 0.116083 | 0.088634 |
| Matched eight-epoch factored GRU | 32.149866 | 0.866260 | 0.111367 | 0.914008 | 0.850344 |

The unaugmented E6 residual improved the N1 structural comparison by ΔR −0.009377 [−0.013770, −0.004975]. The short-history R fell by 0.002144 while p0 MSE rose by 0.000023. The 64-observation and full-prefix slices include 192 and 576 queries, respectively. Their ratios use the condition-specific reference error over episodes in each slice. The eight-epoch factored control performed poorly across all three seeds. Its large gap to the residual arm reflects the bundle of structured features, offset and signal with a short training budget. The archived factored GRU is the stronger external reference and had a different fit recipe and history exposure.

Mask augmentation increased ΔR by 0.000901 [0.000130, 0.001671] on released validation and by 0.001629 [0.000506, 0.002774] when both E6 arms consumed deterministic 24+40 masks. On the 576 originally full-prefix rows alone, the latter change was +0.001703 [0.000433, 0.003050]. The naturally sparse 192 rows were preserved by the evaluation mask. The unaugmented E6 seeds ranged from 54.812529 to 55.434440 points. The augmented seeds ranged from 54.798822 to 55.441482. Ensemble scores use the equal probability average. All reported intervals use 10,000 condition-stratified paired episode draws. [Incumbent comparisons](scores_vs_hybrid.json) · [Matched comparisons](scores_vs_matched.json) · [Masked comparisons](scores_mask_vs_matched.json) · [Slices and class contributions](slices.csv)

Across released validation, the observed −1 to +1 win gap was 7.3545 percentage points in the odds, 7.8793 in E6 and 6.3038 in the N1 structural control. At a fixed history, forcing only the recorded-cutoff signal from −1 to +1 produced a 6.3875-point average change for E6, compared with 6.3670 points for the structure. The residual changed the structured signal response by +0.0205 points. [Signal diagnostics](signal_slopes.json)

## Training and inference scope

The structured parent is explicitly an N1 proxy using causal per-tick 28-pair filtering. Population parameters were refit from the training histories outside each held episode fold. The inference parent was refit on all 96 main training histories and frozen before validation or test prediction. Observed training history supplies pseudo-endpoint outcomes and released train labels supply the actual-cutoff outcomes. The neural model has 128 recurrent units, a learned 8-dimensional joint type/signal embedding, previous-payout token, previous-tick observation bit, tick divided by 256, and N1's causal posterior summaries. The signal is neutral at historical ticks. The E6 win correction starts at zero and has a 0.01 squared-logit penalty. The positive head uses four learned logits. Actual-cutoff supervision gets weight 8 in all arms. Each arm shares seeds 101, 202 and 303, batch size 48 and eight epochs. The factored ablation uses the same network and neutral signal with structural channels and win offset zeroed.

Augmentation considers actual training endpoints and the two fixed historical pseudo-endpoint ticks 128 and 192. A prefix is eligible when it has 40 contiguous recent observations and more than 24 observed older ticks. Eligibility is grouped by episode. SHA-256 ranks the eligible episodes per endpoint and selects the first floor(n/4), giving 18/72, 17/69 and 18/72 at ticks 128, 192 and actual, respectively. Earlier observations are selected with the existing deterministic N1-suite stress mask. Each masked sequence supervises only its own endpoint. The paired control has the same supervised endpoint on an unmasked prefix with the same targets, weights and batch order. The 25% coverage applies to these predefined eligible cohorts. [Fold membership and mask cohorts](full/folds_and_masks.json)

The archived factored GRU and incumbent hybrid used observed pre-cutoff histories across main train, validation and test in their archived fits. Hybrid swaps inherit this history exposure. The size swap holds archived win mass and replaces the archived GRU in the equal joint Pair-L31 and GRU mixture before taking the conditional size distribution. The win swap holds the archived hybrid's conditional size distribution fixed. The standalone E6 forecasts use main training fits only.

## Readiness and replay

The 12-episode, one-seed smoke across all three arms completed in 3.12 seconds. Full fit/predict ran in 108.14 seconds with one numerical thread, including 68.42 seconds for training, invariant checks and serialization, 23.42 seconds for all validation diagnostics and 11.86 seconds for all test forecasts. All nine full runs had zero logit change through the tested endpoints after perturbing future outcomes or signal. The maximum Torch/NumPy logit difference was 1.43e−6. A fresh NumPy-only run finished in 36.49 seconds and reproduced 51 forecast CSVs and the counterfactual archive byte for byte. Torch imports and train reads were checked separately on replay. The public checker accepted all 51 CSVs. The organizer package replay is pending a candidate that passes the selection gate. [Invariance and parity checks](full/fit_checks.json) · [Replay verification](replay_parity.log) · [Format checks](format.log)

The fit/predict guard traces show reads of validation and test query/history inputs and no reads of protected answers. Validation targets, conditions and reference forecasts were opened only after the forecast hashes had been frozen. The weights NPZ includes the nine networks and the structured priors required by replay. [Fit manifest](full/manifest.json) · [Forecast hashes](full/forecast_freeze.json) · [Weights](full/weights.npz) · [Access trace](full_access.jsonl)

## Reproduction commands

Use the participant-kit root and fresh output and trace names when rerunning. The archived logs use the original names below.

```sh
A=signal_experiments/next_stage/20261107_dispatch
D="$A/neural/attempt_01"
python3 -B "$A/coordinator/audit_run.py" --stage fit-predict --trace "$D/smoke_access.jsonl" "$D/run.py" --smoke --out "$D/smoke"
python3 -B "$A/coordinator/audit_run.py" --stage fit-predict --trace "$D/full_access.jsonl" "$D/run.py" --out "$D/full"
python3 -B "$A/coordinator/audit_run.py" --stage score --trace "$D/score_access.jsonl" "$D/score_e6.py" "$D/full"
python3 -B "$A/coordinator/audit_run.py" --stage fit-predict --trace "$D/replay_access.jsonl" "$D/run.py" --replay "$D/full/weights.npz" --out "$D/replay"
python3 -B "$D/verify_replay.py" "$D"
for p in "$D"/full/validation_*.csv; do python3 -B check_submission.py "$p" --split validation; done
for p in "$D"/full/test_*.csv; do python3 -B check_submission.py "$p"; done
```
