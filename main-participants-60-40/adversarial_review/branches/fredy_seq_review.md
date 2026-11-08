# Fredy sequence-arm review

Reviewed snapshot: branch `fredy/bakeoff-gbdt-seq`, commit `5ba591c91d3589a2511b946dcf908b194e1390aa`, under `adversarial_review/branches/fredy`. All line references below are relative to that snapshot. `AGENTS.md` was read first. This review did not load validation answers, score the validation/audit targets, train a full-size model, install dependencies, or modify model source. The shared review's parent agent independently handles cross-branch scoring.

**Verdict:** this is a credible sequence challenger around the prior Bayesian model's performance, distinct from the earlier 31.6-point temporal CNN. It is not presently a stronger evidenced choice than the independently rescorable Amir candidates near 55.3. The sequence scores are selected full-validation development results with missing model/prediction artifacts. The current packaging path also rejects the sequence arm.

## Architecture and legitimate information

- `src/tdx/arms/seq/arm.py:64–80` is a real shared-weight GRUCell, with payout-token embedding width 8, machine-type embedding width 4, dropout 0.1, and a five-class linear output. The no-message H=128 model has 55,237 parameters. It consumes the full absolute-tick history, including a distinct missing token. There is no motif selector, moving-average smoothing head, transformer, or manual Bayesian lag-pair component hidden behind the SEQ name.
- The optional message is the mean of all eight machines' hidden states, included on the following tick (`:23`, `:29`, `:76–78`). It includes the machine's own state and is permutation-equivariant; it is not a peer-only statistic. The reported high-scoring runs use `gru128_nomsg`.
- **No cutoff-signal leakage:** `signal` is never read by this arm. Inputs are payout tokens, type, and optionally the previous peer aggregate. This complies with `AGENTS.md`'s rule that query signal is unavailable at historical ticks, although it also discards signal at the legal cutoff.
- `_masked` at `:8–10` removes all outcomes at or after each episode's cutoff. The head after consuming tick t is trained on tick t+1 (`:89–90`); a cutoff prediction selects the head after tick cutoff-1 (`:92–95`, `:109–114`). Missing targets are masked out. This is correctly aligned. The loader separately rejects history at or after cutoff (`src/tdx/data.py:95–100`).
- `src/tdx/fit.py:10–11` allows train/validation/test **histories** in transductive development mode, but only train cutoff labels and no soft validation targets. `SeqArm.fit` respects those lists (`arm.py:47–56`), and `run.fit_round` strips unauthorized supervision (`src/tdx/run.py:22–28`). The branch contract explicitly permits fitting on released validation/test histories. These predictions are valid transductive forecasts, but the evaluation episodes are not held out from history-based model fitting.
- Final mode deliberately uses validation soft targets (`src/tdx/fit.py:13–14`). Such a final refit must not be rescored as held-out validation. The normal runner scores only dev mode and removes old dev-score files in final mode (`src/tdx/run.py:65–76`). No evidence found that the logged dev scores were produced from final-mode soft-target fitting.

## Findings affecting the decision

**[P1] The sequence arm cannot be exported through the existing runtime packager.** `src/tdx/arms/seq/arm.py:42` imports Torch inside `fit`. `src/tdx/package/adapters.py:49–57` traverses imports anywhere in the AST, including methods; `:116–117` rejects all Torch imports. A direct, read-only reproduction of `copy_runtime(['seq'], temporary_directory)` fails with `ValueError: torch is forbidden in shipped runtime graph: torch.nn.functional`. The source can perform NumPy-only prediction once loaded, but the static package builder cannot accept its module. A runtime-only inference module or explicit export adapter is needed before claiming submission readiness. The actual pipeline configuration still selects GBDT, not SEQ (`config/pipeline.json:2–3`).

**[P1 evidence gap] The 53-point results cannot be independently reproduced or rescored from the supplied branch artifacts.** `runs/LOG.md:50–52` records three GRU128 no-message dev runs: 53.097964, 53.577773, and 53.382474 points. `claude_output/notes/pilot_results.md:3` explicitly says the epoch count was chosen by watching validation and identifies seeds 1–3. There are no supplied `runs/seq/**` prediction arrays, CSVs, checkpoint weights, or per-run metadata. The notes confirm these run directories are git-ignored. The log does not identify the epoch count, full override set, source hash, or exact model hash for each row. Consequently the reported lead seed cannot currently be matched to a retrievable model. Reusing the same variant path also overwrites artifacts by default (`src/tdx/run.py:62–70`, `:82–92`).

**[P2] The implemented training/evaluation protocol differs from the sequence specification and omits an independent epoch-selection check.** `claude_output/specs/W3_SEQ.md:13` calls for up to 40 epochs, early stopping on 12 whole held-out training episodes, 128-tick sampled windows, and batch size 16. Actual `src/tdx/arms/seq/arm.py:35–38` defaults to 100 epochs and batch 32; `:85–97` uses every epoch without early stopping or any held-out training episodes, and processes complete 256-tick arrays. The shipped variants use 100 epochs (`src/tdx/arms/seq/__init__.py:3–6`). Pilot documentation acknowledges validation-based epoch choice. This is not direct target leakage through features, but full-validation scores are development evidence after selection, not fresh confirmation. No learning-curve or train-held-out report is supplied.

**[P2] Historical loss weights do not match the contest's episode/condition weighting.** Historical cross-entropy is summed over observed next ticks (`arm.py:89–96`), so dense histories contribute more fitting mass than the 64-observation short-history episodes; train cutoff labels receive an additional weight of 20. The contest gives equal episode then condition weight. This is a modeling choice rather than a scoring bug, but it can bias training toward long histories. Short-history R is also the weakest of the four conditions in each of the three reported high-scoring seeds (`runs/LOG.md:50–52`).

**[P2 reproducibility gap] NumPy/Torch agreement does not establish reproducible GPU training.** Export parity is checked after converting the fitted model to float64 (`arm.py:98–107`), which is a useful deployment check. The fitting path seeds Torch and NumPy (`:45`) but does not request deterministic Torch algorithms. No repeated full-training checkpoint hashes, NumPy inference timing for all 1,536 test queries, or fresh-folder package reproduction are supplied for the 53-point sequence runs. A frozen checkpoint could make training nondeterminism irrelevant to inference, but that checkpoint is currently missing.

## Score interpretation and ranking

`src/tdx/score.py:30–37` computes the headline points on **all 96 validation episodes** and separately emits tuning/audit summaries. Therefore the sequence headline must not be compared with the previous Bayesian model's **48-episode tuning** score of 53.382758 merely because the numbers are close.

| Candidate | Comparable 96-episode development points / 60 | Evidence in this review |
| --- | ---: | --- |
| Amir M6 | 55.29658 | Parent independently rescored supplied forecasts |
| Amir M3 | 55.26837 | Parent independently rescored supplied forecasts |
| Fredy GRU128 no-message, three seeds | 53.097964–53.577773 | Branch log only; no supplied forecasts/checkpoint; epochs selected on validation |
| Previous uniform Bayesian switch96 + logistic payout | 53.037228 | Parent independently rescored saved forecasts; earlier frozen audit also exists |
| Earlier temporal CNN | 31.597320 on 48-episode tuning | Different architecture, training setup and evaluation subset; not evidence against Fredy's GRU |

The sequence model is a plausible ensemble candidate because its recurrent full-history learner differs from explicit lag-pair filtering. No complementarity, blend improvement, or statistically supported advantage over the Bayesian/Amir models can be claimed without its saved per-query forecasts. The 53.578 best seed is selected from several validation-monitored runs; neither the mean nor best seed constitutes a new held-out audit. Preserve the previous audit's historical meaning, and do not label this branch's full-validation development score as independent audit evidence.

## Fresh checks performed

Environment: the suggested `/Users/andrewvong/ml_env/bin/python3` is Python 3.9.6; `tdx.data` uses union type annotations requiring Python 3.10+, so it cannot be imported normally there. An already-installed `/opt/homebrew/bin/python3.10` provides Python 3.10.20, NumPy 2.2.6 and Torch 2.11.0. No dependencies were installed. Pytest is absent, so the original two test functions were invoked directly with their existing assertions, a temporary model directory, and a loader redirect to the actual kit public inputs. Torch/BLAS threads were limited to two; cache storage was redirected to a temporary directory. This is not verification under the organizer's Python 3.13 environment.

| Check | Result |
| --- | --- |
| Original `test_numpy_matches_torch_and_roundtrips`, H=16, one train-only epoch | Pass, 1.025 s; probabilities normalized; checkpoint reload predictions bit-identical |
| Original `test_prediction_ignores_ticks_at_or_after_cutoff`, H=8, one train-only epoch | Pass, 0.219 s; future-storage mutation has no effect |
| All data access during those tests inside `audit_dev_files` | Zero forbidden validation-answer or non-input test-file opens |
| Synthetic transductive H=8/message model, one epoch, 2 episodes per split | All weights bit-identical after mutating future storage, every signal, all validation/test cutoff labels, and all soft targets; permitted train cutoff labels held fixed |
| Synthetic Torch/NumPy logit parity | Maximum difference 2.22e-16 |
| Synthetic future-storage/signal prediction mutation | Maximum difference 0 |
| Synthetic machine permutation equivariance | Maximum difference 5.55e-17 |
| Synthetic finite-state/output and normalization check | All finite; maximum row-sum error 2.22e-16 |
| Static sequence export reproduction | Fails with the Torch-import error described above |

Synthetic method: NumPy `default_rng(20261107)`, two episodes per train/validation/test split, eight machines, 28 storage ticks, cutoffs 20 and 24, random tokens in {-1,0,1,2,3,4} before cutoff, random legal types/signals/labels/soft distributions. Two identically seeded H=8/message GRUs were fitted for one epoch with the transductive dev spec. The second received independently corrupted at/after-cutoff storage and disallowed supervision fields. Predictions were also compared under machine permutation `[4,1,7,3,0,6,2,5]`. NumPy emitted floating-point matmul warnings in this local environment, but explicit weight/output finiteness, normalization, parity and mutation checks all passed; no full-size saved checkpoint was available to test.

Machine-readable results: `adversarial_review/branches/fredy_seq_checks.json`. This report and that JSON are the only durable outputs created by this sequence review.
