# Adversarial review across all remaining branches — November 7, 2026

**Recommendation changes to Amir's M3, `m3_lag_pair_flip`.** It is the strongest defensible main-test candidate among the branch models now examined. On the same 96 validation episodes it scores **55.268370 / 60**, compared with **53.037228** for the previous checkout's switching Bayesian/logistic model. M3 improves all four conditions and 79 of 96 episodes. A fresh deterministic fit and prediction replay matches its saved validation forecasts to 1.34e-14.

M6 has the highest saved score, 55.296582, but only exceeds M3 by 0.028212 points and inherits a concrete batch-dependent feature defect from M5. M4 is a credible alternative at 55.282990; its tiny 0.014620-point advantage is insufficient to make it the default when simplicity and transfer robustness matter. Hidden test answers remain unavailable. This is a model recommendation supported by observed development results, not a guaranteed test ranking.

## Coverage and exclusions

GitHub branch enumeration was exhausted and all three heads were verified against local Git objects before snapshots were extracted. No branch checkout, merge, push or model source edit was performed.

| Branch | Reviewed commit | Scope |
|---|---|---|
| `amir_workspace` | `c339b469de33bbc2b2c68ea739eeec835139d42b` | All six M1–M6 models, shared components/selection, both saved result batches, validation and test CSVs |
| `fredy/bakeoff-gbdt-seq` | `5ba591c91d3589a2511b946dcf908b194e1390aa` | All five arm implementations and declared variants: pooled, GLM, GBDT, GRU sequence, regime; supervision/scoring, blend/refit and package paths |
| `main` | `c7da3c55ac14b65448aa9f293601467178d3593e` | README only; no models |

The previous checkout's custom forecasting suite was excluded from another code review. Its frozen saved forecasts serve only as fixed benchmarks. Amir's 11 organizer-kit Python modules are byte-identical to the kit already examined and were deduplicated. The current repository has one worktree and no additional attached worktrees. [Coverage and source hashes](coverage.json), [snapshot manifest](manifest.json).

Three GPT-6 Astra xhigh reviewers examined Amir's family, Fredy's sequence model, and Fredy's other arms. The root reviewer independently checked common-data scoring, branch coverage and the refit/package pipeline. Detailed source findings are linked below.

## Comparable ranking

All rows below refer to **full validation: 96 episodes, 768 queries, 24 episodes per condition**. Amir scores and previous-checkout benchmarks were recomputed from actual prediction CSVs. Fredy's learned-arm scores are reported pilot results because its run arrays and model checkpoints are git-ignored and absent from the published branch.

| Candidate | Points / 60 | Evidence and decision |
|---|---:|---|
| Amir M6 ensemble | 55.296582 | Independently rescored; negligible edge over M3 and inherits M5 defect |
| Amir M4 shared-flip model | 55.282990 | Independently rescored; tiny favorable improvement, extra coupling |
| **Amir M3 explicit flip-time model** | **55.268370** | **Independently rescored and replayed; recommended** |
| Amir M5 boosted corrector | 54.987969 | Independently rescored; worse than M3 and defective batch feature |
| Fredy GRU128 without peer message | 53.097964–53.577773 | Three logged seeds; validation-selected epochs; no supplied forecast/checkpoint artifacts |
| Previous checkout Bayesian switch96 + logistic | 53.037228 | Fixed benchmark formed by joining its saved tuning and audit predictions |
| Amir M2 lag-pair model | 52.597509 | Independently rescored; strong static rules, weaker changing regimes |
| Previous checkout boosted/logistic | 49.766798 | Fixed benchmark only |
| Fredy best logged GBDT, `trans_64_l15` | 36.395373 | Log-only; substantially weaker than pair/sequence models |
| Fredy best logged GLM, `l8` | 31.093246 | Log-only; no credible advantage over pooled_recent |
| Amir M1 hierarchical frequency | 31.091807 | Independently rescored; cannot express lag interactions |
| Pooled_recent control | 30.825854 | Independently rescored; shared kit control |
| Fredy best logged regime, `d2_routed` | 30.779477 | Log-only; no improvement over pooled_recent |
| Fredy pooled/context/recent controls | 29.685186–30.825854 | Frequency controls; not competitive |

The original M1/M2 batch reproduces the same predictions/scores as their later batch. Fredy additionally declares GRU128 with peer messages and GRU192 without messages; no committed full-validation score for either establishes competitiveness. Its GLM regularization/recency variants, all eight declared GBDT variants and all three regime variants were covered through implementation and recorded pilots. The factorized GLM is implemented and numerically checked but has no retained real-data score; its performance remains unranked. There is no separate trained W5 stacker artifact: the implemented final blend fits constrained weights to validation targets and deliberately claims no cross-validated score.

The 55-point branch results must not be compared directly with a previous 48-episode tuning score. The common 96-episode comparison above eliminates that mismatch. All Amir models were already evaluated on all 96 validation episodes; splitting those predictions using the earlier checkout's tuning/audit partition does **not** produce a new independent audit.

## Why M3 is a stronger generalization bet

M3 models a persistent lag pair with a possible single direction reversal at an explicit tick. It integrates uncertainty over pair, direction and flip time using Beta-binomial evidence. Before-flip observations can still identify the same pair after reversing their direction. This reuses older evidence that a generic forgetting/reset mechanism may discard or dilute. The positive payout component is strongly shrunk toward learned type/signal priors.

| Condition | M3 loss ratio | Previous Bayesian loss ratio |
|---|---:|---:|
| Stable | .077244 | .101997 |
| Short history | .106894 | .202528 |
| Drift | .079817 | .117095 |
| Shared change | .078493 | .103503 |
| Mean | .085612 | .131281 |

M3's gain is 2.231143 points. Paired delta R is -.045669, with a fixed-forecast, condition-stratified episode-bootstrap 95% interval [-.061022, -.031343]. The advantage remains if any one condition is removed. Raw no-payout MSE falls from .010378 to .004711. It also leads separately on the old tuning half, 55.409835 versus 53.382758, and old audit half, 55.123644 versus 52.665123; those are retrospective descriptive partitions for this branch.

This does not make M3 immune to overfitting. Its learned pair-frequency, high/low win rates and flip-time distribution are main-round assumptions. Repeated flips, changed lag identity, different reversal timing or weaker rules could hurt it in adaptation. Its internal pseudo-query selection also reuses empirical priors learned from the same training histories, so it is not fully episode-held-out evidence. The external validation advantage and successful replay are what make the recommendation credible.

## Material adversarial findings

1. **M5 and M6 have a batch-dependent feature.** M3's auxiliary `features()` computes `n_recent` with `obs[:, -64:]`, relative to the batch's maximum tick, instead of each machine's cutoff. This feature is consumed by M5, which is also part of M6. Adding an unrelated later-cutoff episode can change a forecast's features despite leaving its own episode unchanged. The reviewer reproduced undercounting in 749/768 validation rows, by as many as 64 observations. M3/M4 direct forecasts do not use this auxiliary count. Fixed-full-batch saved scores remain valid measurements, but M5/M6 are fragile to chunking and cutoff composition. [Source](amir/models/m3_lag_pair_flip/model.py#L236), [review and diagnostics](amir_review.md).

2. **Amir's internal validation is weaker than its descriptions imply.** M1–M4 pseudo-query predictions use population/rule priors learned from the full training histories, including evaluated endpoints. M5's GroupKFold holds out correction training rows, but the base M4 was fitted on all episodes before features were built. M6's weights rely on these upstream predictions. This is not demonstrated contamination of external validation targets; it makes internal CV gains optimistic and especially weak support for tiny additions to M3. Earlier pseudo-cutoffs also retain the original query signal; the kit does not establish whether that signal was available historically. If it changes over time, this is another internal-validation mismatch, not proven external leakage: actual validation/test queries supply their own signal. [Selection implementation](amir/models/common/selection.py), [boosted fit](amir/models/m5_boosted_corrector/model.py).

3. **M6's headline win is too small to support the extra correction.** Relative to M3, delta R is -.000554 with 95% interval [-.001284, +.000161]. Its selected weights are approximately .8348 M3 and .1652 M5. M5 alone is worse than M3 in all four conditions, with delta R +.005536 and interval [+.001256, +.010020]. M4's delta R is -.000287 with interval [-.000566, -.0000684], so it would be inaccurate to call that fixed-forecast result a statistical tie. Its practical gain is nevertheless only .014620 points, mainly short history, and the interval does not adjust for model development or selection. Keep M4 as a close alternative.

4. **Fredy's GRU score is development evidence with incomplete artifacts.** This is a real GRU trained over full tick sequences with machine-type embeddings, unlike the earlier self-only CNN. It can learn meaningful temporal rules. Development fits can legally use validation/test histories as self-supervision; they do not receive validation cutoff answers. However, the notes explicitly state that epochs were selected while watching validation, and the branch publishes only score logs, not the prediction arrays or fitted weights needed for independent rescoring or replay. Do not label the 53.578 best seed an untouched holdout result. [Pilot disclosure](fredy/claude_output/notes/pilot_results.md), [sequence review](fredy_seq_review.md).

5. **Fredy's package builder rejects its own GRU.** The runtime import scanner traverses imports inside methods, finds the PyTorch import inside `SeqArm.fit`, and rejects it even though prediction uses NumPy. The reviewer reproduced `ValueError: torch is forbidden in shipped runtime graph: torch.nn.functional`. This is an actual delivery blocker for the generic SEQ package path. A tested separation of training-only code from exported inference is needed before packaging that arm. [Adapter scanner](fredy/src/tdx/package/adapters.py), [check result](fredy_seq_checks.json).

6. **Fredy's default pipeline chooses the weaker GBDT.** `config/pipeline.json` selects only `gbdt/trans_64_l15` with `drop_neighbour=true`, approximately 36.395 development points, while the branch's GRU logs are above 53. Running the default final pipeline does not automatically select the strongest observed model. Final mode also changes supervision by including validation labels/soft targets, so dev scores are not independent evaluation of those final fitted models. Final fitting is allowed, but its performance should not be described as already measured. [Pipeline config](fredy/config/pipeline.json), [supervision permissions](fredy/src/tdx/fit.py).

The default blend routing distinguishes only short versus long histories: its stable, drift and shared-change rows enter with identical weights and are not separately identifiable. Regime changepoint accuracy also needs a base-rate comparison: its recorded 73.958% is below the 75% obtained by always predicting no shared change. These are additional reasons not to infer forecasting gains from component diagnostics.

No demonstrated use of external-validation targets inside Amir's forecast methods was found. Missing observations and causal cutoff masking were checked. Fredy's dev supervision boundary strips validation labels, conditions and soft targets; final-mode use is explicit. These positive checks do not remove selection bias from repeatedly inspected public validation.

## Verification and limits

- Independently rescored all 12 Amir saved validation CSVs from its two batches and the two fixed full-validation benchmarks. Verified every corresponding saved test CSV covers exactly 1,536 IDs and contains valid probabilities; this checks format, not test accuracy.
- All 15 released CSVs in Amir's kit match the local kit byte for byte. Metadata differs only in whitespace and decodes identically.
- Fresh M3 fit plus validation prediction replay passed at maximum absolute difference 1.3323e-14, in 9.20 seconds with two numerical threads. The replay used the available NumPy/SciPy environment rather than the branch's exact pins, and is not an organizer-container packaging verification. [Replay script](amir_m3_replay.py), [result](amir_m3_replay.json).
- Fredy's two original SEQ tests passed under the existing Python 3.10 environment, including causal input/answer-access guards, save/load and NumPy inference checks. The small smoke tests do not reproduce its 100-epoch pilot or demonstrate Python 3.13 offline packaging. [Checks](fredy_seq_checks.json).
- Fredy nonsequence review passed 26 targeted checks: original numerical/causality tests plus explicit dataset-partition and blend-identifiability probes. They ran in the available Python 3.9 numerical environment with type-annotation compatibility because the installed Python 3.10 SciPy extension failed. LightGBM itself was not exercised; engine-argument capture is not a trained-model reproduction. [Checks and limitations](fredy_other_checks.json).
- Native LightGBM/published trained artifacts are unavailable in the current environments. Fredy's learned-arm rankings therefore retain their log-only status; no synthetic or smoke result is substituted for a benchmark score.

All confidence intervals here resample episodes while holding saved forecasts fixed. They do not account for the entire model search or forecast a hidden-test confidence interval. The original validation audit is already consumed, and the newly inspected branches used the whole validation release. Further tuning against these comparisons needs new data for independent confirmation.

## Evidence

- [Complete coverage, branch tips and source hashes](coverage.json)
- [Independent scores, both old partitions, paired comparisons and test-format checks](cross_branch_scores.json)
- [Independent scoring script](compare_saved.py)
- [Amir all-model review](amir_review.md)
- [Fredy sequence review](fredy_seq_review.md)
- [Fredy GLM/GBDT/regime/pooled review](fredy_other_review.md)
- [Previous checkout review, retained as historical scope](../REVIEW.md)
