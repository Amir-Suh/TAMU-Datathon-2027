## Current recommendation across branches

**Use Amir's M3 explicit flip-time lag-pair model as the main-round candidate.** The expanded adversarial review covered all three remote branches and found that the earlier checkout review had missed a stronger model family. On the same **96 validation episodes and 768 queries**, M3 scores **55.268 / 60**, compared with **53.037** for the previous switching Bayesian/logistic recommendation. M3 improves every condition and 79 of 96 episodes. A fresh fit reproduced its saved predictions within **1.34e-14**. [Full branch review](main-participants-60-40/adversarial_review/branches/REVIEW.md), [M3 source and design](https://github.com/Amir-Suh/TAMU-Datathon-2027/blob/c339b469de33bbc2b2c68ea739eeec835139d42b/models/m3_lag_pair_flip/MODEL.md).

The reviewed heads are `amir_workspace` at `c339b46`, `fredy/bakeoff-gbdt-seq` at `5ba591c`, and `main` at `c7da3c5` (README only). All six Amir models and all five Fredy arm implementations, including declared variants and blend/packaging paths, were reviewed. The previously reviewed custom suite and 11 identical organizer-kit modules were excluded from duplicate code review. [Coverage and source hashes](main-participants-60-40/adversarial_review/branches/coverage.json).

| Model | Full-validation points / 60 | Assessment |
|---|---:|---|
| **Amir M3: explicit flip time** | **55.268** | Recommended; independently rescored and replayed |
| Amir M4: shared flip time | 55.283 | Close alternative; only 0.015 points above M3 |
| Amir M6: ensemble | **55.297** | Highest saved score; only 0.028 above M3 and inherits M5's feature defect |
| Amir M5: boosted corrector | 54.988 | Worse than M3; batch-dependent feature and incomplete base cross-fitting |
| Fredy GRU128, no peer message | 53.098–53.578 | Three logged seeds; validation-selected epochs; fitted artifacts absent from branch |
| Previous checkout: switching Bayesian + logistic head | 53.037 | Strong fallback; combined saved tuning and audit predictions |
| Amir M2: lag-pair rules | 52.598 | Strong static model, weaker under changing regimes |
| Previous checkout: boosted + logistic head | 49.767 | Earlier benchmark |
| Fredy GBDT, best logged variant | 36.395 | Substantially behind pair and sequence models |
| Fredy GLM, best logged variant | 31.093 | No clear advantage over frequency control |
| Amir M1: hierarchical frequency | 31.092 | Weak baseline |
| Pooled_recent control | 30.826 | Shared kit control |
| Fredy regime model, best logged variant | 30.779 | No demonstrated gain over control |

Amir and previous-checkout scores above were independently recomputed from saved CSVs. Fredy's learned-arm scores are **reported pilot results**: its prediction arrays and model checkpoints are git-ignored and were not published with the branch. Its peer-message and larger GRU variants have no committed result establishing superiority. The prior checkout's full model table remains in the [earlier review](main-participants-60-40/adversarial_review/REVIEW.md).

M3 models a persistent lag pair whose win/loss direction may reverse once, integrating uncertainty about when the reversal happened. It can reuse observations from both sides of that reversal. Its advantage over the previous Bayesian model is **2.231 points**, with paired `ΔR=-0.04567` and a descriptive 95% episode-bootstrap interval of `[-0.06102, -0.03134]`. M6's much smaller advantage over M3 has an interval crossing zero. M4's tiny favorable interval does exclude zero for these fixed forecasts, but its practical gain is small and does not account for model selection.

The most important review findings are:

- **M5/M6 depend on unrelated prediction rows.** Their recent-history feature counts the batch's last 64 columns instead of each machine's 64 pre-cutoff ticks. It is wrong on 749 of 768 validation rows. M3/M4 direct forecasts are unaffected; fixing the correction requires retraining it.
- **Training selection is not fully held out.** Amir's pseudo-query scores reuse full-training empirical priors, and M5 cross-fits only the classifier after fitting its base on all training episodes. This weakens the case for small corrections and ensemble gains. No direct external-validation-target leakage invalidating M3 was found.
- **Fredy's GRU is competitive but not ready through its generic exporter.** The exporter rejects a training-only PyTorch import even though inference uses NumPy. The default pipeline also still selects the 36.395-point GBDT, not the GRU.
- **There is no fresh audit left in these comparisons.** Amir evaluated all 96 validation episodes, and Fredy selected epochs while watching validation. Reusing the earlier 48/48 partition is descriptive, not a new independent audit. Confidence intervals hold forecasts fixed and do not guarantee a test score.

Main-test generalization looks strongest for M3/M4, but their learned pair prevalence and single-flip assumptions need renewed evaluation in adaptation. Keep the previous uniform-prior switching Bayesian model as a fallback. The complete [review](main-participants-60-40/adversarial_review/branches/REVIEW.md) includes limitations, source locations, targeted checks and common-data comparisons.

M3's source and test-output runner live on `amir_workspace`, with an inspected snapshot under `main-participants-60-40/adversarial_review/branches/amir/`. In that branch's checkout, its existing command is:

```sh
python -m models.run --models m3 --out results/m3-fresh
```

Use a fresh output directory and the branch's dependencies. The runner refits on train, writes validation and test forecasts, and checks test IDs/probabilities. All 12 saved test CSVs across Amir's two run batches passed independent format/coverage checks during review. An organizer-compatible code ZIP and offline reproduction check are still required before upload. The starter commands below generate the pooled starter, not M3.

