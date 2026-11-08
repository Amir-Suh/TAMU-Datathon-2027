## Main-round model selection

**Current recommendation: switching Bayesian lag pairs (`switch_96`) with the self-only logistic positive payout head.** It scored **53.383 / 60 on tuning** and **52.665 / 60 on the separate audit**, beating the previous boosted/logistic leader's **49.213 audit points**. This is the strongest evidence for generalization among the evaluated models; hidden-test performance remains unknown. [Bayesian experiment](runs/bayesian-lag-pairs/replay/report.json), [frozen audit](runs/bayesian-lag-pairs/audit-switch-96/report.json).

The model learns which pair of earlier win/loss outcomes matters for each machine and allows the pair's effect to change over time. Its frozen settings are a uniform prior over all 28 lag pairs, signed amplitudes `{-2,-1,+1,+2}`, reset hazard `1/96`, no evidence discounting, and a 12-tick win offset with strength 8. The self-only logistic model supplies the conditional probabilities of 1×, 2×, 5× and 20× given a positive payout. The complete audited model bundle is [frozen_models.joblib](runs/bayesian-lag-pairs/audit-switch-96/frozen_models.joblib); the accompanying [freeze manifest](runs/bayesian-lag-pairs/audit-switch-96/freeze.json) records its settings and hashes.

Tuning and audit each contain 48 episodes and 384 queries, with 12 episodes per condition. **The audit has already been used** for the frozen Bayesian-versus-boosted comparison. Further choices informed by that result need fresh episodes for an independent confirmation. A dash below means no saved audit comparison for that configuration.

| Forecast | Tuning points / 60 | Audit points / 60 | Assessment |
|---|---:|---:|---|
| Switching Bayesian pairs + logistic head, uniform pair prior | 53.383 | **52.665** | Recommended frozen candidate |
| Switching Bayesian pairs + logistic head, learned pair prior `rho=0.8` | **53.800** | — | Promising challenger; did not pass its promotion threshold |
| Bayesian/boosted blend, weights fitted outside each tuning fold | 53.393 | — | Only 0.010 above standalone Bayesian; no compelling reason to add a fitted blend |
| Boosted rate-3072 win probability + logistic head | 50.274 | 49.213 | Previous leader; useful fallback |
| Boosted correction, rate 3072 | 49.744 | — | Strong improvement over adaptive control |
| Individual peer lags, rate 3072 | 44.117 | — | No clear gain over matched adaptive control |
| Adaptive lag, rate 1536 / 3072 | 44.104 / 44.098 | — | Rates effectively tied; 3072 selected on grouped training |
| Regime filter, rate 3072 | 43.950 | — | No demonstrated improvement over adaptive control |
| Signed lag motifs | 42.026 | — | Weak under drift and shared changes |
| Learned transition retrieval | 41.233 | — | Behind Bayesian and boosted candidates |
| Boosted correction, rate 48 | 37.775 | — | Superseded by the rate-3072 version |
| Original adaptive lag, rate 48 | 33.225 | — | Original control |
| Self-only causal CNN | 31.597 | — | No competitive sequence-model gain in this experiment |
| Logistic families, fixed retrieval and frequency/fixed-lag controls | 31.133–31.444 | — | Useful controls; logistic payout head remains valuable in the stronger models |

The Bayesian audit gain is **3.453 points**, with paired `ΔR = -0.07993` and a saved 95% episode-bootstrap interval of `[-0.12438, -0.04198]`. It improves short history, drift and shared change; boosted wins the stable audit condition (`R=0.0768` versus Bayesian's `0.0936`). The overall Bayesian advantage remains when any single condition is omitted. These intervals describe fixed forecasts on observed episodes, not a guaranteed test score or uncertainty adjusted for every model tried.

The learned pair prior improves tuning by **0.417 points**, mostly on short histories. Its `ΔR=-0.00871` misses the experiment's required `-0.01`, and [the report records `advances=false`](runs/bayesian-lag-pair-tuning/replay/report.json). Nested training evaluation shows a much smaller gain: cutoff win Brier `0.082812 → 0.082452`, with two folds improving, one worsening and one unchanged. Keep the audited uniform-prior model as the default; the higher tuning score alone does not establish a better test model.

The adversarial review used three GPT-6 Astra reviewers at xhigh reasoning, independently rescored 152 saved tuning CSVs (72 distinct normalized prediction arrays), checked frozen artifacts, and ran 19 targeted tests. All tests passed, and a fresh Bayesian replay reproduced the four saved candidate CSVs with zero maximum probability difference. No causal leakage invalidating the recommended model was found. Public test histories retain the same lag availability and inferred common-pair prevalence; that supports main-round transfer without revealing test accuracy. Remaining risks include short-history uncertainty, repeated use of tuning, fixed lag-pair identity within each Bayesian hypothesis, and changed temporal relationships in adaptation. [Full review and verification](main-participants-60-40/adversarial_review/REVIEW.md). The [earlier experiment summary](EXPERIMENT_RESULTS.md) documents the superseded boosted leader and predates the Bayesian audit.

To replay the Bayesian tuning experiment from the repository root, keep the saved component CSVs and use a fresh output directory:

```sh
python -B run_bayesian_lag_pairs.py \
  --out runs/bayesian-lag-pairs/fresh \
  --compare-to runs/bayesian-lag-pairs/replay
```

This writes tuning forecasts. Before submission, the frozen Bayesian model and logistic head still need a forecast-only entry point for the 1,536 test queries, an offline reproduction check, and the required code ZIP. The starter commands below export the pooled starter; they do not export this recommended model.

